"""
Core training functionality for grammar correction fine-tuning.
Extracted from gemma3_finetune.py to support both CLI and hyperparameter optimization.
"""

from unsloth import FastModel
import json
import os
import torch
from datasets import DatasetDict, load_dataset
from transformers import EarlyStoppingCallback, TrainerCallback
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import logging
import mlflow
import mlflow.data
from typing import Dict, Any, Optional, List
import optuna

# Import custom modules
from .types import TrainingResults
from .callbacks import OptunaCallback
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def setup_mlflow_environment(config: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> None:
    """
    Set up MLflow environment variables and tracking.

    Args:
        config: Configuration dictionary
        trial: Optional Optuna trial for hyperparameter optimization
    """
    # Validate required MLflow environment
    assert os.getenv("MLFLOW_TRACKING_URI") is not None, "Please set MLFLOW_TRACKING_URI environment variable"
    assert os.getenv("MLFLOW_EXPERIMENT_NAME") is not None, "Please set MLFLOW_EXPERIMENT_NAME environment variable"

    if os.getenv("MLFLOW_TAGS") is None:
        logger.warning("MLFLOW_TAGS environment variable not set, tags will be set based on config")

    # Remove MLFLOW_RUN_ID usage as requested - let MLflow auto-generate IDs
    if os.getenv("MLFLOW_RUN_ID") is not None:
        logger.warning("MLFLOW_RUN_ID detected but will be ignored. Use run_name parameter instead.")

    if os.getenv("HF_MLFLOW_LOG_ARTIFACTS") is None:
        logger.warning("HF_MLFLOW_LOG_ARTIFACTS environment variable not set, not logging model artifacts to MLflow")
    else:
        assert os.getenv("HF_MLFLOW_LOG_ARTIFACTS") in ["True", "1"], "HF_MLFLOW_LOG_ARTIFACTS must be 'True' or '1'"
        logger.info("HF_MLFLOW_LOG_ARTIFACTS is set, will log model artifacts to MLflow")

    # Set default MLflow tags if not provided
    model_name = config["model"]["model_name"]
    model_parts = model_name.split("/")[-1].split("-")
    model_3_n = model_parts[1] if len(model_parts) > 1 else "unknown"
    model_params = model_parts[2] if len(model_parts) > 2 else "unknown"
    model_it = model_parts[3] if len(model_parts) > 3 else "unknown"

    quant = "bf16"
    if config["model"]["load_in_8bit"]:
        quant = "Q8_0"
    elif config["model"]["load_in_4bit"]:
        quant = "Q4_0"

    if os.getenv("MLFLOW_TAGS") is None:
        tags = {
            "model": "gemma3",
            "params": model_params,
            "pretrain": model_it,
            "quant": quant
        }

        # Add trial-specific tags if running hyperparameter optimization
        if trial is not None:
            tags.update({
                "trial_number": trial.number,
                "hyperparameter_tuning": "true"
            })

        tags_str = json.dumps(tags)
        os.environ["MLFLOW_TAGS"] = tags_str
        logger.info(f"MLFLOW_TAGS not set; defaulting to: {tags_str}")


def load_model_and_tokenizer(config: Dict[str, Any]) -> tuple:
    """
    Load and configure the model and tokenizer.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config['model']['model_name']}")

    # Load model and tokenizer
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model"]["model_name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        load_in_8bit=config["model"]["load_in_8bit"],
        full_finetuning=config["model"]["full_finetuning"],
    )

    # Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        # LoRA settings
        r=config["peft"]["r"],
        target_modules=config["peft"]["target_modules"],
        lora_alpha=config["peft"]["lora_alpha"],
        lora_dropout=config["peft"]["lora_dropout"],
        bias=config["peft"]["bias"],
        use_gradient_checkpointing=config["peft"]["use_gradient_checkpointing"],
        random_state=config["peft"]["random_state"],
        use_rslora=config["peft"]["use_rslora"],
        loftq_config=config["peft"]["loftq_config"],
    )

    return model, tokenizer


def prepare_dataset(config: Dict[str, Any], tokenizer) -> DatasetDict:
    """
    Load and prepare the training dataset.

    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer for chat template application

    Returns:
        Prepared DatasetDict
    """
    logger.info(f"Loading dataset from: {config['data']['dataset_path']}")

    # Set up chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config["data"]["chat_template"],
    )

    # Load dataset
    dataset_train = load_dataset(config["data"]["dataset_path"], split="train")
    dataset_test = load_dataset(config["data"]["dataset_path"], split="test[:250]")  # First 250 samples
    dataset = DatasetDict({"train": dataset_train, "test": dataset_test})

    # Load system prompt
    with open(config["data"]["system_prompt_path"], "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    def convert_to_chatml(example):
        """Convert example to ChatML format."""
        marked_original = f"[BEGINNING OF CONTENT]\n{example['original']}\n[END OF CONTENT]"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": marked_original},
                {"role": "assistant", "content": example["corrected"]}
            ]
        }

    dataset = dataset.map(convert_to_chatml).remove_columns(["source", "original", "corrected"])

    def formatting_prompts_func(examples):
        """Apply chat template to conversations."""
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    logger.info(f"Dataset prepared: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    return dataset


def create_trainer(
    model,
    tokenizer,
    dataset: DatasetDict,
    config: Dict[str, Any],
    trial: Optional[optuna.Trial] = None
) -> SFTTrainer:
    """
    Create and configure the SFTTrainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Prepared dataset
        config: Configuration dictionary
        trial: Optional Optuna trial for hyperparameter optimization

    Returns:
        Configured SFTTrainer
    """
    # Set up scheduler kwargs
    scheduler_kwargs = {}
    if config["training"]["lr_scheduler_type"] == "cosine_with_min_lr":
        scheduler_kwargs["min_lr_rate"] = 0.1

    # Determine run name
    run_name = None
    if trial is not None:
        run_name = f"trial_{trial.number}"

    # Create training arguments
    training_args = SFTConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"]["max_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        lr_scheduler_kwargs=scheduler_kwargs,
        logging_steps=config["training"]["logging_steps"],
        eval_strategy=config["training"]["eval_strategy"],
        eval_steps=config["training"]["eval_steps"],
        eval_on_start=True,  # Evaluate at start of training
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"].get("save_total_limit", 3),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config["training"]["seed"],
        report_to=config["training"]["report_to"],
        run_name=run_name,  # Use run_name instead of MLFLOW_RUN_ID
        # Add compute_metrics for accuracy tracking
        dataloader_pin_memory=False,  # Avoid potential CUDA issues
    )

    # Create metrics function with tokenizer
    # def compute_metrics_with_tokenizer(eval_pred):
    #     return compute_metrics(eval_pred, tokenizer)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        # compute_metrics=compute_metrics_with_tokenizer,
    )

    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.0,
    )
    trainer.add_callback(early_stopping_callback)

    # Add Optuna callback if trial is provided
    if trial is not None:
        optuna_callback = OptunaCallback(trial, metric_for_best_model="eval_loss")
        trainer.add_callback(optuna_callback)

    # Configure response-only training
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config["masking"]["instruction_part"],
        response_part=config["masking"]["response_part"],
    )

    return trainer


def log_datasets_to_mlflow(dataset: DatasetDict, config: Dict[str, Any]) -> None:
    """
    Log datasets to MLflow for tracking.

    Args:
        dataset: The dataset to log
        config: Configuration dictionary
    """
    logger.info("Creating MLflow dataset objects for tracking...")

    mlflow_train_dataset = mlflow.data.from_huggingface(
        dataset["train"],
        data_dir=config["data"]["dataset_path"],
        name="training_dataset"
    )
    mlflow_test_dataset = mlflow.data.from_huggingface(
        dataset["test"],
        data_dir=config["data"]["dataset_path"],
        name="test_dataset"
    )

    logger.info("Logging datasets to MLflow...")
    mlflow.log_input(mlflow_train_dataset, context="training")
    mlflow.log_input(mlflow_test_dataset, context="evaluation")


def train_model(config: Dict[str, Any], trial: Optional[optuna.Trial] = None) -> TrainingResults:
    """
    Main training function that can be called from both CLI and hyperparameter optimization.

    Args:
        config: Configuration dictionary containing all training parameters
        trial: Optional Optuna trial for hyperparameter optimization

    Returns:
        TrainingResults object containing model, metrics, and training history
    """
    try:
        logger.info("Starting training process...")

        # Setup MLflow environment
        setup_mlflow_environment(config, trial)

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config)

        # Prepare dataset
        dataset = prepare_dataset(config, tokenizer)

        # Log datasets to MLflow
        log_datasets_to_mlflow(dataset, config)

        # Log memory stats
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")

        # Create and configure trainer
        trainer = create_trainer(model, tokenizer, dataset, config, trial)

        # Train the model
        logger.info("Starting model training...")
        trainer_stats = trainer.train()
        print(trainer_stats)
        print(list(trainer_stats.metrics.keys()))


        # Log final memory stats
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
            logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
            logger.info(f"Peak reserved memory = {used_memory} GB.")
            logger.info(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
            logger.info(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # Extract training history
        training_history = {"train_loss": [], "eval_loss": [], "eval_accuracy": []}

        # Get history from Optuna callback if available
        optuna_callback = None
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, OptunaCallback):
                optuna_callback = callback
                break

        if optuna_callback:
            history = optuna_callback.get_training_history()
            training_history.update({
                "eval_loss": history["eval_loss_history"],
                "eval_accuracy": history["eval_accuracy_history"],
                "train_loss": history["train_loss_history"]
            })

        # Extract best metrics
        best_eval_loss = float('inf')
        best_eval_accuracy = None

        if optuna_callback:
            history = optuna_callback.get_training_history()
            best_eval_loss = history["best_eval_loss"]

        logger.info("Training completed successfully!")
        logger.info(f"Best eval_loss: {best_eval_loss}")
        # if best_eval_accuracy is not None:
        #     logger.info(f"Best eval_accuracy: {best_eval_accuracy}")

        return TrainingResults(
            model=model,
            tokenizer=tokenizer,
            trainer_stats=trainer_stats,
            best_eval_loss=best_eval_loss,
            best_eval_accuracy=best_eval_accuracy,
            training_history=training_history
        )

    except optuna.TrialPruned:
        # Handle Optuna pruning
        logger.info("Trial was pruned by Optuna")
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Clean up GPU memory on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise