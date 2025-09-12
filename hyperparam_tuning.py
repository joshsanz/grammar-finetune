"""
Hyperparameter tuning for Gemma 3 grammar correction using Optuna.
Explores 162 hyperparameter combinations with grid search.
"""

import os
import sys
import json
import yaml
import torch
import optuna
from optuna.samplers import GridSampler
import mlflow
import mlflow.optuna
from transformers import set_seed
import logging
from typing import Dict, Any
import subprocess
import tempfile

# Import custom metrics
from metrics import compute_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameter search space
SEARCH_SPACE = {
    "model_type": ["base", "instruction-tuned"],
    "r": [8, 16, 32],
    "lr_scheduler_type": ["cosine", "linear", "constant"],
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "weight_decay": [0.001, 0.01, 0.1]
}

def get_model_name(model_type: str) -> str:
    """Get the full model name based on model type."""
    if model_type == "base":
        return "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit"
    else:  # instruction-tuned
        return "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

def create_trial_config(trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration dictionary for a trial."""
    config = {
        "model": {
            "model_name": get_model_name(trial_params["model_type"]),
            "max_seq_length": 2048,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "full_finetuning": False
        },
        "peft": {
            "r": trial_params["r"],
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": trial_params["r"],  # Always equal to r
            "lora_dropout": 0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407,
            "use_rslora": True,
            "loftq_config": None
        },
        "training": {
            "dataset_text_field": "text",
            "max_length": 768,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 25,
            "max_steps": -1,
            "num_train_epochs": 1,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "learning_rate": trial_params["learning_rate"],
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": trial_params["weight_decay"],
            "lr_scheduler_type": trial_params["lr_scheduler_type"],
            "seed": 42 + trial_params.get("trial_number", 0),
            "output_dir": "outputs",
            "save_steps": 50,
            "save_total_limit": 3,
            "report_to": "mlflow"
        },
        "data": {
            "dataset_path": "data/gec-dataset-small",
            "system_prompt_path": "config/prompt.txt",
            "chat_template": "gemma3"
        },
        "inference": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": 125,
            "cache_implementation": "dynamic"
        },
        "masking": {
            "instruction_part": "<start_of_turn>user\n",
            "response_part": "<start_of_turn>model\n"
        }
    }
    return config

def run_training_trial(config: Dict[str, Any], trial_number: int) -> Dict[str, Any]:
    """Run a single training trial with given configuration."""
    config_path = None
    try:
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name

        # Set environment variables for MLflow
        os.environ["MLFLOW_EXPERIMENT_NAME"] = f"gemma3_hyperparam_tuning_trial_{trial_number}"
        os.environ["MLFLOW_RUN_ID"] = f"trial_{trial_number}"
        os.environ["MLFLOW_TAGS"] = json.dumps({
            "trial_number": trial_number,
            "model_type": config["model"]["model_name"].split("-")[3],  # pt or it
            "r": config["peft"]["r"],
            "lr": config["training"]["learning_rate"],
            "wd": config["training"]["weight_decay"],
            "scheduler": config["training"]["lr_scheduler_type"]
        })

        # Run training script
        cmd = [sys.executable, "gemma3-finetune.py", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout

        if result.returncode != 0:
            logger.error(f"Trial {trial_number} failed: {result.stderr}")
            return {"accuracy": 0.0, "eval_loss": float('inf'), "success": False}

        # Parse results from MLflow or training output
        # For now, return dummy values - will be enhanced with actual metric parsing
        return {
            "accuracy": 0.5,  # Placeholder - will be replaced with actual parsing
            "eval_loss": 1.0,  # Placeholder
            "success": True
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Trial {trial_number} timed out")
        return {"accuracy": 0.0, "eval_loss": float('inf'), "success": False}
    except Exception as e:
        logger.error(f"Trial {trial_number} failed with error: {e}")
        return {"accuracy": 0.0, "eval_loss": float('inf'), "success": False}
    finally:
        # Clean up temporary config file
        if config_path and os.path.exists(config_path):
            os.unlink(config_path)

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for hyperparameter optimization."""
    # Sample hyperparameters
    model_type = trial.suggest_categorical("model_type", SEARCH_SPACE["model_type"])
    r = trial.suggest_categorical("r", SEARCH_SPACE["r"])
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", SEARCH_SPACE["lr_scheduler_type"])
    learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"])
    weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"])

    trial_params = {
        "model_type": model_type,
        "r": r,
        "lr_scheduler_type": lr_scheduler_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "trial_number": trial.number
    }

    logger.info(f"Starting trial {trial.number} with params: {trial_params}")

    # Create configuration
    config = create_trial_config(trial_params)

    # Run training trial
    results = run_training_trial(config, trial.number)

    # Log results to Optuna
    trial.set_user_attr("accuracy", results["accuracy"])
    trial.set_user_attr("eval_loss", results["eval_loss"])
    trial.set_user_attr("success", results["success"])

    # Return primary metric (accuracy) for optimization
    return results["accuracy"]

def main():
    """Main function to run hyperparameter tuning."""
    # Setup MLflow
    mlflow.set_experiment("gemma3_hyperparam_tuning")

    # Create Optuna study with grid search
    search_space = SEARCH_SPACE.copy()
    sampler = GridSampler(search_space)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="gemma3_hyperparam_tuning",
        load_if_exists=True
    )

    # Note: MLflow Optuna integration setup will be handled per trial
    # mlflow.optuna.integration.autolog()  # Commented out until MLflow version confirmed

    # Calculate total number of trials
    total_trials = 2 * 3 * 3 * 3 * 3  # 162 trials
    logger.info(f"Starting hyperparameter tuning with {total_trials} trials")

    # Run optimization
    study.optimize(objective, n_trials=162, timeout=28800)  # 8 hour timeout

    # Log best results
    logger.info("Hyperparameter tuning completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best accuracy: {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")

    # Save study results
    with open("hyperparam_study_results.json", "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_accuracy": study.best_value,
            "best_params": study.best_params,
            "all_trials": [
                {
                    "trial": t.number,
                    "value": t.value,
                    "params": t.params,
                    "user_attrs": t.user_attrs
                } for t in study.trials
            ]
        }, f, indent=2)

    logger.info("Results saved to hyperparam_study_results.json")

if __name__ == "__main__":
    main()
