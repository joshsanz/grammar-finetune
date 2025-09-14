"""
Hyperparameter tuning for Gemma 3 grammar correction using Optuna.
Updated to use in-process training with TPESampler and HyperbandPruner.
"""

import unsloth  # Ensure unsloth is imported before transformers
import os
import sys
import json
import yaml
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import mlflow
import mlflow.optuna
from transformers import set_seed
import logging, coloredlogs
import argparse
from typing import Dict, Any
from datetime import datetime

from src.training import train_model, TrainingResults

# Nested runs not properly attaching to parent run, so we avoid them for now
# os.environ["MLFLOW_NESTED_RUN"] = "true"

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameter search space
SEARCH_SPACE = {
    "model_type": ["base", "instruction-tuned"],
    "r": [16, 32, 64, 128],
    "lr_scheduler_type": ["cosine", "linear"],
    "learning_rate": [5e-5, 2e-4],
    "weight_decay": [0.01, 0.1]
}

def get_model_name(model_type: str) -> str:
    """Get the full model name based on model type."""
    if model_type == "base":
        return "unsloth/gemma-3-4b-pt-unsloth-bnb-4bit"
    else:  # instruction-tuned
        return "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_trial_config(trial_params: Dict[str, Any], base_config_path: str) -> Dict[str, Any]:
    """Create configuration dictionary for a trial by modifying base config."""
    # Load base configuration
    config = load_base_config(base_config_path)

    # Override model name based on trial parameters
    config["model"]["model_name"] = get_model_name(trial_params["model_type"])

    # Override PEFT parameters
    config["peft"]["r"] = trial_params["r"]
    config["peft"]["lora_alpha"] = 2 * trial_params["r"]  # Always equal to 2r
    # If r is 128, reduce batch size to fit in memory and increase gradient accumulation
    if trial_params["r"] == 128:
        config["training"]["per_device_train_batch_size"] = max(1, config["training"]["per_device_train_batch_size"] // 2)
        config["training"]["gradient_accumulation_steps"] = config["training"].get("gradient_accumulation_steps", 1) * 2

    # Override training parameters
    config["training"]["learning_rate"] = trial_params["learning_rate"]
    config["training"]["weight_decay"] = trial_params["weight_decay"]
    config["training"]["lr_scheduler_type"] = trial_params["lr_scheduler_type"]
    # config["training"]["seed"] = 42 + trial_params.get("trial_number", 0)

    return config

def run_training_trial(config: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    """Run a single training trial with given configuration using in-process training."""
    try:
        logger.info(f"Starting trial {trial.number} with in-process training")

        # Set environment variables for MLflow
        # Set MLflow tags for this trial
        tags = {
            "trial_number": trial.number,
            "model_type": config["model"]["model_name"].split("-")[-1] if "-" in config["model"]["model_name"] else "unknown",
            "r": config["peft"]["r"],
            "lr": config["training"]["learning_rate"],
            "wd": config["training"]["weight_decay"],
            "scheduler": config["training"]["lr_scheduler_type"],
            "hyperparameter_tuning": "true"
        }
        os.environ["MLFLOW_TAGS"] = json.dumps(tags)

        # Call the refactored training function with trial support
        results = train_model(config, trial=trial)
        mlflow.end_run()  # Ensure MLflow run is ended after training since nested runs are not working properly
        # Extract metrics from training results
        accuracy = results.best_eval_accuracy if results.best_eval_accuracy is not None else 0.0
        eval_loss = results.best_eval_loss if results.best_eval_loss != float('inf') else 10.0

        logger.info(f"Trial {trial.number} completed - Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "eval_loss": eval_loss,
            "success": True,
            "training_history": results.training_history
        }

    except optuna.TrialPruned:
        logger.warning(f"Trial {trial.number} was pruned")
        mlflow.end_run()  # Ensure MLflow run is ended after pruning
        # Return partial results for pruned trials
        return {
            "accuracy": 0.0,
            "eval_loss": float('inf'),
            "success": False,
            "pruned": True
        }
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        # Clean up GPU memory on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mlflow.end_run()  # Ensure MLflow run is ended
        return {
            "accuracy": 0.0,
            "eval_loss": float('inf'),
            "success": False,
            "error": str(e)
        }

def create_objective(config_path: str):
    """Create objective function with config path."""
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

        # Create configuration using base config
        config = create_trial_config(trial_params, config_path)

        # Run training trial with in-process training
        results = run_training_trial(config, trial)

        # Log results to Optuna
        trial.set_user_attr("accuracy", results["accuracy"])
        trial.set_user_attr("eval_loss", results["eval_loss"])
        trial.set_user_attr("success", results["success"])

        # Return primary metric (accuracy) for optimization
        return results["accuracy"]

    return objective

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Gemma 3 grammar correction")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config/gemma3_4b_tuning_config.yaml",
        help="Path to base configuration YAML file"
    )
    return parser.parse_args()

def main():
    """Main function to run hyperparameter tuning."""
    # Parse arguments
    args = parse_args()
    config_path = args.config_path
    config = load_base_config(config_path)
    optuna_params = config['optuna']

    logger.info(f"Using base configuration from: {config_path}")

    # Setup MLflow
    experiment_name = "gemma3_hyper_tuning"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create Optuna study with TPE sampler and Hyperband pruner
    sampler = TPESampler(seed=optuna_params["seed"])
    pruner = HyperbandPruner(
        min_resource=optuna_params["prune_min_evals"],
        max_resource=optuna_params["prune_max_evals"],
        reduction_factor=optuna_params["reduction_factor"],
    )

    study = optuna.create_study(
        direction=optuna_params["direction"],
        sampler=sampler,
        pruner=pruner,
        study_name=optuna_params["study_name"],
        load_if_exists=True
    )

    logger.info(f"Starting hyperparameter tuning with up to {optuna_params["n_trials"]} trials using TPE sampler")
    logger.info("Pruning will be performed by HyperbandPruner based on intermediate results")

    # Create objective function with config path
    objective = create_objective(config_path)

    # Run optimization
    study.optimize(objective,
                    n_trials=optuna_params["n_trials"],
                    timeout=optuna_params["timeout"])

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
