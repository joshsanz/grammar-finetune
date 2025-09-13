#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma-3 Fine-tuning Script for Grammar Error Correction.
Refactored to use the shared training_core module while maintaining CLI compatibility.
"""

import unsloth  # Ensure unsloth is imported before transformers
import json
import os
import sys
import yaml
import logging, coloredlogs
from transformers import TextStreamer
import mlflow

# Import the refactored training core
from src.training import train_model

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger("GemmaFinetune")


def main():
    """Main function for CLI usage of the training script."""

    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/gemma3_270m_config.yaml"
    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Ensure learning rate is properly formatted as float
        config["training"]["learning_rate"] = float(config["training"]["learning_rate"])

        logger.info("Starting training with configuration:")
        logger.info(f"  Model: {config['model']['model_name']}")
        logger.info(f"  Dataset: {config['data']['dataset_path']}")
        logger.info(f"  Learning rate: {config['training']['learning_rate']}")
        logger.info(f"  Epochs: {config['training']['num_train_epochs']}")
        logger.info(f"  Batch size: {config['training']['per_device_train_batch_size']}")

        # Call the refactored training function
        results = train_model(config, trial=None)

        logger.info("Training completed successfully!")
        logger.info(f"Final eval loss: {results.best_eval_loss}")
        if results.best_eval_accuracy is not None:
            logger.info(f"Final eval accuracy: {results.best_eval_accuracy}")

        # Perform inference example (preserving original functionality)
        perform_inference_example(results, config)

        # Save model (preserving original functionality)
        save_model(results, config)

        logger.info("Script completed successfully!")

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def perform_inference_example(results, config):
    """
    Perform a sample inference to demonstrate the trained model.
    Preserves the original inference functionality.
    """
    logger.info("Running inference example...")

    try:
        model = results.model
        tokenizer = results.tokenizer

        # Load a sample from the training dataset for demonstration
        from datasets import load_dataset, Dataset
        dataset_train: Dataset = load_dataset(config["data"]["dataset_path"], split="train") # type: ignore

        # Use a sample from the training data (index 10 as in original)
        sample_idx = min(10, len(dataset_train) - 1)
        sample = dataset_train[sample_idx]

        # Load system prompt
        with open(config["data"]["system_prompt_path"], "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        # Create messages for inference
        marked_original = f"[BEGINNING OF CONTENT]\n{sample['original']}\n[END OF CONTENT]"
        messages = [
            {'role': 'system', 'content': system_prompt},
            {"role": 'user', 'content': marked_original}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ).removeprefix('<bos>')

        logger.info("Inference input:")
        logger.info(f"Original text: {sample['original']}")
        logger.info(f"Expected correction: {sample['corrected']}")
        logger.info("Generated correction:")

        # Generate response
        _ = model.generate(
            **tokenizer(text, return_tensors="pt").to("cuda"),
            max_new_tokens=config["inference"]["max_new_tokens"],
            temperature=config["inference"]["temperature"],
            top_p=config["inference"]["top_p"],
            top_k=config["inference"]["top_k"],
            streamer=TextStreamer(tokenizer, skip_prompt=True),
            cache_implementation=config["inference"]["cache_implementation"],
        )

    except Exception as e:
        logger.warning(f"Inference example failed: {e}")


def save_model(results, config):
    """
    Save the trained model, preserving original functionality.
    """
    logger.info("Saving trained model...")

    try:
        model = results.model
        tokenizer = results.tokenizer

        # Extract model name components for naming
        model_name = config["model"]["model_name"]
        model_parts = model_name.split("/")[-1].split("-")
        model_3_n = model_parts[1] if len(model_parts) > 1 else "gemma3"
        model_params = model_parts[2] if len(model_parts) > 2 else "unknown"
        model_it = model_parts[3] if len(model_parts) > 3 else "unknown"

        model_ft_name = f"gemma-{model_3_n}-{model_params}-{model_it}-gec"

        # Save LoRA adapters locally
        model.save_pretrained(model_ft_name)
        tokenizer.save_pretrained(model_ft_name)
        logger.info(f"Model saved locally to: {model_ft_name}")

        # Log model to MLflow if configured
        try:
            # Create a sample input for model logging
            sample_input = "This is a sample input for model logging."

            mlflow.transformers.log_model(
                transformers_model=model_ft_name,
                artifact_path="model",
                registered_model_name=model_ft_name,
                input_example=sample_input,
                tags=json.loads(os.getenv("MLFLOW_TAGS", "{}"))
            )
            logger.info("Model logged to MLflow successfully")
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")

        # Note about additional saving options
        logger.info(f"LoRA adapters saved to: {model_ft_name}")
        logger.info("To save in other formats (16-bit, 4-bit, GGUF), modify the save options in this function")

    except Exception as e:
        logger.error(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
