#!/usr/bin/env python3
"""
Evaluate model inference on dataset samples.

This utility tests model performance on samples from the grammar correction dataset,
supporting various model types and providing detailed comparison outputs.
"""

import unsloth  # Ensure unsloth is imported before transformers
import argparse
import json
import os
import sys
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import coloredlogs
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger("EvaluateInference")


class ModelLoader:
    """Handle loading different types of models for inference."""

    @staticmethod
    def load_huggingface_model(model_name: str, max_seq_length: int = 2048):
        """Load a model from Hugging Face Hub using Unsloth for compatibility."""
        logger.info(f"Loading Hugging Face model: {model_name}")

        try:
            # Use Unsloth to load HF models for better compatibility
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Switch model to inference mode
            FastLanguageModel.for_inference(model)
            logger.info("Model configured for inference")

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise

    @staticmethod
    def load_unsloth_model(model_path: str, max_seq_length: int = 2048):
        """Load a model using Unsloth (local model or adapters)."""
        logger.info(f"Loading Unsloth model: {model_path}")

        try:
            # Check if this is an adapter directory by looking for adapter_config.json
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                # This is a LoRA adapter directory, need to load base model + adapters
                logger.info("Detected LoRA adapter directory, loading base model + adapters")

                # Read adapter config to get base model
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)

                base_model_name = adapter_config.get("base_model_name_or_path")
                if not base_model_name:
                    raise ValueError("Could not find base_model_name_or_path in adapter_config.json")

                logger.info(f"Base model: {base_model_name}")

                # Load base model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model_name,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )

                # Load adapters
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, model_path)
                logger.info("Adapters loaded successfully")

            else:
                # Regular model loading
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=True,
                )

            # Switch model to inference mode
            FastLanguageModel.for_inference(model)
            logger.info("Model configured for inference")

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load Unsloth model: {e}")
            raise

    @staticmethod
    def load_model_with_adapters(base_model: str, adapter_path: str, max_seq_length: int = 2048):
        """Load a base model and apply LoRA adapters."""
        logger.info(f"Loading base model {base_model} with adapters from {adapter_path}")

        try:
            # Load base model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )

            # Load adapters
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)

            # Switch model to inference mode
            FastLanguageModel.for_inference(model)
            logger.info("Model configured for inference")

            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load model with adapters: {e}")
            raise


class DatasetManager:
    """Handle dataset operations and filtering."""

    def __init__(self, dataset_path: str):
        """Initialize with dataset path."""
        self.dataset_path = dataset_path
        self._dataset = None
        self._sources = None

    def load_dataset(self) -> Dataset:
        """Load the dataset."""
        if self._dataset is None:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            try:
                self._dataset = load_dataset(self.dataset_path, split="train")
                logger.info(f"Dataset loaded: {len(self._dataset)} samples")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise
        return self._dataset

    def get_sources(self) -> Set[str]:
        """Get all unique sources in the dataset."""
        if self._sources is None:
            dataset = self.load_dataset()
            self._sources = set(dataset["source"])
            logger.info(f"Found {len(self._sources)} unique sources")
        return self._sources

    def filter_by_source(self, sources: List[str]) -> Dataset:
        """Filter dataset by source(s)."""
        dataset = self.load_dataset()

        if not sources:
            return dataset

        logger.info(f"Filtering dataset by sources: {sources}")

        # Validate sources exist
        available_sources = self.get_sources()
        invalid_sources = set(sources) - available_sources
        if invalid_sources:
            logger.warning(f"Invalid sources will be ignored: {invalid_sources}")
            sources = [s for s in sources if s in available_sources]

        if not sources:
            logger.error("No valid sources specified")
            return dataset

        # Filter dataset
        filtered = dataset.filter(lambda x: x["source"] in sources)
        logger.info(f"Filtered dataset: {len(filtered)} samples")

        return filtered

    def sample_dataset(self, dataset: Dataset, n_samples: int, seed: Optional[int] = None) -> Dataset:
        """Randomly sample N examples from the dataset."""
        if n_samples >= len(dataset):
            logger.info(f"Requested {n_samples} samples, but dataset only has {len(dataset)}. Using all samples.")
            return dataset

        if seed is not None:
            random.seed(seed)
            logger.info(f"Using random seed: {seed}")

        indices = random.sample(range(len(dataset)), n_samples)
        sampled = dataset.select(indices)

        logger.info(f"Sampled {len(sampled)} examples from dataset")
        return sampled


class InferenceRunner:
    """Handle model inference and evaluation."""

    def __init__(self, model, tokenizer, system_prompt_path: str):
        """Initialize with model, tokenizer, and system prompt."""
        self.model = model
        self.tokenizer = tokenizer

        # Load system prompt
        try:
            with open(system_prompt_path, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load system prompt from {system_prompt_path}: {e}")
            self.system_prompt = "You are a professional copy editor. Correct the following text for grammar, spelling, and punctuation while preserving the author's style."

        # Setup chat template
        try:
            self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma3")
        except Exception as e:
            logger.warning(f"Failed to set chat template: {e}")

    def generate_correction(
        self,
        original_text: str,
        max_new_tokens: int = 128,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 64
    ) -> str:
        """Generate a correction for the given text."""

        # Format input using chat template
        marked_original = f"[BEGINNING OF CONTENT]\n{original_text}\n[END OF CONTENT]"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": marked_original}
        ]

        try:
            # Apply chat template (following working example pattern)
            formatted_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ).removeprefix('<bos>')

            # Debug: Log the formatted input
            logger.debug(f"Formatted input: {formatted_input}")

            # Tokenize input
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)

            # Generate response
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            # Decode only the generated part
            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[ERROR: {str(e)}]"

    def evaluate_samples(self, dataset: Dataset, **generation_kwargs) -> List[Dict[str, Any]]:
        """Evaluate model on dataset samples."""
        results = []

        logger.info(f"Starting evaluation on {len(dataset)} samples...")

        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}")

            original = sample["original"]
            expected = sample["corrected"]
            source = sample["source"]

            # Generate correction
            generated = self.generate_correction(original, **generation_kwargs)

            # Store result
            result = {
                "index": i,
                "source": source,
                "original": original,
                "expected": expected,
                "generated": generated,
                "matches_expected": generated.strip() == expected.strip()
            }

            results.append(result)

            # Show progress
            if (i + 1) % 10 == 0 or i == len(dataset) - 1:
                accuracy = sum(r["matches_expected"] for r in results) / len(results)
                logger.info(f"Progress: {i+1}/{len(dataset)}, Current accuracy: {accuracy:.2%}")

        return results


def print_results_summary(results: List[Dict[str, Any]]):
    """Print a summary of evaluation results."""
    total_samples = len(results)
    matches = sum(r["matches_expected"] for r in results)
    accuracy = matches / total_samples if total_samples > 0 else 0.0

    # Group by source
    source_stats = {}
    for result in results:
        source = result["source"]
        if source not in source_stats:
            source_stats[source] = {"total": 0, "matches": 0}
        source_stats[source]["total"] += 1
        source_stats[source]["matches"] += result["matches_expected"]

    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total samples: {total_samples}")
    print(f"Exact matches: {matches}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print()

    print("Per-source breakdown:")
    for source, stats in sorted(source_stats.items()):
        source_accuracy = stats["matches"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"  {source}: {stats['matches']}/{stats['total']} ({source_accuracy:.2%})")


def print_sample_comparisons(results: List[Dict[str, Any]], n_examples: int = 3):
    """Print detailed comparisons for a few examples."""
    print("\n" + "="*60)
    print("SAMPLE COMPARISONS")
    print("="*60)

    # Show mix of correct and incorrect examples
    correct_examples = [r for r in results if r["matches_expected"]]
    incorrect_examples = [r for r in results if not r["matches_expected"]]

    examples_to_show = []
    examples_to_show.extend(correct_examples[:n_examples//2])
    examples_to_show.extend(incorrect_examples[:n_examples - len(examples_to_show)])

    for i, result in enumerate(examples_to_show):
        status = "✓" if result["matches_expected"] else "✗"
        print(f"\nExample {i+1} {status} (Source: {result['source']})")
        print("-" * 40)
        print(f"Original:  {result['original']}")
        print(f"Expected:  {result['expected']}")
        print(f"Generated: {result['generated']}")


def save_results(results: List[Dict[str, Any]], output_path: str, format: str = "json"):
    """Save results to file."""
    logger.info(f"Saving results to: {output_path}")

    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        logger.error(f"Unsupported output format: {format}")


def main():
    """Main function for the evaluation utility."""

    parser = argparse.ArgumentParser(
        description="Evaluate model inference on grammar correction dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate local model on 50 random samples
  python evaluate_inference.py --model ./my-model --samples 50

  # Evaluate HF model on specific sources
  python evaluate_inference.py --hf-model unsloth/gemma-3-4b-pt-unsloth-bnb-4bit --sources "grammarly/coedit" "jfleg"

  # Evaluate with base model + adapters
  python evaluate_inference.py --base-model unsloth/gemma-3-4b-pt-unsloth-bnb-4bit --adapters ./adapters

  # List available sources in dataset
  python evaluate_inference.py --list-sources
        """
    )

    # Model specification (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument(
        "--model", "-m",
        help="Path to local model directory (Unsloth format) or LoRA adapter directory (auto-detects base model)"
    )
    model_group.add_argument(
        "--hf-model",
        help="Hugging Face model name/path"
    )
    model_group.add_argument(
        "--base-model",
        help="Base model name (use with --adapters)"
    )

    parser.add_argument(
        "--adapters", "-a",
        help="Path to LoRA adapter directory (use with --base-model)"
    )

    # Dataset options
    parser.add_argument(
        "--dataset", "-d",
        default="data/gec-dataset",
        help="Path to dataset (default: data/gec-dataset)"
    )

    parser.add_argument(
        "--sources", "-s",
        nargs="*",
        help="Filter by source(s). Use --list-sources to see available options"
    )

    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List all available sources in the dataset and exit"
    )

    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for sample selection"
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-k sampling (default: 64)"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file to save detailed results (JSON format)"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of example comparisons to show (default: 3)"
    )

    parser.add_argument(
        "--system-prompt",
        default="config/prompt.txt",
        help="Path to system prompt file (default: config/prompt.txt)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for troubleshooting"
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize dataset manager
        dataset_manager = DatasetManager(args.dataset)

        # Handle --list-sources
        if args.list_sources:
            sources = dataset_manager.get_sources()
            print("Available sources in dataset:")
            for source in sorted(sources):
                print(f"  - {source}")
            return

        # Validate model arguments
        if args.base_model and not args.adapters:
            parser.error("--base-model requires --adapters")

        if not any([args.model, args.hf_model, args.base_model]):
            parser.error("Must specify one of: --model, --hf-model, or --base-model")

        # Load model
        if args.model:
            model, tokenizer = ModelLoader.load_unsloth_model(args.model)
        elif args.hf_model:
            model, tokenizer = ModelLoader.load_huggingface_model(args.hf_model)
        elif args.base_model:
            model, tokenizer = ModelLoader.load_model_with_adapters(args.base_model, args.adapters)

        # Load and filter dataset
        dataset = dataset_manager.filter_by_source(args.sources or [])
        dataset = dataset_manager.sample_dataset(dataset, args.samples, args.seed)

        # Run inference
        runner = InferenceRunner(model, tokenizer, args.system_prompt)
        results = runner.evaluate_samples(
            dataset,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        # Print results
        print_results_summary(results)
        print_sample_comparisons(results, args.examples)

        # Save detailed results if requested
        if args.output:
            save_results(results, args.output)

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Add import for torch here to avoid issues
    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required but not installed")
        sys.exit(1)

    main()