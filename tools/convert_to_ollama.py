#!/usr/bin/env python3
"""
Convert trained LoRA adapters to Ollama models with quantization.

This utility merges LoRA adapters with their base models and creates Ollama-compatible
models with quantization. It generates a Modelfile with the appropriate system prompt
and configuration for grammar error correction tasks.

Example usage:
    python convert_to_ollama.py gemma-3-4b-pt-gec --quant q8_0
    python convert_to_ollama.py ./my_model --quant q4_k_m --name my-corrector
"""

from unsloth import FastModel  # Ensure unsloth is imported before transformers
import argparse
import os
import sys
import logging
import shutil
import subprocess
import coloredlogs

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger("ConvertToOllama")


def read_system_prompt(prompt_path: str) -> str:
    """
    Read the system prompt from the specified file.

    Args:
        prompt_path: Path to the system prompt file

    Returns:
        The system prompt text
    """
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to read system prompt: {e}")
        raise


def validate_model_path(model_path: str) -> bool:
    """
    Validate that the model path exists and contains required files.

    Checks for either LoRA adapter files (adapter_config.json, adapter_model.safetensors)
    or full model files (config.json, model.safetensors).

    Args:
        model_path: Path to the model directory to validate

    Returns:
        True if the path contains valid model or adapter files, False otherwise
    """
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return False

    # Check for adapter files (LoRA) or model files
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    model_files = ["config.json", "model.safetensors"]

    has_adapters = all(os.path.exists(os.path.join(model_path, f)) for f in adapter_files)
    has_model = all(os.path.exists(os.path.join(model_path, f)) for f in model_files)

    if not (has_adapters or has_model):
        logger.error(f"Model path does not contain valid adapter or model files: {model_path}")
        return False

    return True


def generate_modelfile(
    output_path: str,
    model_path: str,
    system_prompt: str,
    temperature: float = 0.3,
    top_p: float = 0.95,
    top_k: int = 64
) -> str:
    """
    Generate an Ollama Modelfile with the system prompt and parameters.

    Args:
        output_path: Directory where the Modelfile will be saved
        model_path: Path to the merged model directory
        system_prompt: System prompt text for grammar correction
        temperature: Sampling temperature (default: 0.3)
        top_p: Top-p sampling parameter (default: 0.95)
        top_k: Top-k sampling parameter (default: 64)

    Returns:
        Modelfile content as a string
    """
    model_rel_path = os.path.relpath(model_path, output_path)
    modelfile_content = f"""FROM {model_rel_path}/

SYSTEM \"\"\"
{system_prompt}
\"\"\"

TEMPLATE \"\"\"{{{{- range $i, $_ := .Messages }}}}
{{{{- $last := eq (len (slice $.Messages $i)) 1 }}}}
{{{{- if or (eq .Role "user") (eq .Role "system") }}}}<start_of_turn>user
{{{{ .Content }}}}<end_of_turn>
{{{{ if $last }}}}<start_of_turn>model
{{{{ end }}}}
{{{{- else if eq .Role "assistant" }}}}<start_of_turn>model
{{{{ .Content }}}}{{{{ if not $last }}}}<end_of_turn>
{{{{ end }}}}
{{{{- end }}}}
{{{{- end }}}}
\"\"\"

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}

# Grammar correction specific parameters
PARAMETER stop <end_of_turn>
"""

    return modelfile_content


def convert_and_create_ollama(
    model_path: str,
    output_path: str,
    ollama_name: str = "grammar-corrector"
) -> tuple[str, str]:
    """
    Load LoRA model and merge adapters to create a merged model for Ollama.

    This function loads a LoRA-adapted model using Unsloth, merges the adapters
    with the base model using PEFT's merge_and_unload method, and saves the
    merged model and tokenizer to the specified output directory.

    Args:
        model_path: Path to the LoRA adapter directory
        output_path: Directory to save the merged model
        ollama_name: Name for the output model directory

    Returns:
        Tuple of (merged_model_path, ollama_model_name)

    Raises:
        Exception: If model loading or merging fails
    """
    logger.info(f"Loading model from: {model_path}")

    # Load the model
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,  # Default sequence length
            load_in_4bit=False,  # Load in full precision before merging
            load_in_8bit=False,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Merge LoRA adapters using PEFT implementation
    merged_model_dir = ollama_name
    merged_model_path = os.path.join(output_path, merged_model_dir)

    logger.info("Merging LoRA adapters using PEFT implementation")
    logger.info(f"Merged model output directory: {merged_model_path}")

    try:
        # Merge adapters with base model using PEFT
        merged_model = model.merge_and_unload(safe_merge=True)

        # Save merged model and tokenizer
        merged_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)

        logger.info("Merged model and tokenizer saved successfully")

    except Exception as e:
        logger.error(f"Model merging failed: {e}")
        raise

    return merged_model_path, ollama_name


def main():
    """Main function for the conversion utility."""

    parser = argparse.ArgumentParser(
        description="Convert trained models to GGUF format for Ollama deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert LoRA adapters with q8_0 quantization
  python convert_to_ollama.py gemma-3-4b-pt-gec --quant q8_0

  # Convert to q4_k_m with custom Ollama name
  python convert_to_ollama.py ./my_model --quant q4_k_m --name my-corrector

  # Specify custom output directory and system prompt
  python convert_to_ollama.py ./model --output ./ollama_models --prompt ./custom_prompt.txt

Supported quantization methods:
  - q8_0: 8-bit quantization (higher quality, larger size)
  - q4_k_m: 4-bit quantization (recommended for most use cases)
  - q4_k_s: 4-bit quantization (smaller, faster variant)
        """
    )

    parser.add_argument(
        "model_path",
        help="Path to the trained model or LoRA adapter directory"
    )

    parser.add_argument(
        "--quant", "-q",
        choices=["q8_0", "q4_k_m", "q4_k_s"],
        default="q8_0",
        help="Quantization method (default: q8_0)"
    )

    parser.add_argument(
        "--name", "-n",
        default="grammar-corrector",
        help="Name for the Ollama model (default: grammar-corrector)"
    )

    parser.add_argument(
        "--output", "-o",
        default="./ollama_models",
        help="Output directory for GGUF and Modelfile (default: ./ollama_models)"
    )

    parser.add_argument(
        "--prompt", "-p",
        default="config/prompt.txt",
        help="Path to system prompt file (default: config/prompt.txt)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for Modelfile (default: 0.3)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Top-k sampling parameter (default: 64)"
    )

    args = parser.parse_args()

    try:
        # Validate inputs
        if not validate_model_path(args.model_path):
            sys.exit(1)

        # Read system prompt
        logger.info(f"Reading system prompt from: {args.prompt}")
        system_prompt = read_system_prompt(args.prompt)

        # Convert and merge model
        logger.info("Starting model conversion...")
        merged_model_path, ollama_model_name = convert_and_create_ollama(
            model_path=args.model_path,
            output_path=args.output,
            ollama_name=args.name
        )

        # Generate Modelfile
        logger.info("Generating Ollama Modelfile...")
        modelfile_content = generate_modelfile(
            output_path=args.output,
            model_path=merged_model_path,
            system_prompt=system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )

        # Save Modelfile
        modelfile_path = os.path.join(args.output, "Modelfile")
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)

        logger.info(f"Modelfile saved to: {modelfile_path}")

        # Create Ollama model with quantization
        logger.info(f"Creating Ollama model with quantization: {args.quant}")
        ollama_cmd = [
            "ollama", "create", f"{args.name}:{args.quant}",
            "-f", os.path.abspath(modelfile_path),
            "--quantize", args.quant
        ]

        try:
            result = subprocess.run(ollama_cmd, check=True, capture_output=True, text=True,
                                    cwd=args.output, env=os.environ.copy())
            logger.info(f"Ollama model created successfully: {args.name}:{args.quant}")
            if result.stdout:
                logger.info(f"Ollama output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create Ollama model: {e}")
            logger.error(f"Command: {' '.join(ollama_cmd)}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            # Don't exit, still show success message for the conversion part

        # Success message with usage instructions
        logger.info("Conversion completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"1. Run the model: ollama run {args.name}:{args.quant}")
        logger.info("")
        logger.info("Files created:")
        logger.info(f"  - Merged model: {merged_model_path}")
        logger.info(f"  - Modelfile: {modelfile_path}")
        logger.info(f"  - Ollama model: {args.name}:{args.quant}")

    except KeyboardInterrupt:
        logger.info("Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()