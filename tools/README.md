# Grammar Correction Tools

This directory contains utilities for working with trained grammar correction models.

## Available Tools

### 1. `convert_to_ollama.py`

Converts trained LoRA adapters or merged models to GGUF format for Ollama deployment.

**Features:**
- Supports multiple quantization methods (q8_0, q4_k_m, q4_k_s)
- Auto-generates Ollama Modelfile with system prompt
- Handles both LoRA adapters and full models
- Fallback to PyTorch format when GGUF conversion fails

**Usage Examples:**
```bash
# Convert with q4_k_m quantization (recommended)
python tools/convert_to_ollama.py gemma-3-4b-pt-gec --quant q4_k_m

# Convert with q8_0 quantization
python tools/convert_to_ollama.py gemma-3-4b-pt-gec --quant q8_0

# Custom Ollama name and output directory
python tools/convert_to_ollama.py ./my_model --name my-corrector --output ./ollama_models
```

**Requirements:**
- GGUF conversion requires llama.cpp to be installed
- After compiling llama.cpp from source in the root directory, run 
    `from unsloth_zoo.llama_cpp import _download_convert_hf_to_gguf; _download_convert_hf_to_gguf()`
- Available quantization options: q8_0 (default), q4_k_m, q4_k_s

### 2. `evaluate_inference.py`

Evaluates model performance on dataset samples with detailed comparison outputs.

**Features:**
- Multiple model loading options (local, HuggingFace, base+adapters)
- Dataset filtering by source (books, grammar datasets, etc.)
- Random sampling with optional seed for reproducibility
- Detailed accuracy metrics and example comparisons
- JSON output for detailed results

**Usage Examples:**
```bash
# List available dataset sources
python tools/evaluate_inference.py --list-sources

# Evaluate local model on 50 samples
python tools/evaluate_inference.py --model ./my-model --samples 50

# Evaluate HuggingFace model on specific sources
python tools/evaluate_inference.py --hf-model unsloth/gemma-3-4b-pt-unsloth-bnb-4bit --sources "grammarly/coedit" "jfleg"

# Evaluate with base model + adapters
python tools/evaluate_inference.py --base-model unsloth/gemma-3-4b-pt-unsloth-bnb-4bit --adapters ./adapters

# Save detailed results to JSON
python tools/evaluate_inference.py --model ./my-model --output results.json
```

## Dataset Sources

Available sources in the dataset include:
- `grammarly/coedit` - Grammar correction dataset
- `jhu-clsp/jfleg` - JFLEG grammar correction
- `agentlans/grammar-correction` - Grammar correction dataset
- Various book sources (e.g., `epubs/jane-austen_emma`)

## Integration with Ollama

After using `convert_to_ollama.py`:

1. Navigate to output directory: `cd ./ollama_models`
2. Create Ollama model: `ollama create grammar-corrector -f Modelfile`
3. Run the model: `ollama run grammar-corrector`

## Notes

- Both utilities use Unsloth for model loading for consistency
- The evaluation utility applies the same chat template used during training
- Models are loaded with 4-bit quantization by default for memory efficiency
- Generation parameters can be customized via command-line arguments

## Troubleshooting

- **GGUF conversion fails**: Try different quantization options or ensure llama.cpp is properly installed
- **Memory issues**: Reduce batch size or use smaller models for evaluation
- **Chat template issues**: Ensure the system prompt path is correct (`config/prompt.txt`)
