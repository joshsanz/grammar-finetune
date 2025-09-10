# Grammar Error Correction Fine Tuning

## Installation

```sh
uv venv
uv sync
source .venv/bin/activate
```

## Dataset

Add any books in `.epub` format you wish to have included as positive examples of formatting
and grammatically correct text under `data/`.
Once added, convert them into the expected markdown format with

```sh
cd data/
python epub-to-clean-md.py input.epub output_directory/
```

for each book. Then combine all the books into a single HF Dataset with

```sh
python books-to-dataset.py -o books/ -i input1/ \[input2/ ...]
```

Finally, combine the clean examples dataset with grammatical error datasets in
`config/datasets.txt` with

```sh
python merge-all-datasets.py --clean books/ --grammar ../config/datasets.txt --out output_dir/
```

**Automated Workflow:**
Run the complete dataset creation pipeline with:
```sh
./create-dataset.sh
```
This script handles EPUB conversion, dataset creation, and merging automatically.

## Fine Tuning

### Unsloth Training
The project uses Unsloth for efficient fine-tuning with LoRA (Low-Rank Adaptation) adapters. Training scripts are configured via YAML files in the `config/` directory.

**Available Models:**
- `gemma3-270m.py` - Training script for Gemma 270M model
- `gemma3n-4b-conversational.py` - Training script for Gemma 4B conversational model

**Key Features:**
- 4-bit quantization for memory efficiency
- LoRA adapters to reduce trainable parameters
- Response-only training (ignores user input loss)
- Support for multiple model architectures (Gemma-3, Llama, Mistral, etc.)

**Training Configuration:**
Training parameters are specified in YAML config files (e.g., `config/gemma3_270m_config.yaml`):
- Model settings: quantization, sequence length, LoRA parameters
- Training hyperparameters: learning rate, batch size, epochs
- Data processing: chat template, system prompt path
- Output and evaluation settings
- MLOps logging, e.g. WandB or MLflow

To enable MLflow with your hosted server, set environment vars with

```sh
source mlflow_environment
```

**Data Format:**
The training uses ChatML format with conversation structure:
```
<start_of_turn>user
[BEGINNING OF CONTENT]
{original_text_with_errors}
[END OF CONTENT]
<end_of_turn>
<start_of_turn>model
{corrected_text}
<end_of_turn>
```

**Model Saving Options:**
- LoRA adapters only (lightweight)
- Merged 16-bit models for VLLM
- Merged 4-bit quantized models
- GGUF format for llama.cpp compatibility

### MLX Training
For MLX framework support, convert datasets to JSONL format:
```sh
python create-dataset-for-mlx.py -i data/gec-dataset -o data.mlx -p config/prompt.txt
```

The MLX format uses ChatML structure with system/user/assistant roles for conversation-based training.
