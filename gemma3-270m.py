# -*- coding: utf-8 -*-

from unsloth import FastModel # Must import first to patch other libraries
import sys
import torch
import yaml
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only

# Load configuration
with open("config/gemma3_270m_config.yaml", "r") as f:
    config = yaml.safe_load(f)
    config["training"]["learning_rate"] = float(config["training"]["learning_rate"])

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = config["model"]["model_name"],
    max_seq_length = config["model"]["max_seq_length"], # Choose any for long context!
    load_in_4bit = config["model"]["load_in_4bit"],  # 4 bit quantization to reduce memory
    load_in_8bit = config["model"]["load_in_8bit"], # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = config["model"]["full_finetuning"], # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# We now add LoRA adapters so we only need to update a small amount of parameters!

model = FastModel.get_peft_model(
    model,
    r = config["peft"]["r"], # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = config["peft"]["target_modules"],
    lora_alpha = config["peft"]["lora_alpha"],
    lora_dropout = config["peft"]["lora_dropout"], # Supports any, but = 0 is optimized
    bias = config["peft"]["bias"],    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = config["peft"]["use_gradient_checkpointing"], # True or "unsloth" for very long context
    random_state = config["peft"]["random_state"],
    use_rslora = config["peft"]["use_rslora"],  # We support rank stabilized LoRA
    loftq_config = config["peft"]["loftq_config"], # And LoftQ
)


### Data Prep
# We now use the `Gemma-3` format for conversation style finetunes. We use [Thytu's ChessInstruct](https://huggingface.co/datasets/Thytu/ChessInstruct) dataset. Gemma-3 renders multi turn conversations like below:

# ```
# <bos><start_of_turn>user
# Hello!<end_of_turn>
# <start_of_turn>model
# Hey there!<end_of_turn>
# ```

# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` and more.


tokenizer = get_chat_template(
    tokenizer,
    chat_template = config["data"]["chat_template"],
)

dataset = load_dataset(config["data"]["dataset_path"])

# We now use `convert_to_chatml` to try converting datasets to the correct format for finetuning purposes!

# Load system prompt from config file
with open(config["data"]["system_prompt_path"], "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

def convert_to_chatml(example):
    # Add content markers around original text
    marked_original = f"[BEGINNING OF CONTENT]\n{example['original']}\n[END OF CONTENT]"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": marked_original},
            {"role": "assistant", "content": example["corrected"]}
        ]
    }

dataset = dataset.map(convert_to_chatml).remove_columns(["source", "original", "corrected"])

# Let's see how row 100 looks!
print("dataset[100]:")
print(dataset["train"][100])

# We now have to apply the chat template for `Gemma3` onto the conversations, and save it to `text`.

def formatting_prompts_func(examples):
   convos = examples["messages"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# Let's see how the chat template did!

print("dataset[100]['text']:")
print(dataset["train"][100]['text'])

### Train the model
# Now let's train our model. We do 100 steps to speed things up, but you can set
# `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["test[:2000]"], # Evaluate on first 2000 samples of test set
    args = SFTConfig(
        # Data and batching
        dataset_text_field = config["training"]["dataset_text_field"],
        max_length = config["training"]["max_length"], # Ensure this is <= model max length
        # TODO: packing for inputs
        per_device_train_batch_size = config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size = config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"], # Use GA to mimic batch size!
        # Training length
        warmup_steps = config["training"]["warmup_steps"],
        num_train_epochs = config["training"]["num_train_epochs"], # Set this for 1 full training run.
        max_steps = config["training"]["max_steps"],
        eval_strategy = config["training"]["eval_strategy"], # "steps" or "epoch" or "no"
        eval_steps = config["training"]["eval_steps"], # Eval every N steps if using "steps"
        # Optimizer settings
        learning_rate = config["training"]["learning_rate"], # Reduce to 2e-5 for long training runs
        optim = config["training"]["optim"],
        weight_decay = config["training"]["weight_decay"],
        lr_scheduler_type = config["training"]["lr_scheduler_type"],
        logging_steps = config["training"]["logging_steps"],
        seed = config["training"]["seed"],
        # Saving and logging
        output_dir = config["training"]["output_dir"],
        report_to = config["training"]["report_to"], # Use this for WandB etc
    ),
)

# We also use Unsloth's `train_on_completions` method to only train on the assistant
# outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!
trainer = train_on_responses_only(
    trainer,
    instruction_part = config["masking"]["instruction_part"],
    response_part = config["masking"]["response_part"],
)

# Let's verify masking the instruction part is done! Let's print the 100th row again.
tokenizer.decode(trainer.train_dataset[100]["input_ids"])

# Now let's print the masked out example - you should see only the answer is present:
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Let's train the model! To resume a training run, set
# `trainer.train(resume_from_checkpoint = True)`
trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

### Inference
# Let's run the model via Unsloth native inference!
# According to the `Gemma-3` team, the recommended settings for inference are
# `temperature = 1.0, top_p = 0.95, top_k = 64`


messages = [
    {'role': 'system','content': dataset['train']['messages'][10][0]['content']},
    {"role" : 'user', 'content' : dataset['train']['messages'][10][1]['content']}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True, # Must add for generation
).removeprefix('<bos>')

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = config["inference"]["max_new_tokens"],
    temperature = config["inference"]["temperature"], top_p = config["inference"]["top_p"], top_k = config["inference"]["top_k"],
    streamer = TextStreamer(tokenizer, skip_prompt = True),
    cache_implementation = config["inference"]["cache_implementation"], # Pass cache implementation here
)

### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub`
# for an online save or `save_pretrained` for a local save.

# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF,
# scroll down!

model.save_pretrained("gemma3-270m")  # Local saving
tokenizer.save_pretrained("gemma3-270m")
# model.push_to_hub("your_name/gemma3-270m", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/gemma3-270m", token = "...") # Online saving

# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma-3", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = False,
    )

### Saving to float16 for VLLM
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit`
# for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to
# your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your
# personal tokens.


# Merge to 16bit
if False:
    model.save_pretrained_merged("gemma-3-finetune", tokenizer, save_method = "merged_16bit")
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gemma-3-finetune", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False:
    model.save_pretrained_merged("gemma-3-finetune", tokenizer, save_method = "merged_4bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/gemma-3-finetune", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("gemma-3-finetune")
    tokenizer.save_pretrained("gemma-3-finetune")
if False: # Pushing to HF Hub
    model.push_to_hub("hf/gemma-3-finetune", token = "")
    tokenizer.push_to_hub("hf/gemma-3-finetune", token = "")

### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now,
# you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma-3-finetune",
        tokenizer,
        quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )

# Likewise, if you want to instead push to GGUF to your Hugging Face account, set
# `if False` to `if True` and add your Hugging Face token and upload location!

if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "gemma-3-finetune",
        tokenizer,
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = "HF_ACCOUNT/gemma-finetune-gguf",
        token = "hf_...",
    )
