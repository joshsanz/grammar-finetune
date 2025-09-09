# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Format the text pairs from a merged dataset into a format suitable for the MLX LoRA example.
# Usage: python create-dataset-for-mlx.py -i input_directory/ -o output_directory/ [-p prompt_file]

import argparse
import os
from datasets import load_dataset
import json
import warnings
import shutil


def convert_to_chatml(system, user, asst):
    chat = {
        "conversations": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst}
        ]
    }
    return json.dumps(chat, ensure_ascii=False)


def format_for_mlx(example, prompt):
    # Prepare the input text with the prompt
    input_text = "[BEGINNING OF CONTENT]\n" + example['original'] + "\n[END OF CONTENT]"
    # The output is the corrected text (now single values per row)
    output_text = example['corrected']
    return {"text": convert_to_chatml(prompt, input_text, output_text)}


def create_mlx_dataset(input_dir, output_dir, prompt_file):
    os.makedirs(output_dir, exist_ok=True)

    # Load prompt
    if prompt_file and os.path.isfile(prompt_file):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        prompt = "You are a helpful assistant that corrects grammar and spelling mistakes in the provided text."
        warnings.warn(f"No prompt file provided or file does not exist [{prompt_file}]. "
                       f"Using default prompt: {prompt}.")

    # Load dataset
    print(f"Loading dataset from {input_dir}")
    for split in ['train', 'test']:
        dataset = load_dataset(input_dir, split=split)

        formatted_dataset = dataset.map(lambda x: format_for_mlx(x, prompt),
                                        remove_columns=dataset.column_names)  # type: ignore

        output_path = os.path.join(output_dir, f'{split}.jsonl')
        print(f"Saving formatted dataset to {output_path}")
        formatted_dataset.to_json(output_path)  # type: ignore

    # MLX expects test.jsonl to be named valid.jsonl
    shutil.move(os.path.join(output_dir, 'test.jsonl'), os.path.join(output_dir, 'valid.jsonl'))
    print(f"Formatted dataset created at {output_dir}/(train|valid).jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Format a dataset for MLX LoRA example.")
    parser.add_argument('-i', '--input_dir', required=True, help='Input directory containing the dataset.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save the formatted dataset.')
    parser.add_argument('-p', '--prompt_file', default="config/prompt.txt", help='File containing the system prompt to use.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    prompt_file = args.prompt_file

    create_mlx_dataset(input_dir, output_dir, prompt_file)


if __name__ == "__main__":
    main()
