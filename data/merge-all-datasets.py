# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Merge multiple HF datasets into a single dataset for training.
# Usage: python merge-all-datasets.py -c clean_dataset -g list_of_grammar_datasets -o output_directory/

import argparse
import os
from datasets import load_dataset, concatenate_datasets


README = """
# Merged Dataset

This dataset combines a clean text dataset (e.g., books converted from EPUB to markdown)
alongside several grammar correction datasets.
The merged dataset is structured to facilitate training and evaluation of language models
for grammar correction tasks.
The dataset is saved in parquet format, with separate files for training and testing splits.

## Sources

- Clean dataset: {clean_dataset}
- Grammar datasets:
{grammar_datasets}

## Format
The dataset contains the following columns:
- `source`: The source of the text (e.g., book title or grammar dataset name)
- `original`: The original text content
- `corrected`: The corrected text content (or a copy of original if not applicable)
"""


def _source(x, source_name):
    if isinstance(x['original'], list):
        return [source_name] * len(x['original'])
    return source_name


def _remove_command(text):
    if isinstance(text, list):
        return [_remove_command(t) for t in text]
    # Remove any leading command like "Correct this: ..."
    segments = text.split(":")
    return ":".join(segments[1:]).strip() if len(segments) > 1 else text.strip()


def _expand_corrections(examples):
    """Expand examples with multiple corrections into separate rows"""
    expanded_sources = []
    expanded_originals = []
    expanded_corrected = []

    for i, original in enumerate(examples['original']):
        corrected_list = examples['corrected'][i] if isinstance(examples['corrected'][i], list) else [examples['corrected'][i]]
        source = examples['source'][i] if 'source' in examples else None

        for corrected in corrected_list:
            expanded_originals.append(original)
            expanded_corrected.append(corrected)
            if source:
                expanded_sources.append(source)

    result = {
        'original': expanded_originals,
        'corrected': expanded_corrected
    }
    if expanded_sources:
        result['source'] = expanded_sources

    return result


def load_clean_dataset(path, split='train'):
    dataset = load_dataset(path, split=split)
    # For clean dataset, original and corrected are the same
    dataset = dataset.map(lambda x: {'original': x['text'], 'corrected': x['text']},
                          batched=True)
    dataset = dataset.remove_columns(['text', 'chapter'])
    return dataset


def load_grammar_dataset(path, split='train'):
    # Each grammar dataset has its own format which we need to handle
    # Grammarly CoEdit
    if path == "grammarly/coedit":
        split = {"train": "train", "test": "validation"}[split]  # CoEdit has no test split
        dataset = load_dataset(path, split=split)
        # We only want gec tasks, to avoid changing the author's voice
        dataset = dataset.filter(lambda x: x['task'] == 'gec')
        # Rename columns to original and corrected
        dataset = dataset.rename_column('src', 'original')
        dataset = dataset.rename_column('tgt', 'corrected')
        dataset = dataset.remove_columns(['_id', 'task'])
        dataset = dataset.map(lambda x: {"source": _source(x, "grammarly/coedit"),
                                         "original": _remove_command(x['original']),
                                         "corrected": x['corrected']}, batched=True)
    # JHU CLSP JFLEG
    elif path == "jhu-clsp/jfleg":
        split = {"train": "validation", "test": "test"}[split]  # JFLEG has no train split
        dataset = load_dataset(path, split=split)
        dataset = dataset.rename_column('sentence', 'original')
        dataset = dataset.rename_column('corrections', 'corrected')
        dataset = dataset.map(lambda x: {"source": _source(x, "jhu-clsp/jfleg")}, batched=True)
        # JFLEG has multiple corrections - expand into separate rows
        dataset = dataset.map(_expand_corrections, batched=True)
    # AgentLans Grammar Correction
    elif path == "agentlans/grammar-correction":
        # grammar-correction has no test split
        split = {"train": "train", "test": "validation"}[split]
        dataset = load_dataset(path, split=split)
        dataset = dataset.rename_column('input', 'original')
        dataset = dataset.rename_column('output', 'corrected')
        dataset = dataset.map(lambda x: {"source": _source(x, "agentlans/grammar-correction")}, batched=True)
    else:
        raise ValueError(f"Unsupported grammar dataset: {path}")

    return dataset


def merge_datasets(clean_dataset_path, grammar_datasets_paths,
                   output_dir, seed):
    """Merge multiple datasets into a single dataset and save as a parquet file.

    Dataset format:
    - source: source of the text (e.g., book title or grammar dataset name)
    - original: the original text content
    - corrected: the corrected text content (one correction per row, expanded from lists)
    - seed: random seed for shuffling the merged dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load train and test splits of datasets separately.
    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        # Load clean dataset
        print(f"Loading clean dataset from {clean_dataset_path}")
        clean_dataset = load_clean_dataset(clean_dataset_path, split=split)
        print(f"Clean dataset loaded with {len(clean_dataset)} samples.") # type: ignore

        all_datasets = [clean_dataset]

        for grammar_path in grammar_datasets_paths:
            print(f"Loading grammar dataset from {grammar_path}")
            grammar_dataset = load_grammar_dataset(grammar_path, split=split)
            print(f"Grammar dataset loaded with {len(grammar_dataset)} samples.") # type: ignore
            all_datasets.append(grammar_dataset)

        print("Concatenating datasets...")
        merged_dataset = concatenate_datasets(all_datasets) # type: ignore
        print(f"Merged dataset has {len(merged_dataset)} samples.")

        print("Shuffling merged dataset...")
        merged_dataset = merged_dataset.shuffle(seed=seed)
        merged_dataset.flatten_indices()

        output_path = os.path.join(output_dir, split, 'merged_dataset.parquet')
        print(f"Saving merged dataset to {output_path}")
        merged_dataset.to_parquet(output_path)
        print(f"Merged {split} dataset saved successfully.")
    print("All datasets merged and saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple HF datasets into a single dataset.")
    parser.add_argument('-c', '--clean_dataset', required=True, help='Path to the clean dataset (e.g., books dataset).')
    parser.add_argument('-g', '--grammar_datasets', required=True, help='File containing list of grammar datasets to include.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save the merged dataset.')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for shuffling the merged dataset.')
    return parser.parse_args()


def main():
    args = parse_args()
    clean_dataset = args.clean_dataset
    with open(args.grammar_datasets, 'r') as f:
        grammar_datasets = [line.strip() for line in f
                            if line.strip() and not line.startswith('#')]
    output_dir = args.output_dir
    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(README.format(clean_dataset=clean_dataset,
                              grammar_datasets='\n'.join([f"- {d}" for d in grammar_datasets])))

    merge_datasets(clean_dataset, grammar_datasets,
                   output_dir, seed)


if __name__ == "__main__":
    main()
