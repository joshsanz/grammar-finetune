#!/usr/bin/env python3
"""
Subsample a random fraction of train and test datasets.

Usage:
    python subsample-dataset.py -f 0.1 -i data/gec-dataset -o data/small-dataset
"""

import argparse
import math
from pathlib import Path
from datasets import load_dataset, DatasetDict


def subsample_dataset(fraction: float, input_path: str, output_path: str, seed: int = 42):
    """Subsample train and test datasets by the given fraction."""
    input_dir = Path(input_path)
    output_dir = Path(output_path)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_in: DatasetDict = load_dataset(str(input_dir)) # type: ignore

    # Copy over README if exists
    readme_path = input_dir / 'README.md'
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f_in:
            with open(output_dir / 'README.md', 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())

    # Subsample each split
    for split in dataset_in.keys():
        print(f"Found split: {split}")
        ds_split = dataset_in[split]
        print(f" - {len(ds_split)} samples")

        # Shuffle and subsample
        shuffled_split = ds_split.shuffle(seed=seed)
        total_split = len(shuffled_split)
        subset_size = int(math.ceil(fraction * total_split))

        subsampled_split = shuffled_split.filter(
            lambda example, idx: idx < subset_size,
            with_indices=True
        )
        print(f" - Downsampled to {len(subsampled_split)} samples")

        # Save subsampled dataset
        subsampled_split.flatten_indices()
        subsampled_split.to_parquet(str(output_dir / split / 'dataset.parquet'))


def main():
    parser = argparse.ArgumentParser(description="Subsample train and test datasets")
    parser.add_argument("-f", "--fraction", type=float, required=True,
                        help="Fraction of dataset to keep (e.g., 0.1 for 10%)")
    parser.add_argument("-i", "--input", required=True,
                        help="Input dataset directory path")
    parser.add_argument("-o", "--output", required=True,
                        help="Output dataset directory path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")

    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1")

    subsample_dataset(args.fraction, args.input, args.output, args.seed)


if __name__ == "__main__":
    main()