# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Find converted markdown files from epub-to-clean-md.py and create a dataset, saving as a parquet file.
# Usage: python books-to-dataset.py -i input_directory/ [input_directory2/ ...] -o output_directory/

import argparse
import os
import random
from datasets import Dataset


README = """
    # Books Dataset

    This dataset contains books converted from EPUB format to clean markdown files,
    preserving text formatting and structure.
    The text is broken into chunks between 1 and 5 paragraphs long, randomly chosen.
    Chunks overlap by 1 paragraph to reduce memorization.

    ## Sources

    The books were originally sourced from [Standard Ebooks](https://standardebooks.org/) and
    converted using the `epub-to-clean-md.py` script. Books are in the public domain.
    The conversion process involves transforming HTML content into markdown format, ensuring
    that the text is clean and well-structured.

    Books contained in this dataset are:

    {books_list}

    """


def dataset_generator(input_dirs):
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                        i = 0
                        while i < len(paragraphs):
                            chunk_size = random.randint(1, 5)
                            chunk = '\n\n'.join(paragraphs[i:i + chunk_size])
                            yield {"source": input_dir, "chapter": file, "text": chunk}
                            i += max(1, chunk_size - 1)  # Overlap by 1 paragraph
                    print(f"Processed {file_path}")
            print(f"Completed directory {input_dir}")


def create_dataset(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    readme = README.format(books_list='\n'.join(
        [f"- {os.path.basename(d)}" for d in input_dirs]
    ))
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"README created at {readme_path}")

    ds = Dataset.from_generator(dataset_generator, gen_kwargs={'input_dirs': input_dirs})
    # Create a train-test split
    ds = ds.train_test_split(test_size=0.1, seed=42)  # type: ignore
    train_file_path = os.path.join(output_dir, "train", 'books_dataset.parquet')
    ds["train"].to_parquet(train_file_path) # type: ignore
    test_file_path = os.path.join(output_dir, "test", 'books_dataset.parquet')
    ds["test"].to_parquet(test_file_path) # type: ignore
    print(f"Dataset created at {output_dir}/(train|test)/books_dataset.parquet")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset from converted markdown files.")
    parser.add_argument('-i', '--input_dirs', nargs='+', required=True, help='Input directories containing markdown files.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save the dataset.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dirs = args.input_dirs
    output_dir = args.output_dir

    create_dataset(input_dirs, output_dir)


if __name__ == "__main__":
    main()
