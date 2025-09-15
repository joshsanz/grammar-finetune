# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Find converted markdown files from epub-to-clean-md.py and create a dataset, saving as a parquet file.
# Usage: python books-to-dataset.py -i input_directory/ [input_directory2/ ...] -o output_directory/

import argparse
import os
from datasets import Dataset
import spacy
import numpy

nlp = spacy.load("en_core_web_sm")

README = """
    # Books Dataset

    This dataset contains books converted from EPUB format to clean markdown files,
    preserving text formatting and structure.
    The text is broken into chunks between 1 and {npara} paragraphs long, randomly chosen.
    Chunks overlap by 1 paragraph to reduce memorization.

    ## Sources

    The books were originally sourced from [Standard Ebooks](https://standardebooks.org/) and
    converted using the `epub-to-clean-md.py` script. Books are in the public domain.
    The conversion process involves transforming HTML content into markdown format, ensuring
    that the text is clean and well-structured.

    Books contained in this dataset are:

    {books_list}

    """


def split_by_max_size(text, max_words=500):
    doc = nlp(text)
    chunks = []
    start = 0
    current_words = 0
    for sent in doc.sents:
        sent_words = len([t for t in sent if not t.is_punct and not t.is_space])
        if current_words + sent_words > max_words and current_words > 0:
            chunk = text[start:sent.start_char]
            chunks.append(chunk)
            start = sent.start_char
            current_words = sent_words
        else:
            current_words += sent_words
    chunk = text[start:]
    if chunk:
        chunks.append(chunk)
    return chunks


def dataset_generator(input_dirs, npara=5, max_words=500):
    """
    Generates text chunks from Markdown files in the specified input directories.

    Args:
        input_dirs (list of str): List of directory paths to search for Markdown (.md) files.
        npara (int, optional): Maximum number of paragraphs per chunk. Each chunk will contain a random number of paragraphs between 1 and npara. Defaults to 5.
        max_words (int, optional): Maximum number of words per chunk (currently unused in the function). Defaults to 500.

    Yields:
        dict: A dictionary containing:
            - "source": The input directory from which the file was read.
            - "chapter": The filename of the Markdown file.
            - "text": The chunked text consisting of one or more paragraphs.

    Notes:
        - Paragraphs are separated by double newlines ('\n\n').
        - Chunks may overlap by one paragraph to provide context.
        - Only files ending with '.md' are processed.
    """
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
                            # Use a poisson(3) distribution to choose chunk size, clipped at npara
                            chunk_size = 0
                            while chunk_size < 1 or chunk_size > npara:
                                chunk_size = numpy.random.poisson(min(npara, 3))
                            chunk = '\n\n'.join(paragraphs[i:i + chunk_size])

                            # Further split chunk by max_words if needed
                            if max_words > 0:
                                sub_chunks = split_by_max_size(chunk, max_words=max_words)
                                for sub_chunk in sub_chunks:
                                    yield {"source": input_dir, "chapter": file, "text": sub_chunk}
                            else:
                                yield {"source": input_dir, "chapter": file, "text": chunk}
                            i += max(1, chunk_size - 1)  # Overlap by 1 paragraph
                    print(f"Processed {file_path}")
            print(f"Completed directory {input_dir}")


def create_dataset(input_dirs, output_dir, npara=5):
    os.makedirs(output_dir, exist_ok=True)
    readme = README.format(books_list='\n'.join(
        [f"- {os.path.basename(d)}" for d in input_dirs]
    ), npara=npara)
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"README created at {readme_path}")

    ds = Dataset.from_generator(dataset_generator, gen_kwargs={'input_dirs': input_dirs, 'npara': npara})
    # Create a train-test split
    ds = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)  # type: ignore
    train_file_path = os.path.join(output_dir, "train", 'books_dataset.parquet')
    ds["train"].to_parquet(train_file_path)  # type: ignore
    test_file_path = os.path.join(output_dir, "test", 'books_dataset.parquet')
    ds["test"].to_parquet(test_file_path)  # type: ignore
    print(f"Dataset created at {output_dir}/(train|test)/books_dataset.parquet")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset from converted markdown files.")
    parser.add_argument('-i', '--input_dirs', nargs='+', required=True, help='Input directories containing markdown files.')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory to save the dataset.')
    parser.add_argument('-n', '--num-paragraphs', type=int, default=5, help='Maximum number of paragraphs per chunk.')
    parser.add_argument('-m', '--max-words', type=int, default=500, help='Maximum number of words per chunk (-1 to disable).')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dirs = args.input_dirs
    output_dir = args.output_dir
    num_paragraphs = args.num_paragraphs
    create_dataset(input_dirs, output_dir, num_paragraphs)


if __name__ == "__main__":
    main()
