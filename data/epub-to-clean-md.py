# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convert EPUB files to clean text files with html converted to markdown.
# One file per chapter.
# Usage: python epub-to-clean-md.py input.epub output_directory/

from html_to_markdown import convert_to_markdown
import ebooklib
from ebooklib import epub
import os
import sys


def epub_to_clean_md(epub_path, output_dir):
    # Load the EPUB file
    book = epub.read_epub(epub_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the items in the EPUB
    print(f"Processing EPUB: {epub_path}")
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Convert HTML content to markdown
            markdown_content = convert_to_markdown(item.get_content().decode('utf-8'))

            # Create a filename based on the item's title or id
            title = item.get_name().replace('/', '_').replace('\\', '_')
            output_file_path = os.path.join(output_dir, f"{title}.md")

            # Write the markdown content to a text file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            print(f"Converted {item.get_name()} to {output_file_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python epub-to-clean-md.py input.epub output_directory/")
        sys.exit(1)

    epub_path = sys.argv[1]
    output_dir = sys.argv[2]

    epub_to_clean_md(epub_path, output_dir)


if __name__ == "__main__":
    main()
