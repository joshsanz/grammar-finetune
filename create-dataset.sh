#!/bin/bash
# create-dataset.sh
# Script to create a dataset for fine-tuning llms for GEC tasks.

# Test if `python` command exists, otherwise activate the venv
if ! command -v python &> /dev/null
then
    echo "python could not be found, activating venv"
    source .venv/bin/activate
fi

cd data
mkdir -p gec-dataset

# Prepare markdown version of ebooks present
for epub in *.epub; do
    [ -e "$epub" ] || continue
    base="${epub%.epub}"
    out_dir="${base}"
    echo "Processing $epub to $out_dir"
    python epub-to-clean-md.py "$epub"  "$out_dir" || exit 1
done

# Convert book .mds to a single dataset
echo "Creating clean example dataset from books"
python books-to-dataset.py -o books/ -i $(ls *.epub | sed 's/\.epub//g' | tr '\n' ' ') -n 5 || exit 1

# Combine with existing GEC datasets
echo "Merging all datasets into gec-dataset"
python merge-all-datasets.py -o gec-dataset -c books -g ../config/datasets.txt || exit 1

# If mlx-examples exists, create a dataset formatted for mlx (.jsonl)
cd ..
if [ -d "mlx-examples" ]; then
    echo "Creating MLX formatted dataset"
    python create-dataset-for-mlx.py -i data/gec-dataset -o data.mlx -p config/prompts.txt || exit 1
fi