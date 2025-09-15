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
for epub in epubs/*.epub; do
    [ -e "$epub" ] || continue
    base="${epub%.epub}"
    out_dir="${base}"
    echo "Processing $epub to $out_dir"
    python epub-to-clean-md.py "$epub"  "$out_dir" || exit 1
done

# Convert book .mds to a single dataset
echo "Creating clean example dataset from books"
python books-to-dataset.py -o books/ -i $(ls epubs/*.epub | sed 's/\.epub//g' | tr '\n' ' ') -n 5 || exit 1

# Add synthetic errors to the clean dataset
echo "Creating synthetic error dataset from clean examples"
python insert_synthetic_errors.py \
    -i books/ -o synthetic/ \
    -c ../config/error_insertion_config.yaml || exit 1

# Combine with existing GEC datasets
echo "Merging all datasets into gec-dataset"
python merge-all-datasets.py -o gec-dataset -c synthetic/ -g ../config/datasets.txt || exit 1

# Create a smaller dataset for quick testing
echo "Creating small dataset for quick testing"
python subsample-dataset.py -i gec-dataset -o gec-dataset-small -f 0.05 || exit 1

# If mlx-examples exists, create a dataset formatted for mlx (.jsonl)
cd ..
if [ -d "mlx-examples" ]; then
    echo "Creating MLX formatted dataset"
    python create-dataset-for-mlx.py -i data/gec-dataset -o data.mlx -p config/prompts.txt || exit 1
fi