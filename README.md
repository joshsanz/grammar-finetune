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

## Fine Tuning

TBD...
