# Data Processing Scripts

This directory contains scripts for creating and processing datasets for Grammar Error Correction (GEC) training.

## Pipeline Overview

The complete dataset creation pipeline follows this sequence:

1. **EPUB to Markdown**: `epub-to-clean-md.py` - Convert EPUB books to clean markdown
2. **Books to Dataset**: `books-to-dataset.py` - Convert markdown files to HuggingFace Dataset format  
3. **Synthetic Error Insertion**: `insert_synthetic_errors.py` - Add realistic errors to clean text
4. **Dataset Merging**: `merge-all-datasets.py` - Combine clean and error datasets
5. **Dataset Subsampling**: `subsample-dataset.py` - Create smaller datasets for testing
6. **Error Analysis**: `view_synthetic_diffs.py` - Visualize inserted errors

## Scripts

### epub-to-clean-md.py
Converts EPUB files to clean markdown format, preserving structure and formatting.

**Usage:**
```bash
python epub-to-clean-md.py input.epub output_directory/
```

### books-to-dataset.py
Creates HuggingFace datasets from converted markdown files, chunking text into paragraphs with overlap.

**Usage:**
```bash
python books-to-dataset.py -i input_dir1/ [input_dir2/ ...] -o output_dir/ [-n num_paragraphs]
```

**Output format:**
- `source`: Source directory/book name
- `chapter`: Markdown filename  
- `text`: Text chunks (1-5 paragraphs with 1 paragraph overlap)

### insert-synthetic-errors.py
Inserts realistic synthetic errors into clean text datasets to create training pairs.

**Usage:**
```bash
python insert-synthetic-errors.py -i input_dataset/ -o output_dataset/ [-c config.yaml] [--seed 42] [--preserve-ratio 0.3]
```

**Input:** Dataset from `books-to-dataset.py` with fields: `source`, `chapter`, `text`
**Output:** Dataset with fields: `source`, `original` (with errors), `corrected` (clean text)

**Error Types:**
- **Punctuation errors**: Spaces around quotes, missing quotes, punctuation placement
- **Typography errors**: Dropped/swapped letters, dropped/swapped words
- **Spacing errors**: Extra spaces around punctuation, missing spaces between words
- **Homophone substitutions**: Common confusions (there/their/they're, etc.)
- **Contraction errors**: Dropped "n't" from contractions

**Configuration:**
Error rates and behavior can be customized via YAML config file. See `../config/error_insertion_config.yaml` for example.

**Features:**
- **Linguistic awareness**: Uses spaCy for proper text parsing
- **Formatting preservation**: Maintains markdown, capitalization, and paragraph structure
- **Configurable preserve ratio**: Keep some examples unchanged
- **Deterministic**: Reproducible results with seed parameter
- **Quality controls**: Minimum text length, maximum errors per chunk

### merge-all-datasets.py
Merges clean datasets with external grammar correction datasets from HuggingFace Hub.

**Usage:**
```bash
python merge-all-datasets.py -c clean_dataset_path -g config/datasets.txt -o output_dir/
```

**Supported external datasets:**
- `grammarly/coedit` - Grammar error correction tasks
- `jhu-clsp/jfleg` - JFLEG grammar correction dataset  
- `agentlans/grammar-correction` - Grammar correction pairs

### subsample-dataset.py
Creates smaller versions of datasets for quick testing and experimentation.

**Usage:**
```bash
python subsample-dataset.py -i input_dataset/ -o output_dataset/ -f 0.05
```

**Parameters:**
- `-i, --input`: Input dataset directory
- `-o, --output`: Output dataset directory
- `-f, --fraction`: Fraction of data to sample (e.g., 0.05 = 5%)

### view_synthetic_diffs.py
Visualizes synthetic errors by showing side-by-side comparison of original and corrected text with highlighted differences.

**Usage:**
```bash
python view_synthetic_diffs.py -i synthetic_dataset/ [-n num_examples] [--html output.html]
```

**Features:**
- **Terminal output**: Colored diff display in terminal
- **HTML export**: Generate HTML report with highlighted differences
- **Statistics**: Show error type distribution and frequency
- **Sample size control**: Limit number of examples to review

## Testing

### test_synthetic_errors.py
Comprehensive test suite for synthetic error insertion functionality.

**Run tests:**
```bash
# Run all tests with verbose output
python -m pytest test_synthetic_errors.py -v

# Run specific test class
python -m pytest test_synthetic_errors.py::TestSyntheticErrorInsertion -v

# Run with coverage
python -m pytest test_synthetic_errors.py --cov=insert_synthetic_errors
```

**Test coverage:**
- Individual error type validation with expected outputs
- Integration tests for multiple error combinations  
- Configuration parameter validation
- Edge cases and error handling

## Configuration Files

### ../config/error_insertion_config.yaml
Example configuration for synthetic error insertion with documented parameters:

```yaml
error_rates:
  punctuation:
    quote_spaces: 0.05
    missing_quotes: 0.03  
    punctuation_outside_quotes: 0.04
  typography:
    dropped_letters: 0.08
    swapped_letters: 0.06
    # ... etc
preserve_ratio: 0.3
max_errors_per_chunk: 3
min_chunk_length: 50
```

### ../config/datasets.txt
List of external grammar correction datasets to include in merging:
```
grammarly/coedit
jhu-clsp/jfleg  
agentlans/grammar-correction
```

## Data Flow

```
EPUB Files
    ↓ (epub-to-clean-md.py)
Markdown Files  
    ↓ (books-to-dataset.py)
Clean Dataset [source, chapter, text]
    ↓ (insert-synthetic-errors.py)
Error Dataset [source, original, corrected]
    ↓ (merge-all-datasets.py) + External Datasets
Final Training Dataset [source, original, corrected]
    ↓ (subsample-dataset.py) [optional]
Small Test Dataset [source, original, corrected]

Analysis Tools:
- view_synthetic_diffs.py: Visualize inserted errors
- test_synthetic_errors.py: Validate error insertion quality
```

## Output Formats

**After books-to-dataset.py:**
```json
{
  "source": "book_directory_name",
  "chapter": "chapter-01.md", 
  "text": "Clean text content..."
}
```

**After insert-synthetic-errors.py:**
```json
{
  "source": "book_directory_name",
  "original": "Text with syntetic erors...",
  "corrected": "Text with synthetic errors..."
}
```

**Final merged dataset:**
```json
{
  "source": "book_name | dataset_name",
  "original": "Text with errors or clean text",
  "corrected": "Corrected version of text"
}
```

## Dependencies

- `datasets` - HuggingFace datasets library
- `spacy` - Natural language processing (requires `en_core_web_sm` model)
- `yaml` - Configuration file parsing
- `pytest` - Testing framework (development)

**Install spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

## Best Practices

1. **Test error insertion** with sample data before processing large datasets
2. **Use version control** to track configuration changes
3. **Set seeds** for reproducible results in experiments
4. **Monitor error quality** by manually reviewing sample outputs
5. **Tune error rates** based on target use case (formal writing vs. casual text)
6. **Run tests** before making configuration changes