#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Insert synthetic errors into clean book datasets to create training pairs.
# Usage: python insert-synthetic-errors.py -i input_dataset/ -o output_dataset/ [-c config.yaml] [--seed 42] [--preserve-ratio 0.3]

import argparse
import os
import random
import re
from typing import Dict, List, Tuple, Any
import yaml
import spacy
from datasets import load_dataset, Dataset, DatasetDict

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Default configuration for error rates
DEFAULT_CONFIG = {
    'error_rates': {
        'punctuation': {
            'quote_spaces': 0.05,      # Add spaces around quotes
            'missing_quotes': 0.03,     # Remove quotes
            'punctuation_outside_quotes': 0.04,  # Move punctuation outside quotes
        },
        'typography': {
            'dropped_letters': 0.08,    # Remove random letters
            'swapped_letters': 0.06,    # Swap adjacent letters
            'dropped_words': 0.02,      # Remove entire words
            'swapped_words': 0.01,      # Swap adjacent words
        },
        'spacing': {
            'spaces_around_punctuation': 0.06,  # Add/remove spaces around punctuation
            'missing_spaces': 0.04,     # Remove spaces between words
        },
        'homophones': {
            'substitution_rate': 0.10,  # Replace with homophones
        },
        'contractions': {
            'dropped_nt': 0.03,         # Remove "n't" from contractions
        },
        'keyboard_neighbors': {
            'substitution_rate': 0.07,  # Replace with adjacent keyboard keys
        }
    },
    'preserve_ratio': 0.3,  # Ratio of examples to keep unchanged
    'max_errors_per_chunk': 3,  # Maximum errors per text chunk
    'min_chunk_length': 50,  # Minimum length to apply errors
}

# Comprehensive homophone dictionary
HOMOPHONES = {
    'there': ['their', 'they\'re'],
    'their': ['there', 'they\'re'],
    'they\'re': ['there', 'their'],
    'lose': ['loose'],
    'loose': ['lose'],
    'then': ['than'],
    'than': ['then'],
    'affect': ['effect'],
    'effect': ['affect'],
    'here': ['hear'],
    'hear': ['here'],
    'are': ['our'],
    'our': ['are'],
    'by': ['bye', 'buy'],
    'bye': ['by', 'buy'],
    'buy': ['by', 'bye'],
    'weather': ['whether'],
    'whether': ['weather'],
    'to': ['too', 'two'],
    'too': ['to', 'two'],
    'two': ['to', 'too'],
    'you\'re': ['your'],
    'your': ['you\'re'],
    'bear': ['bare'],
    'bare': ['bear'],
    'bearing': ['baring', 'barring'],
    'baring': ['bearing', 'barring'],
    'barring': ['bearing', 'baring'],
    'scaring': ['scarring'],
    'scarring': ['scaring'],
    'sparing': ['sparring'],
    'sparring': ['sparing'],
    'one': ['won'],
    'won': ['one'],
    'break': ['brake'],
    'brake': ['break'],
    'complement': ['compliment'],
    'compliment': ['complement'],
    'allowed': ['aloud'],
    'aloud': ['allowed'],
    'lie': ['lay'],
    'lay': ['lie'],
    'its': ['it\'s'],
    'it\'s': ['its'],
    'principle': ['principal'],
    'principal': ['principle'],
    'lightning': ['lightening'],
    'lightening': ['lightning'],
    'peak': ['peek', 'pique'],
    'peek': ['peak', 'pique'],
    'pique': ['peak', 'peek'],
    'feat': ['feet'],
    'feet': ['feat'],
    'ensure': ['insure'],
    'insure': ['ensure'],
    'were': ['where'],
    'where': ['were'],
}

# Keyboard neighbor dictionary for QWERTY layout
# Maps each letter to its left/right neighbors on the same row (letters only)
KEYBOARD_NEIGHBORS = {
    'q': ['w'], 'w': ['q', 'e'], 'e': ['w', 'r'], 'r': ['e', 't'], 't': ['r', 'y'],
    'y': ['t', 'u'], 'u': ['y', 'i'], 'i': ['u', 'o'], 'o': ['i', 'p'], 'p': ['o'],
    'a': ['s'], 's': ['a', 'd'], 'd': ['s', 'f'], 'f': ['d', 'g'], 'g': ['f', 'h'],
    'h': ['g', 'j'], 'j': ['h', 'k'], 'k': ['j', 'l'], 'l': ['k'], 
    'z': ['x'], 'x': ['z', 'c'], 'c': ['x', 'v'], 'v': ['c', 'b'], 'b': ['v', 'n'],
    'n': ['b', 'm'], 'm': ['n']
}


class SyntheticErrorInserter:
    """Main class for inserting synthetic errors into clean text."""
    
    def __init__(self, config: Dict[str, Any], seed: int = 42):
        self.config = config
        random.seed(seed)
        self.error_rates = config['error_rates']
        self.preserve_ratio = config.get('preserve_ratio', 0.3)
        self.max_errors = config.get('max_errors_per_chunk', 3)
        self.min_length = config.get('min_chunk_length', 50)
    
    def should_preserve(self) -> bool:
        """Determine if this text chunk should be preserved without errors."""
        return random.random() < self.preserve_ratio
    
    def insert_errors(self, text: str) -> str:
        """Insert synthetic errors into text while preserving structure."""
        if len(text) < self.min_length or self.should_preserve():
            return text
            
        # Parse text with spaCy
        doc = nlp(text)
        modified_text = text
        errors_applied = 0
        
        # Track character offsets as we modify text
        offset_shift = 0
        
        # Apply different types of errors
        error_functions = [
            self._insert_punctuation_errors,
            self._insert_typography_errors,
            self._insert_spacing_errors,
            self._insert_homophone_errors,
            self._insert_contraction_errors,
            self._insert_keyboard_neighbor_errors,
        ]
        
        for error_func in error_functions:
            if errors_applied >= self.max_errors:
                break
            modified_text, new_errors = error_func(modified_text, doc)
            errors_applied += new_errors
            
        return modified_text
    
    def _insert_punctuation_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert punctuation-related errors."""
        errors_count = 0
        modified_text = text
        
        # Quote spacing errors
        if random.random() < self.error_rates['punctuation'].get('quote_spaces', 0.0):
            # Add spaces inside quotes: "hello" -> " hello "
            modified_text = re.sub(r'"([^"]*)"', r'" \1 "', modified_text)
            errors_count += 1
            
        # Missing quotes
        if random.random() < self.error_rates['punctuation'].get('missing_quotes', 0.0):
            # Find quoted text and remove quotes
            quote_matches = list(re.finditer(r'"([^"]*)"', modified_text))
            if quote_matches:
                target_match = random.choice(quote_matches)
                quote_removal_type = random.choice(['first', 'last', 'both'])
                
                if quote_removal_type == 'first':
                    # Remove only opening quote
                    modified_text = (modified_text[:target_match.start()] + 
                                   modified_text[target_match.start() + 1:])
                elif quote_removal_type == 'last':
                    # Remove only closing quote
                    modified_text = (modified_text[:target_match.end() - 1] + 
                                   modified_text[target_match.end():])
                else:  # both
                    # Remove both quotes
                    modified_text = (modified_text[:target_match.start()] + 
                                   target_match.group(1) + 
                                   modified_text[target_match.end():])
                errors_count += 1
        
        # Punctuation outside quotes
        if random.random() < self.error_rates['punctuation'].get('punctuation_outside_quotes', 0.0):
            # Move punctuation outside quotes: "Hello," -> "Hello",
            modified_text = re.sub(r'"([^"]*?)([.!?,:;])"', r'"\1"\2', modified_text)
            errors_count += 1
            
        return modified_text, errors_count
    
    def _insert_typography_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert typography errors (dropped/swapped letters and words)."""
        errors_count = 0
        modified_text = text
        
        # Dropped letters
        if random.random() < self.error_rates['typography'].get('dropped_letters', 0.0):
            modified_text_chars = list(modified_text)
            words = [token for token in doc if token.is_alpha and len(token.text) > 1]
            if words:
                target_word = random.choice(words)
                word_start = target_word.idx
                word_text = target_word.text
                if len(word_text) > 1:  # Any character can be dropped
                    char_to_drop = random.randint(0, len(word_text) - 1)  # Any position
                    if word_start + char_to_drop < len(modified_text_chars):
                        modified_text_chars[word_start + char_to_drop] = ''
                        modified_text = ''.join(modified_text_chars)
                        errors_count += 1
        
        # Swapped letters
        if random.random() < self.error_rates['typography'].get('swapped_letters', 0.0):
            modified_text_chars = list(modified_text)
            words = [token for token in doc if token.is_alpha and len(token.text) > 2]
            if words:
                target_word = random.choice(words)
                word_start = target_word.idx
                word_text = target_word.text
                if len(word_text) > 2:
                    swap_pos = random.randint(0, len(word_text) - 2)
                    char1_idx = word_start + swap_pos
                    char2_idx = word_start + swap_pos + 1
                    if char1_idx < len(modified_text_chars) and char2_idx < len(modified_text_chars):
                        modified_text_chars[char1_idx], modified_text_chars[char2_idx] = \
                            modified_text_chars[char2_idx], modified_text_chars[char1_idx]
                        modified_text = ''.join(modified_text_chars)
                        errors_count += 1
        
        
        # Word-level errors
        words = [token.text for token in doc if token.is_alpha]
        
        # Dropped words
        if random.random() < self.error_rates['typography'].get('dropped_words', 0.0) and len(words) >= 3:
            # Select a word token to drop (avoid first and last)
            word_tokens = [token for token in doc if token.is_alpha]
            if len(word_tokens) > 2:
                target_word = random.choice(word_tokens[1:-1])
                start_pos = target_word.idx
                end_pos = start_pos + len(target_word.text)
                
                # Look for surrounding spaces (only ' ' char, preserve \t and \n)
                before_space = start_pos > 0 and modified_text[start_pos - 1] == ' '
                after_space = end_pos < len(modified_text) and modified_text[end_pos] == ' '
                
                if before_space and after_space:
                    # Remove word and one space (preserve paragraph structure)
                    modified_text = modified_text[:start_pos - 1] + modified_text[end_pos:]
                elif before_space or after_space:
                    # Remove word with its adjacent space
                    if before_space:
                        modified_text = modified_text[:start_pos - 1] + modified_text[end_pos:]
                    else:
                        modified_text = modified_text[:start_pos] + modified_text[end_pos + 1:]
                else:
                    # Just remove the word
                    modified_text = modified_text[:start_pos] + modified_text[end_pos:]
                
                errors_count += 1
        
        # Swapped words
        if random.random() < self.error_rates['typography'].get('swapped_words', 0.0) and len(words) > 3:
            # Find two adjacent words to swap
            word_tokens = [token for token in doc if token.is_alpha]
            if len(word_tokens) > 1:
                swap_idx = random.randint(0, len(word_tokens) - 2)
                word1, word2 = word_tokens[swap_idx], word_tokens[swap_idx + 1]
                # Replace with swapped version
                pattern = r'\b' + re.escape(word1.text) + r'\s+' + re.escape(word2.text) + r'\b'
                replacement = f"{word2.text} {word1.text}"
                modified_text = re.sub(pattern, replacement, modified_text, count=1)
                errors_count += 1
        
        return modified_text, errors_count
    
    def _insert_spacing_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert spacing-related errors."""
        errors_count = 0
        modified_text = text
        
        # Spaces around punctuation
        if random.random() < self.error_rates['spacing'].get('spaces_around_punctuation', 0.0):
            # Add spaces before punctuation: "Hello!" -> "Hello !"
            modified_text = re.sub(r'([a-zA-Z])([.!?,:;])', r'\1 \2', modified_text)
            errors_count += 1
        
        # Missing spaces between words
        if random.random() < self.error_rates['spacing'].get('missing_spaces', 0.0):
            # Remove spaces between some words
            words = re.findall(r'\w+\s+\w+', modified_text)
            if words:
                target = random.choice(words)
                modified_text = modified_text.replace(target, target.replace(' ', ''), 1)
                errors_count += 1
                
        return modified_text, errors_count
    
    def _insert_homophone_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert homophone substitution errors."""
        errors_count = 0
        modified_text = text
        
        if random.random() < self.error_rates['homophones'].get('substitution_rate', 0.0):
            # Get all word tokens that have homophones
            homophone_tokens = [token for token in doc if token.is_alpha and token.text.lower() in HOMOPHONES]
            
            if homophone_tokens:
                # Pick a random occurrence
                target_token = random.choice(homophone_tokens)
                target_word = target_token.text
                replacement = random.choice(HOMOPHONES[target_word.lower()])
                
                # Preserve capitalization
                if target_word[0].isupper():
                    replacement = replacement.capitalize()
                
                # Replace the specific occurrence using character positions
                start_pos = target_token.idx
                end_pos = start_pos + len(target_word)
                modified_text = modified_text[:start_pos] + replacement + modified_text[end_pos:]
                errors_count += 1
        
        return modified_text, errors_count
    
    def _insert_contraction_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert contraction-related errors (mainly dropped n't)."""
        errors_count = 0
        modified_text = text
        
        if random.random() < self.error_rates['contractions'].get('dropped_nt', 0.0):
            # Find contractions with "n't"
            contractions = re.findall(r'\w+n\'t\b', modified_text, re.IGNORECASE)
            if contractions:
                target = random.choice(contractions)
                # Remove "n't": "couldn't" -> "could"
                replacement = target.replace("n't", "").replace("N'T", "")
                modified_text = modified_text.replace(target, replacement, 1)
                errors_count += 1
        
        return modified_text, errors_count
    
    def _insert_keyboard_neighbor_errors(self, text: str, doc) -> Tuple[str, int]:
        """Insert keyboard neighbor substitution errors."""
        errors_count = 0
        modified_text = text
        
        if random.random() < self.error_rates['keyboard_neighbors'].get('substitution_rate', 0.0):
            # Get all alphabetic tokens that have keyboard neighbors
            neighbor_tokens = [token for token in doc if token.is_alpha and 
                             any(c.lower() in KEYBOARD_NEIGHBORS for c in token.text)]
            
            if neighbor_tokens:
                # Pick a random token
                target_token = random.choice(neighbor_tokens)
                target_word = target_token.text
                
                # Find characters in the word that have neighbors
                neighbor_chars = [(i, c) for i, c in enumerate(target_word) 
                                if c.lower() in KEYBOARD_NEIGHBORS]
                
                if neighbor_chars:
                    # Pick a random character to replace
                    char_idx, char = random.choice(neighbor_chars)
                    replacement = random.choice(KEYBOARD_NEIGHBORS[char.lower()])
                    
                    # Preserve capitalization
                    if char.isupper():
                        replacement = replacement.upper()
                    
                    # Replace the character using token position
                    start_pos = target_token.idx + char_idx
                    end_pos = start_pos + 1
                    modified_text = modified_text[:start_pos] + replacement + modified_text[end_pos:]
                    errors_count += 1
        
        return modified_text, errors_count


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Merge with defaults
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        return config
    return DEFAULT_CONFIG


def process_dataset(input_path: str, output_path: str, config: Dict[str, Any], seed: int):
    """Process the clean dataset and insert synthetic errors."""
    print(f"Loading dataset from {input_path}")
    
    # Load the dataset (assumes it's in HuggingFace format)
    dataset = load_dataset(input_path)
    
    inserter = SyntheticErrorInserter(config, seed)
    
    def process_examples(examples):
        """Process a batch of examples."""
        originals = []
        corrected = []
        sources = []
        
        for i in range(len(examples['text'])):
            clean_text = examples['text'][i]
            error_text = inserter.insert_errors(clean_text)
            
            originals.append(error_text)
            corrected.append(clean_text)
            sources.append(examples['source'][i])
        
        return {
            'original': originals,
            'corrected': corrected,
            'source': sources
        }
    
    print("Processing dataset with synthetic errors...")
    processed_dataset = dataset.map(process_examples, batched=True, batch_size=100)
    
    # Remove the old 'text' column and keep only the new format
    processed_dataset = processed_dataset.remove_columns(['text', 'chapter'])
    
    # Save the processed dataset
    os.makedirs(output_path, exist_ok=True)
    
    for split in processed_dataset.keys():
        split_path = os.path.join(output_path, split)
        os.makedirs(split_path, exist_ok=True)
        output_file = os.path.join(split_path, 'books_with_errors.parquet')
        processed_dataset[split].to_parquet(output_file)
        print(f"Saved {split} split to {output_file}")
    
    # Create README
    readme_content = f"""# Books Dataset with Synthetic Errors

This dataset contains books with synthetically inserted errors for grammar correction training.

## Error Types
- Punctuation errors (quotes, placement)
- Typography errors (dropped/swapped letters and words)  
- Spacing errors (around punctuation, between words)
- Homophone substitutions
- Contraction errors (dropped "n't")

## Configuration
- Preserve ratio: {config.get('preserve_ratio', 0.3)} (fraction kept without errors)
- Max errors per chunk: {config.get('max_errors_per_chunk', 3)}
- Seed: {seed}

## Format
- `original`: Text with synthetic errors
- `corrected`: Clean original text
- `source`: Source directory/book
"""
    
    with open(os.path.join(output_path, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"Dataset processing complete. Output saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Insert synthetic errors into clean book datasets."
    )
    parser.add_argument(
        '-i', '--input', required=True,
        help='Input dataset directory (from books-to-dataset.py)'
    )
    parser.add_argument(
        '-o', '--output', required=True,
        help='Output directory for dataset with errors'
    )
    parser.add_argument(
        '-c', '--config', 
        help='Configuration YAML file (optional)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--preserve-ratio', type=float,
        help='Ratio of examples to preserve without errors (overrides config)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override preserve ratio if provided
    if args.preserve_ratio is not None:
        config['preserve_ratio'] = args.preserve_ratio
    
    # Process the dataset
    process_dataset(args.input, args.output, config, args.seed)


if __name__ == "__main__":
    main()