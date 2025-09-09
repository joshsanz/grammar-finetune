#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence Length Analysis Tool for Training Optimization

This script analyzes your dataset to help determine optimal max_length and packing settings:
- Samples and tokenizes data exactly as SFTTrainer would
- Visualizes sequence length distribution
- Estimates truncation impact for different max_length values
- Provides recommendations for optimal settings

Usage: python analyze_sequence_lengths.py [--config config/gemma3_270m_config.yaml] [--samples 1000]
"""

import argparse
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
    return config

def setup_tokenizer(config):
    """Setup tokenizer exactly as in training script"""
    print("Setting up tokenizer...")

    # Load model and tokenizer (we only need tokenizer)
    model, tokenizer = FastModel.from_pretrained(
        model_name=config["model"]["model_name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        load_in_8bit=config["model"]["load_in_8bit"],
        full_finetuning=config["model"]["full_finetuning"],
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=config["data"]["chat_template"],
    )

    return tokenizer

def prepare_dataset(config, tokenizer, num_samples=1000):
    """Prepare dataset exactly as in training script"""
    print(f"Loading and preparing {num_samples} samples from dataset...")

    # Load dataset
    dataset = load_dataset(config["data"]["dataset_path"])

    # Load system prompt
    with open(config["data"]["system_prompt_path"], "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    def convert_to_chatml(example):
        """Convert to ChatML format exactly as in training"""
        marked_original = f"[BEGINNING OF CONTENT]\n{example['original']}\n[END OF CONTENT]"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": marked_original},
                {"role": "assistant", "content": example["corrected"]}
            ]
        }

    def formatting_prompts_func(examples):
        """Apply chat template exactly as in training"""
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>')
                for convo in convos]
        return {"text": texts}

    # Process dataset
    dataset = dataset.map(convert_to_chatml).remove_columns(["source", "original", "corrected"])
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Sample random examples
    train_dataset = dataset["train"]
    total_samples = len(train_dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))

    sampled_texts = [train_dataset[i]["text"] for i in sample_indices]

    print(f"Sampled {len(sampled_texts)} examples from {total_samples} total examples")
    return sampled_texts

def tokenize_and_analyze(texts, tokenizer):
    """Tokenize texts and return sequence lengths"""
    print("Tokenizing samples...")

    sequence_lengths = []
    total_tokens = 0

    for text in texts:
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        length = len(tokens["input_ids"][0])
        sequence_lengths.append(length)
        total_tokens += length

    return sequence_lengths, total_tokens

def calculate_truncation_stats(sequence_lengths, max_lengths):
    """Calculate truncation statistics for different max_length values"""
    stats = {}
    total_tokens = sum(sequence_lengths)

    for max_len in max_lengths:
        truncated_tokens = sum(max(0, length - max_len) for length in sequence_lengths)
        truncated_samples = sum(1 for length in sequence_lengths if length > max_len)

        stats[max_len] = {
            'truncated_tokens': truncated_tokens,
            'truncated_samples': truncated_samples,
            'truncation_percentage': (truncated_tokens / total_tokens) * 100,
            'samples_affected_percentage': (truncated_samples / len(sequence_lengths)) * 100
        }

    return stats

def create_visualizations(sequence_lengths, truncation_stats):
    """Create and save visualization plots"""
    print("Creating visualizations...")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sequence Length Analysis for Training Optimization', fontsize=16, fontweight='bold')

    # 1. Histogram of sequence lengths
    ax1.hist(sequence_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(sequence_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(sequence_lengths):.0f}')
    ax1.axvline(np.median(sequence_lengths), color='green', linestyle='--', label=f'Median: {np.median(sequence_lengths):.0f}')
    ax1.axvline(np.percentile(sequence_lengths, 95), color='orange', linestyle='--', label=f'95th percentile: {np.percentile(sequence_lengths, 95):.0f}')
    ax1.set_xlabel('Sequence Length (tokens)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sequence Lengths')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2.boxplot(sequence_lengths, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax2.set_ylabel('Sequence Length (tokens)')
    ax2.set_title('Sequence Length Box Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Truncation percentage vs max_length
    max_lengths = sorted(truncation_stats.keys())
    truncation_percentages = [truncation_stats[ml]['truncation_percentage'] for ml in max_lengths]

    ax3.plot(max_lengths, truncation_percentages, 'bo-', linewidth=2, markersize=6)
    ax3.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='1% threshold')
    ax3.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5% threshold')
    ax3.set_xlabel('Max Length')
    ax3.set_ylabel('Truncation Percentage (%)')
    ax3.set_title('Token Truncation vs Max Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Samples affected percentage vs max_length
    samples_affected = [truncation_stats[ml]['samples_affected_percentage'] for ml in max_lengths]

    ax4.plot(max_lengths, samples_affected, 'ro-', linewidth=2, markersize=6)
    ax4.set_xlabel('Max Length')
    ax4.set_ylabel('Samples Affected (%)')
    ax4.set_title('Percentage of Samples Truncated vs Max Length')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'sequence_length_analysis.png'")

    return fig

def find_optimal_max_length(truncation_stats, threshold=5.0):
    """Find the optimal max_length based on truncation threshold"""
    for max_len in sorted(truncation_stats.keys()):
        if truncation_stats[max_len]['truncation_percentage'] <= threshold:
            return max_len
    return max(truncation_stats.keys())

def calculate_packing_efficiency(sequence_lengths, max_length):
    """Calculate potential efficiency gains from packing"""
    total_samples = len(sequence_lengths)
    total_tokens = sum(min(length, max_length) for length in sequence_lengths)
    total_padded_tokens = total_samples * max_length

    efficiency_without_packing = total_tokens / total_padded_tokens

    # Simulate simple packing
    packed_sequences = 0
    current_length = 0

    for length in sorted(sequence_lengths):
        if current_length + min(length, max_length) <= max_length:
            current_length += min(length, max_length)
        else:
            packed_sequences += 1
            current_length = min(length, max_length)

    if current_length > 0:
        packed_sequences += 1

    efficiency_with_packing = total_tokens / (packed_sequences * max_length)

    return {
        'without_packing': efficiency_without_packing,
        'with_packing': efficiency_with_packing,
        'improvement': efficiency_with_packing / efficiency_without_packing,
        'sequences_without_packing': total_samples,
        'sequences_with_packing': packed_sequences
    }

def print_analysis_report(sequence_lengths, truncation_stats, config):
    """Print comprehensive analysis report"""
    print("\n" + "="*80)
    print("SEQUENCE LENGTH ANALYSIS REPORT")
    print("="*80)

    # Basic statistics
    print("\nBASIC STATISTICS:")
    print(f"  Total samples analyzed: {len(sequence_lengths):,}")
    print(f"  Mean sequence length: {np.mean(sequence_lengths):.1f} tokens")
    print(f"  Median sequence length: {np.median(sequence_lengths):.1f} tokens")
    print(f"  Std deviation: {np.std(sequence_lengths):.1f} tokens")
    print(f"  Min length: {min(sequence_lengths)} tokens")
    print(f"  Max length: {max(sequence_lengths)} tokens")
    print(f"  95th percentile: {np.percentile(sequence_lengths, 95):.1f} tokens")
    print(f"  99th percentile: {np.percentile(sequence_lengths, 99):.1f} tokens")

    # Current config analysis
    current_max_length = config["training"]["max_length"]
    if current_max_length in truncation_stats:
        current_stats = truncation_stats[current_max_length]
        print(f"\nCURRENT CONFIG ANALYSIS (max_length = {current_max_length}):")
        print(f"  Tokens truncated: {current_stats['truncated_tokens']:,} ({current_stats['truncation_percentage']:.2f}%)")
        print(f"  Samples affected: {current_stats['truncated_samples']:,} ({current_stats['samples_affected_percentage']:.1f}%)")

    # Find optimal max_length for different thresholds
    optimal_max_length_1pct = find_optimal_max_length(truncation_stats, threshold=1.0)
    optimal_max_length_5pct = find_optimal_max_length(truncation_stats, threshold=5.0)

    optimal_stats_1pct = truncation_stats[optimal_max_length_1pct]
    optimal_stats_5pct = truncation_stats[optimal_max_length_5pct]

    print(f"\nRECOMMENDED MAX_LENGTH (≤1% token truncation): {optimal_max_length_1pct}")
    print(f"  Tokens truncated: {optimal_stats_1pct['truncated_tokens']:,} ({optimal_stats_1pct['truncation_percentage']:.2f}%)")
    print(f"  Samples affected: {optimal_stats_1pct['truncated_samples']:,} ({optimal_stats_1pct['samples_affected_percentage']:.1f}%)")

    print(f"\nRECOMMENDED MAX_LENGTH (≤5% token truncation): {optimal_max_length_5pct}")
    print(f"  Tokens truncated: {optimal_stats_5pct['truncated_tokens']:,} ({optimal_stats_5pct['truncation_percentage']:.2f}%)")
    print(f"  Samples affected: {optimal_stats_5pct['truncated_samples']:,} ({optimal_stats_5pct['samples_affected_percentage']:.1f}%)")

    # Packing analysis - current config
    current_packing_stats = calculate_packing_efficiency(sequence_lengths, current_max_length)
    print(f"\nPACKING ANALYSIS (current max_length = {current_max_length}):")
    print(f"  Efficiency without packing: {current_packing_stats['without_packing']:.1%}")
    print(f"  Efficiency with packing: {current_packing_stats['with_packing']:.1%}")
    print(f"  Improvement factor: {current_packing_stats['improvement']:.2f}x")
    print(f"  Sequences without packing: {current_packing_stats['sequences_without_packing']:,}")
    print(f"  Sequences with packing: {current_packing_stats['sequences_with_packing']:,}")
    print(f"  Memory savings: {(1 - current_packing_stats['sequences_with_packing']/current_packing_stats['sequences_without_packing']):.1%}")

    # Packing analysis - 1% truncation threshold
    if optimal_max_length_1pct != current_max_length:
        packing_stats_1pct = calculate_packing_efficiency(sequence_lengths, optimal_max_length_1pct)
        print(f"\nPACKING ANALYSIS (1% truncation max_length = {optimal_max_length_1pct}):")
        print(f"  Efficiency without packing: {packing_stats_1pct['without_packing']:.1%}")
        print(f"  Efficiency with packing: {packing_stats_1pct['with_packing']:.1%}")
        print(f"  Improvement factor: {packing_stats_1pct['improvement']:.2f}x")
        print(f"  Sequences without packing: {packing_stats_1pct['sequences_without_packing']:,}")
        print(f"  Sequences with packing: {packing_stats_1pct['sequences_with_packing']:,}")
        print(f"  Memory savings: {(1 - packing_stats_1pct['sequences_with_packing']/packing_stats_1pct['sequences_without_packing']):.1%}")

    # Packing analysis - 5% truncation threshold (if different from 1%)
    if optimal_max_length_5pct != current_max_length and optimal_max_length_5pct != optimal_max_length_1pct:
        packing_stats_5pct = calculate_packing_efficiency(sequence_lengths, optimal_max_length_5pct)
        print(f"\nPACKING ANALYSIS (5% truncation max_length = {optimal_max_length_5pct}):")
        print(f"  Efficiency without packing: {packing_stats_5pct['without_packing']:.1%}")
        print(f"  Efficiency with packing: {packing_stats_5pct['with_packing']:.1%}")
        print(f"  Improvement factor: {packing_stats_5pct['improvement']:.2f}x")
        print(f"  Sequences without packing: {packing_stats_5pct['sequences_without_packing']:,}")
        print(f"  Sequences with packing: {packing_stats_5pct['sequences_with_packing']:,}")
        print(f"  Memory savings: {(1 - packing_stats_5pct['sequences_with_packing']/packing_stats_5pct['sequences_without_packing']):.1%}")

    # Recommendations
    print("\nRECOMMENDATIONS:")

    # Max length recommendations
    if optimal_max_length_1pct == optimal_max_length_5pct:
        if optimal_max_length_5pct != current_max_length:
            print(f"  🔧 Consider changing max_length from {current_max_length} to {optimal_max_length_5pct}")
            print(f"     (This achieves ≤1% truncation)")
        else:
            print(f"  ✅ Current max_length ({current_max_length}) is already optimal")
    else:
        print(f"  🔧 Max length options:")
        print(f"     Conservative (≤1% truncation): {optimal_max_length_1pct}")
        print(f"     Balanced (≤5% truncation): {optimal_max_length_5pct}")
        print(f"     Current: {current_max_length}")
        if current_max_length not in [optimal_max_length_1pct, optimal_max_length_5pct]:
            print(f"     Recommendation: Consider {optimal_max_length_5pct} for balanced efficiency")

    # Use current config for packing recommendation
    if current_packing_stats['improvement'] > 1.2:
        print(f"  📦 Packing could improve efficiency by {(current_packing_stats['improvement']-1)*100:.1f}%")
        print(f"     Memory savings: {(1 - current_packing_stats['sequences_with_packing']/current_packing_stats['sequences_without_packing']):.1%}")
        print("     Consider enabling packing in SFTTrainer with:")
        print("     args = SFTConfig(..., packing=True, ...)")
    else:
        print("  ⚪ Packing would provide minimal benefit (<20% improvement)")

    # Additional recommendations for alternative max_lengths
    if optimal_max_length_5pct != current_max_length and 'packing_stats_5pct' in locals():
        print(f"  💡 With 5% truncation max_length ({optimal_max_length_5pct}):")
        print(f"     Packing improvement: {(packing_stats_5pct['improvement']-1)*100:.1f}%")
        print(f"     Memory savings: {(1 - packing_stats_5pct['sequences_with_packing']/packing_stats_5pct['sequences_without_packing']):.1%}")

    if optimal_max_length_1pct != current_max_length and optimal_max_length_1pct != optimal_max_length_5pct and 'packing_stats_1pct' in locals():
        print(f"  💡 With 1% truncation max_length ({optimal_max_length_1pct}):")
        print(f"     Packing improvement: {(packing_stats_1pct['improvement']-1)*100:.1f}%")
        print(f"     Memory savings: {(1 - packing_stats_1pct['sequences_with_packing']/packing_stats_1pct['sequences_without_packing']):.1%}")

    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze sequence lengths for training optimization")
    parser.add_argument("--config", default="config/gemma3_270m_config.yaml", help="Path to config file")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to analyze")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    print("Starting sequence length analysis...")
    print(f"Config: {args.config}")
    print(f"Samples: {args.samples}")

    # Load configuration
    config = load_config(args.config)

    # Setup tokenizer
    tokenizer = setup_tokenizer(config)

    # Prepare dataset
    texts = prepare_dataset(config, tokenizer, args.samples)

    # Tokenize and analyze
    sequence_lengths, total_tokens = tokenize_and_analyze(texts, tokenizer)

    # Define max_length values to test
    max_seq_length = max(sequence_lengths)
    test_max_lengths = list(range(512, min(max_seq_length + 512, 8192), 256))
    if config["training"]["max_length"] not in test_max_lengths:
        test_max_lengths.append(config["training"]["max_length"])
    test_max_lengths = sorted(set(test_max_lengths))

    # Calculate truncation statistics
    truncation_stats = calculate_truncation_stats(sequence_lengths, test_max_lengths)

    # Generate visualizations
    if not args.no_plot:
        try:
            create_visualizations(sequence_lengths, truncation_stats)
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
            print("Install matplotlib and seaborn to enable plotting")

    # Print analysis report
    print_analysis_report(sequence_lengths, truncation_stats, config)

if __name__ == "__main__":
    main()