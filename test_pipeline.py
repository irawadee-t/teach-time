"""
Test script to verify the generation and filtering pipeline on a small batch.
"""

import sys
import asyncio
from pathlib import Path
import subprocess

# Test with small numbers
TEST_NUM_SEEDS = 2
TEST_SAMPLES_PER_SEED = 2
TEST_TOTAL = TEST_NUM_SEEDS * TEST_SAMPLES_PER_SEED

print(f"Testing pipeline with {TEST_TOTAL} samples ({TEST_NUM_SEEDS} seeds × {TEST_SAMPLES_PER_SEED} variations)")

# Import and run generation
print("\n" + "="*60)
print("PART 1: GENERATION TEST")
print("="*60 + "\n")

import generate_samples

# Override configuration for testing
generate_samples.NUM_SEEDS = TEST_NUM_SEEDS
generate_samples.SAMPLES_PER_SEED = TEST_SAMPLES_PER_SEED
generate_samples.OUTPUT_PATH = "synthetic/test_samples_generated.jsonl"

# Run generation
print("Starting generation...")
samples = asyncio.run(generate_samples.generate_all_samples(
    csv_path="Text Tutoring Prompts.csv",
    output_path="synthetic/test_samples_generated.jsonl",
    num_seeds=TEST_NUM_SEEDS,
    samples_per_seed=TEST_SAMPLES_PER_SEED
))

if len(samples) < TEST_TOTAL:
    print(f"\n⚠️  Warning: Only generated {len(samples)}/{TEST_TOTAL} samples")
    print("Some samples may have failed. Check for errors above.")
else:
    print(f"\n✓ Generation test successful! Generated {len(samples)} samples")

# Test filtering
print("\n" + "="*60)
print("PART 2: FILTERING TEST")
print("="*60 + "\n")

import filter_samples

# Override configuration for testing
filter_samples.INPUT_PATH = "synthetic/test_samples_generated.jsonl"
filter_samples.OUTPUT_PATH = "synthetic/test_finetune_openai.jsonl"
filter_samples.STATS_PATH = "synthetic/test_filtering_statistics.json"

print("Starting filtering...")
filtered = asyncio.run(filter_samples.filter_samples(
    input_path="synthetic/test_samples_generated.jsonl",
    output_path="synthetic/test_finetune_openai.jsonl",
    stats_path="synthetic/test_filtering_statistics.json"
))

print(f"\n✓ Filtering test successful! Kept {len(filtered)} samples")

# Validate OpenAI format
print("\n" + "="*60)
print("PART 3: FORMAT VALIDATION")
print("="*60 + "\n")

import validate_openai_format

is_valid = validate_openai_format.validate_openai_jsonl("synthetic/test_finetune_openai.jsonl")

if is_valid:
    validate_openai_format.print_sample("synthetic/test_finetune_openai.jsonl", num_samples=1)

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print(f"Generated: {len(samples)} samples")
print(f"Filtered:  {len(filtered)} samples")
print(f"Keep rate: {100*len(filtered)/len(samples):.1f}%" if samples else "N/A")
print(f"Format:    {'✓ Valid' if is_valid else '❌ Invalid'}")
print("\nTest files created in synthetic/ folder:")
print("  - test_samples_generated.jsonl (with metadata)")
print("  - test_finetune_openai.jsonl (OpenAI format)")
print("  - test_filtering_statistics.json")
print("\n✓ Pipeline test complete! You can now run the full generation.")
