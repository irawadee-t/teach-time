"""
Split TutorBench CSV into stratified train/test sets.
Test set: 100 samples, stratified by SUBJECT and bloom_taxonomy
Train set: remaining samples
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
INPUT_CSV = "/Users/zane/Library/Application Support/com.conductor.app/uploads/originals/647217ba-d7cf-49b5-a21e-b3355df3f9b6.csv"
OUTPUT_DIR = "/Users/zane/conductor/workspaces/teach-time/dublin/evals"

def main():
    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} samples")

    # Create stratification key combining subject + bloom level
    # Handle 'none' in bloom_taxonomy
    df['bloom_clean'] = df['bloom_taxonomy'].fillna('none').str.lower()
    df['strat_key'] = df['SUBJECT'] + '_' + df['bloom_clean']

    # Check distribution
    print("\nStratification key distribution:")
    print(df['strat_key'].value_counts())

    # Use SUBJECT only for stratification (more robust with small dataset)
    # This ensures balanced subjects in test set
    test_size = 100

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['SUBJECT'],
        random_state=42
    )

    # Drop helper columns before saving
    cols_to_drop = ['bloom_clean', 'strat_key']
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    # Save to CSV
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n=== Split Complete ===")
    print(f"Train set: {len(train_df)} samples -> {train_path}")
    print(f"Test set:  {len(test_df)} samples -> {test_path}")

    # Verify stratification in test set
    print("\n=== Test Set Distribution ===")
    print("By Subject:")
    print(test_df['SUBJECT'].value_counts())
    print("\nBy Bloom Taxonomy:")
    print(test_df['bloom_taxonomy'].value_counts())

if __name__ == "__main__":
    main()
