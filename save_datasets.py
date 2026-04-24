"""
Save Train, Test, and Validation Datasets as CSV Files
This script creates separate CSV files for train, test, and validation splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Set random seed for reproducibility
RANDOM_STATE = 42

print("="*70)
print("SAVING TRAIN, TEST, AND VALIDATION DATASETS")
print("="*70)

# Load datasets
print("\n1. Loading True and Fake news datasets...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

print(f"   ✓ Loaded True.csv: {len(true_df)} articles")
print(f"   ✓ Loaded Fake.csv: {len(fake_df)} articles")

# Label the data
true_df['label'] = 1  # Real news
fake_df['label'] = 0  # Fake news

# Combine datasets
print("\n2. Combining datasets...")
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"   ✓ Combined dataset size: {len(df)} articles")
print(f"   ✓ Label distribution: {df['label'].value_counts().to_dict()}")

# Create train/test/validation splits
print("\n3. Creating train/test/validation splits...")
print("   Split strategy: 80% Train, 10% Test, 10% Validation")

# First split: 80% (train+test) and 10% (validation)
X_temp, X_val, y_temp, y_val = train_test_split(
    df, df['label'],
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=df['label']
)

# Second split: Split temp into 80% train and 10% test (of original)
# Since temp is 90% of data, 10/90 ≈ 0.1111 to get 10% of original
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.1111,  # 0.1111 * 0.9 ≈ 0.1 (10% of original)
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"\n   Training set: {len(X_train)} articles ({len(X_train)/len(df)*100:.1f}%)")
print(f"      - Real: {(X_train['label'] == 1).sum()}")
print(f"      - Fake: {(X_train['label'] == 0).sum()}")

print(f"\n   Test set: {len(X_test)} articles ({len(X_test)/len(df)*100:.1f}%)")
print(f"      - Real: {(X_test['label'] == 1).sum()}")
print(f"      - Fake: {(X_test['label'] == 0).sum()}")

print(f"\n   Validation set: {len(X_val)} articles ({len(X_val)/len(df)*100:.1f}%)")
print(f"      - Real: {(X_val['label'] == 1).sum()}")
print(f"      - Fake: {(X_val['label'] == 0).sum()}")

# Create output directory if it doesn't exist
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

# Save datasets
print(f"\n4. Saving datasets to '{output_dir}/' directory...")

train_path = os.path.join(output_dir, "train_dataset.csv")
X_train.to_csv(train_path, index=False)
print(f"   ✓ Saved: {train_path}")

test_path = os.path.join(output_dir, "test_dataset.csv")
X_test.to_csv(test_path, index=False)
print(f"   ✓ Saved: {test_path}")

val_path = os.path.join(output_dir, "validation_dataset.csv")
X_val.to_csv(val_path, index=False)
print(f"   ✓ Saved: {val_path}")

# Save a combined metadata file with statistics
metadata = {
    "Total Articles": len(df),
    "Train Size": len(X_train),
    "Test Size": len(X_test),
    "Validation Size": len(X_val),
    "Train Real Count": int((X_train['label'] == 1).sum()),
    "Train Fake Count": int((X_train['label'] == 0).sum()),
    "Test Real Count": int((X_test['label'] == 1).sum()),
    "Test Fake Count": int((X_test['label'] == 0).sum()),
    "Val Real Count": int((X_val['label'] == 1).sum()),
    "Val Fake Count": int((X_val['label'] == 0).sum()),
    "Train Percentage": f"{len(X_train)/len(df)*100:.1f}%",
    "Test Percentage": f"{len(X_test)/len(df)*100:.1f}%",
    "Validation Percentage": f"{len(X_val)/len(df)*100:.1f}%",
}

import json
metadata_path = os.path.join(output_dir, "dataset_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved: {metadata_path}")

print("\n" + "="*70)
print("✓ SUCCESSFULLY SAVED ALL DATASETS!")
print("="*70)
print(f"\nDatasets location: {os.path.abspath(output_dir)}/")
print("\nYou can now use these datasets for:")
print("  - Model training with custom splits")
print("  - Cross-validation experiments")
print("  - Dataset analysis and exploration")
print("="*70)
