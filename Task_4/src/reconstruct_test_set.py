"""
Reconstruct the test set used by random_forest_glove_style_c_p50.pkl.

Replicates the exact split logic from build_split_style_bow_glove.py:
  1. Load Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv (the labeled 500 rows)
  2. Clean and assign row_id (same logic as the pipeline)
  3. train_test_split(test_size=0.2, stratify=ground_truth, random_state=42)
  4. Join test row_ids back to the original Iran_War_Sentiment.csv

Output: Task_4/data/test_set_with_ground_truth.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Task_4/
TASK_3_ROOT = PROJECT_ROOT.parent / "Task_3"

# --- Paths ---
labeled_csv = TASK_3_ROOT / "Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv"
original_csv = PROJECT_ROOT.parent / "Task_2" / "Iran_War_Sentiment.csv"
existing_test_features = PROJECT_ROOT / "data" / "glove_test_style_c.csv"
output_csv = PROJECT_ROOT / "data" / "test_set_with_ground_truth.csv"

# --- 1. Replicate the exact split (same as build_split_style_bow_glove.py lines 437-463) ---
df_raw = pd.read_csv(labeled_csv)
print(f"Labeled CSV loaded: {len(df_raw)} rows")

df_clean = df_raw.copy()
df_clean["sentiment_text"] = df_clean["sentiment_text"].fillna("").astype(str).str.strip()
df_clean["ground_truth"] = (
    df_clean["ground_truth"].fillna("unknown").astype(str).str.strip().str.lower()
)
df_clean = df_clean[df_clean["sentiment_text"] != ""].copy().reset_index(drop=True)
df_clean.insert(0, "row_id", np.arange(len(df_clean), dtype=int))

print(f"Rows after cleaning: {len(df_clean)}")
print(f"Label distribution:\n{df_clean['ground_truth'].value_counts()}")

df_train, df_test = train_test_split(
    df_clean,
    test_size=0.2,
    stratify=df_clean["ground_truth"],
    random_state=42,
)
df_test = df_test.sort_values("row_id").reset_index(drop=True)
print(f"\nTest split size: {len(df_test)} rows")

# --- 2. Validate against the existing feature file ---
if existing_test_features.exists():
    df_existing = pd.read_csv(existing_test_features)
    existing_row_ids = set(df_existing["row_id"].values)
    reconstructed_row_ids = set(df_test["row_id"].values)

    if existing_row_ids == reconstructed_row_ids:
        print("✅ Row IDs match the existing glove_test_style_c.csv perfectly!")
    else:
        diff = existing_row_ids.symmetric_difference(reconstructed_row_ids)
        print(f"⚠️  Row ID mismatch! {len(diff)} IDs differ.")
        print(f"   In existing but not reconstructed: {existing_row_ids - reconstructed_row_ids}")
        print(f"   In reconstructed but not existing: {reconstructed_row_ids - existing_row_ids}")
else:
    print("ℹ️  No existing glove_test_style_c.csv found to validate against.")

# --- 3. Join with original Iran_War_Sentiment.csv ---
# The labeled CSV is the first 500 rows of Iran_War_Sentiment.csv
# (based on --limit 500 in the pipeline). The row order is preserved,
# so row_id maps directly to the positional index in the labeled CSV,
# which itself maps to the same positional index in the original CSV.

df_original = pd.read_csv(original_csv)
print(f"\nOriginal CSV loaded: {len(df_original)} rows, columns: {list(df_original.columns)}")

# The labeled CSV was derived from the first 500 rows of the original.
# We use the labeled CSV as the source of truth since it has ground_truth.
# The row_id is the positional index after cleaning empty rows, so we
# can directly index into df_clean.

# Build the output: original columns + ground_truth
test_indices = df_test["row_id"].values

# Get the original text rows corresponding to test row_ids.
# df_clean was built from df_raw (labeled CSV) which came from the first 500
# of the original CSV. We'll use df_raw with the same cleaning to get the
# original data columns, then attach ground_truth.

# df_clean already has all columns from the labeled CSV plus row_id.
# We just need to select the test rows and keep useful columns.
output_df = df_test.copy()

# Reorder: put ground_truth as the last column
all_cols = [c for c in output_df.columns if c != "ground_truth"]
column_order = all_cols + ["ground_truth"]
output_df = output_df[column_order]

output_df.to_csv(output_csv, index=False)
print(f"\n✅ Test set saved to: {output_csv}")
print(f"   Rows: {len(output_df)}")
print(f"   Columns: {list(output_df.columns)}")
print(f"\nGround truth distribution in test set:")
print(output_df["ground_truth"].value_counts())
