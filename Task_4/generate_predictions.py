"""
Generate a CSV with final_text, ground_truth, and model_prediction columns
using the trained Random Forest model artifacts.
Predictions are made on the held-out TEST SET only (20% of data).
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "Task_3" / "Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv"
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
OUTPUT_PATH = SCRIPT_DIR / "model_predictions.csv"

VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "sentiment_model.joblib"
LABEL_CLASSES_PATH = ARTIFACT_DIR / "label_classes.joblib"
TEST_INDICES_PATH = ARTIFACT_DIR / "test_indices.joblib"


def main():
    # ── 1. Load model artifacts ──────────────────────────────
    print("Loading model artifacts...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    classes = joblib.load(LABEL_CLASSES_PATH)
    test_indices = joblib.load(TEST_INDICES_PATH)
    print(f"  Model type    : {type(model).__name__}")
    print(f"  Classes       : {list(classes)}")
    print(f"  Test set size : {len(test_indices)}")

    # ── 2. Load and clean data ───────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["final_text", "ground_truth"])
    df["final_text"] = df["final_text"].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.lower()
    df = df[df["final_text"] != ""]
    df = df.reset_index(drop=True)

    # ── 3. Extract test set ──────────────────────────────────
    df_test = df.iloc[test_indices].copy()
    print(f"  Test samples: {len(df_test)}")

    # ── 4. Generate predictions ──────────────────────────────
    print("Generating predictions on test set...")
    X_test = vectorizer.transform(df_test["final_text"].values)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # ── 5. Build output DataFrame ────────────────────────────
    output_df = pd.DataFrame({
        "final_text": df_test["final_text"].values,
        "ground_truth": df_test["ground_truth"].values,
        "model_prediction": predictions,
    })

    # Add confidence (max probability) for each prediction
    output_df["confidence"] = probabilities.max(axis=1).round(4)

    # Add per-class probabilities
    for i, cls in enumerate(classes):
        output_df[f"prob_{cls}"] = probabilities[:, i].round(4)

    # ── 6. Summary stats ────────────────────────────────────
    accuracy = (output_df["ground_truth"] == output_df["model_prediction"]).mean()
    correct = (output_df["ground_truth"] == output_df["model_prediction"]).sum()
    total = len(output_df)
    print(f"\n  Test Accuracy: {accuracy:.4f}  ({correct}/{total})")
    print(f"\n  Prediction distribution:")
    print(output_df["model_prediction"].value_counts().to_string(header=False))
    print(f"\n  Ground truth distribution:")
    print(output_df["ground_truth"].value_counts().to_string(header=False))

    # ── 7. Save ──────────────────────────────────────────────
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved {len(output_df)} test-set rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
