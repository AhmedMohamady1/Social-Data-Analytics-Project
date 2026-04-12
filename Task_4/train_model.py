"""
Train a Random Forest sentiment classifier on the labeled dataset.
Saves the trained model and TF-IDF vectorizer as .joblib files.
Replace the model later by swapping the .joblib artifacts.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "Task_3" / "Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv"
MODEL_DIR = SCRIPT_DIR / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_classes.joblib"


def main():
    # ── 1. Load data ─────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["final_text", "ground_truth"])
    df["final_text"] = df["final_text"].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.lower()
    df = df[df["final_text"] != ""]

    X_text = df["final_text"].to_numpy(dtype=str)
    y = df["ground_truth"].to_numpy(dtype=str)
    classes = np.array(sorted(set(y)))

    print(f"  Samples : {len(df)}")
    print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 2. TF-IDF vectorisation ──────────────────────────────
    print("Fitting TF-IDF vectorizer (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(X_text)
    print(f"  Feature matrix: {X.shape}")

    # ── 3. Handle class imbalance via oversampling ───────────
    print("Oversampling minority classes...")
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"  After oversampling: {dict(zip(*np.unique(y_res, return_counts=True)))}")

    # ── 4. Train Random Forest ───────────────────────────────
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)

    # ── 5. Evaluate via Stratified K-Fold on original data ───
    print("\nStratified 5-Fold Cross-Validation on original (unsampled) data:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    print(f"  Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  F1 Macro : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")

    # Full-data classification report (for reference)
    y_pred = model.predict(X)
    print("\nFull-data Classification Report:")
    print(classification_report(y, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred, labels=classes))

    # ── 6. Save artifacts ────────────────────────────────────
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(classes, LABEL_ENCODER_PATH)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print(f"  Vectorizer  : {VECTORIZER_PATH.name}")
    print(f"  Model       : {MODEL_PATH.name}")
    print(f"  Label classes: {LABEL_ENCODER_PATH.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
