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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
TEST_INDICES_PATH = MODEL_DIR / "test_indices.joblib"


def main():
    # ── 1. Load data ─────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["final_text", "ground_truth"])
    df["final_text"] = df["final_text"].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.lower()
    df = df[df["final_text"] != ""]
    df = df.reset_index(drop=True)

    X_text = df["final_text"].to_numpy(dtype=str)
    y = df["ground_truth"].to_numpy(dtype=str)
    classes = np.array(sorted(set(y)))

    print(f"  Total samples : {len(df)}")
    print(f"  Classes       : {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 2. Train / Test split (80/20) ────────────────────────
    print("\nSplitting data 80/20...")
    X_text_train, X_text_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_text, y, np.arange(len(df)),
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"  Train : {len(X_text_train)}  |  Test : {len(X_text_test)}")

    # ── 3. TF-IDF vectorisation (fit on train only) ──────────
    print("Fitting TF-IDF vectorizer on training data (unigrams + bigrams)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_text_train)
    X_test = vectorizer.transform(X_text_test)
    print(f"  Train feature matrix : {X_train.shape}")
    print(f"  Test feature matrix  : {X_test.shape}")

    # ── 4. Handle class imbalance via oversampling (train only)
    print("Oversampling minority classes (training set only)...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print(f"  After oversampling: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    # ── 5. Train Random Forest ───────────────────────────────
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    # ── 6. Evaluate on held-out test set ─────────────────────
    y_test_pred = model.predict(X_test)
    print("\n═══ Test Set Evaluation (20% held-out) ═══")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred, labels=classes))

    # ── 7. Cross-validation on training data ─────────────────
    print("\nStratified 5-Fold Cross-Validation on training data:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    print(f"  Accuracy : {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    print(f"  F1 Macro : {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")

    # ── 8. Save artifacts ────────────────────────────────────
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(classes, LABEL_ENCODER_PATH)
    joblib.dump(idx_test, TEST_INDICES_PATH)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print(f"  Vectorizer    : {VECTORIZER_PATH.name}")
    print(f"  Model         : {MODEL_PATH.name}")
    print(f"  Label classes : {LABEL_ENCODER_PATH.name}")
    print(f"  Test indices  : {TEST_INDICES_PATH.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
