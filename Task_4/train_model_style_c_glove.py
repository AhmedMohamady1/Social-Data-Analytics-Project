"""
Train a Random Forest sentiment classifier on the style_c preprocessed dataset
using GloVe document embeddings.

Input style_c data is expected from Task_3 preprocessing output:
Task_3/preprocessing_temp/Cleaned_Iran_War_Sentiment_style_c.csv
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim import downloader as api
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
STYLE_C_DATA_PATH = (
    PROJECT_ROOT
    / "Task_3"
    / "preprocessing_temp"
    / "Cleaned_Iran_War_Sentiment_style_c.csv"
)
MODEL_DIR = SCRIPT_DIR / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"

EMBEDDER_INFO_PATH = MODEL_DIR / "glove_style_c_embedder_info.joblib"
MODEL_PATH = MODEL_DIR / "sentiment_model_style_c_glove.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_classes_style_c_glove.joblib"
TEST_ROW_IDS_PATH = MODEL_DIR / "test_row_ids_style_c_glove.joblib"


def text_to_glove_embeddings(texts: np.ndarray, glove_model) -> tuple[np.ndarray, int, int, float]:
    """Convert text rows into mean GloVe vectors."""
    dim = int(glove_model.vector_size)
    vectors = np.zeros((len(texts), dim), dtype=np.float32)

    tokens_total = 0
    tokens_covered = 0

    for row_idx, text in enumerate(texts):
        token_count = 0
        vector_sum = np.zeros(dim, dtype=np.float32)

        for token in str(text).split():
            tokens_total += 1
            if token in glove_model:
                vector_sum += glove_model[token]
                token_count += 1
                tokens_covered += 1

        if token_count > 0:
            vectors[row_idx] = vector_sum / token_count

    coverage_rate = float(tokens_covered / tokens_total) if tokens_total > 0 else 0.0
    return vectors, tokens_total, tokens_covered, coverage_rate


def main() -> None:
    # -- 1. Load style_c data --------------------------------------------------
    print("Loading style_c preprocessed data...")
    if not STYLE_C_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Style_c dataset not found: {STYLE_C_DATA_PATH}. "
            "Run Task_3/text_representation.ipynb preprocessing cells first."
        )

    df = pd.read_csv(STYLE_C_DATA_PATH)

    text_column_candidates = ["final_text_style_c", "final_text", "sentiment_text"]
    text_column = next((col for col in text_column_candidates if col in df.columns), None)
    if text_column is None:
        raise ValueError(
            "No usable text column found. Expected one of: "
            f"{text_column_candidates}"
        )
    if "ground_truth" not in df.columns:
        raise ValueError("Missing required column: ground_truth")

    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=int)

    df = df.dropna(subset=[text_column, "ground_truth"]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.lower()
    df = df[df[text_column] != ""].reset_index(drop=True)

    X_text = df[text_column].to_numpy(dtype=str)
    y = df["ground_truth"].to_numpy(dtype=str)
    row_ids = df["row_id"].to_numpy()
    classes = np.array(sorted(set(y)))

    print(f"  Total samples : {len(df)}")
    print(f"  Text column   : {text_column}")
    print(f"  Classes       : {dict(zip(*np.unique(y, return_counts=True)))}")

    # -- 2. Train / Test split (80/20) ----------------------------------------
    print("\nSplitting data 80/20...")
    X_text_train, X_text_test, y_train, y_test, row_id_train, row_id_test = train_test_split(
        X_text,
        y,
        row_ids,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"  Train : {len(X_text_train)}  |  Test : {len(X_text_test)}")

    # -- 3. GloVe embedding (fit transform style) ------------------------------
    print(f"Loading GloVe model: {GLOVE_MODEL_NAME}")
    glove_model = api.load(GLOVE_MODEL_NAME)

    print("Converting text to GloVe document embeddings...")
    X_train, train_tokens_total, train_tokens_covered, train_coverage = text_to_glove_embeddings(
        X_text_train,
        glove_model,
    )
    X_test, test_tokens_total, test_tokens_covered, test_coverage = text_to_glove_embeddings(
        X_text_test,
        glove_model,
    )

    print(f"  Train feature matrix : {X_train.shape}")
    print(f"  Test feature matrix  : {X_test.shape}")
    print(
        f"  Train token coverage : {train_tokens_covered}/{train_tokens_total} "
        f"({train_coverage:.2%})"
    )
    print(
        f"  Test token coverage  : {test_tokens_covered}/{test_tokens_total} "
        f"({test_coverage:.2%})"
    )

    # -- 4. Handle class imbalance via oversampling (train only) ---------------
    print("Oversampling minority classes (training set only)...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    print(f"  After oversampling: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

    # -- 5. Train Random Forest -------------------------------------------------
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_res, y_train_res)

    # -- 6. Evaluate on held-out test set --------------------------------------
    y_test_pred = model.predict(X_test)
    print("\n=== Test Set Evaluation (20% held-out) ===")
    print(classification_report(y_test, y_test_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred, labels=classes))

    # -- 7. Cross-validation on training data ----------------------------------
    print("\nStratified 5-Fold Cross-Validation on training data:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=False,
    )
    print(f"  Accuracy : {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
    print(f"  F1 Macro : {cv_results['test_f1_macro'].mean():.4f} +/- {cv_results['test_f1_macro'].std():.4f}")

    # -- 8. Save artifacts ------------------------------------------------------
    embedder_info = {
        "representation": "glove_mean",
        "glove_model_name": GLOVE_MODEL_NAME,
        "embedding_dim": int(X_train.shape[1]),
        "source_csv": str(STYLE_C_DATA_PATH),
        "text_column": text_column,
        "train_token_coverage": train_coverage,
        "test_token_coverage": test_coverage,
    }

    joblib.dump(embedder_info, EMBEDDER_INFO_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(classes, LABEL_ENCODER_PATH)
    joblib.dump(row_id_test, TEST_ROW_IDS_PATH)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    print(f"  Embedder info : {EMBEDDER_INFO_PATH.name}")
    print(f"  Model         : {MODEL_PATH.name}")
    print(f"  Label classes : {LABEL_ENCODER_PATH.name}")
    print(f"  Test row IDs  : {TEST_ROW_IDS_PATH.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
