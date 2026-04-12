from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
CLEANING_SCRIPT_PATH = PROJECT_ROOT / "Task_3" / "cleaning_pipeline.py"

STYLE_C_COMMON_ARGS = ["--lang_mode", "drop", "--limit", "500"]
STYLE_C_PROFILE_ARGS = [
    "--convert_emojis",
    "--remove_mastodon_artifacts",
    "--remove_urls",
    "--remove_html_tags",
    "--remove_social_tags",
    "--remove_numbers",
    "--remove_punctuation",
    "--normalize_whitespace",
    "--remove_stopwords",
    "--fix_spelling",
    "--lemmatize",
    "--extract_tags",
]

DEFAULT_BOW_VECTORIZER_PATH = PROJECT_ROOT / "Task_4" / "artifacts" / "bow_vectorizer_style_c.joblib"
DEFAULT_BOW_NGRAM_RANGE = (1, 2)


def convert_text_to_style_c(text: str, *, python_executable: str | None = None) -> str:
    """Apply the exact style_c cleaning pipeline to one text string."""
    raw_text = "" if text is None else str(text).strip()
    if not raw_text:
        raise ValueError("Input text is empty.")

    if not CLEANING_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing cleaning pipeline script: {CLEANING_SCRIPT_PATH}")

    with tempfile.TemporaryDirectory(prefix="style_c_eval_") as temp_dir:
        temp_path = Path(temp_dir)
        input_csv = temp_path / "single_input.csv"
        output_csv = temp_path / "single_output.csv"

        pd.DataFrame(
            {
                "row_id": [0],
                "sentiment_text": [raw_text],
                "ground_truth": ["unknown"],
            }
        ).to_csv(input_csv, index=False)

        cmd = [
            python_executable or sys.executable,
            str(CLEANING_SCRIPT_PATH),
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            *STYLE_C_COMMON_ARGS,
            *STYLE_C_PROFILE_ARGS,
        ]

        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(
                "style_c preprocessing failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stdout: {run.stdout}\n"
                f"Stderr: {run.stderr}"
            )

        if not output_csv.exists():
            raise RuntimeError("style_c preprocessing completed but did not produce output CSV.")

        df_out = pd.read_csv(output_csv)
        if df_out.empty:
            raise ValueError(
                "The text was dropped by style_c preprocessing. "
                "This usually happens for non-English text when lang_mode=drop."
            )

        output_col_candidates = ["final_text_style_c", "final_text", "sentiment_text"]
        output_col = next((col for col in output_col_candidates if col in df_out.columns), None)
        if output_col is None:
            raise ValueError(
                "No processed text column found in preprocessing output. "
                f"Expected one of {output_col_candidates}."
            )

        processed_text = str(df_out.iloc[0][output_col]).strip()
        if not processed_text:
            raise ValueError("style_c preprocessing returned empty text.")

        return processed_text


@lru_cache(maxsize=2)
def _load_bow_vectorizer(vectorizer_path: str):
    """Load and cache a fitted CountVectorizer."""
    return joblib.load(vectorizer_path)


def apply_bow_to_text(
    text: str,
    *,
    vectorizer_path: str | Path | None = DEFAULT_BOW_VECTORIZER_PATH,
    ngram_range: tuple[int, int] = DEFAULT_BOW_NGRAM_RANGE,
    fit_local_if_missing: bool = True,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Convert one preprocessed text string into a BoW vector."""
    text = str(text).strip()
    if not text:
        raise ValueError("Cannot apply Bag-of-Words on empty text.")

    vectorizer = None
    vectorizer_source = ""
    vectorizer_path_used = ""

    if vectorizer_path is not None:
        path_obj = Path(vectorizer_path)
        if path_obj.exists():
            vectorizer = _load_bow_vectorizer(str(path_obj.resolve()))
            vectorizer_source = "artifact"
            vectorizer_path_used = str(path_obj)
        elif not fit_local_if_missing:
            raise FileNotFoundError(
                f"BoW vectorizer not found at {path_obj}. "
                "Train/save a CountVectorizer first or enable fit_local_if_missing."
            )

    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        vectorizer_source = "local_fit_single_text"

    if vectorizer_source == "artifact":
        X_bow = vectorizer.transform([text])
    else:
        X_bow = vectorizer.fit_transform([text])

    bow_vector = X_bow.toarray().astype(np.float32)[0]
    feature_count = int(X_bow.shape[1])
    non_zero_entries = int(X_bow.nnz)
    density = float(non_zero_entries / feature_count) if feature_count > 0 else 0.0

    sample_features: list[str] = []
    if hasattr(vectorizer, "get_feature_names_out"):
        sample_features = vectorizer.get_feature_names_out()[:50].tolist()

    metadata: dict[str, float | int | str | bool] = {
        "representation": "bow",
        "feature_count": feature_count,
        "non_zero_entries": non_zero_entries,
        "density": density,
        "vectorizer_source": vectorizer_source,
        "vectorizer_path": vectorizer_path_used,
        "fit_local_if_missing": bool(fit_local_if_missing),
        "feature_name_sample": ", ".join(sample_features),
    }

    return bow_vector, metadata


def text_to_style_c_bow(
    text: str,
    *,
    vectorizer_path: str | Path | None = DEFAULT_BOW_VECTORIZER_PATH,
    ngram_range: tuple[int, int] = DEFAULT_BOW_NGRAM_RANGE,
    fit_local_if_missing: bool = True,
    python_executable: str | None = None,
) -> tuple[str, np.ndarray, dict[str, float | int | str | bool]]:
    """
    Main inference helper:
    1) Convert raw text to style_c
    2) Convert style_c text to BoW vector

    Returns (style_c_text, bow_vector, metadata).
    """
    style_c_text = convert_text_to_style_c(text, python_executable=python_executable)
    bow_vector, metadata = apply_bow_to_text(
        style_c_text,
        vectorizer_path=vectorizer_path,
        ngram_range=ngram_range,
        fit_local_if_missing=fit_local_if_missing,
    )
    return style_c_text, bow_vector, metadata


def text_to_style_c_bow_feature_row(
    text: str,
    *,
    vectorizer_path: str | Path | None = DEFAULT_BOW_VECTORIZER_PATH,
    ngram_range: tuple[int, int] = DEFAULT_BOW_NGRAM_RANGE,
    fit_local_if_missing: bool = True,
    python_executable: str | None = None,
) -> tuple[str, np.ndarray, dict[str, float | int | str | bool]]:
    """
    Convenience wrapper that returns a 2D feature row of shape (1, feature_count),
    ready for scikit-learn model.predict.
    """
    style_c_text, bow_vector, metadata = text_to_style_c_bow(
        text,
        vectorizer_path=vectorizer_path,
        ngram_range=ngram_range,
        fit_local_if_missing=fit_local_if_missing,
        python_executable=python_executable,
    )
    return style_c_text, bow_vector.reshape(1, -1), metadata


if __name__ == "__main__":
    sample = "War escalations continue and civilians are paying the highest price."
    processed, feature_row, info = text_to_style_c_bow_feature_row(sample)
    print(f"Processed style_c text: {processed}")
    print(f"Feature row shape: {feature_row.shape}")
    print(f"BoW info: {info}")
