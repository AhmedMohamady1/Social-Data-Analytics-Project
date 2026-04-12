from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from gensim import downloader as api

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
CLEANING_SCRIPT_PATH = PROJECT_ROOT / "Task_3" / "cleaning_pipeline.py"

GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"

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
def _load_glove_model(model_name: str = GLOVE_MODEL_NAME):
    """Load and cache GloVe model to avoid repeated downloads/loads."""
    return api.load(model_name)


def apply_glove_to_text(
    text: str,
    *,
    glove_model_name: str = GLOVE_MODEL_NAME,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Convert one preprocessed text string into a mean GloVe vector."""
    text = str(text).strip()
    if not text:
        raise ValueError("Cannot apply GloVe on empty text.")

    glove_model = _load_glove_model(glove_model_name)
    vector_dim = int(glove_model.vector_size)

    vector = np.zeros(vector_dim, dtype=np.float32)
    tokens = text.split()
    tokens_total = len(tokens)
    tokens_covered = 0

    for token in tokens:
        if token in glove_model:
            vector += glove_model[token]
            tokens_covered += 1

    if tokens_covered > 0:
        vector /= float(tokens_covered)

    coverage_rate = float(tokens_covered / tokens_total) if tokens_total > 0 else 0.0
    metadata: dict[str, float | int | str] = {
        "glove_model_name": glove_model_name,
        "embedding_dim": vector_dim,
        "tokens_total": int(tokens_total),
        "tokens_covered": int(tokens_covered),
        "coverage_rate": coverage_rate,
    }

    return vector, metadata


def text_to_style_c_glove(
    text: str,
    *,
    glove_model_name: str = GLOVE_MODEL_NAME,
    python_executable: str | None = None,
) -> tuple[str, np.ndarray, dict[str, float | int | str]]:
    """
    Main inference helper:
    1) Convert raw text to style_c
    2) Convert style_c text to GloVe embedding

    Returns (style_c_text, embedding_vector, metadata).
    """
    style_c_text = convert_text_to_style_c(text, python_executable=python_executable)
    glove_vector, metadata = apply_glove_to_text(style_c_text, glove_model_name=glove_model_name)
    return style_c_text, glove_vector, metadata


def text_to_style_c_glove_feature_row(
    text: str,
    *,
    glove_model_name: str = GLOVE_MODEL_NAME,
    python_executable: str | None = None,
) -> tuple[str, np.ndarray, dict[str, float | int | str]]:

    style_c_text, glove_vector, metadata = text_to_style_c_glove(
        text,
        glove_model_name=glove_model_name,
        python_executable=python_executable,
    )
    return style_c_text, glove_vector.reshape(1, -1), metadata


if __name__ == "__main__":
    sample = "War escalations continue and civilians are paying the highest price."
    processed, feature_row, info = text_to_style_c_glove_feature_row(sample)
    print(f"Processed style_c text: {processed}")
    print(f"Feature row shape: {feature_row.shape}")
    print(f"Coverage info: {info}")
