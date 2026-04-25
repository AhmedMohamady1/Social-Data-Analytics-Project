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

from src.config import TASK_3_ROOT, GLOVE_MODEL_NAME

@lru_cache(maxsize=1)
def __get_preprocessing_pipeline():
    # Insert path to Task_3/src into sys.path to import cleaning_pipeline
    task3_src = str(TASK_3_ROOT / "src")
    if task3_src not in sys.path:
        sys.path.append(task3_src)
    
    from cleaning_pipeline import PreprocessingPipeline
    return PreprocessingPipeline()

def convert_text_to_style_c(text: str, *, python_executable: str | None = None) -> str:
    """Apply the exact style_c cleaning pipeline to one text string IN MEMORY."""
    raw_text = "" if text is None else str(text).strip()
    if not raw_text:
        raise ValueError("Input text is empty.")

    from langdetect import detect
    try:
        if detect(raw_text) != 'en':
            raise ValueError("The text was dropped by style_c preprocessing. This usually happens for non-English text when lang_mode=drop.")
    except Exception as e:
        if "No features in text" in str(e) or "dropped by style_c" in str(e):
            raise ValueError("The text was dropped by style_c preprocessing. This usually happens for non-English text when lang_mode=drop.")

    pipeline = __get_preprocessing_pipeline()

    processed = raw_text
    processed = pipeline.convert_emojis(processed)
    processed = pipeline.remove_mastodon_artifacts(processed)
    processed = pipeline.remove_urls(processed)
    processed = pipeline.remove_html_tags(processed)
    processed = pipeline.remove_social_tags(processed)
    processed = pipeline.remove_numbers(processed)
    processed = pipeline.remove_punctuation(processed)
    processed = pipeline.normalize_whitespace(processed)

    processed = str(processed).lower()

    processed = pipeline.fix_spelling(processed)
    processed = pipeline.remove_stopwords(processed)
    processed = pipeline.lemmatize_text(processed)
    
    if not processed.strip():
        raise ValueError("style_c preprocessing returned empty text.")
    
    return processed.strip()


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
