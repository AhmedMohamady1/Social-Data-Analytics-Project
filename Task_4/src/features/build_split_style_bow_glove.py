
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import gensim.downloader as api
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import TASK_3_ROOT, CLEANING_SCRIPT_PATH, DATA_DIR, PROJECT_ROOT


BOW_NGRAM_RANGE = (1, 2)
TFIDF_NGRAM_RANGE = (1, 2)
GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"

STYLE_PROFILE_ARGS = {
    "original": [
        "--convert_emojis",
        "--remove_urls",
        "--normalize_whitespace",
    ],
    "style_b": [
        "--remove_mastodon_artifacts",
        "--remove_html_tags",
        "--remove_social_tags",
        "--remove_numbers",
        "--remove_punctuation",
        "--normalize_whitespace",
        "--lemmatize",
    ],
    "style_c": [
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
    ],
}
COMMON_STYLE_ARGS = ["--lang_mode", "drop", "--limit", "500"]


def load_positive_train_rows(positive_csv_path: Path, next_row_id_start: int) -> pd.DataFrame:
    """Load additional rows and prepare them for train-only augmentation."""
    if not positive_csv_path.exists():
        raise FileNotFoundError(f"Positive train CSV not found: {positive_csv_path}")

    df_positive = pd.read_csv(positive_csv_path)
    required_columns = ["sentiment_text", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in df_positive.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in positive_train_csv: {missing_columns}"
        )

    df_positive = df_positive.copy()
    df_positive["sentiment_text"] = (
        df_positive["sentiment_text"].fillna("").astype(str).str.strip()
    )
    df_positive["ground_truth"] = (
        df_positive["ground_truth"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    df_positive = df_positive[df_positive["sentiment_text"] != ""].reset_index(drop=True)

    if df_positive.empty:
        raise ValueError("positive_train_csv has no non-empty sentiment_text rows.")

    if "row_id" in df_positive.columns:
        df_positive = df_positive.drop(columns=["row_id"])

    df_positive.insert(
        0,
        "row_id",
        np.arange(next_row_id_start, next_row_id_start + len(df_positive), dtype=int),
    )
    return df_positive


def run_cleaning_style(
    *,
    split_name: str,
    style_name: str,
    cleaning_script_path: Path,
    input_csv: Path,
    output_csv: Path,
) -> None:
    """Run cleaning_pipeline.py for one split/style pair."""
    cmd = [
        sys.executable,
        str(cleaning_script_path),
        "--input",
        str(input_csv),
        "--output",
        str(output_csv),
        *COMMON_STYLE_ARGS,
        *STYLE_PROFILE_ARGS[style_name],
    ]

    run = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"[{split_name}/{style_name}] cleaning stdout:")
    print(run.stdout.strip())
    if run.stderr.strip():
        print(f"[{split_name}/{style_name}] cleaning warnings:")
        print(run.stderr.strip())


def preprocess_split(
    *,
    split_name: str,
    df_split: pd.DataFrame,
    preprocessing_temp_dir: Path,
    cleaning_script_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.Series], dict[str, Path]]:
    """Preprocess one split into original/style_b/style_c and align by row_id."""
    split_input_csv = preprocessing_temp_dir / f"working_input_{split_name}.csv"
    # Keep all original split columns in temp input so preprocessing outputs retain full headers.
    df_for_styles = df_split.copy()
    df_for_styles.to_csv(split_input_csv, index=False)

    style_files = {
        "original": preprocessing_temp_dir / f"Cleaned_Iran_War_Sentiment_{split_name}_style_original.csv",
        "style_b": preprocessing_temp_dir / f"Cleaned_Iran_War_Sentiment_{split_name}_style_b.csv",
        "style_c": preprocessing_temp_dir / f"Cleaned_Iran_War_Sentiment_{split_name}_style_c.csv",
    }

    for style_name, out_path in style_files.items():
        run_cleaning_style(
            split_name=split_name,
            style_name=style_name,
            cleaning_script_path=cleaning_script_path,
            input_csv=split_input_csv,
            output_csv=out_path,
        )

    preprocessed_datasets = {
        "original": pd.read_csv(style_files["original"]),
        "style_b": pd.read_csv(style_files["style_b"]),
        "style_c": pd.read_csv(style_files["style_c"]),
    }

    for style_name, frame in preprocessed_datasets.items():
        for required_col in ["row_id", "final_text"]:
            if required_col not in frame.columns:
                raise ValueError(
                    f"Missing column '{required_col}' in {split_name}/{style_name} output"
                )

    preprocessed_datasets["original"] = preprocessed_datasets["original"].rename(
        columns={"final_text": "final_text_original"}
    )
    preprocessed_datasets["style_b"] = preprocessed_datasets["style_b"].rename(
        columns={"final_text": "final_text_style_b"}
    )
    preprocessed_datasets["style_c"] = preprocessed_datasets["style_c"].rename(
        columns={"final_text": "final_text_style_c"}
    )

    common_row_ids = sorted(
        set(preprocessed_datasets["original"]["row_id"]).intersection(
            preprocessed_datasets["style_b"]["row_id"],
            preprocessed_datasets["style_c"]["row_id"],
        )
    )
    if not common_row_ids:
        raise ValueError(f"No overlapping row_id values found in split '{split_name}'.")

    df_split_aligned = (
        df_split[df_split["row_id"].isin(common_row_ids)]
        .sort_values("row_id")
        .reset_index(drop=True)
        .copy()
    )

    for style_name in ["original", "style_b", "style_c"]:
        preprocessed_datasets[style_name] = (
            preprocessed_datasets[style_name][
                preprocessed_datasets[style_name]["row_id"].isin(common_row_ids)
            ]
            .sort_values("row_id")
            .reset_index(drop=True)
            .copy()
        )

    # Save aligned preprocessed split outputs back to disk.
    preprocessed_datasets["original"].to_csv(style_files["original"], index=False)
    preprocessed_datasets["style_b"].to_csv(style_files["style_b"], index=False)
    preprocessed_datasets["style_c"].to_csv(style_files["style_c"], index=False)

    assert len(preprocessed_datasets["original"]) == len(df_split_aligned)
    assert len(preprocessed_datasets["style_b"]) == len(df_split_aligned)
    assert len(preprocessed_datasets["style_c"]) == len(df_split_aligned)
    assert preprocessed_datasets["original"]["row_id"].equals(df_split_aligned["row_id"])
    assert preprocessed_datasets["style_b"]["row_id"].equals(df_split_aligned["row_id"])
    assert preprocessed_datasets["style_c"]["row_id"].equals(df_split_aligned["row_id"])

    style_texts = {
        "original": preprocessed_datasets["original"]["final_text_original"]
        .fillna("")
        .astype(str)
        .str.strip(),
        "style_b": preprocessed_datasets["style_b"]["final_text_style_b"]
        .fillna("")
        .astype(str)
        .str.strip(),
        "style_c": preprocessed_datasets["style_c"]["final_text_style_c"]
        .fillna("")
        .astype(str)
        .str.strip(),
    }

    base_text = style_texts["original"]
    for style_name, series in style_texts.items():
        changed_rows = int((series != base_text).sum())
        print(
            f"[{split_name}] {style_name}: {changed_rows} rows differ "
            "from original style output"
        )

    return df_split_aligned, preprocessed_datasets, style_texts, style_files


def build_bow_and_glove_for_split(
    *,
    split_name: str,
    df_split_aligned: pd.DataFrame,
    style_texts: dict[str, pd.Series],
    features_dir: Path,
    glove_model,
) -> list[dict[str, object]]:
    """Build BoW and weighted GloVe features for one split across all styles."""
    split_style_summary: list[dict[str, object]] = []
    glove_dim = int(glove_model.vector_size)

    for style_name, text_series in style_texts.items():
        text_series = text_series.fillna("").astype(str).str.strip()
        if int((text_series != "").sum()) == 0:
            raise ValueError(
                f"Split '{split_name}' style '{style_name}' has no non-empty text."
            )

        bow_vectorizer = CountVectorizer(ngram_range=BOW_NGRAM_RANGE)
        X_bow = bow_vectorizer.fit_transform(text_series)
        bow_columns = [f"bow_f{idx}" for idx in range(X_bow.shape[1])]

        bow_df = pd.DataFrame(X_bow.toarray().astype(np.int16), columns=bow_columns)
        bow_df.insert(0, "ground_truth", df_split_aligned["ground_truth"].values)
        bow_df.insert(0, "row_id", df_split_aligned["row_id"].values)

        bow_csv_path = features_dir / f"bow_{split_name}_{style_name}.csv"
        bow_json_path = features_dir / f"bow_{split_name}_{style_name}.json"
        bow_metadata_path = features_dir / f"bow_{split_name}_{style_name}_metadata.json"
        bow_df.to_csv(bow_csv_path, index=False)
        bow_df.to_json(bow_json_path, orient="records", force_ascii=True)

        tfidf_vectorizer = TfidfVectorizer(ngram_range=TFIDF_NGRAM_RANGE)
        X_tfidf = tfidf_vectorizer.fit_transform(text_series)

        vocab = tfidf_vectorizer.vocabulary_
        idf = tfidf_vectorizer.idf_
        term_idf = {term: float(idf[idx]) for term, idx in vocab.items()}

        glove_vectors = np.zeros((len(text_series), glove_dim), dtype=np.float32)
        glove_tokens_covered = 0
        glove_tokens_total = 0

        for row_idx, text in enumerate(text_series):
            weighted_sum = np.zeros(glove_dim, dtype=np.float32)
            weight_total = 0.0
            for token in text.split():
                glove_tokens_total += 1
                if token in glove_model:
                    weight = term_idf.get(token, 1.0)
                    weighted_sum += glove_model[token] * weight
                    weight_total += weight
                    glove_tokens_covered += 1
            if weight_total > 0.0:
                glove_vectors[row_idx] = weighted_sum / weight_total

        glove_columns = [f"glove_f{idx}" for idx in range(glove_dim)]
        glove_df = pd.DataFrame(glove_vectors, columns=glove_columns)
        glove_df.insert(0, "ground_truth", df_split_aligned["ground_truth"].values)
        glove_df.insert(0, "row_id", df_split_aligned["row_id"].values)

        glove_csv_path = features_dir / f"glove_{split_name}_{style_name}.csv"
        glove_json_path = features_dir / f"glove_{split_name}_{style_name}.json"
        glove_metadata_path = features_dir / f"glove_{split_name}_{style_name}_metadata.json"
        glove_df.to_csv(glove_csv_path, index=False)
        glove_df.to_json(glove_json_path, orient="records", force_ascii=True)

        bow_density = (
            float(X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]))
            if X_bow.shape[1] > 0
            else 0.0
        )
        glove_non_zero = int(np.count_nonzero(glove_vectors))
        glove_density = (
            float(glove_non_zero / (glove_vectors.shape[0] * glove_vectors.shape[1]))
            if glove_vectors.shape[1] > 0
            else 0.0
        )
        coverage_rate = (
            float(glove_tokens_covered / glove_tokens_total)
            if glove_tokens_total > 0
            else 0.0
        )

        bow_metadata_payload = {
            "split": split_name,
            "style": style_name,
            "representation": "bow",
            "ngram_range": [int(BOW_NGRAM_RANGE[0]), int(BOW_NGRAM_RANGE[1])],
            "rows": int(X_bow.shape[0]),
            "features": int(X_bow.shape[1]),
            "non_zero_entries": int(X_bow.nnz),
            "density": bow_density,
            "feature_name_sample": bow_vectorizer.get_feature_names_out()[:50].tolist(),
        }
        with bow_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(bow_metadata_payload, f, indent=2, ensure_ascii=True)

        glove_metadata_payload = {
            "split": split_name,
            "style": style_name,
            "representation": "glove",
            "model_name": GLOVE_MODEL_NAME,
            "rows": int(glove_vectors.shape[0]),
            "dimensions": int(glove_vectors.shape[1]),
            "non_zero_entries": glove_non_zero,
            "density": glove_density,
            "tokens_total": int(glove_tokens_total),
            "tokens_covered": int(glove_tokens_covered),
            "coverage_rate": coverage_rate,
        }
        with glove_metadata_path.open("w", encoding="utf-8") as f:
            json.dump(glove_metadata_payload, f, indent=2, ensure_ascii=True)

        split_style_summary.append(
            {
                "split": split_name,
                "style": style_name,
                "rows": int(len(text_series)),
                "bow_features": int(X_bow.shape[1]),
                "bow_non_zero_entries": int(X_bow.nnz),
                "bow_density": bow_density,
                "bow_csv": str(bow_csv_path),
                "bow_json": str(bow_json_path),
                "bow_metadata_json": str(bow_metadata_path),
                "bow_metadata": bow_metadata_payload,
                "glove_dimensions": glove_dim,
                "glove_non_zero_entries": glove_non_zero,
                "glove_density": glove_density,
                "glove_tokens_total": int(glove_tokens_total),
                "glove_tokens_covered": int(glove_tokens_covered),
                "glove_coverage_rate": coverage_rate,
                "glove_csv": str(glove_csv_path),
                "glove_json": str(glove_json_path),
                "glove_metadata_json": str(glove_metadata_path),
                "glove_metadata": glove_metadata_payload,
            }
        )

    return split_style_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split-first style preprocessing + BoW/GloVe feature builder "
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=TASK_3_ROOT / "Cleaned_Iran_War_Sentiment_with_Sentiment_Labels.csv",
    )
    parser.add_argument(
        "--cleaning_script",
        type=Path,
        default=CLEANING_SCRIPT_PATH,
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DATA_DIR / "split_style_bow_glove_outputs",
    )
    parser.add_argument(
        "--positive_train_csv",
        type=Path,
        default=DATA_DIR / "positive.csv",
        help="CSV file to append to the train split only after splitting.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    input_csv_path = args.input_csv
    cleaning_script_path = args.cleaning_script
    output_root = args.output_root
    positive_train_csv_path = args.positive_train_csv

    preprocessing_temp_dir = output_root / "preprocessing_temp"
    features_dir = output_root / "three_style_features"
    output_root.mkdir(parents=True, exist_ok=True)
    preprocessing_temp_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest_json = output_root / "split_dataset_manifest.json"
    summary_json = features_dir / "split_style_feature_summary.json"

    print(f"Input CSV: {input_csv_path}")
    print(f"Cleaning script: {cleaning_script_path}")
    print(f"Positive train CSV: {positive_train_csv_path}")
    print(f"Output root: {output_root}")
    print(f"Preprocessing temp dir: {preprocessing_temp_dir}")
    print(f"Features dir: {features_dir}")

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    if not cleaning_script_path.exists():
        raise FileNotFoundError(f"Cleaning script not found: {cleaning_script_path}")

    df_raw = pd.read_csv(input_csv_path)
    required_columns = ["sentiment_text", "ground_truth"]
    missing_columns = [col for col in required_columns if col not in df_raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df_clean = df_raw.copy()
    df_clean["sentiment_text"] = df_clean["sentiment_text"].fillna("").astype(str).str.strip()
    df_clean["ground_truth"] = (
        df_clean["ground_truth"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    df_clean = df_clean[df_clean["sentiment_text"] != ""].copy().reset_index(drop=True)
    df_clean.insert(0, "row_id", np.arange(len(df_clean), dtype=int))

    print(f"Rows loaded: {len(df_raw)}")
    print(f"Rows after cleaning/filtering: {len(df_clean)}")
    print("Label distribution:")
    print(df_clean["ground_truth"].value_counts())

    df_train, df_test = train_test_split(
        df_clean,
        test_size=args.test_size,
        stratify=df_clean["ground_truth"],
        random_state=args.random_state,
    )
    df_train = df_train.sort_values("row_id").reset_index(drop=True)
    df_test = df_test.sort_values("row_id").reset_index(drop=True)

    train_rows_before_augmentation = int(len(df_train))
    test_rows_before_augmentation = int(len(df_test))

    print(f"Train rows (before augmentation): {train_rows_before_augmentation}")
    print(f"Test rows (before augmentation): {test_rows_before_augmentation}")

    next_row_id = int(df_clean["row_id"].max()) + 1
    df_positive_train = load_positive_train_rows(
        positive_csv_path=positive_train_csv_path,
        next_row_id_start=next_row_id,
    )

    # Align schemas to preserve all headers in preprocessed temp outputs.
    for col in df_train.columns:
        if col not in df_positive_train.columns:
            df_positive_train[col] = np.nan

    for col in df_positive_train.columns:
        if col not in df_train.columns:
            df_train[col] = np.nan
            df_test[col] = np.nan

    aligned_columns = list(df_train.columns)
    df_positive_train = df_positive_train[aligned_columns]

    positive_rows_added = int(len(df_positive_train))
    df_train = (
        pd.concat([df_train, df_positive_train], ignore_index=True)
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    print(f"Added rows to train from positive_train_csv: {positive_rows_added}")
    print(f"Train rows (after augmentation): {len(df_train)}")
    print(f"Test rows (after augmentation): {len(df_test)}")
    print("Train label distribution after augmentation:")
    print(df_train["ground_truth"].value_counts())

    split_inputs = {"train": df_train, "test": df_test}

    split_preprocessed_info: dict[str, dict[str, str]] = {}
    split_summaries: list[dict[str, object]] = []

    print(f"Loading GloVe model: {GLOVE_MODEL_NAME}")
    glove_model = api.load(GLOVE_MODEL_NAME)

    for split_name, df_split in split_inputs.items():
        print(f"\n=== Processing split: {split_name} ===")

        df_split_aligned, _, style_texts, style_files = preprocess_split(
            split_name=split_name,
            df_split=df_split,
            preprocessing_temp_dir=preprocessing_temp_dir,
            cleaning_script_path=cleaning_script_path,
        )

        split_preprocessed_info[split_name] = {
            style: str(path) for style, path in style_files.items()
        }

        split_summary = build_bow_and_glove_for_split(
            split_name=split_name,
            df_split_aligned=df_split_aligned,
            style_texts=style_texts,
            features_dir=features_dir,
            glove_model=glove_model,
        )
        split_summaries.extend(split_summary)

    summary_df = pd.DataFrame(split_summaries)

    for split_name in ["train", "test"]:
        split_df = summary_df[summary_df["split"] == split_name]
        if split_df.empty:
            raise ValueError(f"No feature summary rows found for split '{split_name}'.")
        if split_df["rows"].nunique() != 1:
            raise ValueError(f"Row count differs across styles in split '{split_name}'.")
        if not (split_df["bow_features"] > 0).all():
            raise ValueError(f"At least one style has empty BoW vocabulary in split '{split_name}'.")
        if not (split_df["glove_dimensions"] > 0).all():
            raise ValueError(f"At least one style has invalid GloVe dimensions in split '{split_name}'.")

        for _, row in split_df.iterrows():
            required_files = [
                row["bow_csv"],
                row["bow_json"],
                row["bow_metadata_json"],
                row["glove_csv"],
                row["glove_json"],
                row["glove_metadata_json"],
            ]
            for file_path in required_files:
                if not Path(file_path).exists():
                    raise FileNotFoundError(f"Missing output file: {file_path}")

    manifest_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv_path),
        "cleaning_script": str(cleaning_script_path),
        "positive_train_csv": str(positive_train_csv_path),
        "positive_train_rows_added": positive_rows_added,
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "bow_ngram_range": [BOW_NGRAM_RANGE[0], BOW_NGRAM_RANGE[1]],
        "tfidf_ngram_range_for_glove_weighting": [
            TFIDF_NGRAM_RANGE[0],
            TFIDF_NGRAM_RANGE[1],
        ],
        "glove_model_name": GLOVE_MODEL_NAME,
        "pipeline_common_args": COMMON_STYLE_ARGS,
        "pipeline_style_args": STYLE_PROFILE_ARGS,
        "split_preprocessed_style_files": split_preprocessed_info,
        "rows_after_split_before_preprocessing": {
            "train": train_rows_before_augmentation,
            "test": test_rows_before_augmentation,
        },
        "rows_after_train_augmentation_before_preprocessing": {
            "train": int(len(df_train)),
            "test": int(len(df_test)),
        },
        "label_distribution_after_cleaning": {
            k: int(v) for k, v in df_clean["ground_truth"].value_counts().to_dict().items()
        },
        "train_label_distribution_after_augmentation": {
            k: int(v) for k, v in df_train["ground_truth"].value_counts().to_dict().items()
        },
    }

    summary_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representations": ["bow", "glove"],
        "styles": split_summaries,
    }

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=True)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=True)

    print("\nValidation passed for BoW and GloVe across train/test and all three styles.")
    print(f"Manifest saved: {manifest_json}")
    print(f"Feature summary saved: {summary_json}")
    print(f"Feature files saved in: {features_dir}")


if __name__ == "__main__":
    main()
