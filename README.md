# Sentiment Analysis of Public Discourse on the Iran War

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135.3-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E.svg)

> An end-to-end NLP pipeline for analyzing public sentiment on the Iran War using Mastodon social media data — from data collection through SOTA transformer benchmarking.
---

## Overview

This project implements a complete sentiment analysis pipeline tracking public opinion on the Iran War. It progresses through five iterative tasks: data collection, text preprocessing, ground truth labeling, model optimization & deployment, and benchmarking against a State-of-the-Art transformer.

**Key Numbers:**
- **2,000** Mastodon posts scraped via API
- **500** records labeled via multi-LLM majority vote (GPT-4o Mini, Gemini 2.5 Flash, Claude 3.5 Haiku)
- **3** sentiment classes: negative (39%), neutral (60%), positive (1%)
- **18** model configurations evaluated across 3 preprocessing styles × 3 models × 2 representations

---

## Project Structure

```
Social-Data-Analytics-Project/
│
├── Task_2/                         # Data Collection & Preprocessing
│   ├── Data_Collection.ipynb       # Mastodon API scraping (2000 posts)
│   ├── cleaning_pipeline.py        # Modular NLP preprocessing (15 flags)
│   ├── Iran_War_Sentiment.csv      # Raw scraped data
│   └── Cleaned_Iran_War_Sentiment.csv
│
├── Task_3/                         # Labeling, Lexicons & Initial ML
│   ├── data/                       # Labeled dataset + lexicon outputs
│   ├── notebooks/
│   │   ├── ground_truth_labeling.ipynb   # LLM annotation + Fleiss' Kappa
│   │   └── text_representation.ipynb     # BoW & GloVe feature engineering
│   ├── src/
│   │   ├── cleaning_pipeline.py    # Pipeline source
│   │   └── sentiwordnet_model.py   # SWN lexicon model
│   ├── lexicons/                   # Bing Liu + SentiWordNet resources
│   └── ml_models/                  # Baseline ML (DT, NB) training
│
├── Task_4/                         # Optimization & Deployment
│   ├── api/api.py                  # FastAPI inference backend
│   ├── app.py                      # Streamlit frontend dashboard
│   ├── src/                        # Feature transforms, config, utils
│   ├── models/                     # Saved RF model + GloVe embeddings
│   ├── notebooks/
│   │   └── Error_Analysis_RF.ipynb # Model failure mode analysis
│   └── data/                       # Train/test splits, predictions
│
├── Task_5/                         # SOTA Benchmarking & Final Report
│   ├── notebooks/
│   │   └── Task_5_SOTA_Comparison.ipynb  # Full benchmarking notebook
│   ├── reports/
│   │   ├── Task_5_Results_Report.docx    # Final comprehensive report
│   │   └── visuals/                      # All generated plots
│   └── data/                       # Prediction outputs
│
├── Tasks/                          # Assignment instruction PDFs
├── requirements.txt                # All project dependencies
└── README.md
```

---

## Pipeline Overview

| Task | Phase | Description |
|---|---|---|
| **Task 2** | Data Collection | Scraped 2,000 Mastodon posts via API; built modular preprocessing pipeline with 15 configurable flags |
| **Task 3** | Labeling & Baselines | Generated ground truth via 3-LLM majority vote (κ = 0.48); evaluated SentiWordNet and Bing Liu lexicon baselines; trained 12 DT/NB configurations |
| **Task 4** | Optimization & Deployment | Added Random Forest with class weighting (positive = 50); deployed via FastAPI + Streamlit; conducted error analysis |
| **Task 5** | SOTA Benchmarking | Compared optimized RF against `cardiffnlp/twitter-roberta-base-sentiment-latest` (125M params); produced comprehensive results report |

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Lexicon (SentiWordNet) | 0.55–0.60 | ~0.38 |
| Lexicon (Bing Liu) | 0.41–0.47 | ~0.38 |
| Baseline ML (DT/NB) | 0.59–0.63 | 0.39–0.41 |
| **Optimized RF** | **0.75** | 0.49 |
| **SOTA RoBERTa** | 0.63 | **0.59** |

The optimized Random Forest achieves the highest accuracy by excelling at the majority class (neutral), while the SOTA transformer achieves the best macro-F1 through more balanced recall across all classes.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Core** | Python, NumPy, Pandas, SciPy |
| **NLP** | NLTK, SpaCy, Gensim, SymSpellPy, emoji, langdetect |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn |
| **Deep Learning** | PyTorch, Hugging Face Transformers |
| **LLM APIs** | LangChain, OpenRouter (GPT-4o Mini, Gemini, Claude) |
| **Deployment** | FastAPI, Uvicorn, Streamlit |
| **Visualization** | Matplotlib, Seaborn |

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/AhmedMohamady1/Social-Data-Analytics-Project.git
cd Social-Data-Analytics-Project

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create `.env` files where needed:

```bash
# Task_3/.env — for LLM ground truth labeling
OPENROUTER_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Task_5/.env — for Hugging Face SOTA model
HF_TOKEN=your_huggingface_token
```

---

## Usage

### Running the Deployment (Task 4)

```bash
# Terminal 1: Start the FastAPI backend
cd Task_4
uvicorn api.api:app --reload

# Terminal 2: Start the Streamlit frontend
cd Task_4
streamlit run app.py
```

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

### Running the SOTA Benchmark (Task 5)

Open and run `Task_5/notebooks/Task_5_SOTA_Comparison.ipynb` in Jupyter. The notebook loads the SOTA model locally (cached after first download) and runs inference on both the 100-row test set and the full 500-row labeled dataset.

---

## Preprocessing Pipeline Flags

The cleaning pipeline (`cleaning_pipeline.py`) supports fine-grained control via CLI flags:

```bash
python cleaning_pipeline.py --input data.csv --output cleaned.csv \
    --convert_emojis \
    --remove_urls \
    --remove_html_tags \
    --remove_mastodon_artifacts \
    --remove_social_tags \
    --remove_numbers \
    --remove_punctuation \
    --normalize_whitespace \
    --remove_stopwords \
    --fix_spelling \
    --lemmatize \
    --extract_tags \
    --lang_mode drop \
    --no_lowercase
```
