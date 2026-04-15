# Task 4: Model Deployment

This module deploys the Sentiment Analysis pipeline using a Fast-API backend service coupled with a Streamlit front-end dashboard. The underlying prediction model leverages a Random Forest architecture trained on GloVe Style C embeddings, configured specifically for the Iran-War social dataset.

## Directory Structure

```text
Task_4/
├── app.py                   # Streamlit web dashboard
├── api/
│   └── api.py               # FastAPI backend with startup NLP caching
├── src/
│   ├── config.py            # Centralized environmental path constants
│   ├── features/            # Transformation and splitting scripts
│   └── models/              # Model training scripts and hyperparameter tuners
├── models/                  # Serialized .pkl binary pipeline artifacts
├── data/                    # Processed CSVs and cache files
└── reports/                 # Captured visuals, metrics, and CSV results
```

## Running the Application

To interact with the sentiment analyzer, you need to boot both the API (to serve predictions) and the Streamlit app (to visualize them).

> Make sure your virtual environment is activated and you are actively inside the `Task_4` directory before executing the commands.

**1. Start the FastAPI Service:**
```bash
uvicorn api.api:app --reload
```
*(Available at: http://127.0.0.1:8000. It will spend approx ~15 seconds executing a warmup thread before validating inputs!)*

**2. Start the Streamlit Web Application:**
```bash
streamlit run app.py
```
*(Available at: http://localhost:8501. The UI will automatically buffer utilizing a loading state until the backend is fully mounted!)*
