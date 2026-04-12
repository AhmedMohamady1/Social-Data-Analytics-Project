"""
FastAPI sentiment prediction API.

Endpoints
─────────
POST /predict   → { "text": "..." }  →  { "sentiment": "...", "confidence": 0.94 }
GET  /health    → health check
GET  /info      → model metadata
"""

import os
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
VECTORIZER_PATH = ARTIFACT_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACT_DIR / "sentiment_model.joblib"
LABEL_CLASSES_PATH = ARTIFACT_DIR / "label_classes.joblib"

# ── Load model artifacts at startup ──────────────────────────
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    classes = joblib.load(LABEL_CLASSES_PATH)
    return vectorizer, model, classes

vectorizer, model, classes = _load_artifacts()

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive / negative / neutral) from text input.",
    version="1.0.0",
)


# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["I love this product"])


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str


class InfoResponse(BaseModel):
    model_type: str
    classes: list[str]
    vectorizer_features: int


# ── Endpoints ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment for the given text."""
    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty.")

        X = vectorizer.transform([text])
        proba = model.predict_proba(X)[0]
        predicted_idx = int(np.argmax(proba))
        predicted_label = classes[predicted_idx]
        confidence = float(round(proba[predicted_idx], 4))

        probabilities = {
            str(cls): float(round(p, 4)) for cls, p in zip(classes, proba)
        }

        return PredictResponse(
            sentiment=predicted_label,
            confidence=confidence,
            probabilities=probabilities,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse)
def info():
    """Return model metadata."""
    return InfoResponse(
        model_type=type(model).__name__,
        classes=[str(c) for c in classes],
        vectorizer_features=len(vectorizer.get_feature_names_out()),
    )
