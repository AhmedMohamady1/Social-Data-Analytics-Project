"""
FastAPI sentiment prediction API.

Endpoints
─────────
POST /predict   → { "text": "..." }  →  { "sentiment": "...", "confidence": 0.94 }
GET  /health    → health check
GET  /info      → model metadata
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

# Support running app via uvicorn from the root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.style_c_glove_transform import text_to_style_c_glove_feature_row
from src.config import MODELS_DIR

# ── Paths ────────────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "random_forest_glove_style_c_p50.pkl"

# ── Load model artifacts at startup ──────────────────────────
def _load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}."
        )
    model_obj = joblib.load(MODEL_PATH)
    
    if isinstance(model_obj, dict) and "model" in model_obj:
        model = model_obj["model"]
    else:
        # The saved .pkl might actually be just the LabelEncoder due to a bug in earlier iterations
        model = model_obj if not hasattr(model_obj, "classes_") else None
    
    # Classes are sorted alphabetically by scikit-learn LabelEncoder
    classes = ["negative", "neutral", "positive"]
    
    return model, classes

model, classes = _load_artifacts()

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description="Predict sentiment (positive / negative / neutral) from text input.",
    version="1.0.0",
)

import threading

is_warming_up = True

@app.on_event("startup")
def warmup_models():
    """Run a dummy prediction in the background so the server doesn't block."""
    def warmup_task():
        global is_warming_up
        print("Warming up NLP caching...")
        try:
            # Use a full English sentence so langdetect doesn't drop it
            text_to_style_c_glove_feature_row("This is a proper English sentence to warm up the NLP models quickly and safely.", python_executable=sys.executable)
            print("Warmup complete!")
        except Exception as e:
            print(f"Warmup failed: {e}")
        finally:
            is_warming_up = False

    t = threading.Thread(target=warmup_task)
    t.start()
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

        try:
            processed_text, feature_row, info_meta = text_to_style_c_glove_feature_row(text, python_executable=sys.executable)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

        if model is None:
            raise HTTPException(status_code=500, detail="The loaded model is actually a LabelEncoder. Please fix the saving logic in Random_Forest.py.")

        proba = model.predict_proba(feature_row)[0]
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


@app.get("/status")
def status():
    """Detailed status indicating if background warmup is finished."""
    return {"status": "warming_up" if is_warming_up else "ready"}


@app.get("/info", response_model=InfoResponse)
def info():
    """Return model metadata."""
    return InfoResponse(
        model_type=type(model).__name__,
        classes=[str(c) for c in classes],
        vectorizer_features=model.n_features_in_ if hasattr(model, "n_features_in_") else 0,
    )
