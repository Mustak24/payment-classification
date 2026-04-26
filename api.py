# =============================================================================
# api.py — FastAPI REST server
# Run:  uvicorn api:app --host 0.0.0.0 --port 8000
# =============================================================================

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import config
from predict import get_classifier, PredictionResult

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("api")


# ── Lifespan: load model once at startup ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading classifier on startup...")
    get_classifier()          # warms up the singleton
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SMS Classifier API",
    description="Classifies SMS/message text into CREDIT, DEBIT, UNKNOWN (or custom classes).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Raw message text")
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override confidence threshold (0–1). Default from config.",
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class PredictResponse(BaseModel):
    text:            str
    predicted_class: str
    confidence:      float
    is_uncertain:    bool
    all_scores:      dict[str, float]
    latency_ms:      float


class BatchPredictRequest(BaseModel):
    texts:     list[str] = Field(..., min_length=1, max_length=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("texts list must not be empty")
        for t in v:
            if not t.strip():
                raise ValueError("Each text must be non-empty")
        return v


class BatchPredictResponse(BaseModel):
    results:    list[PredictResponse]
    total:      int
    latency_ms: float


class HealthResponse(BaseModel):
    status:    str
    classes:   list[str]
    vocab_size: int
    trained_at: str
    test_accuracy: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns model metadata. Use for liveness / readiness probes."""
    clf = get_classifier()
    meta = clf.metadata
    return HealthResponse(
        status="ok",
        classes=meta.get("classes", []),
        vocab_size=meta.get("vocabulary_size", 0),
        trained_at=meta.get("trained_at", "unknown"),
        test_accuracy=meta.get("test_accuracy", 0.0),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Classify a single message."""
    clf = get_classifier()
    t0  = time.perf_counter()
    try:
        result: PredictionResult = clf.predict(request.text, threshold=request.threshold)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        **result.to_dict(),
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Classify up to 100 messages in one call."""
    clf = get_classifier()
    t0  = time.perf_counter()
    try:
        results = clf.predict_batch(request.texts, threshold=request.threshold)
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return BatchPredictResponse(
        results=[PredictResponse(**r.to_dict(), latency_ms=0) for r in results],
        total=len(results),
        latency_ms=latency,
    )


@app.get("/model/info", tags=["Monitoring"])
def model_info():
    """Full metadata about the trained model."""
    return get_classifier().metadata
