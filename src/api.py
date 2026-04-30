"""
api.py
Shield-Fi FastAPI microservice — real-time transaction risk scoring.

Endpoints:
  POST /score          — score a single transaction
  POST /score/batch    — score up to 100 transactions
  GET  /health         — liveness check
  GET  /model/info     — model metadata

Run:
  uvicorn api:app --reload --port 8000
"""

import time
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# ── Model registry (loaded once at startup) ───────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "models"

_model = None
_eng   = None
_metrics = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _eng, _metrics
    print("🔄  Loading model artifacts...")
    _model = joblib.load(MODEL_DIR / "xgb_model.joblib")
    _eng   = joblib.load(MODEL_DIR / "feature_engineer.joblib")
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        _metrics = json.loads(metrics_path.read_text())
    print("✅  Shield-Fi ready.")
    yield
    print("🛑  Shutting down.")


app = FastAPI(
    title="Shield-Fi Fraud Detection API",
    description="Real-time transaction risk scoring using XGBoost + SMOTE",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

VALID_CATEGORIES = ["grocery","restaurant","gas","online","pharmacy","travel","retail"]
VALID_COUNTRIES  = ["US","CA","GB","AU","XX"]

class TransactionRequest(BaseModel):
    transaction_id:      str              = Field(..., example="TXN00012345")
    timestamp:           int              = Field(..., ge=0, example=1_700_000_000, description="Unix timestamp")
    amount:              float            = Field(..., gt=0, le=100_000, example=129.99)
    merchant_category:   str              = Field(..., example="online")
    card_present:        int              = Field(..., ge=0, le=1, example=0)
    country_code:        str              = Field(..., example="US")
    hour_of_day:         int              = Field(..., ge=0, le=23, example=3)
    user_id:             int              = Field(..., gt=0, example=42001)

    @field_validator("merchant_category")
    @classmethod
    def validate_merchant(cls, v):
        if v not in VALID_CATEGORIES:
            raise ValueError(f"merchant_category must be one of {VALID_CATEGORIES}")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country(cls, v):
        if v not in VALID_COUNTRIES:
            return "XX"  # treat unknown countries as foreign/high-risk
        return v


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., description="0-1 probability of fraud")
    risk_level:       str    = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    decision:         str    = Field(..., description="APPROVE | REVIEW | DECLINE")
    latency_ms:       float


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    results:    List[ScoreResponse]
    total:      int
    latency_ms: float


# ── Scoring logic ─────────────────────────────────────────────────────────────

THRESHOLDS = {
    "LOW":      (0.00, 0.20),
    "MEDIUM":   (0.20, 0.50),
    "HIGH":     (0.50, 0.80),
    "CRITICAL": (0.80, 1.01),
}

def _risk_level(prob: float) -> str:
    for level, (lo, hi) in THRESHOLDS.items():
        if lo <= prob < hi:
            return level
    return "CRITICAL"

def _decision(risk: str) -> str:
    return {"LOW": "APPROVE", "MEDIUM": "REVIEW",
            "HIGH": "REVIEW", "CRITICAL": "DECLINE"}[risk]

def _score_df(df: pd.DataFrame) -> np.ndarray:
    """Run feature engineering + model inference. Returns fraud probabilities."""
    X = _eng.transform(df)
    return _model.predict_proba(X)[:, 1]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info", tags=["system"])
def model_info():
    return {
        "model_type": "XGBoost + SMOTE",
        "metrics": _metrics,
        "risk_thresholds": THRESHOLDS,
    }


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score_transaction(req: TransactionRequest):
    t0 = time.perf_counter()
    try:
        df   = pd.DataFrame([req.model_dump()])
        prob = float(_score_df(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    risk = _risk_level(prob)
    return ScoreResponse(
        transaction_id    = req.transaction_id,
        fraud_probability = round(prob, 6),
        risk_level        = risk,
        decision          = _decision(risk),
        latency_ms        = round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post("/score/batch", response_model=BatchResponse, tags=["scoring"])
def score_batch(req: BatchRequest):
    t0 = time.perf_counter()
    try:
        df    = pd.DataFrame([t.model_dump() for t in req.transactions])
        probs = _score_df(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {e}")

    results = []
    for txn, prob in zip(req.transactions, probs):
        prob  = float(prob)
        risk  = _risk_level(prob)
        results.append(ScoreResponse(
            transaction_id    = txn.transaction_id,
            fraud_probability = round(prob, 6),
            risk_level        = risk,
            decision          = _decision(risk),
            latency_ms        = 0,   # filled below for batch
        ))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return BatchResponse(results=results, total=len(results), latency_ms=elapsed)
