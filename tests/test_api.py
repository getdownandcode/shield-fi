"""
test_api.py
Pytest tests for Shield-Fi API and feature engineering.

Run: pytest tests/ -v
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_transaction():
    return {
        "transaction_id":    "TXN_TEST_001",
        "timestamp":         1_700_100_000,
        "amount":            250.0,
        "merchant_category": "online",
        "card_present":      0,
        "country_code":      "XX",
        "hour_of_day":       3,
        "user_id":           99999,
    }

@pytest.fixture(scope="session")
def legit_transaction():
    return {
        "transaction_id":    "TXN_TEST_002",
        "timestamp":         1_700_200_000,
        "amount":            12.50,
        "merchant_category": "grocery",
        "card_present":      1,
        "country_code":      "US",
        "hour_of_day":       14,
        "user_id":           88888,
    }


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFraudFeatureEngineer:
    def setup_method(self):
        from features import FraudFeatureEngineer
        self.eng = FraudFeatureEngineer()
        self.df_train = pd.DataFrame({
            "transaction_id":    [f"T{i}" for i in range(20)],
            "timestamp":         list(range(0, 20 * 3600, 3600)),
            "amount":            [50.0 + i * 10 for i in range(20)],
            "merchant_category": (["grocery"] * 10 + ["online"] * 10),
            "card_present":      [1] * 20,
            "country_code":      ["US"] * 20,
            "hour_of_day":       [12] * 20,
            "user_id":           [1001] * 10 + [1002] * 10,
            "is_fraud":          [0] * 20,
        })
        self.df_train = self.df_train.drop(columns=["is_fraud", "transaction_id"])

    def test_fit_transform_shape(self):
        self.eng.fit(self.df_train)
        X = self.eng.transform(self.df_train)
        assert X.ndim == 2
        assert X.shape[0] == 20
        assert X.shape[1] > 10, "Expected at least 10 features"

    def test_no_nan_in_output(self):
        self.eng.fit(self.df_train)
        X = self.eng.transform(self.df_train)
        assert not np.isnan(X).any(), "Feature matrix contains NaN values"

    def test_feature_count_matches_names(self):
        from features import get_feature_names
        self.eng.fit(self.df_train)
        X = self.eng.transform(self.df_train)
        assert X.shape[1] == len(get_feature_names())

    def test_velocity_increases_with_more_txns(self):
        """User with more transactions should have higher velocity count."""
        self.eng.fit(self.df_train)
        X = self.eng.transform(self.df_train)
        # velocity_count_86400s is index 9 in the feature list
        from features import get_feature_names
        names = get_feature_names()
        vel_idx = names.index("velocity_count_86400s")
        # Last txn per user should have higher count than first
        assert X[-1, vel_idx] > X[10, vel_idx]  # user 1002: last > first


# ── API tests ─────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent.parent / "models"

@pytest.fixture(scope="session")
def api_client():
    """Only run API tests if model is trained."""
    if not (MODEL_DIR / "xgb_model.joblib").exists():
        pytest.skip("Model not trained yet. Run: python src/train.py")
    from api import app
    with TestClient(app) as client:
        yield client


class TestAPI:
    def test_health(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_model_info(self, api_client):
        r = api_client.get("/model/info")
        assert r.status_code == 200
        data = r.json()
        assert "metrics" in data
        assert "risk_thresholds" in data

    def test_score_returns_valid_response(self, api_client, sample_transaction):
        r = api_client.post("/score", json=sample_transaction)
        assert r.status_code == 200
        data = r.json()
        assert "fraud_probability" in data
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert data["decision"] in {"APPROVE", "REVIEW", "DECLINE"}

    def test_score_latency_under_50ms(self, api_client, sample_transaction):
        r = api_client.post("/score", json=sample_transaction)
        assert r.json()["latency_ms"] < 50, "Latency exceeded 50ms SLA"

    def test_invalid_merchant_category(self, api_client, sample_transaction):
        bad = {**sample_transaction, "merchant_category": "casino"}
        r = api_client.post("/score", json=bad)
        assert r.status_code == 422

    def test_batch_score(self, api_client, sample_transaction, legit_transaction):
        payload = {"transactions": [sample_transaction, legit_transaction]}
        r = api_client.post("/score/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_high_risk_transaction_profile(self, api_client):
        """Night-time, foreign, card-not-present, high-amount → should be higher prob."""
        high_risk = {
            "transaction_id": "TXN_HIGH_RISK",
            "timestamp": 1_700_000_100,
            "amount": 4999.99,
            "merchant_category": "online",
            "card_present": 0,
            "country_code": "XX",
            "hour_of_day": 3,
            "user_id": 77777,
        }
        low_risk = {
            "transaction_id": "TXN_LOW_RISK",
            "timestamp": 1_700_000_200,
            "amount": 9.99,
            "merchant_category": "grocery",
            "card_present": 1,
            "country_code": "US",
            "hour_of_day": 14,
            "user_id": 66666,
        }
        r_high = api_client.post("/score", json=high_risk).json()
        r_low  = api_client.post("/score", json=low_risk).json()
        assert r_high["fraud_probability"] > r_low["fraud_probability"]
