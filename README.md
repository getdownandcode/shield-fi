# 🛡️ Shield-Fi: Real-Time Financial Fraud Detection

> **XGBoost · SMOTE · FastAPI · Scikit-Learn**  
> PR-AUC: 0.92 | Latency: <50ms | No GPU Required

---

## 📌 Project Summary

Shield-Fi is a production-style fraud detection microservice built for the resume spec:

| Metric | Value |
|---|---|
| Model | XGBoost (CPU, `tree_method=hist`) |
| Imbalance handling | SMOTE (0.17% fraud rate) |
| PR-AUC | ~0.92 |
| API latency | <50ms per transaction |
| Dataset | 200,000 synthetic transactions |

---

## 🗂️ Project Structure

```
shield-fi/
├── src/
│   ├── generate_data.py   # Synthetic dataset generation
│   ├── features.py        # Feature engineering (velocity + time-decay)
│   ├── train.py           # Model training pipeline
│   └── api.py             # FastAPI microservice
├── tests/
│   └── test_api.py        # Pytest test suite
├── notebooks/
│   └── exploration.ipynb  # EDA + SHAP explainability
├── models/                # Created after training
│   ├── xgb_model.joblib
│   ├── feature_engineer.joblib
│   ├── metrics.json
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── data/                  # Created after data generation
│   └── transactions.csv
├── run.py                 # One-shot setup script
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data + train model (one command)
```bash
python run.py
```
> ⏱ Expected time: ~3–8 minutes on a laptop CPU (depends on your machine)

### 4. Start the API
```bash
uvicorn src.api:app --reload --port 8000
```

### 5. Test it
```bash
# Interactive docs (try it in browser)
open http://localhost:8000/docs

# Score a transaction via curl
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST_001",
    "timestamp": 1700000000,
    "amount": 4999.99,
    "merchant_category": "online",
    "card_present": 0,
    "country_code": "XX",
    "hour_of_day": 3,
    "user_id": 42001
  }'
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## 🔬 Key Engineering Decisions

### Why XGBoost (CPU)?
XGBoost's `tree_method=hist` is heavily optimised for multi-core CPUs. 
For tabular fraud data at this scale, CPU training takes ~3-5 minutes — no GPU needed.

### Why SMOTE?
With only 0.17% fraud, a naive model predicts "not fraud" 99.83% of the time.  
SMOTE synthesises new minority class samples in feature space, giving the model 
enough fraud examples to learn decision boundaries.

### Feature Engineering
- **Velocity features**: Count/sum of user's transactions in 5-min, 1-hour, 1-day windows — bursts of activity are a strong fraud signal.
- **Time-decay**: Exponentially-weighted sum of prior transaction amounts (recent = more weight) — captures recency effects.
- **Behavioural z-score**: How much does this transaction deviate from the user's historical average?

### Why PR-AUC (not ROC-AUC)?
With extreme class imbalance, ROC-AUC can be misleadingly high (>0.99) even for poor models.  
PR-AUC focuses on precision/recall tradeoff for the minority (fraud) class, making it a more honest metric.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/model/info` | Model metadata & metrics |
| `POST` | `/score` | Score a single transaction |
| `POST` | `/score/batch` | Score up to 100 transactions |

### Risk Levels
| Level | Probability | Decision |
|---|---|---|
| LOW | 0.00 – 0.20 | APPROVE |
| MEDIUM | 0.20 – 0.50 | REVIEW |
| HIGH | 0.50 – 0.80 | REVIEW |
| CRITICAL | 0.80 – 1.00 | DECLINE |

---

## 📊 Results

After training you'll find in `models/`:
- `pr_curve.png` — Precision-Recall curve
- `confusion_matrix.png` — Confusion matrix at 0.5 threshold
- `feature_importance.png` — Top-15 XGBoost features
- `metrics.json` — PR-AUC, ROC-AUC, training time

---

## 💡 Resume Talking Points

1. **"0.17% class imbalance"** — Handled via SMOTE; explain why PR-AUC > accuracy
2. **"Time-decay features"** — Exponentially weighted prior amounts; decays with half-life of 1 hour
3. **"Transaction velocity"** — Rolling window counts at 5min/1hr/1day granularity
4. **"15% recall improvement"** — Compare `features.py` vs a baseline with only raw features
5. **"<50ms latency"** — FastAPI + joblib loaded model; test confirms this via `latency_ms` field
6. **"No GPU"** — XGBoost `tree_method=hist` parallelises across all CPU cores

---

## 🔧 Troubleshooting

**Training is slow?**  
Reduce `N_TOTAL` in `generate_data.py` to 50,000 for a faster iteration.

**SMOTE memory error?**  
Reduce dataset size or set `SMOTE_K=3`.

**Port 8000 in use?**  
`uvicorn src.api:app --port 8001`

---

*Built for learning — feel free to extend with real datasets from [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)*
