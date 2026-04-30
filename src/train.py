"""
train.py
Trains the Shield-Fi XGBoost fraud detection model.

Pipeline:
  1. Load raw data
  2. Feature engineering
  3. Train/test split (stratified)
  4. SMOTE oversampling on training set only
  5. XGBoost training (CPU, no GPU needed)
  6. Evaluation: PR-AUC, ROC-AUC, classification report
  7. Save model + feature engineer to disk
"""

import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    classification_report, confusion_matrix,
    precision_recall_curve, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from features import FraudFeatureEngineer, get_feature_names

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/transactions.csv"
MODEL_DIR   = Path("models")
SEED        = 42
TEST_SIZE   = 0.20
SMOTE_K     = 5           # SMOTE neighbours (safe for small minority class)

XGB_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 1,   # Will be overridden after SMOTE balances classes
    "eval_metric":      "aucpr",
    "use_label_encoder": False,
    "tree_method":      "hist",   # CPU-optimised histogram method
    "device":           "cpu",
    "random_state":     SEED,
    "n_jobs":           -1,       # Use all CPU cores
    "verbosity":        1,
}


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape} | Fraud rate: {df['is_fraud'].mean()*100:.3f}%")

    X_raw = df.drop(columns=["is_fraud", "transaction_id"])
    y     = df["is_fraud"].values

    # ── 2. Feature engineering ────────────────────────────────────────────────
    print("\n[2/6] Engineering features (velocity + time-decay)...")
    eng = FraudFeatureEngineer()
    eng.fit(X_raw, y)
    X = eng.transform(X_raw)
    feature_names = get_feature_names()
    print(f"  Feature matrix: {X.shape}")

    # ── 3. Train / test split ─────────────────────────────────────────────────
    print("\n[3/6] Splitting train/test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"  Train: {X_train.shape} | Fraud in train: {y_train.sum()}")
    print(f"  Test : {X_test.shape}  | Fraud in test : {y_test.sum()}")

    # ── 4. SMOTE ──────────────────────────────────────────────────────────────
    print("\n[4/6] Applying SMOTE to training set...")
    smote = SMOTE(k_neighbors=SMOTE_K, random_state=SEED, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Resampled shape: {X_res.shape} | Fraud: {y_res.sum()} ({y_res.mean()*100:.1f}%)")

    # ── 5. Train XGBoost ──────────────────────────────────────────────────────
    print("\n[5/6] Training XGBoost (CPU)...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_res, y_res,
        eval_set=[(X_test, y_test)],
        verbose=50,
        early_stopping_rounds=30,
    )
    print(f"  Best iteration: {model.best_iteration}")

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    print("\n[6/6] Evaluating model...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc  = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\n  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    metrics = {
        "pr_auc":         round(pr_auc,  4),
        "roc_auc":        round(roc_auc, 4),
        "best_iteration": int(model.best_iteration),
        "n_train":        int(X_res.shape[0]),
        "n_test":         int(X_test.shape[0]),
        "training_time_s": round(time.time() - t0, 1),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_pr_curve(y_test, y_prob, pr_auc)
    _plot_confusion_matrix(y_test, y_pred)
    _plot_feature_importance(model, feature_names)

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(model, MODEL_DIR / "xgb_model.joblib")
    joblib.dump(eng,   MODEL_DIR / "feature_engineer.joblib")
    print(f"\n✅ Model saved to {MODEL_DIR}/")
    print(f"   Total time: {time.time()-t0:.1f}s")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_pr_curve(y_true, y_prob, pr_auc):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, color="#e74c3c", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Shield-Fi")
    plt.legend(); plt.tight_layout()
    plt.savefig(MODEL_DIR / "pr_curve.png", dpi=150)
    plt.close()
    print("  Saved pr_curve.png")


def _plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"])
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.title("Confusion Matrix"); plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("  Saved confusion_matrix.png")


def _plot_feature_importance(model, feature_names):
    scores = model.feature_importances_
    fi = pd.Series(scores, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    fi.head(15).plot(kind="bar", color="#2980b9")
    plt.title("Top-15 Feature Importances (gain)")
    plt.ylabel("Importance"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print("  Saved feature_importance.png")


if __name__ == "__main__":
    main()
