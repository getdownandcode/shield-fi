"""
generate_data.py
Generates a synthetic transaction dataset mimicking real-world fraud patterns.
- ~200,000 transactions
- ~0.17% fraud rate (matching the resume spec)
- Realistic features: amount, time, merchant category, velocity, etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

N_TOTAL = 200_000
FRAUD_RATE = 0.0017  # 0.17%
N_FRAUD = int(N_TOTAL * FRAUD_RATE)
N_LEGIT = N_TOTAL - N_FRAUD


def generate_dataset(output_path: str = "data/transactions.csv"):
    print(f"Generating {N_TOTAL:,} transactions ({N_FRAUD} fraud, {N_LEGIT:,} legit)...")

    # --- Legitimate transactions ---
    legit = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(N_LEGIT)],
        "timestamp": np.random.randint(0, 30 * 24 * 3600, N_LEGIT),  # seconds in 30 days
        "amount": np.abs(np.random.lognormal(mean=4.0, sigma=1.2, size=N_LEGIT)).clip(0.5, 5000),
        "merchant_category": np.random.choice(
            ["grocery", "restaurant", "gas", "online", "pharmacy", "travel", "retail"],
            p=[0.25, 0.20, 0.15, 0.18, 0.08, 0.07, 0.07],
            size=N_LEGIT
        ),
        "card_present": np.random.choice([1, 0], p=[0.75, 0.25], size=N_LEGIT),
        "country_code": np.random.choice(["US", "CA", "GB", "AU"], p=[0.80, 0.08, 0.07, 0.05], size=N_LEGIT),
        "hour_of_day": np.random.choice(range(24), p=_legit_hour_dist(), size=N_LEGIT),
        "user_id": np.random.randint(1000, 50000, N_LEGIT),
        "is_fraud": 0
    })

    # --- Fraudulent transactions ---
    fraud = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(N_LEGIT, N_TOTAL)],
        "timestamp": np.random.randint(0, 30 * 24 * 3600, N_FRAUD),
        # Fraud tends to be higher-value, unusual amounts
        "amount": np.abs(np.random.lognormal(mean=5.5, sigma=1.5, size=N_FRAUD)).clip(10, 8000),
        "merchant_category": np.random.choice(
            ["online", "travel", "retail", "grocery", "restaurant", "gas", "pharmacy"],
            p=[0.40, 0.20, 0.18, 0.08, 0.05, 0.05, 0.04],
            size=N_FRAUD
        ),
        "card_present": np.random.choice([1, 0], p=[0.20, 0.80], size=N_FRAUD),  # mostly card-not-present
        "country_code": np.random.choice(["US", "CA", "GB", "AU", "XX"], p=[0.40, 0.10, 0.15, 0.10, 0.25], size=N_FRAUD),
        "hour_of_day": np.random.choice(range(24), p=_fraud_hour_dist(), size=N_FRAUD),
        "user_id": np.random.randint(1000, 50000, N_FRAUD),
        "is_fraud": 1
    })

    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path} | Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%")
    return df


def _legit_hour_dist():
    """Legitimate purchases peak during daytime hours."""
    weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 (night)
        0.03, 0.05, 0.07, 0.07, 0.07, 0.07,  # 6-11 (morning)
        0.07, 0.07, 0.07, 0.07, 0.06, 0.06,  # 12-17 (afternoon)
        0.05, 0.05, 0.04, 0.03, 0.02, 0.01   # 18-23 (evening)
    ], dtype=float)
    return weights / weights.sum()


def _fraud_hour_dist():
    """Fraud peaks in early morning hours (2-4am) when users are asleep."""
    weights = np.array([
        0.06, 0.07, 0.10, 0.10, 0.08, 0.05,  # 0-5 (night - fraud peak)
        0.03, 0.03, 0.04, 0.04, 0.04, 0.04,  # 6-11
        0.04, 0.04, 0.04, 0.04, 0.04, 0.04,  # 12-17
        0.04, 0.04, 0.04, 0.04, 0.03, 0.03   # 18-23
    ], dtype=float)
    return weights / weights.sum()


if __name__ == "__main__":
    generate_dataset()
