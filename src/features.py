"""
features.py
Feature engineering pipeline for Shield-Fi fraud detection.

Key engineered features:
  - Time-decay: recency-weighted transaction history
  - Transaction velocity: count/sum of transactions in rolling windows
  - Behavioural: deviation from user's typical spending
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── Constants ──────────────────────────────────────────────────────────────────
VELOCITY_WINDOWS = [300, 3600, 86400]   # 5 min, 1 hour, 1 day (in seconds)
DECAY_HALFLIFE   = 3600                  # 1 hour half-life for time-decay feature
CATEGORY_ORDER   = ["grocery","restaurant","gas","online","pharmacy","travel","retail"]
COUNTRY_ORDER    = ["US","CA","GB","AU","XX"]


# ── Main feature pipeline ──────────────────────────────────────────────────────
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw transaction DataFrame into a model-ready feature matrix.
    Safe to use inside sklearn Pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Compute per-user stats from training data for behavioural features
        self.user_stats_ = (
            X.groupby("user_id")["amount"]
            .agg(user_mean_amount="mean", user_std_amount="std")
            .fillna(0)
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = X.copy()
        df = self._base_features(df)
        df = self._velocity_features(df)
        df = self._time_decay_features(df)
        df = self._behavioural_features(df)
        return df[self._feature_cols()].values

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns and extract time components."""
        # Merchant category ordinal
        df["merchant_cat_enc"] = pd.Categorical(
            df["merchant_category"], categories=CATEGORY_ORDER
        ).codes.astype(np.int8)

        # Country risk flag (XX = unknown/foreign = higher risk)
        df["country_risk"] = (df["country_code"] == "XX").astype(np.int8)
        df["country_enc"]  = pd.Categorical(
            df["country_code"], categories=COUNTRY_ORDER
        ).codes.astype(np.int8)

        # Time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["is_night"] = df["hour_of_day"].between(0, 5).astype(np.int8)

        # Amount features
        df["log_amount"] = np.log1p(df["amount"])

        return df

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-user transaction velocity over rolling time windows.
        Velocity = number of transactions by the same user in the last W seconds.
        """
        df = df.sort_values("timestamp")

        for w in VELOCITY_WINDOWS:
            col_cnt = f"velocity_count_{w}s"
            col_sum = f"velocity_sum_{w}s"
            counts, sums = [], []

            for _, grp in df.groupby("user_id"):
                ts  = grp["timestamp"].values
                amt = grp["amount"].values
                cnt_arr = np.zeros(len(ts), dtype=np.int32)
                sum_arr = np.zeros(len(ts), dtype=np.float32)
                left = 0
                run_sum = 0.0
                for right in range(len(ts)):
                    while ts[right] - ts[left] > w:
                        run_sum -= amt[left]
                        left += 1
                    cnt_arr[right] = right - left        # exclude self
                    sum_arr[right] = run_sum
                    run_sum += amt[right]
                counts.append(pd.Series(cnt_arr, index=grp.index))
                sums.append(pd.Series(sum_arr,  index=grp.index))

            df[col_cnt] = pd.concat(counts)
            df[col_sum] = pd.concat(sums)

        return df

    def _time_decay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-decay weighted amount: more recent transactions count more.
        decay_weight = sum of exp(-Δt / half_life) for prior txns by user.
        """
        df = df.sort_values("timestamp")
        decay_col = []

        for _, grp in df.groupby("user_id"):
            ts  = grp["timestamp"].values
            amt = grp["amount"].values
            decay = np.zeros(len(ts), dtype=np.float32)
            for i in range(1, len(ts)):
                dt = ts[i] - ts[:i]
                weights = np.exp(-dt / DECAY_HALFLIFE)
                decay[i] = float(np.dot(weights, amt[:i]))
            decay_col.append(pd.Series(decay, index=grp.index))

        df["time_decay_amount"] = pd.concat(decay_col)
        return df

    def _behavioural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deviation of current amount from user's historical mean."""
        df = df.join(self.user_stats_, on="user_id", how="left")
        df["user_mean_amount"] = df["user_mean_amount"].fillna(df["amount"].mean())
        df["user_std_amount"]  = df["user_std_amount"].fillna(df["amount"].std()).clip(lower=1e-6)
        df["amount_z_score"]   = (df["amount"] - df["user_mean_amount"]) / df["user_std_amount"]
        return df

    def _feature_cols(self):
        base = [
            "log_amount", "amount_z_score",
            "merchant_cat_enc", "country_enc", "country_risk",
            "card_present",
            "hour_sin", "hour_cos", "is_night",
        ]
        velocity = [
            f"{prefix}_{w}s"
            for prefix in ["velocity_count", "velocity_sum"]
            for w in VELOCITY_WINDOWS
        ]
        decay = ["time_decay_amount"]
        return base + velocity + decay

    @property
    def feature_names(self):
        return self._feature_cols()


def get_feature_names() -> list:
    """Return feature names in the same order as FraudFeatureEngineer.transform()."""
    eng = FraudFeatureEngineer()
    return eng._feature_cols()
