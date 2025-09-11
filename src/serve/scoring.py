# src/serve/scoring.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
TRIP_FEATURES_DIR = Path("data/trip_features")

# -------------------
# GBM Frequency Model
# -------------------
class GBMFreq:
    def __init__(self, models_dir: Path = MODELS_DIR):
        meta = json.loads((models_dir / "gbm_meta.json").read_text())
        self.features = meta.get("features") or meta.get("feature_names")
        self.model = joblib.load(models_dir / "gbm_freq.pkl")
        cal_path = models_dir / "gbm_cal.pkl"
        self.cal = joblib.load(cal_path) if cal_path.exists() else None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.cal is not None:
            return self.cal.predict_proba(X)[:, 1]
        return self.model.predict_proba(X)[:, 1]

# -------------------
# GLM Severity Model
# -------------------
class GLMSeverity:
    def __init__(self, models_dir: Path = MODELS_DIR):
        obj = json.loads((models_dir / "glm_sev.json").read_text())
        self.beta = np.array(obj["beta"], dtype=float)  # includes intercept at [0]
        self.sigma2 = float(obj["sigma2"])
        std = obj["standardize"]
        self.features = std["features"]
        self.mu = np.array(std["mu"], dtype=float)
        self.sigma = np.array(std["sigma"], dtype=float)

    def expected_severity(self, df: pd.DataFrame) -> np.ndarray:
        # ensure required derived feature for sev training is present
        if "log_exposure100" in self.features and "log_exposure100" not in df.columns:
            df = df.copy()
            df["log_exposure100"] = np.log((df["exposure_km"] / 100.0).clip(lower=1e-6))

        X = df[self.features].to_numpy(float)
        Xs = (X - self.mu) / self.sigma
        X1 = np.hstack([np.ones((Xs.shape[0], 1)), Xs])  # add intercept
        mu_hat = np.clip(X1 @ self.beta, -2.0, 6.0)
        # mean of lognormal = exp(mu + 0.5*sigma^2)
        return np.exp(mu_hat + 0.5 * self.sigma2)

# -------------------
# Scoring utilities
# -------------------
class Scorer:
    def __init__(self):
        self.freq = GBMFreq()
        self.sev = GLMSeverity()
        self.freq_features = self.freq.features
        self.sev_features = self.sev.features

    def _ensure_log_exposure(self, df: pd.DataFrame) -> pd.DataFrame:
        if "log_exposure100" in self.freq_features and "log_exposure100" not in df.columns:
            df = df.copy()
            df["log_exposure100"] = np.log((df["exposure_km"] / 100.0).clip(lower=1e-6))
        return df

    def score_trip_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_log_exposure(df)
        Xf = df[self.freq_features].to_numpy(float)
        p = self.freq.predict_proba(Xf)
        sev = self.sev.expected_severity(df)
        # expected cost per 100km for each row:
        exp_cost_per100 = (p * sev) / df["exposure_km"].clip(lower=1e-6).to_numpy() * 100.0
        out = df.copy()
        out["p_claim"] = p
        out["sev_mean"] = sev
        out["expected_cost_per100km"] = exp_cost_per100
        return out

    def score_one_trip(self, row: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([row])
        scored = self.score_trip_df(df).iloc[0]
        return {
            "p_claim": float(scored["p_claim"]),
            "sev_mean": float(scored["sev_mean"]),
            "expected_cost_per100km": float(scored["expected_cost_per100km"]),
        }

    def aggregate_driver(self, driver_id: str, window_days: int = 30) -> Dict[str, Any]:
        parts = list(TRIP_FEATURES_DIR.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError("No trip_features parquet found. Run Phase 2 first.")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        # filter by driver + time window
        df = df[df["driver_id"] == driver_id].copy()
        if "start_ts" in df.columns:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=window_days)
            df = df[df["start_ts"] >= cutoff]
        if df.empty:
            return {"driver_id": driver_id, "n_trips": 0, "exposure_km": 0.0,
                    "p_claim_weighted": 0.0, "sev_mean_weighted": 0.0,
                    "expected_cost_per100km": 0.0}

        scored = self.score_trip_df(df)
        w = scored["exposure_km"].clip(lower=1e-6).to_numpy()
        wsum = float(w.sum())
        p_w = float(np.average(scored["p_claim"].to_numpy(), weights=w))
        sev_w = float(np.average(scored["sev_mean"].to_numpy(), weights=w))
        # aggregate expected cost per 100km across trips (exposure-weighted)
        cost_per100_w = float(np.average(scored["expected_cost_per100km"].to_numpy(), weights=w))

        return {
            "driver_id": driver_id,
            "n_trips": int(len(scored)),
            "exposure_km": float(wsum),
            "p_claim_weighted": p_w,
            "sev_mean_weighted": sev_w,
            "expected_cost_per100km": cost_per100_w,
        }

# singleton scorer for API usage
SCORER: Optional[Scorer] = None

def get_scorer() -> Scorer:
    global SCORER
    if SCORER is None:
        SCORER = Scorer()
    return SCORER
