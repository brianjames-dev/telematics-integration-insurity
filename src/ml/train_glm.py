# src/ml/train_glm.py
from __future__ import annotations

"""
Train baseline GLMs:
  - Frequency: Logistic GLM with exposure offset and ridge regularization
  - Severity : Lognormal regression on positives with ridge regularization

Artifacts written to:
  models/glm_freq.json  (includes standardization stats for frequency)
  models/glm_sev.json   (includes standardization stats for severity)
  models/metrics_glm.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.ml.glm import LogisticGLM, LognormalReg
from src.ml.metrics import roc_auc, decile_lift, rmsle, mae, mape

# ----------------------------
# Feature configuration
# ----------------------------
# Frequency features (used for claim probability)
FEATURES = [
    "night_ratio",
    "p95_over_limit_kph",
    "harsh_brakes_per100km",
    "phone_motion_ratio",
    "rain_flag",
    "mean_speed_kph",
    "crash_grid_index",
]

# Severity adds an explicit exposure term used in the simulator:
#   log_exposure100 = log(exposure_km / 100)
# We'll append this when training severity.
SEV_EXTRA = ["log_exposure100"]


# ----------------------------
# IO helpers
# ----------------------------
def _read_parquet_tree(path_or_glob: str) -> pd.DataFrame:
    base = Path(path_or_glob)
    parts = (
        list(base.rglob("*.parquet")) if base.exists()
        else list(Path().glob(path_or_glob))
    )
    if not parts:
        raise FileNotFoundError(f"No parquet files found under: {path_or_glob}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def load_trip_features(path: str) -> pd.DataFrame:
    return _read_parquet_tree(path)


def load_labels(path: str) -> pd.DataFrame:
    return _read_parquet_tree(path)


def build_offset(feats: pd.DataFrame) -> np.ndarray:
    # exposure offset in log-space (per 100 km)
    return np.log((feats["exposure_km"] / 100.0).clip(lower=1e-6)).to_numpy()


# ----------------------------
# Split (grouped by driver)
# ----------------------------
def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    label_col: str,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Make a test set by selecting whole groups (drivers), preferring to include
    some groups with positives and some without.
    """
    rng = np.random.default_rng(seed)
    grp = (
        df.groupby(group_col)[label_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "pos", "count": "n"})
    )

    pos_groups = grp[grp["pos"] > 0][group_col].to_list()
    neg_groups = grp[grp["pos"] == 0][group_col].to_list()
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    target = int(np.ceil(df.shape[0] * test_frac))
    test_groups: list[str] = []
    total = 0
    i = j = 0
    while total < target and (i < len(pos_groups) or j < len(neg_groups)):
        if i < len(pos_groups):
            g = pos_groups[i]
            i += 1
            if g not in test_groups:
                test_groups.append(g)
                total += int(grp.loc[grp[group_col] == g, "n"].iloc[0])
                if total >= target:
                    break
        if j < len(neg_groups):
            g = neg_groups[j]
            j += 1
            if g not in test_groups:
                test_groups.append(g)
                total += int(grp.loc[grp[group_col] == g, "n"].iloc[0])

    if not test_groups:
        # fallback: random groups until we hit target size
        groups = grp[group_col].to_list()
        rng.shuffle(groups)
        total = 0
        for g in groups:
            test_groups.append(g)
            total += int(grp.loc[grp[group_col] == g, "n"].iloc[0])
            if total >= target:
                break

    mask_test = df[group_col].isin(set(test_groups))
    return ~mask_test, mask_test


# ----------------------------
# Main training routine
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/trip_features", help="Parquet dir or glob for trip_features")
    ap.add_argument("--labels", default="data/labels_trip", help="Parquet dir or glob for labels")
    ap.add_argument("--models", default="models", help="Output directory for model artifacts")
    ap.add_argument("--metrics_out", default="models/metrics_glm.json", help="Path to write metrics JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l2-freq", type=float, default=1.0, help="Ridge strength for frequency GLM")
    ap.add_argument("--l2-sev", type=float, default=1.0, help="Ridge strength for severity regressor")
    args = ap.parse_args()

    Path(args.models).mkdir(parents=True, exist_ok=True)

    # Load and join
    feats = load_trip_features(args.features)
    lbl = load_labels(args.labels)
    df = feats.merge(
        lbl[["driver_id", "trip_id", "claim", "severity"]],
        on=["driver_id", "trip_id"],
        how="inner",
    ).reset_index(drop=True)

    # Grouped split by driver (no leakage)
    mask_train, mask_test = stratified_group_split(
        df, group_col="driver_id", label_col="claim", test_frac=0.2, seed=args.seed
    )
    tr = df[mask_train].reset_index(drop=True).copy()
    te = df[mask_test].reset_index(drop=True).copy()

    # ----------------------------
    # Frequency model (Logistic GLM)
    # ----------------------------
    # Z-score standardization on TRAIN ONLY
    mu = tr[FEATURES].mean()
    sigma = tr[FEATURES].std().replace(0, 1.0)

    X_tr_raw = tr[FEATURES].to_numpy(float)
    X_te_raw = te[FEATURES].to_numpy(float)
    X_tr = (X_tr_raw - mu.values) / sigma.values
    X_te = (X_te_raw - mu.values) / sigma.values

    off_tr = build_offset(tr)
    off_te = build_offset(te)

    y_tr = tr["claim"].to_numpy(float)
    y_te = te["claim"].to_numpy(float)

    logit_glm = LogisticGLM.fit(
        X_tr, y_tr, off_tr, FEATURES,
        l2=args.l2_freq, max_iter=200, tol=1e-6
    )
    p_tr = np.clip(logit_glm.predict_prob(X_tr, off_tr), 1e-6, 1 - 1e-6)
    p_te = np.clip(logit_glm.predict_prob(X_te, off_te), 1e-6, 1 - 1e-6)

    # Metrics (frequency)
    try:
        auc = roc_auc(y_te.astype(int), p_te)
    except Exception:
        auc = 0.5
    lift = decile_lift(y_te.astype(int), p_te)

    # Save frequency model with scaler
    freq_obj = logit_glm.to_json()
    freq_obj["standardize"] = {
        "mu": mu[FEATURES].tolist(),
        "sigma": sigma[FEATURES].tolist(),
        "features": FEATURES,
    }
    with open(Path(args.models) / "glm_freq.json", "w") as f:
        json.dump(freq_obj, f, indent=2)

    # ----------------------------
    # Severity model (Lognormal on positives)
    # ----------------------------
    # Add severity-specific feature used by the simulator
    tr.loc[:, "log_exposure100"] = np.log((tr["exposure_km"] / 100.0).clip(lower=1e-6))
    te.loc[:, "log_exposure100"] = np.log((te["exposure_km"] / 100.0).clip(lower=1e-6))
    SEV_FEATURES = FEATURES + SEV_EXTRA

    pos_tr = tr["claim"] == 1
    if pos_tr.sum() >= 5:
        Xs_raw = tr.loc[pos_tr, SEV_FEATURES].to_numpy(float)
        ys = tr.loc[pos_tr, "severity"].to_numpy(float)

        # Standardize on TRAIN POSITIVES only (severity is conditional on claim)
        mu_sev = Xs_raw.mean(axis=0)
        sigma_sev = Xs_raw.std(axis=0)
        sigma_sev[sigma_sev == 0] = 1.0
        Xs = (Xs_raw - mu_sev) / sigma_sev

        sev = LognormalReg.fit(Xs, ys, SEV_FEATURES, l2=args.l2_sev)

        # Evaluate on test positives if they exist
        pos_te = te["claim"] == 1
        if pos_te.sum() >= 3:
            Xt_raw = te.loc[pos_te, SEV_FEATURES].to_numpy(float)
            Xt = (Xt_raw - mu_sev) / sigma_sev
            y_true_sev = te.loc[pos_te, "severity"].to_numpy(float)
            y_pred_sev = sev.predict_mean(Xt, max_mu_clip=6.0)
            sev_rmsle = rmsle(y_true_sev, y_pred_sev)
            sev_mae = mae(y_true_sev, y_pred_sev)
            sev_mape = mape(y_true_sev, y_pred_sev)
        else:
            sev_rmsle = sev_mae = sev_mape = float("nan")
    else:
        # Not enough positives to fit a meaningful model
        sev = LognormalReg(feature_names=["intercept"] + SEV_FEATURES,
                           beta=np.zeros(len(SEV_FEATURES) + 1), sigma2=0.0)
        mu_sev = np.zeros(len(SEV_FEATURES))
        sigma_sev = np.ones(len(SEV_FEATURES))
        sev_rmsle = sev_mae = sev_mape = float("nan")

    # Save severity model with its own scaler
    sev_obj = sev.to_json()
    sev_obj["standardize"] = {
        "mu": mu_sev.tolist(),
        "sigma": sigma_sev.tolist(),
        "features": SEV_FEATURES,
    }
    with open(Path(args.models) / "glm_sev.json", "w") as f:
        json.dump(sev_obj, f, indent=2)

    # ----------------------------
    # Persist metrics + console summary
    # ----------------------------
    metrics = {
        "freq_auc": float(auc),
        "freq_base_rate": float(y_te.mean()) if len(y_te) else float("nan"),
        "freq_decile_lift": lift,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_pos_train": int(y_tr.sum()),
        "n_pos_test": int(y_te.sum()),
        "sev_rmsle": float(sev_rmsle),
        "sev_mae": float(sev_mae),
        "sev_mape": float(sev_mape),
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[PH3-TRAIN] AUC={metrics['freq_auc']:.3f} base={metrics['freq_base_rate']:.4f} pos_te={metrics['n_pos_test']}")
    print(f"[PH3-TRAIN] sev_rmsle={metrics['sev_rmsle']} mae={metrics['sev_mae']} mape={metrics['sev_mape']}")


if __name__ == "__main__":
    main()
