# src/ml/train_gbm.py
import argparse
import json
import math
import warnings
import inspect
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


# ----------------------------
# Feature config / monotonicity
# ----------------------------
FEATS = [
    "night_ratio",
    "p95_over_limit_kph",
    "harsh_brakes_per100km",
    "phone_motion_ratio",
    "rain_flag",
    "mean_speed_kph",
    "crash_grid_index",
    "log_exposure100",  # exposure-based offset feature (monotone +)
]

# Domain-driven monotonic constraints: risk ↑ with each of these
MONO_MAP = {
    "night_ratio": +1,
    "p95_over_limit_kph": +1,
    "harsh_brakes_per100km": +1,
    "phone_motion_ratio": +1,
    "rain_flag": +1,
    "mean_speed_kph": +1,
    "crash_grid_index": +1,
    "log_exposure100": +1,
}


def _read_parquet_dir(path: str) -> pd.DataFrame:
    parts = list(Path(path).rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files under: {path}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def build_dataset(features_dir: str, labels_dir: str) -> pd.DataFrame:
    feats = _read_parquet_dir(features_dir)
    labels = _read_parquet_dir(labels_dir)

    # Required columns check (will raise clear message if something's missing)
    needed = {
        "driver_id", "trip_id", "exposure_km",
        "night_ratio", "p95_over_limit_kph", "harsh_brakes_per100km",
        "phone_motion_ratio", "rain_flag", "mean_speed_kph", "crash_grid_index",
    }
    missing = [c for c in needed if c not in feats.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    if "claim" not in labels.columns:
        raise KeyError("Labels table is missing 'claim' column")

    df = feats.merge(
        labels[["driver_id", "trip_id", "claim"]],
        on=["driver_id", "trip_id"],
        how="inner",
        validate="one_to_one",
    )

    # Derive exposure log feature (used as a weak offset-like input)
    df["log_exposure100"] = np.log((df["exposure_km"] / 100.0).clip(lower=1e-6))

    # Clean up NaNs/Infs
    for c in FEATS:
        if c not in df.columns:
            raise KeyError(f"Expected feature not found after merge: {c}")

    # rain_flag: set missing to 0 (no rain)
    if "rain_flag" in df.columns:
        df["rain_flag"] = df["rain_flag"].fillna(0)

    # ensure numeric dtypes
    for c in FEATS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATS + ["claim"]).copy()

    # Sample weights: proportional to exposure (safe minimum)
    df["weight"] = (df["exposure_km"] / 100.0).clip(lower=1e-3)

    # y
    df["claim"] = df["claim"].astype(int)
    return df


def make_monotone_vector() -> List[int]:
    return [int(MONO_MAP.get(f, 0)) for f in FEATS]


def safe_split(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Try to create a train/cal split with both classes in cal where possible.
    Fallbacks:
      - different test_size proportions,
      - StratifiedKFold pick the best split,
      - ultimately use all data for train and skip calibration.
    Returns: X_tr, y_tr, w_tr, X_cal, y_cal, w_cal, split_tag
    """
    def has_both_classes(y_arr: np.ndarray) -> bool:
        u = np.unique(y_arr)
        return len(u) == 2

    n = len(y)
    if n == 0:
        raise ValueError("Empty dataset after cleaning; cannot train GBM.")

    # Try several stratified splits
    for test_size in (0.3, 0.25, 0.2):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        for tr_idx, cal_idx in sss.split(X, y):
            if len(tr_idx) > 0 and len(cal_idx) > 0:
                if has_both_classes(y[cal_idx]) and has_both_classes(y[tr_idx]):
                    return (
                        X[tr_idx], y[tr_idx], w[tr_idx],
                        X[cal_idx], y[cal_idx], w[cal_idx],
                        f"SSS_{int(test_size*100)}",
                    )

    # Try KFold and pick a fold with both classes on the held-out chunk
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, cal_idx in skf.split(X, y):
        if len(tr_idx) > 0 and len(cal_idx) > 0:
            if has_both_classes(y[cal_idx]) and has_both_classes(y[tr_idx]):
                return (
                    X[tr_idx], y[tr_idx], w[tr_idx],
                    X[cal_idx], y[cal_idx], w[cal_idx],
                    "SKF_5",
                )

    # Final fallback: no proper cal fold; train on all, skip calibration
    warnings.warn("Could not construct a calibration fold with both classes. "
                  "Training on all data; metrics will be in-sample.")
    return X, y, w, np.empty((0, X.shape[1])), np.empty((0,), int), np.empty((0,)), "ALL_TRAIN"


def version_aware_hgbc_kwargs(
    learning_rate: float,
    max_leaf_nodes: int,
    max_depth: int,
    n_estimators: int,
    l2: float,
    early_stopping: bool,
    random_state: int,
    monotonic_cst: List[int],
) -> dict:
    params = inspect.signature(HistGradientBoostingClassifier.__init__).parameters
    kw = dict(
        loss="log_loss",
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_bins=255,
        l2_regularization=l2,
        early_stopping=early_stopping,
        random_state=random_state,
    )
    if "n_estimators" in params:
        kw["n_estimators"] = n_estimators
    elif "max_iter" in params:
        kw["max_iter"] = n_estimators
    else:
        warnings.warn("Neither 'n_estimators' nor 'max_iter' supported; using sklearn default.")

    if "max_depth" in params:
        kw["max_depth"] = max_depth

    if early_stopping and "validation_fraction" in params:
        kw["validation_fraction"] = 0.1

    if "monotonic_cst" in params:
        kw["monotonic_cst"] = monotonic_cst
    else:
        warnings.warn("This sklearn version does not support 'monotonic_cst' — training without constraints.")

    return kw


def train_calibrated_gbm(
    X_tr, y_tr, w_tr, X_cal, y_cal, w_cal,
    monotonic_cst: List[int],
    max_depth: int, learning_rate: float, n_estimators: int, max_leaf_nodes: int,
    l2: float, early_stopping: bool, random_state: int,
    calib_method: str,
):
    kwargs = version_aware_hgbc_kwargs(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_depth=max_depth,
        n_estimators=n_estimators,
        l2=l2,
        early_stopping=early_stopping,
        random_state=random_state,
        monotonic_cst=monotonic_cst,
    )
    gbm = HistGradientBoostingClassifier(**kwargs)
    if X_tr.shape[0] == 0:
        raise ValueError("Training set is empty after splitting. Cannot fit GBM.")
    gbm.fit(X_tr, y_tr, sample_weight=w_tr)

    cal = None
    # Calibrate only if we truly have both classes in cal
    if calib_method in ("isotonic", "sigmoid") and X_cal.shape[0] > 0:
        if len(np.unique(y_cal)) == 2:
            cal = CalibratedClassifierCV(gbm, cv="prefit", method=calib_method)
            cal.fit(X_cal, y_cal, sample_weight=w_cal)
        else:
            warnings.warn("Calibration fold does not contain both classes. Skipping calibration.")
            cal = None
    return gbm, cal


def predict_proba(model, calibrator, X) -> np.ndarray:
    if calibrator is not None:
        return calibrator.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def eval_metrics(y_true: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    metrics = {}
    # AUC/PR-AUC only if both classes present
    classes = np.unique(y_true)
    if len(classes) == 2 and len(y_true) >= 3:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, p))
        except Exception:
            metrics["auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, p))
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Brier
    try:
        metrics["brier"] = float(brier_score_loss(y_true, p))
    except Exception:
        metrics["brier"] = float("nan")

    # decile lift: handle tiny samples
    try:
        n = len(p)
        order = np.argsort(-p)
        y_sorted = y_true[order]
        p_sorted = p[order]
        k = max(1, n // 10)
        lifts = []
        for i in range(10):
            start = i * k
            stop = (i + 1) * k if i < 9 else n
            if start >= n:
                lifts.append(float("nan"))
                continue
            y_bin = y_sorted[start:stop]
            rate = y_bin.mean() if len(y_bin) > 0 else float("nan")
            lifts.append(rate / base_rate if base_rate > 0 and not math.isnan(rate) else float("nan"))
        metrics["decile_lift"] = {"lifts": lifts, "base_rate": base_rate}
    except Exception:
        metrics["decile_lift"] = {"lifts": [float("nan")] * 10, "base_rate": base_rate}

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir of trip_features parquet")
    ap.add_argument("--labels", required=True, help="dir of labels parquet")
    ap.add_argument("--models", required=True, help="output dir for model artifacts")
    ap.add_argument("--metrics_out", required=True, help="path to write metrics json")

    # GBM hyperparams
    ap.add_argument("--learning_rate", type=float, default=0.08)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--max_leaf_nodes", type=int, default=31)
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--early_stopping", type=int, default=1)
    ap.add_argument("--calib_method", type=str, default="isotonic", choices=["isotonic", "sigmoid", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ablations", type=int, default=1)
    args = ap.parse_args()

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = build_dataset(args.features, args.labels)
    base_rate = float(df["claim"].mean())
    X = df[FEATS].to_numpy(float)
    y = df["claim"].to_numpy(int)
    w = df["weight"].to_numpy(float)

    mono_vec = make_monotone_vector()

    # Split (robust)
    X_tr, y_tr, w_tr, X_cal, y_cal, w_cal, split_tag = safe_split(X, y, w, args.seed)

    # Train + calibrate
    gbm, cal = train_calibrated_gbm(
        X_tr, y_tr, w_tr, X_cal, y_cal, w_cal,
        monotonic_cst=mono_vec,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_leaf_nodes=args.max_leaf_nodes,
        l2=args.l2,
        early_stopping=bool(args.early_stopping),
        random_state=args.seed,
        calib_method=args.calib_method if args.calib_method != "none" else "",
    )

    # Evaluate (use calibration fold if present; else in-sample train)
    eval_on = "calibration" if X_cal.shape[0] > 0 else "train"
    X_eval = X_cal if X_cal.shape[0] > 0 else X_tr
    y_eval = y_cal if X_cal.shape[0] > 0 else y_tr

    p_eval = predict_proba(gbm, cal, X_eval)
    metrics = eval_metrics(y_eval, p_eval, base_rate)
    metrics.update({
        "base_rate": base_rate,
        "eval_on": eval_on,
        "split_tag": split_tag,
        "n_total": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_eval": int(len(y_eval)),
        "n_pos_eval": int(y_eval.sum()),
        "n_neg_eval": int(len(y_eval) - y_eval.sum()),
    })

    # Persist
    joblib.dump(gbm, models_dir / "gbm_freq.pkl")
    if cal is not None:
        joblib.dump(cal, models_dir / "gbm_cal.pkl")

    mono_vec = make_monotone_vector()
    meta = {
        # New fields
        "feature_names": FEATS,
        "monotonic": {k: int(MONO_MAP[k]) for k in FEATS},
        "supports_monotone": "monotonic_cst" in inspect.signature(HistGradientBoostingClassifier.__init__).parameters,
        "version_adapted_params": True,
        "seed": args.seed,

        # Legacy fields expected by older verifiers
        "features": FEATS,
        "monotonic_cst": mono_vec,
    }
    (models_dir / "gbm_meta.json").write_text(json.dumps(meta, indent=2))
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))

    # Console summary
    auc = metrics.get("auc", float("nan"))
    pr = metrics.get("pr_auc", float("nan"))
    lifts = metrics.get("decile_lift", {}).get("lifts", [])
    lift1 = lifts[0] if lifts else float("nan")
    print(f"[PH4-TRAIN] eval={eval_on} split={metrics['split_tag']} "
          f"AUC={auc if not math.isnan(auc) else 'nan'} "
          f"PR-AUC={pr if not math.isnan(pr) else 'nan'} "
          f"base={metrics['base_rate']:.4f} lift1={lift1 if isinstance(lift1,(int,float)) else 'nan'} "
          f"pos_eval={metrics['n_pos_eval']} neg_eval={metrics['n_neg_eval']}")


if __name__ == "__main__":
    main()
