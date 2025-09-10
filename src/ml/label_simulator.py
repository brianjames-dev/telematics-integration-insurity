from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RNG_SEED = 1337

LABEL_FEATS = [
    "exposure_km","night_ratio","p95_over_limit_kph",
    "harsh_brakes_per100km","phone_motion_ratio","rain_flag",
    "mean_speed_kph","crash_grid_index","start_ts","driver_id","trip_id"
]

def load_trip_features(path: str) -> pd.DataFrame:
    base = Path(path)
    parts = list(base.rglob("*.parquet")) if base.exists() else list(Path().glob(path))
    if not parts:
        raise FileNotFoundError(f"No parquet under {path}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    missing = [c for c in LABEL_FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"trip_features missing columns: {missing}")
    df["dt"] = pd.to_datetime(df["start_ts"], utc=True).dt.strftime("%Y-%m-%d")
    return df

def _mean_prob(lin: np.ndarray, intercept: float) -> float:
    p = 1.0 / (1.0 + np.exp(-(lin + intercept)))
    return float(p.mean())

def _calibrate_intercept(lin: np.ndarray, target_rate: float) -> float:
    """Find b s.t. mean(sigmoid(lin + b)) ~= target_rate, with dynamic bracketing."""
    # start with a modest bracket and expand until target is between f(lo) and f(hi)
    lo, hi = -20.0, 20.0
    f_lo = _mean_prob(lin, lo) - target_rate
    f_hi = _mean_prob(lin, hi) - target_rate
    expand = 0
    while f_lo > 0 or f_hi < 0:
        # widen bracket exponentially
        lo -= 20.0
        hi += 20.0
        f_lo = _mean_prob(lin, lo) - target_rate
        f_hi = _mean_prob(lin, hi) - target_rate
        expand += 1
        if expand > 50:
            # fallback: give up expanding; return mid which minimizes absolute error
            return 0.5 * (lo + hi)
    # bisection
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = _mean_prob(lin, mid) - target_rate
        if m > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def simulate_labels(feats: pd.DataFrame, seed: int = RNG_SEED, target_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = feats.copy()

    # --- sanitize/clamp only for label generation (features remain unchanged on disk) ---
    exp100 = (df["exposure_km"] / 100.0).clip(lower=1e-6)
    night = df["night_ratio"].clip(0, 1)
    over = df["p95_over_limit_kph"].clip(0, 40)               # cap overspeed
    brakes = df["harsh_brakes_per100km"].clip(0, 15)          # cap event rate
    phone = df["phone_motion_ratio"].clip(0, 1)
    rain  = df["rain_flag"].clip(0, 1)
    meanv = df["mean_speed_kph"].clip(0, 120)
    crash = df["crash_grid_index"].clip(0, 1)

    # exposure offset (per 100 km)
    offset = np.log(exp100)

    # linear predictor WITHOUT intercept
    lin = (
        offset
        + 0.8  * night
        + 0.015 * over
        + 0.04 * brakes
        + 0.8  * phone
        + 0.25 * rain
        + 0.001 * meanv
        + 0.6  * (crash - 0.5)
    ).to_numpy(dtype=float)

    # calibrate intercept to hit target rate
    intercept = _calibrate_intercept(lin, target_rate=target_rate)

    p_claim = 1.0 / (1.0 + np.exp(-(lin + intercept)))
    p_claim = np.clip(p_claim, 1e-4, 1 - 1e-4)
    claim = (rng.uniform(size=len(df)) < p_claim).astype(int)

    # --- severity (given claim): lognormal with mild signals ---
    log_mu = (
        6.2
        + 0.005 * over
        + 0.4   * (crash - 0.5)
        + 0.12  * phone
        + 0.12  * night
        + 0.20  * np.log(exp100)
    ).to_numpy(dtype=float)
    sigma = 0.7
    severity = np.where(
        claim == 1,
        np.exp(rng.normal(loc=log_mu, scale=sigma)),
        0.0,
    )

    out = df[["driver_id","trip_id","dt"]].copy()
    out["claim"] = claim.astype(int)
    out["severity"] = severity.astype(float)
    out["p_claim_true"] = p_claim.astype(float)
    out["intercept_used"] = float(intercept)
    return out

def write_partitioned(df: pd.DataFrame, base: Path) -> int:
    base.mkdir(parents=True, exist_ok=True)
    total = 0
    for (driver_id, dt), g in df.groupby(["driver_id","dt"]):
        out_dir = base / f"driver_id={driver_id}" / f"dt={dt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in out_dir.glob("*.parquet"):
            p.unlink()
        out_path = out_dir / "part-labels.parquet"
        g.to_parquet(out_path, index=False)
        total += len(g)
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/trip_features")
    ap.add_argument("--out", default="data/labels_trip")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--target-rate", type=float, default=0.03)
    args = ap.parse_args()

    feats = load_trip_features(args.features)
    lbl = simulate_labels(feats, seed=args.seed, target_rate=args.target_rate)
    n = write_partitioned(lbl, Path(args.out))
    achieved = float(lbl["claim"].mean()) if len(lbl) else 0.0
    print(f"[PH3-LABELS] wrote={n} rows to {args.out}; target_rate={args.target_rate:.3%} achieved={achieved:.3%}; intercept={lbl['intercept_used'].iloc[0]:.3f}")

if __name__ == "__main__":
    main()
