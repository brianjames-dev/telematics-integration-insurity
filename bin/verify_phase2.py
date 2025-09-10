#!/usr/bin/env python
import argparse, sys, pathlib
import numpy as np
import pandas as pd

REQ_FEAT_COLS = [
    "driver_id","trip_id","start_ts","end_ts","exposure_km","duration_min",
    "night_ratio","urban_km","highway_km","mean_speed_kph",
    "p95_over_limit_kph","pct_time_over_limit_10",
    "harsh_brakes_per100km","harsh_accels_per100km","cornering_per100km",
    "phone_motion_ratio","rain_flag","crash_grid_index","theft_grid_index","feature_version"
]

def load_parquets(path: str) -> pd.DataFrame:
    base = pathlib.Path(path)
    parts = list(base.rglob("*.parquet")) if base.exists() else list(pathlib.Path().glob(path))
    if not parts:
        raise SystemExit(f"No parquet files found under '{path}'")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

def fail(msg: str):
    print(f"[VERIFY-PH2] FAIL: {msg}")
    sys.exit(1)

def ok(msg: str):
    print(f"[VERIFY-PH2] OK: {msg}")

def ensure_dt_col(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return a copy that has a 'dt' column (YYYY-MM-DD)."""
    out = df.copy()
    if "dt" in out.columns:
        return out
    if "start_dt" in out.columns:
        out["dt"] = out["start_dt"]
        return out
    if "start_ts" in out.columns:
        out["dt"] = pd.to_datetime(out["start_ts"], utc=True).dt.strftime("%Y-%m-%d")
        return out
    fail(f"{name} table has neither 'dt' nor 'start_dt' nor 'start_ts' to derive a date")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/trip_features")
    ap.add_argument("--trips", default="data/trips")
    ap.add_argument("--daily", default="data/driver_daily")
    args = ap.parse_args()

    feats = load_parquets(args.features)
    trips = load_parquets(args.trips)
    daily = load_parquets(args.daily)

    # --- Column presence & dtypes ---
    missing = [c for c in REQ_FEAT_COLS if c not in feats.columns]
    if missing: fail(f"trip_features missing columns: {missing}")
    ok("trip_features columns present")

    # --- No NaNs in critical fields ---
    crit = ["driver_id","trip_id","start_ts","end_ts","exposure_km","duration_min","night_ratio","mean_speed_kph"]
    if feats[crit].isna().any().any(): fail("NaNs detected in critical feature columns")
    ok("no NaNs in critical columns")

    # --- Ranges & sanity checks per row ---
    if not ((feats["night_ratio"]>=0).all() and (feats["night_ratio"]<=1).all()):
        fail("night_ratio out of [0,1]")
    if not ((feats["pct_time_over_limit_10"]>=0).all() and (feats["pct_time_over_limit_10"]<=1).all()):
        fail("pct_time_over_limit_10 out of [0,1]")
    if (feats["exposure_km"]<=0).any(): fail("non-positive exposure_km")
    if (feats["exposure_km"]>600).any(): fail("exposure_km > 600 km (implausible single trip)")
    if (feats["duration_min"]<=0).any(): fail("non-positive duration_min")
    if (feats["duration_min"]>300).any(): fail("duration_min > 300 min (beyond 4h cap + buffer)")
    if (feats["mean_speed_kph"]<0).any() or (feats["mean_speed_kph"]>160).any():
        fail("mean_speed_kph outside [0,160]")
    for c in ["harsh_brakes_per100km","harsh_accels_per100km","cornering_per100km"]:
        if ~np.isfinite(feats[c]).all(): fail(f"non-finite values in {c}")
    ok("row-wise ranges look sane")

    # --- Urban + highway allocation close to exposure ---
    alloc_gap = (feats["urban_km"] + feats["highway_km"]) - feats["exposure_km"]
    if (alloc_gap.abs() > 1e-3).any(): fail("urban_km + highway_km != exposure_km (tolerance 1e-3)")
    ok("urban/highway allocation matches exposure")

    # --- Trips table coherence ---
    if (trips["start_ts"] > trips["end_ts"]).any(): fail("trip start_ts > end_ts")
    if (trips["exposure_km"]<=0).any() or (trips["duration_min"]<=0).any():
        fail("trips table has non-positive exposure/duration")
    ok("trips table coherent")

    # --- Driver-daily consistency (exposure sums) ---
    feats2 = ensure_dt_col(feats, name="trip_features")
    daily2 = ensure_dt_col(daily, name="driver_daily")
    exp_from_feats = feats2.groupby(["driver_id","dt"])["exposure_km"].sum().reset_index()
    exp_from_feats.rename(columns={"exposure_km":"exp_feats"}, inplace=True)

    if "exposure_km_day" not in daily2.columns:
        fail("driver_daily missing exposure_km_day")
    merged = daily2.merge(exp_from_feats, on=["driver_id","dt"], how="left")
    if (merged["exp_feats"].isna()).any(): fail("driver_daily keys not found in trip_features")
    if ((merged["exposure_km_day"] - merged["exp_feats"]).abs() > 1e-6).any():
        fail("driver_daily exposure_km_day != sum(trip_features.exposure_km)")
    ok("driver_daily exposure matches summed trip exposures")

    # --- Summary ---
    print("\n[VERIFY-PH2] Summary")
    print(f" drivers={feats['driver_id'].nunique()} trips={len(trips)} features={len(feats)} days={daily2['dt'].nunique()}")
    print(f" exposure_km: min={feats['exposure_km'].min():.2f} p50={feats['exposure_km'].median():.2f} max={feats['exposure_km'].max():.2f}")
    samp = feats[["driver_id","trip_id","exposure_km","duration_min","mean_speed_kph","night_ratio"]].head(5)
    print(samp.to_string(index=False))
    ok("Phase 2 verification passed")

if __name__ == "__main__":
    sys.exit(main())
