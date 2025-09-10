from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import numpy as np
import pandas as pd

from src.processing.feature_defs import (
    IDLE_GAP_SEC, LOW_SPEED_MPS, LOW_SPEED_IDLE_SEC, MAX_TRIP_SEC,
    grid_cell, haversine_km, speed_limit_kph_for_cell, is_rain_cell_day, risk_index,
    flag_harsh_brake, flag_harsh_accel, flag_corner, phone_motion_mask,
    per100km, night_mask
)
from src.serve.schemas import TripFeatures


def _trip_id(driver_id: str, start_ts: pd.Timestamp) -> str:
    key = f"{driver_id}|{start_ts.isoformat()}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"T_{h}"


def _split_trips_driver(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Return list of (start_idx, end_idx_inclusive) row ranges for trips.
    Splits by:
      - gap >= IDLE_GAP_SEC
      - continuous low-speed window >= LOW_SPEED_IDLE_SEC
      - hard-cap MAX_TRIP_SEC
    """
    idx = df.index.to_numpy()
    ts = df["ts"].astype("int64").to_numpy() // 1_000_000_000  # seconds
    speed = df["speed_mps"].to_numpy()

    ranges = []
    start = 0
    low_start = None
    for i in range(1, len(df)):
        # split on idle gap
        if (ts[i] - ts[i-1]) >= IDLE_GAP_SEC:
            ranges.append((start, i-1))
            start = i
            low_start = None
            continue

        # track continuous low-speed
        if speed[i] < LOW_SPEED_MPS:
            if low_start is None:
                low_start = i-1
            # end trip when continuous low-speed window passes threshold
            if (ts[i] - ts[low_start]) >= LOW_SPEED_IDLE_SEC:
                ranges.append((start, i))
                start = i + 1
                low_start = None
                continue
        else:
            low_start = None

        # hard cap
        if (ts[i] - ts[start]) >= MAX_TRIP_SEC:
            ranges.append((start, i))
            start = i + 1
            low_start = None

    # tail
    if start <= len(df) - 1:
        ranges.append((start, len(df) - 1))
    # drop degenerate
    ranges = [(s, e) for (s, e) in ranges if e >= s]
    return ranges


def _trip_features_from_slice(g: pd.DataFrame) -> Dict:
    """
    Compute TripFeatures from a trip slice (sorted by ts).
    """
    # 1) Time deltas in seconds (float)
    dt_sec = (
        g["ts"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    )
    dt_sec = np.clip(dt_sec, 0.0, None)

    # 2) Distance via speed integration (robust to GPS jitter)
    speed_mps = g["speed_mps"].to_numpy()
    step_km = (speed_mps * dt_sec) / 1000.0
    exposure_km = float(step_km.sum())

    # 3) Grid + synthetic speed limit per ping (for overlimit + urban/highway split)
    lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy()
    cells = np.array([grid_cell(a, b) for a, b in zip(lat, lon)])
    sl_kph = np.array([speed_limit_kph_for_cell(c) for c in cells])

    # 4) Speeds and overspeed features
    speed_kph = speed_mps * 3.6
    over = np.maximum(0.0, speed_kph - sl_kph)
    p95_over = float(np.percentile(over, 95)) if len(over) else 0.0
    pct_over_10 = float(np.mean(speed_kph > (sl_kph + 10.0))) if len(over) else 0.0

    # 5) Events
    accel = g["accel_mps2"].to_numpy()
    gyro = g["gyro_z"].to_numpy()
    n_brake = int(flag_harsh_brake(accel).sum())
    n_accel = int(flag_harsh_accel(accel).sum())
    n_corner = int(flag_corner(gyro).sum())
    brakes_100 = per100km(n_brake, exposure_km)
    accels_100 = per100km(n_accel, exposure_km)
    corner_100 = per100km(n_corner, exposure_km)

    # 6) Phone motion
    phone_ratio = float(np.mean(phone_motion_mask(accel, gyro))) if len(g) else 0.0

    # 7) Context
    day_key = pd.to_datetime(g["ts"].iloc[0], utc=True).strftime("%Y-%m-%d")
    rain = int(np.max([is_rain_cell_day(c, day_key) for c in np.unique(cells)]) > 0)
    crash_idx = float(np.mean([risk_index(c, "crash") for c in cells]))
    theft_idx = float(np.mean([risk_index(c, "theft") for c in cells]))

    # 8) Urban vs highway distance (allocate integrated distance by limit)
    urban_km = float(step_km[sl_kph <= 80.0].sum())
    highway_km = float(step_km[sl_kph > 80.0].sum())

    # 9) Durations & night
    duration_min = float(dt_sec.sum() / 60.0)
    # time-weighted mean speed (kph)
    mean_speed_kph = float((speed_mps * dt_sec).sum() / (dt_sec.sum() + 1e-9) * 3.6)
    night_ratio = float(np.mean(night_mask(g["ts"]))) if len(g) else 0.0

    # 10) IDs & payload (validated by Pydantic)
    driver_id = str(g["driver_id"].iloc[0])
    start_ts = pd.to_datetime(g["ts"].iloc[0], utc=True)
    end_ts = pd.to_datetime(g["ts"].iloc[-1], utc=True)
    trip_id = _trip_id(driver_id, start_ts)

    payload = {
        "driver_id": driver_id,
        "trip_id": trip_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "exposure_km": exposure_km,
        "duration_min": duration_min,
        "night_ratio": night_ratio,
        "urban_km": urban_km,
        "highway_km": highway_km,
        "mean_speed_kph": mean_speed_kph,
        "p95_over_limit_kph": p95_over,
        "pct_time_over_limit_10": pct_over_10,
        "harsh_brakes_per100km": brakes_100,
        "harsh_accels_per100km": accels_100,
        "cornering_per100km": corner_100,
        "phone_motion_ratio": phone_ratio,
        "rain_flag": rain,
        "crash_grid_index": crash_idx,
        "theft_grid_index": theft_idx,
        "feature_version": 1,
    }
    TripFeatures.model_validate(payload)
    return payload


def aggregate_trips(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Input: pings dataframe with columns [driver_id, ts, lat, lon, speed_mps, accel_mps2, gyro_z, ...]
    Output:
        trips_df: driver_id, trip_id, start_ts, end_ts, exposure_km, duration_min
        trip_features_df: TripFeatures schema fields
    """
    # ensure proper dtypes & sort
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values(["driver_id", "ts"]).reset_index(drop=True)

    trips = []
    feats = []

    for d, g in df.groupby("driver_id", sort=False):
        g = g.reset_index(drop=True)
        ranges = _split_trips_driver(g)
        for s, e in ranges:
            slice_df = g.iloc[s : e + 1]
            if len(slice_df) < 2:
                continue
            # compute features
            feat = _trip_features_from_slice(slice_df)
            feats.append(feat)
            trips.append({
                "driver_id": feat["driver_id"],
                "trip_id": feat["trip_id"],
                "start_ts": feat["start_ts"],
                "end_ts": feat["end_ts"],
                "exposure_km": feat["exposure_km"],
                "duration_min": feat["duration_min"],
            })

    trips_df = pd.DataFrame(trips)
    feats_df = pd.DataFrame(feats)
    return trips_df, feats_df


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def write_partitioned(df: pd.DataFrame, base: Path, key_ts: str = "start_ts") -> int:
    base.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return 0
    df = df.copy()
    df["dt"] = pd.to_datetime(df[key_ts], utc=True).dt.strftime("%Y-%m-%d")
    total = 0
    for (driver_id, dt), g in df.groupby(["driver_id", "dt"]):
        out_dir = base / f"driver_id={driver_id}" / f"dt={dt}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # CLEAN OUT old parts so we don't mix old/new rows
        for p in out_dir.glob("*.parquet"):
            p.unlink()
        # deterministic filename
        fname = f"part-{_sha1_hex(driver_id + '|' + dt)[:12]}.parquet"
        out_path = out_dir / fname
        g.drop(columns=["dt"], inplace=True)
        g.to_parquet(out_path, index=False)
        total += len(g)
    return total


def to_driver_daily(trip_features: pd.DataFrame) -> pd.DataFrame:
    """Simple driver-day aggregate (exposure + a few means)."""
    if trip_features.empty:
        return trip_features.copy()
    df = trip_features.copy()
    df["dt"] = pd.to_datetime(df["start_ts"], utc=True).dt.strftime("%Y-%m-%d")

    def _agg(g):
        w = g["exposure_km"].replace(0, 1e-6)
        return pd.Series({
            "exposure_km_day": g["exposure_km"].sum(),
            "mean_night_ratio": np.average(g["night_ratio"], weights=w),
            "mean_p95_over_limit_kph": np.average(g["p95_over_limit_kph"], weights=w),
            "harsh_brakes_per100km_day": np.average(g["harsh_brakes_per100km"], weights=w),
        })

    try:
        agg = df.groupby(["driver_id", "dt"]).apply(_agg, include_groups=False).reset_index()
    except TypeError:
        # Older pandas without include_groups
        agg = df.groupby(["driver_id", "dt"]).apply(_agg).reset_index()

    return agg


def load_pings(input_path_glob: str) -> pd.DataFrame:
    parts = list(Path(input_path_glob).rglob("*.parquet")) if Path(input_path_glob).exists() else []
    if not parts:
        # also allow path patterns like data/pings_run1/driver_id=*/dt=*/part-*.parquet
        parts = list(Path().glob(input_path_glob))
    if not parts:
        raise FileNotFoundError(f"No parquet files found under '{input_path_glob}'")
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/pings_run1")
    ap.add_argument("--out_trips", type=str, default="data/trips")
    ap.add_argument("--out_features", type=str, default="data/trip_features")
    ap.add_argument("--out_daily", type=str, default="data/driver_daily")
    args = ap.parse_args()

    in_path = args.input
    # accept a directory or a glob path
    if Path(in_path).is_dir():
        glob_pat = str(Path(in_path) / "driver_id=*/dt=*/part-*.parquet")
    else:
        glob_pat = in_path

    pings = load_pings(glob_pat)
    trips_df, feats_df = aggregate_trips(pings)

    n_trips = write_partitioned(trips_df, Path(args.out_trips), key_ts="start_ts")
    n_feats = write_partitioned(feats_df, Path(args.out_features), key_ts="start_ts")

    daily = to_driver_daily(feats_df)
    _ = write_partitioned(
        daily.rename(columns={"dt": "start_dt"}).assign(
            start_ts=pd.to_datetime(daily["dt"], utc=True)
        ),
        Path(args.out_daily),
        key_ts="start_ts",
    )

    print(f"[PH2] trips={len(trips_df)} trip_features={len(feats_df)} written_trips={n_trips} written_features={n_feats}")


if __name__ == "__main__":
    main()
