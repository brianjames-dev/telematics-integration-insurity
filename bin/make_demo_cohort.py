#!/usr/bin/env python
# bin/make_demo_cohort.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/demo_batch")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# Helper to draw truncated normals
def tnorm(mu, sigma, lo, hi, size):
    x = rng.normal(mu, sigma, size=size)
    return np.clip(x, lo, hi)

def make_cohort(prefix: str, n_drivers: int, trips_per_driver: int, profile: str) -> pd.DataFrame:
    rows = []
    for d in range(n_drivers):
        driver_id = f"{prefix}_{d+1:03d}"
        for t in range(trips_per_driver):
            trip_id = f"T_{rng.integers(1e12):x}"

            # Profiles: change the means to push risk up/down
            if profile == "safe":
                night_ratio            = tnorm(0.05, 0.05, 0.00, 0.20, 1)[0]
                p95_over_limit_kph     = tnorm(2.0,  2.0,  0.0,  8.0, 1)[0]
                harsh_brakes_per100km  = tnorm(8.0,  6.0,  0.0, 30.0, 1)[0]
                phone_motion_ratio     = tnorm(0.05, 0.05, 0.00, 0.20, 1)[0]
                rain_flag              = int(rng.random() < 0.20)
                mean_speed_kph         = tnorm(45.0, 8.0,  25.0, 65.0, 1)[0]
                crash_grid_index       = tnorm(0.20, 0.10, 0.00, 0.50, 1)[0]
            elif profile == "risky":
                night_ratio            = tnorm(0.45, 0.20, 0.10, 1.00, 1)[0]
                p95_over_limit_kph     = tnorm(25.0, 8.0,  8.0,  45.0, 1)[0]
                harsh_brakes_per100km  = tnorm(80.0, 40.0, 10.0, 200.0, 1)[0]
                phone_motion_ratio     = tnorm(0.25, 0.10, 0.05, 0.60, 1)[0]
                rain_flag              = int(rng.random() < 0.60)
                mean_speed_kph         = tnorm(70.0, 10.0, 50.0, 95.0, 1)[0]
                crash_grid_index       = tnorm(0.75, 0.15, 0.30, 1.00, 1)[0]
            else:  # average
                night_ratio            = tnorm(0.20, 0.15, 0.00, 0.60, 1)[0]
                p95_over_limit_kph     = tnorm(12.0, 6.0,  0.0,  30.0, 1)[0]
                harsh_brakes_per100km  = tnorm(35.0, 20.0, 5.0, 120.0, 1)[0]
                phone_motion_ratio     = tnorm(0.12, 0.08, 0.00, 0.40, 1)[0]
                rain_flag              = int(rng.random() < 0.35)
                mean_speed_kph         = tnorm(55.0, 10.0, 35.0, 80.0, 1)[0]
                crash_grid_index       = tnorm(0.45, 0.20, 0.10, 0.85, 1)[0]

            exposure_km = tnorm(30.0, 15.0, 5.0, 120.0, 1)[0]

            rows.append(dict(
                driver_id=driver_id, trip_id=trip_id,
                exposure_km=float(exposure_km),
                night_ratio=float(night_ratio),
                p95_over_limit_kph=float(p95_over_limit_kph),
                harsh_brakes_per100km=float(harsh_brakes_per100km),
                phone_motion_ratio=float(phone_motion_ratio),
                rain_flag=int(rain_flag),
                mean_speed_kph=float(mean_speed_kph),
                crash_grid_index=float(crash_grid_index),
            ))
    return pd.DataFrame(rows)

def main():
    safe   = make_cohort("SAFE",   n_drivers=5, trips_per_driver=20, profile="safe")
    avg    = make_cohort("AVG",    n_drivers=5, trips_per_driver=20, profile="avg")
    risky  = make_cohort("RISKY",  n_drivers=5, trips_per_driver=20, profile="risky")
    df = pd.concat([safe, avg, risky], ignore_index=True)
    out = OUT / "trips.csv"
    df.to_csv(out, index=False)
    print(f"[demo cohort] wrote {out}  rows={len(df)}  drivers={df['driver_id'].nunique()}")

if __name__ == "__main__":
    main()
