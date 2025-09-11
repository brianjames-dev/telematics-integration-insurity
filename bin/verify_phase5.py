#!/usr/bin/env python
# bin/verify_phase5.py
import sys, json, pathlib
from typing import Dict, Any

import pandas as pd
from fastapi.testclient import TestClient

# Ensure project root on path when running directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.serve.api import app  # uses your api.py
client = TestClient(app)

def _read_first_feature_row() -> Dict[str, Any]:
    # find any parquet under data/trip_features
    parts = sorted(pathlib.Path("data/trip_features").rglob("*.parquet"))
    assert parts, "No trip_features parquet found. Run Phase 2 first."
    df = pd.read_parquet(parts[0])
    # coerce to python types we need for TripFeatureIn payload
    row = df.iloc[0].to_dict()
    pay = {
        "driver_id": str(row["driver_id"]),
        "trip_id": str(row.get("trip_id", "")) or None,
        "exposure_km": float(row["exposure_km"]),
        "night_ratio": float(row["night_ratio"]),
        "p95_over_limit_kph": float(row["p95_over_limit_kph"]),
        "harsh_brakes_per100km": float(row["harsh_brakes_per100km"]),
        "phone_motion_ratio": float(row["phone_motion_ratio"]),
        "rain_flag": int(row.get("rain_flag", 0)),
        "mean_speed_kph": float(row["mean_speed_kph"]),
        "crash_grid_index": float(row["crash_grid_index"]),
    }
    return pay

def main() -> int:
    # 1) health
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("ok") is True, "health failed"
    print("[VERIFY-PH5] /health ok")

    # 2) /score/trip with a real feature row
    payload = _read_first_feature_row()
    r = client.post("/score/trip", json=payload)
    assert r.status_code == 200, f"/score/trip http {r.status_code}"
    out = r.json()
    for k in ("p_claim", "sev_mean", "expected_cost_per100km"):
        assert k in out, f"/score/trip missing {k}"
        assert 0 <= float(out["p_claim"]) <= 1, "p_claim out of [0,1]"
        assert float(out["sev_mean"]) >= 0, "sev_mean negative?"
    print(f"[VERIFY-PH5] /score/trip ok  p_claim={out['p_claim']:.3f}  sev={out['sev_mean']:.1f}  exp_cost={out['expected_cost_per100km']:.2f}")

    # 3) /score/driver (use driver_id from the payload)
    driver_id = payload["driver_id"]
    r = client.get(f"/score/driver/{driver_id}", params={"window_days": 365})
    assert r.status_code == 200, f"/score/driver http {r.status_code}"
    d = r.json()
    assert d["driver_id"] == driver_id, "driver_id mismatch"
    assert int(d["n_trips"]) >= 1, "no trips returned for driver window"
    print(f"[VERIFY-PH5] /score/driver ok  trips={d['n_trips']} exp_km={d['exposure_km']:.2f}")

    # --- Quote ---
    q = {
        "driver_id": driver_id,
        "window_days": 365,
        "annual_km": 12_000,
        "lae_ratio": 0.10,
        "expense_ratio": 0.25,
        "target_margin": 0.05,
        "min_premium": 1500.0,          # floor, guarantees > 0
        "prior_premium": 2400.0,       # enable caps around a realistic prior
        "max_change": 0.15,            # ±15% cap
    }
    r = client.post("/price/quote", json=q)
    r.raise_for_status()
    resp = r.json()
    pricing = resp["pricing"]
    prem = float(pricing["premium"])
    print(f"[VERIFY-PH5] /price/quote ok  premium={prem:.2f}  details={pricing}")
    assert prem > 0, "premium not positive"

    print("[VERIFY-PH5] Phase 5 verification passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
