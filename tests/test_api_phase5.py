import pathlib
import pandas as pd
from fastapi.testclient import TestClient

from src.serve.api import app
client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True

def _first_payload():
    parts = sorted(pathlib.Path("data/trip_features").rglob("*.parquet"))
    assert parts, "Run Phase 2 first"
    df = pd.read_parquet(parts[0])
    row = df.iloc[0].to_dict()
    return {
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

def test_score_trip_and_driver_and_quote():
    payload = _first_payload()

    # /score/trip
    r = client.post("/score/trip", json=payload)
    assert r.status_code == 200
    t = r.json()
    assert 0.0 <= t["p_claim"] <= 1.0
    assert t["sev_mean"] >= 0.0
    assert t["expected_cost_per100km"] >= 0.0

    # /score/driver
    did = payload["driver_id"]
    r = client.get(f"/score/driver/{did}", params={"window_days": 365})
    assert r.status_code == 200
    d = r.json()
    assert d["driver_id"] == did
    assert d["n_trips"] >= 1
    assert d["expected_cost_per100km"] >= 0.0

    # /price/quote (with prior to exercise caps)
    q = {
        "driver_id": did, "window_days": 365,
        "annual_km": 12_000, "lae_ratio": 0.10, "expense_ratio": 0.25,
        "target_margin": 0.05, "min_premium": 300.0,
        "prior_premium": 500.0, "max_change": 0.15,
    }
    r = client.post("/price/quote", json=q)
    assert r.status_code == 200
    resp = r.json()
    prem = float(resp["pricing"]["premium"])
    assert prem >= q["min_premium"]  # floor
    # If prior given, premium must be within caps
    lo, hi = 500*(1-0.15), 500*(1+0.15)
    assert lo <= prem <= hi
