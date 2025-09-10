import pandas as pd, numpy as np
from src.ml.label_simulator import simulate_labels

def test_label_shapes_deterministic():
    df = pd.DataFrame({
        "driver_id":["D_1"]*5, "trip_id":[f"T{i}" for i in range(5)],
        "start_ts": pd.date_range("2025-09-09", periods=5, freq="1h", tz="UTC"),
        "exposure_km": np.linspace(10, 50, 5),
        "night_ratio": np.linspace(0,1,5),
        "p95_over_limit_kph": np.linspace(0,30,5),
        "harsh_brakes_per100km": np.linspace(0,5,5),
        "phone_motion_ratio": np.linspace(0,0.5,5),
        "rain_flag": [0,1,0,0,1],
        "mean_speed_kph": np.linspace(40,70,5),
        "crash_grid_index": np.linspace(0.4,0.6,5),
    })
    df["dt"] = df["start_ts"].dt.strftime("%Y-%m-%d")
    out = simulate_labels(df, seed=123)
    assert set(["driver_id","trip_id","dt","claim","severity"]).issubset(out.columns)
    assert len(out) == len(df)
    assert (out["severity"] >= 0).all()
