import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.processing.trip_aggregator import aggregate_trips
from src.serve.schemas import TripFeatures

def test_single_trip_features_contract():
    t0 = datetime(2025, 9, 9, 10, 0, 0, tzinfo=timezone.utc)
    ts = [t0 + timedelta(seconds=i*10) for i in range(60)]
    df = pd.DataFrame({
        "driver_id": ["D_007"] * len(ts),
        "ts": pd.to_datetime(ts, utc=True),
        "lat": np.linspace(37.77, 37.78, len(ts)),
        "lon": np.linspace(-122.42, -122.41, len(ts)),
        "speed_mps": np.clip(np.random.normal(10.0, 1.0, len(ts)), 0, None),
        "accel_mps2": np.random.normal(0.0, 0.4, len(ts)),
        "gyro_z": np.random.normal(0.0, 0.1, len(ts)),
        "gps_acc_m": np.random.uniform(2.0, 6.0, len(ts)),
        "source": ["phone"] * len(ts),
        "ingest_id": ["00000000-0000-0000-0000-000000000000"] * len(ts),
        "schema_version": [1] * len(ts),
    })
    trips, feats = aggregate_trips(df)
    assert len(trips) == 1
    assert len(feats) == 1
    # Contract validation
    TripFeatures.model_validate(feats.iloc[0].to_dict())
    # Reasonable ranges
    assert feats["exposure_km"].iloc[0] > 0.01
    assert 0.0 <= feats["night_ratio"].iloc[0] <= 1.0
