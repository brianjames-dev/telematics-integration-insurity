from datetime import datetime, timezone
from src.serve.schemas import TelemetryPing, TripFeatures

def test_ping_schema_example_validates():
    ping = TelemetryPing.model_validate({
        "driver_id": "D_001",
        "trip_hint_id": "H20250909_0001",
        "ts": "2025-09-09T10:03:02.120Z",
        "lat": 37.7749,
        "lon": -122.4194,
        "gps_acc_m": 4.0,
        "speed_mps": 12.3,
        "accel_mps2": -0.3,
        "gyro_z": 0.05,
        "source": "phone",
        "ingest_id": "9f1a1f4e-b2a5-4bc1-9c8e-0a9b3c35d6c1",
        "schema_version": 1
    })
    assert ping.driver_id == "D_001"
    assert ping.ts.tzinfo is not None

def test_trip_features_contract_fields():
    tf = TripFeatures.model_validate({
        "driver_id": "D_001",
        "trip_id": "T_1",
        "start_ts": "2025-09-09T10:00:00Z",
        "end_ts": "2025-09-09T10:30:00Z",
        "exposure_km": 12.3,
        "duration_min": 30.0,
        "night_ratio": 0.2,
        "urban_km": 6.0,
        "highway_km": 6.3,
        "mean_speed_kph": 40.0,
        "p95_over_limit_kph": 8.0,
        "pct_time_over_limit_10": 0.05,
        "harsh_brakes_per100km": 1.2,
        "harsh_accels_per100km": 0.9,
        "cornering_per100km": 0.1,
        "phone_motion_ratio": 0.05,
        "rain_flag": 0,
        "crash_grid_index": 0.2,
        "theft_grid_index": 0.1,
        "feature_version": 1
    })
    assert tf.exposure_km >= 0
    assert 0 <= tf.night_ratio <= 1
