from pathlib import Path
import json
import pandas as pd
from src.ingest.ingest import process

def test_ingest_partitions_and_dedup(tmp_path: Path):
    # Create a small NDJSON with a duplicate ping
    ndjson = tmp_path / "sample.ndjson"
    rows = [
        {
            "driver_id": "D_001",
            "trip_hint_id": "H1",
            "ts": "2025-09-09T10:00:00Z",
            "lat": 0.0, "lon": 0.0, "gps_acc_m": 3.0,
            "speed_mps": 10.0, "accel_mps2": 0.0, "gyro_z": 0.0,
            "source": "phone", "ingest_id": "11111111-1111-1111-1111-111111111111",
            "schema_version": 1,
        },
        {
            "driver_id": "D_001",  # duplicate ts should be de-duped
            "trip_hint_id": "H1",
            "ts": "2025-09-09T10:00:00Z",
            "lat": 0.1, "lon": 0.1, "gps_acc_m": 3.0,
            "speed_mps": 10.1, "accel_mps2": 0.0, "gyro_z": 0.0,
            "source": "phone", "ingest_id": "22222222-2222-2222-2222-222222222222",
            "schema_version": 1,
        },
        {
            "driver_id": "D_002",
            "trip_hint_id": "H2",
            "ts": "2025-09-09T11:00:00Z",
            "lat": 1.0, "lon": 1.0, "gps_acc_m": 4.0,
            "speed_mps": 12.0, "accel_mps2": 0.1, "gyro_z": 0.01,
            "source": "phone", "ingest_id": "33333333-3333-3333-3333-333333333333",
            "schema_version": 1,
        },
    ]
    ndjson.write_text("\n".join(json.dumps(r) for r in rows))

    out_dir = tmp_path / "pings"
    written = process(str(ndjson), str(out_dir))
    # 3 rows → 2 unique rows expected
    assert written == 2

    # Check partition structure exists
    assert any(p.name.startswith("driver_id=") for p in out_dir.iterdir())

    # Read back and verify unique ts per driver
    parts = list(out_dir.rglob("*.parquet"))
    assert parts, "No parquet files written"
    dfs = [pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    assert df.drop_duplicates(subset=["driver_id", "ts"]).shape[0] == df.shape[0]
