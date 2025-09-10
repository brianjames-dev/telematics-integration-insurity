from pathlib import Path
import json
from src.simulator.generate_trips import generate_ndjson

def test_simulator_golden(tmp_path: Path):
    out = tmp_path / "pings.ndjson"
    generate_ndjson(out, drivers=1, trips_per_driver=2, hz=1.0, seed=123)
    lines = out.read_text().strip().splitlines()
    assert len(lines) > 0
    # Validate ordering and required keys on first 10 rows
    prev_ts = None
    for row in lines[:10]:
        obj = json.loads(row)
        assert "driver_id" in obj and "ts" in obj
        if prev_ts:
            assert obj["ts"] >= prev_ts
        prev_ts = obj["ts"]
