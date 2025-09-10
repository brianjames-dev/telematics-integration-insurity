from __future__ import annotations
import argparse
import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

from pydantic import ValidationError
from src.serve.schemas import TelemetryPing


# ---------- Personas (tunable) ----------
PERSONAS = {
    "safe": {
        "base_speed_mps": (8, 14),        # ~18-31 mph
        "overspeed_bias": 0.5,            # lower => less overspeed
        "brake_prob": 0.02,
        "accel_prob": 0.02,
        "night_frac": 0.1,
    },
    "aggressive": {
        "base_speed_mps": (12, 20),       # ~27-45 mph
        "overspeed_bias": 1.6,
        "brake_prob": 0.08,
        "accel_prob": 0.08,
        "night_frac": 0.2,
    },
    "night_owl": {
        "base_speed_mps": (9, 16),
        "overspeed_bias": 1.0,
        "brake_prob": 0.03,
        "accel_prob": 0.03,
        "night_frac": 0.65,
    },
}

START_COORDS = (37.7749, -122.4194)  # SF-ish


def _jitter_coord(lat: float, lon: float, meters: float) -> Tuple[float, float]:
    # crude lat/lon jitter; good enough for POC
    dlat = meters / 111_111.0
    dlon = meters / (111_111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def _is_night(ts: datetime) -> bool:
    hour = ts.astimezone(timezone.utc).hour  # pretend local ~ UTC for POC
    return (hour >= 22) or (hour < 5)


def _gen_trip_pings(
    driver_id: str,
    start_ts: datetime,
    duration_min: int,
    hz: float,
    persona: Dict,
    trip_idx: int,
) -> Iterable[Dict]:
    steps = max(1, int(duration_min * 60 * hz))
    lat, lon = START_COORDS
    dt = 1.0 / hz

    # choose baseline speed for trip
    base_speed = random.uniform(*persona["base_speed_mps"])
    for i in range(steps):
        ts = start_ts + timedelta(seconds=i * dt)

        # Decide if this instant should be night; nudge schedule to persona
        if random.random() < persona["night_frac"]:
            # shift timestamp into night hours randomly (POC trick)
            hour = random.choice([22, 23, 0, 1, 2, 3, 4])
            ts = ts.replace(hour=hour, tzinfo=timezone.utc)

        # speed dynamics + overspeed bias
        speed = max(
            0.0,
            random.gauss(base_speed, 1.5) + random.random() * persona["overspeed_bias"]
        )

        # accel/gyro synthetic events
        accel = 0.0
        gyro = 0.0
        if random.random() < persona["brake_prob"]:
            accel = random.uniform(-4.0, -2.5)  # harsh brake spike
        elif random.random() < persona["accel_prob"]:
            accel = random.uniform(2.5, 4.0)   # harsh accel spike

        # light turning
        if random.random() < 0.02:
            gyro = random.uniform(0.35, 0.8) * random.choice([1, -1])

        # simple walk for gps
        lat, lon = _jitter_coord(lat, lon, meters=max(0.5, speed * dt))

        yield {
            "driver_id": driver_id,
            "trip_hint_id": f"H{ts.strftime('%Y%m%d')}_{trip_idx:04d}",
            "ts": ts.isoformat().replace("+00:00", "Z"),
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "gps_acc_m": round(random.uniform(2.0, 8.0), 2),
            "speed_mps": round(speed, 2),
            "accel_mps2": round(accel if accel != 0.0 else random.uniform(-0.5, 0.5), 2),
            "gyro_z": round(gyro if gyro != 0.0 else random.uniform(-0.1, 0.1), 2),
            "source": "phone",
            "ingest_id": str(uuid4()),
            "schema_version": 1,
        }


def generate_ndjson(
    out_path: Path,
    drivers: int,
    trips_per_driver: int,
    hz: float,
    seed: int,
) -> None:
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    personas = list(PERSONAS.keys())

    with out_path.open("w", encoding="utf-8") as f:
        for d in range(drivers):
            driver_id = f"D_{d+1:03d}"
            persona_key = personas[d % len(personas)]
            persona = PERSONAS[persona_key]

            # stagger driver starts
            trip_start = datetime(2025, 9, 9, 10, 0, 0, tzinfo=timezone.utc) + timedelta(
                minutes=30 * d
            )

            for t in range(trips_per_driver):
                duration = random.randint(10, 35)  # minutes
                for ping in _gen_trip_pings(driver_id, trip_start, duration, hz, persona, t):
                    # Validate against schema on the fly (catch bugs early)
                    try:
                        TelemetryPing.model_validate(ping)
                    except ValidationError as e:
                        raise RuntimeError(f"Bad ping: {e}") from e
                    f.write(json.dumps(ping) + "\n")
                # next trip after idle
                trip_start = trip_start + timedelta(minutes=duration + random.randint(6, 22))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drivers", type=int, default=3)
    parser.add_argument("--trips", type=int, default=10)
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/tmp/pings.ndjson")
    parser.add_argument("--golden", action="store_true", help="Also emit tiny golden sample")
    args = parser.parse_args()

    out = Path(args.out)
    generate_ndjson(out, args.drivers, args.trips, args.hz, args.seed)

    if args.golden:
        golden = Path("data/tmp/golden/pings_small.ndjson")
        golden.parent.mkdir(parents=True, exist_ok=True)
        generate_ndjson(golden, drivers=1, trips_per_driver=2, hz=1.0, seed=7)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
