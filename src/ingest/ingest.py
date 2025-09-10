from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import pandas as pd


REQUIRED_COLS = [
    "driver_id", "ts", "lat", "lon", "gps_acc_m",
    "speed_mps", "accel_mps2", "gyro_z", "source", "ingest_id", "schema_version"
]


def _read_ndjson(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = pd.DataFrame(rows)
    # basic schema normalization
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # parse timestamp + derive dt partition
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["dt"] = df["ts"].dt.strftime("%Y-%m-%d")
    return df


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer (driver_id, ts); if ingest_id duplicates too, last wins (stable)
    if "ingest_id" in df.columns:
        df = df.sort_values(["driver_id", "ts", "ingest_id"])
    else:
        df = df.sort_values(["driver_id", "ts"])
    df = df.drop_duplicates(subset=["driver_id", "ts"], keep="last")
    return df


def write_partitioned_parquet(df: pd.DataFrame, out_dir: str = "data/pings") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for (driver_id, dt), g in df.groupby(["driver_id", "dt"]):
        part_dir = out / f"driver_id={driver_id}" / f"dt={dt}"
        part_dir.mkdir(parents=True, exist_ok=True)
        fname = f"part-{uuid4().hex}.parquet"
        (part_dir / fname).resolve()
        g.drop(columns=["dt"], inplace=True)
        g.to_parquet(part_dir / fname, index=False)
        total += len(g)
    return total


def process(input_glob: str, out_dir: str = "data/pings") -> int:
    paths = sorted(glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")
    df = _read_ndjson(paths)
    before = len(df)
    df = _dedup(df)
    written = write_partitioned_parquet(df, out_dir)
    print(f"Ingested {before} → wrote {written} rows to {out_dir}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/tmp/*.ndjson")
    ap.add_argument("--out", type=str, default="data/pings")
    args = ap.parse_args()
    process(args.input, args.out)


if __name__ == "__main__":
    main()
