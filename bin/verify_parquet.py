#!/usr/bin/env python
import argparse, pathlib, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/pings", help="Base path to parquet partitions")
    args = ap.parse_args()
    parts = list(pathlib.Path(args.path).rglob("*.parquet"))
    if not parts:
        raise SystemExit(f"No parquet files under {args.path}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    uniq = df.drop_duplicates(subset=["driver_id","ts"]).shape[0]
    print(f"[VERIFY] rows={len(df)}  unique(driver_id,ts)={uniq}  drivers={df['driver_id'].nunique()}")
    print(df.head(5)[["driver_id","ts","speed_mps","accel_mps2","gps_acc_m"]])
    if uniq != len(df):
        raise SystemExit("Dedup failed — duplicate (driver_id, ts) still present")

if __name__ == "__main__":
    main()
