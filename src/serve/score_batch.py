# src/serve/score_batch.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from .scoring import get_scorer
from .score_helpers import risk_score_from_ec100

REQ_COLS = [
    "driver_id","trip_id","exposure_km","night_ratio","p95_over_limit_kph",
    "harsh_brakes_per100km","phone_motion_ratio","rain_flag",
    "mean_speed_kph","crash_grid_index",
]

def score_df(df: pd.DataFrame) -> pd.DataFrame:
    scorer = get_scorer()
    outs = []
    for _, r in df.iterrows():
        row = {k: r[k] for k in REQ_COLS if k in r}
        o = scorer.score_one_trip(row)
        p = float(o["p_claim"]); sev = float(o["sev_mean"])
        exp = float(row["exposure_km"])
        ec_trip = p * sev
        ec100 = (ec_trip / max(exp, 1e-6)) * 100.0
        outs.append({
            **row,
            "p_claim": p,
            "sev_mean": sev,
            "ec100_trip": ec_trip,
            "ec100": ec100,
            "risk_0_100": float(risk_score_from_ec100(ec100)),
        })
    return pd.DataFrame(outs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="CSV of trips to score")
    ap.add_argument("--out", dest="out", required=True, help="CSV to write scored rows")
    ap.add_argument("--driver_out", dest="driver_out", default="", help="Optional per-driver summary csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    for c in REQ_COLS:
        if c not in df.columns:
            raise SystemExit(f"Missing column in input: {c}")

    scored = score_df(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.out, index=False)
    print(f"[score_batch] wrote {args.out} rows={len(scored)}")

    if args.driver_out:
        grp = scored.groupby("driver_id").apply(
            lambda g: pd.Series({
                "n_trips": len(g),
                "exposure_km": g["exposure_km"].sum(),
                "risk_avg": np.average(g["risk_0_100"], weights=g["exposure_km"]),
                "ec100_avg": np.average(g["ec100"],   weights=g["exposure_km"]),
            })
        ).reset_index()
        Path(args.driver_out).parent.mkdir(parents=True, exist_ok=True)
        grp.to_csv(args.driver_out, index=False)
        print(f"[score_batch] wrote {args.driver_out} rows={len(grp)}")

if __name__ == "__main__":
    main()
