#!/usr/bin/env python
import sys, pathlib, pandas as pd, numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serve.scoring import get_scorer
from src.serve.pricing import price_from_risk

def main(days=30, annual_km=12_000, prior=2400.0):
    sc = get_scorer()
    # who to score: all drivers seen in features
    parts = list(pathlib.Path("data/trip_features").rglob("*.parquet"))
    assert parts, "run Phase 2 first"
    df = pd.concat([pd.read_parquet(p)[["driver_id"]] for p in parts]).drop_duplicates()

    rows = []
    for d in df["driver_id"]:
        agg = sc.aggregate_driver(d, window_days=days)
        if agg["n_trips"] == 0: 
            continue
        price = price_from_risk(
            expected_cost_per100km=agg["expected_cost_per100km"],
            annual_km=annual_km,
            lae_ratio=0.10, expense_ratio=0.25, target_margin=0.05,
            min_premium=1500.0, prior_premium=prior, max_change=0.15,
        )
        rows.append({
            "driver_id": d,
            "n_trips": agg["n_trips"],
            "exposure_km": agg["exposure_km"],
            "p_claim_weighted": agg["p_claim_weighted"],
            "sev_mean_weighted": agg["sev_mean_weighted"],
            "expected_cost_per100km": agg["expected_cost_per100km"],
            "premium": price["premium"],
            "cap_reason": price["cap_reason"],
        })
    out = pathlib.Path("models/batch_scores.csv")
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} rows={len(rows)}")

if __name__ == "__main__":
    main()
