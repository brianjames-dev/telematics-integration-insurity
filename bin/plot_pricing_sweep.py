#!/usr/bin/env python
import sys, json, pathlib, numpy as np
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.serve.pricing import price_from_risk

def sweep_ec():
    prior, max_chg, min_p = 2400.0, 0.15, 1500.0
    xs = np.linspace(0, 30, 121)            # expected cost / 100km
    raw, after, final = [], [], []
    for ec in xs:
        out = price_from_risk(
            expected_cost_per100km=float(ec),
            annual_km=12_000,               # fixed km
            lae_ratio=0.10, expense_ratio=0.25, target_margin=0.05,
            min_premium=min_p, prior_premium=prior, max_change=max_chg,
        )
        raw.append(out["raw_premium"]); after.append(min(max(out["raw_premium"], out["bounds"]["cap_lower"]), out["bounds"]["cap_upper"]))
        final.append(out["premium"])
    return xs, np.array(raw), np.array(after), np.array(final), dict(min=min_p, lo=prior*(1-max_chg), hi=prior*(1+max_chg))

def sweep_km():
    prior, max_chg, min_p = 2400.0, 0.15, 1500.0
    xs = np.linspace(0, 30000, 121)         # annual km
    raw, after, final = [], [], []
    for km in xs:
        out = price_from_risk(
            expected_cost_per100km=12.0,    # fixed EC
            annual_km=float(km),
            lae_ratio=0.10, expense_ratio=0.25, target_margin=0.05,
            min_premium=min_p, prior_premium=prior, max_change=max_chg,
        )
        raw.append(out["raw_premium"]); after.append(min(max(out["raw_premium"], out["bounds"]["cap_lower"]), out["bounds"]["cap_upper"]))
        final.append(out["premium"])
    return xs, np.array(raw), np.array(after), np.array(final), dict(min=min_p, lo=prior*(1-max_chg), hi=prior*(1+max_chg))

def main(mode="ec"):
    if mode == "km":
        xs, raw, after, fin, b = sweep_km()
        xlabel = "Annual km"
        outpng = "models/pricing_sweep_km.png"
    else:
        xs, raw, after, fin, b = sweep_ec()
        xlabel = "Expected cost per 100 km"
        outpng = "models/pricing_sweep_ec.png"

    plt.figure(figsize=(11,7))
    plt.plot(xs, raw, label="Raw premium")
    plt.plot(xs, after, label="After prior caps")
    plt.plot(xs, fin, label="Final (with min floor)")
    for y, lbl in ((b["min"], "min premium floor"), (b["lo"], "cap lower"), (b["hi"], "cap upper")):
        plt.axhline(y, ls="--", alpha=0.4); plt.text(xs.max()*0.98, y+15, lbl, ha="right", va="bottom", fontsize=10)
    plt.ylim(-50, max(fin.max(), b["hi"])*1.05)
    plt.xlabel(xlabel); plt.ylabel("Premium"); plt.title(f"Pricing vs {xlabel.lower()}")
    plt.legend(); pathlib.Path("models").mkdir(exist_ok=True); plt.savefig(outpng, bbox_inches="tight")
    print("wrote", outpng)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ec"
    main(mode)
