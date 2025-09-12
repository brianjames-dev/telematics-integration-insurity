# src/serve/dashboard.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import re
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .scoring import get_scorer
from .pricing import price_from_risk
from .score_helpers import risk_score_from_ec100, exposure_weighted_avg

router = APIRouter(tags=["dashboard"], include_in_schema=False)
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Where the “real” features live
FEATURES_DIR = Path("data/trip_features")

# Expected trip feature columns used by the model
TRIP_FEATURE_COLUMNS = [
    "driver_id", "trip_id", "exposure_km",
    "night_ratio", "p95_over_limit_kph", "harsh_brakes_per100km",
    "phone_motion_ratio", "rain_flag", "mean_speed_kph", "crash_grid_index",
]

# --- Pricing config for the dashboard (tune freely) --------------------------
PRICING_CFG = {
    "lae_ratio": 0.10,         # 10% loss-adjustment expense
    "expense_ratio": 0.25,     # 25% fixed/variable expenses
    "target_margin": 0.05,     # 5% margin
    "min_premium": 900.0,      # demo-friendly annual floor ($75/mo)
    "max_change": 0.25,        # allow ±25% movement vs prior
    "ec_calibration": 12.0,    # *** key knob: scale down hot model ECs ***
}

def _estimate_annual_km(exposure_km: float, window_days: int) -> float:
    """Scale the observed exposure in the window to an annual estimate."""
    factor = 365.0 / max(1.0, float(window_days))
    # clip to sane bounds for demos (3k–30k km)
    return float(max(3000.0, min(30000.0, exposure_km * factor)))

def _prior_from_driver_score(score_0_100: float) -> float:
    """
    Map a 0–100 driver score to a prior annual premium.
    0  -> $1,200 (very safe), 100 -> $3,000 (very risky).
    """
    score = max(0.0, min(100.0, float(score_0_100)))
    prior = 1200.0 + (score / 100.0) * 1800.0
    # round to the nearest $10 just for nicer numbers
    return float(round(prior / 10.0) * 10.0)

# ---------- helpers: real data ----------
def _list_real_driver_ids(maxn: int = 200) -> List[str]:
    parts = list(FEATURES_DIR.rglob("*.parquet"))
    if not parts:
        return []
    dfs = [pd.read_parquet(p, columns=["driver_id"]).drop_duplicates() for p in parts]
    ids = pd.concat(dfs, ignore_index=True)["driver_id"].astype(str).unique().tolist()
    ids.sort()
    return ids[:maxn]

def _load_real_trips(driver_id: str, limit: int = 200) -> pd.DataFrame:
    parts = list(FEATURES_DIR.rglob("*.parquet"))
    if not parts:
        raise HTTPException(status_code=404, detail="No trip_features found. Run Phase 2.")
    dfs = [pd.read_parquet(p, columns=list({*TRIP_FEATURE_COLUMNS, "start_ts"})) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["driver_id"] == driver_id].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No trips for driver {driver_id}.")
    if "trip_id" in df.columns:
        df = df.sort_values("trip_id")
    return df.tail(limit)

# ---------- helpers: demo data ----------
# We’ll support either src/serve/demo_trips.csv (preferred) or data/demo_batch/scored_trips.csv (fallback)
_DEMO_PREFERRED = Path(__file__).with_name("demo_trips.csv")
_DEMO_FALLBACK = Path("data/demo_batch/scored_trips.csv")

def _demo_source_path() -> Path:
    if _DEMO_PREFERRED.exists():
        return _DEMO_PREFERRED
    if _DEMO_FALLBACK.exists():
        return _DEMO_FALLBACK
    raise HTTPException(status_code=500, detail="No demo CSV found. Expected src/serve/demo_trips.csv or data/demo_batch/scored_trips.csv.")

def _list_demo_driver_ids() -> List[str]:
    p = _demo_source_path()
    df = pd.read_csv(p, usecols=["driver_id"])
    ids = df["driver_id"].astype(str).unique().tolist()
    # sort “naturally”: SAFE_1, SAFE_2, SAFE_10 -> SAFE_1, SAFE_2, SAFE_10
    def natkey(s: str):
        m = re.match(r"^([A-Za-z]+)_?(\d+)$", s)
        return (m.group(1).upper(), int(m.group(2))) if m else (s.upper(), 0)
    ids.sort(key=natkey)
    return ids

def _load_demo_trips_df(driver_id: str, limit: int = 200) -> pd.DataFrame:
    p = _demo_source_path()
    df = pd.read_csv(p)
    if "driver_id" not in df.columns:
        raise HTTPException(status_code=500, detail=f"Demo CSV {p} missing 'driver_id' column.")
    df = df[df["driver_id"].astype(str) == str(driver_id)].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No demo trips for driver {driver_id}.")
    # keep only known feature columns if present
    cols = [c for c in TRIP_FEATURE_COLUMNS if c in df.columns]
    df = df[cols].copy()
    # sort by trip_id when available
    if "trip_id" in df.columns:
        df = df.sort_values("trip_id")
    # coerce numeric columns (skip id columns)
    numeric_cols = [c for c in cols if c not in ("driver_id", "trip_id")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.tail(limit)

# ---------- shared scoring ----------
def _score_trips(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], float]:
    """Return (trip_rows, exposure-weighted 0–100 driver score)."""
    scorer = get_scorer()
    out_rows: List[Dict[str, Any]] = []
    weights: List[Tuple[float, float]] = []  # (score, exposure_km)

    for _, r in df.iterrows():
        features = {k: r[k] for k in TRIP_FEATURE_COLUMNS if k in r}
        m = scorer.score_one_trip(features)
        p, sev = float(m["p_claim"]), float(m["sev_mean"])
        km = float(r.get("exposure_km", 0.0) or 0.0)
        ec_trip = p * sev
        ec100   = (ec_trip / max(km, 1e-6)) * 100.0
        score01 = risk_score_from_ec100(ec100)

        row = {k: (r[k] if k in r else None) for k in TRIP_FEATURE_COLUMNS}
        row.update({"p_claim": p, "sev_mean": sev, "ec100_trip": ec100, "risk_score_0_100": score01})
        out_rows.append(row)
        weights.append((score01, km))

    driver_score = exposure_weighted_avg(weights)
    return out_rows, driver_score

def _resolve_driver_id(requested: str | None, pool: List[str]) -> str:
    """Best-effort mapping: exact, case-insensitive, or prefix/number tolerant."""
    if not pool:
        return requested or ""
    if not requested:
        return pool[0]
    cand = requested.strip()
    if cand in pool:
        return cand
    # case-insensitive
    lower_map = {d.lower(): d for d in pool}
    if cand.lower() in lower_map:
        return lower_map[cand.lower()]
    # prefix + number, e.g. safe_1 -> SAFE_1 / SAFE_001
    m = re.match(r"^([A-Za-z]+)_?(\d+)$", cand)
    if m:
        pref, num = m.group(1).upper(), int(m.group(2))
        for w in (1, 2, 3):
            s = f"{pref}_{num:0{w}d}"
            if s in pool:
                return s
        s = f"{pref}_{num}"
        if s in pool:
            return s
    # prefix fallback
    for d in pool:
        if d.lower().startswith(cand.lower()):
            return d
    return pool[0]

# ---------- route ----------
@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    src: str = Query("real", pattern="^(real|demo)$"),
    driver: str | None = None,
    window_days: int = Query(365, ge=1, le=3650),
    limit: int = Query(200, ge=10, le=1000),
) -> HTMLResponse:

    # Build driver list based on source
    if src == "demo":
        drivers = _list_demo_driver_ids()
    else:
        drivers = _list_real_driver_ids()

    if not drivers:
        raise HTTPException(status_code=404, detail=f"No drivers available for source '{src}'.")

    # Pick a valid driver ID
    selected = _resolve_driver_id(driver, drivers)

    # Load trips & compute scores
    if src == "demo":
        df = _load_demo_trips_df(selected, limit=limit)
        trips, driver_score = _score_trips(df)
        # synthesize an 'agg' summary for the header
        exp = float(sum(float(t.get("exposure_km", 0.0) or 0.0) for t in trips))
        ec_vals = [float(t["ec100_trip"]) for t in trips if t.get("ec100_trip") is not None]
        agg = {
            "driver_id": selected,
            "n_trips": len(trips),
            "exposure_km": exp,
            "p_claim_weighted": 0.0,  # not used in the UI
            "sev_mean_weighted": 0.0, # not used in the UI
            "expected_cost_per100km": (sum(ec_vals) / len(ec_vals)) if ec_vals else 0.0,
        }
    else:  # real
        scorer = get_scorer()
        agg = scorer.aggregate_driver(selected, window_days=window_days)
        if agg["n_trips"] == 0:
            raise HTTPException(status_code=404, detail=f"No trips in window for driver {selected}.")
        df = _load_real_trips(selected, limit=limit)
        trips, driver_score = _score_trips(df)

    # pricing card
    annual_km_est = _estimate_annual_km(float(agg["exposure_km"]), window_days)
    prior_demo = _prior_from_driver_score(float(driver_score))

    pricing = price_from_risk(
        expected_cost_per100km=float(agg["expected_cost_per100km"]),
        annual_km=annual_km_est,
        lae_ratio=PRICING_CFG["lae_ratio"],
        expense_ratio=PRICING_CFG["expense_ratio"],
        target_margin=PRICING_CFG["target_margin"],
        min_premium=PRICING_CFG["min_premium"],
        prior_premium=prior_demo,
        max_change=PRICING_CFG["max_change"],
        ec_calibration=PRICING_CFG["ec_calibration"],  # <-- new!
    )

    # chart data (newest first visually looks nice; reverse if you want oldest first)
    labels = [str(t.get("trip_id", "")) for t in trips]
    scores = [float(t.get("risk_score_0_100", 0.0)) for t in trips]

    ctx = {
        "request": request,
        "src": src,
        "sources": ["real", "demo"],
        "drivers": drivers,
        "driver": selected,
        "driver_score": round(float(driver_score), 1),
        "agg": agg,
        "pricing": pricing,
        "trips": trips[::-1],
        "chart_labels": labels[::-1],
        "chart_scores": scores[::-1],
    }
    return TEMPLATES.TemplateResponse("driver.html", ctx)
