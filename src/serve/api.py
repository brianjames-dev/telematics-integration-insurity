# src/serve/api.py
from __future__ import annotations
from typing import Optional, Dict, Any, Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .scoring import get_scorer
from .pricing import price_from_risk

app = FastAPI(title="Telematics UBI API", version="0.1")

# ---------- Schemas ----------
FloatGE0 = Annotated[float, Field(ge=0)]
FloatGT0 = Annotated[float, Field(gt=0)]
Float01  = Annotated[float, Field(ge=0, le=1)]
Int01    = Annotated[int,   Field(ge=0, le=1)]

class TripFeatureIn(BaseModel):
    driver_id: str
    trip_id: Optional[str] = None
    exposure_km: FloatGE0 = Field(..., description="Distance for this trip")
    night_ratio: Float01
    p95_over_limit_kph: FloatGE0
    harsh_brakes_per100km: FloatGE0
    phone_motion_ratio: Float01
    rain_flag: Int01
    mean_speed_kph: FloatGE0
    crash_grid_index: Float01

class TripScoreOut(BaseModel):
    p_claim: float
    sev_mean: float
    expected_cost_per100km: float

class DriverScoreOut(BaseModel):
    driver_id: str
    n_trips: int
    exposure_km: float
    p_claim_weighted: float
    sev_mean_weighted: float
    expected_cost_per100km: float

class QuoteIn(BaseModel):
    driver_id: str
    window_days: Annotated[int, Field(ge=1, le=365)] = 30
    annual_km: FloatGT0 = 12000
    lae_ratio: Annotated[float, Field(ge=0, le=0.5)] = 0.10
    expense_ratio: Annotated[float, Field(ge=0, le=0.6)] = 0.25
    target_margin: Annotated[float, Field(ge=0, le=0.3)] = 0.05
    min_premium: FloatGE0 = 300.0
    prior_premium: Optional[FloatGT0] = None
    max_change: Annotated[float, Field(ge=0, le=0.5)] = 0.15

class QuoteOut(BaseModel):
    driver: DriverScoreOut
    # Important: pricing is a nested structure (premium, loads, bounds, inputs, ...)
    # Make this a generic dict unless you define full submodels.
    pricing: Dict[str, Any]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/score/trip", response_model=TripScoreOut)
def score_trip(payload: TripFeatureIn):
    scorer = get_scorer()
    out = scorer.score_one_trip(payload.model_dump())
    return out

@app.get("/score/driver/{driver_id}", response_model=DriverScoreOut)
def score_driver(
    driver_id: str,
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
):
    scorer = get_scorer()
    out = scorer.aggregate_driver(driver_id, window_days=window_days)
    if out["n_trips"] == 0:
        raise HTTPException(status_code=404, detail="No trips in window for driver")
    return out

@app.post("/price/quote", response_model=QuoteOut)
def price_quote(q: QuoteIn):
    scorer = get_scorer()
    d = scorer.aggregate_driver(q.driver_id, window_days=q.window_days)
    if d["n_trips"] == 0:
        raise HTTPException(status_code=404, detail="No trips in window for driver")
    pricing = price_from_risk(
        expected_cost_per100km=d["expected_cost_per100km"],
        annual_km=q.annual_km,
        lae_ratio=q.lae_ratio,
        expense_ratio=q.expense_ratio,
        target_margin=q.target_margin,
        min_premium=q.min_premium,
        prior_premium=q.prior_premium,
        max_change=q.max_change,
    )
    return {"driver": d, "pricing": pricing}
