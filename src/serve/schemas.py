from __future__ import annotations
from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# --------- Raw Telemetry Ping (ingest contract) ---------
class TelemetryPing(BaseModel):
    driver_id: str = Field(..., description="Pseudonymous driver ID")
    trip_hint_id: Optional[str] = Field(
        None, description="Optional rolling trip hint from device"
    )
    ts: datetime = Field(..., description="UTC timestamp")
    lat: float
    lon: float
    gps_acc_m: float = Field(..., ge=0.0)
    speed_mps: float = Field(..., ge=0.0)
    accel_mps2: float
    gyro_z: float
    source: Literal["phone", "obd"] = "phone"
    ingest_id: UUID = Field(default_factory=uuid4)
    schema_version: int = 1

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": "D_001",
                "trip_hint_id": "H20250909_0001",
                "ts": "2025-09-09T10:03:02.120Z",
                "lat": 37.7749,
                "lon": -122.4194,
                "gps_acc_m": 4.0,
                "speed_mps": 12.3,
                "accel_mps2": -0.3,
                "gyro_z": 0.05,
                "source": "phone",
                "ingest_id": "9f1a1f4e-b2a5-4bc1-9c8e-0a9b3c35d6c1",
                "schema_version": 1,
            }
        }
    }


# --------- Trip Features (canonical for ML/pricing) ---------
class TripFeatures(BaseModel):
    driver_id: str
    trip_id: str
    start_ts: datetime
    end_ts: datetime
    exposure_km: float = Field(..., ge=0)
    duration_min: float = Field(..., ge=0)

    night_ratio: float = Field(..., ge=0, le=1)
    urban_km: float = Field(..., ge=0)
    highway_km: float = Field(..., ge=0)

    mean_speed_kph: float = Field(..., ge=0)
    p95_over_limit_kph: float = Field(..., ge=0)
    pct_time_over_limit_10: float = Field(..., ge=0, le=1)

    harsh_brakes_per100km: float = Field(..., ge=0)
    harsh_accels_per100km: float = Field(..., ge=0)
    cornering_per100km: float = Field(..., ge=0)

    phone_motion_ratio: float = Field(..., ge=0, le=1)
    rain_flag: int = Field(..., ge=0, le=1)
    crash_grid_index: float = Field(..., ge=0)
    theft_grid_index: float = Field(..., ge=0)

    feature_version: int = 1

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": "D_001",
                "trip_id": "T_20250909_0001",
                "start_ts": "2025-09-09T10:01:02Z",
                "end_ts": "2025-09-09T10:31:44Z",
                "exposure_km": 14.2,
                "duration_min": 30.7,
                "night_ratio": 0.18,
                "urban_km": 8.2,
                "highway_km": 6.0,
                "mean_speed_kph": 38.1,
                "p95_over_limit_kph": 9.7,
                "pct_time_over_limit_10": 0.06,
                "harsh_brakes_per100km": 1.4,
                "harsh_accels_per100km": 0.8,
                "cornering_per100km": 0.0,
                "phone_motion_ratio": 0.03,
                "rain_flag": 0,
                "crash_grid_index": 0.3,
                "theft_grid_index": 0.1,
                "feature_version": 1,
            }
        }
    }


# --------- API contracts ---------
class FeatureContrib(BaseModel):
    feature: str
    contrib: float  # positive raises risk, negative lowers


class ScoreTripResponse(BaseModel):
    risk_score: float
    explain: List[FeatureContrib] = []


class PolicySummary(BaseModel):
    base_rate: float
    prev_mult: float
    prev_ewma: float
    exposure_to_date_km: float
    trips_to_date: int


class PeriodFeatures(BaseModel):
    driver_id: str
    period_exposure_km: float
    mean_risk_score: float


class PriceRequest(BaseModel):
    policy: PolicySummary
    period_features: PeriodFeatures


class PriceResponse(BaseModel):
    premium: float
    multiplier: float
    guardrails_applied: bool
    explain: List[FeatureContrib] = []
