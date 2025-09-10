from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


# ------------- Thresholds (explicit, easy to tune) -------------
IDLE_GAP_SEC = 600         # 10 min gap starts a new trip
LOW_SPEED_MPS = 1.0        # "stopped" threshold
LOW_SPEED_IDLE_SEC = 180   # >=3 min continuous low speed ends a trip
MAX_TRIP_SEC = 4 * 3600    # hard cap 4 hours

BRAKE_MPS2 = -3.0
ACCEL_MPS2 = 2.5
CORNER_RAD_S = 0.35

PHONE_ACCEL = 1.5
PHONE_GYRO = 0.20

GRID_DEG = 0.01  # ~1.1km in lat; coarse but fine for POC


# ------------- Geo / hashing utilities -------------
def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)


def grid_cell(lat: float, lon: float, deg: float = GRID_DEG) -> str:
    return f"g_{int(np.floor(lat / deg))}_{int(np.floor(lon / deg))}"


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    R = 6371.0088
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------- Synthetic contextual layers (deterministic) -------------
def speed_limit_kph_for_cell(cell: str) -> float:
    """Return a stable synthetic speed limit per grid cell."""
    h = _stable_hash("speed:" + cell) % 3
    # ~35/45/65 mph ≈ 56/72/105 kph
    return [56.0, 72.0, 105.0][h]


def is_rain_cell_day(cell: str, day_key: str) -> int:
    """~20% rainy days per cell-day (stable)."""
    prob = (_stable_hash("rain:" + cell + ":" + day_key) % 100) / 100.0
    return int(prob < 0.2)


def risk_index(cell: str, kind: str) -> float:
    """Crash/theft index in [0,1], stable by cell+kind."""
    return ((_stable_hash(f"{kind}:{cell}") % 1000) / 1000.0)


# ------------- Event flags / phone motion -------------
def flag_harsh_brake(accel_mps2: np.ndarray) -> np.ndarray:
    return (accel_mps2 <= BRAKE_MPS2).astype(int)

def flag_harsh_accel(accel_mps2: np.ndarray) -> np.ndarray:
    return (accel_mps2 >= ACCEL_MPS2).astype(int)

def flag_corner(gyro_z: np.ndarray) -> np.ndarray:
    return (np.abs(gyro_z) >= CORNER_RAD_S).astype(int)

def phone_motion_mask(accel_mps2: np.ndarray, gyro_z: np.ndarray) -> np.ndarray:
    return ((np.abs(accel_mps2) > PHONE_ACCEL) | (np.abs(gyro_z) > PHONE_GYRO)).astype(int)


# ------------- Utilities -------------
def per100km(count: float, km: float) -> float:
    if km <= 1e-6:
        return 0.0
    return (count * 100.0) / km


def night_mask(ts: pd.Series) -> np.ndarray:
    # treat timestamps as UTC; night = 22..23 or 0..4
    h = ts.dt.hour.values
    return ((h >= 22) | (h <= 4)).astype(int)
