# src/serve/score_helpers.py
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np

# Piecewise anchors: (EC/100km -> risk score)
# Tune these once you see typical EC values in your data.
# Example mapping:
#   $0 -> 0, $50 -> 25, $100 -> 50, $200 -> 75, $400+ -> 100
_RISK_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.0),
    (50.0, 25.0),
    (100.0, 50.0),
    (200.0, 75.0),
    (400.0, 100.0),
)

def risk_score_from_ec100(ec100: float) -> float:
    xs = np.array([a[0] for a in _RISK_ANCHORS], dtype=float)
    ys = np.array([a[1] for a in _RISK_ANCHORS], dtype=float)
    ec = float(max(xs.min(), min(xs.max(), ec100)))  # clamp to [min,max]
    score = np.interp(ec, xs, ys)
    return float(max(0.0, min(100.0, score)))

def exposure_weighted_avg(pairs: Iterable[Tuple[float, float]]) -> float:
    """pairs = [(score, exposure_km), ...]"""
    num = 0.0
    den = 0.0
    for s, w in pairs:
        num += float(s) * float(max(w, 0.0))
        den += float(max(w, 0.0))
    return float(num / den) if den > 0 else 0.0
