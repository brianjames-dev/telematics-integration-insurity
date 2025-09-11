# src/serve/pricing.py

from __future__ import annotations
from typing import Optional, Dict, Any

def price_from_risk(
    *,
    expected_cost_per100km: float,
    annual_km: float,
    lae_ratio: float,
    expense_ratio: float,
    target_margin: float,
    min_premium: float,
    prior_premium: Optional[float] = None,
    max_change: float = 0.15,
) -> Dict[str, Any]:
    """
    Simple premium builder:
      1) base_loss = expected_cost_per100km * (annual_km / 100)
      2) LAE & expense as % of base_loss
      3) indicated (raw) premium p_raw = (base_loss + lae + expense) / (1 - target_margin)
      4) optional caps around prior_premium: [prior*(1-max_change), prior*(1+max_change)]
      5) floor at min_premium
    """
    ec_100 = float(expected_cost_per100km)
    km_year = float(annual_km)

    # 1) Base loss and loads
    base_loss = ec_100 * (km_year / 100.0)
    lae = base_loss * float(lae_ratio)
    expense = base_loss * float(expense_ratio)

    # 3) Indicated premium before caps/floor
    denom = max(1e-9, 1.0 - float(target_margin))
    p_raw = (base_loss + lae + expense) / denom

    # 4) Apply caps if prior provided
    prior = float(prior_premium) if prior_premium is not None else None
    cap_lower = cap_upper = None
    if prior is not None:
        cap_lower = prior * (1.0 - float(max_change))
        cap_upper = prior * (1.0 + float(max_change))
        p_capped = min(max(p_raw, cap_lower), cap_upper)
    else:
        p_capped = p_raw

    # 5) Apply minimum floor
    premium_out = max(p_capped, float(min_premium))

    # Margin dollars at the final premium (may be negative if capped/floored)
    margin_dollars = premium_out - (base_loss + lae + expense)

    # Cap reason for explainability
    cap_reason = (
        "upper" if (cap_upper is not None and abs(p_capped - cap_upper) < 1e-9)
        else "lower" if (cap_lower is not None and abs(p_capped - cap_lower) < 1e-9)
        else "floor" if (abs(premium_out - float(min_premium)) < 1e-9)
        else None
    )

    return {
        "premium": float(round(premium_out, 2)),
        "raw_premium": float(round(p_raw, 2)),   # before caps/floor
        "cap_reason": cap_reason,                # 'upper' | 'lower' | 'floor' | None
        "loads": {
            "lae": float(round(lae, 2)),
            "expense": float(round(expense, 2)),
            "margin": float(round(margin_dollars, 2)),
        },
        "bounds": {
            "min_premium": float(min_premium),
            "cap_lower": float(cap_lower) if cap_lower is not None else None,
            "cap_upper": float(cap_upper) if cap_upper is not None else None,
        },
        "inputs": {
            "expected_cost_per100km": float(ec_100),
            "annual_km": float(km_year),
            "lae_ratio": float(lae_ratio),
            "expense_ratio": float(expense_ratio),
            "target_margin": float(target_margin),
            "prior_premium": float(prior) if prior is not None else None,
            "max_change": float(max_change),
        },
    }
