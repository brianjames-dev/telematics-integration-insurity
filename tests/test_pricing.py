# tests/test_pricing.py
import math
import pytest
from src.serve.pricing import price_from_risk

def _p(**kw):
    # sensible defaults; override per test
    base = dict(
        expected_cost_per100km=15.0,
        annual_km=12_000.0,
        lae_ratio=0.10,
        expense_ratio=0.25,
        target_margin=0.05,
        min_premium=300.0,
        prior_premium=None,  # no caps by default
        max_change=0.15,
    )
    base.update(kw)
    return price_from_risk(**base)

def test_min_floor_applies_without_prior():
    # Make the indicated premium tiny by using small EC and km
    out = _p(expected_cost_per100km=0.1, annual_km=1000, prior_premium=None)
    assert out["raw_premium"] < 300
    assert out["premium"] == 300.0
    assert out["cap_reason"] == "floor"

def test_upper_cap_applies__realistic_band():
    # Prior $2,400, ±15% -> cap_upper=$2,760
    out = _p(
        expected_cost_per100km=60.0,
        annual_km=15_000,
        min_premium=1500.0,
        prior_premium=2400.0,
        max_change=0.15,
    )
    assert math.isclose(out["bounds"]["cap_upper"], 2400.0 * 1.15, abs_tol=1e-6)
    assert math.isclose(out["premium"], out["bounds"]["cap_upper"], abs_tol=1e-6)
    assert out["cap_reason"] == "upper"

def test_lower_cap_applies__realistic_band():
    # Prior $2,400, ±15% -> cap_lower=$2,040 (and above min)
    out = _p(
        expected_cost_per100km=5.0,
        annual_km=5_000,
        min_premium=1000.0,
        prior_premium=2400.0,
        max_change=0.15,
    )
    assert math.isclose(out["bounds"]["cap_lower"], 2400.0 * 0.85, abs_tol=1e-6)
    assert math.isclose(out["premium"], out["bounds"]["cap_lower"], abs_tol=1e-6)
    assert out["cap_reason"] == "lower"

def test_no_prior_no_floor_equals_raw():
    # Use big EC so raw >> min; with no prior, premium == raw
    out = _p(expected_cost_per100km=50.0, annual_km=20_000, min_premium=0.0, prior_premium=None)
    assert out["premium"] == pytest.approx(out["raw_premium"])
    assert out["cap_reason"] is None

def test_monotone_wrt_ec_and_km_without_caps():
    a = _p(expected_cost_per100km=10.0, annual_km=10_000, min_premium=0.0, prior_premium=None)
    b = _p(expected_cost_per100km=12.0, annual_km=10_000, min_premium=0.0, prior_premium=None)
    c = _p(expected_cost_per100km=12.0, annual_km=12_000, min_premium=0.0, prior_premium=None)
    assert b["premium"] > a["premium"]  # EC↑ ⇒ premium↑
    assert c["premium"] > b["premium"]  # km↑ ⇒ premium↑

def test_output_shapes_and_types():
    out = _p()
    # core keys
    for k in ("premium", "raw_premium", "cap_reason", "loads", "bounds", "inputs"):
        assert k in out
    assert isinstance(out["premium"], float)
    assert isinstance(out["raw_premium"], float)
    assert set(out["loads"].keys()) == {"lae", "expense", "margin"}
    assert set(out["bounds"].keys()) == {"min_premium", "cap_lower", "cap_upper"}
    # inputs echoing
    assert out["inputs"]["expected_cost_per100km"] > 0
