# Pricing Guardrails – Telematics UBI (v1)

## Purpose

Define rules to ensure premiums remain fair, stable, and compliant with regulatory requirements while incorporating telematics risk scores.

---

## Base Premium

- Derived from traditional factors (age, vehicle, territory, prior claims).
- Telematics applies a **multiplier** on top of base rate.

---

## Risk Multiplier Calculation

- Input = standardized risk score from GLM/GBM model.
- Formula: `mult = 1 + α * risk_score` (α = 0.5 default).
- Multiplier clamped: **0.60 ≤ mult ≤ 1.80**.

---

## Smoothing

- Exponential weighted moving average (EWMA) applied:
  - `EWMA = 0.7 * prev + 0.3 * current`
- Prevents volatility due to single trip anomalies.

---

## Guardrails

- **Monthly cap**: premium change limited to ±10%.
- **Annual cap**: premium change limited to ±25%.
- **Activation rule**: require ≥300 km or ≥10 trips before telematics pricing applies.
- **Minimum premium**: no lower than regulatory/state minimum.

---

## Transparency

- Factor breakdown must be shown to driver:
  - Positive contributors (e.g., harsh braking +3%).
  - Negative contributors (e.g., highway driving −2%).
- Dashboard and API responses include top feature impacts.

---

## Compliance Considerations

- Guardrails and formulas must be documented in any rate filing.
- GLM baseline available for regulator review.
- GBM advanced model accompanied by monotonic constraints and calibration documentation.

---

## Monitoring

- Track % of drivers hitting caps.
- Track calibration drift quarterly.
- Track distribution of multipliers (should cluster near 1.0).

---

## Future Enhancements

- Driver reward programs (gamification) integrated on top of multiplier.
- Contextual risk adjustments (e.g., storm events) as add-ons, with clear guardrails.
