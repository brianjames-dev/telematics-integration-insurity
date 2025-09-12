# Model Card – Telematics Risk Scoring (v1)

## Model Overview

- **Name**: Telematics Risk Scoring v1
- **Developers**: Brian James
- **Type**: Hybrid approach – GLM baseline, GBM advanced
- **Objective**: Predict expected claim frequency/severity from trip-level aggregates for usage-based insurance (UBI) pricing.

---

## Intended Use

- **Primary**: Support auto insurance premium calculation by integrating telematics-based risk scoring into pricing engine.
- **Users**: Actuaries, data scientists, pricing analysts.
- **Scope**: Proof-of-concept, simulated labels, not for production deployment without real claims data.

---

## Data

- **Input Sources**:
  - Raw telemetry pings (speed, accel, gyro, GPS) – retained short-term.
  - Trip aggregates (canonical features).
  - Contextual joins: weather flag, crash/theft indices.
- **Synthetic Labels**:
  - Frequency simulated via Poisson process with logistic base risk.
  - Severity simulated via Lognormal distribution conditional on claim.
- **Bias Considerations**:
  - No protected attributes (gender, race) used.
  - Geography/time may proxy demographics; mitigated with coarse grids and clipping.

---

## Features (v1)

- `harsh_brakes_per100km`
- `harsh_accels_per100km`
- `cornering_per100km`
- `night_ratio`
- `p95_over_limit_kph`, `pct_time_over_limit_10`
- `crash_grid_index`, `theft_grid_index`
- `rain_flag`
- `urban_km`, `highway_km`
- `phone_motion_ratio`
- Normalized per 100 km; clipped at p99.

---

## Models

### GLM Baseline

- **Frequency**: Poisson GLM with log link, offset = log(km).
- **Severity**: Gamma GLM with log link on claims only.
- **Pure premium**: freq_hat × sev_hat.

### GBM Advanced

- **Library**: LightGBM/XGBoost.
- **Target**: pure premium (continuous) or two-stage (probability + severity).
- **Constraints**: Monotonicity enforced for clear risk features.
- **Calibration**: Isotonic regression / Platt scaling.

---

## Metrics

- **Frequency**: AUC, PR-AUC, logloss, calibration curves.
- **Severity**: RMSLE, Gamma deviance.
- **Pure Premium**: MAE, MAPE, lift by deciles, Gini.
- **Validation**: Time-based splits; ablation studies.

---

## Explainability

- **GLM**: Coefficients, sign and magnitude of effects.
- **GBM**: SHAP values (global + per driver), partial dependence plots.
- **API**: `/scoreTrip` and `/price` return factor breakdowns for transparency.

---

## Limitations

- Labels are simulated; calibration is relative, not absolute.
- Raw GPS/telemetry are not retained long-term.
- Does not account for fraud/tamper detection beyond heuristics.
- Requires regulatory review before production.

---

## Ethical Considerations

- **Privacy**: Pseudonymized IDs, coarse grids, data retention limits.
- **Fairness**: No protected attributes; monitor proxy features.
- **Transparency**: Clear factor breakdown in dashboard.

---

## Future Work

- Incorporate real claim labels.
- Sequence models (LSTM/Transformer) for direct event detection.
- Additional context (traffic, weather APIs).
- Fairness audits across demographic/geographic subgroups.
