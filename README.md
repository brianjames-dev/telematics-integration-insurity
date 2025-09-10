# Insurity Telematics UBI – Proof of Concept

## Overview

This repository contains a proof-of-concept (POC) system for **usage-based auto insurance (UBI)** using telematics data.

The system demonstrates:

- Real-time ingestion of simulated telematics pings.
- Trip/session aggregation into insurer-friendly features.
- Risk scoring via GLM (baseline) and GBM (advanced).
- Pricing engine with guardrails and smoothing.
- APIs for scoring and pricing.
- Streamlit dashboard for drivers and insurers.
- Documentation on data, models, guardrails, and compliance.

---

## Repo Structure

.
├── src
│ ├── simulator/ # generate synthetic pings & trips
│ ├── ingest/ # ingestion service (asyncio; kafka optional)
│ ├── processing/ # trip aggregator, feature definitions
│ ├── ml/ # train_glm.py, train_gbm.py, calibrate.py
│ ├── serve/ # FastAPI: api.py, pricing.py, schemas.py
│ └── ui/ # Streamlit dashboard
├── models/ # saved model weights, calibration files
├── data/ # raw pings, trips, trip_features, driver_daily
├── docs/ # design_doc.md, data_dictionary.md, model_card.md, pricing_guardrails.md
├── bin/ # helper scripts (make_demo.sh, run_api.sh, run_ui.sh)
├── docker-compose.yml
├── docker-compose.kafka.yml # optional profile
└── README.md

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker & docker-compose (optional for running services)
- Recommended: virtual environment (`venv` or `conda`)

### Install dependencies

```bash
pip install -r requirements.txt
```

# 1. Generate synthetic trips

python src/simulator/generate_trips.py

# 2. Train models (GLM + GBM)

python src/ml/train_glm.py
python src/ml/train_gbm.py

# 3. Start API

python src/serve/api.py

# 4. Launch dashboard

streamlit run src/ui/streamlit_app.py

bash bin/make_demo.sh

---

## API Endpoints (FastAPI)

- **POST /scoreTrip** → risk score + top feature explanations
- **POST /price** → premium, multiplier, guardrail info
- **GET /healthz** → service health check

See `src/serve/schemas.py` for request/response formats.

---

## Dashboard (Streamlit)

### Driver View

- Current premium multiplier & trend
- Last 7 trips with key metrics
- Top 3 contributing factors + improvement tips

### Insurer View

- Calibration curve
- Decile lift chart
- Multiplier distribution
- Guardrail hit rates

---

## Documentation

- [Design Document](./docs/design_doc.md) – architecture, data flow, roadmap
- [Data Dictionary](./docs/data_dictionary.md) – schemas and feature definitions
- [Model Card](./docs/model_card.md) – models, features, metrics, limitations
- [Pricing Guardrails](./docs/pricing_guardrails.md) – premium smoothing & caps

---

## Tech Stack

- **Language**: Python 3.11
- **Data**: Parquet + DuckDB (local query)
- **Streaming**: asyncio (lean) or Kafka (optional profile)
- **ML**: statsmodels (GLM), LightGBM/XGBoost (GBM), scikit-learn (calibration)
- **Serving**: FastAPI
- **UI**: Streamlit
- **Orchestration**: docker-compose

---

## Limitations

- Labels simulated; not calibrated to real claims
- No production-grade fraud/tamper detection
- No protected attributes, but location/time may proxy risk factors

---

## Roadmap

1. Integrate real claims data for training/validation
2. Add fraud detection via anomaly scoring on raw pings
3. Expand context joins with live weather/traffic APIs
4. Deploy as containerized microservices with Kafka pipelines
5. Perform fairness audits across driver subgroups

---

## License

[MIT License](LICENSE) (placeholder – update as needed)
