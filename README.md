# Insurity Telematics UBI – Proof of Concept

## Overview

This proof-of-concept (POC) demonstrates a usage-based insurance (UBI) workflow on synthetic telematics data:

- Real-time(ish) ingestion of simulated pings ➝ trip aggregation ➝ feature generation
- Risk scoring using a baseline GLM (optional GBM scripts are included but not required)
- Pricing engine with guardrails & smoothing
- FastAPI service for scoring/pricing + Streamlit dashboard to explore results

---

## Repo Structure

.
├── src
│ ├── simulator/ # synthetic pings & trips generator
│ ├── ingest/ # asyncio-based ingestion (Kafka optional; not required)
│ ├── processing/ # trip aggregation, feature engineering
│ ├── ml/ # train_glm.py (+ optional train_gbm.py), calibrate.py
│ ├── serve/ # FastAPI app: api.py, pricing.py, schemas.py
│ └── ui/ # Streamlit dashboard
├── models/ # trained GLM weights / calibration artifacts
├── data/ # generated: pings, trips, features, per-driver summaries
├── docs/ # design/documents (design_doc.md, model_card.md, etc.)
├── bin/ # helper scripts (optional)
├── docker-compose.yml # optional (not required for local demo)
└── README.md

---

## Quickstart

Run the full demo locally with Docker in **two commands**.

## Prerequisites

- **Docker Desktop** (includes Docker Compose v2)
  - macOS: https://docs.docker.com/desktop/setup/install/mac-install/
  - Windows (with **WSL2** backend): https://docs.docker.com/desktop/features/wsl/
  - Product page (all platforms): https://www.docker.com/products/docker-desktop/

> Windows users: run commands in **WSL2 (Ubuntu)** or **Git Bash** so `bash` scripts work.

## Quick Start (run commands from the project root)

```bash
docker compose build
bash bin/run_demo.sh
```

---

## API Endpoints (FastAPI)

- **POST /scoreTrip** → risk score + top feature explanations
- **POST /price** → premium, multiplier, guardrail info
- **GET /healthz** → service health check

See `src/serve/schemas.py` for request/response formats.

---

## Notes on Components Used

- Ingestion: runs with asyncio + local simulator by default.
- Modeling: the GLM is sufficient to run end-to-end; GBM artifacts are optional and not needed for the quickstart.
- Docker: You can containerize with docker-compose.yml, but you can also run scripts to gain the same result.

## Troubleshooting

- Python version: ensure 3.11+; mismatched versions can cause dependency errors.
- Ports: if 8000 or 8501 are in use, set alternate ports.
- Permissions on Windows: run PowerShell as admin if activation scripts are blocked (Set-ExecutionPolicy -Scope Process Bypass).
- Fresh data: if you change generators or features, re-run the simulator and re-train the GLM to refresh ./data and ./models.

---

## Documentation

- [Design Document](./docs/design_doc.md) – architecture, data flow, roadmap
- [Data Dictionary](./docs/data_dictionary.md) – schemas and feature definitions
- [Model Card](./docs/model_card.md) – models, features, metrics, limitations
- [Pricing Guardrails](./docs/pricing_guardrails.md) – premium smoothing & caps

---
