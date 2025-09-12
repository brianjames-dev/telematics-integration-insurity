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

Run the full demo locally with Docker in **two commands** (from the project root):

    docker compose build
    bash bin/run_demo.sh

---

## Prerequisites

- **Docker Desktop** (includes Docker Compose v2)
  - macOS: https://docs.docker.com/desktop/setup/install/mac-install/
  - Windows (with **WSL2** backend): https://docs.docker.com/desktop/features/wsl/
  - Product page (all platforms): https://www.docker.com/products/docker-desktop/

> **Windows:** run commands in **WSL2 (Ubuntu)** or **Git Bash** so `bash` scripts work.

---

## Tuning the demo (optional)

You can override environment variables when running `bin/run_demo.sh` to control data size and model behavior:

    DRIVERS=10 TRIPS=50 HZ=1.0 TARGET_RATE=0.05 GBM_TREES=300 bash bin/run_demo.sh

**Phase 1 (simulation → pings → parquet)**

- `DRIVERS` — number of distinct drivers (e.g., `10`)
- `TRIPS` — trips **per driver** (e.g., `50`) → total trips ≈ `DRIVERS × TRIPS`
- `HZ` — ping frequency (samples/sec). Higher = more rows per trip (e.g., `1.0`)

**Phase 3 (labels + GLM)**

- `TARGET_RATE` — average claim rate simulated. Higher ⇒ more risky trips overall (e.g., `0.03`)
- `L2_SEV`, `L2_FREQ` — GLM regularization. Higher ⇒ smoother/shrunk coefficients (e.g., `10`, `1.0`)

**Phase 4 (GBM + calibration)**

- `GBM_LR`, `GBM_TREES`, `GBM_MAX_DEPTH`, `GBM_MAX_LEAVES` — model capacity/smoothness
- `GBM_CALIB` — `isotonic` (flexible) or `sigmoid` (Platt scaling)
- `SEED` — reproducible training (e.g., `42`)

**How these settings shape the demo**

- **Volume / richness:** `DRIVERS × TRIPS` sets table sizes; `HZ` scales ping file size.
- **Risk mix:** `TARGET_RATE` shifts the distribution of risk scores (higher rate ⇒ more high-risk trips/drivers).
- **Smoothness vs detail:** higher GLM L2 = smoother; larger GBM depth/leaves/trees = more expressive (but risk overfit). Calibration affects how probabilities map to observed rates.

---

## Reset to a clean state (optional)

Simulate a fresh machine by removing containers/volumes and generated artifacts:

    docker compose down -v --remove-orphans || true
    docker image prune -f
    docker network prune -f
    rm -rf data/ models/        # optional: wipe generated outputs
    docker compose build
    bash bin/run_demo.sh

> If port `8080` is busy:  
> `PORT=8090 bash bin/run_demo.sh` → open `http://localhost:8090/dashboard`.

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
