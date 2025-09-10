# Phase 1 Runbook — Simulation → Ingest → Verify → Tests

## 1) Purpose & Scope

**Goal:** establish a clean, repeatable **data landing zone** for telematics pings:

- Generate deterministic, persona-based **raw pings** (NDJSON).
- Ingest to **partitioned, de-duplicated Parquet** (analytics-ready).
- **Verify** integrity and run **unit tests** that lock our contracts.

**Why this matters:** If raw data handling is flaky (duplicates, bad timestamps, slow format), trip features, models, and pricing will be unreliable. Phase 1 makes the foundation solid before Phase 2+.

---

## 2) Prerequisites

- Python 3.11+ (with `pandas`, `pyarrow`, `pydantic`, `pytest`)
- Repo layout includes:
  - `src/simulator/generate_trips.py`
  - `src/ingest/ingest.py`
  - `src/serve/schemas.py` (pydantic models)
  - `bin/make_demo.sh`, `bin/verify_parquet.py`, `bin/run_tests.sh` (exec)
  - `tests/` with schema/simulator/ingest unit tests
- Ensure `src/__init__.py` exists (so `python -m src.…` works)

---

## 3) One-Command Demo (recommended)

Runs **clean → simulate → ingest → verify → tests** with clear logs.

    bin/make_demo.sh

Variants:

- Keep existing data and re-run:

  CLEAN=0 bin/make_demo.sh --out data/pings_run1

- Control data volume / sampling:

  bin/make_demo.sh --drivers 3 --trips 10 --hz 2.0

**Expected success log (example):**

> \[HH:MM:SS] Running Phase 1  
> \[HH:MM:SS] Cleaning data directories  
> \[HH:MM:SS] Simulator: drivers=3 trips=10 hz=1.0  
> \[HH:MM:SS] Ingest → Parquet (OUT=data/pings_run1)  
> Ingested 47700 → wrote 45545 rows to data/pings_run1  
> \[HH:MM:SS] Verify Parquet & de-dup  
> \[VERIFY] rows=45545 unique(driver_id,ts)=45545 drivers=3  
> \[HH:MM:SS] Run unit tests  
> ....  
> 4 passed in 0.26s  
> \[HH:MM:SS] Phase 1 completed successfully

---

## 4) Step-by-Step (what happens + why)

### Step A — Clean (default)

- **What:** Remove prior `data/tmp` and `data/pings*` (unless `CLEAN=0`), or write to a fresh timestamped OUT dir.
- **Why:** Ingest appends new Parquet parts; repeated runs into the same folder create apparent duplicates across runs. Cleaning (or unique OUT) avoids false failures in verification.

**Command run by the script:**

- removes: `data/tmp`, `data/pings*`
- recreates: `data/tmp`

**Success:** folders removed/recreated; no errors.

---

### Step B — Simulate NDJSON pings

- **What:** Generate persona-based driver trips at 1–5 Hz with realistic noise.
- **Why:** Personas (safe/aggressive/night-owl) create learnable risk differences; NDJSON is human-readable for spot checks; seeded for reproducibility.

**Command run by the script:**

- `python -m src.simulator.generate_trips --drivers {N} --trips {M} --hz {Hz} --golden`

**Outputs:**

- `data/tmp/pings.ndjson` (main)
- `data/tmp/golden/pings_small.ndjson` (tiny sample used in tests)

**Quick check:**

- `head -n 5 data/tmp/pings.ndjson` (values look sensible; timestamps increasing)

---

### Step C — Ingest → partitioned Parquet + de-dup

- **What:** Convert NDJSON to Parquet under Hive-style partitions; **de-dup** within the batch by `(driver_id, ts)`.
- **Why:** Parquet is compressed/columnar → fast analytics; partitioning by driver/date enables selective scans; early de-dup keeps later event counts correct.

**Command run by the script:**

- `python -m src.ingest.ingest --input "data/tmp/*.ndjson" --out "{OUT}"`

**Partition tree (example):**

- `data/pings_run1/driver_id=D_001/dt=2025-09-09/part-*.parquet`
- `data/pings_run1/driver_id=D_002/dt=2025-09-09/part-*.parquet`

**Success:** Parquet files exist in each `(driver_id, dt)`; ingest prints `Ingested X → wrote Y`.

---

### Step D — Verify read-back & de-dup

- **What:** Read Parquet back; assert `rows == unique(driver_id, ts)`; print sample rows.
- **Why:** Objective, fast proof that the landing zone is clean and queryable.

**Command run by the script:**

- `bin/verify_parquet.py --path {OUT}`

**Success criteria:**

- `rows == unique(driver_id, ts)` (no duplicates after ingest)
- Sample rows show sensible `speed_mps`, `accel_mps2`, timestamps with UTC tz

**Note:** If you re-ingest to the **same** OUT across multiple runs (with `CLEAN=0`), total rows will exceed uniques; that’s expected append behavior. Use a **fresh OUT** or clean.

---

### Step E — Unit tests

- **What:** Run fast tests that lock contracts.
- **Why:** Prevent regressions and schema drift.

**Command run by the script:**

- `pytest -q` (or `bin/run_tests.sh`)

**What’s covered:**

- **Schemas:** `TelemetryPing` and `TripFeatures` validate; range constraints (e.g., `night_ratio ∈ [0,1]`)
- **Simulator:** timestamps non-decreasing in a small sample
- **Ingest:** partitions created; batch de-dup works on crafted duplicates

**Success:** `4 passed` (or more as tests grow)

---

## 5) Success Criteria (Definition of Done)

- **NDJSON** exists at `data/tmp/pings.ndjson`; quick `head` looks sane.
- **Parquet partitions** under `{OUT}/driver_id=*/dt=*` with non-zero parts.
- **Verifier** prints `rows == unique(driver_id, ts)` and driver count.
- **Unit tests** pass (`pytest -q` → all green).
- The entire flow is reproducible via **one command**: `bin/make_demo.sh`.

---

## 6) Common Pitfalls & Fixes

- **Verifier says duplicates remain:**  
  You re-ingested into the same OUT across runs.  
  **Fix:** run with clean (default) or specify a new `--out`.

- **“Unknown arg: #” or “event not found: /usr/bin/env” in zsh:**  
  You pasted comment lines/shebangs into the interactive shell.  
  **Fix:** run scripts, don’t paste their internals. Optional: `setopt interactivecomments`.

- **Makefile “missing separator”:**  
  Recipe lines must start with a **TAB**, not spaces.  
  **Fix:** replace leading spaces with a real TAB.

- **Empty Parquet listing:**  
  Wrong `--input` glob or empty NDJSON.  
  **Fix:** check `ls data/tmp/*.ndjson`; re-run simulator.

---

## 7) Artifacts Produced

- `data/tmp/pings.ndjson` (raw pings, human-readable)
- `data/tmp/golden/pings_small.ndjson` (tiny sample for tests)
- `{OUT}/driver_id=*/dt=*/part-*.parquet` (analytics layer)
- Unit test outputs (stdout + `.pytest_cache`)

---

## 8) Design Choices (reasoning snapshot)

- **NDJSON in → Parquet out:** NDJSON is streamable & debuggable; Parquet is efficient for analytics.
- **Partition by driver/date:** supports selective reads by customer or period.
- **De-dup at ingest:** avoids inflated event counts later.
- **Seeded personas:** create learnable risk variation; keep runs reproducible.
- **Single orchestrator script:** one predictable entry point for humans/CI; easy to extend to Phase 2+.

---

## 9) Command Reference

- Run end-to-end Phase 1:

      bin/make_demo.sh

- Keep data & re-run into a specific folder:

      CLEAN=0 bin/make_demo.sh --out data/pings_run1

- Scale:

      bin/make_demo.sh --drivers 3 --trips 10 --hz 2.0

- Verify any Parquet path:

      bin/verify_parquet.py --path data/pings_run1

- Unit tests:

      pytest -q
      # or
      bin/run_tests.sh

- Clean data dirs (if you kept a Makefile target):

      make data-clean

---

## 10) Next Phase (preview)

**Phase 2 — Trip Sessionization & Feature Aggregation**

- Segment pings into trips (idle gap rules).
- Compute filed, explainable features (e.g., `harsh_brakes_per100km`, `night_ratio`, `p95_over_limit_kph`).
- Produce `trip_features` and `driver_daily` tables with p99 clipping.
- Add tests for sessionization, event detection, and feature contracts.
