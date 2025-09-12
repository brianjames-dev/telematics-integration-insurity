#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# make_demo.sh — project runner
#   - Phase 1: simulator → ingest → verify → tests
#   - Phase 2: aggregate trips & features → verify → tests
#   - Phase 3: simulate labels → train GLM → verify → tests
#   - Phase 4: train GBM (monotone) + calibration → verify → tests
#   - Phase 5: API smoke (in-process) → verify
# ---------------------------

PHASE="${PHASE:-1}"
CLEAN="${CLEAN:-1}"
DRIVERS="${DRIVERS:-2}"
TRIPS="${TRIPS:-3}"
HZ="${HZ:-1.0}"
OUT="${OUT:-}"                 # Phase 1 parquet out dir (auto-timestamped if empty)
PINGS_IN="${PINGS_IN:-}"       # Phase 2 input pings dir/glob (auto-detect latest if empty)
OUT_TRIPS="${OUT_TRIPS:-data/trips}"
OUT_FEATS="${OUT_FEATS:-data/trip_features}"
OUT_DAILY="${OUT_DAILY:-data/driver_daily}"

usage() {
  cat <<USAGE
Usage:
  Phase 1: bin/make_demo.sh [--phase 1] [--clean|--no-clean] [--drivers N] [--trips N] [--hz HZ] [--out DIR]
  Phase 2: bin/make_demo.sh --phase 2 [--pings DIR|GLOB] [--out-trips DIR] [--out-features DIR] [--out-daily DIR] [--clean]
  Phase 3: bin/make_demo.sh --phase 3 [--clean]
           (env: TARGET_RATE=0.03 L2_SEV=10 L2_FREQ=1.0)
  Phase 4: bin/make_demo.sh --phase 4 [--clean]
           (env: GBM_LR=0.08 GBM_MAX_DEPTH=3 GBM_MAX_LEAVES=31 GBM_TREES=300 GBM_L2=0.0 GBM_CALIB=isotonic SEED=42)
  Phase 5: bin/make_demo.sh --phase 5 [--clean]
           (expects models from phases 3+4; runs bin/verify_phase5.py)

Examples:
  bin/make_demo.sh
  CLEAN=0 bin/make_demo.sh --phase 1 --out data/pings_run1
  bin/make_demo.sh --phase 2 --pings data/pings_run1
  TARGET_RATE=0.06 L2_SEV=10 bin/make_demo.sh --phase 3
  GBM_CALIB=sigmoid SEED=123 bin/make_demo.sh --phase 4
  bin/make_demo.sh --phase 5
USAGE
}

# ---- arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)    PHASE="${2:?}"; shift 2;;
    --clean)    CLEAN=1; shift;;
    --no-clean) CLEAN=0; shift;;
    --drivers)  DRIVERS="${2:?}"; shift 2;;
    --trips)    TRIPS="${2:?}"; shift 2;;
    --hz)       HZ="${2:?}"; shift 2;;
    --out)      OUT="${2:?}"; shift 2;;
    --pings|--pings-in) PINGS_IN="${2:?}"; shift 2;;
    --out-trips)   OUT_TRIPS="${2:?}"; shift 2;;
    --out-features|--out-feats) OUT_FEATS="${2:?}"; shift 2;;
    --out-daily)   OUT_DAILY="${2:?}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log()  { printf "\n\033[1;34m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
need python

latest_pings_dir() {
  ls -1dt data/pings* 2>/dev/null | head -n 1 || true
}

phase1() {
  if [[ -z "${OUT}" ]]; then
    OUT="data/pings_phase1_$(date +%Y%m%d_%H%M%S)"
  fi
  if [[ "${CLEAN}" -eq 1 ]]; then
    log "Cleaning data directories"
    rm -rf data/pings data/tmp "${OUT}"
    mkdir -p data/tmp
  else
    mkdir -p data/tmp
  fi

  log "Simulator: drivers=${DRIVERS} trips=${TRIPS} hz=${HZ}"
  python -m src.simulator.generate_trips --drivers "${DRIVERS}" --trips "${TRIPS}" --hz "${HZ}" --golden

  log "Ingest → Parquet (OUT=${OUT})"
  python -m src.ingest.ingest --input "data/tmp/*.ndjson" --out "${OUT}"

  if [[ -x bin/verify_parquet.py ]]; then
    log "Verify Phase 1 Parquet & de-dup"
    python bin/verify_parquet.py --path "${OUT}"
  fi

  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests"
    pytest -q -k 'not phase5'
  fi

  log "Phase 1 completed successfully"
  echo "Outputs:"
  echo "  - NDJSON: data/tmp/pings.ndjson"
  echo "  - Parquet: ${OUT}/driver_id=*/dt=*/part-*.parquet"
}

phase2() {
  local INP="${PINGS_IN:-}"
  if [[ -z "${INP}" ]]; then
    INP="$(latest_pings_dir)"
    [[ -z "${INP}" ]] && die "No pings directory found. Provide --pings data/pings_run1 or run phase 1 first."
  fi

  if [[ "${CLEAN}" -eq 1 ]]; then
    log "Cleaning phase 2 outputs"
    rm -rf "${OUT_TRIPS}" "${OUT_FEATS}" "${OUT_DAILY}"
  fi

  log "Phase 2: aggregate trips & features"
  python -m src.processing.trip_aggregator \
    --input "${INP}" \
    --out_trips "${OUT_TRIPS}" \
    --out_features "${OUT_FEATS}" \
    --out_daily "${OUT_DAILY}"

  if [[ -x bin/verify_phase2.py ]]; then
    log "Verify Phase 2 feature tables"
    python bin/verify_phase2.py --features "${OUT_FEATS}" --trips "${OUT_TRIPS}" --daily "${OUT_DAILY}"
  fi

  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests"
    pytest -q -k 'not phase5'
  fi

  log "Phase 2 completed successfully"
  echo "Outputs:"
  echo "  - Trips:      ${OUT_TRIPS}/driver_id=*/dt=*/part-*.parquet"
  echo "  - Features:   ${OUT_FEATS}/driver_id=*/dt=*/part-*.parquet"
  echo "  - DriverDay:  ${OUT_DAILY}/driver_id=*/dt=*/part-*.parquet"
}

phase3() {
  local PINGS
  PINGS="$(ls -1dt data/pings* 2>/dev/null | head -n 1 || true)"
  [[ -z "${PINGS}" ]] && die "No pings directory found. Run phase 1 first."
  local FEATURES="data/trip_features"
  [[ ! -d "${FEATURES}" ]] && die "No trip_features found. Run phase 2 first."

  if [[ "${CLEAN}" -eq 1 ]]; then
    log "Cleaning models/ and data/labels_trip"
    rm -rf models data/labels_trip
  fi
  mkdir -p models

  TARGET_RATE="${TARGET_RATE:-0.03}"
  log "Phase 3: simulate labels (target_rate=${TARGET_RATE})"
  python -m src.ml.label_simulator --features "${FEATURES}" --out data/labels_trip --target-rate "${TARGET_RATE}"

  log "Phase 3: train GLM (freq + severity)"
  python -m src.ml.train_glm \
    --features "${FEATURES}" --labels data/labels_trip \
    --models models --metrics_out models/metrics_glm.json \
    --l2-sev "${L2_SEV:-10}" --l2-freq "${L2_FREQ:-1.0}"

  if [[ -x bin/verify_phase3.py ]]; then
    log "Verify Phase 3 metrics + artifacts"
    python bin/verify_phase3.py --models models --metrics models/metrics_glm.json
  fi

  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests"
    pytest -q -k 'not phase5'
  fi

  log "Phase 3 completed successfully"
  echo "Artifacts in ./models and ./data/labels_trip"
}

phase4() {
  if [[ ! -d data/trip_features ]]; then die "Phase 2 outputs not found (data/trip_features)"; fi
  if [[ ! -d data/labels_trip ]]; then die "Phase 3 labels not found (data/labels_trip)"; fi

  if [[ "${CLEAN}" -eq 1 ]]; then
    log "Cleaning GBM artifacts"
    rm -f models/gbm_*.pkl models/gbm_meta.json
  fi
  mkdir -p models

  log "Phase 4: train GBM freq with monotone constraints + calibration"
  python -m src.ml.train_gbm \
    --features data/trip_features \
    --labels data/labels_trip \
    --models models \
    --metrics_out models/metrics_gbm.json \
    --learning_rate "${GBM_LR:-0.08}" \
    --max_depth "${GBM_MAX_DEPTH:-3}" \
    --max_leaf_nodes "${GBM_MAX_LEAVES:-31}" \
    --n_estimators "${GBM_TREES:-300}" \
    --l2 "${GBM_L2:-0.0}" \
    --early_stopping 1 \
    --calib_method "${GBM_CALIB:-isotonic}" \
    --ablations 1 \
    --seed "${SEED:-42}"

  if [[ -x bin/verify_phase4.py ]]; then
    log "Verify Phase 4 GBM metrics + monotonicity"
    python bin/verify_phase4.py --models models --features data/trip_features --labels data/labels_trip
  fi

  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests"
    pytest -q -k 'not phase5'
  fi

  log "Phase 4 completed successfully"
  echo "Artifacts: models/gbm_freq*.pkl, models/gbm_meta.json"
}

phase5() {
  # requires Phase 2 features + models from Phase 3 & 4
  if [[ ! -d data/trip_features ]]; then die "Phase 2 outputs not found (data/trip_features)"; fi
  if [[ ! -f models/glm_sev.json || ! -f models/gbm_meta.json || ! -f models/gbm_freq.pkl ]]; then
    die "Models not found. Run Phase 3 (GLM) and Phase 4 (GBM) first."
  fi
  if [[ ! -x bin/verify_phase5.py ]]; then
    die "Missing bin/verify_phase5.py (API smoke test)."
  fi

  # optional: export model paths for api.py to discover
  export UBI_MODELS_DIR="models"
  export UBI_MODEL_GLM_SEV="models/glm_sev.json"
  export UBI_MODEL_GBM_FREQ="models/gbm_freq.pkl"
  export UBI_MODEL_GBM_CAL="models/gbm_cal.pkl"  # may not exist; api.py should handle gracefully

  log "Phase 5: API smoke (in-process)"
  python bin/verify_phase5.py
  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests (phase 5 only)"
    pytest -q -k 'phase5'
  fi
  log "Phase 5 completed successfully"
}

# ---- final dispatcher
case "${PHASE}" in
  1) log "Running Phase 1"; phase1;;
  2) log "Running Phase 2"; phase2;;
  3) log "Running Phase 3"; phase3;;
  4) log "Running Phase 4"; phase4;;
  5) log "Running Phase 5"; phase5;;
  *) die "Unsupported phase: ${PHASE} (use 1, 2, 3, 4, or 5)";;
esac
