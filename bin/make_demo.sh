#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# make_demo.sh — project runner
#   - Phase 1: simulator → ingest → verify → tests
#   - Phase 2: aggregate trips & features → verify → tests
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
    -h|--help)
      cat <<USAGE
Usage:
  Phase 1: bin/make_demo.sh [--phase 1] [--clean|--no-clean] [--drivers N] [--trips N] [--hz HZ] [--out DIR]
  Phase 2: bin/make_demo.sh --phase 2 [--pings DIR|GLOB] [--out-trips DIR] [--out-features DIR] [--out-daily DIR] [--clean]

Examples:
  bin/make_demo.sh
  CLEAN=0 bin/make_demo.sh --phase 1 --out data/pings_run1
  bin/make_demo.sh --phase 2 --pings data/pings_run1
USAGE
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() { printf "\n\033[1;34m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

need python

latest_pings_dir() {
  # find the most recent pings* directory
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
    pytest -q
  fi

  log "Phase 1 completed successfully"
  echo "Outputs:"
  echo "  - NDJSON: data/tmp/pings.ndjson"
  echo "  - Parquet: ${OUT}/driver_id=*/dt=*/part-*.parquet"
}

phase2() {
  # determine input pings
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
    pytest -q
  fi

  log "Phase 2 completed successfully"
  echo "Outputs:"
  echo "  - Trips:      ${OUT_TRIPS}/driver_id=*/dt=*/part-*.parquet"
  echo "  - Features:   ${OUT_FEATS}/driver_id=*/dt=*/part-*.parquet"
  echo "  - DriverDay:  ${OUT_DAILY}/driver_id=*/dt=*/part-*.parquet"
}

case "${PHASE}" in
  1) log "Running Phase 1"; phase1;;
  2) log "Running Phase 2"; phase2;;
  *) die "Unsupported phase: ${PHASE} (use 1 or 2)";;
esac
