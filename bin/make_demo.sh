#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# make_demo.sh — project runner
#   - Phase 1: simulator → ingest → verify → tests
#   - Extensible: add Phase 2/3 later
# ---------------------------

# ---- defaults (override with env or flags) ----
PHASE="${PHASE:-1}"            # which phase to run (currently supports 1)
CLEAN="${CLEAN:-1}"            # 1=clean data before run, 0=keep
DRIVERS="${DRIVERS:-2}"
TRIPS="${TRIPS:-3}"
HZ="${HZ:-1.0}"
OUT="${OUT:-}"                 # default set below; unique per run

# ---- tiny arg parser (flags override env) ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)    PHASE="${2:?}"; shift 2;;
    --clean)    CLEAN=1; shift;;
    --no-clean) CLEAN=0; shift;;
    --drivers)  DRIVERS="${2:?}"; shift 2;;
    --trips)    TRIPS="${2:?}"; shift 2;;
    --hz)       HZ="${2:?}"; shift 2;;
    --out)      OUT="${2:?}"; shift 2;;
    -h|--help)
      cat <<USAGE
Usage: bin/make_demo.sh [--phase 1] [--clean|--no-clean] [--drivers N] [--trips N] [--hz HZ] [--out DIR]

Examples:
  bin/make_demo.sh                    # run Phase 1 with defaults, clean first
  CLEAN=0 bin/make_demo.sh --phase 1  # keep existing data, rerun simulator+ingest
  bin/make_demo.sh --drivers 3 --trips 10 --out data/pings_run1
USAGE
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# ---- helpers ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

log() { printf "\n\033[1;34m[%s]\033[0m %s\n" "$(date +%H:%M:%S)" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# ensure required tools exist
need python
need sed
# pytest is optional; we detect later

# choose OUT dir (unique per run if not provided)
if [[ -z "${OUT}" ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  OUT="data/pings_phase${PHASE}_${stamp}"
fi

# ensure src is a package for `python -m`
[[ -f src/__init__.py ]] || : > src/__init__.py

# ---- PHASE 1: simulator → ingest → verify → tests ----
phase1() {
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
    log "Verify Parquet & de-dup"
    python bin/verify_parquet.py --path "${OUT}"
  else
    log "Verifier not found (bin/verify_parquet.py). Skipping."
  fi

  if command -v pytest >/dev/null 2>&1; then
    log "Run unit tests"
    pytest -q
  else
    log "pytest not installed; skipping unit tests"
  fi

  log "Phase 1 completed successfully"
  echo "Outputs:"
  echo "  - NDJSON: data/tmp/pings.ndjson (and data/tmp/golden/pings_small.ndjson)"
  echo "  - Parquet: ${OUT}/driver_id=*/dt=*/part-*.parquet"
}

case "${PHASE}" in
  1) log "Running Phase 1"; phase1;;
  *) die "Unsupported phase: ${PHASE}. Only phase 1 is implemented here.";;
esac
