#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-api}"     # change if your compose service isn't "api"
DEFAULT_PORT=8080
HEALTHCHECK_PATH="${HEALTHCHECK_PATH:-/healthz}"
DASHBOARD_PATH="${DASHBOARD_PATH:-/dashboard}"

# --- sanity checks ---
command -v docker >/dev/null || { echo "Docker is required."; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon not running."; exit 1; }

# --- resolve PORT from env or .env file ---
PORT="${PORT:-}"
if [ -z "${PORT}" ] && [ -f .env ]; then
  PORT="$(grep -E '^PORT=' .env | cut -d= -f2 || true)"
fi
PORT="${PORT:-$DEFAULT_PORT}"

# --- run all phases inside the container so host needs nothing but Docker ---
echo ">> Running demo pipeline inside container: ${SERVICE}"
docker compose run --rm "$SERVICE" bash -lc '
  set -e
  chmod +x bin/make_demo.sh
  PHASE=1 CLEAN=1 DRIVERS=10 TRIPS=50 HZ=1.0 bin/make_demo.sh --phase 1
  PHASE=2 CLEAN=1 bin/make_demo.sh --phase 2
  TARGET_RATE=0.03 L2_SEV=10 L2_FREQ=1.0 PHASE=3 CLEAN=1 bin/make_demo.sh --phase 3
  GBM_LR=0.08 GBM_MAX_DEPTH=3 GBM_MAX_LEAVES=31 GBM_TREES=300 GBM_CALIB=isotonic SEED=42 PHASE=4 CLEAN=1 bin/make_demo.sh --phase 4
  PHASE=5 CLEAN=0 bin/make_demo.sh --phase 5
'

# --- bring up the API in the background ---
echo ">> Starting API"
docker compose up -d "$SERVICE"

BASE_URL="http://localhost:${PORT}"
HEALTH_URL="${BASE_URL}${HEALTHCHECK_PATH}"
DASHBOARD_URL="${BASE_URL}${DASHBOARD_PATH}"

# --- light wait for health (optional; ignore if no /healthz) ---
if command -v curl >/dev/null 2>&1; then
  echo ">> Probing ${HEALTH_URL}"
  curl -fsS "${HEALTH_URL}" >/dev/null 2>&1 || true
fi

echo "------------------------------------------------------------"
echo "✅ Ready! Dashboard: ${DASHBOARD_URL}"
echo "If it didn't open automatically, paste the URL in your browser."
echo "------------------------------------------------------------"

# --- try to open the browser (macOS/Linux/Windows) ---
( command -v open >/dev/null 2>&1 && open "$DASHBOARD_URL" ) || \
( command -v xdg-open >/dev/null 2>&1 && xdg-open "$DASHBOARD_URL" ) || \
( command -v start >/dev/null 2>&1 && start "$DASHBOARD_URL" ) || true
