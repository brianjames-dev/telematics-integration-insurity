#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
exec uvicorn src.serve.api:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --workers "${WORKERS:-1}"
