#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ubi-api:dev}"

case "${1:-}" in
  build)
    docker build -t "$IMAGE" .
    ;;
  serve)
    docker run --rm -p 8000:8000 \
      -v "$PWD/models:/app/models" \
      -v "$PWD/data:/app/data:ro" \
      "$IMAGE"
    ;;
  verify)
    docker run --rm \
      -v "$PWD/models:/app/models" \
      -v "$PWD/data:/app/data:ro" \
      "$IMAGE" \
      python bin/verify_phase5.py
    ;;
  shell)
    docker run -it --rm \
      -v "$PWD/models:/app/models" \
      -v "$PWD/data:/app/data:ro" \
      "$IMAGE" bash
    ;;
  score-batch)
    docker run --rm \
      -v "$PWD:/app" \
      -v "$PWD/models:/app/models" \
      -v "$PWD/data:/app/data:ro" \
      "$IMAGE" \
      python bin/score_batch.py
    ;;
  *)
    echo "Usage: $0 {build|serve|verify|shell|score-batch}" >&2
    exit 2
    ;;
esac
