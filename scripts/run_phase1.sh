#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli train-matrix \
  --config "${1:-configs/phase1.yaml}" \
  --dimensions 512 256 128 64 32 16 \
  --seeds 0 1 2
