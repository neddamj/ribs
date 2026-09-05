#!/usr/bin/env bash
set -euo pipefail

config_root="${1:-configs}"
decision_record="${2:?usage: scripts/run_phase2.sh [config-root] PHASE2_DECISION.yaml}"
python_bin="${PYTHON_BIN:-python}"
if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for the prescribed Phase 2 matrix; refusing CPU execution" >&2
  exit 2
fi
if ! PYTHONPATH="${PYTHONPATH:-}:src" "${python_bin}" -m ribs.cli analyze-identity-followup \
  --output-root "${OUTPUT_ROOT:-outputs}"; then
  echo "The three-seed identity follow-up must complete before Phase 2" >&2
  exit 2
fi
PYTHONPATH="${PYTHONPATH:-}:src" "${python_bin}" -m ribs.cli train-family-matrix \
  --decision-record "${decision_record}" --config "${config_root}/vib.yaml" \
  --parameter beta --values 0 0.0001 0.0003 0.001 0.003 0.01 --seeds 0 1 2
PYTHONPATH="${PYTHONPATH:-}:src" "${python_bin}" -m ribs.cli train-family-matrix \
  --decision-record "${decision_record}" --config "${config_root}/vq.yaml" \
  --parameter codebook_size --values 512 256 128 64 32 16 --seeds 0 1 2
PYTHONPATH="${PYTHONPATH:-}:src" "${python_bin}" -m ribs.cli train-family-matrix \
  --decision-record "${decision_record}" --config "${config_root}/quantized.yaml" \
  --parameter bits --values FP32 8 6 4 3 2 --seeds 0 1 2
PYTHONPATH="${PYTHONPATH:-}:src" "${python_bin}" -m ribs.cli train-family-matrix \
  --decision-record "${decision_record}" --config "${config_root}/autoencoder.yaml" \
  --parameter dz --values 512 256 128 64 32 --seeds 0 1 2
"$(dirname "$0")/run_phase2_postprocess.sh" "${config_root}" "${decision_record}"
