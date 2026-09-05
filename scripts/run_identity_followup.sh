#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/phase1.yaml}"
python_bin="${PYTHON_BIN:-python}"
output_root="${OUTPUT_ROOT:-outputs}"
export PYTHONPATH="${PYTHONPATH:-}:src"
if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for the prescribed identity follow-up; refusing CPU execution" >&2
  exit 2
fi
"${python_bin}" -m ribs.cli train-identity-matrix \
  --config "${config}" --seeds 0 1 2

has_completed_kind() {
  local run_dir="$1"
  local prefix="$2"
  local require_attack_protocol="${3:-false}"
  local require_full_final="${4:-false}"
  local candidate
  while IFS= read -r candidate; do
    [[ -f "${candidate}/COMPLETED" ]] || continue
    if [[ "${require_full_final}" == true ]]; then
      local metadata="${candidate}/config.json"
      [[ -f "${metadata}" ]] || metadata="${candidate}/clean_final.json"
      if [[ ! -f "${metadata}" ]] || ! "${python_bin}" -c \
        'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("split") == "final" and x.get("max_samples") is None else 1)' \
        "${metadata}"; then
        continue
      fi
    fi
    if [[ "${require_attack_protocol}" == false ]]; then
      return 0
    fi
    if [[ -f "${candidate}/config.json" ]] && "${python_bin}" -c \
      'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("attack_protocol_version") == 2 else 1)' \
      "${candidate}/config.json"; then
      return 0
    fi
  done < <(find "${run_dir}/evaluations" -mindepth 1 -maxdepth 1 -type d \
    -name "${prefix}*-attempt*" -print 2>/dev/null | sort)
  return 1
}

has_completed_analysis() {
  local run_dir="$1"
  local kind="$2"
  local artifact="$3"
  local legacy="$4"
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/COMPLETED" && -f "${candidate}/${artifact}" ]] && \
       "${python_bin}" -c \
         'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["max_samples"] is None else 1)' \
         "${candidate}/config.json"; then
      return 0
    fi
  done < <(find "${run_dir}/analysis" -mindepth 1 -maxdepth 1 -type d \
    -name "${kind}-*-attempt*" -print 2>/dev/null | sort)
  return 1
}

while IFS= read -r run_dir; do
  [[ -f "${run_dir}/artifacts/latents_development_tune.safetensors" ]] || \
    "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split development_tune
  [[ -f "${run_dir}/artifacts/latents_final.safetensors" ]] || \
    "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split final
  has_completed_kind "${run_dir}" "clean-" false true || \
    "${python_bin}" -m ribs.cli evaluate --run-dir "${run_dir}"
  has_completed_kind "${run_dir}" "robustness-" true true || \
    "${python_bin}" -m ribs.cli attack --run-dir "${run_dir}"
  has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || \
    "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment geometry
  has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json || \
    "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment invariance
done < <(find "${output_root}/identity" -mindepth 1 -maxdepth 1 -type d -name 'identity-*' \
  -print 2>/dev/null | sort)

"${python_bin}" -m ribs.cli analyze-identity-followup --output-root "${output_root}"
