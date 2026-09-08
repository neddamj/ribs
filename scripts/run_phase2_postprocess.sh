#!/usr/bin/env bash
set -euo pipefail

config_root="${1:-configs}"
decision_record="${2:?usage: scripts/run_phase2_postprocess.sh [config-root] PHASE2_DECISION.yaml}"
output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:src"
read -r -a phase2_families <<< "${PHASE2_FAMILIES:-vib vq quantized autoencoder}"

if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for the prescribed Phase 2 final evaluation; refusing CPU execution" >&2
  exit 2
fi

reference_run=""
while IFS= read -r candidate; do
  if [[ -f "${candidate}/resolved_config.yaml" ]] && "${python_bin}" -c \
    'import sys,yaml; x=yaml.safe_load(open(sys.argv[1])); raise SystemExit(0 if x.get("seed") == 0 and x.get("model",{}).get("family") == "identity" else 1)' \
    "${candidate}/resolved_config.yaml"; then
    reference_run="${candidate}"
    break
  fi
done < <(find "${output_root}/identity" -mindepth 2 -maxdepth 2 -type f -name COMPLETED \
  -printf '%h\n' 2>/dev/null | sort)
if [[ -z "${reference_run}" ]]; then
  echo "A completed seed-0 identity reference run is required for Phase 2 postprocessing" >&2
  exit 1
fi

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
      [[ -f "${metadata}" ]] || metadata="${candidate}/reconstruction_final.json"
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

process_run() {
  local run_dir="$1"
  local family="$2"
  echo "START phase2 postprocess family=${family} run=${run_dir} $(date -Is)"
  [[ -f "${run_dir}/artifacts/latents_development_tune.safetensors" ]] || \
    "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split development_tune
  [[ -f "${run_dir}/artifacts/latents_final.safetensors" ]] || \
    "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split final
  if [[ "${family}" == autoencoder ]]; then
    has_completed_kind "${run_dir}" "autoencoder-clean-" false true || \
      "${python_bin}" -m ribs.cli evaluate-autoencoder --run-dir "${run_dir}" \
        --reference-run-dir "${reference_run}"
    has_completed_kind "${run_dir}" "autoencoder-robustness-" true true || \
      "${python_bin}" -m ribs.cli autoencoder-attack --run-dir "${run_dir}" \
        --reference-run-dir "${reference_run}"
    has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || \
      "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment geometry
    has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json || \
      "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment invariance \
        --reference-run-dir "${reference_run}"
  else
    has_completed_kind "${run_dir}" "clean-" false true || \
      "${python_bin}" -m ribs.cli evaluate --run-dir "${run_dir}"
    has_completed_kind "${run_dir}" "robustness-" true true || \
      "${python_bin}" -m ribs.cli attack --run-dir "${run_dir}"
    has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || \
      "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment geometry
    has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json || \
      "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment invariance
    if [[ "${family}" == vq || "${family}" == quantized ]]; then
      has_completed_kind "${run_dir}" "square-" true || \
        "${python_bin}" -m ribs.cli square-attack --run-dir "${run_dir}"
      has_completed_kind "${run_dir}" "transfer-" true || \
        "${python_bin}" -m ribs.cli transfer-attack --run-dir "${run_dir}"
    fi
  fi
  local tuning_artifact
  local tuning_candidates=()
  local collision_lambdas=()
  mapfile -t tuning_candidates < <(
    find "${run_dir}/evaluations" -mindepth 2 -maxdepth 2 -type f \
      -path '*/collision-tuning-*-attempt*/selection.json' -print 2>/dev/null | \
      while read -r path; do
        [[ -f "${path%/selection.json}/COMPLETED" ]] && echo "${path}"
      done | sort
  )
  if [[ "${#tuning_candidates[@]}" -gt 1 ]]; then
    echo "Multiple completed collision tuning artifacts require explicit resolution: ${run_dir}" >&2
    return 1
  fi
  if [[ "${#tuning_candidates[@]}" -eq 0 ]]; then
    read -r -a collision_lambdas <<< "${COLLISION_LAMBDAS:-0.1 0.3 1 3 10}"
    tuning_artifact="$("${python_bin}" -m ribs.cli tune-collision \
      --run-dir "${run_dir}" --reference-run-dir "${reference_run}" \
      --lambdas "${collision_lambdas[@]}" \
      --max-pairs "${COLLISION_TUNING_PAIRS:-200}")"
  else
    tuning_artifact="${tuning_candidates[0]}"
  fi
  has_completed_kind "${run_dir}" "collision-" true true || \
    "${python_bin}" -m ribs.cli collision-attack --run-dir "${run_dir}" \
      --reference-run-dir "${reference_run}" --tuning-artifact "${tuning_artifact}" \
      --max-pairs "${COLLISION_FINAL_PAIRS:-1000}"
  echo "DONE phase2 postprocess family=${family} run=${run_dir} $(date -Is)"
}

duplicate_audit="${PHASE2_AUDIT_PATH:-${output_root}/phase2_duplicate_provenance_$(date -u +%Y%m%dT%H%M%SZ).json}"
mapfile -t canonical_entries < <(
  "${python_bin}" -m ribs.cli phase2-run-dirs \
    --output-root "${output_root}" \
    --families "${phase2_families[@]}" \
    --audit-path "${duplicate_audit}"
)
for entry in "${canonical_entries[@]}"; do
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  process_run "${run_dir}" "${family}"
done

if [[ "${PHASE2_FINALIZE:-true}" == true ]]; then
  "${python_bin}" -m ribs.cli aggregate --output-root "${output_root}" --experiment phase2 \
    --include-families dimensional vib vq quantized autoencoder
  "${python_bin}" -m ribs.cli render --experiment phase2 \
    --summary "${output_root}/phase2_summary.parquet" --output-dir "${output_root}/figures_phase2"
  "${python_bin}" -m ribs.cli validate-phase2 --output-root "${output_root}"
fi
