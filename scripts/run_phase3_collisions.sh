#!/usr/bin/env bash
set -uo pipefail

output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
runtime_amendment="${PHASE2_RUNTIME_AMENDMENT:-configs/phase2_runtime_amendment_20260917.yaml}"
run_tag="${PHASE3_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
read -r -a selected_runs <<< "${PHASE3_RUN_DIRS:-}"
export PYTHONPATH="${PYTHONPATH:-}:src"

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
  echo "A completed seed-0 identity reference run is required" >&2
  exit 1
fi

is_selected_run() {
  local candidate="$1" selected
  [[ "${#selected_runs[@]}" -eq 0 ]] && return 0
  for selected in "${selected_runs[@]}"; do
    [[ "${candidate}" == "${selected}" ]] && return 0
  done
  return 1
}

has_sufficient_collision() {
  local run_dir="$1" candidate
  while IFS= read -r candidate; do
    [[ -f "${candidate}/COMPLETED" && -f "${candidate}/collision_attacks.parquet" ]] || continue
    if "${python_bin}" -c \
      'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("split") == "final" and int(x.get("max_pairs",0)) >= 300 and x.get("attack_protocol_version") == 2 else 1)' \
      "${candidate}/config.json"; then
      return 0
    fi
  done < <(find "${run_dir}/evaluations" -mindepth 1 -maxdepth 1 -type d \
    -name 'collision-*-attempt*' -print 2>/dev/null | sort)
  return 1
}

audit_path="${output_root}/phase3_duplicate_provenance_${run_tag}.json"
mapfile -t canonical_entries < <(
  "${python_bin}" -m ribs.cli phase2-run-dirs --output-root "${output_root}" \
    --families dimensional vq quantized --audit-path "${audit_path}"
)
failed=0
for entry in "${canonical_entries[@]}"; do
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  is_selected_run "${run_dir}" || continue
  IFS=$'\t' read -r _ _ run_collision < <(
    "${python_bin}" -m ribs.cli phase2-runtime-policy \
      --run-dir "${run_dir}" --amendment "${runtime_amendment}"
  )
  [[ "${run_collision}" == 1 ]] || continue
  if has_sufficient_collision "${run_dir}"; then
    echo "SKIP_COMPLETED family=${family} run=${run_dir}"
    continue
  fi
  echo "START phase3 collision family=${family} run=${run_dir} $(date -Is)"
  if ! tuning_artifact="$("${python_bin}" -m ribs.cli tune-collision \
    --run-dir "${run_dir}" --reference-run-dir "${reference_run}" \
    --lambdas 0.3 1.0 3.0 --max-pairs 50)"; then
    echo "FAILED collision tuning family=${family} run=${run_dir}" >&2
    failed=$((failed + 1))
    continue
  fi
  if ! "${python_bin}" -m ribs.cli collision-attack \
    --run-dir "${run_dir}" --reference-run-dir "${reference_run}" \
    --tuning-artifact "${tuning_artifact}" --max-pairs 300; then
    echo "FAILED collision family=${family} run=${run_dir}" >&2
    failed=$((failed + 1))
    continue
  fi
  echo "DONE phase3 collision family=${family} run=${run_dir} $(date -Is)"
done

if [[ "${PHASE3_FINALIZE:-true}" == true ]]; then
  "${python_bin}" -m ribs.cli aggregate --output-root "${output_root}" --experiment phase2 \
    --include-families dimensional vib vq quantized autoencoder \
    --runtime-amendment "${runtime_amendment}" || failed=$((failed + 1))
  "${python_bin}" -m ribs.cli render --experiment phase2 \
    --summary "${output_root}/phase2_summary.parquet" \
    --output-dir "${output_root}/figures_phase2" || failed=$((failed + 1))
fi

(( failed == 0 ))
