#!/usr/bin/env bash
set -uo pipefail

config_root="${1:-configs}"
decision_record="${2:?usage: scripts/run_phase2_postprocess.sh [config-root] PHASE2_DECISION.yaml}"
output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-}:src"
read -r -a phase2_families <<< "${PHASE2_FAMILIES:-vib vq quantized autoencoder}"
read -r -a phase2_skip_run_dirs <<< "${PHASE2_SKIP_RUN_DIRS:-}"
read -r -a phase2_run_dirs <<< "${PHASE2_RUN_DIRS:-}"
run_tag="${PHASE2_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
status_tsv="$(mktemp)"
printf 'family\trun_dir\tstage\tstatus\tduration_seconds\n' >"${status_tsv}"

write_reports() {
  local csv_path="${output_root}/phase2_postprocess_${run_tag}.csv"
  local json_path="${output_root}/phase2_postprocess_${run_tag}.json"
  "${python_bin}" -c 'import csv,json,sys; rows=list(csv.DictReader(open(sys.argv[1]), delimiter="\t")); fields=["family","run_dir","stage","status","duration_seconds"]; out=open(sys.argv[2],"w",newline=""); writer=csv.DictWriter(out,fieldnames=fields); writer.writeheader(); writer.writerows(rows); out.close(); jout=open(sys.argv[3],"w"); json.dump({"runs":rows},jout,indent=2); jout.write("\n"); jout.close()' "${status_tsv}" "${csv_path}" "${json_path}"
  rm -f "${status_tsv}"
  echo "POSTPROCESS_REPORT ${json_path}"
}
trap write_reports EXIT

should_skip_run() {
  local candidate="$1"
  local skipped
  for skipped in "${phase2_skip_run_dirs[@]}"; do
    [[ "${candidate}" == "${skipped}" ]] && return 0
  done
  return 1
}

is_selected_run() {
  local candidate="$1"
  local selected
  [[ "${#phase2_run_dirs[@]}" -eq 0 ]] && return 0
  for selected in "${phase2_run_dirs[@]}"; do
    [[ "${candidate}" == "${selected}" ]] && return 0
  done
  return 1
}

record_stage() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" >>"${status_tsv}"
}

run_stage() {
  local family="$1"
  local run_dir="$2"
  local stage="$3"
  shift 3
  local started=$SECONDS
  if "$@"; then
    record_stage "${family}" "${run_dir}" "${stage}" completed "$((SECONDS - started))"
    return 0
  else
    local status=$?
    record_stage "${family}" "${run_dir}" "${stage}" failed "$((SECONDS - started))"
    echo "FAILED stage=${stage} family=${family} run=${run_dir} exit=${status}" >&2
    return "${status}"
  fi
}

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
  local failed=0
  echo "START phase2 postprocess family=${family} run=${run_dir} $(date -Is)"
  if [[ ! -f "${run_dir}/artifacts/latents_development_tune.safetensors" ]]; then
    run_stage "${family}" "${run_dir}" latents_development_tune \
      "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" \
      --split development_tune || failed=1
  fi
  if [[ ! -f "${run_dir}/artifacts/latents_final.safetensors" ]]; then
    run_stage "${family}" "${run_dir}" latents_final \
      "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" \
      --split final || failed=1
  fi
  if [[ "${family}" == autoencoder ]]; then
    has_completed_kind "${run_dir}" "autoencoder-clean-" false true || run_stage \
      "${family}" "${run_dir}" clean "${python_bin}" -m ribs.cli evaluate-autoencoder \
      --run-dir "${run_dir}" --reference-run-dir "${reference_run}" || failed=1
    has_completed_kind "${run_dir}" "autoencoder-robustness-" true true || run_stage \
      "${family}" "${run_dir}" robustness "${python_bin}" -m ribs.cli autoencoder-attack \
      --run-dir "${run_dir}" --reference-run-dir "${reference_run}" || failed=1
    has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || run_stage \
      "${family}" "${run_dir}" geometry "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment geometry || failed=1
    has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json || run_stage \
      "${family}" "${run_dir}" invariance "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment invariance \
      --reference-run-dir "${reference_run}" || failed=1
  else
    has_completed_kind "${run_dir}" "clean-" false true || run_stage \
      "${family}" "${run_dir}" clean "${python_bin}" -m ribs.cli evaluate \
      --run-dir "${run_dir}" || failed=1
    has_completed_kind "${run_dir}" "robustness-" true true || run_stage \
      "${family}" "${run_dir}" robustness "${python_bin}" -m ribs.cli attack \
      --run-dir "${run_dir}" || failed=1
    has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || run_stage \
      "${family}" "${run_dir}" geometry "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment geometry || failed=1
    has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json || run_stage \
      "${family}" "${run_dir}" invariance "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment invariance || failed=1
    if [[ "${family}" == vq || "${family}" == quantized ]]; then
      has_completed_kind "${run_dir}" "square-" true || run_stage \
        "${family}" "${run_dir}" square "${python_bin}" -m ribs.cli square-attack \
        --run-dir "${run_dir}" || failed=1
      has_completed_kind "${run_dir}" "transfer-" true || run_stage \
        "${family}" "${run_dir}" transfer "${python_bin}" -m ribs.cli transfer-attack \
        --run-dir "${run_dir}" || failed=1
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
    failed=1
    tuning_artifact=""
  fi
  if [[ "${#tuning_candidates[@]}" -eq 0 ]]; then
    read -r -a collision_lambdas <<< "${COLLISION_LAMBDAS:-0.1 0.3 1 3 10}"
    local started=$SECONDS
    if tuning_artifact="$("${python_bin}" -m ribs.cli tune-collision \
      --run-dir "${run_dir}" --reference-run-dir "${reference_run}" \
      --lambdas "${collision_lambdas[@]}" \
      --max-pairs "${COLLISION_TUNING_PAIRS:-200}")"; then
      record_stage "${family}" "${run_dir}" collision_tuning completed "$((SECONDS - started))"
    else
      record_stage "${family}" "${run_dir}" collision_tuning failed "$((SECONDS - started))"
      failed=1
      tuning_artifact=""
    fi
  elif [[ "${#tuning_candidates[@]}" -eq 1 ]]; then
    tuning_artifact="${tuning_candidates[0]}"
  fi
  if [[ -n "${tuning_artifact}" ]]; then
    has_completed_kind "${run_dir}" "collision-" true true || run_stage \
      "${family}" "${run_dir}" collision "${python_bin}" -m ribs.cli collision-attack \
      --run-dir "${run_dir}" --reference-run-dir "${reference_run}" \
      --tuning-artifact "${tuning_artifact}" --max-pairs "${COLLISION_FINAL_PAIRS:-1000}" \
      || failed=1
  else
    record_stage "${family}" "${run_dir}" collision blocked 0
  fi
  echo "DONE phase2 postprocess family=${family} run=${run_dir} $(date -Is)"
  return "${failed}"
}

duplicate_audit="${PHASE2_AUDIT_PATH:-${output_root}/phase2_duplicate_provenance_$(date -u +%Y%m%dT%H%M%SZ).json}"
mapfile -t canonical_entries < <(
  "${python_bin}" -m ribs.cli phase2-run-dirs \
    --output-root "${output_root}" \
    --families "${phase2_families[@]}" \
    --audit-path "${duplicate_audit}"
)
failed_runs=0
for entry in "${canonical_entries[@]}"; do
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  is_selected_run "${run_dir}" || continue
  if should_skip_run "${run_dir}"; then
    echo "SKIP phase2 postprocess family=${family} run=${run_dir} reason=PHASE2_SKIP_RUN_DIRS"
    continue
  fi
  process_run "${run_dir}" "${family}" || failed_runs=$((failed_runs + 1))
done

if [[ "${PHASE2_FINALIZE:-true}" == true ]]; then
  "${python_bin}" -m ribs.cli aggregate --output-root "${output_root}" --experiment phase2 \
    --include-families dimensional vib vq quantized autoencoder || failed_runs=$((failed_runs + 1))
  "${python_bin}" -m ribs.cli render --experiment phase2 \
    --summary "${output_root}/phase2_summary.parquet" \
    --output-dir "${output_root}/figures_phase2" || failed_runs=$((failed_runs + 1))
  "${python_bin}" -m ribs.cli validate-phase2 --output-root "${output_root}" \
    || failed_runs=$((failed_runs + 1))
fi

(( failed_runs == 0 ))
