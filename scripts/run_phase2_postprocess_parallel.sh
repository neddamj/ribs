#!/usr/bin/env bash
set -u -o pipefail

config_root="${1:-configs}"
decision_record="${2:?usage: scripts/run_phase2_postprocess_parallel.sh [config-root] PHASE2_DECISION.yaml}"
output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
gpu_count="${GPU_COUNT:-4}"
run_tag="${PHASE2_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
runtime_amendment="${PHASE2_RUNTIME_AMENDMENT:-configs/phase2_runtime_amendment_20260917.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

if (( gpu_count < 1 )); then
  echo "GPU_COUNT must be positive" >&2
  exit 2
fi
if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for Phase 2 postprocessing" >&2
  exit 2
fi

log_root="${output_root}/phase2_postprocess_logs/${run_tag}"
claim_root="${output_root}/.phase2_postprocess_claims"
mkdir -p "${log_root}" "${claim_root}"
audit_path="${output_root}/phase2_duplicate_provenance_parallel_${run_tag}.json"
mapfile -t discovered < <(
  "${python_bin}" -m ribs.cli phase2-run-dirs --output-root "${output_root}" \
    --families autoencoder vib vq quantized --audit-path "${audit_path}"
)

# Longest families enter the shared queue first; free GPUs still pull the next
# run dynamically, so no GPU remains tied to one family.
declare -a queue=()
for wanted_family in autoencoder vib vq quantized; do
  for entry in "${discovered[@]}"; do
    IFS=$'\t' read -r family run_dir <<< "${entry}"
    [[ "${family}" == "${wanted_family}" ]] && queue+=("${entry}")
  done
done

declare -a pids=()
declare -a labels=()
declare -a gpu_for_pid=()
running=0
failed=0

reap_finished() {
  local wait_for_one="${1:-false}"
  while true; do
    local progressed=0
    for index in "${!pids[@]}"; do
      [[ -n "${pids[index]+present}" ]] || continue
      if ! kill -0 "${pids[index]}" 2>/dev/null; then
        if wait "${pids[index]}"; then
          echo "DONE ${labels[index]}"
        else
          local status=$?
          echo "FAILED ${labels[index]} exit=${status}" >&2
          failed=$((failed + 1))
        fi
        unset 'gpu_for_pid[index]' 'pids[index]' 'labels[index]'
        running=$((running - 1))
        progressed=1
      fi
    done
    [[ "${wait_for_one}" == false || "${progressed}" -eq 1 ]] && return
    sleep 5
  done
}

free_gpu() {
  local candidate assigned occupied
  for candidate in $(seq 0 $((gpu_count - 1))); do
    occupied=0
    for assigned in "${gpu_for_pid[@]}"; do
      [[ "${assigned}" == "${candidate}" ]] && occupied=1
    done
    if (( occupied == 0 )); then
      echo "${candidate}"
      return
    fi
  done
  return 1
}

launch_run() {
  local gpu="$1"
  local family="$2"
  local run_dir="$3"
  local key label log claim
  key="$(printf '%s' "${run_dir}" | sha256sum | cut -c1-12)"
  label="${family}_${key}"
  log="${log_root}/${label}.log"
  claim="${claim_root}/${key}"
  if ! mkdir "${claim}" 2>/dev/null; then
    echo "SKIP_CLAIMED family=${family} run=${run_dir}"
    return
  fi
  echo "START ${label} gpu=${gpu} run=${run_dir}"
  (
    trap 'rmdir "${claim}" 2>/dev/null || true' EXIT
    env CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 PYTHON_BIN="${python_bin}" OUTPUT_ROOT="${output_root}" \
      PHASE2_RUNTIME_AMENDMENT="${runtime_amendment}" \
      PHASE2_FAMILIES="${family}" PHASE2_RUN_DIRS="${run_dir}" PHASE2_FINALIZE=false \
      PHASE2_RUN_TAG="${run_tag}_${key}" \
      PHASE2_AUDIT_PATH="${output_root}/phase2_duplicate_provenance_parallel_${run_tag}_${key}.json" \
      ./scripts/run_phase2_postprocess.sh "${config_root}" "${decision_record}" \
      >"${log}" 2>&1
  ) &
  pids+=("$!")
  labels+=("${label}")
  gpu_for_pid+=("${gpu}")
  running=$((running + 1))
}

for entry in "${queue[@]}"; do
  while (( running >= gpu_count )); do
    reap_finished true
  done
  gpu="$(free_gpu)"
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  launch_run "${gpu}" "${family}" "${run_dir}"
done
while (( running > 0 )); do
  reap_finished true
done

aggregate_failed=0
"${python_bin}" -m ribs.cli aggregate --output-root "${output_root}" --experiment phase2 \
  --include-families dimensional vib vq quantized autoencoder \
  --runtime-amendment "${runtime_amendment}" || aggregate_failed=1
"${python_bin}" -m ribs.cli render --experiment phase2 \
  --summary "${output_root}/phase2_summary.parquet" \
  --output-dir "${output_root}/figures_phase2" || aggregate_failed=1
"${python_bin}" -m ribs.cli validate-phase2 --output-root "${output_root}" \
  --runtime-amendment "${runtime_amendment}" || failed=$((failed + 1))

if (( failed > 0 || aggregate_failed > 0 )); then
  echo "Phase 2 postprocessing finished with worker_failures=${failed} aggregate_failure=${aggregate_failed}" >&2
  exit 1
fi
