#!/usr/bin/env bash
set -uo pipefail

output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
gpu_count="${GPU_COUNT:-4}"
run_tag="${PHASE3_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
runtime_amendment="${PHASE2_RUNTIME_AMENDMENT:-configs/phase2_runtime_amendment_20260917.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

if (( gpu_count < 1 )); then
  echo "GPU_COUNT must be positive" >&2
  exit 2
fi
if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for Phase 3 collision evaluation" >&2
  exit 2
fi

log_root="${output_root}/phase3_collision_logs/${run_tag}"
claim_root="${output_root}/.phase3_collision_claims"
mkdir -p "${log_root}" "${claim_root}"
audit_path="${output_root}/phase3_duplicate_provenance_parallel_${run_tag}.json"
mapfile -t discovered < <(
  "${python_bin}" -m ribs.cli phase2-run-dirs --output-root "${output_root}" \
    --families dimensional vq quantized --audit-path "${audit_path}"
)

declare -a queue=()
for entry in "${discovered[@]}"; do
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  IFS=$'\t' read -r _ _ selected < <(
    "${python_bin}" -m ribs.cli phase2-runtime-policy \
      --run-dir "${run_dir}" --amendment "${runtime_amendment}"
  )
  [[ "${selected}" == 1 ]] && queue+=("${entry}")
done

declare -a pids=() labels=() gpu_for_pid=()
running=0
failed=0

reap_finished() {
  local wait_for_one="${1:-false}" index progressed status
  while true; do
    progressed=0
    for index in "${!pids[@]}"; do
      [[ -n "${pids[index]+present}" ]] || continue
      if ! kill -0 "${pids[index]}" 2>/dev/null; then
        if wait "${pids[index]}"; then
          echo "DONE ${labels[index]}"
        else
          status=$?
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

for entry in "${queue[@]}"; do
  while (( running >= gpu_count )); do
    reap_finished true
  done
  IFS=$'\t' read -r family run_dir <<< "${entry}"
  gpu="$(free_gpu)"
  key="$(printf '%s' "${run_dir}" | sha256sum | cut -c1-12)"
  claim="${claim_root}/${key}"
  if ! mkdir "${claim}" 2>/dev/null; then
    echo "SKIP_CLAIMED family=${family} run=${run_dir}"
    continue
  fi
  label="${family}_${key}"
  echo "START ${label} gpu=${gpu} run=${run_dir}"
  (
    trap 'rmdir "${claim}" 2>/dev/null || true' EXIT
    env CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 PYTHON_BIN="${python_bin}" OUTPUT_ROOT="${output_root}" \
      PHASE2_RUNTIME_AMENDMENT="${runtime_amendment}" PHASE3_RUN_DIRS="${run_dir}" \
      PHASE3_RUN_TAG="${run_tag}_${key}" PHASE3_FINALIZE=false \
      ./scripts/run_phase3_collisions.sh \
      >"${log_root}/${label}.log" 2>&1
  ) &
  pids+=("$!")
  labels+=("${label}")
  gpu_for_pid+=("${gpu}")
  running=$((running + 1))
done

while (( running > 0 )); do
  reap_finished true
done

"${python_bin}" -m ribs.cli aggregate --output-root "${output_root}" --experiment phase2 \
  --include-families dimensional vib vq quantized autoencoder \
  --runtime-amendment "${runtime_amendment}" || failed=$((failed + 1))
"${python_bin}" -m ribs.cli render --experiment phase2 \
  --summary "${output_root}/phase2_summary.parquet" \
  --output-dir "${output_root}/figures_phase2" || failed=$((failed + 1))

(( failed == 0 ))
