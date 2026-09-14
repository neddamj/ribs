#!/usr/bin/env bash
set -u -o pipefail

config_root="${1:-configs}"
decision_record="${2:?usage: scripts/run_phase2_parallel.sh [config-root] PHASE2_DECISION.yaml}"
output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
gpu_count="${GPU_COUNT:-4}"
export PYTHONPATH="${PYTHONPATH:-}:src"
run_tag="${PHASE2_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"

if (( gpu_count < 1 )); then
  echo "GPU_COUNT must be positive" >&2
  exit 2
fi

if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for the prescribed Phase 2 matrix; refusing CPU execution" >&2
  exit 2
fi
if ! "${python_bin}" -m ribs.cli analyze-identity-followup --output-root "${output_root}"; then
  echo "The three-seed identity follow-up must complete before Phase 2" >&2
  exit 2
fi

mkdir -p "${output_root}/phase2_training_logs"
echo "RUN_TAG ${run_tag}"
declare -a pids=()
declare -a labels=()
declare -a gpu_for_pid=()
running=0
failed=0

wait_for_slot() {
  while (( running >= gpu_count )); do
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
        unset 'gpu_for_pid[index]'
        unset 'pids[index]' 'labels[index]'
        running=$((running - 1))
        progressed=1
      fi
    done
    (( progressed )) || sleep 30
  done
}

launch_strength() {
  local gpu="$1"
  local family="$2"
  local parameter="$3"
  local value="$4"
  local config="${config_root}/${family}.yaml"
  local label="${family}_${parameter}_${value}"
  local log="${output_root}/phase2_training_logs/${label}_${run_tag}.log"
  echo "START ${label} gpu=${gpu}"
  (
    env CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 "${python_bin}" -m ribs.cli train-family-matrix \
      --decision-record "${decision_record}" --config "${config}" \
      --parameter "${parameter}" --values "${value}" --seeds 0 1 2 \
      --set data.num_workers=4 >"${log}" 2>&1
  ) &
  pids+=("$!")
  labels+=("${label}")
  gpu_for_pid+=("${gpu}")
  running=$((running + 1))
}

families=(vib vq quantized autoencoder)
parameters=(beta codebook_size bits dz)
values_vib=(0 0.0001 0.0003 0.001 0.003 0.01)
values_vq=(512 256 128 64 32 16)
values_quantized=(FP32 8 6 4 3 2)
values_autoencoder=(512 256 128 64 32)

for family_index in "${!families[@]}"; do
  family="${families[family_index]}"
  parameter="${parameters[family_index]}"
  eval "values=(\"\${values_${family}[@]}\")"
  for value in "${values[@]}"; do
    wait_for_slot
    gpu=""
    for candidate in $(seq 0 $((gpu_count - 1))); do
      occupied=0
      for assigned_gpu in "${gpu_for_pid[@]}"; do
        if [[ "${assigned_gpu}" == "${candidate}" ]]; then
          occupied=1
          break
        fi
      done
      if (( ! occupied )); then
        gpu="${candidate}"
        break
      fi
    done
    if [[ -z "${gpu}" ]]; then
      echo "No free GPU slot available for ${family}_${parameter}_${value}" >&2
      exit 2
    fi
    launch_strength "${gpu}" "${family}" "${parameter}" "${value}"
  done
done

while (( running > 0 )); do
  wait_for_slot
  sleep 1
done

if (( failed )); then
  echo "Phase 2 training ended with ${failed} failed queued jobs; see ${output_root}/phase2_training_logs" >&2
  exit 1
fi

PYTHON_BIN="${python_bin}" OUTPUT_ROOT="${output_root}" \
  GPU_COUNT="${gpu_count}" "$(dirname "$0")/run_phase2_postprocess_parallel.sh" \
  "${config_root}" "${decision_record}"
