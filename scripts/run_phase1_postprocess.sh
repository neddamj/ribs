#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-outputs}"
python_bin="${PYTHON_BIN:-/home/jmadden2/anaconda3/bin/python}"
expected_runs=19

while true; do
  completed="$(find "${output_root}" -mindepth 2 -maxdepth 2 \
    -type f -name COMPLETED 2>/dev/null | wc -l)"
  if [[ "${completed}" -ge "${expected_runs}" ]]; then
    break
  fi
  active_jobs="$(pgrep -f "ribs.cli train --config configs/phase1.yaml" | wc -l || true)"
  if [[ "${active_jobs}" -eq 0 ]]; then
    echo "Training queues ended with only ${completed}/${expected_runs} completed runs" >&2
    exit 1
  fi
  echo "Waiting for Phase 1 training: ${completed}/${expected_runs} completed $(date -Is)"
  sleep 300
done

mkdir -p "${output_root}/phase1_analysis_logs"
mapfile -t run_dirs < <(
  find "${output_root}/dimensional" "${output_root}/identity" \
    -mindepth 1 -maxdepth 1 -type d ! -name ".*" -print 2>/dev/null | sort
)
if [[ "${#run_dirs[@]}" -ne "${expected_runs}" ]]; then
  echo "Expected ${expected_runs} completed run directories, found ${#run_dirs[@]}" >&2
  exit 1
fi

process_run() {
  local gpu="$1"
  local run_dir="$2"
  local common_env=(
    CUBLAS_WORKSPACE_CONFIG=:4096:8
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1
    PYTHONUNBUFFERED=1
    CUDA_VISIBLE_DEVICES="${gpu}"
    PYTHONPATH=src
  )

  echo "START gpu=${gpu} run=${run_dir} $(date -Is)"
  env "${common_env[@]}" "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split development_tune
  env "${common_env[@]}" "${python_bin}" -m ribs.cli extract-latents --run-dir "${run_dir}" --split final
  env "${common_env[@]}" "${python_bin}" -m ribs.cli evaluate --run-dir "${run_dir}"
  env "${common_env[@]}" "${python_bin}" -m ribs.cli attack --run-dir "${run_dir}"
  env "${common_env[@]}" "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment geometry
  env "${common_env[@]}" "${python_bin}" -m ribs.cli analyze --run-dir "${run_dir}" --experiment invariance
  echo "DONE gpu=${gpu} run=${run_dir} $(date -Is)"
}

for gpu in 0 1 2 3; do
  (
    index="${gpu}"
    while [[ "${index}" -lt "${#run_dirs[@]}" ]]; do
      process_run "${gpu}" "${run_dirs[${index}]}"
      index=$((index + 4))
    done
  ) > "${output_root}/phase1_analysis_logs/gpu${gpu}.log" 2>&1 &
done
wait

PYTHONPATH=src "${python_bin}" -m ribs.cli aggregate --output-root "${output_root}"
PYTHONPATH=src "${python_bin}" -m ribs.cli validate-phase1 --output-root "${output_root}"
PYTHONPATH=src "${python_bin}" -m ribs.cli render --summary "${output_root}/phase1_summary.parquet" \
  --output-dir "${output_root}/figures"
