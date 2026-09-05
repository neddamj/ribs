#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-outputs}"
python_bin="${PYTHON_BIN:-/home/jmadden2/anaconda3/bin/python}"
expected_runs=19

while true; do
  completed="$(find "${output_root}" -mindepth 3 -maxdepth 3 \
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
  {
    find "${output_root}/dimensional" -mindepth 1 -maxdepth 1 -type d \
      ! -name ".*" -print 2>/dev/null
    find "${output_root}/identity" -mindepth 1 -maxdepth 1 -type d \
      ! -name ".*" -print 2>/dev/null | while read -r candidate; do
        if [[ -f "${candidate}/resolved_config.yaml" ]] && "${python_bin}" -c \
          'import sys,yaml; x=yaml.safe_load(open(sys.argv[1])); raise SystemExit(0 if x.get("seed") == 0 else 1)' \
          "${candidate}/resolved_config.yaml"; then
          echo "${candidate}"
        fi
      done
  } | sort
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
  if [[ ! -f "${run_dir}/artifacts/latents_development_tune.safetensors" || \
        ! -f "${run_dir}/artifacts/latents_development_tune_index.parquet" ]]; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli extract-latents \
      --run-dir "${run_dir}" --split development_tune
  else
    echo "SKIP existing development_tune latents"
  fi
  if [[ ! -f "${run_dir}/artifacts/latents_final.safetensors" || \
        ! -f "${run_dir}/artifacts/latents_final_index.parquet" ]]; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli extract-latents \
      --run-dir "${run_dir}" --split final
  else
    echo "SKIP existing final latents"
  fi

  clean_count="$(find "${run_dir}/evaluations" -mindepth 2 -maxdepth 2 \
    -type f -name clean_final.parquet 2>/dev/null | while read -r path; do
      candidate="${path%/clean_final.parquet}"
      if [[ -f "${candidate}/COMPLETED" && -f "${candidate}/clean_final.json" ]] && \
         "${python_bin}" -c \
           'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("split") == "final" and x.get("max_samples") is None else 1)' \
           "${candidate}/clean_final.json"; then
        echo "${candidate}"
      fi
    done | wc -l)"
  if [[ "${clean_count}" -eq 0 ]]; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli evaluate --run-dir "${run_dir}"
  elif [[ "${clean_count}" -eq 1 ]]; then
    echo "SKIP existing clean final evaluation"
  else
    echo "Multiple completed clean final evaluations found: ${run_dir}" >&2
    return 1
  fi

  robustness_count="$(find "${run_dir}/evaluations" -mindepth 2 -maxdepth 2 \
    -type f -name input_pgd.parquet 2>/dev/null | while read -r path; do
      candidate="${path%/input_pgd.parquet}"
      if [[ -f "${candidate}/latent_pgd.parquet" && -f "${candidate}/COMPLETED" && \
            -f "${candidate}/config.json" ]] && \
         "${python_bin}" -c \
           'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("attack_protocol_version") == 2 and x.get("split") == "final" and x.get("max_samples") is None else 1)' \
           "${candidate}/config.json"; then
        echo "${candidate}"
      fi
    done | wc -l)"
  if [[ "${robustness_count}" -eq 0 ]]; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli attack --run-dir "${run_dir}"
  elif [[ "${robustness_count}" -eq 1 ]]; then
    echo "SKIP existing robustness evaluation"
  else
    echo "Multiple completed robustness evaluations found: ${run_dir}" >&2
    return 1
  fi

  if ! has_completed_analysis "${run_dir}" geometry geometry.json geometry_final.json || \
     ! has_completed_analysis "${run_dir}" geometry natural_collisions.parquet \
       natural_collisions_final.parquet; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment geometry
  else
    echo "SKIP existing geometry and natural-collision analysis"
  fi
  if ! has_completed_analysis "${run_dir}" invariance invariance.json invariance_final.json; then
    env "${common_env[@]}" "${python_bin}" -m ribs.cli analyze \
      --run-dir "${run_dir}" --experiment invariance
  else
    echo "SKIP existing invariance analysis"
  fi
  echo "DONE gpu=${gpu} run=${run_dir} $(date -Is)"
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
