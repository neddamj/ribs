#!/usr/bin/env bash
set -u -o pipefail

# Run only the registered Phase 2 attack-correctness superseding audits. This
# script never retrains models or removes/replaces an existing evaluation.
output_root="${OUTPUT_ROOT:-outputs}"
python_bin="${PYTHON_BIN:-python}"
amendment="${PHASE2_AUDIT_AMENDMENT:-configs/phase2_attack_audit_amendment_20260921.yaml}"
gpu_count="${GPU_COUNT:-4}"
run_tag="${PHASE2_AUDIT_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
export PYTHONPATH="${PYTHONPATH:-}:src"

if (( gpu_count < 1 )); then
  echo "GPU_COUNT must be positive" >&2
  exit 2
fi
if ! "${python_bin}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is required for the prescribed Phase 2 superseding audits" >&2
  exit 2
fi

log_root="${output_root}/phase2_attack_audit_logs/${run_tag}"
mkdir -p "${log_root}"
mapfile -t run_dirs < <(
  "${python_bin}" - "${output_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = json.loads((root / "phase2_acceptance.json").read_text())
for message in report.get("errors", []):
    prefix = "failed attack correctness diagnostics: "
    if message.startswith(prefix):
        run = Path(message[len(prefix):])
        if run.is_dir():
            existing = []
            for diagnostic in (run / "evaluations").glob(
                "attack-diagnostics-*-attempt*/diagnostics.json"
            ):
                try:
                    value = json.loads(diagnostic.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if value.get("audit_amendment_id") == "phase2-attack-audit-v1-20260921":
                    existing.append(diagnostic)
            if existing:
                continue
            print(run)
PY
)

declare -a pids=()
declare -a labels=()
running=0
failed=0

reap_one() {
  local index
  while true; do
    for index in "${!pids[@]}"; do
      [[ -n "${pids[index]+present}" ]] || continue
      if ! kill -0 "${pids[index]}" 2>/dev/null; then
        if wait "${pids[index]}"; then
          echo "DONE ${labels[index]}"
        else
          echo "FAILED ${labels[index]}" >&2
          failed=$((failed + 1))
        fi
        unset 'pids[index]' 'labels[index]'
        running=$((running - 1))
        return
      fi
    done
    sleep 5
  done
}

gpu=0
for run_dir in "${run_dirs[@]}"; do
  while (( running >= gpu_count )); do
    reap_one
  done
  family="$(basename "$(dirname "${run_dir}")")"
  label="${family}_$(basename "${run_dir}")"
  log="${log_root}/${label}.log"
  echo "START ${label} gpu=${gpu}"
  (
    env CUDA_VISIBLE_DEVICES="${gpu}" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
      PYTHONUNBUFFERED=1 \
      "${python_bin}" -m ribs.cli attack-diagnostics --run-dir "${run_dir}" \
      --diagnostic-samples 128 --diagnostic-tolerance 0.02 \
      --audit-amendment "${amendment}" >"${log}" 2>&1
  ) &
  pids+=($!)
  labels+=("${label}")
  running=$((running + 1))
  gpu=$(( (gpu + 1) % gpu_count ))
done
while (( running > 0 )); do
  reap_one
done

echo "AUDIT_LOG_ROOT ${log_root}"
(( failed == 0 ))
