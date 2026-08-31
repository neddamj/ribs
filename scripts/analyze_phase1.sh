#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-outputs}"
for family in dimensional identity; do
  for run_dir in "${output_root}"/"${family}"/*; do
    if [[ -f "${run_dir}/COMPLETED" ]]; then
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli extract-latents --run-dir "${run_dir}" --split development_tune
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli extract-latents --run-dir "${run_dir}" --split final
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli evaluate --run-dir "${run_dir}"
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli attack --run-dir "${run_dir}"
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli analyze --run-dir "${run_dir}" --experiment geometry
      PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli analyze --run-dir "${run_dir}" --experiment invariance
    fi
  done
done
PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli aggregate --output-root "${output_root}"
PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli validate-phase1 --output-root "${output_root}"
PYTHONPATH="${PYTHONPATH:-}:src" python -m ribs.cli render --summary "${output_root}/phase1_summary.parquet" --output-dir "${output_root}/figures"
