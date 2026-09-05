#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-outputs}"
"$(dirname "$0")/run_phase1_postprocess.sh" "${output_root}"
