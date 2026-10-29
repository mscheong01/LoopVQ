#!/bin/bash
# One loopvq_general cell: (model, benchmark) x the given arms, results in ../outputs.
#
#   ./run_cell.sh <models>/Ouro-1.4B gsm8k_cot_3shot bf16 loopvq_0.5bit
#   FORCE=1 ./run_cell.sh ... ; REPLICATE=2 ./run_cell.sh ...   # re-run / add a replicate
#
# Safe to re-run on any node: an arm already present in outputs/ is reported and skipped, and if
# all of them are present the script exits without starting a server.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
export REASONING=0            # short-task protocol: greedy, single seed
pe_run_cell "$HERE/../outputs" "${1:?usage: run_cell.sh <model> <bench> [arms...]}" \
            "${2:?usage: run_cell.sh <model> <bench> [arms...]}" \
            "${@:3}" || exit 1
