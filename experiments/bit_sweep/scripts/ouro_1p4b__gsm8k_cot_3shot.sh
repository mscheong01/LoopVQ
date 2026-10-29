#!/bin/bash
# bit_sweep cell: Ouro-1.4B x gsm8k_cot_3shot. Arms are the positional arguments.
# Base-model protocol (completion format, greedy), 8 sink and 64 recent tokens kept in bf16.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_MODEL_EVAL=1 PREFIX_TOKENS=8 RECENT_TOKENS=64
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
exec "$HERE/run_cell.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-1.4B" "gsm8k_cot_3shot" "$@"
