#!/bin/bash
# bit_sweep cell: Ouro-2.6B x math500_mv. Arms are the positional arguments.
# Base-model protocol (completion format, greedy), 8 sink and 64 recent tokens kept in bf16.
# Prefill runs in chunks of 2048 tokens for every arm: the loop-coupled prefill holds the K and V
# of (Tc - 1) x Lb slots until a block completes (a memory knob, no score effect).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_MODEL_EVAL=1 PREFIX_TOKENS=8 RECENT_TOKENS=64
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-2048}
exec "$HERE/run_cell.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-2.6B" "math500_mv" "$@"
