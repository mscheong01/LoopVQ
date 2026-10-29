#!/bin/bash
# loopvq_general cell: Huginn-3.5B x evalplus_mbpp. Arms are the positional arguments.
# Huginn-0125 at r = 32 recurrences (2 + 4r + 2 = 132 KV slots).
# Base-model protocol (completion format, greedy), 8 sink and 64 recent tokens kept in bf16.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_MODEL_EVAL=1 PREFIX_TOKENS=8 RECENT_TOKENS=64 SGLANG_HUGINN_STEPS=32
# Memory knobs (no score effect) for a 24 GB card, since a Huginn token is 2.6 MB of bf16 KV:
# prefill chunks of 1024 tokens, and a 256-token bf16 prefix pool (the default reserves 5.3 GB
# for prefix caching, which this protocol does not use).
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-1024}
export SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS=${SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS:-256}
exec "$HERE/run_cell.sh" "${MODELS_DIR:-$HERE/../../../models}/huginn-0125" "evalplus_mbpp" "$@"
