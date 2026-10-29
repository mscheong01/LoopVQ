#!/bin/bash
# loopvq_reasoning cell: Ouro-2.6B-Thinking x aime25. Arms are the positional arguments.
# Prefill runs in chunks of 2048 tokens for every arm: the loop-coupled prefill holds the K and V
# of (Tc - 1) x Lb slots until a block completes (a memory knob, no score effect).
# bf16 runs at TP=2: with 192 KV slots, one 32k sequence needs ~51 GB of KV.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-2048}
[[ "$*" == bf16 ]] && export TP=${TP:-2}
# Quantized arms: MAX_RUNNING requests per replica on a pool of MAX_RUNNING x 34,816 tokens, so
# every request reaches prompt + 32k tokens without retraction. The counts fill a 48 GB card
# with a few GB of headroom (memory and scheduling knobs; results do not depend on them).
# SLOTS_G16 / SLOTS_G32 resize for other cards (12 / 17 on an A100 40GB).
if [[ "$*" != bf16 ]]; then
  export MAX_RUNNING=${MAX_RUNNING:-$( [[ "$*" == *_g16_* ]] && echo ${SLOTS_G16:-13} || echo ${SLOTS_G32:-16} )}
  export REASON_PER_REPLICA=${REASON_PER_REPLICA:-$(( MAX_RUNNING / 3 + 1 ))} CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$MAX_RUNNING}
  export LOOPVQ_MAX_TOKENS=${LOOPVQ_MAX_TOKENS:-$(( MAX_RUNNING * 34816 ))} LOOPVQ_MEM_FRACTION=${LOOPVQ_MEM_FRACTION:-0.85}
fi
exec "$HERE/run_cell.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-2.6B-Thinking" "aime25" "$@"
