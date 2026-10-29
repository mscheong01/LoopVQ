#!/bin/bash
# loopvq_reasoning cell: Ouro-1.4B-Thinking x evalplus_humaneval. Arms are the positional arguments.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_PREFILL_TOKENS=${MAX_PREFILL_TOKENS:-8192}
# Quantized arms: MAX_RUNNING requests per replica on a pool of MAX_RUNNING x 34,816 tokens, so
# every request reaches prompt + 32k tokens without retraction. The counts fill a 48 GB card
# with a few GB of headroom (memory and scheduling knobs; results do not depend on them).
if [[ "$*" != bf16 ]]; then
  export MAX_RUNNING=${MAX_RUNNING:-20}
  export REASON_PER_REPLICA=${REASON_PER_REPLICA:-$(( MAX_RUNNING / 3 + 1 ))} CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$MAX_RUNNING}
  export LOOPVQ_MAX_TOKENS=${LOOPVQ_MAX_TOKENS:-$(( MAX_RUNNING * 34816 ))} LOOPVQ_MEM_FRACTION=${LOOPVQ_MEM_FRACTION:-0.85}
fi
exec "$HERE/run_cell.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-1.4B-Thinking" "evalplus_humaneval" "$@"
