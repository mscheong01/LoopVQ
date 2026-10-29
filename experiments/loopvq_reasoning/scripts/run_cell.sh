#!/bin/bash
# One loopvq_reasoning cell: (Ouro-Thinking model, benchmark) x the given arms, results in ../outputs.
#
#   ./run_cell.sh <model dir> aime24 bf16 tasq_loopvq_0.5bit
#   FORCE=1 ./run_cell.sh ...      # re-run
#
# Thinking-on protocol (config.sh REASONING branch, the chat template's enable_thinking=True),
# with the decoding of the Ouro-Thinking model card (temperature 1.0, top_p 0.7) and a 32k
# generation budget. The quantized arms keep 64 sink and 256 recent tokens in bf16. 3 seeds per
# cell (sampled decoding).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
export REASONING=1
export REASON_SEEDS="${REASON_SEEDS:-0 1 2}"
export REASON_TEMP="${REASON_TEMP:-1.0}" REASON_TOP_P="${REASON_TOP_P:-0.7}"
export REASON_MAX_GEN="${REASON_MAX_GEN:-32768}" REASON_CTX="${REASON_CTX:-40960}"
export PREFIX_TOKENS="${PREFIX_TOKENS:-64}" RECENT_TOKENS="${RECENT_TOKENS:-256}"
# a 32k chain can sit in the server queue for hours behind others; the proxy's 7200 s per-request
# cap would cut it off and score it as an empty answer
export RR_PROXY_TIMEOUT="${RR_PROXY_TIMEOUT:-172800}"
# A system torch's lib/ on LD_LIBRARY_PATH shadows the eval env's own libtorch ("torch._C has no
# attribute TensorBase"). Drop it; both envs carry their own.
export LD_LIBRARY_PATH=$(tr ':' '\n' <<< "${LD_LIBRARY_PATH:-}" | grep -v 'dist-packages/torch' | paste -sd:)
export CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
pe_run_cell "$HERE/../outputs" "${1:?usage: run_cell.sh <model> <bench> [arms...]}" \
            "${2:?usage: run_cell.sh <model> <bench> [arms...]}" \
            "${@:3}" || exit 1
