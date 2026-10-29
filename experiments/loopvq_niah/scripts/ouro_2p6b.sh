#!/bin/bash
# loopvq_niah, Ouro-2.6B (base model, no thinking): 4k/8k/16k/32k/64k x 3 seeds, bf16 + the four
# packed arms at 0.5 and 0.25 bit. Arms are the positional arguments (default: all nine).
#
# Rows are metered with Ouro's own tokenizer. The 64k column is built at 65536 - 256 so prompt + template +
# answer budget stay inside the native window -- no RoPE extrapolation in any cell.
# bf16 KV is 1.5 MB/token here (192 slots = 48 layers x 4 loops): ~50 GB at 32k, ~100 GB at 64k,
# so on 24 GB cards the bf16 row runs at TP=4 up to 32k and TP=8 at 64k. The quantized arms run at
# TP=1; their loop-coupled prefill hold is 1.15 MB/token (see .chain_niah_3090_2p6b.sh).
#
# 64 sink and 256 recent tokens are kept in bf16 (run_model.sh).
# LOOPVQ_MAX_TOKENS pins the packed pool so NIAH_MAX_RUNNING full-length requests are resident
# at once (a memory knob only; scores do not depend on it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NIAH_BUDGET_MODEL=${NIAH_BUDGET_MODEL:-${MODELS_DIR:-$HERE/../../../models}/Ouro-2.6B}
export NIAH_LEN_CAP=${NIAH_LEN_CAP:-65280}
# The driver asks the server for --context-length max_ctx + 1152; the rows themselves stay inside
# 65536 (NIAH_LEN_CAP), so lifting sglang's max_position_embeddings check changes no cell.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export ENV_SERVE=${ENV_SERVE:-loopvq-serve}
export NIAH_MAX_RUNNING=${NIAH_MAX_RUNNING:-4}
export LOOPVQ_MAX_TOKENS=${LOOPVQ_MAX_TOKENS:-$(( NIAH_MAX_RUNNING * (65536 + 128 + 1024) ))}
export CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$NIAH_MAX_RUNNING}
exec "$HERE/run_model.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-2.6B" "$@"
