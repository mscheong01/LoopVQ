#!/bin/bash
# loopvq_longbench, Ouro-1.4B (base model, no thinking): LongBench v1, bf16 + the four packed arms at
# 0.5 and 0.25 bit. Arms are the positional arguments (default: all nine). Serving env as loopvq_niah.
#
# Rows are metered with Ouro's own tokenizer. The 64k column is built at 65536 - 256 so prompt + template +
# answer budget stay inside the native window -- no RoPE extrapolation in any cell.
# bf16 KV is 768 KB/token here (96 slots), so the bf16 row does not fit at TP=1 past ~16k on
# 24 GB cards: run it with TP=4 (run_all.sh does). The quantized arms run at TP=1.
#
# 64 sink and 256 recent tokens are kept in bf16 (run_model.sh).
# LOOPVQ_MAX_TOKENS pins the packed pool so NIAH_MAX_RUNNING full-length requests are resident
# at once (a memory knob only; scores do not depend on it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NIAH_BUDGET_MODEL=${NIAH_BUDGET_MODEL:-${MODELS_DIR:-$HERE/../../../models}/Ouro-1.4B}
export NIAH_LEN_CAP=${NIAH_LEN_CAP:-65280}
# The driver asks the server for --context-length max_ctx + 1152; the rows themselves stay inside
# 65536 (NIAH_LEN_CAP), so lifting sglang's max_position_embeddings check changes no cell.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export ENV_SERVE=${ENV_SERVE:-loopvq-serve}
export NIAH_MAX_RUNNING=${NIAH_MAX_RUNNING:-4}
export LOOPVQ_MAX_TOKENS=${LOOPVQ_MAX_TOKENS:-$(( NIAH_MAX_RUNNING * (65536 + 128 + 1024) ))}
export CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$NIAH_MAX_RUNNING}
exec "$HERE/run_model.sh" "${MODELS_DIR:-$HERE/../../../models}/Ouro-1.4B" "$@"
