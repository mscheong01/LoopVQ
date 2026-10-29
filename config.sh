#!/bin/bash
# Shared configuration for the serving and sweep scripts. Sourced, not run.
#
# Paths are repository-relative and experiment settings are environment-overridable.

# ---- paths ----
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO/scripts/arms.sh"
# canonical_method <name> [model]: a method name (loopvq_0.5bit, ...) or arm name -> the arm name
canonical_method() { arm_name "$1" "${2:-${MODEL:-}}"; }
SCRIPTS="$REPO/scripts"
ARTIFACTS="${ARTIFACTS:-$REPO/experiments/artifacts}"   # codebook bundles (calibration/)
WORK="${WORK:-$REPO/experiments/work}"                  # server logs and raw results
mkdir -p "$WORK"
MODELS_DIR="${MODELS_DIR:-$REPO/models}"                # Ouro-*, huginn-0125 checkpoints

# ---- conda envs ----
# The lm_eval client needs its own env because lm_eval 0.4.9 does not import under transformers
# 5.x, which the serving engine needs. The client talks HTTP and never touches a GPU.
ENV_SERVE="${ENV_SERVE:-loopvq-serve}"     # sglang (python/)
ENV_EVAL="${ENV_EVAL:-loopvq-eval}"        # lm_eval / evalplus client
CONDA_BASE="$(conda info --base 2>/dev/null)"
EVAL_PY="${EVAL_PY:-$CONDA_BASE/envs/$ENV_EVAL/bin/python3}"

# ---- GPU topology (auto) ----
# Run one replica per visible GPU by default. Set GPUS to pin a subset.
detect_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "${CUDA_VISIBLE_DEVICES//,/ }"; return
  fi
  local n; n=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  [[ "${n:-0}" -gt 0 ]] || n=1
  seq -s' ' 0 $((n - 1))
}
GPUS="${GPUS:-$(detect_gpus)}"
read -r -a _GPU_ARR <<< "$GPUS"
# Replicas = visible devices / tensor-parallel degree.
TP="${TP:-1}"
N_REPLICAS=$(( ${#_GPU_ARR[@]} / TP ))
[[ "$N_REPLICAS" -ge 1 ]] || die "TP=$TP exceeds the ${#_GPU_ARR[@]} visible GPU(s) in GPUS='$GPUS'"

# REASONING=1 switches the sweep to thinking-on mode: chat completions with
# chat_template_kwargs.enable_thinking, sampled decoding, a 32k generation budget, and the long-form
# task set.
REASONING="${REASONING:-0}"
CUSTOM_TASKS="${CUSTOM_TASKS:-$REPO/tasks}"
if [[ "$REASONING" == "1" ]]; then
  TASKS=(${REASONING_TASKS:-aime24})
  # 2 per replica: a 32k bf16 KV sequence of a looped model is several GB, so more in flight
  # thrashes the pool.
  REASON_PER_REPLICA="${REASON_PER_REPLICA:-2}"
  NUM_CONCURRENT="${NUM_CONCURRENT:-$(( REASON_PER_REPLICA * N_REPLICAS ))}"
  REASON_TEMP="${REASON_TEMP:-0.6}"
  REASON_TOP_P="${REASON_TOP_P:-0.95}"
  REASON_MAX_GEN="${REASON_MAX_GEN:-32768}"
  # lm_eval applies this timeout to the complete evaluation request queue.
  REASON_TIMEOUT="${REASON_TIMEOUT:-172800}"
else
  TASKS=(${SHORT_TASKS:-gsm8k_cot_3shot})
  SHORT_PER_REPLICA="${SHORT_PER_REPLICA:-16}"
  NUM_CONCURRENT="${NUM_CONCURRENT:-$(( SHORT_PER_REPLICA * N_REPLICAS ))}"
fi
# Server-side request cap per replica.
if [[ "$REASONING" == "1" ]]; then
  MAX_RUNNING="${MAX_RUNNING:-16}"
else
  MAX_RUNNING="${MAX_RUNNING:-$SHORT_PER_REPLICA}"
fi
# bf16 window kept by every quantized arm: the first PREFIX_TOKENS (sink) and the last RECENT_TOKENS.
if [[ "$REASONING" == "1" ]]; then
  PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
  RECENT_TOKENS="${RECENT_TOKENS:-256}"
else
  PREFIX_TOKENS="${PREFIX_TOKENS:-8}"
  RECENT_TOKENS="${RECENT_TOKENS:-64}"
fi
CHUNKED_PREFILL=-1          # no request splitting
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"

# ---- helpers ----
have_conda() { source "$CONDA_BASE/etc/profile.d/conda.sh"; }
say() { echo "[exp] $* $(date -Is)"; }
die() { echo "[exp] FAILED: $*" >&2; exit 1; }

# checkpoint path -> short tag used for artifact dirs and result filenames
model_tag() {
  case "$1" in
    # Thinking variants first: a bare *Ouro-1.4B* glob also matches them and would silently share
    # the base model's artifacts and results directory.
    # Huginn-0125 (recurrent depth, r = 32 by default): 2 + 4r + 2 = 132 KV slots (models/huginn.py)
    *huginn-0125*) echo huginn_3p5b ;;
    *Ouro-1.4B-Thinking*) echo ouro_1p4b_think ;;
    *Ouro-2.6B-Thinking*) echo ouro_2p6b_think ;;
    # Ouro: 24 layers x 4 loop steps = 96 KV slots for the 1.4B, 48 x 4 = 192 for the 2.6B.
    *Ouro-1.4B*) echo ouro_1p4b ;;
    *Ouro-2.6B*) echo ouro_2p6b ;;
    *) basename "$1" | tr 'A-Z.-' 'a-z__' ;;
  esac
}
