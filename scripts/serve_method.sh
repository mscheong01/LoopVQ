#!/usr/bin/env bash
# Serve one (model, method) arm with the vendored engine in ../python.
#
#   scripts/serve_method.sh MODEL METHOD ARTIFACT_DIR [extra sglang args...]
#
# METHOD:
#   bf16        stock BF16 KV cache (the reference arm)
#   vq          the packed VQ KV cache every quantized arm runs on: bf16 sink and recent
#               windows, codebook indices for every other token (CQ, TaSQ, LoopVQ bundles)
#
# ARTIFACT_DIR must contain, for vq (calibration/build_ouro_bundle.py writes all four):
#   k_rotation_qqt_r_h_pbr.pt  v_rotation_sst_r_h_pbr.pt  codebook.pt  vq_v_codebook.pt
#
# Everything after ARTIFACT_DIR is passed to sglang.launch_server verbatim
# (--port, --context-length, --mem-fraction-static, --disable-radix-cache, ...).
#
# Env knobs (all optional):
#   PREFILL_BACKEND=fa3|triton   fa3 needs Hopper+; triton works everywhere
#   TP=1                         tensor parallel size
#   KV_SPLITS                    decode split-K (default: 48 for vq, 8 for bf16)
#   MAX_REQS=16                  --max-running-requests
#   MAX_TOKENS                   pin --max-total-tokens (default: derive from
#                                mem-fraction; pinning erases the capacity
#                                advantage the quantized pool exists to buy)
#   RADIX_CACHE=0|1              prefix cache (default off: eval rows are unique;
#                                prefix-reuse benchmarks set 1)
#   QUANT_GROUP_SIZE             scale group (default: the head dim, 128; 96 for Huginn)
#   SERVE_PYTHON=python3         interpreter of the serving env
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODEL METHOD ARTIFACT_DIR [sglang serve arguments...]" >&2
  exit 2
fi

model=$1
method=$2
artifact_dir=$3
shift 3

repo="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$repo/python:${PYTHONPATH:-}"

unset SGLANG_ENABLE_MIXED_KV_WINDOWS SGLANG_VQ_CODEBOOK_PATH
unset SGLANG_OSCAR_K_ROTATION_PATH SGLANG_OSCAR_V_ROTATION_PATH

# Long-context evals exceed some models' config-derived limit (RoPE
# extrapolation, matching the HF-harness behaviour at 64K+).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# expandable_segments lets the allocator return freed segments to the driver,
# which the capacity-sized quant pools rely on. Incompatible with the custom
# all-reduce's cudaIpc export under TP (vllm#42609), so TP>1 drops it.
if [[ "${TP:-1}" == "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi

qgs_default=128
# Huginn's head_dim is 96: one scale group per head, as 128 is for the D = 128 models
[[ "$model" == *huginn-0125* ]] && qgs_default=96

args=(
  --model-path "$model"
  --trust-remote-code
  --host 127.0.0.1
  --prefill-attention-backend "${PREFILL_BACKEND:-fa3}"
  --decode-attention-backend triton
  --sampling-backend pytorch
  --tp-size "${TP:-1}"
  --max-running-requests "${MAX_REQS:-16}"
)
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-total-tokens "$MAX_TOKENS")
# Huginn ships fp32 weights (config torch_dtype float32): serve bf16 like every other model here,
# at its trained 4096 context (sglang's derived default for this custom config is 2048).
if [[ "$model" == *huginn-0125* ]]; then
  args+=(--dtype bfloat16 --context-length 4096)
  export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
fi
[[ "${RADIX_CACHE:-0}" == "1" ]] || args+=(--disable-radix-cache)

kv_splits_default=8

case "$method" in
  bf16)
    ;;
  vq)
    for f in k_rotation_qqt_r_h_pbr.pt v_rotation_sst_r_h_pbr.pt; do
      if [[ ! -f "$artifact_dir/$f" ]]; then
        echo "missing rotation bundle: $artifact_dir/$f" >&2
        exit 2
      fi
    done
    export SGLANG_ENABLE_MIXED_KV_WINDOWS=1
    export SGLANG_OSCAR_K_ROTATION_PATH="$artifact_dir/k_rotation_qqt_r_h_pbr.pt"
    export SGLANG_OSCAR_V_ROTATION_PATH="$artifact_dir/v_rotation_sst_r_h_pbr.pt"
    # Clipping belongs to the engine's scalar quantizer; 1.0 turns it off (the VQ arms never
    # use it).
    export SGLANG_OSCAR_K_CLIP_RATIO=1.0
    export SGLANG_OSCAR_V_CLIP_RATIO=1.0
    export SGLANG_MIXED_KV_PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
    export SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS:-256}"
    export SGLANG_MIXED_KV_HP_DTYPE=bfloat16
    export SGLANG_MIXED_KV_SCALE_DTYPE="${SCALE_DTYPE:-float16}"
    export SGLANG_SWA_KEEP_PREFIX_TAIL="${SGLANG_SWA_KEEP_PREFIX_TAIL:-1}"   # inert on dense models
    args+=(--kv-cache-dtype int2)
    qgs="${QUANT_GROUP_SIZE:-$qgs_default}"
    [[ "$qgs" != "0" ]] && args+=(--kv-cache-quant-group-size "$qgs")
    if [[ "$method" == vq ]]; then
      for f in codebook.pt vq_v_codebook.pt; do
        if [[ ! -f "$artifact_dir/$f" ]]; then
          echo "missing VQ codebook: $artifact_dir/$f" >&2
          exit 2
        fi
      done
      export SGLANG_VQ_CODEBOOK_PATH="$artifact_dir/codebook.pt"
      export SGLANG_VQ_V_CODEBOOK_PATH="$artifact_dir/vq_v_codebook.pt"
      # keys are stored before RoPE (the decode kernel rotates them), as the bundles are built
      export SGLANG_VQ_PRE_ROPE=1
      # looped models are full MHA (one query head per KV head)
      export SGLANG_VQ2_ALLOW_MHA=1
      kv_splits_default=48
    fi
    ;;
  *)
    echo "unsupported method: $method" >&2
    exit 2
    ;;
esac

args+=(--triton-attention-num-kv-splits "${KV_SPLITS:-$kv_splits_default}")

exec "${SERVE_PYTHON:-python3}" -m sglang.launch_server "${args[@]}" "$@"
