#!/bin/bash
# loopvq_efficiency: serving efficiency of Ouro-Thinking + LoopVQ. This sets the Ouro / LoopVQ
# defaults and runs scripts/run_efficiency.sh (bench_point.py -> sglang's bench_one_batch_server).
#
#   ./run.sh decode     # bs=1 decode latency at 1k / 8k / 32k (64k needs host-RAM offload: excluded)
#   ./run.sh thru       # throughput at bs 1 / 8 / 32 / 64 (+ the arm's capacity), 2048 in / 4096 out
#   ./run.sh prefill    # prefill latency (TTFT) at 4k / 16k / 32k
#   ./run.sh all        # the three panels in a row: the baseline pass
#   ./run.sh profile    # GPU-time breakdown per regime (decode bs1 1k/32k, prefill 16k, thru bs32)
#   CTXS=... BSS=... THRU_OUT=32768 ./run.sh ...  # denser grids / the final reasoning-length shape
#
#   VARIANT=opt_flush EXTRA_ENV="SGLANG_X=1" ./run.sh decode     # a kernel variant, kept apart
#   MODEL_TAG=2p6b ARMS="bf16 loopvq_0.5bit" ./run.sh thru
#
# ONE GPU for everything (GPU_ID, default 0): arms and variants are only comparable when they run
# on the same card, one after another.
#
# Arms: method names of scripts/arms.sh (bf16, loopvq_0.5bit, loopvq_0.25bit, cq_0.5bit, ...),
# mapped to the bundles the accuracy tables use.
# Results: outputs/<VARIANT>/<tag>__<panel>__<arm>__<key>.json, VARIANT defaults to "baseline".
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
EFF_RUN="$HERE/run_efficiency.sh"
PANEL="${1:?usage: run.sh <decode|thru|prefill|all|profile>}"

MODELS_DIR="${MODELS_DIR:-$(cd "$EXP/../.." && pwd)/models}"
case "${MODEL_TAG:-1p4b}" in
  1p4b) export MODEL="${MODEL:-$MODELS_DIR/Ouro-1.4B-Thinking}" ;;
  2p6b) export MODEL="${MODEL:-$MODELS_DIR/Ouro-2.6B-Thinking}" ;;
  *) echo "MODEL_TAG must be 1p4b or 2p6b" >&2; exit 2 ;;
esac

source "$EXP/../../scripts/arms.sh"
arms=()
for a in ${ARMS:-bf16 loopvq_0.5bit loopvq_0.25bit}; do arms+=("$(arm_name "$a" "$MODEL")"); done
export ARMS="${arms[*]}"

# KV window of the general-benchmark accuracy tables (loopvq_general / bit_sweep: 8 FP sink + 64
# recent; efficiency depends on the KV shape and the window, not the weights, so the base-model
# accuracy setting applies to Ouro-Thinking here). (At the reasoning tables' 64 / 256 the
# bf16 arena, 0.26 GB a request, outweighed the quantized pool at bs64.)
export PREFIX_TOKENS="${PREFIX_TOKENS:-8}" RECENT_TOKENS="${RECENT_TOKENS:-64}"
# The accuracy arms pin --max-total-tokens (run_sweep.sh arm_spec: 32768 / 16384); the server takes
# min(profiled, cap), so a huge cap means "the pool the memory fraction buys" -- the real capacity.
export LOOPVQ_MAX_TOKENS="${LOOPVQ_MAX_TOKENS:-1000000000}"
# Up to 32k: a 64k LoopVQ prompt only fits with the loop stash parked in host RAM, and an
# offloaded number is not an efficiency number. bf16 64k does not fit at all.
DEC_CTXS="${CTXS:-1024 8192 32768}"
PRE_CTXS="${CTXS:-4096 16384 32768}"
# Thru: 2048 in / 4096 out while optimising (a bs128 point is minutes, not hours); the final figure
# uses THRU_OUT=32768, the reasoning length the accuracy runs serve. The cell key carries the shape.
export THRU_IN="${THRU_IN:-2048}" THRU_OUT="${THRU_OUT:-4096}"
# A mixed-KV request also holds a bf16 HP arena whatever its length: at 8 / 64 that is (2x8 sink +
# 64+7 recent) x 96 slots ~ 0.07 GB on 1.4B, so bs 128 needs ~9 GB (at 64 / 256 it was 0.26 GB and
# bs 128 did not boot).
export BSS="${BSS:-1 2 4 8 16 32 64 128}"
export GPU_ID="${GPU_ID:-0}"

# ---- memory settings, per panel and arm (a memory knob only: none of these change what is timed) ----
# LoopVQ prefill holds 3 of the 4 loop steps' bf16 K/V for the whole prompt (576 KiB/token on 1.4B):
# 19 GB at 32k, 38 GB at 64k, on top of the pool. 32k prefill OOMed at mem-fraction 0.60 (21.8 GB
# left after the pool). A quantized pool is small, so the quant arms run at a LOW
# fraction (one 32k sequence is ~0.05 GB of vq pool), which leaves the room to the prefill.
# No host-RAM parking anywhere (SGLANG_LOOPVQ_STASH_GPU_GB unset).
#   decode : quant ladder from 0.30; bf16 the default ladder (32k needs ~0.65).
#   prefill: quant 0.30 (32k = 19 GB stash, measured to fit); bf16 0.85.
#   thru   : quant 0.85 like bf16: prefill batches of <= 8192 tokens stash
#            ~4.7 GB and 0.85 still leaves 8.5 GB free on a 48 GB card (bs64 checked).
QUANT_DEC_LADDER="${QUANT_DEC_LADDER:-0.30 0.40 0.50 0.60}"
QUANT_PRE_MEMFRAC="${QUANT_PRE_MEMFRAC:-0.30}"
QUANT_THRU_MEMFRAC="${QUANT_THRU_MEMFRAC:-0.85}"
BF16_MAX_CTX="${BF16_MAX_CTX:-32768}"

# eff <panel> [ctxs]: run_efficiency.sh once per arm with that arm's memory settings
eff() {
  local panel=$1 ctxs=${2:-} a c envs v
  for a in $ARMS; do
    envs=(ARMS="$a")
    if [[ $a == bf16 ]]; then
      if [[ $panel == decode ]]; then
        c=""; for v in $ctxs; do (( v <= BF16_MAX_CTX )) && c+="$v "; done; ctxs_a=$c
      else ctxs_a=$ctxs; fi
    else
      ctxs_a=$ctxs
      case $panel in
        decode)  envs+=(DEC_MEMFRAC_LADDER="$QUANT_DEC_LADDER") ;;
        prefill) envs+=(MEMFRAC="$QUANT_PRE_MEMFRAC") ;;
        thru)    envs+=(THRU_QUANT_MEMFRAC="$QUANT_THRU_MEMFRAC") ;;
      esac
    fi
    [[ $panel == thru || -n "$ctxs_a" ]] || { echo "[run] $a/$panel: no point fits (bf16 > $BF16_MAX_CTX)"; continue; }
    env "${envs[@]}" CTXS="$ctxs_a" bash "$EFF_RUN" "$panel" || return 1
  done
}

export OUTDIR="${OUTDIR:-$EXP/outputs/${VARIANT:-baseline}}"
mkdir -p "$OUTDIR"
# Record what produced this variant once, next to its cells.
{ echo "variant=${VARIANT:-baseline} panel=$PANEL utc=$(date -Is) host=$(hostname)"
  echo "arms=$ARMS model=$MODEL prefix=$PREFIX_TOKENS recent=$RECENT_TOKENS"
  echo "extra_env=${EXTRA_ENV:-} extra_pypath=${EXTRA_PYPATH:-}"
  echo "gpu=$GPU_ID $(nvidia-smi --query-gpu=name,uuid --format=csv,noheader -i "$GPU_ID" 2>/dev/null)"
  echo "commit=$(git -C "$EXP/../.." rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "$EXP/../.." status --porcelain python 2>/dev/null | wc -l)"
} >> "$OUTDIR/RUNS.txt"

# profile: the same servers and shapes, a different per-point script (BENCH_POINT).
#   profile -> outputs/<VARIANT>/profile/  torch-profiler GPU-time breakdown, one point per regime
case "$PANEL" in
  decode)  eff decode "$DEC_CTXS" ;;
  prefill) eff prefill "$PRE_CTXS" ;;
  thru)    eff thru ;;
  all)
    eff decode "$DEC_CTXS" && eff prefill "$PRE_CTXS" && eff thru || exit 1
    python3 "$HERE/summarize.py" "$OUTDIR" ;;
  profile)
    export BENCH_POINT="$HERE/profile_point.py" OUTDIR="$OUTDIR/profile"
    eff decode "${PROF_DEC_CTXS:-1024 32768}" || exit 1
    eff prefill "${PROF_PRE_CTXS:-16384}" || exit 1
    THRU_IN=2048 THRU_OUT=256 BSS="${PROF_BSS:-32}" eff thru || exit 1
    python3 "$HERE/summarize_profile.py" "$OUTDIR" ;;
  *) echo "unknown panel $PANEL" >&2; exit 2 ;;
esac
