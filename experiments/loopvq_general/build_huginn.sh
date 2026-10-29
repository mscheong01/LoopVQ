#!/usr/bin/env bash
# Build the Huginn-3.5B bundles of the main table (r = 32 recurrences, head_dim 96):
#   CQ           cq8b_gG_gq            per-slot codebooks over G contiguous channels
#   TaSQ         loopvq8b_gG_tasq      per slot (Tc = 1), K = W -> N -> P; V plain CQ
#   LoopVQ       loopvq8b_gG_lvgqt4    K and V coded across Tc = 4 recurrences (the 32 form eight
#                                      blocks) -> _lvgqt4e2e after FT
#   TaSQ+LoopVQ  loopvq8b_gG_tqlvt4    TaSQ's W -> N -> P on K, then LoopVQ -> _tqlvt4e2e after FT
# The prelude and coda have no loop axis and are coded per layer as in CQ. G = 16 / 32. Same recipe
# as build.sh (GPQA main minus diamond, 16 x 2048 tokens, 8-bit codes, FT 12 / 4, Adam 1e-3,
# 4 epochs); the Fisher pass keeps 1024 positions per sequence.
#   GPUS="0 1 2 3" bash experiments/loopvq_general/build_huginn.sh
set -u
MODEL=huginn_3p5b
read -r -a GP <<< "${GPUS:-0 1 2 3}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python3}   # the loopvq-build env (README)
A=$R/experiments/artifacts; L=${LOGDIR:-$R/experiments/loopvq_general/logs}; mkdir -p $L
CC=${CALIB_CACHE_DIR:-$R/work/calib_cache}; mkdir -p $CC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${MODELS_DIR:-$R/models}/huginn-0125
CG=$CC/${MODEL}_r32_n16_s2048_p1024_seed0_gpqa_nodiamond.pt
QW=$CC/${MODEL}_r32_tasq_w_gpqa_nodiamond.pt
D="--model $M --n-calib 16 --calib-seq-len 2048 --calib-positions 1024 --calib-data gpqa_nodiamond --calib-cache $CG --abits 8 --kmeans-iters 15"
cd $R/calibration

# Huginn's calibration buffer comes from its functional forward (huginn_torch.py), not the HF capture
[ -f $CG ] || ( CUDA_VISIBLE_DEVICES=${GP[0]} $PY huginn_torch.py calib --model $M --out $CG --n-calib 16 \
  --calib-seq-len 2048 --calib-positions 1024 --calib-data gpqa_nodiamond > $L/${MODEL}_calib.log 2>&1; echo "calib rc=$?" ) &
[ -f $QW ] || ( CUDA_VISIBLE_DEVICES=${GP[1]} $PY tasq_qweights.py --model $M --calib-data gpqa_nodiamond \
  --n-calib 16 --calib-seq-len 2048 --out $QW > $L/${MODEL}_tasq_w.log 2>&1; echo "tasq_w rc=$?" ) &
wait
[ -f $CG ] || { echo "calib cache missing: $CG"; exit 1; }
[ -f $QW ] || { echo "TaSQ W missing: $QW"; exit 1; }

bld() { [ -f $A/${MODEL}_$3/vq_v_codebook.pt ] && { echo "have $3"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY build_ouro_bundle.py $D --group-dim $2 "${@:4}" --out-dir $A/${MODEL}_$3 > $L/${MODEL}_$3.log 2>&1; echo "build $3 rc=$?"; }
ft() { [ -f $A/${MODEL}_$2e2e/vq_v_codebook.pt ] && { echo "have $2e2e"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY huginn_ft.py --model $M --src ${MODEL}_$2 --dst ${MODEL}_$2e2e > $L/${MODEL}_ft_$2.log 2>&1; echo "ft $2 rc=$?"; }

TW="--k-norm pooled --k-cov --k-qweight $QW"
for G in 16 32; do
  bld ${GP[0]} $G cq8b_g${G}_gq --method cq &
  bld ${GP[1]} $G loopvq8b_g${G}_tasq --method loopvq --loops-coupled 1 $TW &
  ( bld ${GP[2]} $G loopvq8b_g${G}_lvgqt4 --method loopvq --loops-coupled 4 --v-method loopvq
    ft ${GP[2]} loopvq8b_g${G}_lvgqt4 ) &
  ( bld ${GP[3]} $G loopvq8b_g${G}_tqlvt4 --method loopvq --loops-coupled 4 --v-method loopvq $TW
    ft ${GP[3]} loopvq8b_g${G}_tqlvt4 ) &
  wait
done
python3 $R/calibration/check_bundles.py $(for G in 16 32; do echo ${MODEL}_cq8b_g${G}_gq ${MODEL}_loopvq8b_g${G}_tasq ${MODEL}_loopvq8b_g${G}_lvgqt4e2e ${MODEL}_loopvq8b_g${G}_tqlvt4e2e; done) \
  && echo BUILD_DONE $MODEL || echo BUILD_FAILED $MODEL
