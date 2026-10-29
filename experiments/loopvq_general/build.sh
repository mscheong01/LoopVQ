#!/usr/bin/env bash
# Build the codebook bundles of the main tables for one Ouro model (base or Thinking):
#   CQ           cq8b_gG_gq          per-slot codebooks over G contiguous channels
#   TaSQ         loopvq8b_gG_tasq    per slot (Tc = 1), K = W -> N -> P; V plain CQ
#   LoopVQ       loopvq8b_gG_lvgq    K and V coded across the 4 loop steps -> _lvgqe2e after FT
#   TaSQ+LoopVQ  loopvq8b_gG_tqlv    TaSQ's W -> N -> P on K, then LoopVQ   -> _tqlve2e after FT
# G = 16 (0.5 bit) and 32 (0.25 bit). Recipe: GPQA main minus diamond, 16 x 2048 tokens (raw text,
# no chat template), Fisher k-means, 8-bit codes, FT 12 train / 4 held out, Adam 1e-3, 4 epochs.
# The Thinking models are refit on their own weights (loopvq_reasoning/build.sh).
#
#   MODEL=ouro_1p4b GPUS="0 1 2 3" bash experiments/loopvq_general/build.sh
#   Huginn-3.5B: experiments/loopvq_general/build_huginn.sh
set -u
MODEL=${MODEL:-ouro_1p4b}
read -r -a GP <<< "${GPUS:-0 1 2 3}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python3}   # the loopvq-build env (README): transformers 4.57.x, Ouro remote code breaks on 5.x
A=$R/experiments/artifacts; L=${LOGDIR:-$R/experiments/loopvq_general/logs}; mkdir -p $L
CC=${CALIB_CACHE_DIR:-$R/work/calib_cache}; mkdir -p $CC   # calibration buffers and TaSQ W, shared by all fits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $MODEL in
  ouro_1p4b) M=Ouro-1.4B ;; ouro_2p6b) M=Ouro-2.6B ;;
  ouro_1p4b_think) M=Ouro-1.4B-Thinking ;; ouro_2p6b_think) M=Ouro-2.6B-Thinking ;;
  *) echo "unknown MODEL $MODEL"; exit 1 ;;
esac
M=${MODELS_DIR:-$R/models}/$M
CG=$CC/${MODEL}_n16_s2048_p2048_seed0_gpqa_nodiamond.pt
QW=$CC/${MODEL}_tasq_w_gpqa_nodiamond.pt
D="--model $M --n-calib 16 --calib-seq-len 2048 --calib-positions 2048 --calib-data gpqa_nodiamond --calib-cache $CG --abits 8 --kmeans-iters 15"
cd $R/calibration

# calibration buffer and TaSQ W once, shared by every fit (--grad-ckpt: the 2.6B Fisher pass at 2048
# tokens does not fit 48 GB without it; recompute only, same gradients)
[ -f $CG ] || ( CUDA_VISIBLE_DEVICES=${GP[0]} $PY build_ouro_bundle.py $D --out-dir /dev/null --calib-only --grad-ckpt \
  > $L/${MODEL}_calib.log 2>&1; echo "calib rc=$?" ) &
[ -f $QW ] || ( CUDA_VISIBLE_DEVICES=${GP[1]} $PY tasq_qweights.py --model $M --calib-data gpqa_nodiamond \
  --n-calib 16 --calib-seq-len 2048 --out $QW > $L/${MODEL}_tasq_w.log 2>&1; echo "tasq_w rc=$?" ) &
wait
[ -f $CG ] || { echo "calib cache missing: $CG"; exit 1; }
[ -f $QW ] || { echo "TaSQ W missing: $QW"; exit 1; }

bld() { [ -f $A/${MODEL}_$3/vq_v_codebook.pt ] && { echo "have $3"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY build_ouro_bundle.py $D --group-dim $2 "${@:4}" --out-dir $A/${MODEL}_$3 > $L/${MODEL}_$3.log 2>&1; echo "build $3 rc=$?"; }
ft() { [ -f $A/${MODEL}_$2e2e/vq_v_codebook.pt ] && { echo "have $2e2e"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY finetune_codebook.py --model $M --src ${MODEL}_$2 --dst ${MODEL}_$2e2e \
  --calib-data gpqa_nodiamond --n-calib 16 --n-val 4 --seq-len 2048 --grad-ckpt --lr 1e-3 --epochs 4 \
  > $L/${MODEL}_ft_$2.log 2>&1; echo "ft $2 rc=$?"; }

TW="--k-norm pooled --k-cov --k-qweight $QW"
for G in 16 32; do
  bld ${GP[0]} $G cq8b_g${G}_gq --method cq &
  bld ${GP[1]} $G loopvq8b_g${G}_tasq --method loopvq --loops-coupled 1 $TW &
  ( bld ${GP[2]} $G loopvq8b_g${G}_lvgq --method loopvq --loops-coupled 4 --v-method loopvq
    ft ${GP[2]} loopvq8b_g${G}_lvgq ) &
  ( bld ${GP[3]} $G loopvq8b_g${G}_tqlv --method loopvq --loops-coupled 4 --v-method loopvq $TW
    ft ${GP[3]} loopvq8b_g${G}_tqlv ) &
  wait
done
python3 $R/calibration/check_bundles.py $(for G in 16 32; do echo ${MODEL}_cq8b_g${G}_gq ${MODEL}_loopvq8b_g${G}_tasq ${MODEL}_loopvq8b_g${G}_lvgqe2e ${MODEL}_loopvq8b_g${G}_tqlve2e; done) \
  && echo BUILD_DONE $MODEL || echo BUILD_FAILED $MODEL
