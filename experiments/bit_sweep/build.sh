#!/usr/bin/env bash
# Build the 1 / 2 / 4-bit bundles of the bit sweep (0.25 / 0.5 bit are loopvq_general's).
#   CQ      cq8b_gG_gq        per-slot codebooks over G contiguous channels
#   LoopVQ  loopvq8b_gG_lvgq  K and V coded loop axis first: Tc = min(G, T) steps x G/Tc channels
#           -> loopvq8b_gG_lvgqe2e after loop-aware fine-tuning
#   TaSQ+LoopVQ  loopvq8b_g8_tqlv -> _tqlve2e, 1 bit only (TaSQ's W -> N -> P on K, then LoopVQ + FT)
#   TaSQ    loopvq8b_gG_tasq  per slot (Tc = 1), K = W -> N -> P (RoPE pairs atomic); V plain CQ
# Same recipe as loopvq_general: GPQA main minus diamond, 16 x 2048 tokens, Fisher k-means, 8-bit
# codes, FT 12 train / 4 held out, Adam 1e-3, 4 epochs.
#   MODEL=ouro_2p6b GPUS="0 1 2 3" bash experiments/bit_sweep/build.sh
set -u
MODEL=${MODEL:-ouro_2p6b}
read -r -a GP <<< "${GPUS:-0 1 2 3}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python3}   # the loopvq-build env (README): transformers 4.57.x, Ouro remote code breaks on 5.x
A=$R/experiments/artifacts; L=$R/experiments/bit_sweep/logs; mkdir -p $L
CC=${CALIB_CACHE_DIR:-$R/work/calib_cache}; mkdir -p $CC   # calibration buffers and TaSQ W, shared by all fits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $MODEL in ouro_2p6b) M=Ouro-2.6B ;; ouro_1p4b) M=Ouro-1.4B ;; esac
M=${MODELS_DIR:-$R/models}/$M
CG=$CC/${MODEL}_n16_s2048_p2048_seed0_gpqa_nodiamond.pt
QW=$CC/${MODEL}_tasq_w_gpqa_nodiamond.pt
[ -f $CG ] || { echo "calib cache missing: $CG (run loopvq_general/build.sh first)"; exit 1; }
[ -f $QW ] || { echo "TaSQ W missing: $QW"; exit 1; }
D="--model $M --n-calib 16 --calib-seq-len 2048 --calib-positions 2048 --calib-data gpqa_nodiamond --calib-cache $CG --abits 8 --kmeans-iters 15"
cd $R/calibration
bld() { [ -f $A/${MODEL}_$3/vq_v_codebook.pt ] && { echo "have $3"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY build_ouro_bundle.py $D --group-dim $2 "${@:4}" --out-dir $A/${MODEL}_$3 > $L/${MODEL}_$3.log 2>&1; echo "build $3 rc=$?"; }
ft() { [ -f $A/${MODEL}_$2e2e/vq_v_codebook.pt ] && { echo "have $2e2e"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY finetune_codebook.py --model $M --src ${MODEL}_$2 --dst ${MODEL}_$2e2e \
  --calib-data gpqa_nodiamond --n-calib 16 --n-val 4 --seq-len 2048 --grad-ckpt --lr 1e-3 --epochs 4 \
  --rep-chunk ${3:-512} > $L/${MODEL}_ft_$2.log 2>&1; echo "ft $2 rc=$?"; }
# G = 2 / 4 have c = 1: 128 code groups per slot, and the served-cache rebuild at 512 tokens per
# chunk runs out of memory for 2.6B on 24 GB. 128 tokens per chunk; same gradient (chunking only).
tc() { echo $(( $1 < 4 ? $1 : 4 )); }
i=0; for G in 2 4 8; do
  bld ${GP[$i]} $G cq8b_g${G}_gq --method cq &
  i=$(( (i + 1) % ${#GP[@]} ))
  ( bld ${GP[$i]} $G loopvq8b_g${G}_lvgq --method loopvq --loops-coupled $(tc $G) --v-method loopvq ) &
  i=$(( (i + 1) % ${#GP[@]} ))
  (( i == 0 )) && wait
done; wait
i=0; for G in 2 4 8; do ft ${GP[$i]} loopvq8b_g${G}_lvgq $( (( G < 8 )) && echo 128 ) & i=$((i + 1)); done
bld ${GP[3]} 8 loopvq8b_g8_tasq --method loopvq --loops-coupled 1 --k-norm pooled --k-cov --k-qweight $QW &
wait
bld ${GP[0]} 4 loopvq8b_g4_tasq --method loopvq --loops-coupled 1 --k-norm pooled --k-cov --k-qweight $QW &
bld ${GP[1]} 2 loopvq8b_g2_tasq --method loopvq --loops-coupled 1 --k-norm pooled --k-cov --k-qweight $QW &
# TaSQ+LoopVQ only at <= 1 bit: G = 2 / 4 leave c = 1 channel per step and P (RoPE pairs) no
# longer fits a group. 0.5 / 0.25 bit are loopvq_general's tqlve2e.
( bld ${GP[2]} 8 loopvq8b_g8_tqlv --method loopvq --loops-coupled 4 --v-method loopvq --k-norm pooled --k-cov --k-qweight $QW
  ft ${GP[2]} loopvq8b_g8_tqlv ) &
wait
python3 $R/calibration/check_bundles.py $(for G in 2 4 8; do echo ${MODEL}_cq8b_g${G}_gq ${MODEL}_loopvq8b_g${G}_lvgqe2e ${MODEL}_loopvq8b_g${G}_tasq; done) ${MODEL}_loopvq8b_g8_tqlve2e \
  && echo BUILD_DONE $MODEL || echo BUILD_FAILED $MODEL
