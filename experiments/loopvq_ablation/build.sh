#!/usr/bin/env bash
# Build the loopvq_ablation bundles (Ouro-1.4B base). Two studies:
#
# (1) Component ablation, G = 16 / 32 (0.5 / 0.25 bit): each of LoopVQ's two components alone,
#     on top of CQ and of TaSQ. Arms (experiments/artifacts/ouro_1p4b_<arm>):
#       base         +loop coupling         +loop-aware FT            +both
#       cq8b_gG_gq   loopvq8b_gG_lvgq       loopvq8b_gG_cq1e2e        loopvq8b_gG_lvgqe2e
#       _tasq        loopvq8b_gG_tqlv       loopvq8b_gG_tasq1e2e      loopvq8b_gG_tqlve2e
#     FT needs the loop-coupled bundle format, so the FT-only arms start from Tc = 1 bundles
#     (loopvq8b_gG_cq1 = CQ, loopvq8b_gG_tasq1 = TaSQ with V coded per slot like CQ); those two
#     sources are evaluated too, to check they match cq8b_gG_gq / _tasq.
#     Every arm but cq1 / tasq1 / their FT exists already (loopvq_general).
#
# (2) Calibration-set sensitivity, G = 32 (0.25 bit), all four methods refit on another corpus:
#       D = wt (WikiText-2 train) / c4 (C4 en) / sp (SlimPajama-6B)
#       cq8b_g32_D   loopvq8b_g32_tasq_D   loopvq8b_g32_lv_De2e   loopvq8b_g32_tqlv_De2e
#     Same recipe as the main runs otherwise (16 x 2048 tokens, Fisher k-means, 8-bit codes,
#     FT 12 train / 4 held out on the same corpus, Adam 1e-3, 4 epochs); TaSQ's W is refit too.
#     GPQA (main minus diamond) is the main-table calibration.
#
# (3) Calibration-set size, G = 32, GPQA with 8 / 32 sequences of 2048 tokens (main: 16):
#       loopvq8b_g32_lv_nNe2e   loopvq8b_g32_tqlv_nNe2e      (FT holds out N/4 sequences)
#
# (4) FT objective, G = 16 / 32: the loop-aware objective (all loop steps read the codes, as served)
#     against a per-step one (finetune_codebook.py --objective per-step: sum over t of the KL with
#     only step t quantized), from the same k-means bundles and with the same recipe otherwise:
#       loopvq8b_gG_lvgqps   loopvq8b_gG_tqlvps        (vs loopvq8b_gG_lvgqe2e / _tqlve2e)
#
#   GPUS="0 1 2 3 4 5 6 7" bash experiments/loopvq_ablation/build.sh
#   ONLY=ftobj bash experiments/loopvq_ablation/build.sh     # study (4) alone
set -u
MODEL=ouro_1p4b
read -r -a GP <<< "${GPUS:-0 1 2 3 4 5 6 7}"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python3}   # the loopvq-build env (README): transformers 4.57.x, Ouro remote code breaks on 5.x
A=$R/experiments/artifacts; L=$R/experiments/loopvq_ablation/logs; mkdir -p $L
CC=${CALIB_CACHE_DIR:-$R/work/calib_cache}; mkdir -p $CC   # calibration buffers and TaSQ W, shared by all fits
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${MODELS_DIR:-$R/models}/Ouro-1.4B
cd $R/calibration

# NC = calibration sequences (16 everywhere but study 3); FT holds out a quarter of them
NC=16
cg() { echo $CC/${MODEL}_n${NC}_s2048_p2048_seed0$( [ $1 = wikitext ] || echo _$1 ).pt; }
qw() { echo $CC/${MODEL}_tasq_w_$1$( [ $NC = 16 ] || echo _n$NC ).pt; }
opts() { echo "--model $M --n-calib $NC --calib-seq-len 2048 --calib-positions 2048 --calib-data $1 --calib-cache $(cg $1) --abits 8 --kmeans-iters 15"; }
# bld <gpu> <corpus> <G> <arm> <builder args...>
bld() { [ -f $A/${MODEL}_$4/vq_v_codebook.pt ] && { echo "have $4"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY build_ouro_bundle.py $(opts $2) --group-dim $3 "${@:5}" --out-dir $A/${MODEL}_$4 > $L/$4.log 2>&1; echo "build $4 rc=$?"; }
# ft <gpu> <corpus> <src arm>   -> <src arm>e2e
ft() { [ -f $A/${MODEL}_$3e2e/vq_v_codebook.pt ] && { echo "have $3e2e"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY finetune_codebook.py --model $M --src ${MODEL}_$3 --dst ${MODEL}_$3e2e \
  --calib-data $2 --n-calib $NC --n-val $((NC / 4)) --seq-len 2048 --grad-ckpt --lr 1e-3 --epochs 4 \
  > $L/ft_$3.log 2>&1; echo "ft $3 rc=$?"; }
# ftps <gpu> <src arm>   -> <src arm>ps (per-step objective, GPQA)
ftps() { [ -f $A/${MODEL}_$2ps/vq_v_codebook.pt ] && { echo "have $2ps"; return; }
  CUDA_VISIBLE_DEVICES=$1 $PY finetune_codebook.py --model $M --src ${MODEL}_$2 --dst ${MODEL}_$2ps \
  --calib-data gpqa_nodiamond --n-calib $NC --n-val $((NC / 4)) --seq-len 2048 --grad-ckpt --lr 1e-3 \
  --epochs 4 --objective per-step > $L/ftps_$2.log 2>&1; echo "ftps $2 rc=$?"; }
FTOBJ=""; for G in 16 32; do FTOBJ="$FTOBJ loopvq8b_g${G}_lvgqps loopvq8b_g${G}_tqlvps"; done
ftobj() { local i=0
  for G in 16 32; do for s in lvgq tqlv; do ftps ${GP[$i]} loopvq8b_g${G}_$s & i=$((i + 1)); done; done
  wait; }
if [ "${ONLY:-}" = ftobj ]; then
  ftobj
  python3 $R/calibration/check_bundles.py $(for a in $FTOBJ; do echo ${MODEL}_$a; done) \
    && echo BUILD_DONE || echo BUILD_FAILED
  exit
fi
tw() { echo "--k-norm pooled --k-cov --k-qweight $(qw $1)"; }

# (1) component ablation: GPQA buffer and W exist (loopvq_general)
[ -f $(cg gpqa_nodiamond) ] && [ -f $(qw gpqa_nodiamond) ] || { echo "GPQA calibration missing"; exit 1; }
i=0
for G in 16 32; do
  ( bld ${GP[$i]} gpqa_nodiamond $G loopvq8b_g${G}_cq1 --method loopvq --loops-coupled 1 --v-method loopvq
    ft ${GP[$i]} gpqa_nodiamond loopvq8b_g${G}_cq1 ) &
  i=$((i + 1))
  ( bld ${GP[$i]} gpqa_nodiamond $G loopvq8b_g${G}_tasq1 --method loopvq --loops-coupled 1 --v-method loopvq $(tw gpqa_nodiamond)
    ft ${GP[$i]} gpqa_nodiamond loopvq8b_g${G}_tasq1 ) &
  i=$((i + 1))
done

# (2) calibration corpora: one buffer at a time (each holds ~51 GB of RAM while it is written),
# W alongside; then the four fits per corpus, FT after the k-means fit
corpus() { local g=$1 c=$2 t=$3
  [ -f $(qw $c) ] || CUDA_VISIBLE_DEVICES=$g $PY tasq_qweights.py --model $M --calib-data $c \
    --n-calib $NC --calib-seq-len 2048 --out $(qw $c) > $L/tasq_w_$c.log 2>&1
  [ -f $(qw $c) ] || { echo "tasq_w $c FAILED"; return 1; }
  bld $g $c 32 cq8b_g32_$t --method cq
  bld $g $c 32 loopvq8b_g32_tasq_$t --method loopvq --loops-coupled 1 $(tw $c)
  bld $g $c 32 loopvq8b_g32_lv_$t --method loopvq --loops-coupled 4 --v-method loopvq
  ft $g $c loopvq8b_g32_lv_$t
  bld $g $c 32 loopvq8b_g32_tqlv_$t --method loopvq --loops-coupled 4 --v-method loopvq $(tw $c)
  ft $g $c loopvq8b_g32_tqlv_$t; }
( for ct in "wikitext wt" "c4 c4" "slimpajama sp"; do
    read -r c t <<< "$ct"
    [ -f $(cg $c) ] || { CUDA_VISIBLE_DEVICES=${GP[7]} $PY build_ouro_bundle.py $(opts $c) --out-dir /dev/null \
      --calib-only --grad-ckpt > $L/calib_$c.log 2>&1; echo "calib $c rc=$?"; }
  done ) &
CAL=$!
# fits of a corpus start once its buffer exists (buffers on GPU 7); one GPU per corpus (4 / 5 / 6),
# its fits in sequence
k=4
for ct in "wikitext wt" "c4 c4" "slimpajama sp"; do
  read -r c t <<< "$ct"
  ( while [ ! -f $(cg $c) ]; do kill -0 $CAL 2>/dev/null || break; sleep 20; done
    [ -f $(cg $c) ] && corpus ${GP[$k]} $c $t || echo "calib $c MISSING" ) &
  k=$((k + 1))
done
wait

# (3) calibration-set size on GPQA: NC = 8 / 32 sequences (16 is the main table). LoopVQ and
# TaSQ+LoopVQ only, 0.25 bit. Run after (1)-(2): the 32-sequence buffer is ~100 GB of RAM while written.
for NC in 8 32; do
  o=$( [ $NC = 8 ] && echo 0 || echo 4 )         # n8 on GPUs 0-3, n32 on 4-7 (fits overlap)
  [ -f $(cg gpqa_nodiamond) ] || { CUDA_VISIBLE_DEVICES=${GP[$o]} $PY build_ouro_bundle.py $(opts gpqa_nodiamond) \
    --out-dir /dev/null --calib-only --grad-ckpt > $L/calib_gpqa_n$NC.log 2>&1; echo "calib n$NC rc=$?"; }
  [ -f $(qw gpqa_nodiamond) ] || CUDA_VISIBLE_DEVICES=${GP[$((o + 1))]} $PY tasq_qweights.py --model $M --calib-data gpqa_nodiamond \
    --n-calib $NC --calib-seq-len 2048 --out $(qw gpqa_nodiamond) > $L/tasq_w_gpqa_n$NC.log 2>&1
  ( bld ${GP[$((o + 2))]} gpqa_nodiamond 32 loopvq8b_g32_lv_n$NC --method loopvq --loops-coupled 4 --v-method loopvq
    ft ${GP[$((o + 2))]} gpqa_nodiamond loopvq8b_g32_lv_n$NC ) &
  ( bld ${GP[$((o + 3))]} gpqa_nodiamond 32 loopvq8b_g32_tqlv_n$NC --method loopvq --loops-coupled 4 --v-method loopvq $(tw gpqa_nodiamond)
    ft ${GP[$((o + 3))]} gpqa_nodiamond loopvq8b_g32_tqlv_n$NC ) &
done
wait
NC=16

# (4) FT objective
ftobj

ARMS=""
for G in 16 32; do ARMS="$ARMS loopvq8b_g${G}_cq1 loopvq8b_g${G}_cq1e2e loopvq8b_g${G}_tasq1 loopvq8b_g${G}_tasq1e2e"; done
for t in wt c4 sp; do ARMS="$ARMS cq8b_g32_$t loopvq8b_g32_tasq_$t loopvq8b_g32_lv_${t}e2e loopvq8b_g32_tqlv_${t}e2e"; done
for n in 8 32; do ARMS="$ARMS loopvq8b_g32_lv_n${n}e2e loopvq8b_g32_tqlv_n${n}e2e"; done
ARMS="$ARMS $FTOBJ"
python3 $R/calibration/check_bundles.py $(for a in $ARMS; do echo ${MODEL}_$a; done) \
  && echo BUILD_DONE || echo BUILD_FAILED
