#!/usr/bin/env bash
# Build the loopvq_ablation bundles (build.sh), then fill every (bench, arm) cell; arms shared with
# loopvq_general are read from its outputs/. Cells already in outputs/ are skipped (run_cell.sh).
#   bash experiments/loopvq_ablation/run_all.sh
set -u
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$G/../.." && pwd)"
MODEL=ouro_1p4b
BENCHES=${BENCHES:-"gsm8k_cot_3shot evalplus_humaneval"}
COMP=""; for g in 16 32; do COMP="$COMP loopvq8b_g${g}_lvgq loopvq8b_g${g}_cq1e2e loopvq8b_g${g}_tqlv loopvq8b_g${g}_tasq1e2e loopvq8b_g${g}_cq1 loopvq8b_g${g}_tasq1"; done
CAL=""; for t in wt c4 sp; do CAL="$CAL cq8b_g32_$t loopvq8b_g32_tasq_$t loopvq8b_g32_lv_${t}e2e loopvq8b_g32_tqlv_${t}e2e"; done
SIZE=""; for n in 8 32; do SIZE="$SIZE loopvq8b_g32_lv_n${n}e2e loopvq8b_g32_tqlv_n${n}e2e"; done
FTOBJ=""; for g in 16 32; do FTOBJ="$FTOBJ loopvq8b_g${g}_lvgqps loopvq8b_g${g}_tqlvps"; done
ARMS=${ARMS:-"$COMP $CAL $SIZE $FTOBJ"}

echo "[ablation] build $(date -Is)"
mkdir -p "$G/logs"; ONLY=${BUILD_ONLY:-} bash "$G/build.sh" 2>&1 | tee -a "$G/logs/build.out" | tail -n 3
for b in $BENCHES; do
  for a in $ARMS; do
    python3 "$R/calibration/check_bundles.py" ${MODEL}_$a --quiet \
      || { echo "WARNING: loopvq_ablation: bundle $a missing or broken -- see loopvq_ablation/logs" >&2; continue; }
    echo "[ablation] $MODEL $b $a $(date -Is)"
    bash "$G/scripts/${MODEL}__$b.sh" "$a"
    ls "$G/outputs/${MODEL}__${b}__${a}"*.json > /dev/null 2>&1 \
      || echo "WARNING: loopvq_ablation $MODEL: $b / $a produced NO RESULT -- see work/$MODEL/logs" >&2
    python3 "$G/make_table.py" > /dev/null 2>&1
  done
done
python3 "$G/make_table.py" 2>/dev/null
echo "ABLATION_DONE $(date -Is)"
