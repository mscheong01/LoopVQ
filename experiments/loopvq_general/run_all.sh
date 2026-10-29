#!/usr/bin/env bash
# Fill the loopvq_general table: every (bench, arm) cell, the table rebuilt after each one.
#
#   bash experiments/loopvq_general/run_all.sh                    # all benches x all arms, 1.4B
#   MODEL=ouro_2p6b bash experiments/loopvq_general/run_all.sh    # same for Ouro-2.6B
#   BENCHES="gsm8k_cot_3shot" ARMS="loopvq_0.5bit" bash .../run_all.sh
#
# Cells already in outputs/ are skipped (run_cell.sh), so this is safe to re-run after a crash.
set -u
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$G/../.." && pwd)"
MODEL=${MODEL:-ouro_1p4b}   # ouro_1p4b / ouro_2p6b / huginn_3p5b
BENCHES=${BENCHES:-"gsm8k_cot_3shot math500_mv evalplus_humaneval evalplus_mbpp"}
ARMS=${ARMS:-"bf16 cq_0.5bit tasq_0.5bit loopvq_0.5bit tasq_loopvq_0.5bit
               cq_0.25bit tasq_0.25bit loopvq_0.25bit tasq_loopvq_0.25bit"}
# method names (scripts/arms.sh) -> arm names, which bundles and outputs/ are keyed by
source "$R/scripts/arms.sh"
ARMS=$(for a in $ARMS; do arm_name "$a" "$MODEL"; done)

names=$(for a in $ARMS; do [[ $a == bf16 ]] || echo ${MODEL}_$a; done)
python3 "$R/calibration/check_bundles.py" $names --quiet \
  || { python3 "$R/calibration/check_bundles.py" $names; echo "BUNDLES MISSING"; exit 1; }

for b in $BENCHES; do
  for a in $ARMS; do
    echo "[general] $MODEL $b $a $(date -Is)"
    bash "$G/scripts/${MODEL}__$b.sh" "$a"
    ls "$G/outputs/${MODEL}__${b}__${a}"*.json > /dev/null 2>&1 \
      || echo "WARNING: loopvq_general $MODEL: $b / $a produced NO RESULT -- see work/$MODEL/logs" >&2
    python3 "$G/make_table.py" > /dev/null 2>&1
  done
done
python3 "$G/make_table.py" 2>/dev/null | tail -n 15
echo "GENERAL_DONE $MODEL $(date -Is)"
