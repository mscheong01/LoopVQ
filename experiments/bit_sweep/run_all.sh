#!/usr/bin/env bash
# Fill the bit_sweep table: every (bench, arm) cell at 1 / 2 / 4 bit; 0.25 / 0.5 bit and bf16 are
# taken from loopvq_general/outputs. Cells already in outputs/ are skipped (run_cell.sh).
#   MODEL=ouro_2p6b GPUS="0 1 2 3" bash experiments/bit_sweep/run_all.sh
set -u
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$G/../.." && pwd)"
MODEL=${MODEL:-ouro_2p6b}
BENCHES=${BENCHES:-"gsm8k_cot_3shot math500_mv evalplus_humaneval evalplus_mbpp"}
ARMS=${ARMS:-"cq_1bit tasq_1bit loopvq_1bit tasq_loopvq_1bit cq_2bit tasq_2bit loopvq_2bit
               cq_4bit tasq_4bit loopvq_4bit"}
source "$R/scripts/arms.sh"   # method names -> arm names
ARMS=$(for a in $ARMS; do arm_name "$a" "$MODEL"; done)
python3 "$R/calibration/check_bundles.py" $(for a in $ARMS; do echo ${MODEL}_$a; done) --quiet \
  || { echo "BUNDLES MISSING"; exit 1; }
for b in $BENCHES; do
  for a in $ARMS; do
    echo "[bit_sweep] $MODEL $b $a $(date -Is)"
    bash "$G/scripts/${MODEL}__$b.sh" "$a"
    ls "$G/outputs/${MODEL}__${b}__${a}"*.json > /dev/null 2>&1 \
      || echo "WARNING: bit_sweep $MODEL: $b / $a produced NO RESULT -- see work/$MODEL/logs" >&2
    python3 "$G/make_table.py" > /dev/null 2>&1
  done
done
python3 "$G/make_table.py" 2>/dev/null
echo "BITSWEEP_DONE $MODEL $(date -Is)"
