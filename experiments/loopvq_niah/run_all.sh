#!/usr/bin/env bash
# Fill the loopvq_niah table (Ouro-1.4B): quantized arms at TP=1 on all GPUs, then bf16 at TP=4
# (its 96-slot bf16 KV does not fit one 24 GB card past ~16k). Cells already in outputs/ are
# skipped, so this is safe to re-run.
#
#   bash experiments/loopvq_niah/run_all.sh
#   ARMS="loopvq_0.5bit" bash experiments/loopvq_niah/run_all.sh
set -u
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$G/../.." && pwd)"
ARMS=${ARMS:-"loopvq_0.5bit tasq_loopvq_0.5bit loopvq_0.25bit tasq_loopvq_0.25bit
               cq_0.5bit tasq_0.5bit cq_0.25bit tasq_0.25bit"}
source "$R/scripts/arms.sh"   # method names -> arm names
ARMS=$(for a in $ARMS; do arm_name "$a" ouro_1p4b; done)
names=$(for a in $ARMS; do echo ouro_1p4b_$a; done)
python3 "$R/calibration/check_bundles.py" $names --quiet || { python3 "$R/calibration/check_bundles.py" $names; echo "BUNDLES MISSING"; exit 1; }
for a in $ARMS; do
  echo "[niah] $a $(date -Is)"
  TP=1 bash "$G/scripts/ouro_1p4b.sh" "$a"
  python3 "$G/make_table.py" > /dev/null 2>&1
done
if [[ "${SKIP_BF16:-0}" != 1 ]]; then
  echo "[niah] bf16 (TP=4) $(date -Is)"
  TP=4 bash "$G/scripts/ouro_1p4b.sh" bf16
fi
python3 "$G/make_table.py" 2>/dev/null | tail -n 12
echo "NIAH_DONE $(date -Is)"
