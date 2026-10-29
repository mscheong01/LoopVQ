#!/usr/bin/env bash
# Fill the loopvq_reasoning table: every (model, bench, arm) cell, 3 seeds each, the table rebuilt
# after each one.
#
#   bash experiments/loopvq_reasoning/run_all.sh                          # both models, all arms
#   MODELS=ouro_2p6b_think ARMS="bf16" bash .../run_all.sh                # a subset
#
# Bundles first: MODEL=<tag> bash experiments/loopvq_reasoning/build.sh (refit on the Thinking
# weights). Cells already in outputs/ are skipped (run_cell.sh), so this is safe to re-run.
set -u
G="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$G/../.." && pwd)"
MODELS=${MODELS:-"ouro_1p4b_think ouro_2p6b_think"}
BENCHES=${BENCHES:-"aime24 aime25 gpqa_diamond evalplus_humaneval evalplus_mbpp"}
ARMS=${ARMS:-"bf16 cq_0.5bit tasq_0.5bit loopvq_0.5bit tasq_loopvq_0.5bit
               cq_0.25bit tasq_0.25bit loopvq_0.25bit tasq_loopvq_0.25bit"}
source "$R/scripts/arms.sh"   # method names -> arm names
METHOD_NAMES=$ARMS

for MODEL in $MODELS; do
  ARMS=$(for a in $METHOD_NAMES; do arm_name "$a" "$MODEL"; done)
  names=$(for a in $ARMS; do [[ $a == bf16 ]] || echo ${MODEL}_$a; done)
  if [[ -n "$names" ]] && ! python3 "$R/calibration/check_bundles.py" $names --quiet; then
    python3 "$R/calibration/check_bundles.py" $names; echo "BUNDLES MISSING for $MODEL"; continue
  fi
  for b in $BENCHES; do
    for a in $ARMS; do
      echo "[reasoning] $MODEL $b $a $(date -Is)"
      bash "$G/scripts/${MODEL}__$b.sh" "$a"
      # fail fast: a missing result is almost always a server crash (OOM etc.) that would hit every
      # later arm too, so stop here and say why instead of burning the rest of the queue
      if ! ls "$G/outputs/${MODEL}__${b}__${a}"*.json > /dev/null 2>&1; then
        why=$(grep -hE "OutOfMemoryError|Error:|Killed" "${PE_WORK:-$R/experiments/work}/$MODEL/logs/server_${a}_gpu"*.log 2>/dev/null | head -1 | cut -c1-200)
        echo "loopvq_reasoning STOPPED: $MODEL $b / $a produced NO RESULT. ${why:-see work/$MODEL/logs}" >&2
        echo "REASONING_STOPPED $MODEL $b $a $(date -Is)"; exit 1
      fi
      python3 "$G/make_table.py" > /dev/null 2>&1
    done
  done
done
python3 "$G/make_table.py" 2>/dev/null | tail -n 20
echo "REASONING_DONE $(date -Is)"
