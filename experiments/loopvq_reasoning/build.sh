#!/usr/bin/env bash
# Build the loopvq_reasoning bundles for Ouro-{1.4B,2.6B}-Thinking: the 8 quantized arms of
# loopvq_general, refit on the Thinking weights (SFT moves them, so the base bundles do not apply).
# Same recipe and arms as loopvq_general/build.sh, which does the work.
#   MODEL=ouro_1p4b_think GPUS="0 1 2 3" bash experiments/loopvq_reasoning/build.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL=${MODEL:-ouro_1p4b_think} LOGDIR=${LOGDIR:-$HERE/logs} exec bash "$HERE/../loopvq_general/build.sh"
