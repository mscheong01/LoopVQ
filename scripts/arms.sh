# Method names -> the arm names bundles and results are stored under. Sourced, not run.
#
#   bf16                    BF16 KV cache
#   cq_<b>bit               CQ
#   tasq_<b>bit             TaSQ
#   loopvq_<b>bit           LoopVQ
#   tasq_loopvq_<b>bit      TaSQ+LoopVQ
#
# <b> is the bitwidth per element: 4, 2, 1, 0.5, or 0.25 (8-bit codes over G = 2 ... 32 elements).
# On Huginn the LoopVQ codes cover 4 of the 32 recurrences, so its arms carry a t4 tag.
# Any other name (fp16, an ablation arm such as loopvq8b_g32_lv_c4e2e) passes through, fp16 as bf16.
#
#   arm_name loopvq_0.5bit ouro_1p4b     -> loopvq8b_g16_lvgqe2e
#   arm_name loopvq_0.5bit huginn_3p5b   -> loopvq8b_g16_lvgqt4e2e

arm_name() {
  local name=$1 model=${2:-} g t4=""
  [[ $name == fp16 ]] && { echo bf16; return; }
  if [[ $name =~ ^(cq|tasq|loopvq|tasq_loopvq)_([0-9.]+)bit$ ]]; then
    case "${BASH_REMATCH[2]}" in
      4) g=2 ;; 2) g=4 ;; 1) g=8 ;; 0.5) g=16 ;; 0.25) g=32 ;;
      *) echo "$name"; return ;;
    esac
    [[ $model == *huginn* ]] && t4=t4
    case "${BASH_REMATCH[1]}" in
      cq)          echo "cq8b_g${g}_gq" ;;
      tasq)        echo "loopvq8b_g${g}_tasq" ;;
      loopvq)      echo "loopvq8b_g${g}_lvgq${t4}e2e" ;;
      tasq_loopvq) echo "loopvq8b_g${g}_tqlv${t4}e2e" ;;
    esac
    return
  fi
  echo "$name"
}
