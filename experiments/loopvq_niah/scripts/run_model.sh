#!/bin/bash
# loopvq_niah, one model: every context length x every seed x every method.
#
#   ./ouro_1p4b.sh                          # wrapper with the Ouro serving env
#   CTX="8192" SEEDS="0" ./ouro_1p4b.sh loopvq_0.5bit
#
# NIAH does not go through lm_eval -- rows come from tasks/niah/build_niah_rows.py and scoring is
# scripts/run_cell.py's contains_answer over the row file. So this script owns its serving loop
# instead of calling scripts/run_sweep.sh, but every serving knob is read from config.sh so the
# two cannot drift.
#
# One server session per method covers all lengths and seeds -- restarting per cell would pay the
# model-load cost 12 times over. Cells already in outputs/ are reported and skipped; if a method
# has nothing left to do its server is never started.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
OUTDIR="$HERE/../outputs"; mkdir -p "$OUTDIR"

MODEL="${1:?usage: run_model.sh <hf-model> [methods...]}"; shift
METHODS=("$@"); [[ $# -eq 0 ]] && METHODS=(bf16 cq_0.5bit tasq_0.5bit loopvq_0.5bit tasq_loopvq_0.5bit
  cq_0.25bit tasq_0.25bit loopvq_0.25bit tasq_loopvq_0.25bit)

# NIAH is the long-context table: it uses the project's long-context residual protocol
# (PREFIX=64/RECENT=256, the setting reports under), not the short-task one.
export PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
export RECENT_TOKENS="${RECENT_TOKENS:-256}"
source "$REPO/config.sh"
for i in "${!METHODS[@]}"; do METHODS[$i]=$(canonical_method "${METHODS[$i]}" "$MODEL"); done
# PREFIX/RECENT are exported above; NUM_CONCURRENT is not one of this path's knobs (NIAH drives
# its own client with NIAH_WORKERS, not lm_eval's num_concurrent), so record the number that is
# actually in flight rather than leaving the provenance field null. MAX_RUNNING is the matching
# server-side cap and comes from config.sh as a plain variable.
export NUM_CONCURRENT="${NIAH_WORKERS:-32}"
export MAX_RUNNING
have_conda; conda activate "$ENV_SERVE"

TAG="$(model_tag "$MODEL")"
# Grid: 4k .. 64k. A 128k prompt must be prefilled in one forward (the quantized arms refuse
# chunked prefill), and its activations do not fit beside the pool on a 24 GB card.
read -r -a CTX_LIST  <<< "${CTX:-4096 8192 16384 32768 65536}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2}"
read -r -a GPU_LIST  <<< "$GPUS"   # detected in config.sh unless pinned
BASE_PORT="${BASE_PORT:-30081}"; PROXY_PORT="${PROXY_PORT:-30080}"
W="$WORK/$TAG/niah"; LOG="$W/logs"; mkdir -p "$LOG"
NIAH_MAX_TOKENS="${NIAH_MAX_TOKENS:-128}"   # raise to 8192 for a thinking model
NIAH_WORKERS="${NIAH_WORKERS:-32}"
# Tensor parallelism. TP=1 puts one replica on each GPU in $GPUS; TP=n groups them n at a time,
# so $GPUS must be a multiple of n and the replica count drops accordingly. The bf16 reference
# row needs TP>=2 at the long lengths on 24 GB cards. The quantized arms fit at TP=1 throughout.
TP="${TP:-1}"
# TP>1 is a BF16-only escape hatch. The quantized arms have never been exercised at TP>1 in this
# fork: there is no per-rank sharding of the codebook or the TaSQ permutation. Refuse rather than
# emit a number from it; the BF16 row is the only one that needs the memory anyway.
if (( TP > 1 )); then
  for _m in "${METHODS[@]}"; do
    # Only the baseline supports tensor parallelism here.
    [[ "$_m" == "bf16" ]] || pe_die "TP=$TP requested with method '$_m'. Only the bf16 baseline is supported at TP>1 (see README). Run: TP=$TP $0 <model> bf16"
  done
fi
# NIAH_SERVER_EXTRA: extra sglang server args, e.g. "--watchdog-timeout 3600" -- a 2.6B 0.25-bit 64k prefill
# with its loop-coupled hold parked in host RAM ran past sglang's default 300 s scheduler watchdog.
PROXY_PID=""; SERVER_PIDS=()

# ---- row files (deterministic given model + length + seed) ----
rows_for() { echo "$W/niah_${1}_${TAG}_s${2}.jsonl"; }
ensure_rows() {
  local ctx=$1 seed=$2 f; f=$(rows_for "$ctx" "$seed")
  [[ -s "$f" ]] && return 0
  pe_say "building rows ctx=$ctx seed=$seed"
  # NIAH_BUDGET_MODEL: tokenizer that meters the haystack (the model's own, set by the wrappers).
  # NIAH_LEN_CAP: build length ceiling, so the longest column still fits the model's native window
  # with the answer budget on top.
  local blen=$ctx; (( blen > ${NIAH_LEN_CAP:-1000000000} )) && blen=$NIAH_LEN_CAP
  ( cd "$REPO" && python3 tasks/niah/build_niah_rows.py --model "$MODEL" \
      --budget-model "${NIAH_BUDGET_MODEL:-$MODEL}" \
      --length "$blen" --seed "$seed" --out "$f" ) > "$LOG/rows_${ctx}_s${seed}.log" 2>&1 \
    || pe_die "row build failed (ctx=$ctx seed=$seed) -- see $LOG/rows_${ctx}_s${seed}.log"
}

cell_file() { echo "$OUTDIR/${TAG}__niah@${1}__${2}__s${3}.json"; }

# ---- serving (mirrors scripts/run_sweep.sh; knobs from config.sh) ----
source_arm_spec() {   # arm -> "engine|artifact|extra_env", by reusing run_sweep's own table
  bash -c 'source "'"$REPO"'/config.sh"; '"$(sed -n '/^arm_spec()/,/^}/p' "$REPO/scripts/run_sweep.sh")"'; MODEL='"$MODEL"'; TAG='"$TAG"'; arm_spec '"$1"
}
start_servers() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ups=() extra=()
  local memfrac="$MEM_FRACTION" prefill="$MAX_PREFILL_TOKENS"
  local maxctx_probe=0; for c in "${CTX_LIST[@]}"; do (( c > maxctx_probe )) && maxctx_probe=$c; done
  case "$arm" in
  # Cap graph capture for the quantized arms. Their decode path has many more kernels per layer
  # than bf16's, so capture uses more memory. Batches above the cap run eager; model outputs are
  # unchanged.
    cq8b_g*|loopvq8b_g*) extra=(--disable-piecewise-cuda-graph --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-8}") ;;
  esac
  # NIAH_MAX_RUNNING caps the requests the server runs at once; cells cut short by a server crash
  # are dropped from the table (table_lib._drop_short_cells). Above 64k the bf16 arm prefills in
  # chunks; the quantized arms cannot, so they get more room beside the pool (NIAH_QUANT_MEMFRAC).
  local chunked="$CHUNKED_PREFILL"
  if (( maxctx_probe > 65536 )) && [[ "$arm" == "bf16" ]]; then
    chunked="${NIAH_BF16_CHUNKED:-${NIAH_FP16_CHUNKED:-8192}}"
  elif [[ -n "${NIAH_QUANT_MEMFRAC:-}" ]]; then
    # One server session serves every length, so the fraction has to satisfy the longest one: the
    # pool must hold one full-length sequence and the rest must absorb the unchunked prefill.
    memfrac="$NIAH_QUANT_MEMFRAC"
  fi
  echo "[pe] $arm: chunked-prefill=$chunked mem-fraction=$memfrac max-ctx=$maxctx_probe"
  local maxctx=$maxctx_probe
  local nrep=$(( ${#GPU_LIST[@]} / TP ))
  (( nrep >= 1 )) || pe_die "TP=$TP needs at least $TP GPUs, got ${#GPU_LIST[@]}"
  (( ${#GPU_LIST[@]} % TP == 0 )) || pe_die "GPUS count ${#GPU_LIST[@]} is not a multiple of TP=$TP"
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i)); ups+=("http://localhost:$port")
    local devs; devs=$(IFS=,; echo "${GPU_LIST[*]:i*TP:TP}")
    ( cd "$REPO" && env $extra_env CUDA_VISIBLE_DEVICES="$devs" PYTHONPATH="$REPO/python" \
        PREFILL_BACKEND=triton TP=$TP PREFIX_TOKENS=$PREFIX_TOKENS RECENT_TOKENS=$RECENT_TOKENS \
        nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
        --port "$port" --max-running-requests "${NIAH_MAX_RUNNING:-8}" \
        --chunked-prefill-size $chunked --max-prefill-tokens "$prefill" \
        --mem-fraction-static "$memfrac" \
        --context-length $((maxctx + NIAH_MAX_TOKENS + 1024)) "${extra[@]}" ${NIAH_SERVER_EXTRA:-} \
        > "$LOG/server_${arm}_rep${i}.log" 2>&1 & )
    sleep 1
    SERVER_PIDS[$i]=$(pgrep -f "sglang.launch_serve[r].*--port $port( |$)" | head -1)
  done
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i))
    for _ in $(seq 1 120); do curl -sf "http://localhost:$port/health" >/dev/null && break; sleep 10; done
    curl -sf "http://localhost:$port/health" >/dev/null || { pe_say "server $port never came up"; return 1; }
  done
  ( cd "$REPO" && RR_PROXY_TIMEOUT=7200 nohup python3 scripts/rr_proxy.py "$PROXY_PORT" "${ups[@]}" \
      > "$LOG/proxy_${arm}.log" 2>&1 & )
  sleep 3; PROXY_PID=$(pgrep -f "rr_proxy.py $PROXY_PORT" | head -1)
}
stop_servers() {
  [[ -n "$PROXY_PID" ]] && kill -9 "$PROXY_PID" 2>/dev/null; PROXY_PID=""
  for pid in "${SERVER_PIDS[@]:-}"; do [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null; done
  SERVER_PIDS=(); sleep 10
}
trap stop_servers EXIT

# ---- main ----
for m in "${METHODS[@]}"; do
  # The served scale dtype, for the provenance (serve_method.sh's float16 VQ default).
  case "$m" in
    cq8b_g*|loopvq8b_g*) export PE_SCALE_DTYPE=float16 ;;
    *)        export PE_SCALE_DTYPE=float32 ;;
  esac
  todo=(); already=()
  for ctx in "${CTX_LIST[@]}"; do for s in "${SEED_LIST[@]}"; do
    if [[ "${FORCE:-0}" != "1" ]] && SEED="$s" REPLICATE="" pe_cell_exists "$OUTDIR" "$TAG" "niah@$ctx" "$m"; then already+=("${ctx}/s${s}")
    else todo+=("${ctx}:${s}"); fi
  done; done
  [[ ${#already[@]} -gt 0 ]] && pe_say "$m: ALREADY RUN, skipping ${#already[@]} cell(s): ${already[*]}"
  if [[ ${#todo[@]} -eq 0 ]]; then pe_say "$m: nothing to do (all cells in outputs/). FORCE=1 to re-run."; continue; fi

  for spec in "${todo[@]}"; do ensure_rows "${spec%%:*}" "${spec##*:}"; done

  IFS='|' read -r engine artifact extra_env <<< "$(source_arm_spec "$m")"
  [[ -e "$artifact" ]] || { pe_say "SKIP $m: no artifact ($artifact) -- see experiments/loopvq_general/build.sh"; continue; }
  pe_say "===== $m: ${#todo[@]} cell(s) ====="
  start_servers "$m" "$engine" "$artifact" "$extra_env" || { stop_servers; continue; }

  for spec in "${todo[@]}"; do
    ctx="${spec%%:*}"; s="${spec##*:}"
    raw="$W/niah${ctx}_${m}_s${s}.jsonl"; rm -f "$raw"
    ( cd "$REPO" && PYTHONPATH="$REPO" python3 scripts/run_cell.py \
        --task ruler_niah --model "$MODEL" --method "$m" --seed "$s" \
        --input "$(rows_for "$ctx" "$s")" --output "$raw" \
        --workers "$NIAH_WORKERS" --max-tokens "$NIAH_MAX_TOKENS" \
        --base-url "http://127.0.0.1:$PROXY_PORT" ) > "$LOG/${m}_${ctx}_s${s}.log" 2>&1
    if [[ -s "$raw" ]]; then
      SEED="$s" python3 "$HERE/emit_niah.py" "$raw" "$(cell_file "$ctx" "$m" "$s")" \
        "$TAG" "niah@${ctx}" "$m" "$s" "$ARTIFACTS/${TAG}_$(pe_artifact_suffix "$m")"
    else
      pe_say "NO RESULT $m/${ctx}/s${s} -- see $LOG/${m}_${ctx}_s${s}.log"
    fi
  done
  stop_servers
done
pe_say "done -- build the table with ../make_table.py"
