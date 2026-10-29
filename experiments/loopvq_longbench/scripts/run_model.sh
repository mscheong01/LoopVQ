#!/bin/bash
# loopvq_longbench, one model: LongBench v1 (16 English tasks, 3750 samples, official protocol) x every method.
# Serving loop copied from experiments/loopvq_niah/scripts/run_model.sh (same knobs, same long-context
# residual protocol PREFIX=64/RECENT=256); rows from ../build_rows.py (prompts middle-truncated to
# 64000 tokens, per-task generation length in each row), generations by scripts/run_cell.py, scoring by
# emit_lbv1.py (official metrics), one outputs/ record per task + avg. A method whose lbv1@avg cell exists is skipped.
#
#   ./ouro_1p4b.sh tasq_loopvq_0.5bit
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../common.sh"
OUTDIR="$HERE/../outputs"; mkdir -p "$OUTDIR"

MODEL="${1:?usage: run_model.sh <hf-model> [methods...]}"; shift
METHODS=("$@"); [[ $# -eq 0 ]] && METHODS=(bf16 cq_0.5bit tasq_0.5bit loopvq_0.5bit tasq_loopvq_0.5bit
  cq_0.25bit tasq_0.25bit loopvq_0.25bit tasq_loopvq_0.25bit)

# LongBench is a long-context table: it uses the project's long-context residual protocol
# (PREFIX=64/RECENT=256, the setting reports under), not the short-task one.
export PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
export RECENT_TOKENS="${RECENT_TOKENS:-256}"
source "$REPO/config.sh"
for i in "${!METHODS[@]}"; do METHODS[$i]=$(canonical_method "${METHODS[$i]}" "$MODEL"); done
# The NIAH_* knob names are shared with the NIAH driver. NUM_CONCURRENT records the client
# workers in flight for the provenance.
export NUM_CONCURRENT="${NIAH_WORKERS:-16}"
export MAX_RUNNING
have_conda; conda activate "$ENV_SERVE"

TAG="$(model_tag "$MODEL")"
# Grid: 4k .. 64k. A 128k prompt must be prefilled in one forward (the quantized arms refuse
# chunked prefill), and its activations do not fit beside the pool on a 24 GB card.
CTX_LIST=(64021)   # longest row (rows/lbv1_<tag>.jsonl n_tokens: 64020) -- +1152 covers the 512-token summaries
read -r -a GPU_LIST  <<< "$GPUS"   # detected in config.sh unless pinned
BASE_PORT="${BASE_PORT:-30081}"; PROXY_PORT="${PROXY_PORT:-30080}"
W="$WORK/$TAG/lbv1"; LOG="$W/logs"; mkdir -p "$LOG"
NIAH_MAX_TOKENS="${NIAH_MAX_TOKENS:-128}"   # raise to 8192 for a thinking model
NIAH_WORKERS="${NIAH_WORKERS:-16}"   # 2 in flight per TP=1 replica on 8 GPUs at NIAH_MAX_RUNNING=1
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
# --nccl-port is pinned per replica (port + 2000): left random, 8 servers booting at once drew a
# torch.distributed port already in use and one replica never came up.
# NIAH_SERVER_EXTRA: extra sglang server args, e.g. "--watchdog-timeout 3600" -- a 2.6B 0.25-bit 64k prefill
# with its loop-coupled hold parked in host RAM ran past sglang's default 300 s scheduler watchdog.
PROXY_PID=""; SERVER_PIDS=()

# LBV1_PART=short|long runs one row subset (rows/lbv1_<tag>.<part>.jsonl, split at 32k tokens) -- the 2.6B
# bf16 row needs TP=8 only for its >32k rows. A finished part waits in $W; once both parts are there they
# are concatenated (row order is irrelevant to the scorer) and scored as the full set.
PART="${LBV1_PART:-}"; SUF="${PART:+.$PART}"
ROWS="$HERE/../rows/lbv1_${TAG}${SUF}.jsonl"
[[ -s "$ROWS" ]] || pe_die "no rows: $ROWS -- run ../build_rows.py --model $MODEL --out $ROWS"

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
        --port "$port" --nccl-port $((port + 2000)) --max-running-requests "${NIAH_MAX_RUNNING:-8}" \
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
  case "$m" in
    cq8b_g*|loopvq8b_g*) export PE_SCALE_DTYPE=float16 ;;
    *)        export PE_SCALE_DTYPE=float32 ;;
  esac
  NEXP=$(wc -l < "$ROWS")
  if [[ -n "$PART" && "${FORCE:-0}" != "1" && $(wc -l < "$W/lbv1_${m}${SUF}.jsonl" 2>/dev/null || echo 0) -eq $NEXP ]]; then
    pe_say "$m: part $PART already complete in $W"; continue; fi
  if [[ "${FORCE:-0}" != "1" && -e "$OUTDIR/${TAG}__lbv1@avg__${m}__s0.json" ]]; then
    pe_say "$m: ALREADY RUN (lbv1@avg in outputs/). FORCE=1 to re-run."; continue; fi
  IFS='|' read -r engine artifact extra_env <<< "$(source_arm_spec "$m")"
  [[ -e "$artifact" ]] || { pe_say "SKIP $m: no artifact ($artifact) -- see experiments/loopvq_general/build.sh"; continue; }
  pe_say "===== $m: LongBench v1 (3750) ====="
  start_servers "$m" "$engine" "$artifact" "$extra_env" || { stop_servers; continue; }
  # LBV1_RESUME=1 keeps the rows already in $raw (e.g. from a run whose node died) and runs only the rest.
  raw="$W/lbv1_${m}${SUF}.jsonl"; resume=()
  if [[ "${LBV1_RESUME:-0}" == "1" ]]; then resume=(--resume); else rm -f "$raw"; fi
  ( cd "$REPO" && PYTHONPATH="$REPO" python3 scripts/run_cell.py "${resume[@]}" \
      --task longbench_v1 --model "$MODEL" --method "$m" --seed 0 \
      --input "$ROWS" --output "$raw" \
      --workers "$NIAH_WORKERS" \
      --base-url "http://127.0.0.1:$PROXY_PORT" ) > "$LOG/${m}_lbv1${SUF}.log" 2>&1   # max_new_tokens per row
  if [[ $(wc -l < "$raw" 2>/dev/null || echo 0) -eq $NEXP ]]; then
    full="$raw"
    if [[ -n "$PART" ]]; then
      full="$W/lbv1_${m}.jsonl"; cat "$W/lbv1_${m}.short.jsonl" "$W/lbv1_${m}.long.jsonl" > "$full" 2>/dev/null
    fi
    if [[ $(wc -l < "$full") -eq 3750 ]]; then
      python3 "$HERE/emit_lbv1.py" "$full" "$OUTDIR" "$TAG" "$m" "$ARTIFACTS/${TAG}_$(pe_artifact_suffix "$m")"
    else
      pe_say "$m: part $PART complete ($NEXP rows); scoring once the other part is in"
    fi
  else
    pe_say "NO RESULT $m/lbv1${SUF} ($(wc -l < "$raw" 2>/dev/null || echo 0)/$NEXP rows) -- see $LOG/${m}_lbv1.log"
  fi
  stop_servers
done
pe_say "done -- build the table with ../make_table.py"
