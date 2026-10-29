#!/bin/bash
# Run the benchmark sweep for one model, one arm at a time.
#
#   ./scripts/run_sweep.sh <models>/Ouro-1.4B bf16 loopvq8b_g16_lvgqe2e   # all node GPUs
#   GPUS="4 5 6 7" ./scripts/run_sweep.sh <models>/Ouro-1.4B cq8b_g16_gq # a subset of GPUs
#
# The experiment drivers (experiments/*/scripts/run_cell.sh) call this through common.sh.
#
# One single-GPU replica per device, so the replica count -- and with it the request concurrency
# and the per-replica server cap -- follows whatever the node has (see config.sh "GPU topology").
#
# Each arm = one single-GPU server per GPU behind a round-robin proxy; the benchmarks then run
# sequentially against the proxy, so every request spreads over all replicas. Arms without an
# artifact are skipped, finished tasks are skipped, so re-running resumes.
# Results: work/<tag>/results/<tag>_<arm>_0_<task>.json
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
have_conda
conda activate "$ENV_SERVE"
export HF_ALLOW_CODE_EVAL=1

MODEL="${1:?usage: run_sweep.sh <model> <arms...>}"; shift
ARMS=("$@"); [[ $# -eq 0 ]] && die "usage: run_sweep.sh <model> <arms...>"
for i in "${!ARMS[@]}"; do ARMS[$i]=$(canonical_method "${ARMS[$i]}" "$MODEL"); done
TAG="$(model_tag "$MODEL")"
W="$WORK/$TAG"; LOG="$W/logs"; RESULTS="$W/results"; mkdir -p "$LOG" "$RESULTS"
# config.sh detects the node's GPUs and derives NUM_CONCURRENT/MAX_RUNNING from the count; this
# just splits its answer. Set GPUS (or CUDA_VISIBLE_DEVICES) to pin a subset.
read -r -a GPU_LIST <<< "$GPUS"
BASE_PORT="${BASE_PORT:-30081}"
PROXY_PORT="${PROXY_PORT:-30080}"
# Ouro's -Thinking variants gate their reasoning block on enable_thinking and default it
# OFF, so the short-task chain has to turn it on explicitly or they answer like a weak
# instruct model. The model card's sampling settings come with it (greedy collapses a
# reasoning model's chain), and the budget has to fit a chain plus the answer.
THINK=()
if [[ "$TAG" == *_think ]]; then
  THINK=(--think --gen_kwargs "temperature=${THINK_TEMP:-1.0},top_p=${THINK_TOP_P:-0.7},do_sample=True,max_gen_toks=${THINK_MAX_GEN:-2048}")
fi
INCLUDE=(); [[ -n "$CUSTOM_TASKS" ]] && INCLUDE=(--include_path "$CUSTOM_TASKS")
# LIMIT=<n> caps docs per task -- smoke tests only; a limited result file is indistinguishable
# from a full one by name, so clear it before the real run.
LIM=(); [[ -n "${LIMIT:-}" ]] && LIM=(--limit "$LIMIT")
# lm_eval applies this timeout to the complete task queue.
SHORT_TIMEOUT="${SHORT_TIMEOUT:-86400}"
[[ -x "$EVAL_PY" ]] || die "eval python not found: $EVAL_PY (see README 'Environments')"
PROXY_PID=""
SERVER_PIDS=()   # pid per GPU slot, so teardown never reaches another sweep's servers

arm_spec() {   # arm -> "engine|artifact|extra_env"
  local d="$ARTIFACTS/$TAG"
  case "$1" in
    # bf16 ignores the artifact contents; serve_method.sh only needs the positional argument, so
    # point it at the repo itself -- the baseline must be runnable before anything is calibrated.
    # The baseline uses the BF16 serving engine.
    bf16) echo "bf16|$REPO|" ;;
    # Codebook arms (8-bit codes, G = 2 ... 32 -> 4 ... 0.25 bits per element) on the packed vq
    # engine. MAX_TOKENS pins the pool to the same token count for every arm of a model (a memory
    # knob; the sizer assumes int2-equivalent bytes, which G=2 doubles).
    cq8b_g*|loopvq8b_g*)
      echo "vq|${d}_${1}|MAX_TOKENS=${LOOPVQ_MAX_TOKENS:-$( [[ $TAG == *2p6b* ]] && echo 16384 || echo 32768 )}" ;;
    *) die "unknown arm: $1" ;;
  esac
}

start_servers() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ups=() extra=()
  local memfrac="$MEM_FRACTION" prefill="$MAX_PREFILL_TOKENS"
  # Optionally cap CUDA-graph capture batches for models with limited free memory.
  local gbs=(); [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]] && gbs=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  local ctx=(); [[ "$REASONING" == "1" ]] && ctx=(--context-length "${REASON_CTX:-40960}")
  case "$arm" in
    loopvq8b_g*|cq8b_g*)
      # decode CUDA graphs on (the vq dispatch is sync-free), piecewise graphs off;
      # LOOPVQ_CUDA_GRAPH=0 turns the decode graphs off too
      extra=(--disable-piecewise-cuda-graph)
      [[ "${LOOPVQ_CUDA_GRAPH:-1}" == 1 ]] || extra=(--disable-cuda-graph --disable-piecewise-cuda-graph)
      memfrac=${LOOPVQ_MEM_FRACTION:-0.70} ;;
  esac
  # NO_CUDA_GRAPH=1: graphs off for any arm (throughput baselines)
  [[ "${NO_CUDA_GRAPH:-0}" == 1 ]] && extra=(--disable-cuda-graph --disable-piecewise-cuda-graph)
  # Allocate one rendezvous port per replica below the ephemeral-port range.
  local NCCL_BASE=$(( 20000 + ($$ % 500) * 16 ))
  local _try
  for _try in $(seq 1 40); do
    local _busy=0 _i _p
    for _i in "${!GPU_LIST[@]}"; do
      _p=$((NCCL_BASE + _i))
      ss -lnt "sport = :$_p" 2>/dev/null | grep -q LISTEN && { _busy=1; break; }
    done
    [[ "$_busy" == "0" ]] && break
    NCCL_BASE=$((NCCL_BASE + 16))
    [[ "$NCCL_BASE" -gt 28000 ]] && NCCL_BASE=20000
  done
  say "$arm: nccl ports $NCCL_BASE-$((NCCL_BASE + ${#GPU_LIST[@]} - 1))"
  # TP>1 groups $GPUS n at a time, one replica per group (config.sh's N_REPLICAS); TP=1 is one
  # replica per GPU as before. Only bf16 is meant to run at TP>1 -- the vq decode kernels have no
  # per-rank codebook sharding (see experiments/loopvq_niah/scripts/run_model.sh).
  (( TP == 1 )) || [[ "$engine" == bf16 && -z "$extra_env" ]] \
    || { echo "[exp] $arm: TP=$TP is supported for the plain bf16 arm only" >&2; return 1; }
  (( ${#GPU_LIST[@]} % TP == 0 )) || { echo "[exp] GPUS count is not a multiple of TP=$TP" >&2; return 1; }
  local nrep=$(( ${#GPU_LIST[@]} / TP )) devs
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i)); ups+=("http://localhost:$port")
    devs=$(IFS=,; echo "${GPU_LIST[*]:i*TP:TP}")
    ( cd "$REPO" && env $extra_env CUDA_VISIBLE_DEVICES="$devs" \
        PYTHONPATH="$REPO/python${PYTHONPATH_EXTRA:+:$PYTHONPATH_EXTRA}" \
        PREFILL_BACKEND=triton TP=$TP PREFIX_TOKENS=$PREFIX_TOKENS RECENT_TOKENS=$RECENT_TOKENS \
        nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
        --port "$port" --nccl-port $((NCCL_BASE + i)) --max-running-requests $MAX_RUNNING \
        --chunked-prefill-size $CHUNKED_PREFILL --max-prefill-tokens "$prefill" \
        --mem-fraction-static "$memfrac" "${gbs[@]}" "${ctx[@]}" "${extra[@]}" \
        > "$LOG/server_${arm}_gpu${devs//,/_}.log" 2>&1 & )
    sleep 1
    SERVER_PIDS[$i]=$(pgrep -f "sglang.launch_serve[r].*--port $port( |$)" | head -1)
  done
  for (( i = 0; i < nrep; i++ )); do
    local port=$((BASE_PORT + i))
    # Give up as soon as the server process is gone (crash at load / graph capture) instead of
    # waiting out the full 20 minutes. The pid can be missing right after launch; look again.
    local dead=0 pid
    for _ in $(seq 1 120); do
      curl -sf "http://localhost:$port/health" >/dev/null && break
      pid="${SERVER_PIDS[$i]:-}"
      [[ -z "$pid" ]] && { pid=$(pgrep -f "sglang.launch_serve[r].*--port $port( |$)" | head -1); SERVER_PIDS[$i]=$pid; }
      [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && { dead=1; break; }
      sleep 10
    done
    curl -sf "http://localhost:$port/health" >/dev/null || {
      devs=$(IFS=_; echo "${GPU_LIST[*]:i*TP:TP}"); tail -30 "$LOG/server_${arm}_gpu${devs}.log" >&2
      echo "[exp] $arm: server :$port $([[ $dead == 1 ]] && echo "exited during startup" || echo "did not start") -- skipping arm" >&2; return 1; }
  done
  ( cd "$REPO" && nohup python3 "$SCRIPTS/rr_proxy.py" "$PROXY_PORT" "${ups[@]}" \
      > "$LOG/proxy_$arm.log" 2>&1 & )
  sleep 3
  PROXY_PID=$(pgrep -f "rr_proxy.py $PROXY_PORT" | head -1)
  say "$arm: ${#GPU_LIST[@]} servers + proxy :$PROXY_PORT ready"
}

stop_servers() {
  [[ -n "$PROXY_PID" ]] && kill -9 "$PROXY_PID" 2>/dev/null
  PROXY_PID=""
  # Kill only servers started by this sweep.
  for i in "${!GPU_LIST[@]}"; do
    local port=$((BASE_PORT + i)) pid="${SERVER_PIDS[$i]:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
    elif [[ -z "$pid" ]]; then
      pkill -9 -f "sglang.launch_serve[r].*$port" 2>/dev/null
    fi
  done
  SERVER_PIDS=()
  # Wait until GPU memory from the previous arm is released.
  local waited=0 busy=0 used g
  while [ "$waited" -lt 120 ]; do
    busy=0
    for g in "${GPU_LIST[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null)
      if [ -n "$used" ] && [ "$used" -gt 2000 ]; then busy=1; fi
    done
    if [ "$busy" -eq 0 ]; then break; fi
    sleep 5; waited=$((waited + 5))
  done
  if [ "$busy" -eq 1 ]; then say "WARNING: GPU memory still held after ${waited}s -- next arm may OOM"; fi
  sleep 3
}
trap stop_servers EXIT

for arm in "${ARMS[@]}"; do
  IFS='|' read -r engine artifact extra_env <<< "$(arm_spec "$arm")"
  [[ -e "$artifact" ]] || { say "SKIP $arm: no artifact ($artifact) -- run build_bundles.sh"; continue; }
  # A build that dies leaves an empty directory, which the -e test above waves through.
  # Gate the codebook arms on content instead (calibration/check_bundles.py).
  case "$arm" in *8b_g*)
    python3 "$REPO/calibration/check_bundles.py" "$(basename "$artifact")" --quiet \
      || { say "SKIP $arm: $(python3 "$REPO/calibration/check_bundles.py" "$(basename "$artifact")")"; continue; } ;;
  esac
  say "===== $arm ====="
  start_servers "$arm" "$engine" "$artifact" "$extra_env" || { stop_servers; continue; }
  for task in "${TASKS[@]}"; do
    if [[ "$REASONING" == "1" ]]; then
      # Run sampled reasoning seeds concurrently against the same servers.
      pids=()
      for S in ${REASON_SEEDS:-0}; do
        out="$RESULTS/${TAG}_${arm}_${S}_${task}.json"
        [[ -f "$out" ]] && { say "$arm/$task seed $S already done"; continue; }
        # evalplus_* in thinking mode: own driver (lm_eval's *_instruct code tasks prime an open
        # code fence, which a thinking reply never fills -- see run_evalplus_reasoning_api.py)
        if [[ "$task" == evalplus_* ]]; then
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_evalplus_reasoning_api.py" --bench "${task#evalplus_}" \
            --seed "$S" --save_postfix "${TAG}_${arm}" --model_name "$MODEL" \
            --base_url "http://localhost:$PROXY_PORT/v1/chat/completions" \
            --output_dir "$RESULTS" --num_concurrent $NUM_CONCURRENT --timeout "$REASON_TIMEOUT" \
            --temperature "$REASON_TEMP" --top_p "$REASON_TOP_P" --max_gen_toks "$REASON_MAX_GEN" \
            "${LIM[@]}" > "$LOG/eval_${arm}_${task}_s${S}.log" 2>&1 ) &
        else
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_reasoning_api.py" --task "$task" --seed "$S" \
            --save_postfix "${TAG}_${arm}" --model_name "$MODEL" \
            --base_url "http://localhost:$PROXY_PORT/v1/chat/completions" \
            --output_dir "$RESULTS" --num_concurrent $NUM_CONCURRENT --timeout "$REASON_TIMEOUT" \
            --temperature "$REASON_TEMP" --top_p "$REASON_TOP_P" --max_gen_toks "$REASON_MAX_GEN" \
            "${INCLUDE[@]}" "${LIM[@]}" > "$LOG/eval_${arm}_${task}_s${S}.log" 2>&1 ) &
        fi
        pids+=($!)
      done
      [[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
      for S in ${REASON_SEEDS:-0}; do
        o="$RESULTS/${TAG}_${arm}_${S}_${task}.json"
        say "$arm/$task s$S: $(tr -d '[:space:]' < "$o" 2>/dev/null || echo "NO RESULT -- see $LOG/eval_${arm}_${task}_s${S}.log")"
      done
      continue
    fi
    out="$RESULTS/${TAG}_${arm}_0_${task}.json"
    [[ -f "$out" ]] && { say "$arm/$task already done"; continue; }
    # Short tasks are greedy and single-seed. Base models get plain few-shot completion without the
    # chat template, the protocol of their reported scores.
    CHATARGS=(--chat_via_completions --fewshot_as_multiturn)
    [[ "${BASE_MODEL_EVAL:-0}" == "1" ]] && CHATARGS=()
    # math500_mv / evalplus_{humaneval,mbpp} are the ResidualQuant-aligned protocol for base models
    # (arXiv 2610.10381 Table 7; bf16 Ouro-1.4B reproduces it: 75.8 / 72.56 / 73.02). They run
    # outside lm_eval but write the same <postfix>_0_<task>.json name, so the skip logic holds.
    API=(--seed 0 --save_postfix "${TAG}_${arm}" --model_name "$MODEL"
         --base_url "http://localhost:$PROXY_PORT/v1/completions" --output_dir "$RESULTS"
         --num_concurrent $NUM_CONCURRENT "${LIM[@]}")
    case "$task" in
      math500_mv)
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_math500_api.py" "${API[@]}" \
            > "$LOG/eval_${arm}_${task}.log" 2>&1 ) ;;
      tput)   # decode throughput, not a score: run with GPUS=<one gpu> so the proxy has one upstream
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/bench_tput_api.py" ${TPUT_POINTS:+--points "$TPUT_POINTS"} \
            "${API[@]}" > "$LOG/eval_${arm}_${task}.log" 2>&1 ) ;;
      evalplus_humaneval|evalplus_mbpp)
        ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_evalplus_api.py" --bench "${task#evalplus_}" \
            --prompt_style plain $([[ $task == evalplus_mbpp ]] && echo --max_tokens 2048) "${API[@]}" \
            > "$LOG/eval_${arm}_${task}.log" 2>&1 ) ;;
      *)
    ( cd "$REPO" && "$EVAL_PY" "$SCRIPTS/run_gsm8k_api.py" --task "$task" --seed 0 \
        --save_postfix "${TAG}_${arm}" "${CHATARGS[@]+"${CHATARGS[@]}"}" "${THINK[@]+"${THINK[@]}"}" \
        --model_name "$MODEL" --base_url "http://localhost:$PROXY_PORT/v1/completions" \
        --output_dir "$RESULTS" --num_concurrent $NUM_CONCURRENT --timeout "$SHORT_TIMEOUT" \
        "${INCLUDE[@]}" "${LIM[@]}" > "$LOG/eval_${arm}_${task}.log" 2>&1 ) ;;
    esac
    say "$arm/$task: $(tr -d '[:space:]' < "$out" 2>/dev/null || echo "NO RESULT -- see $LOG/eval_${arm}_${task}.log")"
  done
  stop_servers
done

say "sweep done -- results in $RESULTS"
