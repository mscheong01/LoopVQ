#!/bin/bash
# Efficiency panels, one at a time, for the arms in $ARMS. Normally called through run.sh, which
# sets the Ouro / LoopVQ defaults.
#
#   MODEL=<models>/Ouro-1.4B-Thinking ARMS="bf16 loopvq_0.5bit" ./run_efficiency.sh decode
#   ./run_efficiency.sh thru       # end-to-end throughput vs batch size
#   ./run_efficiency.sh prefill    # prefill latency
#   OUTDIR=... EXTRA_ENV="SGLANG_X=1" ./run_efficiency.sh thru   # a variant, kept apart
#   THRU_CAP_FROM=<arm> ./run_efficiency.sh thru   # end the other arms' curves at that arm's ceiling
#
# Measurements use the same server path as the accuracy experiments. Each point runs alone on one
# GPU with CUDA graphs enabled; shared serving settings are held fixed across methods.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EFF="$(dirname "$HERE")"
source "$HERE/../../common.sh"
# OUTDIR: where cell JSONs land. Override to measure a variant without touching the recorded
# baseline cells.
OUTDIR="${OUTDIR:-$EFF/outputs}"; mkdir -p "$OUTDIR"

PANEL="${1:?usage: run_efficiency.sh <decode|thru|prefill>}"
MODEL="${MODEL:?set MODEL to a checkpoint (run.sh does)}"

# Figure 4 uses the reference efficiency policy: no attention sink and a 64-token raw tail.
# Export before config.sh so the reasoning model does not substitute the benchmark-only 64/256
# policy. This setting is part of the capacity measurement and is recorded in every result cell.
export PREFIX_TOKENS="${PREFIX_TOKENS:-0}"
export RECENT_TOKENS="${RECENT_TOKENS:-64}"
source "$REPO/config.sh"
have_conda; conda activate "$ENV_SERVE"

TAG="$(model_tag "$MODEL")"
# Native context window, read from the checkpoint. Serving above it engages RoPE scaling, so the
# decode panel's margin is clamped to it.
NATIVE_CTX="$(python3 - "$MODEL" <<'PYX'
import json, glob, os, sys
m = sys.argv[1]
# A local checkout, else the hub cache -- whose location is HF_HUB_CACHE, else $HF_HOME/hub,
# else the default. Hardcoding ~/.cache/huggingface made this step fail on any machine that
# sets either variable.
cand = [os.path.join(m, "config.json")]
hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
cand += sorted(glob.glob(os.path.join(hub, "models--" + m.replace("/", "--"),
                                      "snapshots", "*", "config.json")))
for p in cand:
    if os.path.exists(p):
        print(json.load(open(p)).get("max_position_embeddings", 0) or 0); break
else:
    print(0)
PYX
)"
[[ "${NATIVE_CTX:-0}" -gt 0 ]] || pe_die "could not read max_position_embeddings for $MODEL"
pe_say "native context window: $NATIVE_CTX"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-30091}"
PIECEWISE="${PIECEWISE:-0}"
# Match the effective FlashAttention prefill path across methods.
PREFILL_BACKEND="${PREFILL_BACKEND:-fa3}"
CHUNKED="${CHUNKED:--1}"
# The decode panel times decode steps only, so if a long unchunked prefill does not fit it may set
#   DEC_CHUNKED=8192 SGLANG_ALLOW_CHUNKED_QUANT_PREFILL=1
# (the chunked-prefill refusal guards accuracy). The prefill panel must not: chunking changes it.
DEC_CHUNKED="${DEC_CHUNKED:-$CHUNKED}"
MEMFRAC="${MEMFRAC:-0.85}"
# Decode sweeps choose the smallest pool reservation that fits the longest context, leaving
# sufficient workspace for unchunked prefill.
DEC_MEMFRAC_LADDER="${DEC_MEMFRAC_LADDER:-0.60 0.65 0.70 0.75 0.80 0.85}"
# All arms use the same memory fraction by default. The override remains available for controlled
# capacity studies with a different residual policy.
THRU_QUANT_MEMFRAC="${THRU_QUANT_MEMFRAC:-$MEMFRAC}"
W="$WORK/$TAG/efficiency"; LOG="$W/logs"; mkdir -p "$LOG"
read -r -a ARM_LIST <<< "${ARMS:-bf16}"
for i in "${!ARM_LIST[@]}"; do ARM_LIST[$i]=$(canonical_method "${ARM_LIST[$i]}" "$MODEL"); done

# ---- panel grids ----
# decode: x = TOTAL context. input_len = ctx - output_len so the served context never exceeds the
# model's 131072 native window (no rope scaling anywhere in this figure).
DEC_OUT=64
read -r -a DEC_CTXS <<< "${CTXS:-1024 2048 4096 8192 16384 32768 65536 131072}"
# thru: 2048 in / 32768 out. bf16's grid is TRUNCATED AT ITS MEASURED CAPACITY, not at a guess --
# see cap_batches below, which reads the engine's own max_total_num_tokens.
THRU_IN="${THRU_IN:-2048}"; THRU_OUT="${THRU_OUT:-32768}"
# Non-default shapes include the shape in their cell key. The published 2048/32768 setting keeps
# the bare `bsN` key consumed by the default figure loader.
THRU_KEY=""; (( THRU_IN == 2048 && THRU_OUT == 32768 )) || THRU_KEY="${THRU_IN}x${THRU_OUT}_"
# The grid includes both the BF16 capacity (6) and the quantized capacity (84) reported in the
# paper; each arm is also measured once at its exact engine-reported capacity.
read -r -a THRU_BS <<< "${BSS:-1 2 4 8 16 32 64 128 256}"
# prefill: output_len 4 rather than 1 -- the reported prefill latency is last_ttft either way, but
# a 1-token generation makes (latency - ttft) ~ 0 and the throughput fields meaningless.
PRE_OUT=4
read -r -a PRE_CTXS <<< "${CTXS:-8192 16384 32768}"

source_arm_spec() {
  bash -c 'source "'"$REPO"'/config.sh"; '"$(sed -n '/^arm_spec()/,/^}/p' "$REPO/scripts/run_sweep.sh")"'; MODEL='"$MODEL"'; TAG='"$TAG"'; arm_spec '"$1"
}

gpu_guard() {
  nvidia-smi -L >/dev/null 2>&1 || pe_die "GPU guard: nvidia-smi -L failed"
  python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || pe_die "GPU guard: torch.cuda unavailable"
}

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -9 "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  for _ in $(seq 1 40); do
    pgrep -f "sglang.launch_serve[r].*--port $PORT( |$)" >/dev/null || break
    sleep 3
  done
  sleep 5
}
trap stop_server EXIT

# start_server <arm> <engine> <artifact> <extra_env> <ctx> <maxbs>
start_server() {
  local arm=$1 engine=$2 artifact=$3 extra_env=$4 ctx=$5 maxbs=$6
  # Use a distinct server log for each panel, arm, GPU, and throughput shape.
  local slog="$LOG/server_${PANEL}_${THRU_KEY:-}${arm}_gpu${GPU_ID}.log"
  local extra=()
  [[ "$PIECEWISE" == "1" ]] || extra+=(--disable-piecewise-cuda-graph)
  pe_say "$arm: booting gpu=$GPU_ID ctx=$ctx cuda-graph-max-bs=$maxbs mem-frac=$MEMFRAC chunked=$CHUNKED"
  # EXTRA_ENV / EXTRA_PYPATH inject an out-of-tree patch into the server; both empty by
  # default, so the baseline protocol is unchanged.
  ( cd "$REPO" && env $extra_env ${EXTRA_ENV:-} CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$REPO/python${EXTRA_PYPATH:+:$EXTRA_PYPATH}" \
      PREFILL_BACKEND="$_pb" TP=1 PREFIX_TOKENS="$PREFIX_TOKENS" RECENT_TOKENS="$RECENT_TOKENS" \
      nohup bash scripts/serve_method.sh "$MODEL" "$engine" "$artifact" \
      --port "$PORT" --max-running-requests "$maxbs" \
      --chunked-prefill-size "$CHUNKED" --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
      --mem-fraction-static "$MEMFRAC" --context-length "$ctx" \
      --cuda-graph-max-bs "$maxbs" "${extra[@]}" \
      > "$slog" 2>&1 & )
  sleep 2
  SERVER_PID=$(pgrep -f "sglang.launch_serve[r].*--port $PORT( |$)" | head -1)
  for _ in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    # fail fast instead of burning the whole health window on a dead server
    grep -qE "OutOfMemory|Tried to allocate|SIGQUIT received|Capture cuda graph failed" "$slog" 2>/dev/null \
      && { pe_say "$arm: server DIED during boot -- see $slog"; return 1; }
    sleep 10
  done
  curl -sf "http://localhost:$PORT/health" >/dev/null || { pe_say "$arm: server never came up -- see $slog"; return 1; }
  assert_graphs_on "$arm" "$slog" || return 1
  return 0
}

# The figure is only meaningful if every point ran under CUDA graphs. A boot that skipped or
# fell back to eager decode must not silently become a data point.
assert_graphs_on() {
  local arm=$1 slog=$2
  grep -q "Capture cuda graph end" "$slog" || {
    pe_say "$arm: REFUSING -- no 'Capture cuda graph end' in $slog (graphs did not capture)"; return 1; }
  grep -q "disable_cuda_graph=True" "$slog" && {
    pe_say "$arm: REFUSING -- server reports disable_cuda_graph=True"; return 1; }
  pe_say "$arm: cuda graphs captured ($(grep -c 'Capture cuda graph end' "$slog") capture block(s))"
  return 0
}

max_total_tokens() {
  curl -sf "http://localhost:$PORT/get_server_info" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("max_total_num_tokens",0))' 2>/dev/null \
    || echo 0
}

cell() { echo "$OUTDIR/${TAG}__${PANEL}__${1}__${2}.json"; }

# run_point <arm> <key> <bs> <inlen> <outlen> <artifact> <extra_env>
run_point() {
  local arm=$1 key=$2 bs=$3 inlen=$4 outlen=$5 artifact=$6 extra_env=$7
  local f; f="$(cell "$arm" "$key")"
  # Resume on SUCCESSFUL cells only. A cell whose JSON exists with "ok": false is what a killed or
  # crashed run leaves behind; counting it as done makes a re-run skip the point and the panel come
  # back missing it.
  if [[ -s "$f" && "${FORCE:-0}" != "1" ]]; then
    if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)' "$f" 2>/dev/null; then
      pe_say "$arm/$key: already done, skipping"; return 0
    fi
    pe_say "$arm/$key: previous cell recorded a FAILURE -- re-measuring"
  fi
  # Timeout sized from the cell: output_len * batch at a pessimistic 200 tok/s, plus 30 min for
  # boot and warmup (the largest throughput cells run for hours).
  local _to="${BENCH_TIMEOUT:-}"
  if [[ -z "$_to" ]]; then _to=$(( 1800 + outlen * bs / 200 )); fi
  # BENCH_POINT swaps in another per-point script with the same CLI (loopvq_efficiency uses it for
  # its profile / equivalence passes on this exact server path); unset = the measurement.
  python3 "${BENCH_POINT:-$HERE/bench_point.py}" --base-url "http://localhost:$PORT" --out "$f" --timeout "$_to" \
    --model "$MODEL" --tag "$TAG" --arm "$arm" --panel "$PANEL" --key "$key" \
    --batch-size "$bs" --input-len "$inlen" --output-len "$outlen" \
    --artifact "$artifact" --extra-env "$extra_env" 2>&1 | tee -a "$LOG/${PANEL}_${arm}.log"
}

pe_say "===== efficiency panel '$PANEL' -- model=$TAG arms=${ARM_LIST[*]} gpu=$GPU_ID ====="
gpu_guard

for arm in "${ARM_LIST[@]}"; do
  IFS='|' read -r engine artifact extra_env <<< "$(source_arm_spec "$arm")"
  [[ -e "$artifact" ]] || { pe_say "SKIP $arm: no artifact ($artifact)"; continue; }
  # Use one scale dtype across quantized methods unless explicitly overridden.
  EFF_SCALE_DTYPE="${EFF_SCALE_DTYPE-float16}"
  if [[ -n "$EFF_SCALE_DTYPE" && "$arm" != "bf16" && "$extra_env" != *"SCALE_DTYPE="* ]]; then
    extra_env="$extra_env SCALE_DTYPE=$EFF_SCALE_DTYPE"
    pe_say "$arm: SCALE_DTYPE not pinned by arm_spec -> forcing $EFF_SCALE_DTYPE (figure fairness)"
  fi
  # Quantized prefill requires the Triton wrapper but invokes the same effective FlashAttention
  # kernel used by the BF16 FA3 path. Record the selected backend in each cell's provenance.
  _pb="$PREFILL_BACKEND"
  [[ "$arm" == "bf16" ]] || _pb="${PREFILL_BACKEND_QUANT:-triton}"
  extra_env="$extra_env PREFILL_BACKEND=$_pb"
  _armvar="EXTRA_ENV_${arm^^}"
  if [[ -n "${!_armvar:-}" ]]; then
    extra_env="$extra_env ${!_armvar}"
    pe_say "$arm: per-arm env ${!_armvar}"
  fi

  case "$PANEL" in
  decode)
    maxctx=0; for c in "${DEC_CTXS[@]}"; do (( c > maxctx )) && maxctx=$c; done
    _srvctx=$(( maxctx + DEC_OUT + 1024 ))
    (( _srvctx > NATIVE_CTX )) && _srvctx=$NATIVE_CTX
    # +DEC_OUT+1024: the server refuses a request whose input+output equals its context length.
    # At batch 1 the pool holds one sequence, so take the smallest memory fraction whose pool fits
    # the longest point with a 5% margin; the rest is left to the unchunked prefill.
    _chosen=""
    for _mf in $DEC_MEMFRAC_LADDER; do
      CHUNKED="$DEC_CHUNKED" MEMFRAC="$_mf" \
        start_server "$arm" "$engine" "$artifact" "$extra_env" "$_srvctx" 8 || { stop_server; continue; }
      _mtt=$(max_total_tokens)
      if (( _mtt * 100 >= maxctx * 105 )); then
        pe_say "$arm: mem-fraction $_mf -> pool $_mtt tok, holds the $maxctx point (chosen)"
        _chosen="$_mf"; break
      fi
      pe_say "$arm: mem-fraction $_mf -> pool $_mtt tok, too small for $maxctx; stepping up"
      stop_server
    done
    [[ -n "$_chosen" ]] || { pe_say "SKIP $arm: no mem-fraction in '$DEC_MEMFRAC_LADDER' fits $maxctx"; stop_server; continue; }
    for ctx in "${DEC_CTXS[@]}"; do
      _in=$(( ctx - DEC_OUT ))
      # keep input+output strictly BELOW the served context (see NATIVE_CTX above)
      (( _in + DEC_OUT >= _srvctx )) && _in=$(( _srvctx - DEC_OUT - 1 ))
      run_point "$arm" "ctx${ctx}" 1 "$_in" "$DEC_OUT" "$artifact" "$extra_env"
    done
    stop_server ;;
  prefill)
    maxctx=0; for c in "${PRE_CTXS[@]}"; do (( c > maxctx )) && maxctx=$c; done
    start_server "$arm" "$engine" "$artifact" "$extra_env" $(( maxctx + PRE_OUT + 64 )) 8 \
      || { stop_server; continue; }
    for ctx in "${PRE_CTXS[@]}"; do
      run_point "$arm" "ctx${ctx}" 1 "$ctx" "$PRE_OUT" "$artifact" "$extra_env"
    done
    stop_server ;;
  thru)
    bsmax=0; for b in "${THRU_BS[@]}"; do (( b > bsmax )) && bsmax=$b; done
    # All arms use MEMFRAC by default. THRU_QUANT_MEMFRAC is retained as an explicit override, and
    # the selected value is recorded in each cell's provenance.
    _mf="$MEMFRAC"
    [[ "$arm" == "bf16" ]] || _mf="$THRU_QUANT_MEMFRAC"
    MEMFRAC="$_mf" start_server "$arm" "$engine" "$artifact" "$extra_env" \
      $(( THRU_IN + THRU_OUT + 64 )) "$bsmax" || { stop_server; continue; }
    # Capacity from the ENGINE's own pool, not from a guess: the largest batch whose
    # (input + output) KV fits max_total_num_tokens. This is also the bf16 capacity line.
    mtt=$(max_total_tokens)
    capb=$(( mtt / (THRU_IN + THRU_OUT) ))
    pe_say "$arm: max_total_num_tokens=$mtt -> capacity $capb requests at $((THRU_IN+THRU_OUT)) tok"
    # THRU_CAP_FROM=<arm>: cap this arm's batch grid at that arm's capacity, so the quantized arms
    # end at the same batch. The arm's own capacity is still recorded (capacity_batch_own).
    _capown=$capb; _capsrc="own"
    if [[ -n "${THRU_CAP_FROM:-}" && "$arm" != "${THRU_CAP_FROM}" ]]; then
      _capf="$OUTDIR/${TAG}__thru_capacity__${THRU_KEY}${THRU_CAP_FROM}.json"
      pe_say "$arm: waiting for ${THRU_CAP_FROM}'s capacity ($_capf)"
      for _ in $(seq 1 360); do [[ -s "$_capf" ]] && break; sleep 10; done
      if [[ -s "$_capf" ]]; then
        _capref=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["capacity_batch"])' "$_capf" 2>/dev/null || echo 0)
        if (( _capref > 0 && _capref < capb )); then
          pe_say "$arm: capacity $capb -> $_capref (pinned to ${THRU_CAP_FROM})"
          capb=$_capref; _capsrc="$THRU_CAP_FROM"
        else
          pe_say "$arm: own capacity $capb <= ${THRU_CAP_FROM}'s $_capref -- not raised, grid stays at $capb"
        fi
      else
        pe_die "$arm: THRU_CAP_FROM=${THRU_CAP_FROM} but $_capf never appeared"
      fi
    fi
    echo "{\"model\":\"$TAG\",\"arm\":\"$arm\",\"max_total_num_tokens\":$mtt,\"tokens_per_request\":$((THRU_IN+THRU_OUT)),\"capacity_batch\":$capb,\"capacity_batch_own\":$_capown,\"capacity_source\":\"$_capsrc\"}" \
      > "$OUTDIR/${TAG}__thru_capacity__${THRU_KEY}${arm}.json"
    # The arm's measured capacity is run as a final point after the power-of-two grid, so each curve
    # ends where that arm runs out of memory.
    _ran_max=0
    for bs in "${THRU_BS[@]}"; do
      if (( bs > capb )); then pe_say "$arm/bs${bs}: ABOVE CAPACITY ($capb) -- not run"; continue; fi
      run_point "$arm" "${THRU_KEY}bs${bs}" "$bs" "$THRU_IN" "$THRU_OUT" "$artifact" "$extra_env"
      _ran_max=$bs
    done
    if (( capb > _ran_max && capb <= bsmax )) && [[ "${THRU_SKIP_CAPACITY:-0}" == "1" ]]; then
      # THRU_SKIP_CAPACITY=1 runs the power-of-two grid only and leaves the (most expensive)
      # capacity point for a later pass, which must use the same BSS.
      pe_say "$arm: capacity point bs=$capb SKIPPED (THRU_SKIP_CAPACITY=1)"
    elif (( capb > _ran_max && capb <= bsmax )); then
      pe_say "$arm: capacity point bs=$capb (grid stopped at $_ran_max)"
      run_point "$arm" "${THRU_KEY}bs${capb}" "$capb" "$THRU_IN" "$THRU_OUT" "$artifact" "$extra_env"
    elif (( capb > bsmax )); then
      pe_say "$arm: capacity $capb exceeds --cuda-graph-max-bs $bsmax; not run (would serve eager)"
    fi
    stop_server ;;
  *) pe_die "unknown panel '$PANEL'" ;;
  esac
done
pe_say "===== panel '$PANEL' done -- cells in $OUTDIR ====="
