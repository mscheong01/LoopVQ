#!/bin/bash
# Shared plumbing for the experiment harness. Sourced, not run.
#
# Points the build and sweep scripts at the experiment artifact and work directories and calls
# scripts/run_sweep.sh, so serving, protocol, and method settings stay in config.sh.
set -uo pipefail

PE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$PE")"

export ARTIFACTS="$PE/artifacts"
# PE_WORK: a separate work dir for a concurrent run on another node (logs/results names collide)
export WORK="${PE_WORK:-$PE/work}"

# The metric key per benchmark lives in table_lib.METRICS.

pe_say() { echo "[pe] $* $(date -Is)"; }
pe_die() { echo "[pe] FAILED: $*" >&2; exit 1; }

# emit_result <outdir> <model_tag> <bench> <method> <seed|-> <src_json>
#
# Copies a finished lm_eval result into the table's input directory, with the provenance (bundle
# hash, bf16 windows, load) the table builder checks before averaging cells.
emit_result() {
  local outdir=$1 tag=$2 bench=$3 method=$4 seed=$5 src=$6
  [[ -s "$src" ]] || { echo "[pe] no result to emit: $src" >&2; return 1; }
  local name; name=$(SEED="${seed/#-/}" pe_cell_name "$tag" "$bench" "$method")
  local art="$ARTIFACTS/${tag}_$(pe_artifact_suffix "$method")"
  # bf16 has no bundle; every other arm is identified by its codebook.pt.
  local ahash="n/a"
  [[ -f "$art/codebook.pt" ]] && ahash=$(md5sum "$art/codebook.pt" | cut -c1-12)
  if [[ "$method" != "bf16" && "$ahash" == "n/a" ]]; then
    echo "[pe] WARNING no bundle found under $art -- artifact provenance will be blank" >&2
  fi
  mkdir -p "$outdir"
  python3 - "$src" "$outdir/$name.json" "$tag" "$bench" "$method" "$seed" "$ahash" <<'PY'
import json, os, subprocess, sys, socket, datetime
src, dst, tag, bench, method, seed, ahash = sys.argv[1:8]
res = json.load(open(src))
def git(*a):
    # run in the repo, not the caller's directory
    try:
        return subprocess.check_output(["git", *a], text=True, cwd=os.environ["PE_REPO"],
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
json.dump({
    "model": tag, "bench": bench, "method": method,
    "seed": None if seed == "-" else int(seed),
    "results": res.get("results", res),
    "provenance": {
        "commit": git("rev-parse", "--short", "HEAD"),
        # tracked modifications only
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "dirty_files": git("diff", "--name-only", "HEAD") or None,
        "artifact_md5": ahash,
        "scale_dtype": os.environ.get("PE_SCALE_DTYPE"),
        "prefix_tokens": os.environ.get("PREFIX_TOKENS"),
        "recent_tokens": os.environ.get("RECENT_TOKENS"),
        "num_concurrent": os.environ.get("NUM_CONCURRENT"),
        # with num_concurrent, the load the cell was served under
        "max_running": os.environ.get("MAX_RUNNING"),
        "host": socket.gethostname(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    },
}, open(dst, "w"), indent=1)
print(f"[pe] wrote {dst}")
PY
}

# pe_cell_name <tag> <bench> <method> -> the outputs/ basename (no .json)
pe_cell_name() {
  local n="$1__$2__$3"
  [[ -n "${SEED:-}" ]]      && n="${n}__s${SEED}"
  [[ -n "${REPLICATE:-}" ]] && n="${n}__r${REPLICATE}"
  echo "$n"
}

# A completed legacy baseline also satisfies the skip check. New writes use bf16.
pe_cell_exists() {
  local outdir=$1 model=$2 bench=$3 method=$4
  [[ -s "$outdir/$(pe_cell_name "$model" "$bench" "$method").json" ]] && return 0
  [[ "$method" == "bf16" ]] &&
    [[ -s "$outdir/$(pe_cell_name "$model" "$bench" fp16).json" ]]
}

# pe_run_cell <outdir> <hf-model> <bench> <methods...>
#
# One table cell = one (model, bench) across the method axis; results land in outputs/. A method
# whose output already exists is skipped (FORCE=1 re-runs it), so re-running a script is safe.
pe_run_cell() {
  local outdir=$1 model=$2 bench=$3; shift 3
  local methods=("$@"); [[ $# -eq 0 ]] && pe_die "no arms given (e.g. bf16 loopvq_0.5bit)"
  source "$REPO/config.sh"
  local i
  for i in "${!methods[@]}"; do methods[$i]=$(canonical_method "${methods[$i]}" "$model"); done
  # emit_result reads these from the environment
  export PREFIX_TOKENS RECENT_TOKENS NUM_CONCURRENT MAX_RUNNING PE_REPO="$REPO"
  # checked before anything is served
  [[ -n "${PREFIX_TOKENS:-}" && -n "${RECENT_TOKENS:-}" ]] \
    || pe_die "residual protocol unset after sourcing config.sh -- refusing to run a cell whose provenance cannot be recorded"
  local tag; tag=$(model_tag "$model")
  # A multi-seed (reasoning) cell is one file per seed (..._s0.json) and counts as done only
  # when every seed file is present.
  local seeds="0"
  [[ "${REASONING:-0}" == "1" ]] && seeds="${REASON_SEEDS:-0}"
  # reasoning cells always carry __sN, however many seeds this invocation runs
  local multi=0; [[ "${REASONING:-0}" == "1" ]] && multi=1

  local todo=() done_list=()
  for m in "${methods[@]}"; do
    local complete=1 s
    for s in $seeds; do
      if [[ "$multi" == 1 ]]; then
        SEED="$s" pe_cell_exists "$outdir" "$tag" "$bench" "$m" || complete=0
      else
        pe_cell_exists "$outdir" "$tag" "$bench" "$m" || complete=0
      fi
    done
    if [[ "$complete" == 1 && "${FORCE:-0}" != "1" ]]; then done_list+=("$m"); else todo+=("$m"); fi
  done
  [[ ${#done_list[@]} -gt 0 ]] && pe_say "ALREADY RUN, skipping: ${done_list[*]}  (in $outdir)"
  if [[ ${#todo[@]} -eq 0 ]]; then
    pe_say "nothing to do for $tag/$bench -- all ${#methods[@]} methods already in outputs/. Use FORCE=1 to re-run."
    return 0
  fi
  pe_say "running $tag/$bench: ${todo[*]}"

  local raw="$WORK/$tag/results"
  # The reasoning path is sampled and runs every seed against one set of servers; the short-task
  # path is greedy, so it runs seed 0 only.
  for m in "${todo[@]}"; do
    # the per-token scale dtype serve_method.sh uses (float16 for the vq arms)
    case "$m" in
      cq8b_g*|loopvq8b_g*)  export PE_SCALE_DTYPE=float16 ;;
      *)                    export PE_SCALE_DTYPE=float32 ;;
    esac
    # outputs/ decides what is done, so a stale raw result is removed rather than reused
    for s in $seeds; do rm -f "$raw/${tag}_${m}_${s}_${bench}.json"; done
    if [[ "${REASONING:-0}" == "1" ]]; then
      REASONING_TASKS="$bench" REASON_SEEDS="$seeds" bash "$REPO/scripts/run_sweep.sh" "$model" "$m"
    else
      SHORT_TASKS="$bench" bash "$REPO/scripts/run_sweep.sh" "$model" "$m"
    fi
    for s in $seeds; do
      local tag_seed="-"
      [[ "$multi" == 1 ]] && tag_seed="$s"
      emit_result "$outdir" "$tag" "$bench" "$m" "$tag_seed" "$raw/${tag}_${m}_${s}_${bench}.json" \
        || pe_say "NO RESULT for $m/$bench seed $s -- see $WORK/$tag/logs/eval_${m}_${bench}_s${s}.log"
    done
  done
}

# arm name -> artifact directory suffix, mirroring scripts/run_sweep.sh's arm_spec
pe_artifact_suffix() {
  case "$1" in
    bf16) echo "" ;;
    *)    echo "$1" ;;
  esac
}
