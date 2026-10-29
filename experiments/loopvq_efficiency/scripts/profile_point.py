#!/usr/bin/env python3
"""Profile ONE point on the running server and break its GPU time down by kernel category.

Drop-in for bench_point.py (same CLI; run_efficiency.sh calls it when
BENCH_POINT points here -- see run.sh `profile`). It runs sglang's bench_one_batch_server with the
torch profiler on (--profile --profile-by-stage), then reads the chrome traces it dumps and writes:

    <out>.json            cell: ok, per-stage {category: ms, share}, step count, idle share
    <out>.prof/           the raw traces (open in perfetto.dev)

Categories are regexes over kernel names (CATS below). "idle" is the part of the stage's GPU span
not covered by any kernel -- launch gaps, host syncs, Python between kernels.
Unmatched kernels land in "other" and the top ones are listed so CATS can be extended.
"""
from __future__ import annotations

import argparse, gzip, json, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

CATS = [  # first match wins; order matters
    ("vq_stage1", r"vqwide|vqg_|vq2_|_quant_vqwide|stage1_quant|vq_decode"),
    ("hp_stage1", r"prerope|_fwd_grouped_kernel_stage1|_fwd_kernel_stage1|decode_grouped_att"),
    ("stage2", r"stage2"),
    ("encode_flush", r"encode|flush|_loop_|regroup"),
    ("kv_write", r"set_kv|set_mixed_hp|store_kv|kv_buffer|index_copy|scatter|index_put"),
    ("kv_meta", r"mixed_kv_indices|build_.*indices|create_flashinfer|kv_indices|num_kv_splits"),
    ("prefill_attn", r"flash|fmha|fwd_kernel|varlen|attn_fwd"),
    ("gemm", r"gemm|cutlass|cublas|matmul|xmma|sm\d+_|ampere|_mm_|splitk"),
    ("norm_act_rope", r"rms|norm|silu|gelu|act_and_mul|rotary|rope"),
    ("copy_cast", r"copy|cast|cat|elementwise|fill|memset|Memcpy|Memset|to_copy|contiguous"),
    ("sampling", r"argmax|topk|softmax|sampl|reduce"),
]
CAT_RE = [(n, re.compile(p, re.I)) for n, p in CATS]


def cat_of(name: str) -> str:
    for n, r in CAT_RE:
        if r.search(name):
            return n
    return "other"


def union_ms(iv):
    tot, end = 0.0, -1.0
    for s, e in sorted(iv):
        if s > end:
            tot += e - s; end = e
        elif e > end:
            tot += e - end; end = e
    return tot / 1000.0


def breakdown(trace: Path) -> dict:
    op = gzip.open if trace.suffix == ".gz" else open
    with op(trace, "rt") as f:
        ev = json.load(f).get("traceEvents", [])
    k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k:
        return {}
    by, names, iv = defaultdict(float), defaultdict(float), []
    for e in k:
        d = float(e.get("dur", 0)); s = float(e["ts"])
        c = cat_of(e.get("name", ""))
        by[c] += d / 1000.0
        if c == "other":
            names[e.get("name", "")[:90]] += d / 1000.0
        iv.append((s, s + d))
    span = (max(e for _, e in iv) - min(s for s, _ in iv)) / 1000.0
    busy = union_ms(iv)
    out = {"span_ms": round(span, 3), "busy_ms": round(busy, 3),
           "idle_ms": round(span - busy, 3), "idle_share": round((span - busy) / span, 4) if span else 0,
           "kernels": len(k),
           "by_cat_ms": {c: round(v, 3) for c, v in sorted(by.items(), key=lambda x: -x[1])},
           "top_other": sorted(([n, round(v, 3)] for n, v in names.items()), key=lambda x: -x[1])[:8]}
    tot = sum(by.values())
    out["by_cat_share"] = {c: round(v / tot, 4) for c, v in out["by_cat_ms"].items()} if tot else {}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--base-url", "--out", "--model", "--tag", "--arm", "--panel", "--key"):
        ap.add_argument(a, required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--artifact", default=""); ap.add_argument("--extra-env", default="")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--profile-steps", type=int, default=int(__import__("os").environ.get("PROFILE_STEPS", "8")))
    a = ap.parse_args()

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    pdir = outp.with_suffix(".prof"); pdir.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "sglang.bench_one_batch_server", "--model-path", "None",
           "--base-url", a.base_url, "--batch-size", str(a.batch_size),
           "--input-len", str(a.input_len), "--output-len", str(a.output_len),
           "--no-append-to-github-summary", "--profile", "--profile-by-stage",
           "--profile-steps", str(a.profile_steps), "--profile-output-dir", str(pdir)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
    traces = sorted(pdir.rglob("*.trace.json*"))
    stages = {}
    for t in traces:
        st = "prefill" if re.search(r"extend|prefill", t.name, re.I) else "decode" if "decode" in t.name.lower() else t.stem
        b = breakdown(t)
        if b:
            stages.setdefault(st, b)
    cell = {"model": a.tag, "arm": a.arm, "panel": a.panel, "key": a.key, "batch_size": a.batch_size,
            "input_len": a.input_len, "output_len": a.output_len, "ok": bool(stages),
            "wall_s": round(time.time() - t0, 2), "profile_steps": a.profile_steps,
            "stages": stages, "traces": [str(t) for t in traces], "extra_env": a.extra_env}
    if not stages:
        cell["error"] = {"returncode": p.returncode, "stderr_tail": p.stderr[-3000:], "stdout_tail": p.stdout[-1500:]}
    outp.write_text(json.dumps(cell, indent=2) + "\n")
    print(("[prof] OK   " if cell["ok"] else "[prof] FAIL ") + f"{a.arm}/{a.key} -> {outp}")
    for st, b in stages.items():
        top = ", ".join(f"{c} {v*100:.0f}%" for c, v in list(b["by_cat_share"].items())[:6])
        print(f"        {st}: span {b['span_ms']:.1f} ms, idle {b['idle_share']*100:.0f}% | {top}")
    return 0 if cell["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
