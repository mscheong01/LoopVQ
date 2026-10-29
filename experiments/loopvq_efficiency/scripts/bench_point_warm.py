#!/usr/bin/env python3
"""Prefill point with a same-length warmup: run bench_point.py (the measurement) 1 + WARM_N times on the
same server, keep the first as cold_ttft_s, and write the median-TTFT warm run as the cell.

Plugs into run_efficiency.sh through BENCH_POINT (same CLI as bench_point.py). bench_one_batch_server
only warms up at 1k, so a single-shot cell would time the first request the server sees at that length.

    BENCH_POINT=$PWD/bench_point_warm.py VARIANT=prefill_warm ./run.sh prefill
"""
from __future__ import annotations

import json, os, statistics, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_point.py"


def main() -> int:
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--out") + 1])
    n = int(os.environ.get("WARM_N", "5"))
    runs = []
    for i in range(1 + n):
        tmp = out.with_name(out.stem + f".run{i}.json")
        a = list(argv); a[a.index("--out") + 1] = str(tmp)
        rc = subprocess.run([sys.executable, str(BENCH)] + a).returncode
        cell = json.loads(tmp.read_text()) if tmp.exists() else {"ok": False}
        if rc != 0 or not cell.get("ok"):
            cell["error_run"] = i
            out.write_text(json.dumps(cell, indent=2) + "\n")
            return 1
        runs.append(cell)
    cold, warm = runs[0], runs[1:]
    samples = [c["metrics"]["prefill_latency_s"] for c in warm]
    med = sorted(warm, key=lambda c: c["metrics"]["prefill_latency_s"])[len(warm) // 2]
    cell = dict(med)
    cell["protocol"] = f"same-length warmup + median of {n} (bench_point_warm.py)"
    cell["wall_s"] = round(sum(c["wall_s"] for c in runs), 2)
    cell["metrics"] = dict(med["metrics"], prefill_samples_s=samples,
                           cold_ttft_s=cold["metrics"]["prefill_latency_s"])
    out.write_text(json.dumps(cell, indent=2) + "\n")
    with out.with_suffix(".raw.jsonl").open("w") as f:
        for c in warm:
            f.write(json.dumps(c["metrics"]) + "\n")
    with out.with_suffix(".raw.cold.jsonl").open("w") as f:
        f.write(json.dumps(cold["metrics"]) + "\n")
    for i in range(1 + n):
        for suf in (".json", ".raw.jsonl"):
            p = out.with_name(out.stem + f".run{i}{suf}")
            if p.exists():
                p.unlink()
    print(f"[warm] {out.name}: cold {cold['metrics']['prefill_latency_s']:.3f}s  warm median "
          f"{statistics.median(samples):.3f}s  samples {[round(s, 3) for s in samples]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
