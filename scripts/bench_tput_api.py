"""Decode throughput of one served arm, against a running SGLang server (native /generate).

For each (batch, prompt_len) point: B concurrent requests of random token ids, greedy, EOS
ignored. Run once with 1 new token (prefill + first token) and once with N; the difference is
pure decode, so decode tok/s = B * (N - 1) / (t_N - t_1) and per-step latency = (t_N - t_1) / (N - 1).
Each point is preceded by a short warm-up at the same shape so JIT builds and graph capture do not
land in the timing.

Writes <output_dir>/<save_postfix>_0_tput.json as {"tput": {"B<b>_P<p>": {...}, ...}}.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:30000/v1/completions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_postfix", default="run")
    ap.add_argument("--points", default="1x1024,8x1024,32x1024,1x8192,4x8192",
                    help="comma list of <batch>x<prompt_len>")
    ap.add_argument("--new_tokens", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=os.environ.get("TPUT_DUMP"),
                    help="write each point's greedy output token ids here (path prefix) to compare runs")
    ap.add_argument("--hold", type=float, default=float(os.environ.get("TPUT_HOLD", 0)),
                    help="seconds to keep the server up afterwards (lets a profiler trace finish)")
    a, _ = ap.parse_known_args()   # the sweep passes the common eval flags; ignore the rest
    url = a.base_url.split("/v1/")[0] + "/generate"

    import random
    rng = random.Random(a.seed)

    def batch(b, p, n):
        prompts = [[rng.randrange(100, a.vocab) for _ in range(p)] for _ in range(b)]
        body = lambda ids: {"input_ids": ids, "sampling_params": {
            "max_new_tokens": n, "temperature": 0.0, "ignore_eos": True}}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(b) as ex:
            outs = list(ex.map(lambda ids: requests.post(url, json=body(ids), timeout=3600).json(),
                               prompts))
        dt = time.perf_counter() - t0
        got = [o["meta_info"]["completion_tokens"] for o in outs]
        assert all(g == n for g in got), got
        if a.dump and n > 16:
            with open(f"{a.dump}_B{b}_P{p}.jsonl", "w") as f:
                for o in outs:
                    f.write(json.dumps(o.get("output_ids") or o["text"]) + "\n")
        return dt

    res = {}
    for pt in a.points.split(","):
        b, p = map(int, pt.split("x"))
        batch(b, p, 16)                                   # warm-up at this shape
        t1 = min(batch(b, p, 1) for _ in range(2))
        tn = min(batch(b, p, a.new_tokens) for _ in range(2))
        step = (tn - t1) / (a.new_tokens - 1)
        res[f"B{b}_P{p}"] = {"decode_tok_s": b / step, "step_ms": 1e3 * step,
                             "prefill_s": t1, "batch": b, "prompt": p, "new": a.new_tokens}
        print(pt, json.dumps(res[f"B{b}_P{p}"]), flush=True)
    os.makedirs(a.output_dir, exist_ok=True)
    json.dump({"tput": res}, open(os.path.join(a.output_dir, f"{a.save_postfix}_0_tput.json"), "w"))
    time.sleep(a.hold)


if __name__ == "__main__":
    main()
