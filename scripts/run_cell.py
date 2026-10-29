#!/usr/bin/env python3
"""Resolve and run one task/model/method cell through an OpenAI-compatible server."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen


def contains_answer(prediction, answers):
    """NIAH scoring: the needle counts as found if any accepted answer appears in the
    generation, compared on lowercase alphanumerics with runs of anything else collapsed
    to a single space."""
    def norm(t):
        return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())
    p = norm(prediction)
    return float(any(norm(a) in p for a in answers))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input", type=Path, required=True, help="JSONL rows with prompt and answers")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--base-url", default=os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:30000"))
    p.add_argument("--resume", action="store_true",
                   help="keep the complete rows already in --output (matched on _id) and append the rest")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.open()]
    if args.limit is not None:
        rows = rows[: args.limit]
    mode = "w"
    if args.resume and args.output.exists():
        # A run killed mid-write can leave a torn last line: keep only lines that parse.
        kept = []
        for line in args.output.open():
            try:
                kept.append(json.loads(line))
            except json.JSONDecodeError:
                break
        with args.output.open("w") as dst:
            dst.writelines(json.dumps(r) + "\n" for r in kept)
        done = {r["_id"] for r in kept}
        rows = [r for r in rows if r["_id"] not in done]
        mode = "a"
        print(f"resume: {len(done)} rows kept, {len(rows)} to run", flush=True)

    def generate(index_row):
        index, row = index_row
        body = {"text": row["prompt"], "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_tokens or row.get("max_new_tokens", 1024),
                **row.get("sampling_extra", {})}}   # e.g. LongBench samsum: stop at "\n"
        req = Request(args.base_url + "/generate", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10800) as response:
            result = json.load(response)
        prediction = result["text"]
        answers = row.get("answers", row.get("answer", []))
        return {**row, "answers": answers, "prediction": prediction,
                "correct": contains_answer(prediction, answers),
                "task": args.task, "method": args.method, "seed": args.seed}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, args.output.open(mode) as dst:
        for result in pool.map(generate, enumerate(rows)):
            dst.write(json.dumps(result) + "\n")
            dst.flush()


if __name__ == "__main__":
    main()
