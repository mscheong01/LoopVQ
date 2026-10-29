"""MATH500, zero-shot CoT + Math-Verify, against a running SGLang server.

ResidualQuant (arXiv 2610.10381, Appendix A.2 / Table 7) evaluates Ouro-1.4B on MATH500 with a
0-shot plain-text prompt that asks for step-by-step reasoning and a final answer in \\boxed{},
2048 new tokens, greedy, scored with Math-Verify (bf16 75.0). Their exact wording is not
published; PROMPT below follows that description. Math-Verify accepts mathematically equivalent
forms (1/2 vs 0.5, \\dfrac vs \\frac, ...), which the hendrycks string match in tasks/math500
does not -- on the same 4-shot bf16 completions it alone moves 64.0 to 70.2.

Writes <output_dir>/<save_postfix>_<seed>_math500_mv.json as {"math500_mv": {"acc": ..}}.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROMPT = ("Solve the following math problem step by step. Put your final answer within "
          "\\boxed{{}}.\n\nProblem: {problem}\n\nSolution:")
# "\n\nQuestion:": Ouro base models often box the answer and then invent and solve a new
# "Question: ... Solution:" of their own, whose box Math-Verify would score
STOPS = ["\n\nProblem:", "\n\nQuestion:", "<|endoftext|>", "<|im_end|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", default="http://localhost:30000/v1/completions")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_postfix", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_concurrent", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    from math_verify import parse, verify

    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))[: a.limit]

    def gen(i):
        body = {"model": a.model_name, "prompt": PROMPT.format(problem=ds[i]["problem"]),
                "temperature": 0.0, "max_tokens": a.max_tokens, "stop": STOPS, "seed": a.seed}
        r = requests.post(a.base_url, json=body, timeout=a.timeout)
        r.raise_for_status()
        return i, r.json()["choices"][0]["text"]

    # completions are appended as they finish so an interrupted run resumes where it stopped
    os.makedirs(a.output_dir, exist_ok=True)
    stem = os.path.join(a.output_dir, f"{a.save_postfix}_{a.seed}_math500_mv")
    outs = {}
    if os.path.exists(stem + "_partial.jsonl"):
        for l in open(stem + "_partial.jsonl"):
            r = json.loads(l)
            outs[r["idx"]] = r["completion"]
    todo = [i for i in range(len(ds)) if i not in outs]
    with ThreadPoolExecutor(a.num_concurrent) as ex, open(stem + "_partial.jsonl", "a") as f:
        # one failed request must not discard the rest: the pool would still wait for every
        # other request before the exception surfaced, and their completions would be lost
        failed = 0
        for fut in as_completed([ex.submit(gen, i) for i in todo]):
            try:
                i, t = fut.result()
            except Exception as e:
                failed += 1
                print(f"request failed: {e!r}", file=sys.stderr, flush=True)
                continue
            outs[i] = t
            f.write(json.dumps({"idx": i, "completion": t}) + "\n")
            f.flush()
    if failed:
        sys.exit(f"{failed} requests failed; the {len(outs)} completions are kept in "
                 f"{stem}_partial.jsonl and a re-run resumes from them")

    rows, ok = [], 0
    for i, d in enumerate(ds):
        try:
            hit = bool(verify(parse("$" + d["answer"] + "$"), parse(outs[i])))
        except Exception:
            hit = False
        ok += hit
        rows.append({"idx": i, "answer": d["answer"], "completion": outs[i], "correct": hit})
    res = {"math500_mv": {"acc": ok / len(ds), "n": len(ds), "shots": 0, "scorer": "math_verify"}}
    json.dump(res, open(stem + ".json", "w"))
    with open(stem + "_samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
