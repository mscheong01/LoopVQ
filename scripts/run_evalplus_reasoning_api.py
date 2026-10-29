"""HumanEval+ / MBPP+ in reasoning mode (thinking on) against a running SGLang server.

The thinking-mode counterpart of run_evalplus_api.py, for the loopvq_reasoning table. lm_eval's
`humaneval_instruct` does not work here: it primes the assistant turn with an open code fence, and
a thinking model's reply begins with its reasoning, so the extracted "code" is reasoning text
(Ouro-1.4B-Thinking scored 3.7 that way at bf16).

  * prompt: one user turn, EvalPlus's instruction + the task prompt in a code block, through the
    chat template with enable_thinking=True when the template has the switch
  * sampled decoding (the reasoning protocol's temperature / top-p / budget), one seed per call
  * answer: the LAST fenced code block of the reply (after "</think>" when the server leaves the
    tag in the text; SGLang strips it as a special token), then evalplus.sanitize(entrypoint) as
    in run_evalplus_api.py. A reply cut by the budget usually has no final block and fails, as
    it should; "finished" in the result is the share that stopped on their own.
  * score: untrusted_check on the base and the plus suites; pass@1 over the problems

Writes <output_dir>/<save_postfix>_<seed>_evalplus_<bench>.json as
{"evalplus_<bench>": {"pass@1,base": .., "pass@1,plus": ..}} plus a _samples.jsonl with the raw
reply, so a finish-rate column can be computed without re-running.
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

INSTRUCTION = ("Please provide a self-contained Python script that solves the following problem "
               "in a markdown code block:")
FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.S)


def answer_part(text):
    i = text.rfind("</think>")
    return text if i < 0 else text[i + len("</think>"):]


MAX_LINES = 150    # evalplus.sanitize falls back to an O(lines^2) ast.parse scan on unparsable
                   # input; a budget-cut reply handed over whole (thousands of lines) never returns


def last_code_block(text):
    blocks = FENCE.findall(text)
    if blocks:
        code = blocks[-1]
    else:
        i = text.rfind("```")                   # an unclosed final fence (budget ran out)
        code = text[text.find("\n", i) + 1:] if i >= 0 else ""   # no fence at all: no answer
    return "\n".join(code.splitlines()[-MAX_LINES:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("humaneval", "mbpp"), required=True)
    ap.add_argument("--base_url", default="http://localhost:30000/v1/chat/completions")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_postfix", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_concurrent", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--max_gen_toks", type=int, default=32768)
    ap.add_argument("--timeout", type=int, default=48 * 3600)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    from evalplus.eval import PASS, untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth
    from evalplus.sanitize import sanitize

    if a.bench == "humaneval":
        problems, hashcode, not_none = get_human_eval_plus(), get_human_eval_plus_hash(), []
    else:
        problems, hashcode, not_none = get_mbpp_plus(), get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS
    gt = get_groundtruth(problems, hashcode, not_none)
    tids = sorted(problems, key=lambda t: int(t.rsplit("/", 1)[1]))[: a.limit]

    tok = AutoTokenizer.from_pretrained(a.model_name, trust_remote_code=True)
    hybrid = "enable_thinking" in (tok.chat_template or "")
    print(f"[evalplus-reasoning] enable_thinking switch in template: {hybrid}", flush=True)

    def gen(tid):
        msg = f"{INSTRUCTION}\n```\n{problems[tid]['prompt'].strip()}\n```"
        body = {"model": a.model_name, "messages": [{"role": "user", "content": msg}],
                "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_gen_toks,
                "seed": a.seed}
        if hybrid:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        r = requests.post(a.base_url, json=body, timeout=a.timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        return tid, ch["message"].get("content") or "", ch.get("finish_reason")

    outs, failed = {}, 0
    with ThreadPoolExecutor(a.num_concurrent) as ex:
        futs = [ex.submit(gen, t) for t in tids]
        for f in as_completed(futs):
            try:
                tid, text, fin = f.result()
                outs[tid] = (text, fin)
            except Exception as e:                  # keep the others; fail the cell at the end
                failed += 1
                print(f"[evalplus-reasoning] request failed: {e}", flush=True)
            done = len(outs) + failed
            if done % 10 == 0 or done == len(tids):  # progress, readable with tail on the eval log
                stop = sum(fin == "stop" for _, fin in outs.values())
                print(f"[evalplus-reasoning] {done}/{len(tids)} done ({stop} finished on their own, "
                      f"{failed} failed) {time.strftime('%H:%M:%S')}", flush=True)
    if failed:
        sys.exit(f"{failed} requests failed; no result written")
    task = f"evalplus_{a.bench}"
    stem = os.path.join(a.output_dir, f"{a.save_postfix}_{a.seed}_{task}")
    os.makedirs(a.output_dir, exist_ok=True)
    with open(stem + "_raw.jsonl", "w") as f:   # generations first, so scoring can be redone
        for tid in tids:
            f.write(json.dumps({"task_id": tid, "resp": outs[tid][0], "finish_reason": outs[tid][1]}) + "\n")

    bench = "humaneval" if a.bench == "humaneval" else "mbpp"
    rows, base_ok, plus_ok, finished = [], 0, 0, 0
    for tid in tids:
        p, g = problems[tid], gt[tid]
        text, fin = outs[tid]
        finished += fin == "stop"
        code = last_code_block(answer_part(text))
        san = sanitize(code, entrypoint=p["entry_point"]) if code.strip() else ""
        if code.strip() and not san.strip():
            san = sanitize(p["prompt"] + code, entrypoint=p["entry_point"])
        b, _ = untrusted_check(bench, san, p["base_input"], p["entry_point"],
                               g["base"], p["atol"], g["base_time"])
        q, _ = untrusted_check(bench, san, p["plus_input"], p["entry_point"],
                               g["plus"], p["atol"], g["plus_time"])
        base_ok += b == PASS
        plus_ok += q == PASS
        rows.append({"task_id": tid, "resp": text, "finish_reason": fin, "sanitized": san,
                     "base": str(b), "plus": str(q)})
    n = len(tids)
    res = {task: {"pass@1,base": base_ok / n, "pass@1,plus": plus_ok / n, "n": n,
                  "finished": finished / n, "thinking": hybrid}}
    json.dump(res, open(stem + ".json", "w"))
    with open(stem + "_samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
