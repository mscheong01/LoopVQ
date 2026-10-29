"""HumanEval+ / MBPP+ against a running SGLang server, EvalPlus protocol.

Mirrors the setup LoopCD (arXiv 2609.24196, github.com/hoeng4/LoopCD, loopcd/eval/
eval_code_common.py) uses for Ouro, which is EvalPlus's own codegen path:

  * prompt: when the tokenizer has a chat template (Ouro ships ChatML) the task is wrapped by
    evalplus.provider.utility.make_raw_chat_prompt -- instruction "Please provide a
    self-contained Python script ... in a markdown code block:" and an assistant turn primed
    with "Below is a Python script ...:\\n```python\\n". Otherwise the bare task prompt.
  * zero-shot, greedy, 1024 new tokens, stops BASE_EOS + "\\n```\\n" (chat) or + function
    boundaries (bare)
  * score: evalplus.sanitize(text, entrypoint) (falling back to prompt + text), then
    untrusted_check on the base and the plus test suites.

lm_eval's `humaneval` / 3-shot `mbpp` are NOT this protocol: Ouro under them scored 0.0-33.5 on
HumanEval at bf16, against LoopCD's 69.5 here. GSM8K / MATH500 stay on lm_eval (LoopCD uses
plain few-shot for those too).

Writes <output_dir>/<save_postfix>_<seed>_evalplus_<bench>.json as
{"evalplus_<bench>": {"pass@1,base": .., "pass@1,plus": ..}} plus a _samples.jsonl.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests

BASE_EOS = ["<|endoftext|>", "<|endofmask|>", "</s>", "\nif __name__", "\ndef main(", "\nprint("]
HE_EXTRA_DIRECT = ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
HE_EXTRA_CHAT = ["\n```\n"]
INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following "
                      "problem in a markdown code block:")
RESPONSE_PREFIX = ("Below is a Python script with a self-contained function that solves the "
                   "problem and passes corresponding tests:")


def trim_eos(text, eos):
    cut = min([i for i in (text.find(e) for e in eos) if i != -1], default=len(text))
    return text[:cut].replace("\t", "    ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=("humaneval", "mbpp"), required=True)
    ap.add_argument("--base_url", default="http://localhost:30000/v1/completions")
    ap.add_argument("--model_name", required=True, help="tokenizer path (chat template)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_postfix", default="run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_concurrent", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt_style", choices=("chat", "plain"), default="plain",
                    help="plain: EvalPlus's instruction + open-fence response prefix as plain text, no "
                         "chat template (ResidualQuant, arXiv 2610.10381, Table 7). chat: the same "
                         "turns through the tokenizer's chat template (LoopCD / EvalPlus codegen).")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from evalplus.data import (get_human_eval_plus, get_human_eval_plus_hash,
                               get_mbpp_plus, get_mbpp_plus_hash)
    from evalplus.eval import PASS, untrusted_check
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth
    from evalplus.provider.utility import make_raw_chat_prompt
    from evalplus.sanitize import sanitize

    if a.bench == "humaneval":
        problems, hashcode, not_none = get_human_eval_plus(), get_human_eval_plus_hash(), []
    else:
        problems, hashcode, not_none = get_mbpp_plus(), get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS
    gt = get_groundtruth(problems, hashcode, not_none)
    tids = sorted(problems, key=lambda t: int(t.rsplit("/", 1)[1]))[: a.limit]

    tok = AutoTokenizer.from_pretrained(a.model_name)
    chat = a.prompt_style == "chat" and tok.chat_template is not None
    # both styles end the prompt inside an open ```python fence, so the fence close stops it
    eos = BASE_EOS + HE_EXTRA_CHAT

    def prompt_of(p):
        if chat:
            return make_raw_chat_prompt(p["prompt"], INSTRUCTION_PREFIX, RESPONSE_PREFIX, tok)
        # make_raw_chat_prompt's user / assistant contents, without the chat template
        return (f"{INSTRUCTION_PREFIX}\n```\n{p['prompt'].strip()}\n```\n\n"
                f"{RESPONSE_PREFIX}\n```python\n")

    def gen(tid):
        body = {"model": a.model_name, "prompt": prompt_of(problems[tid]), "temperature": 0.0,
                "max_tokens": a.max_tokens, "stop": eos, "seed": a.seed}
        r = requests.post(a.base_url, json=body, timeout=a.timeout)
        r.raise_for_status()
        return tid, r.json()["choices"][0]["text"]

    with ThreadPoolExecutor(a.num_concurrent) as ex:
        outs = dict(ex.map(gen, tids))

    def bench_name(tid):
        return "humaneval" if "HumanEval" in tid else "mbpp"

    rows, base_ok, plus_ok = [], 0, 0
    for tid in tids:
        p, g = problems[tid], gt[tid]
        text = trim_eos(outs[tid], eos)
        san = sanitize(text, entrypoint=p["entry_point"])
        if not san.strip():
            san = sanitize(p["prompt"] + text, entrypoint=p["entry_point"])
        b, _ = untrusted_check(bench_name(tid), san, p["base_input"], p["entry_point"],
                               g["base"], p["atol"], g["base_time"])
        q, _ = untrusted_check(bench_name(tid), san, p["plus_input"], p["entry_point"],
                               g["plus"], p["atol"], g["plus_time"])
        base_ok += b == PASS
        plus_ok += q == PASS
        rows.append({"task_id": tid, "completion": outs[tid], "sanitized": san,
                     "base": str(b), "plus": str(q)})
    n = len(tids)
    task = f"evalplus_{a.bench}"
    res = {task: {"pass@1,base": base_ok / n, "pass@1,plus": plus_ok / n, "n": n,
                  "chat_template": chat, "prompt_style": a.prompt_style}}
    os.makedirs(a.output_dir, exist_ok=True)
    stem = os.path.join(a.output_dir, f"{a.save_postfix}_{a.seed}_{task}")
    json.dump(res, open(stem + ".json", "w"))
    with open(stem + "_samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
