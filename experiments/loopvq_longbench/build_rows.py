#!/usr/bin/env python3
"""LongBench (v1) rows for the loopvq_longbench table -- official protocol (THUDM/LongBench pred.py):

- the 16 English tasks, all samples (3750), from the dataset's data.zip (data_raw/data/<task>.jsonl);
- per-task prompt and generation length from the official dataset2prompt / dataset2maxlen (lb/);
- the filled prompt is tokenized with the serving model's tokenizer and, when longer than
  --max-len, truncated in the middle (first max_len/2 + last max_len/2 tokens);
- the chat template is applied except for trec, triviaqa, samsum, lcc and repobench-p, which pred.py
  feeds raw ("chat models are better off without build prompts on these tasks");
- samsum stops at the first newline after at least one token (pred.py: eos_token_id += "\\n",
  min_length = context + 1).
Scoring: scripts/emit_lbv1.py (official metrics, lb/longbench_metrics.py).

  python build_rows.py --model <hf-dir> --out rows.jsonl [--max-len 31500]
"""
import argparse
import json
import os

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
         "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
         "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=31500)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    d2p = json.load(open(os.path.join(HERE, "lb/dataset2prompt.json")))
    d2m = json.load(open(os.path.join(HERE, "lb/dataset2maxlen.json")))
    n = n_trunc = 0
    with open(args.out, "w") as f:
        for task in TASKS:
            for i, line in enumerate(open(os.path.join(HERE, f"data_raw/data/{task}.jsonl"))):
                obj = json.loads(line)
                prompt = d2p[task].format(**obj)
                ids = tok(prompt, truncation=False).input_ids
                if len(ids) > args.max_len:
                    n_trunc += 1
                    half = args.max_len // 2
                    prompt = tok.decode(ids[:half], skip_special_tokens=True) + \
                        tok.decode(ids[-half:], skip_special_tokens=True)
                if task not in NO_CHAT:
                    prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=False, add_generation_prompt=True)
                row = {"_id": f"{task}_{i}", "task": task, "prompt": prompt, "answers": obj["answers"],
                       "all_classes": obj["all_classes"], "length": obj["length"],
                       "n_tokens": len(tok(prompt).input_ids), "max_new_tokens": d2m[task]}
                if task == "samsum":
                    row["sampling_extra"] = {"stop": ["\n"], "min_new_tokens": 1}
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"wrote {n} rows to {args.out}; {n_trunc} truncated to {args.max_len} tokens")


if __name__ == "__main__":
    main()
