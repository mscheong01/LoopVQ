#!/usr/bin/env python3
"""Per-sample generated length of the reasoning runs -> outlen_<tag>__<task>.csv, for the stability figure.

    python3 dump_outlen.py ouro_1p4b_think gpqa_diamond      # in the serve env (needs transformers)

CSV columns: arm, seed, doc_id, gen_tokens, cap_hit.
Reads the lm_eval samples of every arm and seed from every experiments/work* dir that has them (a
PE_WORK per node when cells were spread over nodes) and re-tokenizes the response with
the model's tokenizer. cap_hit = the response reached the 32,768-token generation budget
(>= CAP - CAP_SLACK tokens after re-tokenization; detokenize -> retokenize moves a few tokens).
"""
from __future__ import annotations

import csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPS = HERE.parent
CAP, CAP_SLACK = 32768, 64
MODEL_DIR = {"ouro_1p4b_think": "Ouro-1.4B-Thinking", "ouro_2p6b_think": "Ouro-2.6B-Thinking"}


def main() -> int:
    tag, task = sys.argv[1], sys.argv[2]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(EXPS.parents[2] / "models" / MODEL_DIR[tag]), trust_remote_code=True)
    pat = re.compile(rf"^{tag}_(?P<arm>.+)_(?P<seed>\d)_{task}_samples\.jsonl$")
    seen, rows = set(), []
    for wd in sorted(EXPS.glob("work*")):
        rd = wd / tag / "results"
        if not rd.is_dir():
            continue
        for f in sorted(rd.iterdir()):
            m = pat.match(f.name)
            if not m or (m["arm"], m["seed"]) in seen:
                continue
            seen.add((m["arm"], m["seed"]))
            for line in f.open():
                d = json.loads(line)
                r = d["resps"][0]
                r = r[0] if isinstance(r, list) else r
                n = len(tok(r, add_special_tokens=False).input_ids)
                rows.append((m["arm"], int(m["seed"]), d["doc_id"], n, int(n >= CAP - CAP_SLACK)))
            print(f"[outlen] {wd.name}/{f.name}")
    out = HERE / f"outlen_{tag}__{task}.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "seed", "doc_id", "gen_tokens", "cap_hit"]); w.writerows(rows)
    arms = sorted({r[0] for r in rows})
    print(f"\n{out.name}: {len(rows)} samples")
    print(f"{'arm':24s} {'n':>5s} {'mean':>7s} {'p50':>7s} {'cap hit':>8s}")
    for a in arms:
        v = sorted(r[3] for r in rows if r[0] == a); h = sum(r[4] for r in rows if r[0] == a)
        print(f"{a:24s} {len(v):5d} {sum(v) / len(v):7.0f} {v[len(v) // 2]:7d} {100 * h / len(v):7.1f}%")
    near = sorted(r[3] for r in rows if r[3] >= CAP - 512)
    print(f"\nlengths >= {CAP - 512}: {len(near)}; min/max {near[0] if near else '-'} / {near[-1] if near else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
