#!/usr/bin/env python3
"""LongBench (v1) table: Ouro-1.4B (base model, no thinking), the 16 English LongBench tasks
(one column each, standard short names) plus Avg (unweighted task mean, LongBench's own average);
a bf16 row, then one block per KV bit rate (0.5 and 0.25 bit per channel).

Protocol: official LongBench pred.py / eval.py (build_rows.py, scripts/emit_lbv1.py): all 3750
samples, per-task prompts and generation lengths, prompts middle-truncated to 64000 tokens, greedy,
official metrics. 64 FP sink + 256 recent tokens. Reads outputs/ and writes table.tex /
table_float.tex / table.png / table.md. Missing cells print "--".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MODELS = [("ouro_1p4b", "Ouro-1.4B"), ("ouro_2p6b", "Ouro-2.6B")]
TASK_COLS = [("narrativeqa", "NrtvQA"), ("qasper", "Qasper"), ("multifieldqa_en", "MF-en"),
             ("hotpotqa", "HotpotQA"), ("2wikimqa", "2WikiMQA"), ("musique", "Musique"),
             ("gov_report", "GovReport"), ("qmsum", "QMSum"), ("multi_news", "MultiNews"),
             ("trec", "TREC"), ("triviaqa", "TriviaQA"), ("samsum", "SAMSum"),
             ("passage_count", "PCount"), ("passage_retrieval_en", "PRe"),
             ("lcc", "Lcc"), ("repobench-p", "RB-P"), ("avg", "Avg.")]
for key, pretty in TASK_COLS:
    T.METRICS[f"lbv1@{key}"] = ("score,none", 100.0)
    T.PRETTY_BENCH[f"lbv1@{key}"] = pretty

RATES = [("0.5", 16), ("0.25", 32)]
KINDS = [("cq8b_g{G}_gq", "CQ"), ("loopvq8b_g{G}_tasq", "TaSQ"),
         ("loopvq8b_g{G}_lvgqe2e", "LoopVQ"), ("loopvq8b_g{G}_tqlve2e", "TaSQ+LoopVQ")]
NORM_BITS = 16 / (16 * 128)         # TaSQ's pooled-head K scale, one fp16 per (token, slot)
K_NORM = {"TaSQ", "TaSQ+LoopVQ"}
TEX_NAME = {"LoopVQ": r"\textbf{LoopVQ}", "TaSQ+LoopVQ": r"TaSQ+\textbf{LoopVQ}"}
SHADED = {"LoopVQ", "TaSQ+LoopVQ"}

T.METHOD_ORDER = ["bf16"] + [a.format(G=G) for _, G in RATES for a, _ in KINDS]
T.BITS_FMT = T.compact_bits
T.OURS_METHODS = set()
for rate, G in RATES:
    b = 8 / G
    for a, name in KINDS:
        arm = a.format(G=G)
        T.PRETTY_METHOD[arm] = TEX_NAME.get(name, name)
        if name in SHADED:
            T.OURS_METHODS.add(arm)
        T.METHOD_BITS[arm] = (b + (NORM_BITS if name in K_NORM else 0.0), b)
for model, pretty in MODELS:
    T.PRETTY_MODEL[f"{model}@bf16"] = pretty
    for rate, _ in RATES:
        T.PRETTY_MODEL[f"{model}@{rate}"] = f"{rate} bit"

CAPTION = (r"LongBench results of Ouro-1.4B and Ouro-2.6B under 0.5- and 0.25-bit KV cache quantization (8-bit codes, "
           r"$G{=}16$ and $32$) on the 16 English tasks (official metrics; prompts truncated to 64k tokens). "
           r"64 sink and 256 recent tokens kept in full precision. Best quantized result per block in bold.")


def grid(cols):
    blocks, d = [], None
    for model, _ in MODELS:
        for label, methods in [("bf16", ["bf16"])] + [(r, [a.format(G=G) for a, _ in KINDS]) for r, G in RATES]:
            d = T.table_data(OUT, [model], [f"lbv1@{k}" for k, _ in cols], expect_seeds=1, methods=methods)
            for blk in d["blocks"]:
                blk["model"] = f"{model}@{label}"
            blocks += d["blocks"]
    d["blocks"] = blocks
    return d


def to_markdown(d):
    heads = [T.PRETTY_BENCH[c] for c in d["cols"]]
    lines = ["| bits | method | K/V bits | " + " | ".join(heads) + " |", "|---" * (3 + len(heads)) + "|"]
    for blk in d["blocks"]:
        for i, (m, vals) in enumerate(blk["rows"]):
            cs = []
            for j, v in enumerate(vals):
                if v is None:
                    cs.append("--"); continue
                s = f"{v[0]:.2f}"
                cs.append(f"**{s}**" if m in blk["best"][j] and m != "bf16" else s)
            lines.append(f"| {T.PRETTY_MODEL[blk['model']] if i == 0 else ''} | "
                         f"{T.plain(T.PRETTY_METHOD.get(m, m))} | {T.bits_cell(m, blk['model'])} | "
                         + " | ".join(cs) + " |")
    return "\n".join(lines)


def build():
    d = grid(TASK_COLS)
    body = T.to_latex_body_grouped(d, show_std=False)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(HERE, "table_float.tex"), "w") as f:
        f.write(T.to_latex_float("table.tex", CAPTION, "tab:loopvq_longbench", body=body, star=True) + "\n")
    try:
        T.to_png(d, os.path.join(HERE, "table.png"), T.plain_caption(CAPTION), show_std=False)
    except ImportError as e:
        print(f"[table] NOTE no PNG preview ({e})")
    md = to_markdown(d)
    with open(os.path.join(HERE, "table.md"), "w") as f:
        f.write(md + "\n")
    return md


if __name__ == "__main__":
    print(build())
