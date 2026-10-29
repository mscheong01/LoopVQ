#!/usr/bin/env python3
"""Bit sweep: CQ vs LoopVQ on Ouro-2.6B at 4 / 2 / 1 / 0.5 / 0.25 bit per channel, x {GSM8K,
MATH500, HumanEval, MBPP}; a bf16 row, then one block per rate. Feeds figures/motivation (b).

Reads outputs/ (1 / 2 / 4 bit written by scripts/<model>__<bench>.sh via run_cell.sh; 0.25 /
0.5 bit and bf16 are read from ../loopvq_general/outputs) and writes table.tex /
table_float.tex / table.png / table.md beside it. Missing cells print "--".

Arms (bundles in experiments/artifacts/<model>_<arm>; build.sh), 8-bit codes, G = 8 / bits:
  cq8b_gG_gq            CQ: per-slot codebooks over G contiguous channels
  loopvq8b_gG_tasq      TaSQ: per slot, K = W -> N -> P (RoPE pairs atomic); V plain CQ
  loopvq8b_gG_tqlve2e   TaSQ+LoopVQ, G >= 8 only (1 / 0.5 / 0.25 bit)
  loopvq8b_gG_lvgqe2e   LoopVQ: K and V coded loop axis first, Tc = min(G, 4) steps x G/Tc
                        channels, + loop-aware fine-tuning
Both calibrated on GPQA main minus diamond, 16 x 2048 tokens, Fisher-weighted k-means.
Metrics as loopvq_general (GSM8K 3-shot flexible-extract, MATH500 Math-Verify, EvalPlus base).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = [os.path.join(HERE, "outputs"),                                  # this study's own arms
       os.path.join(os.path.dirname(HERE), "loopvq_general", "outputs")]  # arms shared with the main table
MODELS = [("ouro_2p6b", "Ouro-2.6B")]
BENCHES = ["gsm8k_cot_3shot", "math500_mv", "evalplus_humaneval", "evalplus_mbpp"]
RATES = [("4", 2), ("2", 4), ("1", 8), ("0.5", 16), ("0.25", 32)]
KINDS = [("cq8b_g{G}_gq", "CQ"), ("loopvq8b_g{G}_tasq", "TaSQ"), ("loopvq8b_g{G}_lvgqe2e", "LoopVQ"),
         ("loopvq8b_g{G}_tqlve2e", "TaSQ+LoopVQ")]
# TaSQ+LoopVQ only at <= 1 bit (G >= 8): below that c = G / 4 < 2 and TaSQ's P cannot keep a
# RoPE pair in one group
def has(kind, G):
    return kind != "TaSQ+LoopVQ" or G >= 8
# K-side pooled-head scale: one fp16 per (token, slot) shared by 16 heads x 128 channels
NORM_BITS = 16 / (16 * 128)


def arms(G):
    return [a.format(G=G) for a, n in KINDS if has(n, G)]


T.METRICS.update({
    "gsm8k_cot_3shot":    ("exact_match,flexible-extract", 100.0),
    "math500_mv":         ("acc", 100.0),
    "evalplus_humaneval": ("pass@1,base", 100.0),
    "evalplus_mbpp":      ("pass@1,base", 100.0),
})
T.PRETTY_BENCH.update({"gsm8k_cot_3shot": "GSM8K", "math500_mv": "MATH500",
                       "evalplus_humaneval": "HumanEval", "evalplus_mbpp": "MBPP"})
T.METHOD_ORDER = ["bf16"] + [a for _, G in RATES for a in arms(G)]
# Paper styling: ours named in bold (only the LoopVQ part of a combined name), TaSQ+LoopVQ rows
# shaded. Previews (md / png) strip the LaTeX via T.plain.
TEX_NAME = {"LoopVQ": r"\textbf{LoopVQ}", "TaSQ+LoopVQ": r"TaSQ+\textbf{LoopVQ}"}
SHADED = {"TaSQ+LoopVQ"}
T.BITS_FMT = T.compact_bits
T.OURS_METHODS = set()
for rate, G in RATES:
    for (a, name) in KINDS:
        arm = a.format(G=G)
        T.PRETTY_METHOD[arm] = TEX_NAME.get(name, name)
        if name in SHADED:
            T.OURS_METHODS.add(arm)
        T.METHOD_BITS[arm] = (8 / G + (NORM_BITS if name.startswith("TaSQ") else 0.0), 8 / G)
for model, pretty in MODELS:
    T.PRETTY_MODEL[f"{model}@bf16"] = pretty
    for rate, _ in RATES:
        T.PRETTY_MODEL[f"{model}@{rate}"] = f"{rate} bit"


def build():
    blocks = []
    for model, _ in MODELS:
        for label, methods in [("bf16", ["bf16"])] + [(r, arms(G)) for r, G in RATES]:
            d = T.table_data(OUT, [model], BENCHES, methods=methods, average=True)
            for blk in d["blocks"]:
                blk["model"] = f"{model}@{label}"
            blocks += d["blocks"]
    d["blocks"] = blocks
    body = T.to_latex_body_grouped(d, show_std=False)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(HERE, "table_float.tex"), "w") as f:
        f.write(T.to_latex_float("table.tex", CAPTION, "tab:bit_sweep", body=body, star=True) + "\n")
    try:
        T.to_png(d, os.path.join(HERE, "table.png"), T.plain_caption(CAPTION), show_std=False)
    except ImportError as e:
        print(f"[table] NOTE no PNG preview ({e})")
    md = to_markdown(d)
    with open(os.path.join(HERE, "table.md"), "w") as f:
        f.write(md + "\n")
    return md


def to_markdown(d):
    heads = [T.PRETTY_BENCH[c] for c in d["cols"]] + ["Avg."]
    lines = ["| bits | method | K/V bits | " + " | ".join(heads) + " |",
             "|---" * (3 + len(heads)) + "|"]
    for blk in d["blocks"]:
        for i, (m, vals) in enumerate(blk["rows"]):
            cs = ["--" if v is None else (f"**{v[0]:.1f}**" if m in blk["best"][j] and m != "bf16"
                                          else f"{v[0]:.1f}") for j, v in enumerate(vals)]
            a = blk["avg"][i]
            cs.append("--" if a is None else f"{a[0]:.1f}")
            lines.append(f"| {T.PRETTY_MODEL[blk['model']] if i == 0 else ''} | "
                         f"{T.plain(T.PRETTY_METHOD.get(m, m))} | {T.bits_cell(m, blk['model'])} | " + " | ".join(cs) + " |")
    return "\n".join(lines)


CAPTION = (r"Accuracy (\%) of Ouro-2.6B from 4 to 0.25 bits per KV coordinate (8-bit codes, "
           r"$G{=}2,\dots,32$). TaSQ+LoopVQ is evaluated at $\le$1 bit, where each loop step "
           r"holds at least one RoPE pair. Benchmarks as in Table~\ref{tab:loopvq_general}. "
           r"Best quantized result per block in bold.")

if __name__ == "__main__":
    print(build())
