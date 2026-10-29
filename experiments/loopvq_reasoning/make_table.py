#!/usr/bin/env python3
"""loopvq reasoning table: Ouro-1.4B-Thinking and Ouro-2.6B-Thinking, thinking on, methods x
{AIME'24, GPQA-Diamond}; per model a bf16 row, then one block per KV bit rate (0.5 and 0.25 bit per channel).

Same arms as loopvq_general (8-bit codes, G = 16 / 32), with the reasoning window (64 FP sink + 256 recent
tokens) and the bundles refit on the Thinking weights (build.sh). Decoding follows the
Ouro-Thinking model card: temperature 1.0, top_p 0.7, 32k generation budget. Three sampling
seeds per cell, folded to mean+-std.

Reads outputs/ (written by scripts/<model>__<bench>.sh via run_cell.sh) and writes, beside it,
table.tex / table_float.tex / table.png (table_lib) and table.md. Missing cells print "--".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MODELS = [("ouro_1p4b_think", "Ouro-1.4B-Thinking"), ("ouro_2p6b_think", "Ouro-2.6B-Thinking")]
BENCHES = ["aime24", "aime25", "gpqa_diamond", "evalplus_humaneval", "evalplus_mbpp"]
RATES = [("0.5", 16), ("0.25", 32)]
KINDS = [("cq8b_g{G}_gq", "CQ"), ("loopvq8b_g{G}_tasq", "TaSQ"),
         ("loopvq8b_g{G}_lvgqe2e", "LoopVQ"), ("loopvq8b_g{G}_tqlve2e", "TaSQ+LoopVQ")]
# K-side pooled-head scale: one fp16 per (token, slot) shared by 16 heads x 128 channels
NORM_BITS = 16 / (16 * 128)
K_NORM = {"TaSQ", "TaSQ+LoopVQ"}
SEEDS = 3


def arms(G):
    return [a.format(G=G) for a, _ in KINDS]


# HumanEval in thinking mode (scripts/run_evalplus_reasoning_api.py): EvalPlus HumanEval+ (plus suite) pass@1
T.METRICS["evalplus_humaneval"] = ("pass@1,plus", 100.0)
T.PRETTY_BENCH["evalplus_humaneval"] = "HumanEval+"
T.METRICS["evalplus_mbpp"] = ("pass@1,plus", 100.0)
T.PRETTY_BENCH["evalplus_mbpp"] = "MBPP+"
T.METHOD_ORDER = ["bf16"] + [a for _, G in RATES for a in arms(G)]
# Paper styling: ours named in bold (only the LoopVQ part of a combined name), LoopVQ and TaSQ+LoopVQ rows
# shaded. Previews (md / png) strip the LaTeX via T.plain.
TEX_NAME = {"LoopVQ": r"\textbf{LoopVQ}", "TaSQ+LoopVQ": r"TaSQ+\textbf{LoopVQ}"}
SHADED = {"LoopVQ", "TaSQ+LoopVQ"}
T.BITS_FMT = T.compact_bits
T.OURS_METHODS = set()
for rate, G in RATES:
    b = 8 / G
    for (a, name) in KINDS:
        arm = a.format(G=G)
        T.PRETTY_METHOD[arm] = TEX_NAME.get(name, name)
        if name in SHADED:
            T.OURS_METHODS.add(arm)
        T.METHOD_BITS[arm] = (b + (NORM_BITS if name in K_NORM else 0.0), b)
for model, pretty in MODELS:
    T.PRETTY_MODEL[f"{model}@bf16"] = pretty
    for rate, _ in RATES:
        T.PRETTY_MODEL[f"{model}@{rate}"] = f"{rate} bit"


def build():
    blocks = []
    for model, _ in MODELS:
        for label, methods in [("bf16", ["bf16"])] + [(r, arms(G)) for r, G in RATES]:
            d = T.table_data(OUT, [model], BENCHES, expect_seeds=SEEDS, methods=methods, average=True)
            for blk in d["blocks"]:
                blk["model"] = f"{model}@{label}"
            blocks += d["blocks"]
    d["blocks"] = blocks
    body = T.to_latex_body_grouped(d, show_std=True)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(HERE, "table_float.tex"), "w") as f:
        f.write(T.to_latex_float("table.tex", CAPTION, "tab:loopvq_reasoning", body=body, star=True) + "\n")
    try:
        T.to_png(d, os.path.join(HERE, "table.png"), T.plain_caption(CAPTION), show_std=True)
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
            cs = []
            for j, v in enumerate(vals):
                if v is None:
                    cs.append("--")
                    continue
                s = f"{v[0]:.1f} ± {v[1]:.1f}" + ("" if v[2] == SEEDS else f" (n={v[2]})")
                cs.append(f"**{s}**" if m in blk["best"][j] and m != "bf16" else s)
            a = blk["avg"][i]
            cs.append("--" if a is None else f"{a[0]:.1f} ± {a[1]:.1f}")
            lines.append(f"| {T.PRETTY_MODEL[blk['model']] if i == 0 else ''} | "
                         f"{T.plain(T.PRETTY_METHOD.get(m, m))} | {T.bits_cell(m, blk['model'])} | " + " | ".join(cs) + " |")
    return "\n".join(lines)


CAPTION = (r"Accuracy (\%) of Ouro-1.4B-Thinking and Ouro-2.6B-Thinking with thinking enabled, under "
           r"0.5- and 0.25-bit KV cache quantization (8-bit codes, $G{=}16$ and $32$). AIME'24, AIME'25, "
           r"GPQA-Diamond, HumanEval+, and MBPP+, mean$\pm$std over 3 sampling seeds (temperature 1.0, top-$p$ 0.7, 32k-token "
           r"budget; 64 sink and 256 recent tokens kept in full precision). Best quantized result per "
           r"block in bold.")

if __name__ == "__main__":
    print(build())
