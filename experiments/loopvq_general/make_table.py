#!/usr/bin/env python3
"""loopvq general-benchmark table: Ouro-1.4B, Ouro-2.6B and Huginn-3.5B, methods x {GSM8K, MATH500, HumanEval,
MBPP}; per model a bf16 row, then one block per KV bit rate (0.5 and 0.25 bit per channel).

Reads outputs/ (written by scripts/<model>__<bench>.sh via run_cell.sh) and writes, beside it:
table.tex / table_float.tex / table.png (table_lib) and table.md (quick view). Missing cells
print "--", so it can be rebuilt at any point of the sweep (run_all.sh does after every cell).

Arms (scripts/run_sweep.sh arm_spec, bundles in experiments/artifacts/<model>_<arm>), all
calibrated on GPQA main minus diamond, 16 x 2048 tokens, Fisher-weighted k-means (CQ Eq.6):
  cq8b_g{16,32}_gq            CQ: per slot (loop step x layer) codebooks, G contiguous channels
  loopvq8b_g{16,32}_tasq      TaSQ: per slot, K = W (query-guided weight) -> N (pooled-head
                              norm) -> P (covariance grouping, RoPE pairs atomic); V plain CQ
  loopvq8b_g{16,32}_lvgqe2e   loopvq: K and V coded across the 4 loop steps (c = G/4 channels
                              x 4 steps per code) + loop-aware end-to-end codebook fine-tuning
  loopvq8b_g{16,32}_tqlve2e   TaSQ + loopvq: TaSQ's W -> N -> P on K, then loopvq + fine-tuning
abits 8 throughout: G = 16 -> 0.5 bit, G = 32 -> 0.25 bit per channel. Benchmarks follow the
ResidualQuant protocol (GSM8K 3-shot CoT flexible-extract, MATH500 0-shot Math-Verify,
EvalPlus HumanEval / MBPP base pass@1).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MODELS = [("ouro_1p4b", "Ouro-1.4B"), ("ouro_2p6b", "Ouro-2.6B"), ("huginn_3p5b", "Huginn-3.5B")]
# Huginn-0125 at r = 32 recurrences (132 KV slots; models/huginn.py). Its LoopVQ rows code Tc = 4 of
# the 32 loop steps together (c = G/4 channels, as on Ouro), not Tc = G: Tc = G leaves c = 1, whose
# decode reads G x the index traffic of CQ, and cannot hold TaSQ's RoPE pairs.
BENCHES = ["gsm8k_cot_3shot", "math500_mv", "evalplus_humaneval", "evalplus_mbpp"]
RATES = [("0.5", 16), ("0.25", 32)]
KINDS = [("cq8b_g{G}_gq", "CQ"), ("loopvq8b_g{G}_tasq", "TaSQ"),
         ("loopvq8b_g{G}_lvgqe2e", "LoopVQ"), ("loopvq8b_g{G}_tqlve2e", "TaSQ+LoopVQ")]
# K-side pooled-head scale: one fp16 per (token, slot) shared by 16 heads x 128 channels
NORM_BITS = 16 / (16 * 128)
K_NORM = {"TaSQ", "TaSQ+LoopVQ"}


def arms(G):
    return [a.format(G=G) for a, _ in KINDS]


T.METRICS.update({
    "gsm8k_cot_3shot":    ("exact_match,flexible-extract", 100.0),
    "math500_mv":         ("acc", 100.0),
    "evalplus_humaneval": ("pass@1,base", 100.0),
    "evalplus_mbpp":      ("pass@1,base", 100.0),
})
T.PRETTY_BENCH.update({"gsm8k_cot_3shot": "GSM8K", "math500_mv": "MATH500",
                       "evalplus_humaneval": "HumanEval", "evalplus_mbpp": "MBPP"})
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


# Per-model arm overrides: Huginn's LoopVQ rows are the Tc = 4 bundles (see MODELS).
ARM_OVERRIDE = {"huginn_3p5b": {"LoopVQ": "loopvq8b_g{G}_lvgqt4e2e",
                                "TaSQ+LoopVQ": "loopvq8b_g{G}_tqlvt4e2e"}}
# Huginn's pooled-head K scale: one fp16 per (token, slot) shared by 55 heads x 96 channels
HUGINN_NORM_BITS = 16 / (55 * 96)
for rate, G in RATES:
    b = 8 / G
    for name, a in ARM_OVERRIDE["huginn_3p5b"].items():
        T.PRETTY_METHOD[a.format(G=G)] = TEX_NAME.get(name, name)
        if name in SHADED:
            T.OURS_METHODS.add(a.format(G=G))
        T.METHOD_BITS[a.format(G=G)] = (b + (HUGINN_NORM_BITS if name in K_NORM else 0.0), b)
    for name in K_NORM:
        arm = dict(ARM_OVERRIDE["huginn_3p5b"], TaSQ="loopvq8b_g{G}_tasq")[name].format(G=G)
        T.MODEL_METHOD_BITS[(f"huginn_3p5b@{rate}", arm)] = (b + HUGINN_NORM_BITS, b)


def model_arms(model, G):
    over = ARM_OVERRIDE.get(model, {})
    return [over.get(n, a).format(G=G) for a, n in KINDS]


def build():
    blocks = []
    for model, _ in MODELS:
        for label, methods in [("bf16", ["bf16"])] + [(r, model_arms(model, G)) for r, G in RATES]:
            d = T.table_data(OUT, [model], BENCHES, methods=methods, average=True)
            for blk in d["blocks"]:
                blk["model"] = f"{model}@{label}"
            blocks += d["blocks"]
    d["blocks"] = blocks
    body = T.to_latex_body_grouped(d, show_std=False)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(HERE, "table_float.tex"), "w") as f:
        f.write(T.to_latex_float("table.tex", CAPTION, "tab:loopvq_general", body=body, star=True) + "\n")
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


CAPTION = (r"Accuracy (\%) of Ouro-1.4B, Ouro-2.6B, and Huginn-3.5B under 0.5- and 0.25-bit KV cache "
           r"quantization (8-bit codes, $G{=}16$ and $32$). GSM8K 3-shot CoT, MATH500 0-shot, and "
           r"EvalPlus HumanEval / MBPP base pass@1. On Huginn-3.5B, LoopVQ couples $T_c{=}4$ of the "
           r"32 recurrences. Best quantized result per block in bold.")

if __name__ == "__main__":
    print(build())
