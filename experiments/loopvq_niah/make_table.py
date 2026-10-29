#!/usr/bin/env python3
"""loopvq NIAH table: Ouro-1.4B and Ouro-2.6B (base models, no thinking), RULER NIAH accuracy versus
context length; per model a bf16 row, then one block per KV bit rate (0.5 and 0.25 bit per channel).

Grid and scoring as in RULER NIAH: 4k-64k, 3 seeds
(needle placement / haystack sampling), each cell the mean over the eight RULER-NIAH subtasks
(200 rows), 64 FP sink + 256 recent tokens. Reads outputs/ (written by scripts/ouro_{1p4b,2p6b}.sh) and
writes table.tex / table_float.tex / table.png / table.md beside it. Missing cells print "--".

Arms are the loopvq_general ones (bundles in experiments/artifacts/ouro_{1p4b,2p6b}_<arm>):
  cq8b_g{16,32}_gq, loopvq8b_g{16,32}_tasq, loopvq8b_g{16,32}_lvgqe2e, loopvq8b_g{16,32}_tqlve2e
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MODELS = [("ouro_1p4b", "Ouro-1.4B"), ("ouro_2p6b", "Ouro-2.6B")]
CTX = [4096, 8192, 16384, 32768, 65536]     # 65536 = Ouro's max_position_embeddings
BENCHES = [f"niah@{c}" for c in CTX]
for b, c in zip(BENCHES, CTX):
    T.METRICS[b] = ("exact_match,none", 100.0)
    T.PRETTY_BENCH[b] = f"{c // 1024}k"

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

CAPTION = (r"RULER NIAH accuracy (\%) of Ouro-1.4B and Ouro-2.6B versus context length under 0.5- and 0.25-bit KV "
           r"cache quantization (8-bit codes, $G{=}16$ and $32$). Each cell averages the eight "
           r"RULER-NIAH subtasks over 3 seeds; 64 sink and 256 recent tokens kept in full precision. "
           r"Best quantized result per block in bold.")


def build():
    blocks, d = [], None
    for model, _ in MODELS:
        for label, methods in [("bf16", ["bf16"])] + [(r, [a.format(G=G) for a, _ in KINDS]) for r, G in RATES]:
            d = T.table_data(OUT, [model], BENCHES, expect_seeds=3, methods=methods, average=True)
            for blk in d["blocks"]:
                blk["model"] = f"{model}@{label}"
            blocks += d["blocks"]
    d["blocks"] = blocks
    body = T.to_latex_body_grouped(d, show_std=False)
    with open(os.path.join(HERE, "table.tex"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(HERE, "table_float.tex"), "w") as f:
        f.write(T.to_latex_float("table.tex", CAPTION, "tab:loopvq_niah", body=body, star=True) + "\n")
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
                    cs.append("--"); continue
                s = f"{v[0]:.1f} ± {v[1]:.1f}" if len(v) > 1 and v[1] is not None else f"{v[0]:.1f}"
                cs.append(f"**{s}**" if m in blk["best"][j] and m != "bf16" else s)
            a = blk["avg"][i]
            cs.append("--" if a is None else f"{a[0]:.1f}")
            lines.append(f"| {T.PRETTY_MODEL[blk['model']] if i == 0 else ''} | "
                         f"{T.plain(T.PRETTY_METHOD.get(m, m))} | {T.bits_cell(m, blk['model'])} | "
                         + " | ".join(cs) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build())
