#!/usr/bin/env python3
"""loopvq_ablation on Ouro-1.4B (GSM8K 3-shot flexible-extract, HumanEval base pass@1): the
component ablation (each LoopVQ component alone on CQ and on TaSQ, 0.5 / 0.25 bit) and the
calibration-set sensitivity (all four methods refit on WikiText-2 / C4 / SlimPajama, 0.25 bit),
and the calibration-set size (LoopVQ / TaSQ+LoopVQ on 8 / 16 / 32 GPQA sequences, 0.25 bit).

Reads outputs/ (new arms written by scripts/<model>__<bench>.sh via run_cell.sh; arms shared with
the main table are read from ../loopvq_general/outputs) and writes table.md. Arms: build.sh.
Missing cells print "--". The paper figure is figures/ablation (reads the same outputs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import table_lib as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = [os.path.join(HERE, "outputs"),                                  # this study's own arms
       os.path.join(os.path.dirname(HERE), "loopvq_general", "outputs")]  # arms shared with the main table
MODEL = "ouro_1p4b"
MODELS = [(MODEL, "Ouro-1.4B")]
BENCHES = ["gsm8k_cot_3shot", "evalplus_humaneval"]
RATES = [("0.5", 16), ("0.25", 32)]
# (base, +loop coupling, +loop-aware FT, +both); FT-only arms start from the Tc = 1 sources
COMPONENTS = [("CQ", "cq8b_g{G}_gq", "loopvq8b_g{G}_lvgq", "loopvq8b_g{G}_cq1e2e", "loopvq8b_g{G}_lvgqe2e"),
              ("TaSQ", "loopvq8b_g{G}_tasq", "loopvq8b_g{G}_tqlv", "loopvq8b_g{G}_tasq1e2e", "loopvq8b_g{G}_tqlve2e")]
SOURCES = [("CQ (Tc=1 source)", "loopvq8b_g{G}_cq1"), ("TaSQ (Tc=1 source)", "loopvq8b_g{G}_tasq1")]
CORPORA = [("GPQA", None), ("WikiText-2", "wt"), ("C4", "c4"), ("SlimPajama", "sp")]
CAL_METHODS = [("CQ", "cq8b_g32_gq", "cq8b_g32_{t}"),
               ("TaSQ", "loopvq8b_g32_tasq", "loopvq8b_g32_tasq_{t}"),
               ("LoopVQ", "loopvq8b_g32_lvgqe2e", "loopvq8b_g32_lv_{t}e2e"),
               ("TaSQ+LoopVQ", "loopvq8b_g32_tqlve2e", "loopvq8b_g32_tqlv_{t}e2e")]
# calibration-set size (GPQA): sequences of 2048 tokens; 16 is the main table
SIZES = [8, 16, 32]
SIZE_METHODS = [("LoopVQ", "loopvq8b_g32_lvgqe2e", "loopvq8b_g32_lv_n{n}e2e"),
                ("TaSQ+LoopVQ", "loopvq8b_g32_tqlve2e", "loopvq8b_g32_tqlv_n{n}e2e")]
# FT objective (LoopVQ, TaSQ+LoopVQ): k-means, per-step FT objective, loop-aware (joint) FT
FTOBJ = [("LoopVQ", "loopvq8b_g{G}_lvgq", "loopvq8b_g{G}_lvgqps", "loopvq8b_g{G}_lvgqe2e"),
         ("TaSQ+LoopVQ", "loopvq8b_g{G}_tqlv", "loopvq8b_g{G}_tqlvps", "loopvq8b_g{G}_tqlve2e")]
NORM_BITS = 16 / (16 * 128)

T.METRICS.update({"gsm8k_cot_3shot": ("exact_match,flexible-extract", 100.0),
                  "evalplus_humaneval": ("pass@1,base", 100.0)})
T.PRETTY_BENCH.update({"gsm8k_cot_3shot": "GSM8K", "evalplus_humaneval": "HumanEval"})
T.PRETTY_MODEL[f"{MODEL}@bf16"] = "Ouro-1.4B"
T.METHOD_ORDER = ["bf16"]


def reg(arm, name, G):
    T.PRETTY_METHOD[arm] = name
    T.METHOD_BITS[arm] = (8 / G + (NORM_BITS if name.startswith("TaSQ") else 0.0), 8 / G)
    if arm not in T.METHOD_ORDER:
        T.METHOD_ORDER.append(arm)


def comp_arms(G):
    out = []
    for base, *arms in COMPONENTS:
        for tag, a in zip(("", " +coupling", " +FT", " +both"), arms):
            out.append((f"{base}{tag}", a.format(G=G)))
    return out + [(n, a.format(G=G)) for n, a in SOURCES]


def cal_arm(gpqa, pat, t):
    return gpqa if t is None else pat.format(t=t)


for _, G in RATES:
    for name, arm in comp_arms(G):
        reg(arm, name, G)
for name, gpqa, pat in CAL_METHODS:
    for cname, t in CORPORA:
        reg(cal_arm(gpqa, pat, t), f"{name} [{cname}]", 32)


for _, G in RATES:
    for name, *arms in FTOBJ:
        for tag, a in zip((" [k-means]", " [per-step FT]", " [loop-aware FT]"), arms):
            reg(a.format(G=G), name + tag, G)


def size_arm(main, pat, n):
    return main if n == 16 else pat.format(n=n)


for name, main, pat in SIZE_METHODS:
    for n in SIZES:
        reg(size_arm(main, pat, n), f"{name} [GPQA x{n}]", 32)


def build():
    cells = T.load(OUT)

    def sc(arm, bench):
        a = T.agg(cells.get((MODEL, bench, arm), []), bench)
        return "--" if a is None else f"{a[0]:.1f}"

    heads = [T.PRETTY_BENCH[b] for b in BENCHES]
    ref = " | ".join(sc("bf16", b) for b in BENCHES)
    L = ["## Component ablation (Ouro-1.4B)", "",
         "| bits | method | " + " | ".join(heads) + " |", "|---" * (2 + len(heads)) + "|",
         f"| 16 | BF16 | {ref} |"]
    for rate, G in RATES:
        for i, (name, arm) in enumerate(comp_arms(G)):
            L.append(f"| {rate if i == 0 else ''} | {name} | " + " | ".join(sc(arm, b) for b in BENCHES) + " |")
    L += ["", "## Calibration set (Ouro-1.4B, 0.25 bit)", ""]
    for b in BENCHES:
        L += [f"**{T.PRETTY_BENCH[b]}** (bf16 {sc('bf16', b)})", "",
              "| method | " + " | ".join(c for c, _ in CORPORA) + " |", "|---" * (1 + len(CORPORA)) + "|"]
        for name, gpqa, pat in CAL_METHODS:
            L.append(f"| {name} | " + " | ".join(sc(cal_arm(gpqa, pat, t), b) for _, t in CORPORA) + " |")
        L.append("")
    L += ["## Calibration-set size (Ouro-1.4B, GPQA, 0.25 bit)", "",
          "| method | bench | " + " | ".join(f"{n} seq" for n in SIZES) + " |", "|---" * (2 + len(SIZES)) + "|"]
    for name, main, pat in SIZE_METHODS:
        for b in BENCHES:
            L.append(f"| {name} | {T.PRETTY_BENCH[b]} | " + " | ".join(sc(size_arm(main, pat, n), b) for n in SIZES) + " |")
    L += ["", "## FT objective (Ouro-1.4B, GPQA)", "",
          "| bits | method | bench | k-means | per-step FT | loop-aware FT |", "|---" * 6 + "|"]
    for rate, G in RATES:
        for name, *arms in FTOBJ:
            for b in BENCHES:
                L.append(f"| {rate} | {name} | {T.PRETTY_BENCH[b]} | "
                         + " | ".join(sc(a.format(G=G), b) for a in arms) + " |")
    md = "\n".join(L)
    with open(os.path.join(HERE, "table.md"), "w") as f:
        f.write(md + "\n")
    return md


if __name__ == "__main__":
    print(build())
