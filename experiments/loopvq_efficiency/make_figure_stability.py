#!/usr/bin/env python3
"""Reasoning-stability figure: per-problem GPQA-Diamond output length, cap hits boxed.

    python3 make_figure_stability.py [--rate 0.5|0.25] [--arms bf16,cq,tasq,lvq,tqlv]
                                                           [--models 1p4b,2p6b] [--task gpqa_diamond]
    -> figure_stability_<task>_<rate>.pdf / .png

For each arm, the share of problems (x 3 seeds) whose generation length falls in each bin, with the cap-hit share closing the
curve at the 32,768-token budget. One panel per model. Data: outlen_<tag>__<task>.csv from
dump_outlen.py (serve env); plotting needs matplotlib.
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
CAP = 32768
MODELS = {"1p4b": ("ouro_1p4b_think", "Ouro-1.4B-Thinking"), "2p6b": ("ouro_2p6b_think", "Ouro-2.6B-Thinking")}
# arm key -> (label, colour, bundle name at 0.5 b, at 0.25 b)
ARMS = {
    "bf16": ("BF16", "#4C4C4C", "bf16", "bf16"),
    "cq":   ("CQ", "#D1495B", "cq8b_g16_gq", "cq8b_g32_gq"),
    "tasq": ("TaSQ", "#E8A33D", "loopvq8b_g16_tasq", "loopvq8b_g32_tasq"),
    "lvq":  ("LoopVQ", "#2E6FBA", "loopvq8b_g16_lvgqe2e", "loopvq8b_g32_lvgqe2e"),
    "tqlv": ("TaSQ+LoopVQ", "#0F7B6C", "loopvq8b_g16_tqlve2e", "loopvq8b_g32_tqlve2e"),
}


def load(tag: str, task: str) -> dict[str, list[tuple[int, int]]]:
    p = HERE / f"outlen_{tag}__{task}.csv"
    if not p.exists():
        raise SystemExit(f"missing {p}: run dump_outlen.py {tag} {task} in the serve env first")
    out: dict[str, list[tuple[int, int]]] = {}
    for r in csv.DictReader(p.open()):
        out.setdefault(r["arm"], []).append((int(r["gen_tokens"]), int(r["cap_hit"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="0.5", choices=("0.5", "0.25"))
    ap.add_argument("--arms", default="bf16,cq,tasq,lvq,tqlv")
    ap.add_argument("--models", default="1p4b,2p6b")
    ap.add_argument("--task", default="gpqa_diamond")
    ap.add_argument("--binw", type=int, default=2048)
    a = ap.parse_args()
    arms = a.arms.split(","); models = a.models.split(",")
    col = 2 if a.rate == "0.5" else 3
    nbin = CAP // a.binw
    centers = [(j + 0.5) * a.binw for j in range(nbin)]

    fig, axes = plt.subplots(1, len(models), figsize=(6.6 * len(models), 4.2), squeeze=False)
    for ax, mk in zip(axes[0], models):
        tag, title = MODELS[mk]
        data = load(tag, a.task)
        ax.grid(True, ls=":", lw=0.7, c="#BBBBBB", alpha=0.8)
        ax.axvspan(CAP - a.binw, CAP + a.binw, color="#B00020", alpha=0.07, zorder=1)
        ymax = 0.0
        labels = []
        for k in arms:
            lbl, c = ARMS[k][0], ARMS[k][1]
            vals = data.get(ARMS[k][col], [])
            if not vals:
                print(f"[fig] {tag}: no data for {ARMS[k][col]}"); continue
            n = len(vals)
            counts = [0] * nbin
            for v, hit in vals:
                if not hit:
                    counts[min(v // a.binw, nbin - 1)] += 1
            pct = [100 * x / n for x in counts]
            rate = 100 * sum(h for _, h in vals) / n
            mean = sum(v for v, _ in vals) / n
            ax.plot(centers + [CAP], pct + [rate], color=c, lw=2.0 if k != "bf16" else 2.4,
                    ls="-" if k != "bf16" else "--", label=f"{lbl}  (mean {mean / 1000:.1f}k)",
                    zorder=3, solid_joinstyle="round")
            labels.append((rate, c))
            ymax = max([ymax, rate] + pct)
        # cap-hit labels, spread apart where the rates nearly coincide
        sep, prev = ymax * 0.06, None
        for rate, c in sorted(labels):
            y = rate if prev is None else max(rate, prev + sep)
            ax.annotate(f"{rate:.1f}%", xy=(CAP, rate), xytext=(CAP * 1.012, y), ha="left", va="center",
                        fontsize=11, color=c, fontweight="bold", zorder=5)
            prev = y
        ax.set_ylim(0, ymax * 1.18)
        ax.text(CAP - a.binw * 1.1, ymax * 1.10, "cap hit", ha="right", va="center",
                fontsize=11, color="#B00020", fontweight="bold")
        ax.set_xlim(0, CAP * 1.16)
        ax.xaxis.set_major_locator(MultipleLocator(8192))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(v) // 1024}k" if v else "0"))
        ax.set_xlabel("Generated tokens per problem")
        ax.set_ylabel("Problems (%)")
        ax.set_title(f"{title}, {a.rate} bit", fontsize=11)
        ax.legend(frameon=False, fontsize=9.5, loc="center right", bbox_to_anchor=(0.86, 0.55))
    fig.tight_layout(w_pad=1.5, pad=0.6)
    for ext in ("pdf", "png"):
        p = HERE / f"figure_stability_{a.task}_{a.rate.replace('.', 'p')}b.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"[fig] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
