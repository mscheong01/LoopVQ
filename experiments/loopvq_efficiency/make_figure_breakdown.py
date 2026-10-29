#!/usr/bin/env python3
"""Kernel-time breakdown figure for the discussion: where GPU time goes, bf16 vs LoopVQ, per regime.

    python3 make_figure_breakdown.py [outputs/m3_profile/profile]   # needs matplotlib

Input: profile_point.py cells (run.sh profile; m3_profile = milestone3 code, window 8/64, 1.4B,
measured on GPU 1 while GPU 0 ran the 2.6B sweep -> the figure shows time RELATIVE to bf16, not ms).
Each bar is GPU time per decode step (bs1 points), per generated token (thru: step / batch size, bf16
at its KV capacity, LoopVQ at bs64) or per prefill (16k), normalised to the bf16 bar of that panel.
Writes figure_breakdown.{pdf,png} and breakdown_table.csv next to this script.
"""
import csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs/m3_profile/profile"

ARMS = [("bf16", "BF16"), ("loopvq8b_g16_lvgqe2e", "LoopVQ 0.5b"), ("loopvq8b_g32_lvgqe2e", "LoopVQ 0.25b")]
# (panel title, cell panel, key prefix, stage, normalise per generated token)
POINTS = [("Decode, bs=1, 1k ctx", "decode", "ctx1024", "decode", False),
          ("Decode, bs=1, 8k ctx", "decode", "ctx8192", "decode", False),
          ("Decode, bs=1, 32k ctx", "decode", "ctx32768", "decode", False),
          ("Decode, bs=16 (per token)", "thru", "2048x256_bs16", "decode", True),
          ("Prefill, 16k", "prefill", "ctx16384", "prefill", False)]
# paper categories <- profile_point CATS (+ idle)
GROUPS = [("Linear layers", "#2a78d6"),
          ("Attention, compressed KV", "#eb6834"),
          ("Attention, BF16 KV", "#1baf7a"),
          ("KV encode / write", "#eda100"),
          ("Other + idle", "#c8c6bf")]


def group(cat_ms: dict, stage: str, idle: float, arm: str) -> dict:
    g = dict.fromkeys((n for n, _ in GROUPS), 0.0)
    for c, v in cat_ms.items():
        if c == "gemm":
            k = "Linear layers"
        elif c == "vq_stage1":
            k = "Attention, compressed KV"
        elif c in ("hp_stage1", "stage2"):
            # bf16: the whole cache; LoopVQ: the sink/recent window (+ split merge of both parts)
            k = "Attention, BF16 KV"
        elif c == "prefill_attn" and stage == "prefill":
            k = "Attention, BF16 KV"   # prefill attends to the bf16 K/V of the prompt for every arm
        elif c in ("encode_flush", "kv_write", "kv_meta"):
            k = "KV encode / write"
        else:
            k = "Other + idle"
        g[k] += v
    g["Other + idle"] += idle
    return g


def load(panel, keypfx, arm):
    hits = [f for f in PDIR.glob(f"*__{panel}__{arm}__{keypfx}*.json")]
    assert len(hits) == 1, (panel, keypfx, arm, hits)
    return json.loads(hits[0].read_text())


def main():
    rows, data = [], []
    for title, panel, kp, stage, per_tok in POINTS:
        bars = []
        for arm, label in ARMS:
            j = load(panel, kp, arm)
            b = j["stages"][stage]
            div = (j["profile_steps"] * j["batch_size"]) if per_tok else (j["profile_steps"] if stage == "decode" else 1)
            g = {k: v / div for k, v in group(b["by_cat_ms"], stage, b["idle_ms"], arm).items()}
            bars.append((label, g, j["batch_size"]))
        ref = sum(bars[0][1].values())
        data.append((title, [(l, {k: v / ref for k, v in g.items()}, bs) for l, g, bs in bars]))
        for l, g, bs in bars:
            rows.append({"panel": title.replace("\n", " "), "arm": l, "batch_size": bs,
                         "total_rel_bf16": round(sum(g.values()) / ref, 4),
                         **{k: round(v / ref, 4) for k, v in g.items()}})

    with open(HERE / "breakdown_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#0b0b0b"})
    fig, axes = plt.subplots(1, len(data), figsize=(7.0, 1.85), sharey=True)
    for ax, (title, bars) in zip(axes, data):
        for i, (label, g, bs) in enumerate(bars):
            y, left = len(bars) - 1 - i, 0.0
            for name, col in GROUPS:
                v = g[name]
                if v > 0:
                    ax.barh(y, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.8)
                left += v
            txt = f"{left:.2f}" + (f"  (bs {bs})" if "per token" in title else "")
            ax.text(left + 0.03, y, txt, va="center", ha="left", fontsize=6.5, color="#52514e")
        ax.set_title(title, fontsize=7.5, color="#0b0b0b")
        ax.set_xlim(0, 1.55 if "per token" not in title else 1.75)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([l for l, _, _ in reversed(bars)])
        ax.axvline(1.0, color="#52514e", lw=0.5, ls=(0, (2, 2)), zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
    fig.supxlabel("GPU time relative to BF16", fontsize=7.5, y=0.02)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in GROUPS]
    fig.legend(handles, [n for n, _ in GROUPS], loc="upper center", ncol=len(GROUPS), frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 1.07), handlelength=1.0, columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"figure_breakdown.{ext}", bbox_inches="tight", dpi=200)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
