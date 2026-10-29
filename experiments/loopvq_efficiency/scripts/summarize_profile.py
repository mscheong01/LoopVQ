#!/usr/bin/env python3
"""Roll profile_point.py cells into one table: where each arm's GPU time goes, per regime.

    python3 summarize_profile.py outputs/baseline/profile     # writes PROFILE.md there, prints it

One row per (point, stage, arm). ms columns are per profiled span (decode: PROFILE_STEPS steps;
prefill: the extend forward). "idle" = span not covered by any kernel (launch gaps, host syncs).
Category columns are milliseconds, so arms can be compared directly at the same point.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

CATS = ["vq_stage1", "hp_stage1", "stage2", "encode_flush", "kv_write", "kv_meta", "prefill_attn",
        "gemm", "norm_act_rope", "copy_cast", "sampling", "other"]
SHORT = {"bf16": "bf16", "loopvq8b_g16_lvgqe2e": "loopvq_0.5bit", "loopvq8b_g32_lvgqe2e": "loopvq_0.25bit",
         "cq8b_g16_gq": "cq_0.5bit", "cq8b_g32_gq": "cq_0.25bit"}


def main() -> int:
    d = Path(sys.argv[1]).resolve()
    rows = []
    for f in sorted(d.glob("*.json")):
        if "__thru_capacity__" in f.name:
            continue
        j = json.loads(f.read_text())
        if not j.get("ok"):
            rows.append((f"{j.get('panel')}/{j.get('key')}", "-", j.get("arm"), None)); continue
        for st, b in j["stages"].items():
            rows.append((f"{j['panel']}/{j['key']}", st, j["arm"], b))
    used = [c for c in CATS if any(b and b["by_cat_ms"].get(c) for *_, b in rows)]
    hdr = ["point", "stage", "arm", "span ms", "idle ms", "idle %"] + used
    out = [f"# profile: {d.parent.name}\n", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for pt, st, arm, b in sorted(rows, key=lambda r: (r[0], r[1], list(SHORT).index(r[2]) if r[2] in SHORT else 9)):
        if b is None:
            out.append(f"| {pt} | {st} | {SHORT.get(arm, arm)} | FAIL |" + " |" * (len(hdr) - 4)); continue
        cells = [pt, st, SHORT.get(arm, arm), f"{b['span_ms']:.1f}", f"{b['idle_ms']:.1f}",
                 f"{b['idle_share']*100:.0f}"] + [f"{b['by_cat_ms'].get(c, 0):.1f}" for c in used]
        out.append("| " + " | ".join(cells) + " |")
    others = {(pt, st, arm): b["top_other"][:3] for pt, st, arm, b in rows if b and b.get("top_other")}
    if others:
        out.append("\nLargest unclassified kernels (extend CATS in profile_point.py if these matter):")
        for (pt, st, arm), lst in others.items():
            out.append(f"- {pt} {st} {SHORT.get(arm, arm)}: " + "; ".join(f"{n} {v:.1f} ms" for n, v in lst))
    text = "\n".join(out) + "\n"
    (d / "PROFILE.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
