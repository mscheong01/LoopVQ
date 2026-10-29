#!/usr/bin/env python3
"""One markdown summary of a loopvq_efficiency variant: where each arm stands against bf16.

    python3 summarize.py outputs/baseline            # writes outputs/baseline/SUMMARY.md, prints it
    python3 summarize.py outputs/opt_flush           # also shows the change vs outputs/baseline

Per panel: the metric per arm, then "x bf16" (>1 = slower than bf16 for latency, faster for
throughput, so every ratio reads "how much better than bf16"). A non-baseline variant adds
"vs baseline" for the same arm and point. Failed cells show as FAIL with the reason's first line.
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

ORDER = ["bf16", "loopvq8b_g16_lvgqe2e", "loopvq8b_g32_lvgqe2e", "loopvq8b_g16_tqlve2e", "loopvq8b_g32_tqlve2e",
         "cq8b_g16_gq", "cq8b_g32_gq"]
SHORT = {"bf16": "bf16", "loopvq8b_g16_lvgqe2e": "loopvq_0.5bit", "loopvq8b_g32_lvgqe2e": "loopvq_0.25bit",
         "loopvq8b_g16_tqlve2e": "tasq_loopvq_0.5bit", "loopvq8b_g32_tqlve2e": "tasq_loopvq_0.25bit",
         "cq8b_g16_gq": "cq_0.5bit", "cq8b_g32_gq": "cq_0.25bit"}
# metric, unit, higher_is_better
PANELS = {
    "decode": ("decode_latency_ms_per_token", "ms/tok", False),
    "prefill": ("prefill_latency_s", "s TTFT", False),
    "thru": ("overall_throughput_tok_s", "tok/s E2E", True),
}


def load(d: Path):
    cells, caps = {}, {}
    for f in d.glob("*.json"):
        if f.name.endswith(".raw.json"):
            continue
        parts = f.stem.split("__")
        try:
            j = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if len(parts) >= 3 and parts[1] == "thru_capacity":
            caps[(parts[0], j.get("arm"))] = j
        elif len(parts) == 4:
            cells[(parts[0], parts[1], parts[2], parts[3])] = j
    return cells, caps


def keyorder(k: str):
    m = re.search(r"(\d+)$", k)
    return (re.sub(r"\d+$", "", k), int(m.group(1)) if m else 0)


def val(cell, metric):
    if not cell:
        return None
    if not cell.get("ok"):
        return "FAIL"
    return (cell.get("metrics") or {}).get(metric)


def fmt(v):
    if v is None:
        return "--"
    if v == "FAIL":
        return "FAIL"
    return f"{v:.3g}" if v < 100 else f"{v:,.0f}"


def main() -> int:
    d = Path(sys.argv[1]).resolve()
    base = d.parent / "baseline"
    if "--no-base" in sys.argv[2:]:      # e.g. an H100 run: the Ada baseline is not its reference
        base = d
    cells, caps = load(d)
    bcells, _ = load(base) if base != d and base.exists() else ({}, {})
    out = [f"# {d.name}\n"]
    models = sorted({k[0] for k in cells})
    for model in models:
        out.append(f"## {model}\n")
        for panel, (metric, unit, hib) in PANELS.items():
            keys = sorted({k[3] for k in cells if k[0] == model and k[1] == panel}, key=keyorder)
            arms = [a for a in ORDER if any(k[0] == model and k[1] == panel and k[2] == a for k in cells)]
            # a quick-mode variant measures no bf16: take bf16 from the baseline (same GPU)
            bf16_from_base = "bf16" not in arms and any(
                k[0] == model and k[1] == panel and k[2] == "bf16" for k in bcells)
            if not keys:
                continue
            out.append(f"### {panel} ({unit})\n")
            hdr = (["point"] + (["bf16 (baseline)"] if bf16_from_base else []) + [SHORT.get(a, a) for a in arms]
                   + [f"{SHORT.get(a, a)} x bf16" for a in arms if a != "bf16"])
            if bcells:
                hdr += [f"{SHORT.get(a, a)} vs baseline" for a in arms]
            out.append("| " + " | ".join(hdr) + " |")
            out.append("|" + "---|" * len(hdr))
            for k in keys:
                vs = {a: val(cells.get((model, panel, a, k)), metric) for a in arms}
                row = [k] + ([fmt(val(bcells.get((model, panel, "bf16", k)), metric))] if bf16_from_base else []) \
                    + [fmt(vs[a]) for a in arms]
                ref = vs.get("bf16")
                if bf16_from_base:
                    ref = val(bcells.get((model, panel, "bf16", k)), metric)
                for a in arms:
                    if a == "bf16":
                        continue
                    v = vs[a]
                    if isinstance(v, float) and isinstance(ref, float) and v > 0 and ref > 0:
                        row.append(f"{(v / ref if hib else ref / v):.2f}x")
                    else:
                        row.append("--")
                if bcells:
                    for a in arms:
                        v, b = vs[a], val(bcells.get((model, panel, a, k)), metric)
                        if isinstance(v, float) and isinstance(b, float) and v > 0 and b > 0:
                            row.append(f"{(v / b if hib else b / v):.2f}x")
                        else:
                            row.append("--")
                out.append("| " + " | ".join(row) + " |")
            if panel == "thru":
                cl = [f"{SHORT.get(a, a)} {caps[(model, a)]['capacity_batch']}" for a in arms if (model, a) in caps]
                if cl:
                    out.append(f"\nCapacity from the KV pool alone (requests at this shape): {', '.join(cl)}. "
                               "For mixed-KV arms this ignores the per-request bf16 HP arena (~0.26 GB on 1.4B), "
                               "which caps the real batch lower.")
            fails = [(k, a, (c.get('error') or {}).get('stderr_tail', '').strip().splitlines()[-1:] or ['?'])
                     for (m, p, a, k), c in cells.items() if m == model and p == panel and not c.get("ok")]
            for k, a, why in fails:
                out.append(f"- FAIL {a}/{k}: {why[0][:160]}")
            out.append("")
    text = "\n".join(out) + "\n"
    (d / "SUMMARY.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
