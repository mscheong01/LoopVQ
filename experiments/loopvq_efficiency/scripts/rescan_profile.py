#!/usr/bin/env python3
"""Re-classify existing profile cells from their saved traces after CATS changes (no GPU needed).

    python3 rescan_profile.py outputs/<VARIANT>/profile && python3 summarize_profile.py outputs/<VARIANT>/profile
"""
import json, re, sys
from pathlib import Path
from profile_point import breakdown

for f in sorted(Path(sys.argv[1]).glob("*.json")):
    j = json.loads(f.read_text())
    if not j.get("traces"):
        continue
    stages = {}
    for t in map(Path, j["traces"]):
        st = "prefill" if re.search(r"extend|prefill", t.name, re.I) else "decode" if "decode" in t.name.lower() else t.stem
        b = breakdown(t)
        if b:
            stages.setdefault(st, b)
    j["stages"] = stages
    f.write_text(json.dumps(j, indent=2) + "\n")
    print("rescanned", f.name)
