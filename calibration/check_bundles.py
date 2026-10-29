#!/usr/bin/env python3
"""Inventory bundles by CONTENT, not by directory existence.

build_ouro_bundle creates its output directory before it fits anything, so a build
that dies -- OOM, kill, anything -- leaves an empty directory behind. Drivers that
gate on `[ -d $artifact ]` then treat that as a usable bundle and hand it to the
server, which fails much later and much less legibly. Every 2.6B "bundle" in this
tree was an empty directory of exactly that kind, and the sweep's own check waved
them through.

Exit status is nonzero if any requested bundle is unusable, so a driver can gate on
it directly:

    python3 calibration/check_bundles.py ouro_2p6b_cq8b_g8 || skip

With no arguments it prints the whole inventory.
"""
import argparse
import json
import os
import sys

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "experiments", "artifacts")
NEEDED = ("codebook.pt", "vq_v_codebook.pt")


def status(name):
    d = os.path.join(ART, name)
    if not os.path.isdir(d):
        return False, "no directory", None
    missing = [f for f in NEEDED if not os.path.isfile(os.path.join(d, f))]
    if missing:
        return False, "missing " + ",".join(missing), None
    for f in NEEDED:
        if os.path.getsize(os.path.join(d, f)) == 0:
            return False, f"{f} is empty", None
    info = None
    p = os.path.join(d, "build_info.json")
    if os.path.isfile(p):
        try:
            b = json.load(open(p))
            lc = b.get("loop_coupled")
            info = (f"{b.get('bits_per_channel')}b/ch G={b.get('group_dim')} "
                    + (f"Tc={lc['loops']} c={lc['channels']}" if lc else "channel"))
        except Exception as e:
            return False, f"build_info unreadable: {e}", None
    return True, "ok", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    names = a.names or sorted(x for x in os.listdir(ART)
                              if os.path.isdir(os.path.join(ART, x)))
    bad = 0
    for n in names:
        ok, why, info = status(n)
        bad += not ok
        if not a.quiet:
            print("%-30s %-8s %s" % (n, "OK" if ok else "BAD",
                                     info or ("" if ok else why)))
    if not a.quiet and not a.names:
        print(f"\n{len(names) - bad} usable, {bad} unusable")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
