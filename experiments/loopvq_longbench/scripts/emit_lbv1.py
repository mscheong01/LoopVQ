#!/usr/bin/env python3
"""Score one run_cell.py LongBench (v1) JSONL with the official scorer and write one outputs/ record
per task (lbv1@<task>) plus lbv1@avg, the unweighted mean over the 16 tasks (LongBench's "Avg").

Scorer as THUDM/LongBench eval.py: per sample the max over reference answers of the task metric
(lb/longbench_metrics.py: qa_f1 / rouge / classification / retrieval / count / code_sim); for trec,
triviaqa and samsum only the first line of the generation is scored; task score = 100 x mean.

  emit_lbv1.py <raw.jsonl> <outdir> <tag> <method> <artifact_dir>
"""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lb"))
from longbench_metrics import (classification_score, code_sim_score, count_score, qa_f1_score,  # noqa: E402
                               retrieval_score, rouge_score)

DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score, "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score, "musique": qa_f1_score,
    "gov_report": rouge_score, "qmsum": rouge_score, "multi_news": rouge_score,
    "trec": classification_score, "triviaqa": qa_f1_score, "samsum": rouge_score,
    "passage_count": count_score, "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
N_EXPECTED = 3750


def main():
    raw, outdir, tag, method, artifact = sys.argv[1:6]
    rows = [json.loads(l) for l in open(raw) if l.strip()]
    if len(rows) != N_EXPECTED:
        raise SystemExit(f"[pe] LongBench run has {len(rows)}/{N_EXPECTED} rows, not scoring: {raw}")
    per = {}
    for r in rows:
        # run_cell.py overwrites "task" with its --task label; the row id is "<task>_<index>"
        t, pred = r["_id"].rsplit("_", 1)[0], r["prediction"]
        if t in FIRST_LINE:
            pred = pred.lstrip("\n").split("\n")[0]
        s = max(DATASET2METRIC[t](pred, gt, all_classes=r["all_classes"]) for gt in r["answers"])
        per.setdefault(t, []).append(s)
    scores = {t: 100 * sum(v) / len(v) for t, v in per.items()}
    scores["avg"] = sum(scores[t] for t in DATASET2METRIC) / len(DATASET2METRIC)

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True, cwd=outdir).strip()
        except Exception:
            return "unknown"

    ahash = "n/a"
    cb = os.path.join(artifact, "codebook.pt")
    if os.path.exists(cb):
        h = hashlib.md5()
        with open(cb, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ahash = h.hexdigest()[:12]
    os.makedirs(outdir, exist_ok=True)
    for key, v in scores.items():
        bench = f"lbv1@{key}"
        json.dump({
            "model": tag, "bench": bench, "method": method, "seed": 0,
            # stored on the 0-1 scale like every other cell; the table multiplies by 100
            "results": {bench: {"score,none": v / 100, "n_rows": len(per.get(key, rows))}},
            "provenance": {
                "commit": git("rev-parse", "--short", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "artifact_md5": ahash, "scale_dtype": os.environ.get("PE_SCALE_DTYPE"),
                "prefix_tokens": os.environ.get("PREFIX_TOKENS"), "recent_tokens": os.environ.get("RECENT_TOKENS"),
                "rows_file": os.path.basename(raw), "host": socket.gethostname(),
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            },
        }, open(os.path.join(outdir, f"{tag}__{bench}__{method}__s0.json"), "w"), indent=1)
    print(f"[pe] {tag}/lbv1/{method}: avg {scores['avg']:.2f} -> {outdir}")


if __name__ == "__main__":
    main()
