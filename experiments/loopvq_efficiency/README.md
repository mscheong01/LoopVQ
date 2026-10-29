# LoopVQ efficiency

Serving efficiency of LoopVQ on Ouro-Thinking (Section 5.4 and the appendix of the paper).

| panel | what | shape |
|---|---|---|
| `decode` | batch-1 decode latency vs context length | 1k / 8k / 32k, 64 output tokens |
| `thru` | end-to-end throughput vs batch size, up to each arm's pool capacity | 2048 in / 4096 out (`THRU_OUT=32768` for the reasoning length) |
| `prefill` | prefill latency (TTFT) | 4k / 16k / 32k |
| `profile` | GPU-time breakdown per regime (torch profiler) | decode 1k / 32k, prefill 16k, thru bs 32 |

Arms take the method names of the top-level README (`bf16`, `loopvq_0.5bit`, `loopvq_0.25bit`,
`tasq_loopvq_0.5bit`, `cq_0.5bit`, ...); CQ shares the decode kernel without loop coupling. Any other
bundle name is passed through unchanged.

## Run

```bash
scripts/run.sh all                                     # decode, prefill and thru for bf16 and LoopVQ 0.5 / 0.25 bit
MODEL_TAG=2p6b ARMS="bf16 loopvq_0.5bit" scripts/run.sh thru   # Ouro-2.6B-Thinking
BENCH_POINT=$PWD/scripts/bench_point_warm.py VARIANT=prefill_warm scripts/run.sh prefill
scripts/run.sh profile                                 # then make_figure_breakdown.py
python3 scripts/summarize.py outputs/baseline          # tables vs bf16
```

Everything runs on **one GPU** (`GPU_ID`, default 0), one point after another; arms are only
compared on the same card. The server path is the one the accuracy runs use (`scripts/serve_method.sh`,
CUDA graphs asserted). The settings that differ from the accuracy runs:

- KV window: 8 sink + 64 recent bf16 tokens, as in the general benchmarks.
- `LOOPVQ_MAX_TOKENS` is lifted (the accuracy arms pin `--max-total-tokens`), so the pool is what the
  memory fraction buys and `thru` measures the real capacity.
- Contexts stop at 32k: a 64k LoopVQ prompt only fits with the loop-coupled prefill buffer offloaded
  to host RAM, and bf16 64k does not fit on a 48 GB card.
- The memory fraction is set per panel and arm (`scripts/run.sh`); it changes no timed work.

Cells land in `outputs/<VARIANT>/` (default `baseline`); `RUNS.txt` in each variant records the
environment and commit that produced it.

## Reasoning stability

`dump_outlen.py <tag> gpqa_diamond` collects the per-problem output length of the reasoning runs
(`experiments/loopvq_reasoning`), and `make_figure_stability.py` plots its distribution per arm.
