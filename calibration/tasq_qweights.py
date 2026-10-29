"""TaSQ's query-guided key channel weights (W) for every Ouro KV slot.

Same definition as the TaSQ repository's codebooks/make_tasq.py pass 1, per slot = ut*L + layer instead of per layer:

  M = E over valid causal (t, p<=t) query-key pairs of R_p^T q_t q_t^T R_p,   W = diag(M),
  stored as w = sqrt(W / mean(W)) per (slot, head), so mean(w^2) = 1 and k^w = diag(w) k.

Ouro runs each layer T times; the query that reads slot ut*L + l is layer l's query at loop step
ut (each step attends to its own slot), so the moments are accumulated per (layer, call index).
Ouro is MHA (NREP = 1) without QK-norm; q_proj output + RoPE is the served query.

The closed form (suffix sums of the per-token diagonal and RoPE-pair cross moments) is checked
against the full einsum on head 0 of the first window.

    python tasq_qweights.py --model <models>/Ouro-1.4B --calib-data gpqa_nodiamond \
        --n-calib 16 --calib-seq-len 2048 --out work/calib_cache/ouro_1p4b_tasq_w_gpqa_nodiamond.pt
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from build_ouro_bundle import calib_texts
from ouro_cq import rotate_half


def causal_diag_m(q, cos, sin, NKV, NREP):
    """post-RoPE queries [NQ, S, D] (rotate-half layout), cos/sin [S, D] -> sum over causal pairs
    (t, p <= t) of diag(R_p^T q_t q_t^T R_p), [NKV, D] -- make_tasq.py pass 1's closed form."""
    D = q.shape[-1]; HALF = D // 2
    SGN = torch.cat([torch.ones(HALF), -torch.ones(HALF)]).to(q.device).double()
    qt = q.view(NKV, NREP, -1, D)
    c2, s2, csx = cos ** 2, sin ** 2, cos * sin
    d_t = (qt ** 2).sum(1) / NREP
    x_t = (qt[..., :HALF] * qt[..., HALF:]).sum(1) / NREP
    Sd = torch.flip(torch.cumsum(torch.flip(d_t, [1]), 1), [1])
    Sx = torch.flip(torch.cumsum(torch.flip(x_t, [1]), 1), [1])
    Sd_sw = torch.cat([Sd[..., HALF:], Sd[..., :HALF]], -1)
    Sx_f = torch.cat([Sx, Sx], -1)
    return (c2 * Sd + s2 * Sd_sw + 2.0 * SGN * csx * Sx_f).sum(1)


def huginn_main(args, dev):
    """Huginn: the query that reads slot s is the block application that wrote it (each block
    application attends to its own slot), captured post-RoPE by huginn_torch's query hook in the
    NeoX channel order the bundles use. MHA, so NREP = 1."""
    from huginn_torch import HuginnF
    m = HuginnF(args.model, device=dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = calib_texts(args, tok, dev)
    H, D = m.H, m.D
    num = torch.zeros(m.n_slots, H, D, dtype=torch.double, device=dev)
    den = torch.zeros((), dtype=torch.double, device=dev)
    for bi, ids in enumerate(texts):
        S = ids.shape[1]
        cos = torch.cat([m.cos[:S], m.cos[:S]], -1).double()
        sin = torch.cat([m.sin[:S], m.sin[:S]], -1).double()

        def qh(slot, q):
            num[slot] += causal_diag_m(q[0].transpose(0, 1).double(), cos, sin, H, 1)
        with torch.no_grad():
            m.forward(ids, qh=qh)
        den += torch.arange(S, 0, -1, device=dev, dtype=torch.double).sum()
        print(f"  {bi + 1}/{len(texts)}", flush=True)
    basis = (num / den).cpu().numpy()
    weps = np.clip(basis.max(-1, keepdims=True) * 1e-6, 1e-30, None)
    b = np.clip(basis, weps, None)
    w = torch.from_numpy(np.sqrt(b / b.mean(-1, keepdims=True))).float()
    key = {"model": str(Path(args.model).resolve()), "n_calib": args.n_calib,
           "seq_len": args.calib_seq_len, "seed": args.seed, "data": args.calib_data,
           "huginn_steps": m.T}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": json.dumps(key), "w": w,
                "definition": "make_tasq.py causal diag(M), per slot (Huginn engine slot order, NeoX channels)"},
               args.out)
    print(f"[done] w {tuple(w.shape)} -> {args.out}; w range [{w.min():.3f}, {w.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-cache", default=os.environ.get("CALIB_DATA_DIR", str(Path(__file__).resolve().parent / ".." / "work" / "calib_data")))
    ap.add_argument("--calib-data", choices=("wikitext", "mix", "gpqa_nodiamond", "c4", "slimpajama"), default="gpqa_nodiamond")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--calib-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    from huginn_torch import is_huginn
    if is_huginn(cfg):
        return huginn_main(args, dev)
    L, T = cfg.num_hidden_layers, cfg.total_ut_steps
    NQ, NKV, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    NREP, HALF = NQ // NKV, D // 2
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    texts = calib_texts(args, tok, dev)

    num = torch.zeros(L * T, NKV, D, dtype=torch.double, device=dev)
    den = torch.zeros((), dtype=torch.double, device=dev)
    SGN = torch.cat([torch.ones(HALF), -torch.ones(HALF)]).to(dev).double()
    st = {"calls": [0] * L, "cs": None, "checked": False}

    def hook(li):
        def h(_, __, o):
            ut = st["calls"][li]; st["calls"][li] += 1
            cos, sin = st["cs"]                                                  # [T, D] float64
            q = o[0].view(-1, NQ, D).transpose(0, 1).double()                    # [NQ, T, D] pre-RoPE
            q = q * cos + rotate_half(q) * sin                                   # post-RoPE query
            qt = q.view(NKV, NREP, -1, D)
            c2, s2, csx = cos ** 2, sin ** 2, cos * sin
            d_t = (qt ** 2).sum(1) / NREP                                        # [NKV, T, D]
            x_t = (qt[..., :HALF] * qt[..., HALF:]).sum(1) / NREP                # [NKV, T, HALF]
            Sd = torch.flip(torch.cumsum(torch.flip(d_t, [1]), 1), [1])          # suffix sums over t
            Sx = torch.flip(torch.cumsum(torch.flip(x_t, [1]), 1), [1])
            Sd_sw = torch.cat([Sd[..., HALF:], Sd[..., :HALF]], -1)
            Sx_f = torch.cat([Sx, Sx], -1)
            n_ = (c2 * Sd + s2 * Sd_sw + 2.0 * SGN * csx * Sx_f).sum(1)          # [NKV, D]
            num[ut * L + li] += n_
            if not st["checked"]:
                st["checked"] = True
                O = torch.einsum("rtd,rte->tde", qt[0], qt[0]) / NREP
                S = torch.flip(torch.cumsum(torch.flip(O, [0]), 0), [0])
                Sm = torch.zeros(D, D, dtype=torch.double, device=dev)
                idx = torch.arange(HALF)
                Sm[idx, idx + HALF] = -1.0
                Sm[idx + HALF, idx] = 1.0
                Rm = torch.diag_embed(cos) + sin.unsqueeze(-1) * Sm.unsqueeze(0)
                ref = torch.einsum("pdc,pde,pec->c", Rm, S, Rm)
                rel = ((n_[0] - ref).abs().max() / ref.abs().max()).item()
                print(f"  [check] causal diag(M): closed form vs full einsum, max rel {rel:.3e}", flush=True)
                assert rel < 1e-6, "causal diag(M) closed form disagrees with the einsum"
        return h

    hs = [model.model.layers[li].self_attn.q_proj.register_forward_hook(hook(li)) for li in range(L)]
    with torch.no_grad():
        for bi, ids in enumerate(texts):
            S = ids.shape[1]
            ref = torch.empty(1, S, 1, device=dev, dtype=torch.float32)
            cos, sin = model.model.rotary_emb(ref, torch.arange(S, device=dev).unsqueeze(0))
            st["cs"] = (cos[0].double(), sin[0].double())
            st["calls"] = [0] * L
            model(ids, use_cache=False)
            assert st["calls"] == [T] * L, f"expected {T} calls per layer, got {st['calls']}"
            den += torch.arange(S, 0, -1, device=dev, dtype=torch.double).sum()
            print(f"  {bi + 1}/{len(texts)}", flush=True)
    for h in hs:
        h.remove()

    basis = (num / den).cpu().numpy()                                            # [L*T, NKV, D]
    weps = np.clip(basis.max(-1, keepdims=True) * 1e-6, 1e-30, None)
    b = np.clip(basis, weps, None)
    w = torch.from_numpy(np.sqrt(b / b.mean(-1, keepdims=True))).float()        # mean(w^2) = 1
    key = {"model": str(Path(args.model).resolve()), "n_calib": args.n_calib,
           "seq_len": args.calib_seq_len, "seed": args.seed, "data": args.calib_data}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": json.dumps(key), "w": w, "definition": "make_tasq.py causal diag(M), per slot"},
               args.out)
    print(f"[done] w {tuple(w.shape)} -> {args.out}; w range [{w.min():.3f}, {w.max():.3f}]")
    for sl in (0, L - 1, L, 2 * L - 1, (T - 1) * L, T * L - 1):
        print(f"  slot {sl:3d} h0 w min/max {w[sl, 0].min():.3f}/{w[sl, 0].max():.3f}")


if __name__ == "__main__":
    main()
