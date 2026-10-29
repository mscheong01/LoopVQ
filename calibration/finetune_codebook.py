"""End-to-end, loop-aware codebook refinement: fit the codes to the FINAL loop's output.

k-means (with CQ's Fisher weight or any of the variants tried) fits each code layer to its own
reconstruction error, weighted at best by a first-order, per-coordinate sensitivity. But an
early loop step's KV error does not reach the logits directly: it changes that step's attention,
the hidden state the next steps start from, their KV, and so on through all T steps. This tunes
the codebook entries themselves against what serving cares about --
KL(p_clean || p_quantised) of the final loop's next-token distribution -- by backpropagating
through the whole loop recurrence:

  per step: clean forward (no grad) -> every slot's K (pre-RoPE) / V
            assign codes with the CURRENT codebook exactly as served (forward map, pooled norm,
            loop scale, coupled codes)                                         (no grad)
            reconstruct from the codebook parameters (differentiable gather), second forward
            with those K/V, KL to the clean logits, backward into the entries, Adam step

Sequences 0..(n-val-1) of the calibration set train, the last --n-val are held out; the
codebook with the best held-out KL is written back in the bundle's own format (same files,
same layout -- the serving path does not change).

    python finetune_codebook.py --src ouro_1p4b_loopvq8b_g16_a7 --dst ouro_1p4b_loopvq8b_g16_a7e2e
"""
import argparse
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_ouro_bundle import calib_texts, enable_layer_checkpointing, load_wikitext, make_fisher_cache_cls
from ouro_cq import RopeInverter, rotate_half

HERE = Path(__file__).resolve().parent
ART = HERE / ".." / "experiments" / "artifacts"
_spec = importlib.util.spec_from_file_location(
    "loopvq_pack", HERE / ".." / "python" / "sglang" / "srt" / "mem_cache" / "loopvq_pack.py")
LP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(LP)


class Coded(torch.nn.Module):
    """One loop-coupled bundle file (K or V) with trainable coupled entries [Lc,H,NG',KC,G]."""

    def __init__(self, path, device):
        super().__init__()
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.blob = b
        lc = b["loop_coupled"]
        self.Lb, self.Tc, self.c = lc["base_layers"], lc["loops"], lc["channels"]
        L, H, D, _ = b["forward"].shape
        self.L, self.H, self.D, self.T = L, H, D, L // self.Lb
        self.ident = bool(torch.equal(b["forward"], torch.eye(D).expand_as(b["forward"]))) \
            and not bool(b["mean"].any())
        self.fwd = None if self.ident else b["forward"].float().to(device)
        self.inv = None if self.ident else b["inverse"].float().to(device)
        self.norm = bool(b.get("pool_heads_scale", False))
        self.ls = b["loop_scale"].float().to(device) if b.get("loop_scale") is not None else None
        self.cw = b["chan_weight"].float().to(device) if b.get("chan_weight") is not None else None
        cb = LP.build_coupled_codebook(b["codebooks"], self.Lb, self.T, self.Tc, self.c, H, D)
        self.cb = torch.nn.Parameter(cb.float().to(device))                  # [Lc,H,NG',KC,G]
        self.cl = torch.tensor([LP.code_layer_of(s, self.Lb, self.Tc) for s in range(L)], device=device)
        self.step = torch.tensor([LP.step_of(s, self.Lb, self.Tc) for s in range(L)], device=device)

    def to_coded(self, x):
        """[L,n,H,D] natural -> (coded rows [L,n,H,D], per-(slot,token) norm or None)."""
        r = x if self.ident else torch.einsum("lnhd,lhde->lnhe", x, self.fwd)
        sc = None
        if self.cw is not None:             # TaSQ's W before the pooled norm, as the pool does
            r = r * self.cw[:, None]
        if self.norm:
            sc = r.pow(2).mean(dim=(2, 3), keepdim=True).sqrt().clamp_min(1e-8)
            r = r / sc
        if self.ls is not None:
            r = r * self.ls[:, None]
        return r, sc

    def from_coded(self, r, sc):
        if self.ls is not None:
            r = r / self.ls[:, None]
        if sc is not None:
            r = r * sc
        if self.cw is not None:
            r = r / self.cw[:, None]
        return r if self.ident else torch.einsum("lnhe,lhed->lnhd", r, self.inv)

    @torch.no_grad()
    def assign(self, r):
        v = LP.regroup_for_encode(r, self.Lb, self.T, self.Tc, self.c)       # [Lc,n,H,NG',G]
        Lc, n, H, NG, G = v.shape
        # int16 (K <= 2^15): int64 codes are 3.2 GB per kind on 2.6B at G = 2 and OOM'd the backward
        idx = torch.empty(Lc, n, H, NG, dtype=torch.int16, device=r.device)
        cb = self.cb.detach()
        for cl in range(Lc):
            vv = v[cl].permute(1, 2, 0, 3).reshape(H * NG, n, G)
            c = cb[cl].reshape(H * NG, -1, G)
            # chunked over tokens: one [H*NG, n, KC] distance block is 1 GB at 0.5 b/ch, n=2048
            a_ = torch.cat([torch.cdist(vv[:, i:i + 512], c).argmin(-1) for i in range(0, n, 512)], 1)
            idx[cl] = a_.view(H, NG, n).permute(2, 0, 1)
        return idx

    def decode(self, idx):
        """codes [Lc,n,H,NG'] -> coded rows [L,n,H,D], differentiable in self.cb."""
        Lc, n, H, NG = idx.shape
        G = self.cb.shape[-1]
        h = torch.arange(H, device=idx.device).view(1, 1, H, 1)
        g = torch.arange(NG, device=idx.device).view(1, 1, 1, NG)
        cl = torch.arange(Lc, device=idx.device).view(Lc, 1, 1, 1)
        # gather from a bf16 view of the fp32 entries: the gathered [Lc,n,H,NG',G] block (and its
        # backward scatter) at half the bytes -- at 0.5 b/ch x 2048 tokens the fp32 version did
        # not fit next to the model's checkpointed backward on 24 GB
        q = self.cb.to(torch.bfloat16)[cl, h, g, idx.long()].float()         # [Lc,n,H,NG',G]
        return LP.ungroup_after_decode(q, self.Lb, self.T, self.Tc, self.c)  # [L,n,H,D]

    def save(self, path):
        b = dict(self.blob)
        cbs = {}
        cb = self.cb.detach().cpu().to(torch.float16)
        NG = self.D // (self.c * self.Tc)
        for blk in range(self.T // self.Tc):
            for l in range(self.Lb):
                for hh in range(self.H):
                    for gg in range(self.D // self.c):
                        slot = l + (blk * self.Tc + gg // NG) * self.Lb
                        cbs.setdefault((slot, hh), [None] * NG)[gg % NG] = cb[blk * self.Lb + l, hh, gg].contiguous()
        b["codebooks"] = cbs
        torch.save(b, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / ".." / ".." / ".." / "models" / "Ouro-1.4B"))
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sink", type=int, default=8)
    ap.add_argument("--train", default="kv", choices=("k", "v", "kv"))
    ap.add_argument("--data-cache", default=os.environ.get("CALIB_DATA_DIR", str(HERE / ".." / "work" / "calib_data")))
    ap.add_argument("--calib-data", default="wikitext", choices=("wikitext", "mix", "gpqa_nodiamond", "c4", "slimpajama"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compound", type=int, default=1,
                    help="passes of error compounding in the objective (1 = clean-cache substitution)")
    ap.add_argument("--eval-compound", default=None,
                    help="e.g. 1,2,3: print the src bundle's held-out KL at each R and exit")
    ap.add_argument("--rep-chunk", type=int, default=512,
                    help="tokens per chunk of the encode and of the decode replay in the two-stage backward")
    ap.add_argument("--grad-ckpt", action="store_true", help="needed at 2048 tokens on 24 GB")
    ap.add_argument("--objective", default="joint", choices=("joint", "per-step"),
                    help="joint: KL with every loop step reading the codes (the recipe); per-step "
                         "(ablation, loop-unaware): sum over t of the KL with only loop step t "
                         "quantized, the other steps exact, so the errors of different steps never "
                         "interact; held-out selection uses the same objective")
    # figure instrumentation (figures/ft_kl); defaults leave the recipe and its outputs unchanged
    ap.add_argument("--trace", default=None,
                    help="write a JSON trace: held-out KL every --eval-every train steps and, "
                         "with --step-breakdown, per-loop-step KL at the start and the end")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--step-breakdown", action="store_true",
                    help="held-out KL with only loop step t's slots quantized (others exact), per t")
    ap.add_argument("--no-save", action="store_true", help="do not write the dst bundle")
    # figures/ft_kl panel (b): perplexity of the src bundle under clean-cache substitution
    ap.add_argument("--eval-ppl", type=int, default=0,
                    help="N > 0: perplexity on the first N wikitext-2 TEST windows of --seq-len "
                         "tokens (positions after --sink), bf16 and with the src bundle; then exit")
    ap.add_argument("--ppl-out", default=None, help="append the --eval-ppl result to this JSON")
    a = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 trust_remote_code=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rope = RopeInverter(model.model.rotary_emb)
    cfg = model.config
    n_slots = cfg.num_hidden_layers * (getattr(cfg, "total_ut_steps", 1) or 1)
    cl = a.seq_len
    a.calib_seq_len = cl                      # calib_texts reads the builder's field names
    texts = calib_texts(a, tok, dev)
    if a.grad_ckpt:
        enable_layer_checkpointing(model)
    tr, va = texts[:a.n_calib - a.n_val], texts[a.n_calib - a.n_val:]
    src = ART / a.src
    qk, qv = Coded(src / "codebook.pt", dev), Coded(src / "vq_v_codebook.pt", dev)
    rf = torch.load(src / "v_rotation_sst_r_h_pbr.pt", map_location="cpu", weights_only=False)
    assert rf.get("objective", "identity") == "identity", "V rotation bundles not handled here"
    params = ([qk.cb] if "k" in a.train else []) + ([qv.cb] if "v" in a.train else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    base = make_fisher_cache_cls(type(model(torch.tensor([[1]], device=dev), use_cache=True,
                                            logits_to_keep=1).past_key_values))

    class Rep(base):
        rep = None
        capture = None                        # dict slot -> (k, v) computed BEFORE replacement

        def update(self, k, v, layer_idx, cache_kwargs=None):
            if Rep.capture is not None:
                Rep.capture[layer_idx] = (k.detach(), v.detach())
            if Rep.rep is not None:
                k, v = Rep.rep[0][layer_idx].to(k.dtype), Rep.rep[1][layer_idx].to(v.dtype)
            return super().update(k, v, layer_idx, cache_kwargs)

    cs = rope.cos_sin(torch.arange(cl, device=dev), dev)
    cos, sin = cs[0][0], cs[1][0]

    chunks = [slice(i, min(i + a.rep_chunk, cl)) for i in range(0, cl, a.rep_chunk)]

    def encode(kpost, vv):
        """post-RoPE K / V [L,H,S,D] -> codes and scales under the current codebooks"""
        enc = []
        # per token chunk (every step is per token): full-slot fp32 copies are 1.6 GB on 1.4B
        # and 3.2 GB on 2.6B at 2048 tokens, and unrope makes several
        for sl in chunks:
            kx = rope.unrope(kpost[:, :, sl].float(), (cos[:, sl], sin[:, sl])).permute(0, 2, 1, 3)
            rk, sk_ = qk.to_coded(kx); del kx
            rv, sv_ = qv.to_coded(vv[:, :, sl].float().permute(0, 2, 1, 3))
            enc.append((qk.assign(rk), sk_, qv.assign(rv), sv_)); del rk, rv
        cat1 = lambda xs: None if xs[0] is None else torch.cat(xs, 1)
        return tuple(cat1([e[j] for e in enc]) for j in range(4))

    def rebuild(codes, sl):
        """served K (post-RoPE) / V rows [L,H,s,D] of token slice sl, differentiable in cb"""
        ik, sk, iv, sv = codes
        kr = qk.from_coded(qk.decode(ik[:, sl]), None if sk is None else sk[:, sl]).permute(0, 2, 1, 3)
        kr = kr * cos[:, sl] + rotate_half(kr) * sin[:, sl]                # cos [1,S,D]
        vr = qv.from_coded(qv.decode(iv[:, sl]), None if sv is None else sv[:, sl]).permute(0, 2, 1, 3)
        return kr, vr

    @torch.no_grad()
    def served_cache(codes, ks0, vs0):
        # bf16 per chunk: the same values as casting after the cat, without holding every chunk's
        # fp32 rows at once (2.6B at G = 2: OOM on 24 GB otherwise)
        parts = [tuple(t.to(torch.bfloat16) for t in rebuild(codes, sl)) for sl in chunks]
        kr = torch.cat([p_[0] for p_ in parts], 2)
        vr = torch.cat([p_[1] for p_ in parts], 2)
        kr[:, :, :a.sink] = ks0.to(torch.bfloat16)                              # FP sink prefix
        vr[:, :, :a.sink] = vs0.to(torch.bfloat16)
        return kr, vr

    def run(ids, train, R=None, only_step=None, zero=True, step=True):
        """R = 1: quantise the clean cache (clean-cache substitution). R > 1: R-1 extra passes,
        each re-computing every slot's K/V from the hidden states of a forward that already
        attends to the quantised cache, then quantising those -- the error compounding of
        generated context, which converges to the served fixed point under the causal mask.
        Gradients flow through the last quantisation only."""
        R = R or a.compound
        with torch.no_grad():
            Rep.rep = None
            c1 = Rep(n_slots)
            o = model(ids, past_key_values=c1, use_cache=True)
            lp = F.log_softmax(o.logits[0, :-1].float(), -1)
            kpost = torch.stack([c1.key_cache[i] for i in range(n_slots)])[:, 0]       # [L,H,S,D] bf16
            vv = torch.stack([c1.value_cache[i] for i in range(n_slots)])[:, 0]
            del c1, o
            ks0, vs0 = kpost[:, :, :a.sink].float(), vv[:, :, :a.sink].float()        # exact sink rows
            codes = encode(kpost, vv)
            # only_step: slots of every other loop step keep their exact K/V (slot = t * Lb + l)
            exact = None if only_step is None else (kpost.clone(), vv.clone())
            del kpost, vv
            for _ in range(R - 1):
                kr, vr = served_cache(codes, ks0, vs0)
                Rep.rep, Rep.capture = (kr.unsqueeze(1), vr.unsqueeze(1)), {}
                model(ids, past_key_values=Rep(n_slots), use_cache=True, logits_to_keep=1)
                del kr, vr
                kpost = torch.stack([Rep.capture[i][0] for i in range(n_slots)])[:, 0]
                vv = torch.stack([Rep.capture[i][1] for i in range(n_slots)])[:, 0]
                Rep.rep, Rep.capture = None, None
                codes = encode(kpost, vv)
                del kpost, vv

        # Two-stage backward: the model's backward stops at bf16 leaves holding the served K/V,
        # then the decode is replayed per token chunk with those leaves' grads. Same chain rule
        # as one graph, but the fp32 decode/RoPE intermediates of all L*T slots (3 GB each on
        # Ouro-2.6B at 2048 tokens) are never held at once.
        kr, vr = served_cache(codes, ks0, vs0)
        if only_step is not None:
            keep = torch.ones(n_slots, dtype=torch.bool, device=kr.device)
            keep[only_step * qk.Lb:(only_step + 1) * qk.Lb] = False
            kr[keep], vr[keep] = exact[0][keep].to(kr.dtype), exact[1][keep].to(vr.dtype)
            del exact
        kr.requires_grad_(train); vr.requires_grad_(train)
        with torch.set_grad_enabled(train):
            Rep.rep = (kr.unsqueeze(1), vr.unsqueeze(1))
            c2 = Rep(n_slots)
            o2 = model(ids, past_key_values=c2, use_cache=True)
            lq = F.log_softmax(o2.logits[0, :-1].float(), -1)
            kl = F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)[a.sink:].mean()
            c2.frozen = True                  # checkpoint recompute must reuse, not append
            if train:
                if zero:
                    opt.zero_grad(set_to_none=True)
                kl.backward()
                gk, gv = kr.grad, vr.grad
                gk[:, :, :a.sink] = 0; gv[:, :, :a.sink] = 0                    # sink rows are exact
                if only_step is not None:                                       # so are other steps
                    gk[keep] = 0; gv[keep] = 0
                del o2, lq, c2
                Rep.rep = None
                for sl in chunks:
                    k_, v_ = rebuild(codes, sl)
                    torch.autograd.backward([k_, v_], [gk[:, :, sl].float(), gv[:, :, sl].float()])
                if step:
                    opt.step()
        Rep.rep = None
        return kl.item()

    def val(R=None, only_step=None):
        if only_step is None and a.objective == "per-step":
            return sum(val(R, t) for t in range(qk.T))
        return sum(run(x, False, R, only_step) for x in va) / len(va)

    def train_step(x):
        if a.objective == "joint":
            return run(x, True)
        # per-step: one Adam step on the summed gradients of the T single-step objectives
        return sum(run(x, True, only_step=t, zero=t == 0, step=t == qk.T - 1) for t in range(qk.T))

    trace = {"src": a.src, "lr": a.lr, "n_train": len(tr), "n_val": len(va), "curve": []}

    def breakdown():
        return [val(only_step=t) for t in range(qk.T)]

    @torch.no_grad()
    def nll(ids):
        """summed next-token NLL after the sink, clean and with every slot reading the src
        bundle's reconstruction of the clean cache (the fine-tuning objective's q)"""
        Rep.rep = None
        c1 = Rep(n_slots)
        o = model(ids, past_key_values=c1, use_cache=True)
        tgt = ids[0, 1:]
        lp = F.log_softmax(o.logits[0, :-1].float(), -1).gather(-1, tgt[:, None])[:, 0]
        kpost = torch.stack([c1.key_cache[i] for i in range(n_slots)])[:, 0]
        vv = torch.stack([c1.value_cache[i] for i in range(n_slots)])[:, 0]
        del c1, o
        ks0, vs0 = kpost[:, :, :a.sink].float(), vv[:, :, :a.sink].float()
        codes = encode(kpost, vv)
        del kpost, vv
        kr, vr = served_cache(codes, ks0, vs0)
        Rep.rep = (kr.unsqueeze(1), vr.unsqueeze(1))
        o2 = model(ids, past_key_values=Rep(n_slots), use_cache=True)
        lq = F.log_softmax(o2.logits[0, :-1].float(), -1).gather(-1, tgt[:, None])[:, 0]
        Rep.rep = None
        return -lp[a.sink:].sum().item(), -lq[a.sink:].sum().item(), lp[a.sink:].numel()

    if a.eval_ppl:
        import math
        wt = tok(load_wikitext(a.data_cache, "test"), return_tensors="pt").input_ids
        n = min(a.eval_ppl, wt.shape[1] // cl)
        tot = [0.0, 0.0, 0]
        for i in range(n):
            r = nll(wt[:, i * cl:(i + 1) * cl].to(dev))
            tot = [tot[0] + r[0], tot[1] + r[1], tot[2] + r[2]]
        res = {"bf16": math.exp(tot[0] / tot[2]), "quant": math.exp(tot[1] / tot[2]),
               "windows": n, "seq_len": cl, "sink": a.sink, "tokens": tot[2]}
        print(f"[ppl] {a.src}: bf16 {res['bf16']:.3f}  quantized {res['quant']:.3f}  "
              f"({n} x {cl} tokens)", flush=True)
        if a.ppl_out:
            f = Path(a.ppl_out)
            d = json.loads(f.read_text()) if f.exists() else {}
            d[a.src] = res
            f.write_text(json.dumps(d, indent=1))
        return

    if a.eval_compound:
        for R in map(int, a.eval_compound.split(",")):
            print(f"[compound] R={R}: held-out KL {val(R):.4f}", flush=True)
        return

    best = val()
    print(f"[e2e] start: held-out KL {best:.4f}", flush=True)
    trace["curve"].append((0, best))
    if a.step_breakdown:
        trace["steps_start"] = breakdown()
        print(f"[e2e] per-step KL (start): {[round(x, 4) for x in trace['steps_start']]}", flush=True)
    n_step = 0
    best_state = (qk.cb.detach().clone(), qv.cb.detach().clone())
    t0 = time.time()
    for ep in range(a.epochs):
        trl = 0.0
        for x in tr:
            trl += train_step(x) / len(tr)
            n_step += 1
            if a.eval_every and n_step % a.eval_every == 0 and n_step % len(tr):
                trace["curve"].append((n_step, val()))
        v = val()
        print(f"[e2e] epoch {ep + 1}: train KL {trl:.4f}  held-out KL {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        trace["curve"].append((n_step, v))
        if v < best:
            best = v
            best_state = (qk.cb.detach().clone(), qv.cb.detach().clone())
    with torch.no_grad():
        qk.cb.copy_(best_state[0]); qv.cb.copy_(best_state[1])
    joint = sum(run(x, False) for x in va) / len(va)          # the served objective, for reference
    print(f"[e2e] held-out joint KL of the selected codebooks {joint:.4f}", flush=True)
    if a.trace:
        trace["best"] = best
        if a.step_breakdown:
            trace["steps_end"] = breakdown()
            print(f"[e2e] per-step KL (best): {[round(x, 4) for x in trace['steps_end']]}", flush=True)
        Path(a.trace).write_text(json.dumps(trace, indent=1))
    if a.no_save:
        return
    dst = ART / a.dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("codebook.pt", "vq_v_codebook.pt", "build_info.json"):
            shutil.copy(f, dst / f.name)
    qk.save(dst / "codebook.pt"); qv.save(dst / "vq_v_codebook.pt")
    info = json.loads((src / "build_info.json").read_text())
    info["e2e_finetune"] = {"src": a.src, "lr": a.lr, "epochs": a.epochs, "train": a.train,
                            "held_out_kl": best, "held_out_joint_kl": joint, "compound": a.compound,
                            "objective": "final-loop KL(p_clean || p_quant)" if a.objective == "joint"
                            else "sum_t KL(p_clean || p_quant, only loop step t quantized)"}
    (dst / "build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[e2e] best held-out KL {best:.4f} -> {dst}")


if __name__ == "__main__":
    main()
