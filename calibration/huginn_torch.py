"""Huginn-0125 as a plain functional forward over the HF weights, for calibration and fine-tuning.

Same computation as raven_modeling_minimal.py, written so that every block application exposes its
KV: one full-sequence pass, in which the k / v a block computes ARE that slot's cache for all
positions. A hook `kv(slot, k, v) -> (k, v)` sees each slot's pre-RoPE key and value and may
replace them (fine-tuning feeds reconstructed K / V back in) or record them (calibration).

Conventions shared with the sglang port (python/sglang/srt/models/huginn.py):
  * engine slot order: core slot t*4 + l (t < r, l < 4), then prelude 4r + i, then coda 4r + 2 + i
  * keys / queries in NeoX channel order: channel j < D/2 is HF's channel 2j, j >= D/2 is
    2(j - D/2) + 1, so RoPE is rotate-half with frequency j for the pair (j, j + D/2)
  * s0 from the fixed per-position table (initial_state_table), not a fresh random draw

Calibration (Fisher-weighted k-means input, build_ouro_bundle.py's cache format):

    CUDA_VISIBLE_DEVICES=0 python huginn_torch.py calib --out <calib cache dir>/<name>.pt \
        --n-calib 16 --calib-seq-len 2048 --calib-positions 1024 --calib-data gpqa_nodiamond

writes {k, v, gk, gv}: [n_slots, N, 1, H, P, D] bf16 with the key build_ouro_bundle.calib_key
expects, so `build_ouro_bundle.py --model <huginn> --calib-cache <it>` fits bundles from it.
gk / gv are the gradient of the data cross-entropy w.r.t. each slot's pre-RoPE K / V -- the total
derivative in the unquantised network, as the Ouro capture's retain_grad gives.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
MODEL = HERE / ".." / ".." / ".." / "models" / "huginn-0125"
sys.path.insert(0, str(HERE))


def initial_state_table(n_embd: int, n_pos: int, seed: int) -> torch.Tensor:
    from huginn_ref import initial_state_table as t
    return t(n_embd, n_pos, seed)


def is_huginn(cfg) -> bool:
    return getattr(cfg, "model_type", "") == "huginn_raven"


def huginn_geometry(cfg, steps=None):
    """(Lb, T, n_slots, H, D): core blocks, recurrences, all slots, KV heads, head dim."""
    T = int(steps or os.environ.get("SGLANG_HUGINN_STEPS", 0) or cfg.mean_recurrence)
    Lb = cfg.n_layers_in_recurrent_block
    return Lb, T, Lb * T + cfg.n_layers_in_prelude + cfg.n_layers_in_coda, cfg.n_heads, cfg.n_embd // cfg.n_heads


class HuginnF:
    def __init__(self, path=MODEL, steps=None, device="cuda", dtype=torch.bfloat16, seed=0):
        from transformers import AutoModelForCausalLM
        self.hf = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True,
                                                       dtype=dtype).to(device).eval()
        for p in self.hf.parameters():
            p.requires_grad_(False)
        cfg = self.cfg = self.hf.config
        self.Lb, self.T, self.n_slots, self.H, self.D = huginn_geometry(cfg, steps)
        self.P, self.C = cfg.n_layers_in_prelude, cfg.n_layers_in_coda
        self.device, self.dtype = device, dtype
        D = self.D
        self.perm = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)]).to(device)
        inv = 1.0 / (cfg.rope_base ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        ang = torch.outer(torch.arange(cfg.block_size, dtype=torch.float32), inv)       # [pos, D/2]
        self.cos, self.sin = ang.cos().to(device), ang.sin().to(device)
        self.s0 = initial_state_table(cfg.n_embd, cfg.block_size, seed).to(device)
        self.emb_scale = math.sqrt(cfg.n_embd)
        tr = self.hf.transformer
        self.blocks = ([(tr.core_block[l], t * self.Lb + l) for t in range(self.T) for l in range(self.Lb)])

    # slot bookkeeping ------------------------------------------------------------------------
    def prelude_slot(self, i):
        return self.Lb * self.T + i

    def coda_slot(self, i):
        return self.Lb * self.T + self.P + i

    def rope(self, x, S):
        """x [B,S,H,D] NeoX order -> rotated (rotate-half, frequency j on (j, j + D/2))."""
        h = self.D // 2
        c, s = self.cos[:S, None, :], self.sin[:S, None, :]
        x1, x2 = x[..., :h].float(), x[..., h:].float()
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)

    def unrope(self, x, pos):
        """inverse rotation of [..., P, D] rows at absolute positions `pos` (orthogonal)."""
        h = self.D // 2
        c, s = self.cos[pos], self.sin[pos]
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c + x2 * s, x2 * c - x1 * s], -1)

    # forward ---------------------------------------------------------------------------------
    def block(self, blk, x, slot, kv, qh=None):
        B, S, E = x.shape
        H, D = self.H, self.D
        a = blk.attn
        q, k, v = a.Wqkv(blk.norm_1(x)).split(a.chunks, dim=2)
        q = q.view(B, S, H, D)
        k = k.view(B, S, H, D)
        v = v.view(B, S, H, D)
        qb, kb = a.qk_bias.split(1, dim=0)
        q, k = (q + qb).to(q.dtype), (k + kb).to(q.dtype)
        q, k = q[..., self.perm], k[..., self.perm]         # HF interleaved -> NeoX order
        if kv is not None:
            k, v = kv(slot, k, v)
        q, k = self.rope(q, S), self.rope(k, S)
        if qh is not None:
            qh(slot, q)                                       # post-RoPE query [B,S,H,D], NeoX order
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           is_causal=True)
        y = a.proj(y.transpose(1, 2).reshape(B, S, E))
        x = blk.norm_2(y + x)
        return blk.norm_4(blk.mlp(blk.norm_3(x)) + x)

    def _run(self, blk, x, slot, kv, ckpt, qh=None):
        if ckpt:
            return checkpoint(self.block, blk, x, slot, kv, qh, use_reentrant=False)
        return self.block(blk, x, slot, kv, qh)

    def forward(self, ids, kv=None, ckpt=False, embeds_grad=False, qh=None):
        """ids [B,S] -> logits [B,S,V] fp32. kv: optional hook (slot, k, v) -> (k, v)."""
        tr = self.hf.transformer
        S = ids.shape[1]
        e = tr.wte(ids) * self.emb_scale
        if embeds_grad:
            e = e.detach().requires_grad_()       # the graph for KV gradients, weights frozen
        for i, blk in enumerate(tr.prelude):
            e = self._run(blk, e, self.prelude_slot(i), kv, ckpt, qh)
        x = self.s0[:S].to(e.dtype).expand_as(e)
        for t in range(self.T):
            x = tr.adapter(torch.cat([x, e], dim=-1))
            for l, blk in enumerate(tr.core_block):
                x = self._run(blk, x, t * self.Lb + l, kv, ckpt, qh)
        x = tr.ln_f(x)
        for i, blk in enumerate(tr.coda):
            x = self._run(blk, x, self.coda_slot(i), kv, ckpt, qh)
        x = tr.ln_f(x)
        return self.hf.lm_head(x).float()


# --------------------------------------------------------------------------------------------
def calibrate(args):
    from transformers import AutoTokenizer
    from build_ouro_bundle import CALIB_FIELDS, calib_key, calib_texts
    dev = "cuda"
    m = HuginnF(args.model, steps=args.steps, device=dev)
    tok = AutoTokenizer.from_pretrained(args.model)
    texts = calib_texts(args, tok, dev)
    P = min(args.calib_positions, args.calib_seq_len)
    buf = {f: torch.empty(m.n_slots, len(texts), 1, m.H, P, m.D, dtype=torch.bfloat16)
           for f in CALIB_FIELDS}
    print(f"[calib] huginn r={m.T}: {m.n_slots} slots, buffer {4 * buf['k'].nbytes / 2**30:.1f} GiB "
          f"({len(texts)} seq x {P} of {args.calib_seq_len} positions)", flush=True)
    t0 = time.time()
    for n, ids in enumerate(texts):
        S = ids.shape[1]
        keep = (torch.randperm(S, generator=torch.Generator(device=dev).manual_seed(args.seed + n),
                               device=dev)[:P].sort().values if S > P else torch.arange(S, device=dev))

        def kv(slot, k, v, n=n):
            # rows [1,S,H,D] -> buffer [1,H,P,D] at the kept positions; gradients via tensor hooks
            with torch.no_grad():
                buf["k"][slot, n].copy_(k[:, keep].transpose(1, 2).cpu())
                buf["v"][slot, n].copy_(v[:, keep].transpose(1, 2).cpu())
            if k.requires_grad:
                def save(field, s=slot):
                    def h(g):                     # record only: a hook returning a tensor replaces the grad
                        buf[field][s, n].copy_(g[:, keep].transpose(1, 2).cpu())
                    return h
                k.register_hook(save("gk"))
                v.register_hook(save("gv"))
            return k, v

        lg = m.forward(ids, kv=kv, ckpt=True, embeds_grad=True)
        loss = F.cross_entropy(lg[0, :-1], ids[0, 1:])
        loss.backward()
        del lg, loss
        torch.cuda.empty_cache()
        print(f"    calib {n + 1}/{len(texts)}: {S} tok -> {len(keep)} positions "
              f"({time.time() - t0:.0f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    key = calib_key(args, args.model, m.n_slots)
    key["huginn_steps"] = m.T
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    torch.save({"key": json.dumps(key), **buf}, tmp)
    tmp.rename(out)
    print(f"[calib] saved -> {out}")


def check(args):
    """HuginnF vs the HF remote code (s0 patched to the same table): last-position logits."""
    import types
    m = HuginnF(args.model, steps=args.steps)
    hf = m.hf
    tab = m.s0

    def init_state(self, input_embeds, scale=1.0):
        return tab[:input_embeds.shape[1]].to(input_embeds.dtype)[None].expand_as(input_embeds).clone()

    hf.initialize_state = types.MethodType(init_state, hf)
    ids = torch.randint(0, 65000, (1, 96), device="cuda", generator=torch.Generator(device="cuda").manual_seed(0))
    with torch.no_grad():
        a = m.forward(ids)[0]
        b = hf(ids, num_steps=m.T).logits[0].float()
    err = (a - b).abs().max().item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"[check] max |logit diff| {err:.4f} (logit scale {b.abs().max().item():.1f}); argmax agree {agree:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("calib", "check"))
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--calib-seq-len", type=int, default=2048)
    ap.add_argument("--calib-positions", type=int, default=1024)
    ap.add_argument("--calib-data", default="gpqa_nodiamond")
    ap.add_argument("--data-cache", default=os.environ.get("CALIB_DATA_DIR", str(HERE / ".." / "work" / "calib_data")))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.calib_target = "ce"
    (calibrate if a.cmd == "calib" else check)(a)


if __name__ == "__main__":
    main()
