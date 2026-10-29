"""Loop-aware codebook fine-tuning (and held-out KL evaluation) for Huginn-0125 bundles.

The objective and protocol are finetune_codebook.py's (method.tex, Sec. loop-aware FT):
KL(p_clean || p_quant) of the final output, every slot's cache replaced by its reconstruction
from the codes of the clean cache, sink positions exact, Adam on the entries, assignments
recomputed from the current entries at every step, best held-out epoch kept.

What differs is the plumbing, because Huginn has 132 slots of 55 x 96:
  * the model is huginn_torch.HuginnF (one full-sequence pass; a hook sees each slot's k / v)
  * a bundle is viewed as CODE LAYERS. A code layer is a set of Tc member slots sharing one
    codebook of width G = c * Tc (entry position i*Tc + m = channel i of member m). Loop-coupled
    core blocks are Tc-slot code layers; slots without a loop axis (Huginn's prelude / coda,
    every slot of a CQ / TaSQ bundle) are 1-slot code layers with c = G.
  * clean K / V go to CPU and are encoded one code layer at a time; in the quantised pass each
    slot is decoded inside the hook from the only entry coordinates it reads ([..., m::Tc]),
    so no [slots, tokens, H, D] reconstruction is ever materialised.

    CUDA_VISIBLE_DEVICES=0 python huginn_ft.py --src huginn_3p5b_loopvq8b_g32_lvgq --dst huginn_3p5b_loopvq8b_g32_lvgqe2e
    CUDA_VISIBLE_DEVICES=0 python huginn_ft.py --src huginn_3p5b_cq8b_g32_gq --eval-only
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from build_ouro_bundle import calib_texts
from huginn_torch import MODEL, HuginnF

HERE = Path(__file__).resolve().parent
ART = HERE / ".." / "experiments" / "artifacts"


class Book(torch.nn.Module):
    """One bundle file (K or V) as trainable code layers."""

    def __init__(self, path, device):
        super().__init__()
        b = torch.load(path, map_location="cpu", weights_only=False)
        self.blob = b
        L, H, D, _ = b["forward"].shape
        self.L, self.H, self.D = L, H, D
        self.ident = bool(torch.equal(b["forward"], torch.eye(D).expand_as(b["forward"]))) \
            and not bool(b["mean"].any())
        self.fwd = None if self.ident else b["forward"].float().to(device)
        self.inv = None if self.ident else b["inverse"].float().to(device)
        self.norm = bool(b.get("pool_heads_scale", False))
        self.cw = b["chan_weight"].float().to(device) if b.get("chan_weight") is not None else None
        assert b.get("loop_scale") is None, "loop_scale bundles are not handled here"
        cbs = b["codebooks"]
        G = cbs[(0, 0)][0].shape[1]
        NG = D // G                                           # bundle groups per slot
        lc = b.get("loop_coupled")
        self.layers = []                                      # [(members, c)]
        ents = []
        if lc:
            Lb, Tc, c = lc["base_layers"], lc["loops"], lc["channels"]
            plain = list(lc.get("plain_slots", []))
            T = (L - len(plain)) // Lb
            for blk in range(T // Tc):
                for l in range(Lb):
                    members = [l + (blk * Tc + m) * Lb for m in range(Tc)]
                    # inverse of ouro_cq.loop_slot_map: coupled group g sits at bundle slot
                    # l + (blk*Tc + g // NG) * Lb, group g % NG (loopvq_pack.build_coupled_codebook)
                    e = torch.stack([torch.stack([cbs[(l + (blk * Tc + g // NG) * Lb, h)][g % NG]
                                                  for g in range(D // c)]) for h in range(H)])
                    self.layers.append((members, c)); ents.append(e)       # [H, D/c, K, G]
        else:
            plain = list(range(L))
        for s in plain:
            e = torch.stack([torch.stack(cbs[(s, h)]) for h in range(H)])  # [H, D/G, K, G]
            self.layers.append(([s], G)); ents.append(e)
        self.cb = torch.nn.ParameterList([torch.nn.Parameter(e.float().to(device)) for e in ents])
        self.where = {}                                       # slot -> (code layer, member idx)
        for i, (members, _) in enumerate(self.layers):
            for m, s in enumerate(members):
                self.where[s] = (i, m)
        assert sorted(self.where) == list(range(L)), "bundle does not cover every slot"
        self.G = G

    # coordinate maps (Coded's, per slot) ------------------------------------------------------
    def to_coded(self, x, s):
        """[n,H,D] natural rows of slot s -> (coded rows, pooled scale [n,1,1] or None)."""
        r = x if self.ident else torch.einsum("nhd,hde->nhe", x, self.fwd[s])
        sc = None
        if self.cw is not None:
            r = r * self.cw[s]
        if self.norm:
            sc = r.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
            r = r / sc
        return r, sc

    def from_coded(self, r, sc, s):
        if sc is not None:
            r = r * sc
        if self.cw is not None:
            r = r / self.cw[s]
        return r if self.ident else torch.einsum("nhe,hed->nhd", r, self.inv[s])

    @torch.no_grad()
    def assign(self, i, rows):
        """code layer i, rows = coded [n,H,D] of each member -> codes [n,H,D/c] (nearest entry)."""
        members, c = self.layers[i]
        Tc = len(members)
        n, H, D = rows[0].shape
        z = torch.stack(rows, -1).view(n, H, D // c, c * Tc)   # entry pos i*Tc + m
        cb = self.cb[i].detach()
        out = torch.empty(n, H, D // c, dtype=torch.long, device=z.device)
        for h in range(H):
            for j in range(0, n, 512):
                out[j:j + 512, h] = torch.cdist(z[j:j + 512, h].transpose(0, 1), cb[h]).argmin(-1).T
        return out

    def decode(self, s, codes):
        """slot s, its code layer's codes [n,H,D/c] -> coded rows [n,H,D], differentiable."""
        i, m = self.where[s]
        Tc, c = len(self.layers[i][0]), self.layers[i][1]
        part = self.cb[i][..., m::Tc]                          # [H, D/c, K, c]: what slot s reads
        n, H, NG = codes.shape
        h = torch.arange(H, device=codes.device).view(1, H, 1)
        g = torch.arange(NG, device=codes.device).view(1, 1, NG)
        return part[h, g, codes].reshape(n, H, NG * c)

    def save(self, path):
        b = dict(self.blob)
        cbs = {}
        D, G = self.D, self.G
        NG = D // G
        lc = b.get("loop_coupled")
        for i, (members, c) in enumerate(self.layers):
            e = self.cb[i].detach().cpu().to(torch.float16)
            if len(members) == 1 and c == G:
                for h in range(self.H):
                    cbs[(members[0], h)] = [e[h, g].contiguous() for g in range(NG)]
                continue
            Lb, Tc = lc["base_layers"], len(members)
            l, blk = members[0] % Lb, members[0] // Lb // Tc
            for h in range(self.H):
                for g in range(D // c):
                    cbs.setdefault((l + (blk * Tc + g // NG) * Lb, h), [None] * NG)[g % NG] = e[h, g].contiguous()
        assert len(cbs) == self.L * self.H and all(t is not None for v in cbs.values() for t in v)
        b["codebooks"] = cbs
        torch.save(b, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--eval-only", action="store_true", help="print the src bundle's held-out KL")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sink", type=int, default=8)
    ap.add_argument("--train", default="kv", choices=("k", "v", "kv"))
    ap.add_argument("--data-cache", default=os.environ.get("CALIB_DATA_DIR", str(HERE / ".." / "work" / "calib_data")))
    ap.add_argument("--calib-data", default="gpqa_nodiamond")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert a.eval_only or a.dst, "--dst (or --eval-only)"
    dev = "cuda"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    m = HuginnF(a.model, device=dev)
    a.calib_seq_len = a.seq_len
    texts = calib_texts(a, tok, dev)
    tr, va = texts[:a.n_calib - a.n_val], texts[a.n_calib - a.n_val:]
    src = ART / a.src
    bk, bv = Book(src / "codebook.pt", dev), Book(src / "vq_v_codebook.pt", dev)
    assert bk.L == m.n_slots, (bk.L, m.n_slots)
    params = (list(bk.cb) if "k" in a.train else []) + (list(bv.cb) if "v" in a.train else [])
    opt = torch.optim.Adam(params, lr=a.lr) if not a.eval_only else None

    def run(ids, train):
        S = ids.shape[1]
        cap = {}
        with torch.no_grad():
            def grab(slot, k, v):
                cap[slot] = (k[0].to("cpu", torch.bfloat16), v[0].to("cpu", torch.bfloat16))   # [S,H,D]
                return k, v
            lp = F.log_softmax(m.forward(ids, kv=grab), -1)[0, :-1]
            codes, sink = {}, {}
            for book, j, store in ((bk, 0, "k"), (bv, 1, "v")):
                cs, ss = [], {}
                for i, (members, _) in enumerate(book.layers):
                    rows = []
                    for s in members:
                        r, sc = book.to_coded(cap[s][j].to(dev).float(), s)
                        rows.append(r); ss[s] = sc
                    cs.append(book.assign(i, rows))
                codes[store] = (cs, ss)
            for s, (k, v) in cap.items():
                sink[s] = (k[:a.sink].to(dev), v[:a.sink].to(dev))
            del cap

        def rep(slot, k, v):
            out = []
            for book, store, x in ((bk, "k", k), (bv, "v", v)):
                cs, ss = codes[store]
                i, _ = book.where[slot]
                r = book.from_coded(book.decode(slot, cs[i]), ss[slot], slot).to(x.dtype)   # [S,H,D]
                r = torch.cat([sink[slot][0 if store == "k" else 1], r[a.sink:]], 0)
                out.append(r.unsqueeze(0))
            return out[0], out[1]

        with torch.set_grad_enabled(train):
            lq = F.log_softmax(m.forward(ids, kv=rep, ckpt=train), -1)[0, :-1]
            kl = F.kl_div(lq, lp, log_target=True, reduction="none").sum(-1)[a.sink:].mean()
            if train:
                opt.zero_grad(set_to_none=True)
                kl.backward()
                opt.step()
        return kl.item()

    def val():
        return sum(run(x, False) for x in va) / len(va)

    t0 = time.time()
    best = val()
    print(f"[e2e] {a.src}: held-out KL {best:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if a.eval_only:
        return
    best_state = [p.detach().clone() for p in params]
    for ep in range(a.epochs):
        trl = sum(run(x, True) for x in tr) / len(tr)
        v = val()
        print(f"[e2e] epoch {ep + 1}: train KL {trl:.4f}  held-out KL {v:.4f}  ({time.time() - t0:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
        if v < best:
            best = v
            best_state = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, s in zip(params, best_state):
            p.copy_(s)
    dst = ART / a.dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("codebook.pt", "vq_v_codebook.pt", "build_info.json"):
            shutil.copy(f, dst / f.name)
    bk.save(dst / "codebook.pt"); bv.save(dst / "vq_v_codebook.pt")
    info = json.loads((src / "build_info.json").read_text())
    info["e2e_finetune"] = {"src": a.src, "lr": a.lr, "epochs": a.epochs, "train": a.train,
                            "held_out_kl": best, "objective": "final KL(p_clean || p_quant)",
                            "tool": "calibration/huginn_ft.py"}
    (dst / "build_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"[e2e] best held-out KL {best:.4f} -> {dst}")


if __name__ == "__main__":
    main()
