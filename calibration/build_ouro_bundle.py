"""Codebook bundle builder for looped models (Ouro, Huginn): CQ, TaSQ, and LoopVQ.

Fits the codebooks directly (Fisher-weighted k-means++ on pre-RoPE keys, one codebook per channel
group) and writes them in the bundle schema of the serving engine.

The KV slots of Ouro are slot = ut*L + layer, L*T of them. Placing the slots on the bundle's
"layer" axis as they are gives the per-loop-step baseline (CQ, TaSQ): an independent codebook for
every loop step. --method loopvq couples Tc consecutive loop steps of one layer instead.

Bundle schema (python/sglang/srt/mem_cache/vq_codebook.py):
    forward / inverse : [L,H,D,D]   identity (CQ has no transform)
    mean              : [L,H,D]     0
    codebooks         : {(l,h): [ [K,G] x (D//G) ]}
    bounds            : [(g*G, (g+1)*G, bits)]
    pertoken_norm     : False

K groups are contiguous channels within a head, and so are V groups: with wide-G serving the pool
encodes V without reordering (the strided `_vq_v_perm` is the fp8 G=4 path only), so the codebooks
are fitted in that coordinate system (LOOPVQ_V_LAYOUT=strided restores the strided layout).
"""
import argparse, json, os, time, urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ouro_cq import GROUPERS, RopeInverter, group_loop, loop_slot_map, weighted_kmeans

WIKI = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1"


def load_wikitext(cache_dir, split):
    import pandas as pd
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"wikitext2-{split}.parquet"
    if not f.exists():
        urllib.request.urlretrieve(f"{WIKI}/{split}-00000-of-00001.parquet", f)
    return "\n\n".join(pd.read_parquet(f).text.tolist())


class Calib:
    """Per-slot (pre-RoPE K, V) of one calibration sequence and their gradients, on the CPU.

    Each field is a sequence indexed by slot returning [1,H,P,D], a view of the shared buffer."""
    __slots__ = ("k", "v", "gk", "gv")

    def __init__(self, k, v, gk, gv):
        self.k, self.v, self.gk, self.gv = k, v, gk, gv


CALIB_FIELDS = ("k", "v", "gk", "gv")
UNIFORM_W = False      # --kmeans-weight uniform: plain k-means, no Fisher (set in main)
ATTN = None            # --attn-stats (collect_attn_stats.py): {"q_energy", "att_mass"}
ATTN_W = False         # --kmeans-weight attn: per-vector weight = attention mass received
LS_POW = 1.0           # --loop-scale-pow: s = importance ** (pow / 2)
FISHER_NORM = False    # --fisher-norm-fix: Eq.6 weight of a normalised bundle from (g*sigma)^2
FISHER_POOL = "none"   # --fisher-pool heads: token weight = its Eq.6 weight averaged over heads/groups
FISHER_POW = 1.0       # --fisher-pow: tempered Eq.6 weight w**pow
LOOP_WEIGHT = "sum"    # --loop-weight: combine a coupled vector's per-loop Fisher (see fit_loop_slot)
QW = None              # --k-qweight (tasq_qweights.py): TaSQ's W, [n_slots, H, D] natural channels


def k_transform(x, slot, norm):
    """TaSQ's key order before grouping, natural channels: k^w = w * k  [W], then
    k^wn = k^w / RMS_pooled(k^w)  [N]. Returns (k^wn, s) with s [1,1,P,1] or None."""
    if QW is not None:
        x = x * QW[slot].to(x.device).view(1, x.shape[1], 1, x.shape[-1])
    if not norm:
        return x, None
    s = x.pow(2).mean(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-8)
    return x / s, s


def calib_views(buf):
    """buf[f] : [n_slots, N, 1, H, P, D] -> one Calib per sample (no copy)."""
    n_slots, N = buf["k"].shape[:2]
    return [Calib(*([buf[f][i, n] for i in range(n_slots)] for f in CALIB_FIELDS))
            for n in range(N)]


def calib_key(args, model_path, n_slots):
    key = {"model": str(Path(model_path).resolve()), "n_calib": args.n_calib,
           "seq_len": args.calib_seq_len, "positions": args.calib_positions,
           "seed": args.seed, "n_slots": n_slots}
    if getattr(args, "calib_target", "ce") != "ce":     # "ce" keeps the original caches valid
        key["target"] = args.calib_target
    if getattr(args, "calib_data", "wikitext") != "wikitext":
        key["data"] = args.calib_data
    return key


def calib_texts(args, tok, dev):
    """--n-calib sequences of --calib-seq-len tokens.

    wikitext  wikitext-2 train (the original calibration).
    mix       one third each: wikitext-2 train; GSM8K TRAIN worked solutions in the eval's own
              "Q: ... A: ..." completion format (the test split stays unseen); Python source
              from the interpreter's standard library (no HumanEval / MBPP overlap) -- the
              served workloads are math reasoning and code, not encyclopaedia prose.
    c4 / slimpajama  shuffled documents (seed) of the C4 en train shard 0 (HF cache) / the
              SlimPajama-6B train shard 0 (<data-cache>/slimpajama6b-train-00000.parquet),
              joined by blank lines -- the calibration-set sensitivity ablation.
    """
    cl = args.calib_seq_len
    if args.calib_data in ("c4", "slimpajama"):
        import random
        if args.calib_data == "c4":
            import gzip, glob as _g
            f = _g.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--allenai--c4/snapshots/"
                                           "*/en/c4-train.00000-of-01024.json.gz"))[0]
            docs = [json.loads(l)["text"] for l in gzip.open(f, "rt")]
        else:
            import pandas as pd
            docs = pd.read_parquet(Path(args.data_cache) / "slimpajama6b-train-00000.parquet",
                                   columns=["text"])["text"].tolist()
        random.Random(args.seed).shuffle(docs)
        need, parts, n = args.n_calib * cl, [], 0
        for d in docs:                      # tokenize only as many documents as needed
            parts.append(d.strip()); n += len(d) // 3
            if n > 2 * need:
                break
        ids = tok("\n\n".join(parts), return_tensors="pt").input_ids
        assert ids.shape[1] >= need, f"{args.calib_data}: {ids.shape[1]} tokens < {need}"
        print(f"[calib] {args.calib_data}: {len(parts)} documents, {ids.shape[1]} tokens")
        return [ids[:, i * cl:(i + 1) * cl].to(dev) for i in range(args.n_calib)]
    if args.calib_data == "gpqa_nodiamond":
        # GPQA main minus every diamond question (Record ID): graduate-level science questions
        # with their choices, answer and explanation; diamond stays unseen as an eval set
        import random
        from datasets import load_dataset
        main_ = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        dia = {x["Record ID"] for x in load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")}
        rows = [x for x in main_ if x["Record ID"] not in dia]
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        docs = []
        for x in rows:
            opts = [x["Correct Answer"], x["Incorrect Answer 1"], x["Incorrect Answer 2"], x["Incorrect Answer 3"]]
            opts = [o.strip() for o in opts]
            order = list(range(4)); rng.shuffle(order)
            letters = "ABCD"
            ch = "\n".join(f"({letters[i]}) {opts[j]}" for i, j in enumerate(order))
            ans = letters[order.index(0)]
            docs.append(f"Question: {x['Question'].strip()}\n\nChoices:\n{ch}\n\n"
                        f"Answer: ({ans}) {opts[0]}\n\nExplanation: {x['Explanation'].strip()}")
        ids = tok("\n\n".join(docs), return_tensors="pt").input_ids
        assert ids.shape[1] >= args.n_calib * cl, f"GPQA main-diamond has {ids.shape[1]} tokens"
        print(f"[calib] gpqa main - diamond: {len(rows)} questions, {ids.shape[1]} tokens")
        return [ids[:, i * cl:(i + 1) * cl].to(dev) for i in range(args.n_calib)]
    wiki = tok(load_wikitext(args.data_cache, "train")[:4_000_000], return_tensors="pt").input_ids
    if args.calib_data == "wikitext":
        return [wiki[:, i * cl:(i + 1) * cl].to(dev) for i in range(args.n_calib)]
    import glob, random, sysconfig
    from datasets import load_dataset
    g = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(args.seed)
    idx = list(range(len(g))); rng.shuffle(idx)
    math_txt = "\n\n".join(f"Q: {g[i]['question']}\nA: {g[i]['answer']}" for i in idx[:3000])
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    rng.shuffle(files)
    code_txt = "\n\n".join(open(f, errors="ignore").read() for f in files[:200])
    streams = [wiki, tok(math_txt, return_tensors="pt").input_ids, tok(code_txt, return_tensors="pt").input_ids]
    out, used = [], [0, 0, 0]
    for i in range(args.n_calib):
        k = i % 3
        st = streams[k]
        assert (used[k] + 1) * cl <= st.shape[1], f"calib source {k} too short"
        out.append(st[:, used[k] * cl:(used[k] + 1) * cl].to(dev)); used[k] += 1
    return out


def calib_loss(o, ids, target, seed):
    """The scalar whose gradient w.r.t. every KV slot becomes the Fisher weight.

    ce        CE on the data's next token -- the empirical Fisher (CQ's choice).
    kl        CE on a label drawn from the model's OWN final-loop distribution. Its gradient's
              second moment is the true Fisher of p_model: the second-order term of
              KL(p || p_quantised) at the final loop's output, which is the error that matters
              when an early loop's KV is quantised -- the data label adds noise unrelated to how
              far the output distribution moves.
    logit_rp  random Gaussian projection of the final-loop logits (Hutchinson): E[g g^T] =
              J^T J, the plain size of the logit change, not weighted by p.
    """
    lg = o.logits[0, :-1].float()
    if target == "ce":
        return F.cross_entropy(lg, ids[0, 1:])
    gen = torch.Generator(device=lg.device).manual_seed(seed)
    if target == "kl":
        with torch.no_grad():
            y = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(-1)
        return F.cross_entropy(lg, y)
    if target == "logit_rp":
        r = torch.randn(lg.shape, generator=gen, device=lg.device)
        return (lg * r).sum() / lg.shape[0] / lg.shape[1] ** 0.5
    raise ValueError(target)


def make_fisher_cache_cls(base_cls):
    class FisherCache(base_cls):
        def get_mask_sizes(self, cache_position, layer_idx):
            # Ouro's UniversalTransformerCache does not fill self.layers, so the default
            # get_mask_sizes of transformers>=4.56 ignores the past cache.
            return self.get_seq_length() + cache_position.shape[0], 0

        def update(self, k, v, layer_idx, cache_kwargs=None):
            if getattr(self, "frozen", False):
                # gradient-checkpoint recompute during backward: hand back the tensors the forward
                # used (appending would grow the cache and lose the retained grads)
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            if k.requires_grad:
                k.retain_grad(); v.retain_grad()
            return super().update(k, v, layer_idx, cache_kwargs)
    return FisherCache


def enable_layer_checkpointing(model):
    """Non-reentrant activation checkpointing per decoder layer, keeping the cache.

    Ouro runs every layer T times, so a backward over a 2048-token sequence holds 96 layer
    passes of activations and does not fit 24 GB. transformers' own GradientCheckpointingLayer
    path drops past_key_values, which the Fisher capture needs, so each layer's forward is
    wrapped directly; FisherCache.frozen makes the recompute reuse the forward's K/V."""
    from torch.utils.checkpoint import checkpoint
    for layer in model.model.layers:
        orig = layer.forward
        def fwd(*args, _orig=orig, **kw):
            return checkpoint(_orig, *args, use_reentrant=False, **kw)
        layer.forward = fwd


def calibrate(model, tok, texts, n_slots, rope, n_pos, seed, device, H, D, target="ce"):
    """Fill the shared slot-major buffer {k,v,gk,gv}: [n_slots, N, 1, H, P, D] bf16.

    Samples are written straight into the preallocated buffer rather than stacked: for the 2.6B
    model it is ~48 GB, and stacking a list first would double the peak."""
    probe = model(torch.tensor([[1]], device=device), use_cache=True,
                  logits_to_keep=1).past_key_values
    FisherCache = make_fisher_cache_cls(type(probe))
    del probe; torch.cuda.empty_cache()

    lens = {t.shape[1] for t in texts}
    assert len(lens) == 1, f"calibration sequences differ in length: {lens}"
    P = min(lens.pop(), n_pos)
    buf = {f: torch.empty(n_slots, len(texts), 1, H, P, D, dtype=torch.bfloat16)
           for f in CALIB_FIELDS}
    print(f"[calib] buffer {4 * buf['k'].nbytes / 2**30:.1f} GiB "
          f"({n_slots} slots x {len(texts)} seq x {P} pos)", flush=True)

    for n, ids in enumerate(texts):
        cache = FisherCache(n_slots)
        o = None
        if target in ("loop_exits", "exit_pdf", "last2_exits"):
            # Ouro exits after every loop step (lm_head on each step's normed hidden state; it is
            # trained on the exit-distribution-weighted sum of these). Fisher of
            #   loop_exits   the mean exit CE: how much a KV coordinate moves EVERY step's prediction
            #   exit_pdf     the expected CE under the model's own exit distribution (its training
            #                objective; the gate probabilities are weights, not differentiated)
            #   last2_exits  the mean CE of the last two steps
            _, hs, gates = model.model(ids, past_key_values=cache, use_cache=True)
            ces = [F.cross_entropy(model.lm_head(h)[0, :-1].float(), ids[0, 1:], reduction="none") for h in hs]
            if target == "loop_exits":
                loss = sum(c.mean() for c in ces) / len(ces)
            elif target == "last2_exits":
                loss = sum(c.mean() for c in ces[-2:]) / 2
            else:
                rem, pdf = None, []
                for i, gt in enumerate(gates):
                    lam = torch.sigmoid(gt.squeeze(-1).float().detach())[0, :-1]
                    rem = torch.ones_like(lam) if rem is None else rem
                    p_i = lam * rem if i < len(gates) - 1 else rem
                    if i < len(gates) - 1:
                        rem = rem * (1 - lam)
                    pdf.append(p_i)
                loss = sum((p * c).mean() for p, c in zip(pdf, ces))
        else:
            o = model(ids, past_key_values=cache, use_cache=True)
            loss = calib_loss(o, ids, target, seed + 7919 * n)
        cache.frozen = True
        loss.backward()
        del o

        S = cache.get_seq_length()
        keep = (torch.randperm(S, generator=torch.Generator(device=device).manual_seed(seed + n),
                               device=device)[:n_pos].sort().values
                if S > n_pos else torch.arange(S, device=device))
        cs = rope.cos_sin(keep, device)      # angles that undo RoPE (at the actual positions)

        with torch.no_grad():
            for i in range(n_slots):
                # detach is required: key_cache[i] is a graph tensor kept with retain_grad for
                # the backward pass; without it every calibration sample would hold on to the
                # whole autograd graph (memory blowup, and out= ops fail).
                kt = cache.key_cache[i][:, :, keep, :].detach().float()
                gt = cache.key_cache[i].grad[:, :, keep, :].detach().float()
                # move K and its gradient to the same pre-RoPE coordinates (RoPE is orthogonal).
                buf["k"][i, n].copy_(rope.unrope(kt, cs))
                buf["gk"][i, n].copy_(rope.unrope(gt, cs))
                buf["v"][i, n].copy_(cache.value_cache[i][:, :, keep, :].detach())
                buf["gv"][i, n].copy_(cache.value_cache[i].grad[:, :, keep, :].detach())
        model.zero_grad(set_to_none=True)
        del cache
        torch.cuda.empty_cache()
        print(f"    calib {n+1}/{len(texts)}: {S} tok -> {len(keep)} positions", flush=True)
    return buf


def load_or_calibrate(args, n_slots, H, D, dev):
    """Build the calibration buffer, or mmap it from --calib-cache.

    The calibration does not depend on G or the method. Fitting several bundles in parallel would
    otherwise hold one copy of the buffer (2.6B: ~48 GB) in RAM per process, ~380 GB for eight,
    and kill the node. Saved once and opened with mmap, the page cache is shared.
    """
    key = calib_key(args, args.model, n_slots)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    from huginn_torch import huginn_geometry, is_huginn
    if is_huginn(cfg):
        key["huginn_steps"] = huginn_geometry(cfg)[1]
    cpath = Path(args.calib_cache) if args.calib_cache else None
    if cpath is not None and cpath.exists():
        blob = torch.load(cpath, mmap=True, weights_only=True)
        got = json.loads(blob["key"])
        assert got == key, f"calibration cache mismatch {cpath}:\n  cache={got}\n  want ={key}"
        print(f"[calib] mmap {cpath}", flush=True)
        return {f: blob[f] for f in CALIB_FIELDS}

    assert not is_huginn(cfg), f"Huginn calibration: run huginn_torch.py calib --out {cpath}"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    rope = RopeInverter(model.model.rotary_emb)

    cl = args.calib_seq_len
    texts = calib_texts(args, tok, dev)
    print(f"[calib] {args.n_calib} x {cl} tok (wikitext-2 train) -> "
          f"{args.calib_positions} positions each")
    t0 = time.time()
    if args.grad_ckpt:
        enable_layer_checkpointing(model)
    buf = calibrate(model, tok, texts, n_slots, rope, args.calib_positions, args.seed, dev, H, D,
                    target=args.calib_target)
    print(f"[calib] done in {time.time()-t0:.0f}s")
    del model, texts; torch.cuda.empty_cache()

    if cpath is not None:
        # write to a temporary file and rename, so a half-written file is never taken as a cache
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_name(cpath.name + ".partial")
        torch.save({"key": json.dumps(key), **buf}, tmp)
        tmp.rename(cpath)
        print(f"[calib] saved -> {cpath}", flush=True)
    return buf


def pooled_norm(x):
    """[1,H,P,D] -> x / RMS over all heads and channels, one scalar per token (TaSQ's N).

    The served counterpart is the pool's pertoken_norm + pool_heads_scale path, which stores
    that scalar per (token, slot) and multiplies it back at decode."""
    return x / x.pow(2).mean(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-8)


def apply_perm(x, perm):
    """[1,H,P,D] natural channels -> stored order. perm [H,D]: stored channel i holds natural
    channel perm[h, i]. None = identity."""
    if perm is None:
        return x
    return x.gather(-1, perm.to(x.device).view(1, perm.shape[0], 1, -1).expand_as(x))


@torch.no_grad()
def pair_cov_perm(samples, slots, G, norm, device):
    """TaSQ's covariance grouping (TaSQ repository, codebooks/make_tasq.py:match_groups) for keys, RoPE pairs
    {j, j+D/2} kept atomic, over one block of len(slots) loop steps.

    Units are pairs taken across all the block's steps (2*Tc coordinates); iterated min-weight
    perfect matching on det(Sigma_sub)^(1/|sub|) merges them until a group has G coordinates.
    Returns perm [H, D] (stored -> natural) whose groups read [j, j+half, k, k+half, ...] --
    the pair-adjacent layout the PERM_ROPE decode expects -- or None if a pair cannot fit in
    a group (c = G/Tc < 2). Unweighted covariance of the (optionally normalised) keys.
    """
    import numpy as np
    from ouro_cq import match_units
    Tc = len(slots)
    if G < 2 * Tc:
        return None
    rows = []
    for smp in samples:
        # Cov(k^wn): weighted then pooled-normalised, as TaSQ's make_tasq.py pass 2
        per = [k_transform(smp.k[sl].to(device, torch.float32), sl, norm)[0] for sl in slots]
        rows.append(torch.cat(per, dim=-1)[0])                           # [H,P,Tc*D]
    Z = torch.cat(rows, dim=1)                                           # [H,N,Tc*D]
    H, _, TD = Z.shape
    D = TD // Tc
    half = D // 2
    perm = torch.empty(H, D, dtype=torch.long)
    for h in range(H):
        Zc = Z[h] - Z[h].mean(0)
        S = (Zc.T @ Zc / Zc.shape[0]).double().cpu().numpy()
        eps = max(float(np.diag(S).mean()) * 1e-4, 1e-30)
        units = [[m * D + ch for ch in (j, j + half) for m in range(Tc)] for j in range(half)]
        groups = match_units(S, units, G, eps)
        chans = []
        for grp in groups:
            # channel order inside a group follows unit order: [j, j+half, k, k+half, ...]
            for coord in grp:
                if coord < D:
                    chans.append(coord)
        perm[h] = torch.tensor(chans)
        assert sorted(chans) == list(range(D))
        assert all(perm[h, 2 * p + 1] == perm[h, 2 * p] + half for p in range(half)), "pair split"
    return perm


@torch.no_grad()
def v_cov_perm(samples, slots, G, norm, device):
    """Covariance grouping for V (no RoPE, so single channels are the units), one block.

    The permutation is SHARED BY ALL HEADS: it is served through the pool's per-slot V rotation
    (v @ R_v before storage, o @ R_v^T after attention), which is one [D, D] matrix per slot.
    Covariance is therefore pooled over heads. Returns perm [D] (stored -> natural).
    """
    import numpy as np
    from ouro_cq import match_units
    Tc = len(slots)
    rows = []
    for smp in samples:
        per = [smp.v[sl].to(device, torch.float32) for sl in slots]
        if norm:
            per = [pooled_norm(x) for x in per]
        rows.append(torch.cat(per, dim=-1)[0])                           # [H,P,Tc*D]
    Z = torch.cat(rows, dim=1)                                           # [H,N,Tc*D]
    H, _, TD = Z.shape
    D = TD // Tc
    Zc = Z - Z.mean(1, keepdim=True)
    S = torch.einsum("hnd,hne->de", Zc, Zc).double().cpu().numpy() / (H * Zc.shape[1])
    eps = max(float(np.diag(S).mean()) * 1e-4, 1e-30)
    units = [[m * D + ch for m in range(Tc)] for ch in range(D)]
    chans = [coord for grp in match_units(S, units, G, eps) for coord in grp if coord < D]
    assert sorted(chans) == list(range(D))
    return torch.tensor(chans)


@torch.no_grad()
@torch.no_grad()
def loop_sensitivity_scale(samples, slots, kind, norm, perm, mode, device, clip=16.0):
    """Static per-(slot, head, stored channel) scale s = sqrt(mean Fisher), [Tc, H, D].

    A loopvq code couples Tc loop steps, and CQ's Eq.6 gives each vector ONE weight, so every
    step's coordinates count the same inside the code. Their Fisher does not: Ouro-1.4B's K at
    layer 0 is ~30x more sensitive at loop 0 than at loop 3, and from layer ~14 on loop 3
    dominates. Fitting and encoding s*x instead of x makes
    the code's metric that Fisher -- the final-loop CE gradient, which already carries an early
    step's downstream effect through the later loops -- and decode divides s back out.

    mode: "ch" one factor per (step, head, channel); "loop" one per (step, head). The Fisher is
    taken in the coordinates actually quantised: for a normalised bundle x/sigma has gradient
    g*sigma. Normalised per head (mean Fisher 1) and clipped to [1/clip, clip] so a channel the
    calibration barely sees is not given unbounded error.
    """
    Tc = len(slots)
    if mode == "qe":
        # forward-only importance: query energy per RoPE pair (collect_attn_stats.py), natural
        # channel order -> stored order
        fis = torch.stack([apply_perm(ATTN["q_energy"][sl].to(device).view(1, -1, 1, ATTN["q_energy"].shape[-1]),
                                      perm)[0, :, 0] for sl in slots])            # [Tc,H,D]
        fis = fis / fis.mean(dim=(0, 2), keepdim=True).clamp_min(1e-30)
        return fis.pow(LS_POW / 2).clamp(1.0 / clip, clip)
    acc = None
    n = 0
    for s in samples:
        X, Gr = (s.k, s.gk) if kind == "k" else (s.v, s.gv)
        for m, sl in enumerate(slots):
            g = Gr[sl].to(device, torch.float32)
            if norm:
                x = X[sl].to(device, torch.float32)
                g = g * x.pow(2).mean(dim=(1, 3), keepdim=True).sqrt().clamp_min(1e-8)
            g = apply_perm(g, perm)                                    # [1,H,P,D]
            f = g.pow(2).sum(dim=(0, 2))                               # [H,D]
            if acc is None:
                acc = torch.zeros(Tc, *f.shape, device=device)
            acc[m] += f
        n += Gr[slots[0]].shape[2]
    fis = acc / n                                                      # [Tc,H,D]
    if mode == "loop":
        fis = fis.mean(-1, keepdim=True).expand_as(fis)
    fis = fis / fis.mean(dim=(0, 2), keepdim=True).clamp_min(1e-30)
    return fis.pow(LS_POW / 2).clamp(1.0 / clip, clip)


def fit_loop_slot(samples, slots, c_ch, G, K, iters, seed, device, kind="k", norm=False,
                  perm=None, scale=None):
    """Head-chunked front of _fit_loop_slot. The rows of a coupled fit are independent codebooks,
    head-major, so heads can be fit in chunks and concatenated -- exactly the same fit, as long as
    no transform mixes heads (pooled norm, TaSQ's W -> N) or carries per-head state (perm, scale,
    attention weights). Huginn at Tc = 32 needs it: one code layer of 32 slots x 55 heads x 16k
    vectors is 11 GB of fp32 input before k-means starts."""
    X0 = samples[0].k[slots[0]]
    H, P_, D_ = X0.shape[1], sum(s.k[slots[0]].shape[2] for s in samples), X0.shape[3]
    est = 3 * 4 * H * P_ * D_ * len(slots)                    # x, its grouping, the gradients
    plain = not norm and perm is None and scale is None and not ATTN_W and not (kind == "k" and QW is not None)
    if not plain or est <= 6 * 2**30:
        return _fit_loop_slot(samples, slots, c_ch, G, K, iters, seed, device, kind, norm, perm, scale)
    hc = max(1, int(H * 6 * 2**30 // est))
    out = []
    for h0 in range(0, H, hc):
        sub = [Calib(*([t[:, h0:h0 + hc] for t in getattr(s, f)] for f in CALIB_FIELDS)) for s in samples]
        out.append(_fit_loop_slot(sub, slots, c_ch, G, K, iters, seed, device, kind, norm, None, None))
        torch.cuda.empty_cache()
    return torch.cat(out, 0)


def _fit_loop_slot(samples, slots, c_ch, G, K, iters, seed, device, kind="k", norm=False,
                   perm=None, scale=None):
    """loopvq: fit the codebooks of one base layer with its Tc slots coupled along the loop axis.

    scale [Tc,H,D] (loop_sensitivity_scale): fit in the space s*x; the Eq.6 weight of a scaled
    vector is sum((g/s)^2), the same first-order loss in the new coordinates."""
    xs, ws = [], []
    for si, s in enumerate(samples):
        X, Gr = (s.k, s.gk) if kind == "k" else (s.v, s.gv)
        tq = kind == "k" and QW is not None
        if tq:
            tr = [k_transform(X[sl].to(device, torch.float32), sl, norm) for sl in slots]
            per = [x for x, _ in tr]
        else:
            per = [X[sl].to(device, torch.float32) for sl in slots]
            if norm:
                per = [pooled_norm(x) for x in per]
        per = [apply_perm(x, perm) for x in per]
        # gradients only when the weight is Fisher's: the attn / uniform modes never read them (and
        # holding them pushed the attn build past 24 GB)
        need_g = not (ATTN_W or UNIFORM_W)
        if need_g and tq:
            # TaSQ's Fisher in the coded coordinates u = w k / s (build_bundles.sh defaults:
            # --fisher-native-coord and --ptnorm-weight): dk_i = (s / w_i) du_i, so the native-space
            # distortion sum_i g_i^2 dk_i^2 is sum_i (g_i s / w_i)^2 du_i^2
            gper = []
            for (_, sg), sl in zip(tr, slots):
                g = Gr[sl].to(device, torch.float32) / QW[sl].to(device).view(1, -1, 1, QW.shape[-1])
                gper.append(apply_perm(g * sg if sg is not None else g, perm))
        else:
            gper = [apply_perm(Gr[sl].to(device, torch.float32), perm) for sl in slots] if need_g else None
        if need_g and norm and FISHER_NORM and not tq:
            # the codes quantise x/sigma, whose gradient is g*sigma -- the raw g undercounts
            # tokens with a large norm by sigma^2
            sig = [X[sl].to(device, torch.float32).pow(2).mean(dim=(1, 3), keepdim=True).sqrt()
                   .clamp_min(1e-8) for sl in slots]
            gper = [g * sg for g, sg in zip(gper, sig)]
        if scale is not None:
            per = [x * scale[m].view(1, -1, 1, scale.shape[-1]) for m, x in enumerate(per)]
            if need_g:
                gper = [g / scale[m].view(1, -1, 1, scale.shape[-1]) for m, g in enumerate(gper)]
        xs.append(group_loop(per, c_ch))
        if ATTN_W:
            # attention mass of this token summed over the Tc steps the code covers, the same
            # for every group of a head: [H, P] -> [H*(D/c), P] in group_loop's row order
            m = sum(ATTN["att_mass"][sl, si].to(device) for sl in slots)          # [H, P]
            NGc = per[0].shape[-1] // c_ch
            ws.append(m.unsqueeze(1).expand(-1, NGc, -1).reshape(-1, m.shape[-1]))
        else:
            if need_g and LOOP_WEIGHT != "sum":
                g2 = group_loop(gper, c_ch) ** 2                                  # [rows, P, c*Tc]
                ws.append(g2.view(*g2.shape[:2], c_ch, len(slots)).sum(2))        # per-loop [rows,P,Tc]
                continue
            if need_g:
                w = (group_loop(gper, c_ch) ** 2).sum(-1)                         # Eq.6 [H*NG', P]
                if FISHER_POOL == "heads":
                    # one weight per token: a single token's g^2 is one noisy sample per head
                    w = w.mean(dim=0, keepdim=True).expand_as(w).contiguous()     # [H*NG', P] rows share it
                if FISHER_POW != 1.0:
                    w = w.clamp_min(0).pow(FISHER_POW)
                ws.append(w)
            else:
                ws.append(torch.ones(xs[-1].shape[:2], device=device))           # uniform
    x = torch.cat(xs, dim=1); w = torch.cat(ws, dim=1)
    if w.dim() == 3:
        # --loop-weight: how a coupled vector's per-loop sensitivities s_m combine into its weight
        #   balanced  each loop normalised by its own mean first, so no step dominates by scale
        #   max       the most sensitive step decides
        #   early2 / late2   s_m scaled by 2^(Tc-1-m) / 2^m (mean 1): which end of the loop matters
        Tc = w.shape[-1]
        if LOOP_WEIGHT == "balanced":
            w = (w / w.mean(dim=(0, 1), keepdim=True).clamp_min(1e-30)).sum(-1)
        elif LOOP_WEIGHT == "max":
            w = w.max(-1).values
        else:
            e = torch.arange(Tc, device=w.device, dtype=w.dtype)
            gam = 2.0 ** ((Tc - 1 - e) if LOOP_WEIGHT == "early2" else e)
            w = (w * (gam / gam.mean())).sum(-1)
    cb = weighted_kmeans(x, w, K, iters=iters, seed=seed + slots[0])
    del x, w, xs, ws
    return cb


@torch.no_grad()
def fit_slot(samples, slot, kind, grouper, G, K, iters, seed, device, norm=False, perm=None):
    """Fit the codebooks [NB=H*(D//G), K, G] of one slot with Fisher-weighted k-means."""
    xs, ws = [], []
    for s in samples:
        x = (s.k if kind == "k" else s.v)[slot].to(device, torch.float32)
        if norm:
            x = pooled_norm(x)
        x = apply_perm(x, perm)
        g = apply_perm((s.gk if kind == "k" else s.gv)[slot].to(device, torch.float32), perm)
        xs.append(grouper(x, G))
        ws.append((grouper(g, G) ** 2).sum(-1) if not UNIFORM_W
                  else torch.ones(grouper(g, G).shape[:2], device=device))   # Eq.6 / uniform
    x = torch.cat(xs, dim=1); w = torch.cat(ws, dim=1)
    cb = weighted_kmeans(x, w, K, iters=iters, seed=seed + slot)
    del x, w, xs, ws
    return cb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-cache", default=os.environ.get("CALIB_DATA_DIR", str(Path(__file__).resolve().parent / ".." / "work" / "calib_data")))
    ap.add_argument("--group-dim", type=int, default=8, help="CQ coupled channels (COUPLED)")
    ap.add_argument("--abits", type=int, default=10, help="bits per code (ABITS) -> K=2^abits")
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--calib-seq-len", type=int, default=1024,
                    help="with T=4 loop steps the backward pass covers 96 layer applications; 2048 OOMs on 24 GB")
    ap.add_argument("--calib-positions", type=int, default=1024)
    ap.add_argument("--kmeans-iters", type=int, default=15)
    ap.add_argument("--method", choices=("cq", "loopvq"), default="cq")
    ap.add_argument("--loops-coupled", type=int, default=0,
                    help="loopvq: number of loop steps Tc to couple (0 = all total_ut_steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-method", choices=("cq", "loopvq"), default="cq",
                    help="V grouping: per-slot CQ (default) or loop-coupled like K")
    ap.add_argument("--v-norm", choices=("none", "pooled"), default="none",
                    help="pooled: divide each (token, slot) V row by its RMS over all heads and "
                         "channels before fitting (served via pertoken_norm + pool_heads_scale)")
    ap.add_argument("--k-norm", choices=("none", "pooled"), default="none",
                    help="same pooled-head per-token normalisation as --v-norm, on keys")
    ap.add_argument("--k-cov", action="store_true",
                    help="covariance-aware key grouping with RoPE pairs atomic (TaSQ's P), "
                         "served through PERM_ROPE; skipped where a pair cannot fit a group")
    ap.add_argument("--v-cov", action="store_true",
                    help="covariance-aware V grouping, one permutation per block shared by all "
                         "heads, served as the per-slot V rotation (a permutation matrix)")
    ap.add_argument("--k-loop-scale", choices=("none", "ch", "loop", "qe"), default="none",
                    help="loop-sensitivity scaling of K codes (loop_sensitivity_scale): fit and "
                         "encode s*x with s = sqrt(Fisher) per (loop step, head[, channel])")
    ap.add_argument("--v-loop-scale", choices=("none", "ch", "loop"), default="none",
                    help="same for V (loop-coupled V only)")
    ap.add_argument("--k-qweight", default=None,
                    help="tasq_qweights.py output: TaSQ's query-guided key weighting W, applied "
                         "before the pooled norm (W -> N -> P); served as chan_weight")
    ap.add_argument("--k-from", default=None,
                    help="reuse codebook.pt from this bundle instead of fitting K -- so a V-only "
                         "variant differs from its parent in V alone")
    ap.add_argument("--calib-cache", default=None,
                    help="calibration buffer file: opened with mmap if present, else built and saved. "
                         "Parallel fits must share the calibration through it (otherwise OOM)")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="activation checkpointing for the Fisher backward (needed at 2048 tokens)")
    ap.add_argument("--calib-data", choices=("wikitext", "mix", "gpqa_nodiamond", "c4", "slimpajama"), default="wikitext",
                    help="calibration text (calib_texts): wikitext-2, or a wikitext / GSM8K-train / "
                         "Python mix")
    ap.add_argument("--calib-target", choices=("ce", "kl", "logit_rp", "loop_exits", "exit_pdf", "last2_exits"),
                    default="ce",
                    help="what the Fisher gradient is taken of (calib_loss): data CE (default), the "
                         "model's own final-loop distribution (kl, true Fisher), or a random "
                         "projection of the final-loop logits")
    ap.add_argument("--attn-stats", default=None, help="collect_attn_stats.py output (qe / attn modes)")
    ap.add_argument("--loop-scale-pow", type=float, default=1.0,
                    help="s = importance ** (pow/2): 1 = sqrt (full weighting), 0.5 = gentler")
    ap.add_argument("--fisher-pool", choices=("none", "heads"), default="none")
    ap.add_argument("--loop-weight", choices=("sum", "balanced", "max", "early2", "late2"), default="sum")
    ap.add_argument("--fisher-pow", type=float, default=1.0)
    ap.add_argument("--fisher-norm-fix", action="store_true",
                    help="normalised bundles: Eq.6 weight from the gradient of the normalised coordinates")
    ap.add_argument("--kmeans-weight", choices=("fisher", "uniform", "attn"), default="fisher",
                    help="per-vector k-means weight: CQ Eq.6 Fisher (default) or uniform -- if uniform "
                         "matches, calibration needs no backward pass")
    ap.add_argument("--calib-only", action="store_true",
                    help="build --calib-cache and exit (once, before parallel fits)")
    args = ap.parse_args()
    global UNIFORM_W, ATTN, ATTN_W, LS_POW, FISHER_NORM, FISHER_POOL, FISHER_POW
    FISHER_NORM = args.fisher_norm_fix
    FISHER_POOL, FISHER_POW = args.fisher_pool, args.fisher_pow
    global LOOP_WEIGHT, QW
    LOOP_WEIGHT = args.loop_weight
    if args.k_qweight:
        QW = torch.load(args.k_qweight, map_location="cpu", weights_only=True)["w"].float()
        assert args.method == "loopvq", "--k-qweight is wired for loop-coupled K only"
        assert args.k_loop_scale == "none", "--k-qweight with --k-loop-scale is not defined"
    UNIFORM_W = args.kmeans_weight == "uniform"
    ATTN_W = args.kmeans_weight == "attn"
    LS_POW = args.loop_scale_pow
    if args.attn_stats:
        ATTN = torch.load(args.attn_stats, map_location="cpu", weights_only=True)
    assert not (ATTN_W or args.k_loop_scale == "qe") or ATTN is not None, "needs --attn-stats"
    assert not args.calib_only or args.calib_cache, "--calib-only needs --calib-cache"

    dev = "cuda"
    G, K = args.group_dim, 2 ** args.abits

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    from huginn_torch import huginn_geometry, is_huginn
    if is_huginn(cfg):
        # Huginn: Lb = 4 core blocks x T recurrences (engine slots t*Lb + l), then 2 prelude and
        # 2 coda slots that have no loop axis ("plain" slots: per-slot codes in every method).
        # Calibration comes from huginn_torch.py calib (pass --calib-cache).
        L_base, T, n_slots, H, D = huginn_geometry(cfg)
    else:
        L_base, T = cfg.num_hidden_layers, getattr(cfg, "total_ut_steps", 1) or 1
        H, D = cfg.num_key_value_heads, cfg.head_dim
        n_slots = L_base * T
    plain_slots = list(range(L_base * T, n_slots))
    assert D % G == 0
    print(f"[cfg] L={L_base} T={T} slots={n_slots} (plain {plain_slots}) H={H} D={D} G={G} K={K}")

    buf = load_or_calibrate(args, n_slots, H, D, dev)
    if args.calib_only:
        print(f"[done] calib only -> {args.calib_cache}")
        return
    samples = calib_views(buf)
    cl = args.calib_seq_len
    n_vec = sum(s.k[0].shape[2] for s in samples)
    print(f"[calib] {n_vec} vectors per codebook ({n_vec/K:.1f} per centroid)")
    # created right before fitting, so a run that dies in calibration leaves no empty "bundle"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # K: contiguous channels within a head / V: see the module docstring
    per_head_groups = D // G
    eye = torch.eye(D, dtype=torch.float32)
    meta = {
        "forward": eye.unsqueeze(0).unsqueeze(0).expand(n_slots, H, D, D).contiguous(),
        "inverse": eye.unsqueeze(0).unsqueeze(0).expand(n_slots, H, D, D).contiguous(),
        "mean": torch.zeros(n_slots, H, D, dtype=torch.float32),
        "bounds": [(g * G, (g + 1) * G, args.abits) for g in range(per_head_groups)],
        "pertoken_norm": False,
    }

    Tc = (args.loops_coupled or T) if args.method == "loopvq" else 1
    c_ch = G // Tc
    if args.method == "loopvq":
        assert G % Tc == 0 and T % Tc == 0, f"G={G} / T={T} not divisible by Tc={Tc}"
        print(f"[loopvq] K group = {c_ch} channels x {Tc} loop steps")

    # V layout must be the coordinates the pool ENCODES V in: the engine's wide-G layout encodes
    # contiguous groups.
    v_layout = os.environ.get("LOOPVQ_V_LAYOUT", "contiguous")
    assert v_layout in GROUPERS, v_layout
    v_loop, v_norm = args.v_method == "loopvq", args.v_norm == "pooled"
    if v_loop:
        assert Tc > 1 or args.loops_coupled, "--v-method loopvq needs --loops-coupled (or --method loopvq)"
    Tc_v = (args.loops_coupled or T) if v_loop else 1
    for kind, layout, fname in (("k", "contiguous", "codebook.pt"),
                                ("v", v_layout, "vq_v_codebook.pt")):
        if kind == "k" and args.k_from:
            import shutil
            shutil.copy(Path(args.k_from) / fname, out / fname)
            print(f"[k] copied from {args.k_from}")
            continue
        codebooks = {}
        t0 = time.time()
        loop = (args.method == "loopvq") if kind == "k" else v_loop
        norm = (v_norm and kind == "v") or (args.k_norm == "pooled" and kind == "k")
        use_cov = args.k_cov and kind == "k"
        use_vcov = args.v_cov and kind == "v"
        perms = {}                                  # slot -> [H, D] stored -> natural
        ls_mode = args.k_loop_scale if kind == "k" else args.v_loop_scale
        assert ls_mode == "none" or loop, f"--{kind}-loop-scale needs a loop-coupled {kind}"
        loop_scale = torch.ones(n_slots, H, D) if ls_mode != "none" else None
        # Tc / c_ch name THIS kind's coupling inside the loop body; K's are restored after it.
        Tc_k, c_k = Tc, c_ch
        Tc, c_ch = (Tc_k, c_k) if kind == "k" else (Tc_v, G // Tc_v)
        if loop:
            # fitted once per base layer; its D//c_ch groups are split over the Tc slots
            # the layer occupies, so the code count equals CQ's.
            # Tc < T leaves T/Tc blocks per base layer, each its own codebook: block `blk`
            # couples steps blk*Tc .. blk*Tc+Tc-1. (This loop used to cover block 0 only,
            # so a Tc=2 bundle had no codebooks for ut 2-3.)
            n_groups = D // c_ch
            for blk in range(T // Tc):
                for l in range(L_base):
                    slots = [l + (blk * Tc + m) * L_base for m in range(Tc)]
                    perm = pair_cov_perm(samples, slots, G, norm, dev) if use_cov else None
                    if use_vcov:
                        perm = v_cov_perm(samples, slots, G, norm, dev).unsqueeze(0).expand(H, D).contiguous()
                    if perm is not None:
                        perms.update({sl: perm for sl in slots})
                    sc = (loop_sensitivity_scale(samples, slots, kind, norm, perm, ls_mode, dev)
                          if ls_mode != "none" else None)
                    if sc is not None:
                        for m, sl in enumerate(slots):
                            loop_scale[sl] = sc[m].cpu()
                    cb = fit_loop_slot(samples, slots, c_ch, G, K, args.kmeans_iters, args.seed,
                                       dev, kind=kind, norm=norm, perm=perm,
                                       scale=sc).cpu()  # [H*n_groups,K,G]
                    for h in range(H):
                        for g in range(n_groups):
                            slot, gg = loop_slot_map(l, g, L_base, per_head_groups)
                            slot += blk * Tc * L_base
                            codebooks.setdefault((slot, h), [None] * per_head_groups)
                            codebooks[(slot, h)][gg] = (
                                cb[h * n_groups + g].to(torch.float16).contiguous())
                    if (l + 1) % 6 == 0:
                        print(f"    {kind}(loopvq): block {blk+1}/{T // Tc} base layer "
                              f"{l+1}/{L_base} ({time.time()-t0:.0f}s)", flush=True)
                    torch.cuda.empty_cache()
            for slot in plain_slots:
                # no loop axis: an ordinary per-slot code of G channels at the same bit rate, fit
                # through fit_loop_slot with one member so TaSQ's W -> N -> P apply as on the core
                # slots (with c = G and one step, group_loop is the contiguous grouping)
                perm = pair_cov_perm(samples, [slot], G, norm, dev) if use_cov else None
                if perm is not None:
                    perms[slot] = perm
                cb = fit_loop_slot(samples, [slot], G, G, K, args.kmeans_iters, args.seed, dev,
                                   kind=kind, norm=norm, perm=perm).cpu()
                for h in range(H):
                    codebooks[(slot, h)] = [cb[h * per_head_groups + g].to(torch.float16).contiguous()
                                            for g in range(per_head_groups)]
            assert len(codebooks) == n_slots * H, (
                f"loopvq bundle covers {len(codebooks)} of {n_slots * H} (slot, head) seats")
            assert all(t is not None for v in codebooks.values() for t in v)
        else:
            grouper = GROUPERS[layout]
            for slot in range(n_slots):
                perm = pair_cov_perm(samples, [slot], G, norm, dev) if use_cov else None
                if use_vcov:
                    perm = v_cov_perm(samples, [slot], G, norm, dev).unsqueeze(0).expand(H, D).contiguous()
                if perm is not None:
                    perms[slot] = perm
                cb = fit_slot(samples, slot, kind, grouper, G, K, args.kmeans_iters, args.seed,
                              dev, norm=norm, perm=perm).cpu()
                for h in range(H):
                    codebooks[(slot, h)] = [
                        cb[h * per_head_groups + g].to(torch.float16).contiguous()
                        for g in range(per_head_groups)
                    ]
                if (slot + 1) % 24 == 0:
                    print(f"    {kind}: slot {slot+1}/{n_slots} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                torch.cuda.empty_cache()
        extra = ({"loop_coupled": {"loops": Tc, "channels": c_ch, "base_layers": L_base,
                                   **({"plain_slots": plain_slots} if plain_slots else {})}}
                 if loop else {})
        if norm:
            extra.update(pertoken_norm=True, pool_heads_scale=True)
        if kind == "k" and QW is not None:
            assert QW.shape == (n_slots, H, D), QW.shape
            # stored channel i holds natural perm[i]: chan_weight[slot, h, i] = w[slot, h, perm[i]]
            cw = torch.stack([QW[sl].gather(-1, perms[sl]) if sl in perms else QW[sl]
                              for sl in range(n_slots)])
            extra.update(chan_weight=cw.contiguous())
            print(f"[k] TaSQ W: chan_weight in [{cw.min():.3f}, {cw.max():.3f}]")
        if loop_scale is not None:
            extra.update(loop_scale=loop_scale, loop_scale_mode=ls_mode)
            print(f"[{kind}] loop-sensitivity scale ({ls_mode}): s in "
                  f"[{loop_scale.min():.3f}, {loop_scale.max():.3f}]")
        if perms and kind == "v":
            assert len(perms) == n_slots
            v_rot = {sl: torch.nn.functional.one_hot(perms[sl][0], D).float().T.contiguous()
                     for sl in range(n_slots)}            # v @ R puts natural perm[i] at i
            print(f"[v] covariance grouping: {len(perms)} slots permuted (shared across heads)")
        if perms and kind == "k":
            # Every slot or none: PERM_ROPE is a per-bundle mode, and a slot left in natural
            # order would not have its RoPE pairs adjacent.
            assert len(perms) == n_slots, f"permutation for {len(perms)}/{n_slots} slots"
            P = torch.stack([perms[sl] for sl in range(n_slots)])          # [L,H,D]
            # r = k @ forward puts natural channel P[i] at stored position i: forward[P[i], i] = 1
            fwd2 = torch.zeros(n_slots, H, D, D)
            for i in range(D):
                fwd2[:, :, :, i] = torch.nn.functional.one_hot(P[:, :, i], D).float()
            extra.update(forward=fwd2, inverse=fwd2.transpose(-1, -2).contiguous(),
                         perm_rope=True, freq_idx=P[..., 0::2].contiguous(),
                         w_perm=torch.ones(n_slots, H, D))
            print(f"[{kind}] covariance grouping: {len(perms)} slots permuted (RoPE pairs atomic)")
        torch.save({**meta, **extra, "codebooks": codebooks}, out / fname)
        tag = (f"loop({c_ch}c x {Tc}L)" if loop else f"{layout} groups") + (" +norm" if norm else "")
        print(f"[{kind}] {tag} -> {out/fname} ({time.time()-t0:.0f}s)")
        Tc, c_ch = Tc_k, c_k

    # CQ is served without a rotation (objective="identity"); serve_method.sh requires the
    # rotation files to exist, so identity rotations are written.
    for name in ("k_rotation_qqt_r_h_pbr.pt", "v_rotation_sst_r_h_pbr.pt"):
        vperm = name.startswith("v_") and args.v_cov
        torch.save({"format_version": 1,
                    "objective": "covariance_grouping_permutation" if vperm else "identity",
                    "source_grouping": None,
                    "layers": {i: {"layer_id": i, "rotation": v_rot[i] if vperm else torch.eye(D)}
                               for i in range(n_slots)}}, out / name)
    print(f"[rot] identity rotations for {n_slots} slots")

    (out / "build_info.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()), "method": args.method,
        "num_hidden_layers": L_base, "total_ut_steps": T, "bundle_layers": n_slots,
        "heads": H, "head_dim": D, "group_dim": G, "abits": args.abits, "codebook_size": K,
        "bits_per_channel": args.abits / G,
        "k_group_layout": "contiguous", "v_group_layout": v_layout,
        "calibration": {"corpus": args.calib_data, "n_seq": args.n_calib,
                        "seq_len": cl, "positions": args.calib_positions,
                        "vectors_per_codebook": n_vec,
                        "fisher_weighted": True, "pre_rope_keys": True,
                        "kmeans": "k-means++ init, weighted Lloyd (CQ Eq.6)",
                        "kmeans_iters": args.kmeans_iters},
        "loop_coupled": ({"loops": Tc, "channels": c_ch} if args.method == "loopvq" else None),
        "v_method": args.v_method, "v_norm": args.v_norm,
        "k_norm": args.k_norm, "k_cov": args.k_cov, "v_cov": args.v_cov,
        "k_loop_scale": args.k_loop_scale, "v_loop_scale": args.v_loop_scale,
        "kmeans_weight": args.kmeans_weight, "calib_target": args.calib_target,
        "loop_scale_pow": args.loop_scale_pow, "attn_stats": args.attn_stats,
        "calib_data": args.calib_data,
        "fisher_norm_fix": args.fisher_norm_fix, "fisher_pool": args.fisher_pool,
        "fisher_pow": args.fisher_pow, "loop_weight": args.loop_weight,
        "v_loop_coupled": ({"loops": Tc_v, "channels": G // Tc_v} if v_loop else None),
        "k_from": args.k_from, "k_qweight": args.k_qweight,
        "note": ("K coupled along the loop axis; the code count equals cq's"
                 if args.method == "loopvq"
                 else "slot axis = ut*L + layer: an independent codebook per loop step (per-step baseline)"),
    }, indent=2, ensure_ascii=False))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
