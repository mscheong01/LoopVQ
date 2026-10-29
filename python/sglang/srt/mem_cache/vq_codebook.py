"""Group-VQ codebook bundle loader + torch-side encode helpers for the vq2
K quant tier of the unified mixed HP+int2 pool.

Bundle schema (produced by the ``exp/calibration/fit_vq_codebook.py`` trainer):

    forward   [L, H, D, D]   residual map: r = (k - mean) @ forward  (per head)
    inverse   [L, H, D, D]   recon map:    k_hat = r_hat @ inverse + mean
    mean      [L, H, D]
    codebooks {(l, h): list of NG tensors [K, G]}   (fp16 or fp8_e4m3fn)
    bounds    [(start, end, bits)] * NG   -- flat allocation, contiguous groups
    pertoken_norm bool -- per-token RMS scale on r before lookup

Engine storage convention: BOTH tiers hold the residual ``r`` (HP as bf16
rows, quant as VQ indices + per-token RMS scale). Queries are mapped with
``q @ inverse.T`` so ``q_m . r = q . (k - mean)``; the ``-q . mean`` term is
constant across keys for a given query, hence softmax-invariant, and it is
identical for both tiers. Models with a learned attention sink are the
exception: the sink is an extra denominator logit, not a real key, so it must
receive the same per-query ``-q . mean`` shift. Prefill adjusts it explicitly;
decode fuses the adjustment into unified stage 2.

The decode kernel reconstructs codewords from a packed int32 (4x fp8-e5m2
bytes, little-endian => byte i is coord i of the group; e5m2 because sm80
Triton only bitcasts fp8e5). Encode assigns against the *fp8-dequantized*
centroids so the encoder is a true nearest-neighbor for what the decoder
actually reconstructs.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class VQCodebook:
    forward: torch.Tensor    # [L, H, D, D] bf16 -- k-side map (applied to k - mean)
    q_map: torch.Tensor      # [L, H, D, D] bf16 -- inverse.transpose(-1,-2), q-side map
    mean: torch.Tensor       # [L, H, D] bf16
    cb16: torch.Tensor       # [L, H, NG, K, G] fp16 -- fp8-dequantized centroids
    cb_sq: torch.Tensor      # [L, H, NG, K] fp32 -- 0.5 * ||c||^2 per centroid
    cb_packed: torch.Tensor  # [L, H, NG, K] int32 -- fp8 bytes packed little-endian
    num_groups: int          # NG
    codebook_size: int       # K
    group_dim: int           # G
    pertoken_norm: bool
    source_fp8_fmt: str | None      # fp8 format the bundle declares, if any
    centroid_resnap_rel_rmse: float  # error added by re-snapping at load
    fp8_fmt: str = "e5m2"    # "e5m2" | "e4m3" -- decode kernel must bitcast to match
    wide_g: bool = False      # G != 4 -- decode via plain fp16 gather, not packed-int32
    idx_dtype: torch.dtype = torch.uint8  # uint8 (K<=256) or int16 (K>256) index arena
    perm_rope: bool = False  # decode skips q_map entirely (see decode_attention.py's
    # PERM_ROPE branch) -- K is unweighted+rotated via freq_idx/w_perm to land on
    # plain original channels, and Q is read unmapped. Requires freq_idx/w_perm.
    identity_map: bool = False  # forward=inverse=I and mean=0; skip redundant transforms
    freq_idx: torch.Tensor = None  # [L, H, D//2] int64, only used when perm_rope
    w_perm: torch.Tensor = None    # [L, H, D] fp32, only used when perm_rope
    cb16_dec_pair_major: bool = False  # cb16_dec holds each centroid as [e0..e3 | o0..o3]
    # instead of pair-interleaved. A coordinate relabelling inside the group, so the quantiser
    # is unchanged; it lets the decode kernel halve at a register boundary instead of unpacking
    # evens from odds. See PM_FUSED in decode_attention.py.
    cb16_dec: torch.Tensor = None  # [L, H, NG, K, G] fp16 with 1/w folded in; decode reads
    # this instead of cb16 so PERM_ROPE needs no per-element unweight in the KV loop. Built
    # at load for perm_rope bundles only; see the fold note in load_vq_codebook.
    # Loop-coupled K (loopvq bundles only). cb16/cb_sq above then hold the PER-SLOT decode
    # slices -- [L, H, D/c, K, c], an ordinary (NG'=D/c, G'=c) geometry -- while the encoder
    # needs the whole coupled entry, so it gets its own table over the Lc = Lb*T/Tc code
    # layers. See mem_cache/loopvq_pack.py for the layout and the pool for the aliasing.
    loop: dict = None              # {"Lb", "T", "Tc", "c", "Lc"} or None
    cb16_enc: torch.Tensor = None  # [Lc, H, D/c, K, c*Tc] fp16
    cb_sq_enc: torch.Tensor = None  # [Lc, H, D/c, K] fp32
    code_layer: list = None        # engine slot -> code layer, len L
    # Loop-sensitivity scaling (build_ouro_bundle --k/v-loop-scale): static per (slot, head,
    # stored channel) factor s. Encode quantises s*x against cb16_enc (fit in that space); the
    # per-slot decode slices in cb16 are pre-divided by s, so decode -- every kernel -- sees
    # plain coordinates and needs no change. None = unscaled.
    loop_scale: torch.Tensor = None  # [L, H, D] fp32
    # TaSQ's query-guided channel weight W (build_ouro_bundle --k-qweight), static per (slot,
    # head, stored channel). Unlike loop_scale it is applied BEFORE the pooled-head norm (TaSQ's
    # W -> N order: the stored scale is RMS(w*k)); decode slices are pre-divided by w the same way.
    chan_weight: torch.Tensor = None  # [L, H, D] fp32
    # [L, H, D] int64 source channel per stored channel, set only when `forward` is an exact
    # 0/1 permutation, q_map == forward (inverse.T of a permutation) and mean == 0 (TaSQ's
    # PERM_ROPE without weights). vq_map_k / vq_map_q then gather instead of running a D x D
    # matmul per token -- the same values (a 0/1 product with fp32 accumulation is a copy).
    # None otherwise.
    perm_src: torch.Tensor = None
    pool_heads_scale: bool = False  # pertoken_norm pools RMS across ALL KV heads, not
    # per-(token,head) independently like
    # the pool's own pertoken_norm. Decode needs no change (K_Scales_Zeros already stores
    # one value per (token,head); encode just writes the SAME pooled value to every
    # head's slot instead of an independent one). The arena stores one physical copy
    # using the serving configuration's scale dtype (FP16 for the published TaSQ setup).

    def layer(self, idx: int):
        return (
            self.forward[idx],
            self.q_map[idx],
            self.mean[idx],
            self.cb16[idx],
            self.cb_sq[idx],
            self.cb_packed[idx],
        )


def load_vq_codebook(
    path: str,
    *,
    layer_num: int,
    start_layer: int,
    head_num: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    head_start: int = 0,
    fold_decode: bool = False,
) -> VQCodebook:
    """``head_num`` is the pool's LOCAL head count; under tensor parallelism
    the bundle holds all global KV heads and ``head_start`` selects this
    rank's contiguous slice (Megatron-style sharding: rank r owns heads
    [r*local, (r+1)*local))."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("loop_coupled") is not None:
        return _load_loop_coupled(
            blob, path, layer_num=layer_num, start_layer=start_layer,
            head_num=head_num, head_dim=head_dim, device=device, dtype=dtype,
            head_start=head_start,
        )
    F = blob["forward"]
    inv = blob["inverse"]
    mean = blob["mean"]
    bounds = blob["bounds"]
    ptn = bool(blob.get("pertoken_norm", False))
    pool_heads_scale = bool(blob.get("pool_heads_scale", False))
    perm_rope = bool(blob.get("perm_rope", False))
    source_fp8_fmt = blob.get("fp8_fmt")

    L_total, H_total, D, D2 = F.shape
    assert D == D2 == head_dim, f"codebook head_dim {D} != pool head_dim {head_dim}"
    head_end = head_start + head_num
    assert head_end <= H_total and (head_start == 0 or H_total % head_num == 0), (
        f"codebook has {H_total} heads, rank wants [{head_start}, {head_end}) "
        f"— TP size must divide the KV head count (no replication support)"
    )
    H = head_num
    F = F[:, head_start:head_end]
    inv = inv[:, head_start:head_end]
    mean = mean[:, head_start:head_end]
    if perm_rope:
        freq_idx_full = blob["freq_idx"][:, head_start:head_end]
        w_perm_full = blob["w_perm"][:, head_start:head_end]
    end_layer = start_layer + layer_num
    assert end_layer <= L_total, (
        f"codebook has {L_total} layers, pool wants [{start_layer}, {end_layer})"
    )

    # Flat contiguous groups only (the stratified permutation is folded into
    # ``forward``); the decode kernel assumes uniform (K, G) across groups.
    starts = [s for (s, _e, _b) in bounds]
    ends = [e for (_s, e, _b) in bounds]
    NG = len(bounds)
    G = ends[0] - starts[0]
    assert starts == list(range(0, D, G)) and all(
        e - s == G for s, e in zip(starts, ends)
    ), f"vq2 requires uniform contiguous groups, got bounds={bounds[:4]}..."

    cbs = blob["codebooks"]
    K = cbs[(0, 0)][0].shape[0]
    cb = torch.empty((layer_num, H, NG, K, G), dtype=torch.float16)
    for l in range(layer_num):
        for h in range(H):
            entry = cbs[(start_layer + l, head_start + h)]
            for g in range(NG):
                c = entry[g]
                assert c.shape == (K, G), f"codebook ({l},{h},{g}) shape {c.shape}"
                cb[l, h, g] = c.to(torch.float16)

    # The decode kernel gathers fp16 centroids per group (the "wide-G" layout) at every
    # group dim. cb_packed is a placeholder of shape [layer_num, 1] so call sites that
    # index cb_packed[layer_idx] stay in bounds; it is never read.
    wide_g = True
    cb_packed = torch.zeros((layer_num, 1), dtype=torch.int32)
    cb16 = cb.to(torch.float16)
    centroid_resnap_rel_rmse = 0.0
    fmt = "none"
    cb_sq = 0.5 * cb16.to(torch.float32).pow(2).sum(-1)  # [L, H, NG, K]
    # Index width: uint8 caps codebook_size at 256 (the G=4/bpc=1 arm,
    # K=16, fits trivially). This project's 8-dim CQ/TaSQ codebooks are K=1024
    # (10 bits, 1.25 bit/coord) -- indices >255 would silently wrap in uint8,
    # not error, so this must be sized from K rather than left at the old
    # hardcoded uint8.
    idx_dtype = torch.uint8 if K <= 256 else torch.int16

    Fl = F[start_layer:end_layer].to(dtype)
    q_map = inv[start_layer:end_layer].transpose(-1, -2).to(dtype).contiguous()
    mean_l = mean[start_layer:end_layer].to(dtype)
    freq_idx_l = freq_idx_full[start_layer:end_layer].to(device) if perm_rope else None
    w_perm_l = w_perm_full[start_layer:end_layer].to(torch.float32).to(device) if perm_rope else None

    # Identity detection: exact, on the dtype the kernels actually consume, so a
    # bundle that is identity only after rounding is NOT skipped.
    _D = Fl.shape[-1]
    _eye = torch.eye(_D, dtype=Fl.dtype, device=Fl.device)
    identity_map = bool(
        torch.equal(Fl, _eye.expand_as(Fl))
        and torch.equal(q_map, _eye.expand_as(q_map))
        and not mean_l.any()
    )
    if identity_map:
        print(
            "[vq] identity k/q map detected (forward == q_map == I, mean == 0): "
            "skipping both map kernels",
            flush=True,
        )

    out = VQCodebook(
        forward=Fl.to(device),
        q_map=q_map.to(device),
        identity_map=identity_map,
        mean=mean_l.to(device),
        cb16=cb16.to(device),
        cb_sq=cb_sq.to(device),
        cb_packed=cb_packed.to(device),
        num_groups=NG,
        codebook_size=K,
        group_dim=G,
        pertoken_norm=ptn,
        source_fp8_fmt=source_fp8_fmt,
        centroid_resnap_rel_rmse=centroid_resnap_rel_rmse,
        fp8_fmt=fmt,
        wide_g=wide_g,
        idx_dtype=idx_dtype,
        pool_heads_scale=pool_heads_scale,
        perm_rope=perm_rope,
        freq_idx=freq_idx_l,
        w_perm=w_perm_l,
    )
    logger.info(
        "vq2: loaded codebook %s (layers=%d heads=[%d,%d)/%d NG=%d K=%d "
        "G=%d ptn=%s fp8=%s source_fp8=%s resnap_rel_rmse=%.6f wide_g=%s idx_dtype=%s)",
        path,
        layer_num,
        head_start,
        head_end,
        H_total,
        NG,
        K,
        G,
        ptn,
        fmt,
        source_fp8_fmt,
        centroid_resnap_rel_rmse,
        wide_g,
        idx_dtype,
    )
    if fold_decode:
        _vq_fold_weights_into_decode_table(out)
    out.perm_src = _perm_src(out)
    return out


def _load_loop_coupled(blob, path, *, layer_num, start_layer, head_num, head_dim,
                       device, dtype, head_start) -> VQCodebook:
    """Loop-coupled K bundle (calibration/build_ouro_bundle.py --method loopvq).

    A code covers c channels x Tc loop steps of one base layer, so it is shared by Tc
    engine slots. Two tables come out of one set of centroids:

      decode  cb16 [L, H, D/c, K, c]       each slot's stride-Tc share of the entry, so
              the decode kernel and the prefill dequantise see an ordinary
              (NG'=D/c, G'=c) geometry and never learn about loops;
      encode  cb16_enc [Lc, H, D/c, K, c*Tc]  the whole entry, for assigning codes.

    Centroids stay fp16 (wide_g), so the encode and decode tables describe the same codebook.
    """
    from sglang.srt.mem_cache.loopvq_pack import (
        build_coupled_codebook, code_layer_map, slot_codebooks,
    )

    lc = blob["loop_coupled"]
    Tc, c, Lb = int(lc["loops"]), int(lc["channels"]), int(lc["base_layers"])
    F, inv, mean = blob["forward"], blob["inverse"], blob["mean"]
    L_total, H_total, D, _ = F.shape
    assert D == head_dim, f"codebook head_dim {D} != pool head_dim {head_dim}"
    # Slots without a loop axis (Huginn's prelude / coda) sit after the Lb*T loop slots and
    # carry ordinary per-slot codes of G channels: a different decode geometry (NG = D/G,
    # width G) from the loop slots' (D/c, c). The decode kernel reads the geometry off each
    # layer's table, so cb16 / cb_sq become per-slot lists; the pool gives these slots their
    # own index arena (unified_kv_pool, "plain" slots).
    plain = [int(s) for s in lc.get("plain_slots", [])]
    if Tc == 1 and L_total % Lb == 0:
        # Per-slot bundles (TaSQ: Tc = 1, c = G) give a plain slot the same (D/G, G) geometry
        # as a loop slot -- a one-step loop slot -- so it needs no separate arena, and the
        # transforms (perm_rope, chan_weight, pooled norm) apply to every slot alike.
        plain = []
    L_loop = L_total - len(plain)
    assert plain == list(range(L_loop, L_total)), f"plain slots must follow the loop slots: {plain}"
    assert L_loop % Lb == 0, (L_loop, Lb)
    T = L_loop // Lb
    assert T % Tc == 0, (T, Tc)
    # A code block spans Tc slots that sit Lb apart, so a pool owning only some slots
    # (pipeline parallel) would split blocks across ranks. Heads are independent.
    assert start_layer == 0 and layer_num == L_total, (
        f"loop-coupled K needs every KV slot in one pool; got [{start_layer}, "
        f"{start_layer + layer_num}) of {L_total}")
    assert head_start + head_num <= H_total
    # Supported extras (calibration/build_ouro_bundle.py): the pooled-head per-token norm
    # (pertoken_norm + pool_heads_scale, both or neither) and, for keys, a covariance-grouped
    # channel permutation served as PERM_ROPE (forward = permutation matrix, freq_idx, w_perm=1).
    ptn = bool(blob.get("pertoken_norm", False))
    pool_heads = bool(blob.get("pool_heads_scale", False))
    assert ptn == pool_heads, "loop-coupled bundles support only the pooled-head norm"
    perm_rope = bool(blob.get("perm_rope", False))
    G = c * Tc
    b0 = blob["bounds"][0]
    assert b0[1] - b0[0] == G, f"bounds group width {b0[1] - b0[0]} != c*Tc = {G}"
    hs, he = head_start, head_start + head_num
    eye = torch.eye(D, dtype=F.dtype)
    identity = bool(torch.equal(F[:, hs:he], eye.expand_as(F[:, hs:he]))
                    and torch.equal(inv[:, hs:he], eye.expand_as(inv[:, hs:he]))
                    and not mean[:, hs:he].any())
    assert identity or perm_rope, (
        "a loop-coupled bundle's map must be the identity or a PERM_ROPE permutation")
    if perm_rope:
        assert not mean[:, hs:he].any()
        assert torch.all(blob["w_perm"] == 1), "weighted PERM_ROPE is not supported here"

    cbs = blob["codebooks"]
    if hs or head_num != H_total:
        cbs = {(s, h - hs): v for (s, h), v in cbs.items() if hs <= h < he}
    cb_enc = build_coupled_codebook(cbs, Lb, T, Tc, c, head_num, D)   # [Lc,H,D/c,K,G]
    cb_dec = slot_codebooks(cb_enc, Lb, T, Tc)                        # [L_loop,H,D/c,K,c]
    plain_cb = (torch.stack([torch.stack([torch.stack(cbs[(s, h)]) for h in range(head_num)])
                             for s in plain]).to(torch.float16)        # [P,H,D/G,K,G]
                if plain else None)
    if plain:
        # TaSQ's W -> N -> P apply to plain slots as to loop slots (build_ouro_bundle fits them
        # through fit_loop_slot): P is the per-slot forward map / freq_idx, N the per-slot scale
        # arena, W is undone in the decode table below. The loop-sensitivity scale has no
        # meaning without a loop axis.
        assert blob.get("loop_scale") is None, "loop_scale bundles cannot carry plain slots"
    loop_scale = None
    if blob.get("loop_scale") is not None:
        # codes live in the scaled space s*x; undo s once, in the per-slot decode slices
        loop_scale = blob["loop_scale"][:, hs:he].to(torch.float32)              # [L,H,D]
        assert loop_scale.shape == (L_total, head_num, D), loop_scale.shape
        assert bool((loop_scale > 0).all()), "loop_scale must be positive"
        cb_dec = (cb_dec.to(torch.float32)
                  / loop_scale.view(L_total, head_num, D // c, 1, c)).to(cb_dec.dtype).contiguous()
    chan_weight = None
    if blob.get("chan_weight") is not None:
        # codes live in w*k / RMS(w*k); undo w in the per-slot decode slices (the scale is
        # multiplied back by decode as for any pooled-norm bundle)
        chan_weight = blob["chan_weight"][:, hs:he].to(torch.float32)            # [L,H,D]
        assert chan_weight.shape == (L_total, head_num, D), chan_weight.shape
        assert bool((chan_weight > 0).all()), "chan_weight must be positive"
        cb_dec = (cb_dec.to(torch.float32)
                  / chan_weight[:L_loop].view(L_loop, head_num, D // c, 1, c)).to(cb_dec.dtype).contiguous()
    # plain slots: plain_cb stays the ENCODE table (codes live in the weighted space); the
    # decode table they are read through undoes W like the loop slots' cb_dec
    plain_dec = plain_cb
    if plain and chan_weight is not None:
        plain_dec = (plain_cb.to(torch.float32)
                     / chan_weight[L_loop:].view(len(plain), head_num, D // G, 1, G)).to(plain_cb.dtype).contiguous()
    K = cb_enc.shape[3]
    Lc = cb_enc.shape[0]

    out = VQCodebook(
        forward=F[:, hs:he].to(dtype).to(device),
        q_map=inv[:, hs:he].transpose(-1, -2).to(dtype).contiguous().to(device),
        mean=mean[:, hs:he].to(dtype).to(device),
        # a stacked tensor when every slot is loop-coupled (Ouro), else per-slot lists
        cb16=(cb_dec.to(device) if not plain else
              list(cb_dec.to(device).unbind(0)) + list(plain_dec.to(device).unbind(0))),
        cb_sq=((0.5 * cb_dec.to(torch.float32).pow(2).sum(-1)).to(device) if not plain else
               list((0.5 * cb_dec.to(torch.float32).pow(2).sum(-1)).to(device).unbind(0))
               + list((0.5 * plain_dec.to(torch.float32).pow(2).sum(-1)).to(device).unbind(0))),
        cb_packed=torch.zeros((L_total, 1), dtype=torch.int32, device=device),
        num_groups=D // c,
        codebook_size=K,
        group_dim=c,
        pertoken_norm=ptn,
        pool_heads_scale=pool_heads,
        perm_rope=perm_rope,
        freq_idx=(blob["freq_idx"][:, hs:he].to(device) if perm_rope else None),
        w_perm=(blob["w_perm"][:, hs:he].to(torch.float32).to(device) if perm_rope else None),
        source_fp8_fmt=blob.get("fp8_fmt"),
        centroid_resnap_rel_rmse=0.0,
        fp8_fmt="none",
        wide_g=True,
        idx_dtype=_vq_idx_dtype(K),
        identity_map=identity,
        loop={"Lb": Lb, "T": T, "Tc": Tc, "c": c, "Lc": Lc, "plain": plain, "L_loop": L_loop,
              **({"plain_cb16": plain_cb.to(device),
                  "plain_cb_sq": (0.5 * plain_cb.to(torch.float32).pow(2).sum(-1)).to(device),
                  "plain_ng": D // G, "G": G} if plain else {})},
        cb16_enc=cb_enc.to(device),
        cb_sq_enc=(0.5 * cb_enc.to(torch.float32).pow(2).sum(-1)).to(device),
        code_layer=code_layer_map(Lb, T, Tc) + [None] * len(plain),
        loop_scale=loop_scale.to(device) if loop_scale is not None else None,
        chan_weight=chan_weight.to(device) if chan_weight is not None else None,
    )
    logger.info(
        "vq2: loaded LOOP-COUPLED codebook %s (slots=%d = %d base x %d loops, Tc=%d c=%d "
        "-> %d code layers; decode NG'=%d G'=%d, encode G=%d; K=%d heads=[%d,%d)/%d; "
        "pooled-head norm=%s perm_rope=%s loop_scale=%s chan_weight=%s)",
        path, L_total, Lb, T, Tc, c, Lc, D // c, c, G, K, hs, he, H_total, ptn, perm_rope,
        loop_scale is not None, chan_weight is not None,
    )
    out.perm_src = _perm_src(out)
    return out


def _perm_src(vq: "VQCodebook"):
    """Source-channel index of `forward` when it is an exact permutation with zero mean, else
    None. r = k @ F with F[d, e] = 1 iff d = src[e] is r[e] = k[src[e]]."""
    if vq.identity_map or not vq.perm_rope or bool(vq.mean.any()):
        return None
    F = vq.forward
    if not torch.equal(vq.q_map, F):
        return None
    src = F.argmax(dim=-2)
    if not torch.equal(torch.zeros_like(F).scatter_(-2, src.unsqueeze(-2), 1), F):
        return None
    return src.contiguous()


def _vq_fold_weights_into_decode_table(vq: "VQCodebook") -> None:
    """Fold 1/w into a decode-side copy of the codebook, for perm+weight bundles.

    K SIDE ONLY, which is why the caller opts in rather than this keying on perm_rope alone:
    decode unweights K (its RoPE branch needs plain original channels) and never touches w on
    the V side, so folding a V codebook would scale V by 1/w with nothing to undo it. No V
    bundle sets perm_rope today, so the guard below would skip it anyway -- opting in makes
    that a decision rather than a coincidence.

    Decode reconstructs ``k = cb[idx] * s / w`` with w static per (layer, head, stored
    channel), so the division can be pre-applied to the centroids and the KV loop keeps only
    the multiply.

    The encode table is then RE-DERIVED from the folded one. That is the whole point: fp16
    ``cb/w`` is not exactly ``cb``, so an encoder scoring against the original table would
    pick a centroid that is not the centroid decode reconstructs -- the assignment and the
    reconstruction would describe different codebooks. Re-deriving makes both describe the
    same (slightly perturbed, equally valid) set, and cb_sq must be recomputed with it or the
    argmax mixes a new inner product with a stale norm.

    Verified as a quantiser, not just as arithmetic: held-out reconstruction distortion is
    unchanged to four decimals (exp/kernelopt/gate_fold16c.py, worst ratio 1.0000).
    """
    if not getattr(vq, "perm_rope", False) or getattr(vq, "w_perm", None) is None:
        return
    if bool(torch.all(vq.w_perm == 1)):
        # Unweighted permutation (covariance grouping only): 1/w is 1, so the fold would only
        # reorder the decode table into pair-major -- a layout the geometry-general decode
        # kernel (vqwide_ng) declines. Leave the plain table.
        return
    L, H, NG, K, G = vq.cb16.shape
    w5 = vq.w_perm.view(L, H, NG, 1, G)
    dec = (vq.cb16.to(torch.float32) / w5).to(torch.float16).contiguous()
    vq.cb16 = (dec.to(torch.float32) * w5).to(torch.float16).contiguous()
    vq.cb_sq = (0.5 * vq.cb16.to(torch.float32).pow(2).sum(-1)).contiguous()
    # Reorder the DECODE copy only: [e0,o0,e1,o1,...] -> [e0..e3, o0..o3]. The encode table
    # keeps original order, so assignments are bit-identical; the rotation operands then come
    # out of one G-wide load already halved (gate_pmfused.py).
    dec = torch.cat([dec[..., 0::2], dec[..., 1::2]], dim=-1).contiguous()
    vq.cb16_dec = dec
    vq.cb16_dec_pair_major = True
    logger.info(
        "vq2: folded 1/w into the decode codebook: %s %.0f MB",
        tuple(dec.shape), dec.numel() * dec.element_size() / 2**20,
    )


def vq_map_k(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor, perm_src: torch.Tensor = None
) -> torch.Tensor:
    """r = (k - mean) @ forward, per head. k: [T, H, D] (any float dtype).
    Contiguous output: pool writers view the result as [-1, row_dim].
    """
    if perm_src is not None:
        # VQCodebook.perm_src: forward is a pure permutation and mean == 0
        return torch.gather(
            k.to(forward.dtype), -1, perm_src.unsqueeze(0).expand(k.shape[0], -1, -1))
    return vq_map_k_fused(k, forward, mean)


@triton.jit
def _vq_qmap_kernel(
    Q, QMAP, MEAN, OUT,
    n_rows, QH,
    D: tl.constexpr, GRP: tl.constexpr, BLOCK_T: tl.constexpr,
    HAS_MEAN: tl.constexpr, PREC: tl.constexpr, DP: tl.constexpr,
):
    """out[t, h*GRP+g, :] = (q[t, h*GRP+g, :] - mean[h]) @ q_map[h]

    One program handles BLOCK_T rows of a single KV head, reading and writing
    q in place-strided form. This replaces the view/permute/reshape/bmm/
    permute/reshape/contiguous chain with a single kernel: the profile showed
    the copies, not the GEMM, were the bulk of the cost.

    HAS_MEAN + GRP=1 makes the same kernel serve the K map, whose only
    differences are the centering term and the absent GQA group.
    """
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_i = pid * BLOCK_T + tl.arange(0, BLOCK_T)      # index into T*GRP
    mask_i = offs_i < n_rows
    t = offs_i // GRP
    g = offs_i % GRP
    qh = h * GRP + g
    # DP = next power of two >= D (tl.arange needs one); the padded channels load as 0 and
    # are never stored, so a non-power-of-two head_dim (Huginn's 96) maps exactly
    offs_d = tl.arange(0, DP)
    mask_d = offs_d < D

    base = t[:, None] * (QH * D) + qh[:, None] * D + offs_d[None, :]
    mask = mask_i[:, None] & mask_d[None, :]
    qv = tl.load(Q + base, mask=mask, other=0.0)
    if HAS_MEAN:
        qv = qv - tl.load(MEAN + h * D + offs_d, mask=mask_d, other=0.0)[None, :]
    m = tl.load(QMAP + h * D * D + offs_d[:, None] * D + offs_d[None, :],
                mask=mask_d[:, None] & mask_d[None, :], other=0.0)
    # PREC="ieee" on fp32 inputs: tl.dot would otherwise default to TF32 and
    # make the flag a silent numerics change (rel ~8e-4). fp16/bf16 are
    # bit-identical to the torch path either way.
    o = tl.dot(qv, m, input_precision=PREC)              # fp32 accumulate
    tl.store(OUT + base, o.to(qv.dtype), mask=mask)


def vq_map_q_fused(q: torch.Tensor, q_map: torch.Tensor) -> torch.Tensor:
    """Fused equivalent of vq_map_q."""
    T, QH, D = q.shape
    H = q_map.shape[0]
    GRP = QH // H
    qc = q.to(q_map.dtype)
    if not qc.is_contiguous():
        qc = qc.contiguous()
    out = torch.empty_like(qc)
    n_rows = T * GRP
    BLOCK_T = 16 if n_rows >= 16 else 16   # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(n_rows, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        qc, q_map, qc, out, n_rows, QH,
        D=D, GRP=GRP, BLOCK_T=BLOCK_T, HAS_MEAN=False, DP=triton.next_power_of_2(D),
        PREC="ieee" if qc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_k_fused(
    k: torch.Tensor, forward: torch.Tensor, mean: torch.Tensor
) -> torch.Tensor:
    """Fused vq_map_k: (k - mean) @ forward per head.

    The einsum form runs sub/bmm/contiguous as three kernels over the same
    small tensor; measured under CUDA-graph replay (the serving condition,
    where launch cost is already amortised) that is 2.5-2.9x the fused time
    across n=8..512.
    """
    T, H, D = k.shape
    mean = mean.to(forward.dtype)
    kc = k.to(forward.dtype)
    if not kc.is_contiguous():
        kc = kc.contiguous()
    out = torch.empty_like(kc)
    BLOCK_T = 16                     # tl.dot needs >= 16 rows; mask covers the tail
    grid = (triton.cdiv(T, BLOCK_T), H)
    _vq_qmap_kernel[grid](
        kc, forward, mean, out, T, H,
        D=D, GRP=1, BLOCK_T=BLOCK_T, HAS_MEAN=True, DP=triton.next_power_of_2(D),
        PREC="ieee" if kc.dtype == torch.float32 else "tf32",
        num_warps=4, num_stages=2,
    )
    return out


def vq_map_q(q: torch.Tensor, q_map: torch.Tensor, perm_src: torch.Tensor = None) -> torch.Tensor:
    """q_m = q @ inverse.T per KV head, broadcast over the GQA group.

    q: [T, QH, D]; q_map: [H, D, D] with QH % H == 0. Batched-GEMM (bmm) form:
    ~1.5x faster than the einsum at decode (small T) and bit-identical; brings the
    per-head query map to OSCAR's shared-rotation GEMM efficiency. perm_src
    (VQCodebook.perm_src, [H, D]) replaces the GEMM with a gather.
    """
    T, QH, D = q.shape
    H = q_map.shape[0]
    grp = QH // H
    if perm_src is not None:
        src = perm_src.repeat_interleave(grp, dim=0) if grp > 1 else perm_src
        return torch.gather(q.to(q_map.dtype), -1, src.unsqueeze(0).expand(T, -1, -1))
    qd = (
        q.to(q_map.dtype)
        .view(T, H, grp, D)
        .permute(1, 0, 2, 3)
        .reshape(H, T * grp, D)
    )
    out = torch.bmm(qd, q_map).view(H, T, grp, D).permute(1, 0, 2, 3)
    return out.reshape(T, QH, D).contiguous()


@triton.jit
def _tasq_prep_kernel(R, W, PERM, OUT, SC, stride_row, rows_per_w,
                      HD: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
                      HAS_PERM: tl.constexpr, HAS_NORM: tl.constexpr):
    # BLOCK = next power of two of HD (Huginn: 55 heads x 96); masked lanes load 0
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < HD
    src = offs
    if HAS_PERM:
        src = (offs // D) * D + tl.load(PERM + offs, mask=m, other=0)
    x = tl.load(R + row.to(tl.int64) * stride_row + src, mask=m, other=0.0).to(tl.float32)
    x = x * tl.load(W + (row // rows_per_w).to(tl.int64) * HD + offs, mask=m, other=0.0)
    if HAS_NORM:
        sc = tl.maximum(tl.sqrt(tl.sum(x * x, axis=0) / HD), 1e-8)
        x = x / sc
        tl.store(SC + row, sc)
    tl.store(OUT + row.to(tl.int64) * HD + offs, x, mask=m)


def tasq_prep(r: torch.Tensor, w: torch.Tensor, rows_per_w: int, perm_src: torch.Tensor = None,
              norm: bool = True):
    """TaSQ's encode-side transforms in one pass: optional key permutation (perm_src [H, D],
    as vq_map_k), channel weight W, then the pooled-head RMS norm.

    r: [..., H, D] rows (any float dtype; each row's H*D block contiguous, rows strided);
    w: [Lw, H, D] fp32, row i uses w[i // rows_per_w]. Returns (x fp32 [..., H, D],
    sc fp32 [...] or None). Same math as r[perm] * w / RMS(r[perm] * w) in torch; the RMS
    sum runs in a different order, so the last bits can differ."""
    H, D = r.shape[-2], r.shape[-1]
    HD = H * D
    lead = r.shape[:-2]
    r2 = r.reshape(-1, H, D)
    if r2.stride(2) != 1 or r2.stride(1) != D:
        r2 = r2.contiguous()
    n = r2.shape[0]
    out = torch.empty((n, H, D), dtype=torch.float32, device=r.device)
    sc = torch.empty((n,), dtype=torch.float32, device=r.device) if norm else out
    if n:
        _tasq_prep_kernel[(n,)](
            r2, w, perm_src if perm_src is not None else r2, out, sc, r2.stride(0), rows_per_w,
            HD=HD, D=D, BLOCK=triton.next_power_of_2(HD), HAS_PERM=perm_src is not None,
            HAS_NORM=norm, num_warps=8)
    return out.view(*lead, H, D), (sc.view(lead) if norm else None)


def _vq_idx_dtype(K: int) -> torch.dtype:
    """uint8 fits codebook_size<=256 (the G=4 arm); wider codebooks
    (this project's G=8/K=1024 CQ/TaSQ bundles) need int16 -- silently
    wrapping indices past 255 in uint8 is a correctness bug, not an error."""
    return torch.uint8 if K <= 256 else torch.int16


# --------------------------------------------------------------------------- index packing
# A wide-G bundle has 16 groups of 10-bit indices per (token, kv-head). Stored one-per-int16
# that is 2.0 b/coord and six bits of every index are zero; as a little-endian bitstream it is
# 1.25 b/coord, five int32 words per row. The decode kernel unpacks with two gathered word
# loads and shifts -- ALU against what was a 2-byte load, and it buys the KV capacity.
#
# Gated on the GROUP COUNT, never on the index dtype. A previous attempt keyed on dtype, packed
# the int16 K arena and left the uint8 V arena raw, and the kernel -- which unpacks both --
# rejected the pair. K and V must be packed or not packed together.
VQ_PACK_NG = 16


def vq_idx_is_packed(num_groups: int, codebook_size: int | None = None) -> bool:
    """Whether the arena holds a bitstream rather than one raw index per group.

    8-bit codes (K <= 256) are never packed: the bitstream would be exactly the raw
    bytes, so it buys no capacity, and the geometry-general decode kernel
    (triton_ops/vqwide_ng.py) reads raw [slots, H, NG] arenas only. Without this the
    NG=16 point of an abits=8 sweep (G=8 CQ, G=32 loopvq) would be the one rate laid
    out as int32 words and the one rate that kernel declines.
    """
    if codebook_size is not None and vq_idx_bits(codebook_size) == 8:
        return False
    return num_groups == VQ_PACK_NG


def vq_idx_bits(codebook_size: int) -> int:
    """Narrowest supported index width holding `codebook_size` codewords.

    Derived from the codebook, not configured, and it must agree with the decode kernel's
    own `_bits_for` -- the launcher asserts the arena width it computes this way, so a pool
    that packed everything at 10 bits made the K<=256 V arena (8 bits, 4 words) fail the
    check with a 5-word buffer.
    """
    for b in (8, 9, 10, 12):
        if codebook_size <= (1 << b):
            return b
    raise ValueError(f"codebook of {codebook_size} entries exceeds the supported widths")


def vq_pack_words(codebook_size: int, num_groups: int = VQ_PACK_NG) -> int:
    return (num_groups * vq_idx_bits(codebook_size) + 31) // 32


@triton.jit
def _vq_pack_kernel(IDX, OUT, n_rows,
                    BITS: tl.constexpr, NG: tl.constexpr, WORDS: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    for w in tl.static_range(0, WORDS):
        acc = tl.zeros([BLOCK], dtype=tl.int64)
        for g in tl.static_range(0, NG):
            bit = g * BITS
            gw = bit // 32
            off = bit % 32
            # a group lands in word w either wholly or, when it straddles, as the high part
            touches = (gw == w) or (gw + 1 == w and off + BITS > 32)
            if touches:
                v = tl.load(IDX + rows * NG + g, mask=mask, other=0).to(tl.int64) & (
                    (1 << BITS) - 1
                )
                if gw == w:
                    acc |= v << off
                else:
                    acc |= v >> (32 - off)
        acc &= 0xFFFFFFFF
        acc = tl.where(acc >= (1 << 31), acc - (1 << 32), acc)
        tl.store(OUT + rows * WORDS + w, acc.to(tl.int32), mask=mask)


def vq_pack_idx(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., 16] integer indices -> [..., words] int32 bitstream at `bits` per index."""
    assert idx.shape[-1] == VQ_PACK_NG, f"packing is specialised to NG=16, got {idx.shape[-1]}"
    words = (VQ_PACK_NG * bits + 31) // 32
    flat = idx.contiguous().reshape(-1, VQ_PACK_NG)
    n = flat.shape[0]
    out = torch.empty((n, words), dtype=torch.int32, device=idx.device)
    if n == 0:
        return out.reshape(*idx.shape[:-1], words)
    if idx.is_cuda:
        BLOCK = 256
        _vq_pack_kernel[(triton.cdiv(n, BLOCK),)](
            flat.to(torch.int32), out, n,
            BITS=bits, NG=VQ_PACK_NG, WORDS=words, BLOCK=BLOCK,
        )
        return out.reshape(*idx.shape[:-1], words)
    return _vq_pack_idx_torch(idx, bits)


def _vq_pack_idx_torch(idx: torch.Tensor, bits: int) -> torch.Tensor:
    """CPU reference for :func:`vq_pack_idx`; same bit layout."""
    words = (VQ_PACK_NG * bits + 31) // 32
    flat = idx.long().reshape(-1, VQ_PACK_NG)
    out = torch.zeros(flat.shape[0], words, dtype=torch.int64, device=idx.device)
    for g in range(VQ_PACK_NG):
        bit = g * bits
        w, off = bit // 32, bit % 32
        v = flat[:, g] & ((1 << bits) - 1)
        out[:, w] |= v << off
        if off + bits > 32:
            out[:, w + 1] |= v >> (32 - off)
    out &= 0xFFFFFFFF
    out = torch.where(out >= (1 << 31), out - (1 << 32), out)
    return out.to(torch.int32).reshape(*idx.shape[:-1], words)


def vq_unpack_idx(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., words] int32 -> [..., 16] int64. Read side for the prefill prefix dequantise;
    decode unpacks inside the kernel instead."""
    words = packed.shape[-1]
    flat = packed.long().reshape(-1, words) & 0xFFFFFFFF
    out = torch.zeros(flat.shape[0], VQ_PACK_NG, dtype=torch.int64, device=packed.device)
    for g in range(VQ_PACK_NG):
        bit = g * bits
        w, off = bit // 32, bit % 32
        v = (flat[:, w] >> off) & ((1 << bits) - 1)
        if off + bits > 32:
            v = (v | (flat[:, w + 1] << (32 - off))) & ((1 << bits) - 1)
        out[:, g] = v
    return out.reshape(*packed.shape[:-1], VQ_PACK_NG)


def vq_encode(
    r: torch.Tensor,
    cb16: torch.Tensor,
    cb_sq: torch.Tensor,
    *,
    pertoken_norm: bool,
    pool_heads: bool = False,
    token_chunk: int = 2048,
):
    """Assign residual rows to nearest centroids.

    r: [T, H, D] (bf16/fp16/fp32); cb16: [H, NG, K, G]; cb_sq: [H, NG, K].
    Returns (idx [T, H, NG] uint8 or int16, scale fp32 [T, H]).
    (K<=256 -> indices fit uint8; wider codebooks use int16, see _vq_idx_dtype.)

    pool_heads=True computes ONE RMS scale per token, pooled over ALL KV heads
    and channels, and uses it for every head instead of the pool's own default
    of an independent scale per (token, head). See VQCodebook.pool_heads_scale.

    Nearest neighbor via argmax(<c, x> - 0.5 ||c||^2); token-chunked so the
    [T, H, NG, K] score tensor stays bounded.
    """
    T, H, D = r.shape
    NG, K, G = cb16.shape[1], cb16.shape[2], cb16.shape[3]
    idx_dtype = _vq_idx_dtype(K)
    rf = r.to(torch.float32)
    if pool_heads:
        scale = rf.pow(2).mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)  # [T,1,1]
        scale = scale.expand(T, H, 1)
        rn = (rf / scale).to(torch.float16)
        scale = scale.squeeze(-1)
    elif pertoken_norm:
        scale = rf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        rn = (rf / scale).to(torch.float16)
        scale = scale.squeeze(-1)
    else:
        scale = torch.ones((T, H), dtype=torch.float32, device=r.device)
        rn = rf.to(torch.float16)
    rn = rn.view(T, H, NG, G)
    idx = torch.empty((T, H, NG), dtype=idx_dtype, device=r.device)
    for t0 in range(0, T, token_chunk):
        t1 = min(t0 + token_chunk, T)
        scores = torch.einsum("thgc,hgkc->thgk", rn[t0:t1], cb16).to(torch.float32)
        scores -= cb_sq.unsqueeze(0)
        idx[t0:t1] = scores.argmax(-1).to(idx_dtype)
    return idx, scale


@triton.jit
def _vq_encode_kernel(
    R, CB, CBSQ, SCALE, IDX, VALID,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BLOCK_N: tl.constexpr,
    IDX_WIDE: tl.constexpr, HAS_VALID: tl.constexpr,
):
    """Nearest-centroid assign, fused -- never materialises the score tensor.

    grid = (L*H*NG, cdiv(n_tok, BLOCK_N)). Each program owns one
    (layer, head, group) and BLOCK_N tokens: it loads that group's [K, G]
    centroids once, scores the rows against them in registers, and writes the
    argmax straight to the uint8 index arena. The torch path instead builds an
    [L, n, H, NG, K] fp32 tensor (~76 MB at n=64) and makes three passes over
    it to produce ~0.5 MB of indices.

    The per-token RMS scale is precomputed by the caller (a cheap reduction)
    so it is not recomputed redundantly in all NG group-programs.
    """
    lhg = tl.program_id(0)
    pid_n = tl.program_id(1)
    g = lhg % NG
    h = (lhg // NG) % H
    l = lhg // (NG * H)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    if HAS_VALID:
        # Decode-flush plans are shape-static (bs * flush_interval rows every step,
        # CUDA-graph-safe) but on most steps most rows are INVALID (trash-routed):
        # with synchronized requests, 7 of 8 steps carry zero valid rows, so the
        # encode was doing ~8x redundant work. Rows are request-major, so whole
        # programs are usually uniformly invalid -> exit before touching R or CB.
        vld = tl.load(VALID + offs_n, mask=mask_n, other=0)
        mask_n = mask_n & (vld != 0)
        if tl.max(vld, axis=0) == 0:
            return
    offs_g = tl.arange(0, G)

    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + (g * G + offs_g)[None, :],
        mask=mask_n[:, None], other=0.0,
    ).to(tl.float32)
    sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                 mask=mask_n, other=1.0).to(tl.float32)
    x = x / sc[:, None]

    # K-chunked scoring: the original single-shot form materialised a
    # [BLOCK_N, K, G] broadcast intermediate, which at the g8 geometry
    # (K=1024, G=8) is 8x the K=256/G=4 design point and spills to local
    # memory -- profiled at 88% of ALL decode GPU time (152 ms/call) and the
    # cause of the catastrophic CQ/TaSQ prefill on small-L2 cards. Chunking
    # bounds the intermediate at [BLOCK_N, BK, G] while keeping every score's
    # arithmetic (and the first-max tie-break) bit-identical: scores are
    # computed with the same ops per entry, and the cross-chunk update uses
    # strict >, so the earliest index still wins ties exactly like a full
    # tl.argmax.
    BK: tl.constexpr = 128 if K > 128 else K
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        cbp = (CB + ((l * H + h) * NG + g) * (K * G)
               + offs_kc[:, None] * G + offs_g[None, :])
        c = tl.load(cbp).to(tl.float32)                              # [BK, G]
        csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_kc).to(tl.float32)
        scores = tl.sum(x[:, None, :] * c[None, :, :], axis=2) - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    best = best_idx
    if IDX_WIDE:
        store_val = best[:, None].to(tl.int16)
    else:
        store_val = best[:, None].to(tl.uint8)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n
        + h * stride_i_h + g,
        store_val, mask=mask_n[:, None],
    )


@triton.jit
def _vq_encode_dot_kernel(
    R, CB, CBSQ, SCALE, IDX, VALID,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, GP: tl.constexpr, G_REAL: tl.constexpr,
    BLOCK_N: tl.constexpr, BK: tl.constexpr,
    IDX_WIDE: tl.constexpr, HAS_VALID: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    """Same assignment as :func:`_vq_encode_kernel`, scored as a matmul.

    The broadcast form ``tl.sum(x[:, None, :] * c[None, :, :], axis=2)`` never reaches the
    tensor cores and materialises a [BLOCK_N, BK, G] fp32 intermediate, which is why raising
    BLOCK_N made it slower (16 -> 1.90 ms, 32 -> 3.97, 64 -> 20.09): the intermediate grows
    with BLOCK_N and spills. As ``x @ c.T`` the intermediate is [BLOCK_N, BK], 8x smaller, so
    BK can cover the whole codebook in one chunk.

    tl.dot needs a contraction of at least 16 and G is 8, so CB is zero-padded to GP=16 on the
    host and x's padded lanes come from the load mask (other=0.0) -- no staging tensor. Padding
    contributes exactly zero to every score.

    NOT bit-identical to the broadcast form: tl.dot sums the padded contraction in the MMA's
    order, so near-ties can resolve to the other equally-valid argmax. Measured at 5
    disagreements in 6.3M assignments, every one between codewords whose float64 scores differ
    by <1.4e-07 (below fp32 epsilon). See exp/kernelopt/gate_encode.py.
    """
    pid0 = tl.program_id(0)
    g = pid0 % NG
    h = (pid0 // NG) % H
    l = pid0 // (NG * H)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    if HAS_VALID:
        vld = tl.load(VALID + offs_n, mask=mask_n, other=0)
        mask_n = mask_n & (vld != 0)
        if tl.max(vld, axis=0) == 0:
            return
    offs_g = tl.arange(0, GP)
    gmask = offs_g < G_REAL
    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + g * G_REAL + offs_g[None, :],
        mask=mask_n[:, None] & gmask[None, :], other=0.0,
    ).to(tl.float32)
    # HAS_SCALE=False: the bundle has neither pertoken_norm nor pool_heads_scale (CQ's K and V,
    # TaSQ's V), so the caller's scale is all ones and `x / 1.0` is a per-element divide plus a
    # load that compute nothing.
    if HAS_SCALE:
        sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                     mask=mask_n, other=1.0).to(tl.float32)
        x = x / sc[:, None]
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        c = tl.load(CB + ((l * H + h) * NG + g) * (K * GP)
                    + offs_kc[:, None] * GP + offs_g[None, :]).to(tl.float32)
        csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_kc).to(tl.float32)
        # allow_tf32=False: the operands are fp32 here and TF32's 10-bit mantissa would move
        # far more assignments than the MMA reordering does.
        scores = tl.dot(x, tl.trans(c), allow_tf32=False) - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    if IDX_WIDE:
        store_val = best_idx[:, None].to(tl.int16)
    else:
        store_val = best_idx[:, None].to(tl.uint8)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n + h * stride_i_h + g,
        store_val, mask=mask_n[:, None],
    )


@triton.jit
def _vq_encode_mma_kernel(
    R, CB, CBSQ, SCALE, IDX, VALID,
    n_tok,
    stride_r_l, stride_r_n, stride_r_h,
    stride_s_l, stride_s_n,
    stride_i_l, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    BLOCK_N: tl.constexpr, BK: tl.constexpr,
    IDX_WIDE: tl.constexpr, HAS_VALID: tl.constexpr, HAS_SCALE: tl.constexpr,
):
    """Same assignment as :func:`_vq_encode_kernel` for G % 16 == 0, on fp16 tensor cores.

    The broadcast kernel never reaches the tensor cores at the loop-coupled (G = c x Tc = 16 / 32)
    and wide CQ/TaSQ group sizes. This kernel scores ``x @ c.T`` as two fp16 MMAs with fp32
    accumulation: x (fp32 after the scale) is split into an fp16 head plus an fp16 remainder, so
    the product carries ~22 mantissa bits of x against the exact fp16 centroids (fp32-grade scores).

    NOT bit-identical to the broadcast form (MMA summation order): near-ties can resolve to the
    other equally-valid argmax, never to a worse codeword.
    """
    pid0 = tl.program_id(0)
    g = pid0 % NG
    h = (pid0 // NG) % H
    l = pid0 // (NG * H)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    if HAS_VALID:
        vld = tl.load(VALID + offs_n, mask=mask_n, other=0)
        mask_n = mask_n & (vld != 0)
        if tl.max(vld, axis=0) == 0:
            return
    offs_g = tl.arange(0, G)
    x = tl.load(
        R + l * stride_r_l + offs_n[:, None] * stride_r_n + h * stride_r_h
        + g * G + offs_g[None, :],
        mask=mask_n[:, None], other=0.0,
    ).to(tl.float32)
    if HAS_SCALE:
        sc = tl.load(SCALE + l * stride_s_l + offs_n * stride_s_n + h,
                     mask=mask_n, other=1.0).to(tl.float32)
        x = x / sc[:, None]
    x_hi = x.to(tl.float16)
    x_lo = (x - x_hi.to(tl.float32)).to(tl.float16)
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        ct = tl.trans(tl.load(CB + ((l * H + h) * NG + g) * (K * G)
                              + offs_kc[:, None] * G + offs_g[None, :]))          # fp16 [G, BK]
        csq = tl.load(CBSQ + ((l * H + h) * NG + g) * K + offs_kc).to(tl.float32)
        scores = tl.dot(x_hi, ct, out_dtype=tl.float32)
        scores += tl.dot(x_lo, ct, out_dtype=tl.float32)
        scores = scores - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    if IDX_WIDE:
        store_val = best_idx[:, None].to(tl.int16)
    else:
        store_val = best_idx[:, None].to(tl.uint8)
    tl.store(
        IDX + l * stride_i_l + offs_n[:, None] * stride_i_n + h * stride_i_h + g,
        store_val, mask=mask_n[:, None],
    )


@triton.jit
def _vq_encode_mma_steps_kernel(
    P0, P1, P2, P3, CB, CBSQ, IDX,
    n_tok, stride_n, stride_h, stride_i_n, stride_i_h,
    H,
    NG: tl.constexpr, K: tl.constexpr, G: tl.constexpr, TC: tl.constexpr,
    BLOCK_N: tl.constexpr, BK: tl.constexpr, IDX_WIDE: tl.constexpr,
):
    """:func:`_vq_encode_mma_kernel` for one loop-coupled code layer, reading the Tc steps' rows
    in place instead of a ``torch.stack``-ed copy.

    Entry j of group g is channel g*(G/TC) + j//TC of step j%TC (ouro_cq.group_loop), which is
    exactly what ``torch.stack(steps, -1).view(n, H, D/c, c*Tc)`` lays out -- so the scored x, and
    hence every code, is bit-identical to the stacked path. The stack was a 4-tensor interleaved
    copy of every prompt token per code layer and tier (127 ms of a 16k Ouro-1.4B prefill).
    """
    pid0 = tl.program_id(0)
    g = pid0 % NG
    h = pid0 // NG
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tok
    offs_g = tl.arange(0, G)
    C: tl.constexpr = G // TC
    step = offs_g % TC
    ch = g * C + offs_g // TC
    base = offs_n[:, None] * stride_n + h * stride_h + ch[None, :]
    x = tl.load(P0 + base, mask=mask_n[:, None] & (step == 0)[None, :], other=0.0).to(tl.float32)
    if TC > 1:
        x += tl.load(P1 + base, mask=mask_n[:, None] & (step == 1)[None, :], other=0.0).to(tl.float32)
    if TC > 2:
        x += tl.load(P2 + base, mask=mask_n[:, None] & (step == 2)[None, :], other=0.0).to(tl.float32)
    if TC > 3:
        x += tl.load(P3 + base, mask=mask_n[:, None] & (step == 3)[None, :], other=0.0).to(tl.float32)
    x_hi = x.to(tl.float16)
    x_lo = (x - x_hi.to(tl.float32)).to(tl.float16)
    best_val = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    best_idx = tl.zeros((BLOCK_N,), dtype=tl.int32)
    for k0 in tl.static_range(0, K, BK):
        offs_kc = k0 + tl.arange(0, BK)
        ct = tl.trans(tl.load(CB + (h * NG + g) * (K * G) + offs_kc[:, None] * G + offs_g[None, :]))
        csq = tl.load(CBSQ + (h * NG + g) * K + offs_kc).to(tl.float32)
        scores = tl.dot(x_hi, ct, out_dtype=tl.float32)
        scores += tl.dot(x_lo, ct, out_dtype=tl.float32)
        scores = scores - csq[None, :]
        loc_val = tl.max(scores, axis=1)
        loc_arg = tl.argmax(scores, axis=1).to(tl.int32) + k0
        upd = loc_val > best_val
        best_idx = tl.where(upd, loc_arg, best_idx)
        best_val = tl.where(upd, loc_val, best_val)
    if IDX_WIDE:
        store_val = best_idx[:, None].to(tl.int16)
    else:
        store_val = best_idx[:, None].to(tl.uint8)
    tl.store(IDX + offs_n[:, None] * stride_i_n + h * stride_i_h + g, store_val, mask=mask_n[:, None])


def vq_encode_loop_steps(steps, cb16_l, cb_sq_l):
    """Codes [n, H, NG] for one code layer from its Tc steps' rows ([n, H, D] each), or None when
    the steps cannot be read in place (dtype / layout mismatch, Tc > 4, geometry outside the
    tensor-core path) -- the caller then stacks as before. cb16_l: [H, NG, K, G]."""
    Tc = len(steps)
    if not (1 <= Tc <= 4):
        return None
    s0 = steps[0]
    if s0.dim() != 3 or not all(
        s.dtype == s0.dtype and s.shape == s0.shape and s.stride() == s0.stride()
        and s.device == s0.device and s.stride(-1) == 1 for s in steps
    ):
        return None
    n, H, D = s0.shape
    Hc, NG, K, G = cb16_l.shape
    if Hc != H or G % 16 or K % 64 or G % Tc or NG * (G // Tc) != D:
        return None
    idx = torch.empty((n, H, NG), dtype=_vq_idx_dtype(K), device=s0.device)
    P = list(steps) + [s0] * (4 - Tc)
    BLOCK_N = 64
    _vq_encode_mma_steps_kernel[(H * NG, triton.cdiv(n, BLOCK_N))](
        P[0], P[1], P[2], P[3], _fp16_codebook(cb16_l), cb_sq_l.contiguous(), idx,
        n, s0.stride(0), s0.stride(1), idx.stride(0), idx.stride(1),
        H, NG=NG, K=K, G=G, TC=Tc, BLOCK_N=BLOCK_N, BK=min(K, 256),
        IDX_WIDE=(idx.dtype != torch.uint8), num_warps=4, num_stages=2,
    )
    return idx


_VQ_FP16_CACHE: dict = {}


def _fp16_codebook(cb16: torch.Tensor) -> torch.Tensor:
    """cb16 as a contiguous fp16 tensor (a no-op for the fp16 wide_g tables), cached otherwise."""
    if cb16.dtype == torch.float16 and cb16.is_contiguous():
        return cb16
    key = (cb16.data_ptr(), tuple(cb16.shape), cb16.dtype)
    p = _VQ_FP16_CACHE.get(key)
    if p is None:
        p = cb16.to(torch.float16).contiguous()
        _VQ_FP16_CACHE[key] = p
    return p


_VQ_PAD_CACHE: dict = {}


def _padded_codebook(cb16: torch.Tensor) -> torch.Tensor:
    """[L,H,NG,K,G] -> [L,H,NG,K,16] fp16, zero-padded, cached per source tensor.

    fp16 rather than fp32: same indices, same speed, half the residency (~134 MB for a
    32-layer 8-kv-head K=1024 G=8 bundle instead of 268 MB).
    """
    key = (cb16.data_ptr(), tuple(cb16.shape))
    p = _VQ_PAD_CACHE.get(key)
    if p is None:
        L, H, NG, K, G = cb16.shape
        p = torch.zeros((L, H, NG, K, 16), dtype=torch.float16, device=cb16.device)
        p[..., :G] = cb16.to(torch.float16)
        _VQ_PAD_CACHE[key] = p
    return p


def vq_encode_fused(r, cb16, cb_sq, *, pertoken_norm: bool, pool_heads: bool = False,
                    valid: "torch.Tensor | None" = None):
    """Fused flush encode.

    r: [L, n, H, D]; cb16: [L, H, NG, K, G]; cb_sq: [L, H, NG, K].
    Returns (idx uint8 [L, n, H, NG], scale fp32 [L, n, H]) -- same contract as
    vq_encode. pool_heads: see vq_encode.

    Scores through :func:`_vq_encode_dot_kernel` (tensor cores) when the geometry allows, else
    the broadcast kernel. The index dtype contract is preserved either way -- returning int16 unconditionally would break the uint8 arenas of
    K<=256 bundles, which is the shape the V side uses.
    """
    L, n, H, D = r.shape
    NG, K, G = cb16.shape[2], cb16.shape[3], cb16.shape[4]
    idx_dtype = _vq_idx_dtype(K)
    rc = r.contiguous()
    rf = rc.to(torch.float32)
    if G == 8:
        if pool_heads:
            scale = rf.pow(2).mean(dim=(2, 3)).sqrt().clamp_min(1e-8)
            scale = scale.unsqueeze(-1).expand(L, n, H).contiguous()
        elif pertoken_norm:
            scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8).contiguous()
        else:
            # stride-0 view, not a materialised buffer: with HAS_SCALE=False the kernel never
            # reads it, and callers only need the documented [L, n, H] shape back.
            scale = torch.ones((), dtype=torch.float32, device=r.device).expand(L, n, H)
        has_scale = bool(pool_heads or pertoken_norm)
        idx = torch.empty((L, n, H, NG), dtype=idx_dtype, device=r.device)
        valid_i8 = (valid.to(torch.int8).contiguous() if valid is not None
                    else rf.new_zeros(1).to(torch.int8))
        BLOCK_N = 32
        grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
        _vq_encode_dot_kernel[grid](
            rf, _padded_codebook(cb16), cb_sq.contiguous(), scale, idx, valid_i8,
            n,
            rf.stride(0), rf.stride(1), rf.stride(2),
            scale.stride(0), scale.stride(1),
            idx.stride(0), idx.stride(1), idx.stride(2),
            H, NG=NG, K=K, GP=16, G_REAL=G, BLOCK_N=BLOCK_N, BK=K,
            IDX_WIDE=(idx_dtype != torch.uint8),
            HAS_VALID=(valid is not None),
            HAS_SCALE=has_scale,
            num_warps=8, num_stages=2,
        )
        return idx, scale
    if G % 16 == 0 and K % 64 == 0:
        # fp16 tensor-core scoring for the wide groups (loop-coupled G = c*Tc, wide CQ/TaSQ)
        if pool_heads:
            scale = rf.pow(2).mean(dim=(2, 3)).sqrt().clamp_min(1e-8)
            scale = scale.unsqueeze(-1).expand(L, n, H).contiguous()
        elif pertoken_norm:
            scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8).contiguous()
        else:
            scale = torch.ones((), dtype=torch.float32, device=r.device).expand(L, n, H)
        idx = torch.empty((L, n, H, NG), dtype=idx_dtype, device=r.device)
        valid_i8 = (valid.to(torch.int8).contiguous() if valid is not None
                    else rf.new_zeros(1).to(torch.int8))
        BLOCK_N = 64
        grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
        _vq_encode_mma_kernel[grid](
            rf, _fp16_codebook(cb16), cb_sq.contiguous(), scale, idx, valid_i8,
            n,
            rf.stride(0), rf.stride(1), rf.stride(2),
            scale.stride(0), scale.stride(1),
            idx.stride(0), idx.stride(1), idx.stride(2),
            H, NG=NG, K=K, G=G, BLOCK_N=BLOCK_N, BK=min(K, 256),
            IDX_WIDE=(idx_dtype != torch.uint8),
            HAS_VALID=(valid is not None),
            HAS_SCALE=bool(pool_heads or pertoken_norm),
            num_warps=4, num_stages=2,
        )
        return idx, scale
    if pool_heads:
        scale = rf.pow(2).mean(dim=(2, 3)).sqrt().clamp_min(1e-8)     # [L, n]
        scale = scale.unsqueeze(-1).expand(L, n, H).contiguous()
    elif pertoken_norm:
        scale = rf.pow(2).mean(-1).sqrt().clamp_min(1e-8)             # [L, n, H]
    else:
        scale = torch.ones((L, n, H), dtype=torch.float32, device=r.device)
    idx = torch.empty((L, n, H, NG), dtype=idx_dtype, device=r.device)
    cbc, cbsqc = cb16.contiguous(), cb_sq.contiguous()
    BLOCK_N = 16
    grid = (L * H * NG, triton.cdiv(n, BLOCK_N))
    valid_i8 = valid.to(torch.int8).contiguous() if valid is not None else rf.new_zeros(1).to(torch.int8)
    _vq_encode_kernel[grid](
        rf, cbc, cbsqc, scale, idx, valid_i8,
        n,
        rf.stride(0), rf.stride(1), rf.stride(2),
        scale.stride(0), scale.stride(1),
        idx.stride(0), idx.stride(1), idx.stride(2),
        H, NG=NG, K=K, G=G, BLOCK_N=BLOCK_N,
        IDX_WIDE=(idx_dtype != torch.uint8),
        HAS_VALID=(valid is not None),
        num_warps=4, num_stages=2,
    )
    return idx, scale


def vq_encode_single(r, cb16_l, cb_sq_l, *, pertoken_norm, pool_heads: bool = False):
    """Single-layer adapter for :func:`vq_encode_fused`.

    Drop-in for ``vq_encode`` at the prefill/extend write sites, which work one
    layer at a time: r [T, H, D], cb16_l [H, NG, K, G], cb_sq_l [H, NG, K].
    The unfused path materialises a [chunk, H, NG, K] fp32 score tensor (537 MB
    at the 2048-token chunk) and makes three passes over it; the fused kernel
    keeps the reduction in registers.
    """
    idx, scale = vq_encode_fused(
        r.unsqueeze(0),
        cb16_l.unsqueeze(0),
        cb_sq_l.unsqueeze(0),
        pertoken_norm=pertoken_norm,
        pool_heads=pool_heads,
    )
    return idx.squeeze(0), scale.squeeze(0)


