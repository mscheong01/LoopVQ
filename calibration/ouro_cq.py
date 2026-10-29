"""CQ / LoopVQ codebook fitting for looped models.

CQ (arXiv:2405.03917, Sec. 3.2, Eq. 5-6):
  - non-overlapping groups of contiguous channels, b-bit codes per group (CQ-<c>c<b>b)
  - an independent codebook per channel group, (layer, head, group), never across heads
  - Fisher-guided weighted k-means (Eq. 6): a group vector's weight is the sum of squared dL/dA
    over the group
  - k-means++ initialization, clustering of the raw activations (no normalization)
  - keys are quantized before RoPE

LoopVQ: the groups extend along the loop axis instead of only the channel axis. The KV slots of
Ouro are slot = ut*L + layer, so the T slots of one layer are coupled.
"""
import torch

# ------------------------------------------------------------------ undoing RoPE
def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RopeInverter:
    """Turn the cached post-RoPE keys back into pre-RoPE keys.

    Ouro calls apply_rotary_pos_emb before past_key_value.update(), so the cache holds post-RoPE
    keys. The codebooks quantize pre-RoPE keys, so they are fitted on the inverted keys.
    """

    def __init__(self, rotary_emb):
        self.rot = rotary_emb

    def cos_sin(self, positions, device):
        ref = torch.empty(1, 1, positions.numel(), 1, device=device, dtype=torch.float32)
        cos, sin = self.rot(ref, positions.unsqueeze(0).to(device))
        return cos.unsqueeze(1).float(), sin.unsqueeze(1).float()   # [1,1,S,D]

    @staticmethod
    def unrope(k, cs):
        cos, sin = cs
        return k * cos - rotate_half(k) * sin


# ------------------------------------------------- batched weighted k-means (++)
def _argmin_dist(x, c, chunk):
    NB, N, _ = x.shape
    out = torch.empty(NB, N, dtype=torch.long, device=x.device)
    for i in range(0, N, chunk):
        out[:, i:i + chunk] = torch.cdist(x[:, i:i + chunk], c).argmin(-1)
    return out


def _sq_dist_to(x, c, chunk):
    """||x - c||^2 per sample. c:[NB,1,G].  Chunked over N to avoid a [NB,N,G] temporary."""
    NB, N, _ = x.shape
    out = torch.empty(NB, N, device=x.device, dtype=x.dtype)
    for i in range(0, N, chunk):
        out[:, i:i + chunk] = ((x[:, i:i + chunk] - c) ** 2).sum(-1)
    return out


def _kmeanspp_init(x, w, k, gen, chunk=4096):
    NB, N, G = x.shape
    idx0 = torch.multinomial(w.clamp_min(1e-30), 1, generator=gen)
    cents = [torch.gather(x, 1, idx0.unsqueeze(-1).expand(-1, -1, G))]
    d2 = _sq_dist_to(x, cents[0], chunk)
    for _ in range(1, k):
        p = (w * d2).clamp_min(0)
        p = torch.where(p.sum(1, keepdim=True) > 0, p, torch.ones_like(p))
        i = torch.multinomial(p, 1, generator=gen)
        nc = torch.gather(x, 1, i.unsqueeze(-1).expand(-1, -1, G))
        cents.append(nc)
        torch.minimum(d2, _sq_dist_to(x, nc, chunk), out=d2)
    return torch.cat(cents, dim=1)


def weighted_kmeans(x, w, k, iters=15, seed=0, chunk=512):
    """Fisher-weighted Lloyd (CQ Eq.6). x:[NB,N,G], w:[NB,N] -> [NB,k,G].

    The weights do not change the argmin (a common scalar per vector), so the assignment is the
    L2 nearest centroid and only the centroid update is weighted.
    """
    NB, N, G = x.shape
    gen = torch.Generator(device=x.device).manual_seed(seed)
    c = _kmeanspp_init(x, w, k, gen)
    offs = (torch.arange(NB, device=x.device) * k).unsqueeze(1)
    xf, wf = x.reshape(-1, G), w.reshape(-1)
    for _ in range(iters):
        a = _argmin_dist(x, c, chunk)
        fi = (a + offs).reshape(-1)
        wsum = torch.zeros(NB * k, device=x.device, dtype=x.dtype).index_add_(0, fi, wf)
        num = torch.zeros(NB * k, G, device=x.device, dtype=x.dtype).index_add_(
            0, fi, xf * wf.unsqueeze(1))
        new = (num / wsum.clamp_min(1e-30).unsqueeze(1)).view(NB, k, G)
        dead = (wsum <= 0).view(NB, k)
        if dead.any():
            r = torch.multinomial(w.clamp_min(1e-30), k, replacement=True, generator=gen)
            new[dead] = torch.gather(x, 1, r.unsqueeze(-1).expand(-1, -1, G))[dead]
        if torch.allclose(new, c, atol=1e-6):
            return new
        c = new
    return c


# ------------------------------------------------------------------ grouping
def group_contiguous(x, G):
    """[B,H,S,D] -> [H*(D//G), B*S, G]   (head-major, contiguous channels in a head)

    K-side layout: codebooks[(l,h)][g] of a bundle is the g-th run of G contiguous channels of head h.
    """
    B, H, S, D = x.shape
    return x.view(B, H, S, D // G, G).permute(1, 3, 0, 2, 4).reshape(H * (D // G), B * S, G)


def group_strided(x, G):
    """[B,H,S,D] -> [H*(D//G), B*S, G]   group g = channels {g, g+NG, g+2NG, ...}, NG=D//G

    V-side layout of the fp8 G=4 path: UnifiedInt2HPKVPool reorders V by `_vq_v_perm`
    (= [g + m*NG for g in range(NG) for m in range(G)]) and encodes it contiguously, so a V
    codebook for that path is fitted in these strided coordinates.
    """
    B, H, S, D = x.shape
    NG = D // G
    xg = x.view(B, H, S, G, NG)            # channel c = m*NG + g  ->  (m on the G axis, g on the NG axis)
    return xg.permute(1, 4, 0, 2, 3).reshape(H * NG, B * S, G)


GROUPERS = {"contiguous": group_contiguous, "strided": group_strided}


# ------------------------------------------------------------ loop-axis coupling
def group_loop(per_ut, c):
    """list of Tc x [B,H,S,D] -> [H*(D//c), B*S, c*Tc]   (channel major, loop step minor)

    A LoopVQ group: c contiguous channels of one (layer, head) across Tc loop steps. Where CQ
    couples G channels within one slot, this couples c channels x Tc loop steps; G = c*Tc, so the
    code length and the codebook size are the same.
    """
    x = torch.stack(per_ut, dim=-1)                     # [B,H,S,D,Tc]
    B, H, S, D, Tc = x.shape
    y = x.view(B, H, S, D // c, c * Tc)                 # (c, Tc) flattened, channel major
    return y.permute(1, 3, 0, 2, 4).reshape(H * (D // c), B * S, c * Tc)


def loop_slot_map(base_layer, g, Lb, NG):
    """Map LoopVQ group g to its (slot, group) position in the bundle.

    LoopVQ has D//c groups per base layer, and the bundle has NG = D//G positions per slot. Since
    D//c = Tc * NG, the groups of a base layer fill exactly the Tc slots the layer occupies, which
    is why the code count equals CQ's.
    """
    return base_layer + (g // NG) * Lb, g % NG


def _det_cost(S, idx, eps):
    import numpy as np
    sub = S[np.ix_(idx, idx)] + eps * np.eye(len(idx))
    sign, logdet = np.linalg.slogdet(sub)
    return np.inf if sign <= 0 else float(np.exp(logdet / len(idx)))


def match_units(S, units, size, eps):
    """TaSQ's covariance grouping: merge `units` (lists of coordinate ids) pairwise by min-weight
    perfect matching on det(Sigma_sub)^(1/|sub|) until every group has `size` coordinates."""
    import itertools
    import networkx as nx
    while len(units[0]) < size:
        n = len(units)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i, j in itertools.combinations(range(n), 2):
            G.add_edge(i, j, weight=_det_cost(S, units[i] + units[j], eps))
        m = nx.min_weight_matching(G, weight="weight")
        assert len(m) == n // 2
        units = [units[i] + units[j] for i, j in m]
    return units
