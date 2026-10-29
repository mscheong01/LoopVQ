"""Packed loop-coupled VQ: the layout transforms, as pure functions.

A loopvq code covers `c` channels x `Tc` loop steps of ONE base layer, so it is
shared by the Tc KV slots that base layer occupies -- slot `s = ut*Lb + l` with
`ut // Tc` picking the block. CQ spends (Lb*T) x D/G codes per (token, head) and
loopvq spends (Lb*T/Tc) x D/c, and those are equal, so the two sit at the same
bits/channel; what differs is which axis the codes live on.

The engine reaches both the codebook and the index arena through ONE indirection
each -- `vq.cb16[layer_idx]` and `k_buffer[layer_idx]`. That is the whole reason
this is cheap to wire: make those two indirections slot-aware and every existing
consumer (prefill dequantize, decode, flush readback) keeps working unchanged.

  codebook   slot s reads a per-slot slice of its block's coupled codebook,
             `[H, D/c, KC, c]`, produced once at load time.
  arena      the Tc slots of a block ALIAS one arena row set, `[tokens, H, D/c]`.
             Allocating Lb*n_blocks rows instead of Lb*T is what makes the index
             bytes come out equal to CQ's rather than Tc times larger.

Only the flush has to change for real: it must group the channels of Tc slots into
one vector before assigning a code.

ENTRY LAYOUT -- the thing to get right. `calibration/ouro_cq.py:group_loop` builds
a coupled vector as `x[..., D, Tc].view(..., D // c, c * Tc)`, so position `j` of an
entry is channel `j // Tc` of loop step `j % Tc`: **channel major, ut minor**. A
slot's channels are therefore the stride-Tc positions `m, m+Tc, ...`, NOT a
contiguous window. Everything here is tested against `group_loop` itself rather
than against a restatement of it.
"""
from __future__ import annotations

import torch


def code_layer_of(slot: int, Lb: int, Tc: int) -> int:
    """Engine KV slot -> the code layer whose arena row and codebook it shares.

    slot = ut*Lb + l; the Tc slots with the same (l, ut//Tc) share one code layer,
    indexed block-major: `blk*Lb + l`.
    """
    ut, l = divmod(slot, Lb)
    return (ut // Tc) * Lb + l


def step_of(slot: int, Lb: int, Tc: int) -> int:
    """Which loop step within its block this slot is -- the codebook slice to take."""
    return (slot // Lb) % Tc


def code_layer_map(Lb: int, T: int, Tc: int) -> list[int]:
    return [code_layer_of(s, Lb, Tc) for s in range(Lb * T)]


def build_coupled_codebook(codebooks: dict, Lb: int, T: int, Tc: int, c: int,
                           H: int, D: int, dtype=torch.float16) -> torch.Tensor:
    """Bundle's (slot, head)[group] dict -> [Lc, H, D/c, KC, c*Tc].

    Inverts `calibration/ouro_cq.py:loop_slot_map`, which seats loopvq group `g`
    of base layer `l` at bundle slot `l + (g // NG)*Lb`, group `g % NG`, where
    NG = D/(c*Tc) is how many groups the bundle gives one slot. Lc = Lb*(T/Tc),
    indexed block-major to match `code_layer_of`.
    """
    G = c * Tc
    NG = D // G                      # bundle groups per slot
    NGp = D // c                     # loopvq groups per base layer
    n_blocks = T // Tc
    KC = codebooks[(0, 0)][0].shape[0]
    out = torch.empty(n_blocks * Lb, H, NGp, KC, G, dtype=dtype)
    for blk in range(n_blocks):
        for l in range(Lb):
            for h in range(H):
                for g in range(NGp):
                    slot = l + (blk * Tc + (g // NG)) * Lb
                    out[blk * Lb + l, h, g] = codebooks[(slot, h)][g % NG].to(dtype)
    return out


def slot_codebooks(cb_coupled: torch.Tensor, Lb: int, T: int,
                   Tc: int) -> torch.Tensor:
    """[Lc, H, NGp, KC, G] -> [Lb*T, H, NGp, KC, c], one slice per engine slot.

    Position `j` of an entry is channel `j // Tc` of step `j % Tc`, so slot at step
    `m` takes `[..., m::Tc]`. Done once at load time; the decode kernel then sees an
    ordinary (NG' = D/c, G' = c) geometry and never learns about loops.
    """
    G = cb_coupled.shape[-1]
    assert G % Tc == 0, (G, Tc)
    return torch.stack(
        [cb_coupled[code_layer_of(s, Lb, Tc)][..., step_of(s, Lb, Tc)::Tc]
         for s in range(Lb * T)]
    ).contiguous()


def regroup_for_encode(r: torch.Tensor, Lb: int, T: int, Tc: int,
                       c: int) -> torch.Tensor:
    """[Lb*T, n, H, D] -> [Lc, n, H, D/c, c*Tc], the vectors a code is assigned to.

    `r` is indexed by engine slot (= ut*Lb + l). The output is block-major in Lc so
    it lines up with `code_layer_of`, and the last axis carries the builder's
    (channel, step) flatten.
    """
    L, n, H, D = r.shape
    assert L == Lb * T, (L, Lb, T)
    assert D % c == 0, (D, c)
    n_blocks = T // Tc
    x = r.view(n_blocks, Tc, Lb, n, H, D)        # [blk, m, l, n, H, D]
    x = x.permute(0, 2, 3, 4, 5, 1)              # [blk, l, n, H, D, Tc]
    return x.reshape(n_blocks * Lb, n, H, D // c, c * Tc)


