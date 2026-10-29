"""Geometry-general build of the vqwide stage-1 decode kernel.

`vqwide_cuda_stage1` compiles its CUDA source once with `VQW_GEOM_{NG,KC,L,KVG}`
fixed at one GQA shape (NG=16, KC=1024, L=128, KVG=4) and gates on a one-entry table,
`_SUPPORTED_GEOMS = {(16, 1024, 128, 4)}`. That table -- not the kernel body -- is
what pins the engine to NG=16, i.e. group_dim 8, i.e. exactly 1.0 bit/channel.
Ouro needs the other rates (G in {2,4,8,16,32} -> 4.0 .. 0.25 bit/channel), has
KVG=1 (16 Q heads over 16 KV heads, full MHA) and KC=256 (abits=8), so it misses
the entry on three of four axes.

Three things do not survive a change of geometry, and this module supplies all
three:

1. **Shared memory.** The index stage costs CAP*NG*sizeof(index), which at NG=128
   would want far more than the device's opt-in limit for the compiled-in CAP=1024.
   `choose_cap` picks the largest power of two that fits.

2. **The index dtype.** The source hard-codes int16_t, but the pool stores uint8
   codes whenever the codebook has at most 256 entries -- every abits=8 bundle we
   build. Reading those as int16 would double the index bytes and misstate the bit
   rate, so the dtype is lifted to a macro too.

3. **A separate V geometry.** loopvq couples K across loop steps but leaves V as
   ordinary CQ, so one bundle carries K at (NG'=D/c, G'=c) and V at (NG=D/G, G).
   The source uses one NG for both; this module splits the V-side region onto its
   own NGV/GV/KCV macros. The kernel already runs K (phase A) and V (phase C) as
   separate loops, so the split is mechanical -- only the two shared buffers they
   share have to be sized for the larger of the two.

The kernel body itself is untouched: `_CUDA_SRC` is imported verbatim and rewritten
by anchored substitutions whose match counts are asserted, so there is exactly one
copy of the kernel and a silent drift in the source cannot pass unnoticed.
"""
from __future__ import annotations

import functools
import os
import re

import torch

from .vqwide_cuda_stage1 import _CUDA_SRC as _SRC_ORIG
from .vqwide_cuda_stage1 import _MIN_BLOCK_KV

_THR = 256

# cudaDevAttrMaxSharedMemoryPerBlockOptin by compute capability. Conservative:
# sm_80 is 163 KB and sm_90 227 KB, but nothing here needs more than ~75 KB.
_OPTIN_SHMEM = {(7, 5): 64 * 1024, (8, 0): 163 * 1024, (8, 6): 99 * 1024,
                (8, 7): 99 * 1024, (8, 9): 99 * 1024, (9, 0): 227 * 1024}
_DEFAULT_OPTIN = 48 * 1024

_IDX_DTYPES = {"uint8": "uint8_t", "int16": "int16_t"}
_TORCH_IDX = {"uint8": torch.uint8, "int16": torch.int16}


# --------------------------------------------------------------------------- #
# Source rewrite
# --------------------------------------------------------------------------- #

def _expect(src: str, old: str, new: str, n: int = 1) -> str:
    got = src.count(old)
    if got != n:
        raise RuntimeError(
            f"vqwide_ng: anchor appeared {got} times, expected {n}. The kernel "
            f"source changed under this module; re-check the rewrite before "
            f"trusting it.\n  anchor: {old[:90]!r}")
    return src.replace(old, new)


def _build_src() -> str:
    src = _SRC_ORIG

    # (2) index dtype -> macro. Every int16_t in the source is the index type: the
    # two kernel parameters, the staging buffer's declaration AND its
    # reinterpret_cast, the two sizeof terms, and the two data_ptr<> calls.
    n_idx = src.count("int16_t")
    if n_idx != 8:
        raise RuntimeError(
            f"vqwide_ng: source has {n_idx} int16_t uses, expected 8; confirm each "
            "one is the index type before building a uint8 variant")
    src = src.replace("int16_t", "VQW_IDXT")

    # (3) split the V-side region onto its own geometry. The region runs from the
    # V index restage through the end of phase C -- everything after it (the LSE
    # write) is head-indexed, not group-indexed.
    v_start = "    // ---- restage V index rows over the K ones ----\n"
    v_end = "\n    if (tid < KVGw) {"
    if src.count(v_start) != 1 or src.count(v_end) != 1:
        raise RuntimeError("vqwide_ng: cannot locate the V-side region")
    i, j = src.index(v_start), src.index(v_end)
    vmap = {"NGw": "NGVw", "Gw": "GVw", "KCw": "KCVw"}
    # \b keeps KVGw intact: its "Gw" is preceded by a word character.
    v_region = re.sub(r"\b(NGw|Gw|KCw)\b", lambda m: vmap[m.group(1)], src[i:j])
    src = src[:i] + v_region + src[j:]

    # The two buffers K and V share must hold the larger of the two geometries.
    src = _expect(
        src,
        "    VQW_IDXT*  idx_s = reinterpret_cast<VQW_IDXT*>(tbl_s + KCw * Gw);",
        "    VQW_IDXT*  idx_s = reinterpret_cast<VQW_IDXT*>(tbl_s + TBLw);")
    src = _expect(
        src,
        "    float*    qk_s  = reinterpret_cast<float*>(idx_s + CAPw * NGw);",
        "    float*    qk_s  = reinterpret_cast<float*>(idx_s + CAPw * MAXNGw);")
    src = _expect(
        src,
        "    const int sm = (int)(sizeof(__half) * KCw * Gw\n"
        "                         + sizeof(VQW_IDXT) * CAPw * NGw",
        "    const int sm = (int)(sizeof(__half) * TBLw\n"
        "                         + sizeof(VQW_IDXT) * CAPw * MAXNGw",
        n=2)

    # (4) rotate-half partner without assuming HALF = 2^k (Huginn: D = 96, HALF = 48). For a
    # power-of-two HALF these are the same values as `c & (HALF-1)` / `c ^ HALF`.
    src = _expect(src, "const int f = c & (HALFw - 1);",
                  "const int f = (c < HALFw) ? c : c - HALFw;")
    src = _expect(src, "const int cp = c ^ HALFw;   // partner channel",
                  "const int cp = (c < HALFw) ? c + HALFw : c - HALFw;   // partner channel")

    # (1)+(3) macro header, appended to the existing geometry defines.
    src = _expect(
        src,
        "#define CAPw  VQW_SPLIT_CAP\n",
        "#define CAPw  VQW_SPLIT_CAP\n"
        "#ifndef VQW_GEOM_NGV\n"
        "#define VQW_GEOM_NGV VQW_GEOM_NG\n"
        "#endif\n"
        "#ifndef VQW_GEOM_KCV\n"
        "#define VQW_GEOM_KCV VQW_GEOM_KC\n"
        "#endif\n"
        "#define NGVw  VQW_GEOM_NGV\n"
        "#define KCVw  VQW_GEOM_KCV\n"
        "#define GVw   (Lw / NGVw)\n"
        "/* buffers shared by phase A (K) and phase C (V): size for the larger */\n"
        "#define TBLw   (KCw * Gw > KCVw * GVw ? KCw * Gw : KCVw * GVw)\n"
        "#define MAXNGw (NGw > NGVw ? NGw : NGVw)\n")
    return src


_CUDA_SRC = _build_src()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=8)
def optin_shmem(device=None) -> int:
    # cached: get_device_capability per call was ~60 us of host time on every layer of every step
    env = os.environ.get("SGLANG_VQWIDE_SHMEM")
    if env:
        return int(env)
    return _OPTIN_SHMEM.get(torch.cuda.get_device_capability(device), _DEFAULT_OPTIN)


def idx_dtype_name(kc: int) -> str:
    """Match the pool's rule (vq_codebook._vq_idx_dtype): uint8 up to 256 codes."""
    return "uint8" if kc <= 256 else "int16"


def smem_bytes(ng, kc, l, kvg, cap, ngv=None, kcv=None, thr=_THR) -> int:
    """Mirror of the `sm` expression the rewritten source computes."""
    ngv = ng if ngv is None else ngv
    kcv = kc if kcv is None else kcv
    g, gv, half = l // ng, l // ngv, l // 2
    idx_sz = 1 if idx_dtype_name(max(kc, kcv)) == "uint8" else 2
    return (2 * max(kc * g, kcv * gv)      # tbl_s   __half[TBL]
            + idx_sz * cap * max(ng, ngv)  # idx_s   VQW_IDXT[CAP*MAXNG]
            + 4 * cap * kvg                # qk_s    float[CAP*KVG]
            + 4 * kvg * l                  # q_s     float[KVG*L]
            + 4 * l                        # w_s     float[L]     (MODE==2)
            + 8 * half                     # fi_s    long[HALF]   (MODE==2)
            + 4 * (thr // 32) * kvg        # red_s
            + 256)


def _use_v2(ng, l, ngv) -> bool:
    """vqwide_v2 (sync-light layout) serves every geometry it applies to; the rest keep
    vqwide_cuda_stage1's layout."""
    from . import vqwide_v2
    return vqwide_v2.structural_ok(ng, l, ngv)


@functools.lru_cache(maxsize=1)
def _server_context_len() -> int:
    """The server's --context-length (0 when unknown); SGLANG_VQWIDE_G_CTX overrides (benchmarks)."""
    env = os.environ.get("SGLANG_VQWIDE_G_CTX")
    if env:
        return int(env)
    try:
        from sglang.srt.server_args import get_global_server_args
        return int(get_global_server_args().context_length or 0)
    except Exception:
        return 0


def _group_r(B, ng, kc, l, kvg, ngv, kcv, mode, max_kv_splits) -> int:
    """0: not served by vqwide_g. Else the requests per block: vqwide_g.group_r() when B > 1 and every
    request is guaranteed to fit (CAP / R) x max_kv_splits tokens (the server's context length), 1
    otherwise (the same coverage as vqwide_v2)."""
    from . import vqwide_g
    if not vqwide_g.structural_ok(ng, l, ngv, kvg, kc, kcv, mode):
        return 0
    r = vqwide_g.group_r()
    ctx = _server_context_len()
    if B > 1 and r > 1 and 0 < ctx <= (vqwide_g.cap() // r) * int(max_kv_splits):
        return r
    return 1


def choose_cap(ng, kc, l, kvg, ngv=None, kcv=None, thr=_THR, limit=None,
               cap_max=1024) -> int:
    """Largest power-of-two split cap whose shared-memory footprint fits.

    0 means even one minimum block does not fit -- the geometry cannot be served.
    """
    if limit is None:
        limit = optin_shmem()
    sm_fn = smem_bytes
    if _use_v2(ng, l, ngv):
        from . import vqwide_v2
        sm_fn = vqwide_v2.smem_bytes
        cap_max = max(cap_max, 2048)   # no per-token index staging: only qk + slot per token
    cap = cap_max
    while cap >= _MIN_BLOCK_KV:
        if sm_fn(ng, kc, l, kvg, cap, ngv, kcv, thr if sm_fn is smem_bytes else vqwide_v2.thr()) <= limit:
            return cap
        cap //= 2
    return 0


def structural_ok(ng, kc, l, kvg, ngv=None, kcv=None, mode: int = 1) -> bool:
    """Assumptions the kernel body makes, independent of shared memory.

    `mode` matters for one check: PERM_ROPE (mode 2) walks each K codebook entry in
    pairs, `PPG = G/2`, so it needs an even G. Modes 0 and 1 read entries channel by
    channel and are fine at G=1 -- the geometry loopvq lands on when a group's
    channels-per-slot is 1.
    """
    ngv = ng if ngv is None else ngv
    kcv = kc if kcv is None else kcv
    if min(ng, ngv, kc, kcv, l, kvg) <= 0:
        return False
    if l % ng or l % ngv:              # G = L/NG and GV = L/NGV must be exact
        return False
    if mode == 2 and (l // ng) % 2:
        return False
    if l % 2:                          # rotate-half needs an even head dim (any HALF, see (4))
        return False
    if max(kc, kcv) > 32767:           # index range
        return False
    # The K and V arenas are read through one macro, so they must agree on dtype.
    return idx_dtype_name(kc) == idx_dtype_name(kcv)


def supports(ng, kc, l, kvg, seq_len_max, n_splits, ngv=None, kcv=None,
             thr=_THR, mode=1) -> bool:
    if not structural_ok(ng, kc, l, kvg, ngv, kcv, mode):
        return False
    cap = choose_cap(ng, kc, l, kvg, ngv, kcv, thr)
    if cap == 0:
        return False
    per = ((seq_len_max + n_splits - 1) // n_splits + _MIN_BLOCK_KV - 1) \
        // _MIN_BLOCK_KV * _MIN_BLOCK_KV
    return per <= cap


# --------------------------------------------------------------------------- #
# Build + launch
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=32)
def _ext(ng, kc, l, kvg, cap, ngv, kcv, thr=_THR):
    from torch.utils.cpp_extension import load_inline

    idxt = idx_dtype_name(max(kc, kcv))
    name = (f"sgl_vqwide_ng{ng}_kc{kc}_ngv{ngv}_kcv{kcv}"
            f"_l{l}_kvg{kvg}_c{cap}_t{thr}_{idxt}")
    m = load_inline(
        name=name,
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void vqwide_stage1_cuda(torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " int64_t, double, int64_t, int64_t);\n"
            "void vqwide_stage1_init();"
        ),
        cuda_sources=_CUDA_SRC,
        functions=["vqwide_stage1_cuda", "vqwide_stage1_init"],
        extra_cuda_cflags=[
            "-O3",
            f"-DVQW_GEOM_NG={ng}", f"-DVQW_GEOM_KC={kc}",
            f"-DVQW_GEOM_NGV={ngv}", f"-DVQW_GEOM_KCV={kcv}",
            f"-DVQW_GEOM_L={l}", f"-DVQW_GEOM_KVG={kvg}",
            f"-DVQW_SPLIT_CAP={cap}", f"-DVQW_THR={thr}",
            # Single-token type name: "unsigned char" would split into two ninja
            # arguments and the compile fails with a bare `char` left over.
            f"-DVQW_IDXT={_IDX_DTYPES[idxt]}",
        ],
    )
    m.vqwide_stage1_init()
    return m


def geometry(q, k_cb, v_cb):
    """(NG, KC, NGV, KCV, L, KVG) read off the tensors."""
    h_kv, ng, kc, g = k_cb.shape
    _, ngv, kcv, gv = v_cb.shape
    l = ng * g
    if ngv * gv != l:
        raise ValueError(f"V spans {ngv * gv} channels, K spans {l}")
    return ng, kc, ngv, kcv, l, q.shape[1] // h_kv


def _sz_f32(t: torch.Tensor) -> torch.Tensor:
    """Scale arena -> the fp32 the kernel reads, keeping a broadcast view a view.

    The kernel takes fp32 scales and indexes them through their strides, so a pool
    serving fp16 scales (SCALE_DTYPE=float16, the default) must be converted -- but
    the const-scale arena of a ptn-free bundle is a stride-0 view over a single value,
    and `.contiguous()` / a plain `.float()` would materialise it at slots x heads on
    every layer of every step. Convert the one value and re-expand instead.
    """
    if t.dtype == torch.float32:
        return t
    if t.dim() >= 2 and t.stride(0) == 0 and t.stride(1) == 0:
        return t[:1, :1].float().expand(t.shape)
    return t.float()


_F32_CACHE: dict = {}


def _f32_cached(t: torch.Tensor) -> torch.Tensor:
    """fp32 contiguous copy of a static table (the RoPE cos/sin cache), made once.

    Converting it per call copies the whole [max_pos, D] table on every layer of
    every decode step."""
    if t.dtype == torch.float32 and t.is_contiguous():
        return t
    key = (t.data_ptr(), tuple(t.shape), t.dtype)
    out = _F32_CACHE.get(key)
    if out is None:
        out = _F32_CACHE[key] = t.float().contiguous()
    return out


@functools.lru_cache(maxsize=8)
def _n_sm(dev) -> int:
    return torch.cuda.get_device_properties(dev).multi_processor_count


@functools.lru_cache(maxsize=8)
def _dummies(dev):
    """Placeholders for the unused MODE inputs, allocated once (not per layer per step)."""
    return (torch.zeros(1, 1, device=dev, dtype=torch.float32),
            torch.zeros(1, 1, device=dev, dtype=torch.int64))


_WARNED_BOUND: set = set()

# The split fix-up in try_dispatch depends only on the step's shared metadata (kv_indptr,
# num_kv_splits) and the geometry, so it is identical for every slot of a decode step: a slot skips
# it when the last fix-up written this forward had the same key. The key is reset by new_forward()
# (backend metadata init, eager or capture) and when a CUDA-graph capture begins, so a captured
# graph always contains the fix-up once, before its first slot.
_split_fix_key = None
_was_capturing = False


def new_forward() -> None:
    global _split_fix_key
    _split_fix_key = None


def run(q, k_idx, k_cb, v_idx, v_cb, k_sz, v_sz, kv_indptr, kv_indices,
        num_kv_splits, n_splits, att_out, att_lse, sm_scale,
        cos_sin_cache=None, pos_offset=0, freq_idx=None, w_perm=None, thr=_THR, group=0):
    """Same contract as vqwide_cuda_stage1.run; geometry read off the tensors.

    k_cb is [H_kv, NG, KC, G] and v_cb [H_kv, NGV, KCV, GV] -- the two may differ,
    which is what lets a loop-coupled K share a kernel with an ordinary CQ V.
    """
    ng, kc, ngv, kcv, l, kvg = geometry(q, k_cb, v_cb)
    cap = choose_cap(ng, kc, l, kvg, ngv, kcv, thr)
    if cap == 0:
        raise RuntimeError(
            f"vqwide_ng: geometry (NG={ng}, KC={kc}, NGV={ngv}, KCV={kcv}, L={l}, "
            f"KVG={kvg}) needs "
            f"{smem_bytes(ng, kc, l, kvg, _MIN_BLOCK_KV, ngv, kcv, thr)} B of shared "
            f"memory for even one {_MIN_BLOCK_KV}-token block; device opt-in limit "
            f"is {optin_shmem()} B")
    want = _TORCH_IDX[idx_dtype_name(max(kc, kcv))]
    if k_idx.dtype is not want or v_idx.dtype is not want:
        raise TypeError(
            f"vqwide_ng: KC={kc}/KCV={kcv} imply a {want} index arena (the pool's "
            f"own rule), got k_idx={k_idx.dtype} v_idx={v_idx.dtype}. Reading 8-bit "
            "codes as int16 would double the index bytes and misstate the bit rate.")
    mode = 0 if cos_sin_cache is None else (2 if freq_idx is not None else 1)
    dummy_f, dummy_l = _dummies(q.device)
    if group:
        from . import vqwide_g
        k_s, v_s = k_sz, v_sz
        if k_s.dtype not in (torch.float16, torch.float32) or k_s.dtype != v_s.dtype:
            k_s, v_s = _sz_f32(k_sz), _sz_f32(v_sz)
        if mode == 2:
            vqwide_g.ext(ng, kc, l, ngv, kcv, 2, hasw=not _w_unit(w_perm), grouped=group > 1).vqwide_v2_cuda(
                q, k_idx, k_cb.contiguous(), v_idx, v_cb.contiguous(), k_s, v_s,
                _f32_cached(cos_sin_cache), freq_idx.contiguous(), w_perm.float().contiguous(),
                kv_indptr.int(), kv_indices, num_kv_splits.int(),
                att_out, att_lse, n_splits, float(sm_scale), int(pos_offset), 18 if group > 1 else 2,
            )
            return
        vqwide_g.ext(ng, kc, l, ngv, kcv).vqwide_v2_cuda(
            q, k_idx, k_cb.contiguous(), v_idx, v_cb.contiguous(), k_s, v_s,
            _f32_cached(cos_sin_cache), dummy_l, dummy_f,
            kv_indptr.int(), kv_indices, num_kv_splits.int(),
            att_out, att_lse, n_splits, float(sm_scale), int(pos_offset), 17 if group > 1 else 1,
        )
        return
    if _use_v2(ng, l, ngv):
        from . import vqwide_v2
        # v2 reads scales in their stored dtype (fp16 or fp32), so no per-call conversion
        k_s, v_s = k_sz, v_sz
        if k_s.dtype not in (torch.float16, torch.float32) or k_s.dtype != v_s.dtype:
            k_s, v_s = _sz_f32(k_sz), _sz_f32(v_sz)
        vqwide_v2.ext(ng, kc, l, kvg, cap, ngv, kcv, idx_dtype_name(max(kc, kcv)),
                      vqwide_v2.thr()).vqwide_v2_cuda(
            q, k_idx, k_cb.contiguous(), v_idx, v_cb.contiguous(), k_s, v_s,
            _f32_cached(cos_sin_cache) if cos_sin_cache is not None else dummy_f,
            freq_idx.contiguous() if freq_idx is not None else dummy_l,
            w_perm.float().contiguous() if w_perm is not None else dummy_f,
            kv_indptr.int(), kv_indices, num_kv_splits.int(),
            att_out, att_lse, n_splits, float(sm_scale), int(pos_offset), mode,
        )
        return
    _ext(ng, kc, l, kvg, cap, ngv, kcv, thr).vqwide_stage1_cuda(
        q, k_idx, k_cb.contiguous(), v_idx, v_cb.contiguous(),
        _sz_f32(k_sz), _sz_f32(v_sz),
        _f32_cached(cos_sin_cache) if cos_sin_cache is not None else dummy_f,
        freq_idx.contiguous() if freq_idx is not None else dummy_l,
        w_perm.float().contiguous() if w_perm is not None else dummy_f,
        kv_indptr.int(), kv_indices, num_kv_splits.int(),
        att_out, att_lse, n_splits, float(sm_scale), int(pos_offset), mode,
    )


def try_dispatch(*, q, k_idx, k_cb, v_idx, v_cb, k_sz, v_sz, att_out, att_lse,
                 kv_indptr, kv_indices, num_kv_splits, max_kv_splits, sm_scale,
                 cos_sin_cache=None, pos_offset=0, freq_idx=None, w_perm=None,
                 logit_cap=0.0, xai_temperature_len=-1, w_folded=False,
                 pair_major_table=False, q_permuted=False, v_vq=True,
                 idx_split=2, seq_len_max=None) -> bool:
    """Serve one decode through this kernel, or decline.

    Everything the Triton vqwide path carries beyond plain VQ -- folded tables,
    pair-major layouts, the scalar-V tier, permuted Q, logit caps -- is machinery
    this kernel does not have, so the dispatch declines rather than silently
    serving something different. Returning False leaves the caller's own assert to
    fire, which is the honest outcome: an unsupported geometry should fail loudly,
    not fall back to a path that computes another thing.
    """
    # Needs raw [slots, H, NG] index arenas. The caller's idx_split is keyed on dtype:
    # 2 = int16 raw, 1 = uint8 "split" layout, which for 8-bit codes is exactly raw
    # (NG bytes plus zero spill bytes) -- the arena every abits=8 bundle gets. The
    # width check below confirms there is no spill tail before accepting it.
    if not v_vq or idx_split not in (1, 2):
        return False
    # PERM_ROPE (MODE 2) is served -- a covariance-grouped key permutation with RoPE pairs kept
    # adjacent, read through freq_idx / w_perm. What is
    # still declined is TaSQ's folded / pair-major decode tables and permuted Q, which this
    # kernel has no code for.
    if w_folded or pair_major_table or q_permuted:
        return False
    if freq_idx is not None and (cos_sin_cache is None or w_perm is None):
        return False
    if logit_cap:
        return False
    if xai_temperature_len is not None and xai_temperature_len > 0:
        return False
    try:
        ng, kc, ngv, kcv, l, kvg = geometry(q, k_cb, v_cb)
    except ValueError:
        return False
    if q.shape[-1] != l or q.shape[1] % k_cb.shape[0]:
        return False
    if k_idx.shape[-1] != ng or v_idx.shape[-1] != ngv:
        return False                 # packed or spill-tailed arena, not raw
    # Same rule run() launches with. This used to say `else 2`, which applied the
    # PERM_ROPE even-G check to plain pre-RoPE and so declined G'=1 (loopvq 4.0 and
    # 2.0 bit/ch) -- unreachable only because nothing ever got this far in-server.
    mode = 0 if cos_sin_cache is None else (2 if freq_idx is not None else 1)
    if not structural_ok(ng, kc, l, kvg, ngv, kcv, mode):
        return False
    cap = choose_cap(ng, kc, l, kvg, ngv, kcv)
    if cap == 0:
        return False
    want = _TORCH_IDX[idx_dtype_name(max(kc, kcv))]
    if k_idx.dtype is not want or v_idx.dtype is not want:
        return False
    # Every split of every request must fit the shared-memory stage of CAP tokens. The engine
    # picks num_kv_splits PER REQUEST (get_num_kv_splits: batch size, SM count), often far
    # below max_kv_splits, so checking only max_kv_splits is not enough -- and a split longer
    # than CAP does not fail, it returns garbage (an illegal-address crash only when it
    # overruns by a lot). Raise each request's split count
    # to the minimum that fits. num_kv_splits is the engine's own tensor, so stage 2 merges
    # exactly the splits stage 1 wrote.
    #
    # Done entirely on the device, with no host read-back, so the decode can be captured in a
    # CUDA graph (a .max()/.any() here was 2 syncs per layer per step and blocked capture). The
    # one condition that cannot be fixed by raising -- a request longer than CAP x max_kv_splits
    # -- is checked statically instead, against the longest sequence the arena can hold.
    B = q.shape[0]
    _mode = 0 if cos_sin_cache is None else (2 if freq_idx is not None else 1)
    gr = _group_r(B, ng, kc, l, q.shape[1] // k_cb.shape[0], ngv, kcv, _mode, max_kv_splits)
    global _split_fix_key, _was_capturing
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and not _was_capturing:
        _split_fix_key = None          # warm-up runs precede capture: re-emit into the graph
    _was_capturing = capturing
    key = (num_kv_splits.data_ptr(), kv_indptr.data_ptr(), B, q.shape[1], k_cb.shape[0],
           cap, int(max_kv_splits), _use_v2(ng, l, ngv),
           os.environ.get("SGLANG_VQWIDE_V2_SPLIT"), gr)
    if key != _split_fix_key:
        if gr:
            _fix_splits_group(q, k_cb, kv_indptr, num_kv_splits, B, max_kv_splits, gr)
        else:
            _fix_splits(q, k_cb, kv_indptr, num_kv_splits, B, cap, max_kv_splits, ng, l, ngv)
        _split_fix_key = key
    bound = int(seq_len_max) if seq_len_max is not None else int(k_idx.shape[0])
    key = (cap, int(max_kv_splits), bound)
    if bound > cap * int(max_kv_splits) and key not in _WARNED_BOUND:
        _WARNED_BOUND.add(key)
        import logging
        logging.getLogger(__name__).warning(
            "[vqwide_ng] CAP=%d x max_kv_splits=%d covers %d tokens but the arena holds %d; a "
            "request longer than that would read past its split -- raise "
            "--triton-attention-num-kv-splits", cap, int(max_kv_splits), cap * int(max_kv_splits),
            bound)

    run(q, k_idx, k_cb, v_idx, v_cb, k_sz, v_sz, kv_indptr, kv_indices,
        num_kv_splits, int(max_kv_splits), att_out, att_lse, sm_scale,
        cos_sin_cache=cos_sin_cache, pos_offset=pos_offset,
        freq_idx=freq_idx, w_perm=w_perm, group=gr)
    return True


_W_UNIT: dict = {}


def _w_unit(w_perm) -> bool:
    """Whether a PERM_ROPE weight tensor is all ones (loop bundles assert it at load), so vqwide_g can
    take its unweighted MODE 2 build. Checked once per tensor (one host sync) and cached by storage;
    a tensor first seen under graph capture is not checked there and takes the weighted build."""
    key = (w_perm.data_ptr(), tuple(w_perm.shape), w_perm.dtype)
    hit = _W_UNIT.get(key)
    if hit is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        hit = _W_UNIT[key] = bool((w_perm == 1).all().item())
    return hit


def _fix_splits_group(q, k_cb, kv_indptr, num_kv_splits, B, max_kv_splits, r) -> None:
    """vqwide_g split counts: every member of a group of r gets its longest request's count (the
    group shares split boundaries), each split at most CAP / r tokens, and the occupancy floor
    (2 waves of 1 block/SM) over the B / r blocks. Device-only, static under capture (B, r fixed)."""
    from . import vqwide_g
    ct = vqwide_g.cap() // r
    seq = kv_indptr[1:B + 1] - kv_indptr[:B]
    nb = -(-B // r)
    if nb * r > B:
        seq = torch.cat([seq, seq.new_zeros(nb * r - B)])
    gmax = seq.view(nb, r).amax(1)
    want = torch.clamp((gmax + (ct - 1)) // ct, min=1, max=int(max_kv_splits))
    occ = 2 * _n_sm(q.device) // (nb * k_cb.shape[0])
    if occ > 1:
        want = torch.maximum(want, torch.clamp((gmax + 255) // 256, min=1, max=occ))
    num_kv_splits[:B].copy_(want.repeat_interleave(r)[:B].to(num_kv_splits.dtype))


def _fix_splits(q, k_cb, kv_indptr, num_kv_splits, B, cap, max_kv_splits, ng, l, ngv) -> None:
    """Raise / set each request's split count so every split fits CAP (see try_dispatch)."""
    seq_lens = kv_indptr[1:B + 1] - kv_indptr[:B]
    cur = num_kv_splits[:B]
    need = torch.clamp((seq_lens + (cap - 1)) // cap, max=int(max_kv_splits))
    if _use_v2(ng, l, ngv):
        # v2 stages a head's whole codebook per block, so short splits pay that 64 KB copy over
        # few tokens: SET the split count from a target split length (default CAP, i.e. the
        # fewest splits that fit) instead of only raising the engine's choice. Stage 2 skips
        # splits that were not written (attn_lse is reset to -inf per call).
        tgt = min(cap, int(os.environ.get("SGLANG_VQWIDE_V2_SPLIT", cap)))
        want_s = torch.clamp((seq_lens + (tgt - 1)) // tgt, min=1, max=int(max_kv_splits))
        # small batches: enough splits to put ~2 blocks on every SM, never below 256 tokens each.
        # B and H are fixed per captured graph, so this is static under capture.
        # FLOOR, not ceil: v2 runs 1 block/SM, so ceil can overshoot two full waves by a few
        # blocks, which then run as a third wave.
        occ = 2 * _n_sm(q.device) // (B * k_cb.shape[0])
        if occ > 1:
            want_s = torch.maximum(want_s, torch.clamp((seq_lens + 255) // 256, min=1, max=occ))
        cur.copy_(torch.maximum(want_s, need).to(cur.dtype))
    else:
        torch.maximum(cur, need.to(cur.dtype), out=cur)
