# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging
import os

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()

logger = logging.getLogger(__name__)


_MIN_BLOCK_KV = 32


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    if _is_hip and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    extra_kargs = {}
    num_stages = 2
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale_withk,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_grouped_kernel_stage1_prerope(
    Q,
    K_Buffer,    # [cache_size, num_kv_heads, L] -- stored PRE-RoPE
    V_Buffer,
    CosSin,      # [max_pos, L] fp32 -- cache_sin_cache layout: [cos(L/2), sin(L/2)]
    Full_Seq_Len,  # [batch] int -- this request's TOTAL sequence length so far
    FreqIdx,     # [H, L//2] int64, only read when PERM_ROPE (dummy otherwise)
    WPerm,       # [H, L] fp32, only read when PERM_ROPE (dummy otherwise)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_cs_pos,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    L: tl.constexpr,
    LV: tl.constexpr,
    PERM_ROPE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
):
    """HP/exact-tier stage-1 for pre-RoPE caches: K_Buffer holds RAW (un-rotated)
    K, as the CQ / TaSQ / LoopVQ bundles are fit (the bf16 tier is pre-RoPE too,
    not just the quantized bulk).
    RoPE is instead applied here, per cached row, using that row's own
    absolute sequence position -- standard NeoX rotate-half, matching
    sglang's own RotaryEmbedding.forward_native exactly (cos/sin from the
    model's own cos_sin_cache, so this cannot drift from whatever
    base/scaling config the model uses).

    HP indices are scattered in position order, so with PREFIX_TOKENS>0 the tier
    is two bands: the sink rows at [0, PREFIX_TOKENS) followed by the recent tail
    at [full_len - recent, full_len). Rows in the first band are at ``offs_n``
    itself; the rest keep the ``Full_Seq_Len``-derived offset, which is exact for
    a contiguous tail of any length. PREFIX_TOKENS==0 makes the first branch
    unreachable, preserving the no-prefix execution path.

    A dedicated copy of :func:`_fwd_grouped_kernel_stage1` rather than a flag
    on it: that function is the exact-tier kernel for every OTHER method
    (the VQ arm, quarot, turboquant), and must not change behavior for
    them.
    """
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < LV
    offs_half = tl.arange(0, BLOCK_D // 2)
    mask_half = offs_half < (L // 2)

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)
    # Offset for the recent band: its rows are the last (hp_len - PREFIX_TOKENS)
    # positions of the sequence. Sink rows bypass this, see abs_pos below.
    pos_offset = tl.load(Full_Seq_Len + cur_batch) - cur_batch_seq_len

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        if PERM_ROPE:
            # HP/exact tier for perm+weight methods (this project's TaSQ):
            # the VQ pool's own _prepare_hp_kv_tensors applies `forward` (permute+
            # weight) before writing HP rows too, same as the quant tier --
            # for identity `forward` (CQ) that is a no-op, which is why this
            # branch is new here and the non-perm path below still matches
            # CQ exactly. Same fix as the quant-tier vqwide kernel: unweight
            # (WPerm) -> rotate with the ORIGINAL per-pair frequency index
            # (FreqIdx) -> no reweight, landing on plain original channels;
            # Q is gathered at those same channels, never mapped.
            offs_pair = tl.arange(0, BLOCK_D // 2)
            mask_pair = offs_pair < (L // 2)
            fidx = tl.load(FreqIdx + cur_kv_head * (L // 2) + offs_pair, mask=mask_pair, other=0).to(tl.int64)
            w_even = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair, mask=mask_pair, other=1.0).to(tl.float32)
            w_odd = tl.load(WPerm + cur_kv_head * L + 2 * offs_pair + 1, mask=mask_pair, other=1.0).to(tl.float32)
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < L
            fidx_n = tl.broadcast_to(fidx.to(tl.int32)[None, :], (BLOCK_N, BLOCK_D // 2))

            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + fidx[None, :]
            offs_q2 = offs_q1 + L // 2
            q1 = tl.load(Q + offs_q1, mask=(mask_h[:, None]) & (mask_pair[None, :]), other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=(mask_h[:, None]) & (mask_pair[None, :]), other=0.0).to(tl.float32)
        else:
            # q1/q2: the two NeoX rotate-half halves of the (already-RoPE'd, at
            # the query's own position) query -- loaded as two separate halves
            # rather than one full load + slice, since Triton tensors don't
            # support mid-axis integer/slice indexing after the fact (only fresh
            # loads with computed offsets).
            offs_q1 = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_half[None, :]
            offs_q2 = offs_q1 + L // 2
            q1 = tl.load(Q + offs_q1, mask=(mask_h[:, None]) & (mask_half[None, :]), other=0.0).to(tl.float32)
            q2 = tl.load(Q + offs_q2, mask=(mask_h[:, None]) & (mask_half[None, :]), other=0.0).to(tl.float32)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_kv_end
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)

            k_base = kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh

            # Sink rows use their absolute prefix positions.
            abs_pos = tl.where(offs_n < PREFIX_TOKENS, offs_n, pos_offset + offs_n)

            if PERM_ROPE:
                # One contiguous row load split into the (even, odd) pair members, and the
                # cos/sin rows loaded contiguously and reordered to each pair's frequency in
                # registers (tl.gather): the same values as strided/scalar gathers from global
                # memory (bit-identical output), 2.3x faster at bs32 (time_window.py).
                k12 = tl.load(
                    K_Buffer + k_base + offs_d[None, :],
                    mask=mask_n[:, None] & mask_d[None, :], other=0.0,
                ).to(tl.float32)
                k1, k2 = tl.split(tl.reshape(k12, (BLOCK_N, BLOCK_D // 2, 2)))
                cs_nat = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + offs_half[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)
                sn_nat = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + (L // 2 + offs_half)[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)
                cs_row = tl.gather(cs_nat, fidx_n, 1)
                sn_row = tl.gather(sn_nat, fidx_n, 1)
                k1u = k1 / w_even[None, :]
                k2u = k2 / w_odd[None, :]
                o1 = k1u * cs_row - k2u * sn_row
                o2 = k2u * cs_row + k1u * sn_row
            else:
                k1 = tl.load(
                    K_Buffer + k_base + offs_half[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)
                k2 = tl.load(
                    K_Buffer + k_base + (L // 2 + offs_half)[None, :],
                    mask=mask_n[:, None] & mask_half[None, :], other=0.0,
                ).to(tl.float32)

                cs_row = tl.load(
                    CosSin + abs_pos[:, None] * stride_cs_pos + offs_half[None, :],
                    mask=mask_n[:, None] & mask_half[None, :],
                    other=0.0,
                ).to(tl.float32)
                sn_row = tl.load(
                    CosSin
                    + abs_pos[:, None] * stride_cs_pos
                    + (L // 2 + offs_half)[None, :],
                    mask=mask_n[:, None] & mask_half[None, :],
                    other=0.0,
                ).to(tl.float32)
                # NeoX rotate-half: o1 = k1*cos - k2*sin, o2 = k2*cos + k1*sin.
                # q . k_rot = q1.o1 + q2.o2 -- computed as two partial dots
                # instead of concatenating o1/o2 into one BLOCK_D-wide row
                # (Triton has no `torch.cat`; `tl.join` interleaves rather
                # than concatenates, which would silently scramble channel
                # order here).
                o1 = k1 * cs_row - k2 * sn_row
                o2 = k2 * cs_row + k1 * sn_row

            qk = tl.dot(q1.to(tl.float16), tl.trans(o1).to(tl.float16))
            qk += tl.dot(q2.to(tl.float16), tl.trans(o2).to(tl.float16))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & mask_n[None, :], qk, float("-inf")
            )

            v = tl.load(
                V_Buffer
                + kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :],
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // LV
        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd_prerope(
    q,
    k_buffer,
    v_buffer,
    cos_sin_cache,
    full_seq_len,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    freq_idx=None,
    w_perm=None,
    prefix_tokens=0,
):
    """Launcher for :func:`_fwd_grouped_kernel_stage1_prerope`. See that
    kernel's docstring -- this is the HP/exact-tier counterpart of
    :func:`_decode_grouped_att_m_fwd_quant_vqwide`'s pre-RoPE handling.

    full_seq_len: [batch] int -- each request's total sequence length so far
    (prefill + generated tokens up to and including this step); combined with
    this tier's own per-request length (from kv_indptr) to get each cached
    row's absolute position.

    prefix_tokens: SGLANG_MIXED_KV_PREFIX_TOKENS. Rows below it are the always-exact
    sink band and are rotated at their own index.

    freq_idx/w_perm != None activates PERM_ROPE (this project's TaSQ): the
    HP/exact buffer holds permuted+weighted (not raw) K, same as the quant
    tier -- see the kernel docstring."""
    perm_rope = freq_idx is not None
    if not perm_rope:
        freq_idx = _dummy(q, torch.int64)
        w_perm = _dummy(q)
    L = k_buffer.shape[-1]
    LV = v_buffer.shape[-1]
    assert L % 2 == 0, "RoPE rotate-half requires an even head_dim"

    BLOCK_D = triton.next_power_of_2(L)
    BLOCK_DV = triton.next_power_of_2(LV)
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # 32 / 4 is slightly faster than 16 / 2 on the Ouro decode step.
    BLOCK = int(os.environ.get("SGL_PREROPE_BLOCK_N", 32))
    BLOCK_H = int(os.environ.get("SGL_PREROPE_BLOCK_H", 4))
    num_warps = int(os.environ.get("SGL_PREROPE_NUM_WARPS", 4))
    num_stages = int(os.environ.get("SGL_PREROPE_NUM_STAGES", 1))

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        max_kv_splits,
    )

    _fwd_grouped_kernel_stage1_prerope[grid](
        q,
        k_buffer,
        v_buffer,
        cos_sin_cache,
        full_seq_len,
        freq_idx,
        w_perm,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        cos_sin_cache.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        L=L,
        LV=LV,
        PERM_ROPE=perm_rope,
        PREFIX_TOKENS=int(prefix_tokens),
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    O_lse,
    v_scale,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum * v_scale,
        mask=mask_d,
    )
    if WRITE_LSE:
        # Per-seq log-sum-exp = e_max + log(e_sum). O_lse has shape
        # [bs, num_heads]; batch stride = stride_obs // Lv = num_heads.
        tl.store(O_lse + cur_batch * (stride_obs // Lv) + cur_head, e_max + tl.log(e_sum))


def _decode_softmax_reducev_fwd(
    logits,
    lse,
    q,
    o,
    v_scale,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    max_kv_splits,
    sinks=None,
    output_lse=None,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None
    WRITE_LSE = output_lse is not None

    extra_kargs = {}
    if _is_hip:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        lse,
        o,
        output_lse,
        v_scale,
        kv_indptr,
        num_kv_splits,
        sinks,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        HAS_SINK=HAS_SINK,
        WRITE_LSE=WRITE_LSE,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale_withk,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        max_kv_splits,
        sm_scale_withk,
        logit_cap,
        xai_temperature_len,
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        output_lse=output_lse,
    )


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    output_lse=None,
):
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            max_kv_splits,
            sm_scale * k_scale,
            v_scale,
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            output_lse=output_lse,
        )


# ---------------------------------------------------------------------------
# INT2 quantized decode attention kernels
# ---------------------------------------------------------------------------
# INT2 packs 4 values per byte (2-bit crumbs).  Storage is head_dim // 4
# packed uint8 bytes.  Unpacking uses masks 0x03, shifts >> 2, >> 4, >> 6.
# ---------------------------------------------------------------------------


@triton.jit
def _fwd_kernel_stage2_unified_vec(
    Mid_O,
    Mid_O_1,
    O,
    O_lse,
    sink_ptr,
    sink_q_ptr,
    sink_mean_ptr,
    v_scale,
    sink_sm_scale,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_sink_qb,
    stride_sink_qh,
    stride_sink_mh,
    TOTAL_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DQ: tl.constexpr,
    QK_HEAD_DIM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    Lv: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SHIFT_VQ_SINK: tl.constexpr,
):
    """Stage 2 of the unified decode: merges the bf16-tier and quant-tier splits by LSE.

    All LSEs are one vector load, the softmax weights come from one max / exp / sum, and the
    valid rows are one masked [BLOCK_S, BLOCK_DV] load reduced over splits.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < TOTAL_SPLITS
    base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    lse = tl.load(Mid_O_1 + base // Lv + offs_s * (stride_mid_os // Lv), mask=mask_s,
                  other=-float("inf"))
    valid = lse > -float("inf")
    e_max = tl.max(lse, axis=0)
    e_max_safe = tl.where(e_max > -float("inf"), e_max, 0.0)
    w = tl.where(valid, tl.exp(lse - e_max_safe), 0.0)
    e_sum = tl.sum(w, axis=0)
    tv = tl.load(Mid_O + base + offs_s[:, None] * stride_mid_os + offs_d[None, :],
                 mask=valid[:, None] & mask_d[None, :], other=0.0)
    acc = tl.sum(tv * w[:, None], axis=0)

    if HAS_SINK:
        if e_sum > 0.0:
            cur_sink = tl.load(sink_ptr + cur_head)
            if SHIFT_VQ_SINK:
                offs_q = tl.arange(0, BLOCK_DQ)
                q = tl.load(
                    sink_q_ptr + cur_batch * stride_sink_qb + cur_head * stride_sink_qh + offs_q,
                    mask=offs_q < QK_HEAD_DIM, other=0.0,
                ).to(tl.float32)
                kv_head = cur_head // KV_GROUP_NUM
                mean = tl.load(sink_mean_ptr + kv_head * stride_sink_mh + offs_q,
                               mask=offs_q < QK_HEAD_DIM, other=0.0).to(tl.float32)
                cur_sink -= tl.sum(q * mean, axis=0) * sink_sm_scale
            e_sum += tl.exp(cur_sink - e_max)

    safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out = tl.where(e_sum > 0.0, acc / safe_e_sum * v_scale, 0.0)
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, out, mask=mask_d)
    if WRITE_LSE:
        lse_out = tl.where(e_sum > 0.0, e_max + tl.log(safe_e_sum), -float("inf"))
        tl.store(O_lse + cur_batch * (stride_obs // Lv) + cur_head, lse_out)


def _unified_stage2(
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    o: torch.Tensor,
    total_splits: int,
    output_lse=None,
    sinks=None,
    sink_q=None,
    sink_mean=None,
    sink_sm_scale=1.0,
):
    batch, head_num = o.shape[0], o.shape[1]
    Lv = o.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)
    shift_vq_sink = sink_q is not None
    assert shift_vq_sink == (sink_mean is not None)
    qk_head_dim = sink_q.shape[-1] if shift_vq_sink else 1
    block_dq = triton.next_power_of_2(qk_head_dim)
    kv_group_num = (
        sink_q.shape[1] // sink_mean.shape[0] if shift_vq_sink else 1
    )
    grid = (batch, head_num)
    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
    extra_kargs = dict(extra_kargs, BLOCK_S=triton.next_power_of_2(int(total_splits)))
    _fwd_kernel_stage2_unified_vec[grid](
        attn_logits,
        attn_lse,
        o,
        output_lse,
        sinks,
        sink_q,
        sink_mean,
        1.0,
        sink_sm_scale,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        o.stride(0),
        o.stride(1),
        sink_q.stride(0) if shift_vq_sink else 0,
        sink_q.stride(1) if shift_vq_sink else 0,
        sink_mean.stride(0) if shift_vq_sink else 0,
        TOTAL_SPLITS=int(total_splits),
        BLOCK_DV=BLOCK_DV,
        BLOCK_DQ=block_dq,
        QK_HEAD_DIM=qk_head_dim,
        KV_GROUP_NUM=kv_group_num,
        Lv=Lv,
        WRITE_LSE=output_lse is not None,
        HAS_SINK=sinks is not None,
        SHIFT_VQ_SINK=shift_vq_sink,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


# --- launch-time environment, read ONCE -------------------------------------------------
# Keyed by (storage pointer, last-dim width): the .all() below is a device sync, so it must
# run once per arena, never per decode step.
_SCALE_ONE_CACHE: dict = {}


_DUMMY_CACHE: dict = {}


def _dummy(like: torch.Tensor, dtype=None, shape=(1,)) -> torch.Tensor:
    """A cached zero tensor for kernel arguments that a launch never reads (PERM_ROPE / trig
    operands of the modes that do not use them). Allocating them per call was a fill kernel each,
    several per layer slot per decode step (~0.34 ms of a 1k bs=1 Ouro-1.4B LoopVQ step)."""
    key = (like.device, dtype or like.dtype, tuple(shape))
    t = _DUMMY_CACHE.get(key)
    if t is None:
        t = _DUMMY_CACHE[key] = torch.zeros(shape, dtype=dtype or like.dtype, device=like.device)
    return t


def _decode_grouped_att_m_fwd_quant_vqwide(
    q,
    k_idx_buffer,   # [cache_size(+1), num_kv_heads, NG] uint8 / int16 codebook indices
    cb16,           # [num_kv_heads, NG, KC, G] fp16
    v_idx_buffer,
    cb16_v,
    k_scales_zeros,
    v_scales_zeros,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
    cos_sin_cache=None,
    pos_offset=0,
    freq_idx=None,
    w_perm=None,
    w_folded=False,
    pair_major_table=False,
    v_buffer=None,
    v_pack_bits=2,
    q_permuted=False,
):
    """Quant-tier stage 1 (codebook-index K and V), served by the CUDA kernels of vqwide_ng."""
    assert freq_idx is None or cos_sin_cache is not None, "PERM_ROPE requires PRE_ROPE (cos_sin_cache)"
    assert cb16_v is not None, "the quant tier needs a V codebook"
    from .vqwide_ng import try_dispatch

    idx_split = {torch.uint8: 1, torch.int16: 2}.get(k_idx_buffer.dtype, 0)
    if not try_dispatch(
        q=q, k_idx=k_idx_buffer, k_cb=cb16, v_idx=v_idx_buffer, v_cb=cb16_v,
        k_sz=k_scales_zeros, v_sz=v_scales_zeros, att_out=att_out,
        att_lse=att_lse, kv_indptr=kv_indptr, kv_indices=kv_indices,
        num_kv_splits=num_kv_splits, max_kv_splits=max_kv_splits,
        sm_scale=sm_scale, cos_sin_cache=cos_sin_cache, pos_offset=pos_offset,
        freq_idx=freq_idx, w_perm=w_perm, logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len, w_folded=w_folded,
        pair_major_table=pair_major_table, q_permuted=q_permuted,
        v_vq=True, idx_split=idx_split,
    ):
        raise NotImplementedError(
            f"vqwide_ng cannot serve this codebook geometry (K {tuple(cb16.shape)}, "
            f"V {tuple(cb16_v.shape)}, index {k_idx_buffer.dtype} {tuple(k_idx_buffer.shape)})"
        )


def decode_attention_fwd_vq2_unified(
    q,
    hp_k_buffer,
    hp_v_buffer,
    quant_k_idx_buffer,
    cb_packed,
    quant_v_buffer,
    quant_k_scales_zeros,
    quant_v_scales_zeros,
    o,
    hp_kv_indptr,
    hp_kv_indices,
    quant_kv_indptr,
    quant_kv_indices,
    attn_logits,
    attn_lse,
    hp_num_kv_splits,
    quant_num_kv_splits,
    hp_max_kv_splits,
    quant_max_kv_splits,
    sm_scale,
    logit_cap=0.0,
    sinks=None,
    sink_q=None,
    sink_mean=None,
    xai_temperature_len=-1,
    cb_packed_v=None,
    quant_v_idx_buffer=None,
    wide_g=False,
    cb16=None,          # decode-side table: the 1/w-folded copy when the bundle
                        # has one, in which case w_folded must be True
    k_cb_folded=False,
    k_cb_pair_major=False,
    cb16_v=None,
    cos_sin_cache=None,
    full_seq_len=None,
    pos_offset=0,
    prefix_tokens=0,
    freq_idx=None,
    w_perm=None,
    v_pack_bits=2,
    q_permuted=None,
):
    """Unified HP + VQ decode attention: the bf16 tier (sink and recent tokens) and the
    quant tier (codebook indices) each run their own stage-1, and the tier-agnostic
    stage-2 merges them. When cb_packed_v/quant_v_idx_buffer are given, the quant
    tier decodes V as group-VQ too (the VQ-V ablation); else int2 V.

    wide_g=True routes the quant tier through the plain-gather vqwide kernel
    (group_dim != 4 codebooks, e.g. this project's own CQ/TaSQ); requires cb16.
    V follows the same rule as the packed path: cb16_v/quant_v_idx_buffer select
    VQ-V, and their absence selects the scalar tier out of ``quant_v_buffer`` at
    ``v_pack_bits`` bits per coordinate (2 = int2 crumbs, 1 = int1).

    cos_sin_cache != None activates pre-RoPE mode for BOTH tiers (this
    project's CQ/TaSQ convention -- the cache holds raw pre-RoPE K
    everywhere, not just in the quant tier): the HP tier routes through
    ``_decode_grouped_att_m_fwd_prerope`` (needs ``full_seq_len``, see that
    function) and the quant tier's rotation uses ``pos_offset`` (the fixed
    PREFIX_TOKENS constant -- the quant tier always starts at that absolute
    position, unlike the HP tier's per-request offset)."""
    pre_rope = cos_sin_cache is not None
    total_splits = hp_max_kv_splits + quant_max_kv_splits
    assert attn_logits.shape[2] == total_splits

    attn_lse.fill_(float("-inf"))

    hp_logits = attn_logits[:, :, :hp_max_kv_splits, :]
    hp_lse = attn_lse[:, :, :hp_max_kv_splits]
    quant_logits = attn_logits[:, :, hp_max_kv_splits:, :]
    quant_lse = attn_lse[:, :, hp_max_kv_splits:]

    kv_group_num = q.shape[1] // hp_k_buffer.shape[1]
    if kv_group_num == 1 and not get_bool_env_var("SGLANG_VQ2_ALLOW_MHA"):
        raise NotImplementedError(
            "vq2 decode currently supports GQA/MQA (kv_group_num > 1) only. "
            "Set SGLANG_VQ2_ALLOW_MHA=1 to run the grouped kernels at "
            "kv_group_num=1 (Ouro is full MHA)."
        )
    # MHA (kv_group_num == 1) runs the SAME grouped kernels: there
    # cur_kv_head = cur_head_id // cdiv(1, BLOCK_H) = cur_head_id, VALID_BLOCK_H
    # = 1 and mask_h keeps exactly the one lane whose head == cur_head_id, with a
    # grid of cdiv(head_num, 1) = head_num programs. Correct, but it uses 1 of
    # BLOCK_H lanes per program, so it is ~BLOCK_H x off peak occupancy.

    if hp_kv_indices.numel() > 0:
        if pre_rope:
            assert full_seq_len is not None, "pre-RoPE mode requires full_seq_len"
            _decode_grouped_att_m_fwd_prerope(
                q,
                hp_k_buffer,
                hp_v_buffer,
                cos_sin_cache,
                full_seq_len,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
                freq_idx=freq_idx,
                w_perm=w_perm,
                prefix_tokens=prefix_tokens,
            )
        else:
            _decode_grouped_att_m_fwd(
                q,
                hp_k_buffer,
                hp_v_buffer,
                hp_logits,
                hp_lse,
                hp_kv_indptr,
                hp_kv_indices,
                hp_num_kv_splits,
                hp_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
            )

    if quant_kv_indices.numel() > 0:
        if wide_g:
            assert cb16 is not None, "wide_g=True requires cb16"
            assert (cb16_v is None) == (quant_v_idx_buffer is None), (
                "cb16_v and quant_v_idx_buffer select VQ-V together; passing one "
                "without the other would silently read the wrong V arena"
            )
            _decode_grouped_att_m_fwd_quant_vqwide(
                q if q_permuted is None else q_permuted,
                quant_k_idx_buffer,
                cb16,
                quant_v_idx_buffer,
                cb16_v,
                quant_k_scales_zeros,
                quant_v_scales_zeros,
                quant_logits,
                quant_lse,
                quant_kv_indptr,
                quant_kv_indices,
                quant_num_kv_splits,
                quant_max_kv_splits,
                sm_scale,
                logit_cap,
                xai_temperature_len,
                cos_sin_cache=cos_sin_cache if pre_rope else None,
                pos_offset=pos_offset,
                freq_idx=freq_idx,
                w_perm=w_perm,
                w_folded=k_cb_folded,
                pair_major_table=k_cb_pair_major,
                v_buffer=quant_v_buffer,
                v_pack_bits=v_pack_bits,
                q_permuted=q_permuted is not None,
            )
        else:
            raise ValueError("the quant tier needs the wide-G (fp16) codebook layout")

    _unified_stage2(
        attn_logits,
        attn_lse,
        o,
        total_splits=total_splits,
        sinks=sinks,
        sink_q=sink_q,
        sink_mean=sink_mean,
        sink_sm_scale=sm_scale,
    )
    return o
