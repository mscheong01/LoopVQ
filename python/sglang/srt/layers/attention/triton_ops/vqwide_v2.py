"""Sync-light vqwide stage-1 for many-small-group geometries (Ouro loopvq / cq).

`vqwide_cuda_stage1` was written for a GQA geometry: NG=16 groups of G=8, KVG=4
query heads per KV head. It stages one group's codebook at a time into shared memory and, in
the V phase, block-reduces every (group, channel) output separately. On Ouro that structure is
the bottleneck rather than memory: loopvq at 0.5 bit/ch serves NG'=32 groups of G'=4 with
KVG=1, so a block runs 32 table stages (64 __syncthreads) for K and 32*4 block reductions
(256 __syncthreads) for V, while only ~n_tok of its 256 threads have work.

This kernel keeps the same three phases and the same contract (att_out / att_lse per split,
identical split arithmetic, MODE 0/1/2), and changes only how work is laid out:

  * codebooks are read straight from global memory through the read-only path. One head's
    tables are KC x L fp16 = 64 KB at KC=256, all heads ~1 MB -- L2-resident -- so staging them
    bought nothing except the syncs.
  * phase A: TPT lanes per token, each lane summing a strided subset of groups, combined by a
    width-TPT shuffle. No barrier inside the loop.
  * phase C: each thread owns one output channel and a stripe of tokens and accumulates the
    whole split in a register; the stripes are merged once through shared memory.
  * scales are read in their stored dtype (fp16 or fp32), so the per-call fp32 conversion of the
    whole scale arena goes away.

Served from vqwide_ng.run (SGLANG_VQWIDE_V2=1, the default).
"""
from __future__ import annotations

import functools

import torch

_THR = 256
_MIN_BLOCK_KV = 32

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>

#define NGw   VQW_GEOM_NG
#define KCw   VQW_GEOM_KC
#define NGVw  VQW_GEOM_NGV
#define KCVw  VQW_GEOM_KCV
#define Lw    VQW_GEOM_L
#define KVGw  VQW_GEOM_KVG
#define Gw    (Lw / NGw)
#define GVw   (Lw / NGVw)
#define HALFw (Lw / 2)
#define PPGw  (Gw / 2)
#define CAPw  VQW_SPLIT_CAP
#define MAXNGw (NGw > NGVw ? NGw : NGVw)
#define THRw  VQW_THR
#define NWw   (THRw / 32)
/* RoPE pairs per lane in phase A (one warp per token) */
#define PPLw  (HALFw / 32)
/* tokens in flight per warp (phase A) / per thread (phase C) */
#ifndef VQW_UA
#define VQW_UA 4
#endif
#ifndef VQW_UC
#define VQW_UC 8
#endif
#define UAw VQW_UA
/* qk_s doubles as phase C's NW x KVG x L partial-row buffer */
#define QKRw  (CAPw * KVGw > NWw * KVGw * Lw ? CAPw * KVGw : NWw * KVGw * Lw)
#define UCw VQW_UC
/* token stripes in phase C: THR threads over L channels */
#define NSw   (THRw / Lw)
#define VQW_MINBK 32
/* one head's whole codebook, K and V in turn: L x KC fp16 (64 KB at KC=256) */
#define TBLw  (NGw * KCw * Gw > NGVw * KCVw * GVw ? NGw * KCw * Gw : NGVw * KCVw * GVw)

__device__ __forceinline__ float q2f(const __half v)        { return __half2float(v); }
__device__ __forceinline__ float q2f(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float q2f(const float v)         { return v; }

template <typename QT, typename IT, typename ST, int MODE>
__global__ __launch_bounds__(THRw) void vqwide_v2_stage1(
    const QT*        __restrict__ Q,        // [B, H_kv*KVG, L]
    const VQW_IDXT*  __restrict__ K_Idx,    // [slots, H_kv, NG]
    const __half*    __restrict__ K_CB,     // [H_kv, NG, KC, G]   contiguous
    const VQW_IDXT*  __restrict__ V_Idx,    // [slots, H_kv, NGV]
    const __half*    __restrict__ V_CB,     // [H_kv, NGV, KCV, GV] contiguous
    const ST*        __restrict__ K_SZ,     // scale at [slot, h, 0]
    const ST*        __restrict__ V_SZ,
    const float*     __restrict__ CosSin,   // [max_pos, L] cos|sin, MODE>=1
    const long*      __restrict__ FreqIdx,  // [H_kv, HALF]  MODE==2
    const float*     __restrict__ WPerm,    // [H_kv, L]     MODE==2
    const int*       __restrict__ kv_indptr,
    const IT*        __restrict__ kv_indices,
    const int*       __restrict__ num_kv_splits,
    float*           __restrict__ Att_Out,
    float*           __restrict__ Att_Lse,
    float sm_scale, int pos_offset,
    long s_q_b, long s_q_h,
    long s_ki_b, long s_ki_h, long s_vi_b, long s_vi_h,
    long s_ksz_b, long s_ksz_h, long s_vsz_b, long s_vsz_h,
    long s_cs_p,
    long s_o_b, long s_o_h, long s_o_s,
    long s_l_b, long s_l_h, long s_l_s)
{
    // grid z is a small host-chosen width and each block walks its request's splits with that
    // stride. Launching z = max_kv_splits (48) put ~18k blocks on the grid at B=24 when 1-2
    // splits had work: the empty ones still reserve this kernel's ~88 KB of shared memory, one
    // block per SM at a time, and draining them cost more than the real blocks (ncu: 225 waves).
    const int bb = blockIdx.x, kvh = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const int kv_start = kv_indptr[bb];
    const int seq_len  = kv_indptr[bb + 1] - kv_start;
    const int splits   = num_kv_splits[bb];
    const int per = ((seq_len + splits - 1) / splits + VQW_MINBK - 1) / VQW_MINBK * VQW_MINBK;
    if (blockIdx.z >= splits || per * (int)blockIdx.z >= seq_len) return;

    extern __shared__ __align__(16) char smem_raw[];
    // tbl_s first: the int4 staging copies need a 16 B-aligned destination
    __half*   tbl_s = reinterpret_cast<__half*>(smem_raw);      // TBL
    float*    q_s   = reinterpret_cast<float*>(tbl_s + TBLw);   // KVG*L
    float*    w_s   = q_s + KVGw * Lw;                          // L        (MODE==2)
    long*     fi_s  = reinterpret_cast<long*>(w_s + Lw);        // HALF     (MODE==2)
    int*      loc_s = reinterpret_cast<int*>(fi_s + HALFw);     // CAP  token -> arena slot
    float*    qk_s  = reinterpret_cast<float*>(loc_s + CAPw);   // QKR: CAP*KVG scores / p
    float*    out_s = qk_s;                                     // phase C partial rows, after p is dead
    float*    vs_s  = qk_s + QKRw;                              // CAP    V scale per token
    float*    red_s = vs_s + CAPw;                              // NW*KVG
    float*    ml_s  = red_s + NWw * KVGw;                       // 2*KVG  (m, l)

    for (int i = tid; i < KVGw * Lw; i += THRw) {
        const int h = i / Lw, c = i % Lw;
        q_s[i] = q2f(Q[(long)bb * s_q_b + (long)(kvh * KVGw + h) * s_q_h + c]);
    }
    if (MODE == 2) {
        // reciprocal once: the per-pair unweighting below is then a multiply, not an fp32 divide
        for (int i = tid; i < Lw; i += THRw)    w_s[i]  = 1.f / WPerm[(long)kvh * Lw + i];
        for (int i = tid; i < HALFw; i += THRw) fi_s[i] = FreqIdx[(long)kvh * HALFw + i];
    }

  for (int bs = blockIdx.z; bs < splits; bs += gridDim.z) {
    const int s_start = per * bs;
    const int s_end   = min(s_start + per, seq_len);
    if (s_end <= s_start) break;          // uniform over the block; later splits are empty too
    const int n_tok = s_end - s_start;    // <= CAPw, guaranteed by the dispatcher
    // Only the slot of each token is staged. Its code rows are read straight from global in
    // phases A and C, where a warp reads one token's row together (one 32 B sector); staging
    // them byte by byte was most of the kernel's fixed cost and capped CAP through shared memory.
    for (int n = tid; n < n_tok; n += THRw) loc_s[n] = (int)kv_indices[kv_start + s_start + n];
    // stage this head's whole K codebook: every gather below then hits shared memory. Read
    // per token from L2 instead, the 2-8 byte pieces cost a 32 B sector each and dominated
    // the kernel (912 us/call at 3 x 8k tokens).
    {
        const int4* src = reinterpret_cast<const int4*>(K_CB + (long)kvh * ((long)NGw * KCw * Gw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGw * KCw * Gw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

    // ---- phase A: scores. One warp per token, each lane owning PPL RoPE pairs ----
    // A pair is the two stored channels that RoPE rotates together and the natural frequency f
    // it lands on: (2p, 2p+1) -> fi[p] for PERM_ROPE, (f, f+HALF) -> f for plain pre-RoPE, the
    // same with no rotation for MODE 0. Then for every mode
    //     q~ . k = q_f (k1 co - k2 si) + q_{f+HALF} (k2 co + k1 si)
    // with co/si at the token's position (co=1, si=0 when MODE 0). Everything about a lane's pairs
    // except the codes is fixed per head, so it lives in registers; the 32 lanes read one token's
    // cos/sin row together. (The previous layout -- TPT lanes per token, groups strided -- read
    // those rows scattered across 4 tokens per instruction and stalled on them: ncu long-scoreboard
    // was over half the samples.)
    // Vector-gather phase A for plain pre-RoPE G'=4 (LoopVQ 0.5 b on Ouro): one lane per code group
    // reads its token's code once and the whole 4-channel entry as one 8-byte shared load, instead
    // of a 2-byte load per channel; RoPE needs no exchange between lanes because
    //     q~ . k = sum_f k_f (q_f co_f + q_{f+H} si_f) + k_{f+H} (q_{f+H} co_f - q_f si_f).
    // Differs from the generic path only in fp32 summation order (att_out 2e-8). G'=8 is slower
    // this way (two tokens per warp), so it keeps the generic path.
    if (MODE == 1 && KVGw == 1 && Gw == 4 && NGw <= 32) {
        constexpr int LPT = NGw;                 // lanes per token: one code group each
        constexpr int TPW = 32 / LPT;            // tokens per warp per trip
        const int sub = lane / LPT, sl = lane % LPT;
        const int ch0 = sl * 4;
        const bool hi = ch0 >= HALFw;
        const int f0 = hi ? ch0 - HALFw : ch0;
        float qx[4], qy[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int f = f0 + j;
            if (!hi) { qx[j] = q_s[f];         qy[j] = q_s[f + HALFw]; }
            else     { qx[j] = q_s[f + HALFw]; qy[j] = -q_s[f]; }
        }
        const __half* tb = tbl_s + sl * KCw * 4;
        for (int n0 = warp * TPW * UAw; n0 < n_tok; n0 += NWw * TPW * UAw) {
            int id[UAw];
            float ksv[UAw], vsv[UAw];
            float co[UAw][4], si[UAw][4];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                ksv[u] = vsv[u] = 0.f; id[u] = 0;
                if (n < n_tok) {
                    const long loc = loc_s[n];
                    if (sl == 0) {
                        ksv[u] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                    id[u] = (int)__ldg(K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b + sl);
                    const float* cs = CosSin + (long)(pos_offset + s_start + n) * s_cs_p;
#pragma unroll
                    for (int j = 0; j < 4; ++j) { co[u][j] = __ldg(cs + f0 + j); si[u][j] = __ldg(cs + HALFw + f0 + j); }
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) { co[u][j] = si[u][j] = 0.f; }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const uint2 raw = *reinterpret_cast<const uint2*>(tb + id[u] * 4);
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
                const float k[4] = {a.x, a.y, b.x, b.y};
                float part = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) part = fmaf(k[j], fmaf(qx[j], co[u][j], qy[j] * si[u][j]), part);
#pragma unroll
                for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                if (sl == 0 && n < n_tok) { qk_s[n] = part * ksv[u] * sm_scale; vs_s[n] = vsv[u]; }
            }
        }
    } else
    {
        int   sg1[PPLw], sg2[PPLw], fq[PPLw];
        float qa[PPLw][KVGw], qb[PPLw][KVGw], wa[PPLw], wb[PPLw];
#pragma unroll
        for (int i = 0; i < PPLw; ++i) {
            const int pp = lane + 32 * i;               // pair index 0..HALF-1
            int c1, c2, f;
            if (MODE == 2) { c1 = 2 * pp; c2 = 2 * pp + 1; f = (int)fi_s[pp]; }
            else           { c1 = pp; c2 = pp + HALFw; f = pp; }
            // table offsets of the two channels, less the code: (g*KC)*G + j
            sg1[i] = (c1 / Gw) * KCw * Gw + c1 % Gw;
            sg2[i] = (c2 / Gw) * KCw * Gw + c2 % Gw;
            fq[i] = f;
            wa[i] = (MODE == 2) ? w_s[c1] : 1.f;
            wb[i] = (MODE == 2) ? w_s[c2] : 1.f;
#pragma unroll
            for (int h = 0; h < KVGw; ++h) { qa[i][h] = q_s[h * Lw + f]; qb[i][h] = q_s[h * Lw + HALFw + f]; }
        }
        int gi1[PPLw], gi2[PPLw];
#pragma unroll
        for (int i = 0; i < PPLw; ++i) {
            const int c1 = (MODE == 2) ? 2 * (lane + 32 * i) : lane + 32 * i;
            gi1[i] = c1 / Gw;
            gi2[i] = ((MODE == 2) ? c1 + 1 : c1 + HALFw) / Gw;
        }
        // UA tokens per warp per trip, every load of all UA issued before any use: one token at a
        // time is a dependent slot -> code -> table chain per token and left the kernel latency-
        // bound. Validity is uniform over the warp (one token per warp), so the shuffles stay legal.
        for (int n0 = warp * UAw; n0 < n_tok; n0 += NWw * UAw) {
            int id1[UAw][PPLw], id2[UAw][PPLw];
            float co[UAw][PPLw], si[UAw][PPLw];
            float ksv[UAw], vsv[UAw];   // lane 0: this token's K / V scale, in flight with the codes
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u;
                ksv[u] = vsv[u] = 0.f;
                if (n < n_tok) {
                    if (lane == 0) {
                        const long loc = loc_s[n];
                        ksv[u] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                    const VQW_IDXT* ir = K_Idx + (long)kvh * s_ki_h + loc_s[n] * (int)s_ki_b;
                    const float* cs = (MODE >= 1) ? CosSin + (long)(pos_offset + s_start + n) * s_cs_p : nullptr;
#pragma unroll
                    for (int i = 0; i < PPLw; ++i) {
                        id1[u][i] = (int)__ldg(ir + gi1[i]);
                        id2[u][i] = (MODE == 2 && Gw % 2 == 0) ? id1[u][i] : (int)__ldg(ir + gi2[i]);
                        co[u][i] = (MODE >= 1) ? __ldg(cs + fq[i]) : 1.f;
                        si[u][i] = (MODE >= 1) ? __ldg(cs + HALFw + fq[i]) : 0.f;
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < PPLw; ++i) { id1[u][i] = id2[u][i] = 0; co[u][i] = si[u][i] = 0.f; }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u;
                float part[KVGw];
#pragma unroll
                for (int h = 0; h < KVGw; ++h) part[h] = 0.f;
#pragma unroll
                for (int i = 0; i < PPLw; ++i) {
                    float k1, k2;
                    if (MODE == 2 && Gw % 2 == 0) {   // the pair is one aligned half2 of the entry
                        const float2 kk = __half22float2(*reinterpret_cast<const __half2*>(tbl_s + sg1[i] + id1[u][i] * Gw));
                        k1 = kk.x * wa[i]; k2 = kk.y * wb[i];
                    } else {
                        k1 = __half2float(tbl_s[sg1[i] + id1[u][i] * Gw]) * wa[i];
                        k2 = __half2float(tbl_s[sg2[i] + id2[u][i] * Gw]) * wb[i];
                    }
                    const float r1 = k1 * co[u][i] - k2 * si[u][i];
                    const float r2 = k2 * co[u][i] + k1 * si[u][i];
#pragma unroll
                    for (int h = 0; h < KVGw; ++h) part[h] = fmaf(qa[i][h], r1, fmaf(qb[i][h], r2, part[h]));
                }
#pragma unroll
                for (int h = 0; h < KVGw; ++h)
                    for (int o = 16; o; o >>= 1) part[h] += __shfl_xor_sync(0xffffffff, part[h], o);
                if (lane == 0 && n < n_tok) {
                    const float ksc = ksv[u] * sm_scale;
#pragma unroll
                    for (int h = 0; h < KVGw; ++h) qk_s[n * KVGw + h] = part[h] * ksc;
                    vs_s[n] = vsv[u];
                }
            }
        }
    }
    __syncthreads();

    // ---- phase B: exact softmax per head over the split ----
#pragma unroll
    for (int h = 0; h < KVGw; ++h) {
        float m = -INFINITY;
        for (int n = tid; n < n_tok; n += THRw) m = fmaxf(m, qk_s[n * KVGw + h]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) red_s[h * NWw + warp] = m;
    }
    __syncthreads();
    if (tid < KVGw) {
        float bm = red_s[tid * NWw];
        for (int w = 1; w < NWw; ++w) bm = fmaxf(bm, red_s[tid * NWw + w]);
        ml_s[tid] = bm;
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < KVGw; ++h) {
        const float mh = ml_s[h];
        float l = 0.f;
        for (int n = tid; n < n_tok; n += THRw) {
            const float e = __expf(qk_s[n * KVGw + h] - mh);
            qk_s[n * KVGw + h] = e * vs_s[n];   // p with the V scale folded in (v^ = cb * vscale)
            l += e;
        }
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
        if (lane == 0) red_s[h * NWw + warp] = l;
    }
    __syncthreads();
    if (tid < KVGw) {
        float bl = 0.f;
        for (int w = 0; w < NWw; ++w) bl += red_s[tid * NWw + w];
        ml_s[KVGw + tid] = bl;
    }
    {   // V codebook over the K one (phase A is past the barrier above the softmax)
        const int4* src = reinterpret_cast<const int4*>(V_CB + (long)kvh * ((long)NGVw * KCVw * GVw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGVw * KCVw * GVw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

    // ---- phase C: thread = (V group g, token stripe). Each thread owns all GV channels of its
    // group, so one code load and GV/2 half2 table reads serve GV outputs; the previous
    // channel-per-thread layout repeated the same code load for every channel of the group.
    //   NGV >= 32: a warp is 32 consecutive groups of one stripe (NGV/32 warps per stripe)
    //   NGV <  32: a warp is 32/NGV stripes of all groups, folded together by shuffles below
    const __half* vcb = tbl_s;
    constexpr int WPS  = NGVw >= 32 ? NGVw / 32 : 1;
    constexpr int SPW  = NGVw >= 32 ? 1 : 32 / NGVw;
    constexpr int NST  = NGVw >= 32 ? NWw / WPS : NWw * SPW;      // stripes
    constexpr int ROWS = NGVw >= 32 ? NST : NWw;                  // partial rows in out_s
    const int g      = NGVw >= 32 ? (warp % WPS) * 32 + lane : lane % NGVw;
    const int stripe = NGVw >= 32 ? warp / WPS : warp * SPW + lane / NGVw;
    const int svb = (int)s_vi_b;
    const VQW_IDXT* vrow = V_Idx + (long)kvh * s_vi_h + g;
    const __half* vg = vcb + g * KCVw * GVw;
    float acc[KVGw][GVw];
#pragma unroll
    for (int h = 0; h < KVGw; ++h)
#pragma unroll
        for (int j = 0; j < GVw; ++j) acc[h][j] = 0.f;
    for (int n0 = stripe; n0 < n_tok; n0 += NST * UCw) {
        int id[UCw];
#pragma unroll
        for (int u = 0; u < UCw; ++u) {
            const int n = n0 + u * NST;
            id[u] = n < n_tok ? (int)__ldg(vrow + loc_s[n] * svb) : -1;
        }
#pragma unroll
        for (int u = 0; u < UCw; ++u) {
            if (id[u] < 0) continue;
            const int n = n0 + u * NST;
            const __half* e = vg + id[u] * GVw;
            float p[KVGw];
#pragma unroll
            for (int h = 0; h < KVGw; ++h) p[h] = qk_s[n * KVGw + h];
            if (GVw % 2 == 0) {
#pragma unroll
                for (int j2 = 0; j2 < GVw / 2; ++j2) {
                    const float2 v2 = __half22float2(reinterpret_cast<const __half2*>(e)[j2]);
#pragma unroll
                    for (int h = 0; h < KVGw; ++h) {
                        acc[h][2 * j2]     = fmaf(p[h], v2.x, acc[h][2 * j2]);
                        acc[h][2 * j2 + 1] = fmaf(p[h], v2.y, acc[h][2 * j2 + 1]);
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < GVw; ++j) {
                    const float vv = __half2float(e[j]);
#pragma unroll
                    for (int h = 0; h < KVGw; ++h) acc[h][j] = fmaf(p[h], vv, acc[h][j]);
                }
            }
        }
    }
    if (NGVw < 32) {   // fold the warp's stripes (lanes g, g+NGV, ...) into lanes 0..NGV-1
#pragma unroll
        for (int h = 0; h < KVGw; ++h)
#pragma unroll
            for (int j = 0; j < GVw; ++j)
                for (int o = NGVw; o < 32; o <<= 1) acc[h][j] += __shfl_xor_sync(0xffffffff, acc[h][j], o);
    }
    const int row = NGVw >= 32 ? stripe : warp;
    __syncthreads();   // every warp is done reading p from qk_s, which out_s overlays
    if (NGVw >= 32 || lane < NGVw) {
#pragma unroll
        for (int h = 0; h < KVGw; ++h)
#pragma unroll
            for (int j = 0; j < GVw; ++j) out_s[(row * KVGw + h) * Lw + g * GVw + j] = acc[h][j];
    }
    __syncthreads();
    float* out_base = Att_Out + (long)bb * s_o_b + (long)bs * s_o_s;
    for (int i = tid; i < KVGw * Lw; i += THRw) {
        const int h = i / Lw, cc = i % Lw;
        float t = 0.f;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) t += out_s[(r * KVGw + h) * Lw + cc];
        out_base[((long)kvh * KVGw + h) * s_o_h + cc] = t / ml_s[KVGw + h];
    }
    if (tid < KVGw) {
        const long qh = (long)kvh * KVGw + tid;
        Att_Lse[(long)bb * s_l_b + qh * s_l_h + (long)bs * s_l_s] = ml_s[tid] + logf(ml_s[KVGw + tid]);
    }
    __syncthreads();                      // shared buffers are reused by the next split
  }
}

static int smem_bytes() {
    return (int)(sizeof(float) * KVGw * Lw + sizeof(float) * Lw + sizeof(long) * HALFw
                 + sizeof(float) * QKRw + sizeof(float) * CAPw + sizeof(float) * NWw * KVGw
                 + sizeof(float) * 2 * KVGw
                 + sizeof(__half) * TBLw + sizeof(int) * CAPw + 64);
}

#define FOR_ALL(X) \
    X(__half, long, float, 0) X(__half, long, float, 1) X(__half, long, float, 2) \
    X(__nv_bfloat16, long, float, 0) X(__nv_bfloat16, long, float, 1) X(__nv_bfloat16, long, float, 2) \
    X(__half, int, float, 0) X(__half, int, float, 1) X(__half, int, float, 2) \
    X(__nv_bfloat16, int, float, 0) X(__nv_bfloat16, int, float, 1) X(__nv_bfloat16, int, float, 2) \
    X(__half, long, __half, 0) X(__half, long, __half, 1) X(__half, long, __half, 2) \
    X(__nv_bfloat16, long, __half, 0) X(__nv_bfloat16, long, __half, 1) X(__nv_bfloat16, long, __half, 2) \
    X(__half, int, __half, 0) X(__half, int, __half, 1) X(__half, int, __half, 2) \
    X(__nv_bfloat16, int, __half, 0) X(__nv_bfloat16, int, __half, 1) X(__nv_bfloat16, int, __half, 2)

void vqwide_v2_init() {
    const int sm = smem_bytes();
#define REG(QT, IT, ST, MODE) TORCH_CHECK(cudaFuncSetAttribute(                    \
        vqwide_v2_stage1<QT, IT, ST, MODE>,                                         \
        cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess,            \
        "vqwide_v2: cudaFuncSetAttribute failed for ", sm, " bytes");
    FOR_ALL(REG)
#undef REG
}

void vqwide_v2_cuda(torch::Tensor q, torch::Tensor k_idx, torch::Tensor k_cb,
                    torch::Tensor v_idx, torch::Tensor v_cb,
                    torch::Tensor k_sz, torch::Tensor v_sz,
                    torch::Tensor cos_sin, torch::Tensor freq_idx, torch::Tensor w_perm,
                    torch::Tensor kv_indptr, torch::Tensor kv_indices,
                    torch::Tensor splits, torch::Tensor att_out,
                    torch::Tensor att_lse, int64_t n_splits, double sm_scale,
                    int64_t pos_offset, int64_t mode) {
    const int B = q.size(0), H_KV = k_idx.size(1);
    // z: enough blocks to cover the SMs ~2x over, never more than n_splits. Static per (B, H), so
    // a captured CUDA graph keeps a valid grid.
    int dev = 0, n_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    const int zw = (int)std::max<int64_t>(1, std::min<int64_t>(n_splits, (2 * n_sm + B * H_KV - 1) / (B * H_KV)));
    dim3 grid(B, H_KV, zw);
    const int sm = smem_bytes();
    TORCH_CHECK(k_sz.scalar_type() == v_sz.scalar_type(), "vqwide_v2: K/V scale dtypes differ");
    // the kernel indexes slots in 32 bits
    TORCH_CHECK(k_idx.numel() < INT_MAX && v_idx.numel() < INT_MAX, "vqwide_v2: index arena too large for int32 offsets");
#define ARGS(IT, ST)                                                           \
        reinterpret_cast<const VQW_IDXT*>(k_idx.data_ptr()),                   \
        reinterpret_cast<const __half*>(k_cb.data_ptr()),                      \
        reinterpret_cast<const VQW_IDXT*>(v_idx.data_ptr()),                   \
        reinterpret_cast<const __half*>(v_cb.data_ptr()),                      \
        reinterpret_cast<const ST*>(k_sz.data_ptr()),                          \
        reinterpret_cast<const ST*>(v_sz.data_ptr()),                          \
        cos_sin.data_ptr<float>(), freq_idx.data_ptr<long>(),                  \
        w_perm.data_ptr<float>(), kv_indptr.data_ptr<int>(),                   \
        reinterpret_cast<const IT*>(kv_indices.data_ptr()),                    \
        splits.data_ptr<int>(),                                                \
        att_out.data_ptr<float>(), att_lse.data_ptr<float>(),                  \
        (float)sm_scale, (int)pos_offset,                                      \
        q.stride(0), q.stride(1),                                              \
        k_idx.stride(0), k_idx.stride(1), v_idx.stride(0), v_idx.stride(1),    \
        k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1),        \
        cos_sin.stride(0),                                                     \
        att_out.stride(0), att_out.stride(1), att_out.stride(2),               \
        att_lse.stride(0), att_lse.stride(1), att_lse.stride(2)
#define GO(QT, IT, ST, MODE) do {                                              \
        vqwide_v2_stage1<QT, IT, ST, MODE>                                     \
            <<<grid, THRw, sm, at::cuda::getCurrentCUDAStream()>>>(            \
                reinterpret_cast<const QT*>(q.data_ptr()), ARGS(IT, ST));      \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                        \
    } while (0)
#define PICKM(QT, IT, ST) do {                                                 \
        if (mode == 0) GO(QT, IT, ST, 0);                                      \
        else if (mode == 1) GO(QT, IT, ST, 1);                                 \
        else GO(QT, IT, ST, 2);                                                \
    } while (0)
#define PICKS(QT, IT) do {                                                     \
        if (k_sz.scalar_type() == at::kHalf) PICKM(QT, IT, __half);            \
        else PICKM(QT, IT, float);                                             \
    } while (0)
    const bool i64 = kv_indices.scalar_type() == at::kLong;
    if (q.scalar_type() == at::kBFloat16) {
        if (i64) PICKS(__nv_bfloat16, long); else PICKS(__nv_bfloat16, int);
    } else {
        if (i64) PICKS(__half, long); else PICKS(__half, int);
    }
#undef PICKS
#undef PICKM
#undef GO
#undef ARGS
}
"""


def thr() -> int:
    """Block size, SGLANG_VQWIDE_V2_THR (default 1024). The whole-codebook stage holds one block
    per SM, so more threads per block is the only way to more resident warps: on a 3090 at B=24 x
    1k tokens, 256 / 512 / 1024 threads ran 0.80 / 0.78 / 0.63 ns per token-head."""
    import os
    env = os.environ.get("SGLANG_VQWIDE_V2_THR")
    if env:
        return int(env)
    # sm_89 (RTX 6000 Ada) is faster at 512 on the Ouro LoopVQ geometries, sm_86 at 1024.
    return 512 if torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9) else 1024


def unroll() -> tuple:
    """Tokens in flight per warp in phase A / per thread in phase C (SGLANG_VQWIDE_V2_UA / _UC,
    default 4 / 8). The best values depend on the GPU."""
    import os
    return (int(os.environ.get("SGLANG_VQWIDE_V2_UA", 4)), int(os.environ.get("SGLANG_VQWIDE_V2_UC", 8)))


def smem_bytes(ng, kc, l, kvg, cap, ngv=None, kcv=None, thr=_THR) -> int:
    """Mirror of the kernel's smem_bytes()."""
    ngv = ng if ngv is None else ngv
    kcv = kc if kcv is None else kcv
    tbl = max(kc, kcv) * l
    qkr = max(cap * kvg, (thr // 32) * kvg * l)
    return (4 * kvg * l + 4 * l + 8 * (l // 2) + 4 * qkr + 4 * cap + 4 * (thr // 32) * kvg
            + 8 * kvg + 2 * tbl + 4 * cap + 64)


def structural_ok(ng, l, ngv) -> bool:
    # phase C maps THR threads over L channels; phase A gives each of 32 lanes HALF/32 pairs
    t = thr()
    return t % l == 0 and t // l >= 1 and (l // 2) % 32 == 0 and l % ngv == 0 and l % ng == 0


@functools.lru_cache(maxsize=32)
def ext(ng, kc, l, kvg, cap, ngv, kcv, idxt, thr=_THR):
    from torch.utils.cpp_extension import load_inline

    ua, uc = unroll()
    name = f"sgl_vqwide_v2_ua{ua}_uc{uc}_ng{ng}_kc{kc}_ngv{ngv}_kcv{kcv}_l{l}_kvg{kvg}_c{cap}_t{thr}_{idxt}"
    m = load_inline(
        name=name,
        cpp_sources=(
            "#include <torch/extension.h>\n"
            "void vqwide_v2_cuda(torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
            " int64_t, double, int64_t, int64_t);\n"
            "void vqwide_v2_init();"
        ),
        cuda_sources=_CUDA_SRC,
        functions=["vqwide_v2_cuda", "vqwide_v2_init"],
        extra_cuda_cflags=[
            "-O3", "-lineinfo",
            f"-DVQW_GEOM_NG={ng}", f"-DVQW_GEOM_KC={kc}",
            f"-DVQW_GEOM_NGV={ngv}", f"-DVQW_GEOM_KCV={kcv}",
            f"-DVQW_GEOM_L={l}", f"-DVQW_GEOM_KVG={kvg}",
            f"-DVQW_SPLIT_CAP={cap}", f"-DVQW_THR={thr}",
            f"-DVQW_UA={ua}", f"-DVQW_UC={uc}",
            f"-DVQW_IDXT={'uint8_t' if idxt == 'uint8' else 'int16_t'}",
        ],
    )
    m.vqwide_v2_init()
    return m
