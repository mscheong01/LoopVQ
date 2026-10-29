"""Grouped vqwide stage 1 for LoopVQ decode (MODE 1 plain pre-RoPE, KVG 1, G' in {4, 8}, 8-bit codes).

Same contract as vqwide_v2 (att_out / att_lse per split, merged by stage 2), different layout:
  * R requests per block (R = 2 at B > 1, 1 at B = 1). The group shares split boundaries, sized from
    its longest request, so tokens at the same split index share a RoPE position and one cos/sin
    row load serves the R requests. The per-token cos/sin row (L fp32 = 512 B) re-read from L2 by
    every (request, head) block was ~40% of vqwide_v2's time at high batch. The K and V codebooks
    are staged once per R requests.
  * phase A: one lane per RoPE group pair (g, g + NG/2) -- the two halves of the same frequencies --
    so its cos/sin come in as float4 loads and a token needs NG/2 lanes (vqwide_v2: one lane per
    group, the same cos/sin loaded twice).
  * phase C (VQG_PC=1): GPL code groups per lane, one GPL-byte code load and one p read per token.
  * per-request buffers are CAP/R tokens, so shared memory matches vqwide_v2 (fits Ada's 99 KB at
    512 threads).
Math is vqwide_v2's; only fp32 summation order and the split partition differ (merged outputs
agree to ~1e-6 relative).

MODE 2 (PERM_ROPE, TaSQ + LoopVQ: RoPE pairs adjacent inside a code group, read through freq_idx /
w_perm) replaces phase A in its own build (md=2, _md2_source); MODE 1 builds compile _SRC unchanged.
VQG_M2L picks the MODE 2 lane layout: 2 = frequency-major lanes (float4 cos/sin), 1 = one lane per
code group, 3 = two code groups per lane, NG/2 lanes per token as in MODE 1, with an unweighted
build (VQG_HASW 0) when w_perm == 1; bitwise equal to M2L 1 and served for G'=8 grouped launches
(m2l_default).

Served from vqwide_ng.run. Grouping (R=2) needs every request to fit
(CAP/2) x max_kv_splits tokens; vqwide_ng only groups when the server's context length does.
"""
from __future__ import annotations

import functools
import os

import torch

_SRC = r"""
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
#define Gw    (Lw / NGw)
#define GVw   (Lw / NGVw)
#define HALFw (Lw / 2)
#define CAPw  VQW_SPLIT_CAP          /* tokens per split summed over the group */
#define THRw  VQW_THR
#define NWw   (THRw / 32)
#define UCw   VQW_UC
#define MINBK 32
#define TBLw  (NGw * KCw * Gw > NGVw * KCVw * GVw ? NGw * KCw * Gw : NGVw * KCVw * GVw)
#define RMAX  4

__device__ __forceinline__ float q2f(const __half v)        { return __half2float(v); }
__device__ __forceinline__ float q2f(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float q2f(const float v)         { return v; }

template <int GG>
__device__ __forceinline__ void load_entry(const __half* row, float* k) {
    if (GG == 4) {
        const uint2 x = *reinterpret_cast<const uint2*>(row);
        const unsigned w[2] = {x.x, x.y};
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[t]));
            k[2 * t] = a.x; k[2 * t + 1] = a.y;
        }
    } else {
        const uint4 x = *reinterpret_cast<const uint4*>(row);
        const unsigned w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[t]));
            k[2 * t] = a.x; k[2 * t + 1] = a.y;
        }
    }
}

template <typename QT, typename IT, typename ST, int R>
__global__ __launch_bounds__(THRw) void vqg_stage1(
    const QT* __restrict__ Q, const uint8_t* __restrict__ K_Idx, const __half* __restrict__ K_CB,
    const uint8_t* __restrict__ V_Idx, const __half* __restrict__ V_CB,
    const ST* __restrict__ K_SZ, const ST* __restrict__ V_SZ, const float* __restrict__ CosSin,
    const int* __restrict__ kv_indptr, const IT* __restrict__ kv_indices,
    const int* __restrict__ num_kv_splits, float* __restrict__ Att_Out, float* __restrict__ Att_Lse,
    int B, float sm_scale, int pos_offset,
    long s_q_b, long s_q_h, long s_ki_b, long s_ki_h, long s_vi_b, long s_vi_h,
    long s_ksz_b, long s_ksz_h, long s_vsz_b, long s_vsz_h, long s_cs_p,
    long s_o_b, long s_o_h, long s_o_s, long s_l_b, long s_l_h, long s_l_s)
{
    constexpr int CT = CAPw / R;                 // tokens per split per request
    constexpr int UAw = (R == 1) ? VQW_UA1 : VQW_UAG;   // tokens in flight per warp, phase A
    const int gb = blockIdx.x, kvh = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int kv0[R], len[R];
    int maxlen = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int b = gb * R + r;
        kv0[r] = b < B ? kv_indptr[b] : 0;
        len[r] = b < B ? kv_indptr[b + 1] - kv0[r] : 0;
        maxlen = max(maxlen, len[r]);
    }
    const int splits = num_kv_splits[gb * R];   // the dispatcher gives every member the group's count
    const int per = ((maxlen + splits - 1) / splits + MINBK - 1) / MINBK * MINBK;
    if (blockIdx.z >= splits || per * (int)blockIdx.z >= maxlen) return;

    extern __shared__ __align__(16) char smem_raw[];
    __half* tbl_s = reinterpret_cast<__half*>(smem_raw);       // TBL
    float*  q_s   = reinterpret_cast<float*>(tbl_s + TBLw);    // R*L
    int*    loc_s = reinterpret_cast<int*>(q_s + R * Lw);      // R*CT
    float*  qk_s  = reinterpret_cast<float*>(loc_s + R * CT);  // R*CT
    float*  vs_s  = qk_s + R * CT;                             // R*CT
    float*  out_s = vs_s + R * CT;                             // NW*L
    float*  red_s = out_s + NWw * Lw;                          // NW
    float*  ml_s  = red_s + NWw;                               // 2R

    for (int i = tid; i < R * Lw; i += THRw) {
        const int r = i / Lw, c = i % Lw, b = gb * R + r;
        q_s[i] = b < B ? q2f(Q[(long)b * s_q_b + (long)kvh * s_q_h + c]) : 0.f;
    }

  for (int bs = blockIdx.z; bs < splits; bs += gridDim.z) {
    const int s_start = per * bs;
    if (s_start >= maxlen) break;
    int nt[R];
    int nmax = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) { nt[r] = max(0, min(per, len[r] - s_start)); nmax = max(nmax, nt[r]); }
#pragma unroll
    for (int r = 0; r < R; ++r)
        for (int n = tid; n < nt[r]; n += THRw) loc_s[r * CT + n] = (int)kv_indices[kv0[r] + s_start + n];
    {
        const int4* src = reinterpret_cast<const int4*>(K_CB + (long)kvh * ((long)NGw * KCw * Gw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGw * KCw * Gw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

    // ---- phase A: RoPE-pair lanes; one cos/sin load per position serves the R requests ----
    {
        constexpr int GG = Gw, NP = NGw / 2, LPT = NP, TPW = 32 / LPT, NF4 = GG / 4;
        const int sub = lane / LPT, ln = lane % LPT;
        float ql[R][GG], qh[R][GG];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int j = 0; j < GG; ++j) { ql[r][j] = q_s[r * Lw + ln * GG + j]; qh[r][j] = q_s[r * Lw + HALFw + ln * GG + j]; }
        const __half* tl = tbl_s + ln * KCw * GG;
        const __half* th = tbl_s + (ln + NP) * KCw * GG;
        for (int n0 = warp * TPW * UAw; n0 < nmax; n0 += NWw * TPW * UAw) {
            float4 co[UAw][NF4], si[UAw][NF4];
            int il[UAw][R], ih[UAw][R];
            float ksv[UAw][R], vsv[UAw][R];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const int nn = n < nmax ? n : 0;
                const float* cs = CosSin + (long)(pos_offset + s_start + nn) * s_cs_p;
#pragma unroll
                for (int t = 0; t < NF4; ++t) {
                    co[u][t] = __ldg(reinterpret_cast<const float4*>(cs + ln * GG) + t);
                    si[u][t] = __ldg(reinterpret_cast<const float4*>(cs + HALFw + ln * GG) + t);
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const bool ok = n < nt[r];
                    const long loc = ok ? loc_s[r * CT + n] : 0;
                    const uint8_t* ir = K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b;
                    il[u][r] = ok ? (int)__ldg(ir + ln) : 0;
                    ih[u][r] = ok ? (int)__ldg(ir + ln + NP) : 0;
                    ksv[u][r] = vsv[u][r] = 0.f;
                    if (ok && ln == 0) {
                        ksv[u][r] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u][r] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                float c[GG], s[GG];
#pragma unroll
                for (int t = 0; t < NF4; ++t) {
                    c[4 * t] = co[u][t].x; c[4 * t + 1] = co[u][t].y; c[4 * t + 2] = co[u][t].z; c[4 * t + 3] = co[u][t].w;
                    s[4 * t] = si[u][t].x; s[4 * t + 1] = si[u][t].y; s[4 * t + 2] = si[u][t].z; s[4 * t + 3] = si[u][t].w;
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float kl[GG], kh[GG];
                    load_entry<GG>(tl + il[u][r] * GG, kl);
                    load_entry<GG>(th + ih[u][r] * GG, kh);
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < GG; ++j) {
                        part = fmaf(kl[j], fmaf(ql[r][j], c[j], qh[r][j] * s[j]), part);
                        part = fmaf(kh[j], fmaf(qh[r][j], c[j], -ql[r][j] * s[j]), part);
                    }
#pragma unroll
                    for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                    if (ln == 0 && n < nt[r]) { qk_s[r * CT + n] = part * ksv[u][r] * sm_scale; vs_s[r * CT + n] = vsv[u][r]; }
                }
            }
        }
    }
    __syncthreads();

    // ---- phase B: softmax per request ----
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float m = -INFINITY;
        for (int n = tid; n < nt[r]; n += THRw) m = fmaxf(m, qk_s[r * CT + n]);
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) red_s[warp] = m;
        __syncthreads();
        if (tid == 0) { float bm = red_s[0]; for (int w = 1; w < NWw; ++w) bm = fmaxf(bm, red_s[w]); ml_s[r] = bm; }
        __syncthreads();
        const float mh = ml_s[r];
        float l = 0.f;
        for (int n = tid; n < nt[r]; n += THRw) {
            const float e = __expf(qk_s[r * CT + n] - mh);
            qk_s[r * CT + n] = e * vs_s[r * CT + n];
            l += e;
        }
        for (int o = 16; o; o >>= 1) l += __shfl_xor_sync(0xffffffff, l, o);
        if (lane == 0) red_s[warp] = l;
        __syncthreads();
        if (tid == 0) { float bl = 0.f; for (int w = 0; w < NWw; ++w) bl += red_s[w]; ml_s[R + r] = bl; }
        __syncthreads();
    }
    {
        const int4* src = reinterpret_cast<const int4*>(V_CB + (long)kvh * ((long)NGVw * KCVw * GVw));
        int4* dst = reinterpret_cast<int4*>(tbl_s);
#pragma unroll 8
        for (int i = tid; i < NGVw * KCVw * GVw / 8; i += THRw) dst[i] = __ldg(src + i);
    }
    __syncthreads();

#if VQG_PC
    // ---- phase C per request: GPL code groups per lane (one GPL-byte code load and one p read
    // serve GPL*GV outputs), LPT lanes per token, SPW tokens per warp per step ----
    constexpr int GPL = (GVw == 4) ? 4 : 2;
    constexpr int LPC = NGVw / GPL;
    constexpr int SPW = 32 / LPC;
    constexpr int NST = NWw * SPW;
    const int lg = lane % LPC;
    const int stripe = warp * SPW + lane / LPC;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        if (nt[r] <= 0) continue;                  // uniform over the block
        const int b = gb * R + r;
        const uint8_t* vrow = V_Idx + (long)kvh * s_vi_h + lg * GPL;
        float acc[GPL * GVw];
#pragma unroll
        for (int j = 0; j < GPL * GVw; ++j) acc[j] = 0.f;
        for (int n0 = stripe; n0 < nt[r]; n0 += NST * UCw) {
            unsigned cw[UCw];
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                if (n < nt[r]) {
                    const uint8_t* rr = vrow + (long)loc_s[r * CT + n] * s_vi_b;
                    cw[u] = (GPL == 4) ? __ldg(reinterpret_cast<const unsigned*>(rr))
                                       : (unsigned)__ldg(reinterpret_cast<const unsigned short*>(rr));
                } else cw[u] = 0u;
            }
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                if (n >= nt[r]) continue;
                const float p = qk_s[r * CT + n];
#pragma unroll
                for (int a = 0; a < GPL; ++a) {
                    float v[GVw];
                    load_entry<GVw>(tbl_s + ((lg * GPL + a) * KCVw + ((cw[u] >> (8 * a)) & 0xff)) * GVw, v);
#pragma unroll
                    for (int j = 0; j < GVw; ++j) acc[a * GVw + j] = fmaf(p, v[j], acc[a * GVw + j]);
                }
            }
        }
#pragma unroll
        for (int j = 0; j < GPL * GVw; ++j)
            for (int o = LPC; o < 32; o <<= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], o);
        if (lane < LPC) {
#pragma unroll
            for (int j = 0; j < GPL * GVw; ++j) out_s[warp * Lw + lg * GPL * GVw + j] = acc[j];
        }
        __syncthreads();
        float* out_base = Att_Out + (long)b * s_o_b + (long)bs * s_o_s + (long)kvh * s_o_h;
        for (int cc = tid; cc < Lw; cc += THRw) {
            float t = 0.f;
#pragma unroll
            for (int rr = 0; rr < NWw; ++rr) t += out_s[rr * Lw + cc];
            out_base[cc] = t / ml_s[R + r];
        }
        if (tid == 0) Att_Lse[(long)b * s_l_b + (long)kvh * s_l_h + (long)bs * s_l_s] = ml_s[r] + logf(ml_s[R + r]);
        __syncthreads();
    }
#else
    // ---- phase C per request (served layout: thread = (V group, token stripe)) ----
    constexpr int WPS  = NGVw >= 32 ? NGVw / 32 : 1;
    constexpr int SPW  = NGVw >= 32 ? 1 : 32 / NGVw;
    constexpr int NST  = NGVw >= 32 ? NWw / WPS : NWw * SPW;
    constexpr int ROWS = NGVw >= 32 ? NST : NWw;
    const int g      = NGVw >= 32 ? (warp % WPS) * 32 + lane : lane % NGVw;
    const int stripe = NGVw >= 32 ? warp / WPS : warp * SPW + lane / NGVw;
    const __half* vg = tbl_s + g * KCVw * GVw;
#pragma unroll 1
    for (int r = 0; r < R; ++r) {
        if (nt[r] <= 0) continue;                  // uniform over the block
        const int b = gb * R + r;
        const uint8_t* vrow = V_Idx + (long)kvh * s_vi_h + g;
        float acc[GVw];
#pragma unroll
        for (int j = 0; j < GVw; ++j) acc[j] = 0.f;
        for (int n0 = stripe; n0 < nt[r]; n0 += NST * UCw) {
            int id[UCw];
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                const int n = n0 + u * NST;
                id[u] = n < nt[r] ? (int)__ldg(vrow + (long)loc_s[r * CT + n] * s_vi_b) : -1;
            }
#pragma unroll
            for (int u = 0; u < UCw; ++u) {
                if (id[u] < 0) continue;
                const float p = qk_s[r * CT + n0 + u * NST];
                float v[GVw];
                load_entry<GVw>(vg + id[u] * GVw, v);
#pragma unroll
                for (int j = 0; j < GVw; ++j) acc[j] = fmaf(p, v[j], acc[j]);
            }
        }
        if (NGVw < 32) {
#pragma unroll
            for (int j = 0; j < GVw; ++j)
                for (int o = NGVw; o < 32; o <<= 1) acc[j] += __shfl_xor_sync(0xffffffff, acc[j], o);
        }
        const int row = NGVw >= 32 ? stripe : warp;
        if (NGVw >= 32 || lane < NGVw) {
#pragma unroll
            for (int j = 0; j < GVw; ++j) out_s[row * Lw + g * GVw + j] = acc[j];
        }
        __syncthreads();
        float* out_base = Att_Out + (long)b * s_o_b + (long)bs * s_o_s + (long)kvh * s_o_h;
        for (int cc = tid; cc < Lw; cc += THRw) {
            float t = 0.f;
#pragma unroll
            for (int rr = 0; rr < ROWS; ++rr) t += out_s[rr * Lw + cc];
            out_base[cc] = t / ml_s[R + r];
        }
        if (tid == 0) Att_Lse[(long)b * s_l_b + (long)kvh * s_l_h + (long)bs * s_l_s] = ml_s[r] + logf(ml_s[R + r]);
        __syncthreads();
    }
#endif
  }
}

template <int R>
static int smem_bytes_r() {
    return (int)(sizeof(__half) * TBLw + sizeof(float) * R * Lw + (sizeof(int) + 2 * sizeof(float)) * CAPw
                 + sizeof(float) * NWw * Lw + sizeof(float) * NWw + sizeof(float) * 2 * R + 64);
}

#define FOR_ST(X, R) X(__nv_bfloat16, long, __half, R) X(__nv_bfloat16, long, float, R) \
                     X(__nv_bfloat16, int, __half, R) X(__nv_bfloat16, int, float, R) \
                     X(__half, long, __half, R) X(__half, long, float, R) \
                     X(__half, int, __half, R) X(__half, int, float, R)

void vqwide_v2_init() {
#define REG(QT, IT, ST, R) TORCH_CHECK(cudaFuncSetAttribute(vqg_stage1<QT, IT, ST, R>, \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes_r<R>()) == cudaSuccess, "vqg: smem attr");
    FOR_ST(REG, 1) FOR_ST(REG, VQG_R)
#undef REG
}

void vqwide_v2_cuda(torch::Tensor q, torch::Tensor k_idx, torch::Tensor k_cb,
                    torch::Tensor v_idx, torch::Tensor v_cb, torch::Tensor k_sz, torch::Tensor v_sz,
                    torch::Tensor cos_sin, torch::Tensor freq_idx, torch::Tensor w_perm,
                    torch::Tensor kv_indptr, torch::Tensor kv_indices, torch::Tensor splits,
                    torch::Tensor att_out, torch::Tensor att_lse, int64_t n_splits, double sm_scale,
                    int64_t pos_offset, int64_t mode) {
    // mode 1: one request per block; 17: groups of VQG_R (the caller sized the splits for it)
    TORCH_CHECK(mode == 1 || mode == 17, "vqwide_g: MODE 1 (plain pre-RoPE) only, got ", mode);
    const int B = q.size(0), H_KV = k_idx.size(1);
    const int R = mode == 17 ? VQG_R : 1;
    const int NB = (B + R - 1) / R;
    int dev = 0, n_sm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    const int zw = (int)std::max<int64_t>(1, std::min<int64_t>(n_splits, (2 * n_sm + NB * H_KV - 1) / (NB * H_KV)));
    dim3 grid(NB, H_KV, zw);
#define GO(QT, IT, ST, RR) vqg_stage1<QT, IT, ST, RR><<<grid, THRw, smem_bytes_r<RR>(), at::cuda::getCurrentCUDAStream()>>>( \
        reinterpret_cast<const QT*>(q.data_ptr()), k_idx.data_ptr<uint8_t>(),                     \
        reinterpret_cast<const __half*>(k_cb.data_ptr()), v_idx.data_ptr<uint8_t>(),              \
        reinterpret_cast<const __half*>(v_cb.data_ptr()),                                         \
        reinterpret_cast<const ST*>(k_sz.data_ptr()), reinterpret_cast<const ST*>(v_sz.data_ptr()), \
        cos_sin.data_ptr<float>(), kv_indptr.data_ptr<int>(),                                     \
        reinterpret_cast<const IT*>(kv_indices.data_ptr()), splits.data_ptr<int>(),               \
        att_out.data_ptr<float>(), att_lse.data_ptr<float>(), B, (float)sm_scale, (int)pos_offset, \
        q.stride(0), q.stride(1), k_idx.stride(0), k_idx.stride(1), v_idx.stride(0), v_idx.stride(1), \
        k_sz.stride(0), k_sz.stride(1), v_sz.stride(0), v_sz.stride(1), cos_sin.stride(0),        \
        att_out.stride(0), att_out.stride(1), att_out.stride(2),                                  \
        att_lse.stride(0), att_lse.stride(1), att_lse.stride(2))
#define PICKR(QT, IT, ST) do { if (R == 1) GO(QT, IT, ST, 1); else GO(QT, IT, ST, VQG_R); } while (0)
#define PICKS(QT, IT) do { if (k_sz.scalar_type() == at::kHalf) PICKR(QT, IT, __half); else PICKR(QT, IT, float); } while (0)
    const bool i64 = kv_indices.scalar_type() == at::kLong;
    if (q.scalar_type() == at::kBFloat16) { if (i64) PICKS(__nv_bfloat16, long); else PICKS(__nv_bfloat16, int); }
    else { if (i64) PICKS(__half, long); else PICKS(__half, int); }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
#undef GO
#undef PICKR
#undef PICKS
}
"""


# ---- MODE 2 (PERM_ROPE) phase A ----
_A_START = "    // ---- phase A: RoPE-pair lanes; one cos/sin load per position serves the R requests ----\n"
_A_END = "    __syncthreads();\n\n    // ---- phase B: softmax per request ----"

_MD2B = r"""#if VQG_MD == 2 && VQG_M2L == 2
    // ---- phase A, MODE 2, frequency-major lanes: lane = 4 consecutive natural frequencies, so cos/sin
    // are float4 loads; each frequency's stored pair (group, position) comes from the inverse of
    // freq_idx (per-head constants) ----
    {
        constexpr int GG = Gw, PPG = GG / 2, NF = 4, LPT = HALFw / NF, TPW = 32 / LPT;
        __shared__ int inv_s[HALFw];
        for (int i = tid; i < HALFw; i += THRw) inv_s[(int)FreqIdx[(long)kvh * HALFw + i]] = i;
        __syncthreads();
        const int sub = lane / LPT, ln = lane % LPT;
        int gi[NF], off[NF];
        float a1[R][NF], a2[R][NF], b1[R][NF], b2[R][NF];
#pragma unroll
        for (int t = 0; t < NF; ++t) {
            const int f = ln * NF + t, pp = inv_s[f];
            gi[t] = pp / PPG;                                  // code group of the pair
            off[t] = gi[t] * KCw * GG + (pp % PPG) * 2;        // table offset less the code
            const float wa = 1.f / WPerm[(long)kvh * Lw + 2 * pp], wb = 1.f / WPerm[(long)kvh * Lw + 2 * pp + 1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float qa = q_s[r * Lw + f], qb = q_s[r * Lw + HALFw + f];
                a1[r][t] = wa * qa; a2[r][t] = wa * qb; b1[r][t] = wb * qb; b2[r][t] = wb * qa;
            }
        }
        for (int n0 = warp * TPW * UAw; n0 < nmax; n0 += NWw * TPW * UAw) {
            float4 co[UAw], si[UAw];
            int id[UAw][R][NF];
            float ksv[UAw][R], vsv[UAw][R];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const int nn = n < nmax ? n : 0;
                const float* cs = CosSin + (long)(pos_offset + s_start + nn) * s_cs_p;
                co[u] = __ldg(reinterpret_cast<const float4*>(cs + ln * NF));
                si[u] = __ldg(reinterpret_cast<const float4*>(cs + HALFw + ln * NF));
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const bool ok = n < nt[r];
                    const long loc = ok ? loc_s[r * CT + n] : 0;
                    const uint8_t* ir = K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b;
#pragma unroll
                    for (int t = 0; t < NF; ++t) id[u][r][t] = ok ? (int)__ldg(ir + gi[t]) : 0;
                    ksv[u][r] = vsv[u][r] = 0.f;
                    if (ok && ln == 0) {
                        ksv[u][r] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u][r] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const float c[NF] = {co[u].x, co[u].y, co[u].z, co[u].w};
                const float sn[NF] = {si[u].x, si[u].y, si[u].z, si[u].w};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float part = 0.f;
#pragma unroll
                    for (int t = 0; t < NF; ++t) {
                        const float2 kk = __half22float2(*reinterpret_cast<const __half2*>(tbl_s + off[t] + id[u][r][t] * GG));
                        part = fmaf(kk.x, fmaf(a1[r][t], c[t],  a2[r][t] * sn[t]), part);
                        part = fmaf(kk.y, fmaf(b1[r][t], c[t], -b2[r][t] * sn[t]), part);
                    }
#pragma unroll
                    for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                    if (ln == 0 && n < nt[r]) { qk_s[r * CT + n] = part * ksv[u][r] * sm_scale; vs_s[r * CT + n] = vsv[u][r]; }
                }
            }
        }
    }
#elif VQG_MD == 2 && VQG_M2L == 3
    // ---- phase A, MODE 2, two code groups per lane (g, g + NG/2): NG/2 lanes per token, as in MODE 1,
    // so a warp holds twice M2L 1's tokens and the xor tree is one step shorter. Each group's partial
    // is summed in M2L 1's order and the two are added first -- exactly M2L 1's first xor step -- so qk
    // is bit-identical to M2L 1. VQG_HASW 0: w_perm == 1 (loop bundles; asserted at load), 1/w is
    // exactly 1 and q is held once per pair instead of four weighted copies ----
    {
        constexpr int GG = Gw, PPG = GG / 2, NH2 = NGw / 2, LPT = NH2, TPW = 32 / LPT;
        const int sub = lane / LPT, g = lane % LPT;
        int fq[2][PPG];
#if VQG_HASW
        float a1[R][2][PPG], a2[R][2][PPG], b1[R][2][PPG], b2[R][2][PPG];
#else
        float qa[R][2][PPG], qb[R][2][PPG];
#endif
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int j = 0; j < PPG; ++j) {
                const int pp = (g + h * NH2) * PPG + j;
                fq[h][j] = (int)FreqIdx[(long)kvh * HALFw + pp];
#if VQG_HASW
                const float wa = 1.f / WPerm[(long)kvh * Lw + 2 * pp], wb = 1.f / WPerm[(long)kvh * Lw + 2 * pp + 1];
#endif
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float xa = q_s[r * Lw + fq[h][j]], xb = q_s[r * Lw + HALFw + fq[h][j]];
#if VQG_HASW
                    a1[r][h][j] = wa * xa; a2[r][h][j] = wa * xb; b1[r][h][j] = wb * xb; b2[r][h][j] = wb * xa;
#else
                    qa[r][h][j] = xa; qb[r][h][j] = xb;
#endif
                }
            }
        const __half* tb[2] = {tbl_s + g * KCw * GG, tbl_s + (g + NH2) * KCw * GG};
        for (int n0 = warp * TPW * UAw; n0 < nmax; n0 += NWw * TPW * UAw) {
            float co[UAw][2][PPG], si[UAw][2][PPG];
            int id[UAw][R][2];
            float ksv[UAw][R], vsv[UAw][R];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const int nn = n < nmax ? n : 0;
                const float* cs = CosSin + (long)(pos_offset + s_start + nn) * s_cs_p;
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < PPG; ++j) { co[u][h][j] = __ldg(cs + fq[h][j]); si[u][h][j] = __ldg(cs + HALFw + fq[h][j]); }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const bool ok = n < nt[r];
                    const long loc = ok ? loc_s[r * CT + n] : 0;
                    const uint8_t* ir = K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b;
                    id[u][r][0] = ok ? (int)__ldg(ir + g) : 0;
                    id[u][r][1] = ok ? (int)__ldg(ir + g + NH2) : 0;
                    ksv[u][r] = vsv[u][r] = 0.f;
                    if (ok && g == 0) {
                        ksv[u][r] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u][r] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float ph[2];
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        float k[GG];
                        load_entry<GG>(tb[h] + id[u][r][h] * GG, k);
                        float part = 0.f;
#pragma unroll
                        for (int j = 0; j < PPG; ++j) {
#if VQG_HASW
                            part = fmaf(k[2 * j],     fmaf(a1[r][h][j], co[u][h][j],  a2[r][h][j] * si[u][h][j]), part);
                            part = fmaf(k[2 * j + 1], fmaf(b1[r][h][j], co[u][h][j], -b2[r][h][j] * si[u][h][j]), part);
#else
                            part = fmaf(k[2 * j],     fmaf(qa[r][h][j], co[u][h][j],  qb[r][h][j] * si[u][h][j]), part);
                            part = fmaf(k[2 * j + 1], fmaf(qb[r][h][j], co[u][h][j], -qa[r][h][j] * si[u][h][j]), part);
#endif
                        }
                        ph[h] = part;
                    }
                    float part = ph[0] + ph[1];
#pragma unroll
                    for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                    if (g == 0 && n < nt[r]) { qk_s[r * CT + n] = part * ksv[u][r] * sm_scale; vs_s[r * CT + n] = vsv[u][r]; }
                }
            }
        }
    }
#elif VQG_MD == 2
"""

_MD2 = r"""#if VQG_MD == 2
    // ---- phase A, MODE 2 (PERM_ROPE): one lane per code group = G'/2 whole RoPE pairs ----
    {
        constexpr int GG = Gw, PPG = GG / 2, LPT = NGw, TPW = 32 / LPT;
        const int sub = lane / LPT, g = lane % LPT;
        int fq[PPG];
        float a1[R][PPG], a2[R][PPG], b1[R][PPG], b2[R][PPG];
#pragma unroll
        for (int j = 0; j < PPG; ++j) {
            const int pp = g * PPG + j;
            fq[j] = (int)FreqIdx[(long)kvh * HALFw + pp];
            const float wa = 1.f / WPerm[(long)kvh * Lw + 2 * pp], wb = 1.f / WPerm[(long)kvh * Lw + 2 * pp + 1];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float qa = q_s[r * Lw + fq[j]], qb = q_s[r * Lw + HALFw + fq[j]];
                a1[r][j] = wa * qa; a2[r][j] = wa * qb; b1[r][j] = wb * qb; b2[r][j] = wb * qa;
            }
        }
        const __half* tb = tbl_s + g * KCw * GG;
        for (int n0 = warp * TPW * UAw; n0 < nmax; n0 += NWw * TPW * UAw) {
            float co[UAw][PPG], si[UAw][PPG];
            int id[UAw][R];
            float ksv[UAw][R], vsv[UAw][R];
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
                const int nn = n < nmax ? n : 0;
                const float* cs = CosSin + (long)(pos_offset + s_start + nn) * s_cs_p;
#pragma unroll
                for (int j = 0; j < PPG; ++j) { co[u][j] = __ldg(cs + fq[j]); si[u][j] = __ldg(cs + HALFw + fq[j]); }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const bool ok = n < nt[r];
                    const long loc = ok ? loc_s[r * CT + n] : 0;
                    id[u][r] = ok ? (int)__ldg(K_Idx + (long)kvh * s_ki_h + loc * (int)s_ki_b + g) : 0;
                    ksv[u][r] = vsv[u][r] = 0.f;
                    if (ok && g == 0) {
                        ksv[u][r] = q2f(K_SZ[loc * s_ksz_b + (long)kvh * s_ksz_h]);
                        vsv[u][r] = q2f(V_SZ[loc * s_vsz_b + (long)kvh * s_vsz_h]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < UAw; ++u) {
                const int n = n0 + u * TPW + sub;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float k[GG];
                    load_entry<GG>(tb + id[u][r] * GG, k);
                    float part = 0.f;
#pragma unroll
                    for (int j = 0; j < PPG; ++j) {
                        part = fmaf(k[2 * j],     fmaf(a1[r][j], co[u][j],  a2[r][j] * si[u][j]), part);
                        part = fmaf(k[2 * j + 1], fmaf(b1[r][j], co[u][j], -b2[r][j] * si[u][j]), part);
                    }
#pragma unroll
                    for (int o = LPT / 2; o; o >>= 1) part += __shfl_xor_sync(0xffffffff, part, o);
                    if (g == 0 && n < nt[r]) { qk_s[r * CT + n] = part * ksv[u][r] * sm_scale; vs_s[r * CT + n] = vsv[u][r]; }
                }
            }
        }
    }
#else
"""


def _md2_source() -> str:
    """_SRC with the MODE 2 phase A in front of the MODE 1 one (#if VQG_MD == 2 ... #else ... #endif)
    and the FreqIdx / WPerm pointers threaded into the kernel."""
    s = _SRC
    i = s.index(_A_START); j = s.index(_A_END, i)
    s = s[:i] + _MD2B + _MD2[len("#if VQG_MD == 2\n"):] + s[i:j] + "#endif\n" + s[j:]
    for old, new in (
            ("const float* __restrict__ CosSin,\n",
             "const float* __restrict__ CosSin, const long* __restrict__ FreqIdx, const float* __restrict__ WPerm,\n"),
            ("cos_sin.data_ptr<float>(), kv_indptr.data_ptr<int>(),",
             "cos_sin.data_ptr<float>(), freq_idx.data_ptr<long>(), w_perm.data_ptr<float>(), kv_indptr.data_ptr<int>(),"),
            ('TORCH_CHECK(mode == 1 || mode == 17, "vqwide_g: MODE 1 (plain pre-RoPE) only, got ", mode);',
             'TORCH_CHECK((mode & 15) == VQG_MD, "vqwide_g: built for MODE ", VQG_MD, ", got ", mode);'),
            ("const int R = mode == 17 ? VQG_R : 1;", "const int R = mode >= 16 ? VQG_R : 1;")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


def thr() -> int:
    """Threads per block (SGLANG_VQWIDE_G_THR, default 512: at 1024 the phase-C row buffer pushes the
    block past Ada's 99 KB of shared memory)."""
    return int(os.environ.get("SGLANG_VQWIDE_G_THR", 512))


def unroll(g: int) -> tuple:
    """Tokens in flight per warp in phase A at R = 1 / in a group, and per thread in phase C.
    Grouped blocks hold R requests' q and codes, so fewer tokens in flight; G'=8 runs out of
    registers beyond one."""
    ua1 = int(os.environ.get("SGLANG_VQWIDE_G_UA1", 1 if g == 8 else 2))
    uag = int(os.environ.get("SGLANG_VQWIDE_G_UAG", 1))
    uc = int(os.environ.get("SGLANG_VQWIDE_G_UC", 8))
    return ua1, uag, uc


def group_r() -> int:
    return int(os.environ.get("SGLANG_VQWIDE_G_R", 2))


def cap() -> int:
    return int(os.environ.get("SGLANG_VQWIDE_G_CAP", 2048))


def m2l_default(g: int, grouped: bool) -> int:
    """MODE 2 lane layout (see the module docstring): G'=4 -> M2L 2, G'=8 -> M2L 3, at every R.
    M2L 3 is bitwise equal to M2L 1 and faster for G'=8; G'=4 is faster end to end on M2L 2."""
    return 2 if g == 4 else 3


def structural_ok(ng, l, ngv, kvg, kc, kcv, mode) -> bool:
    return (mode in (1, 2) and kvg == 1 and l == 128 and ng * (l // ng) == l and l // ng in (4, 8)
            and l // ngv in (4, 8) and max(kc, kcv) <= 256)


@functools.lru_cache(maxsize=16)
def ext(ng, kc, l, ngv, kcv, md=1, hasw=1, grouped=True):
    from torch.utils.cpp_extension import load_inline
    t = thr(); ua1, uag, uc = unroll(l // ng); r = group_r(); c = cap()
    pc = int(os.environ.get("SGLANG_VQWIDE_G_PC", 1))
    name = f"sgl_vqwide_g_r{r}_c{c}_ua{ua1}x{uag}_uc{uc}_pc{pc}_ng{ng}_ngv{ngv}_kc{kc}_kcv{kcv}_l{l}_t{t}"
    flags = []
    if md == 2:
        # MODE 2 lanes hold 4*PPG*R q constants: one token in flight. Lane layout: see the docstring.
        ua1 = int(os.environ.get("SGLANG_VQWIDE_G_UA1_M2", 1)); uag = int(os.environ.get("SGLANG_VQWIDE_G_UAG_M2", 1))
        m2l = int(os.environ.get("SGLANG_VQWIDE_G_M2L", m2l_default(l // ng, grouped)))
        hw = int(hasw or m2l != 3)
        name = (f"sgl_vqwide_g_md2l{m2l}w{hw}_r{r}_c{c}_ua{ua1}x{uag}_uc{uc}_pc{pc}_ng{ng}_ngv{ngv}_kc{kc}_kcv{kcv}"
                f"_l{l}_t{t}")
        flags = [f"-DVQG_MD=2", f"-DVQG_M2L={m2l}", f"-DVQG_HASW={hw}"]
    m = load_inline(
        name=name,
        cpp_sources=("#include <torch/extension.h>\n"
                     "void vqwide_v2_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
                     " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,"
                     " torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t, int64_t);\n"
                     "void vqwide_v2_init();"),
        cuda_sources=_SRC if md == 1 else _md2_source(),
        functions=["vqwide_v2_cuda", "vqwide_v2_init"],
        extra_cuda_cflags=["-O3", "-lineinfo", f"-DVQW_GEOM_NG={ng}", f"-DVQW_GEOM_KC={kc}",
                           f"-DVQW_GEOM_NGV={ngv}", f"-DVQW_GEOM_KCV={kcv}", f"-DVQW_GEOM_L={l}",
                           f"-DVQW_SPLIT_CAP={c}", f"-DVQW_THR={t}", f"-DVQW_UA1={ua1}", f"-DVQW_UAG={uag}", f"-DVQW_UC={uc}",
                           f"-DVQG_R={r}", f"-DVQG_PC={pc}"] + flags,
    )
    m.vqwide_v2_init()
    return m
