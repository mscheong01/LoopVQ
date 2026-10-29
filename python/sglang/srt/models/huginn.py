# Huginn-0125 (tomg-group-umd/huginn-0125, "RavenForCausalLM"): a recurrent-depth LM.
#
#   e = prelude(wte(ids) * sqrt(n_embd))                 2 SandwichBlocks
#   x = s0                                               random initial state
#   repeat r times:  x = core(adapter([x, e]))          4 SandwichBlocks, shared weights
#   y = ln_f(coda(ln_f(x)))                              2 SandwichBlocks
#
# Every application of a block keeps its own KV, so a token leaves 2 + 4r + 2 slots. HF's
# HuginnDynamicCache indexes them by step_idx (prelude 0,1; core 2 + 4t + l; coda -1,-2). The
# ENGINE slot order here is different on purpose: the 4r core slots come first in Ouro's layout,
#
#     slot = t * 4 + l          t < r recurrence step, l < 4 core block
#     slot = 4r + i             prelude block i
#     slot = 4r + 2 + i         coda block i
#
# so the loop-coupled VQ plumbing (mem_cache/loopvq_pack.py: slot = ut*Lb + l) applies to the
# first Lb*T = 4r slots unchanged, and the 4 non-recurrent slots sit after them.
#
# Faithfulness notes (vs raven_modeling_minimal.py):
#   * RoPE. Huginn rotates INTERLEAVED pairs (2i, 2i+1) with frequency i. The engine's pre-RoPE
#     decode kernels implement NeoX rotate-half (pairs (j, j + D/2)). Permuting the q and k
#     channels of every head by P = [0, 2, ..., D-2, 1, 3, ..., D-1] maps one onto the other
#     exactly, and q.k is invariant under a permutation applied to both, so the rows of Wq, Wk
#     and the qk bias are permuted at load and the model runs NeoX-style. Cached keys are
#     therefore in P order; the bundle builder must capture keys in the same order.
#   * Initial state. HF draws s0 ~ TruncNormal(0, std, +-3 std) * emb_scale fresh on every
#     forward. Serving needs it to be a function of the token, not of the batch, so s0 is a
#     fixed table indexed by absolute position (seed SGLANG_HUGINN_STATE_SEED, default 0).
#     calibration/huginn_torch.py reads the same table.
#   * r = SGLANG_HUGINN_STEPS or config.mean_recurrence (32). Fixed for the server's lifetime,
#     because it sets the number of KV slots. No adaptive exit, no test-time noise (the
#     config's test_time_noise is 0).
import math
import os
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


def huginn_steps(config) -> int:
    return int(os.environ.get("SGLANG_HUGINN_STEPS", 0) or config.mean_recurrence)


def huginn_num_slots(config) -> int:
    return (config.n_layers_in_prelude + config.n_layers_in_coda
            + config.n_layers_in_recurrent_block * huginn_steps(config))


def interleave_to_neox(D: int) -> torch.Tensor:
    """New channel j reads old channel perm[j]: evens first, then odds."""
    return torch.cat([torch.arange(0, D, 2, device="cpu"), torch.arange(1, D, 2, device="cpu")])


def initial_state_table(config, n_pos: int, seed: int) -> torch.Tensor:
    """s0 per absolute position: HF's initialize_state with a fixed generator, [n_pos, E] fp32."""
    std = math.sqrt(2 / (5 * config.n_embd))
    # explicit CPU: the model is constructed under a CUDA device context
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n_pos, config.n_embd, generator=g, device="cpu")
    # trunc_normal_ by resampling the tails, the same distribution HF's init draws from
    x = x * std
    bad = x.abs() > 3 * std
    while bool(bad.any()):
        x[bad] = torch.randn(int(bad.sum()), generator=g, device="cpu") * std
        bad = x.abs() > 3 * std
    return x * math.sqrt(config.n_embd)


class HuginnAttention(nn.Module):
    def __init__(self, config, slots, quant_config=None, prefix=""):
        super().__init__()
        E = config.n_embd
        self.num_heads = config.n_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = E // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.Wqkv = ReplicatedLinear(E, self.q_size + 2 * self.kv_size, bias=False,
                                     quant_config=quant_config, prefix=add_prefix("Wqkv", prefix))
        self.qk_bias = nn.Parameter(torch.zeros(2, 1, self.num_heads, self.head_dim))
        self.proj = ReplicatedLinear(E, E, bias=False, quant_config=quant_config,
                                     prefix=add_prefix("proj", prefix))
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
                                   max_position=config.block_size, base=config.rope_base,
                                   is_neox_style=True)   # see the RoPE note at the top
        self.attn = nn.ModuleDict({
            str(s): RadixAttention(self.num_heads, self.head_dim, self.head_dim ** -0.5,
                                   num_kv_heads=self.num_kv_heads, layer_id=s,
                                   quant_config=quant_config, prefix=add_prefix(f"attn.{s}", prefix))
            for s in slots
        })

    def forward(self, positions, x, forward_batch: ForwardBatch, slot: int):
        qkv, _ = self.Wqkv(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        n = q.shape[0]
        q = (q.view(n, self.num_heads, self.head_dim) + self.qk_bias[0, 0]).view(n, -1)
        k = (k.view(n, self.num_kv_heads, self.head_dim) + self.qk_bias[1, 0]).view(n, -1)
        attn = self.attn[str(slot)]
        if os.environ.get("SGLANG_VQ_PRE_ROPE") == "1":
            # same contract as models/ouro.py: the pool gets raw K, RoPE re-applied at read time
            k_raw = k
            if forward_batch.forward_mode.is_decode_or_idle():
                q_rope, _ = self.rotary_emb(positions, q, k.clone())
                out = attn(q_rope, k_raw, v, forward_batch, save_kv_cache=True,
                           cos_sin_cache=self.rotary_emb.cos_sin_cache)
            else:
                q_rope, k_rope = self.rotary_emb(positions, q, k.clone())
                out = attn(q_rope, k_rope, v, forward_batch, save_kv_cache=False,
                           cos_sin_cache=self.rotary_emb.cos_sin_cache)
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    attn, forward_batch.out_cache_loc,
                    k_raw.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim), is_decode=False)
        else:
            q, k = self.rotary_emb(positions, q, k)
            out = attn(q, k, v, forward_batch)
        out, _ = self.proj(out)
        return out


class HuginnMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        # fc = [gate | up] (HF: chunk(2) -> silu(first) * second), the order SiluAndMul expects
        self.fc = ReplicatedLinear(config.n_embd, 2 * config.intermediate_size, bias=False,
                                   quant_config=quant_config, prefix=add_prefix("fc", prefix))
        self.proj = ReplicatedLinear(config.intermediate_size, config.n_embd, bias=False,
                                     quant_config=quant_config, prefix=add_prefix("proj", prefix))
        self.act = SiluAndMul()

    def forward(self, x):
        h, _ = self.fc(x)
        h, _ = self.proj(self.act(h))
        return h


class SandwichBlock(nn.Module):
    """x = norm_2(attn(norm_1(x)) + x);  x = norm_4(mlp(norm_3(x)) + x)."""

    def __init__(self, config, slots, quant_config=None, prefix=""):
        super().__init__()
        eps = config.norm_eps
        self.norm_1 = RMSNorm(config.n_embd, eps=eps)
        self.attn = HuginnAttention(config, slots, quant_config, add_prefix("attn", prefix))
        self.norm_2 = RMSNorm(config.n_embd, eps=eps)
        self.mlp = HuginnMLP(config, quant_config, add_prefix("mlp", prefix))
        self.norm_3 = RMSNorm(config.n_embd, eps=eps)
        self.norm_4 = RMSNorm(config.n_embd, eps=eps)

    def forward(self, positions, x, forward_batch, slot):
        x = self.norm_2(self.attn(positions, self.norm_1(x), forward_batch, slot) + x)
        return self.norm_4(self.mlp(self.norm_3(x)) + x)


class RavenForCausalLM(nn.Module):
    def __init__(self, config, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        P, C, Lb = config.n_layers_in_prelude, config.n_layers_in_coda, config.n_layers_in_recurrent_block
        self.r = huginn_steps(config)
        self.Lb = Lb
        R = Lb * self.r
        self.n_slots = R + P + C
        self.core_slots = [[t * Lb + l for t in range(self.r)] for l in range(Lb)]
        self.prelude_slots = [R + i for i in range(P)]
        self.coda_slots = [R + P + i for i in range(C)]
        # the engine sizes the pool from end_layer - start_layer (model_runner)
        self.start_layer, self.end_layer = 0, self.n_slots

        self.wte = VocabParallelEmbedding(config.padded_vocab_size, config.n_embd,
                                          prefix=add_prefix("wte", prefix))
        self.prelude = nn.ModuleList(
            SandwichBlock(config, [self.prelude_slots[i]], quant_config, add_prefix(f"prelude.{i}", prefix))
            for i in range(P))
        self.adapter = ReplicatedLinear(2 * config.n_embd, config.n_embd, bias=config.bias,
                                        quant_config=quant_config, prefix=add_prefix("adapter", prefix))
        self.core_block = nn.ModuleList(
            SandwichBlock(config, self.core_slots[l], quant_config, add_prefix(f"core_block.{l}", prefix))
            for l in range(Lb))
        self.coda = nn.ModuleList(
            SandwichBlock(config, [self.coda_slots[i]], quant_config, add_prefix(f"coda.{i}", prefix))
            for i in range(C))
        self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.lm_head = ParallelLMHead(config.padded_vocab_size, config.n_embd,
                                      prefix=add_prefix("lm_head", prefix))
        if config.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.logits_processor = LogitsProcessor(config)
        self.emb_scale = math.sqrt(config.n_embd)
        seed = int(os.environ.get("SGLANG_HUGINN_STATE_SEED", 0))
        self.register_buffer("s0", initial_state_table(config, config.block_size, seed), persistent=False)

    def get_input_embeddings(self):
        return self.wte

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, input_embeds=None):
        e = self.wte(input_ids) if input_embeds is None else input_embeds
        e = e * self.emb_scale
        for i, blk in enumerate(self.prelude):
            e = blk(positions, e, forward_batch, self.prelude_slots[i])
        if self.s0.device != positions.device:
            self.s0 = self.s0.to(positions.device)
        x = self.s0[positions].to(e.dtype)
        for t in range(self.r):
            x, _ = self.adapter(torch.cat([x, e], dim=-1))
            for l, blk in enumerate(self.core_block):
                x = blk(positions, x, forward_batch, t * self.Lb + l)
        x = self.ln_f(x)
        for i, blk in enumerate(self.coda):
            x = blk(positions, x, forward_batch, self.coda_slots[i])
        x = self.ln_f(x)
        return self.logits_processor(input_ids, x, self.lm_head, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        H, D = self.config.n_heads, self.config.n_embd // self.config.n_heads
        E = self.config.n_embd
        perm = interleave_to_neox(D)
        loaded = set()
        for name, w in weights:
            if name == "freqs_cis" or name.endswith(".freqs_cis"):
                continue
            name = name.removeprefix("transformer.")
            if name == "lm_head.weight" and self.config.tie_embeddings:
                continue
            if name.endswith("attn.Wqkv.weight"):
                # rows: q [H*D] | k [H*D] | v [H*D]; permute each q/k head's channels to NeoX order
                q, k, v = w.split([E, E, E], dim=0)
                pm = perm.to(w.device)
                q = q.view(H, D, E)[:, pm].reshape(E, E)
                k = k.view(H, D, E)[:, pm].reshape(E, E)
                w = torch.cat([q, k, v], dim=0)
            elif name.endswith("attn.qk_bias"):
                w = w[..., perm.to(w.device)]
            if name not in params:
                raise KeyError(f"unexpected checkpoint tensor {name!r}")
            p = params[name]
            getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(name)
        missing = set(params) - loaded
        if self.config.tie_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            raise KeyError(f"missing checkpoint tensors: {sorted(missing)[:8]}")

    def get_embed_and_head(self):
        return self.wte.weight, self.lm_head.weight


EntryClass = RavenForCausalLM
