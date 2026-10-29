# Ouro: ByteDance's Looped Language Model (LoopLM).
#
# Ouro runs the SAME stack of `num_hidden_layers` decoder layers `total_ut_steps`
# times (a Universal-Transformer recurrence). Weights are shared across the loop
# steps, but **KV is not**: HF's `UniversalTransformerCache` gives every
# (ut_step, layer) pair its own slot, indexed
#
#     slot = ut_step * num_hidden_layers + layer_idx
#
# (see modeling_ouro.py's OuroAttention.forward). So a model with L layers and
# T loop steps holds L*T layers' worth of KV.
#
# This port mirrors that exactly: each OuroAttention owns T RadixAttention
# instances whose `layer_id` is `ut * L + layer_id`, so SGLang's KV pool lays out
# the same L*T slots. ModelConfig/ModelRunner pick the pool size up from
# `total_ut_steps` the same way they already do for IQuestLoopCoder's `loop_num`.
#
# Consequences for KV quantization, which is why this port exists:
#   - A per-layer method (CQ, TaSQ) applied to the L*T slots gets an independent
#     codebook per (loop step, layer) for free.
#   - LoopVQ instead couples the T slots that share a layer into one code; that
#     lives in the KV pool (mem_cache/unified_kv_pool.py), not here.
#
# Early exit (`early_exit_gate`, `exit_at_step`, `early_exit_threshold`) is NOT
# implemented: like vLLM, this always runs the full `total_ut_steps`. The model
# card documents that limitation for vLLM too. The gate weights are loaded but
# unused so checkpoints round-trip cleanly.
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
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
from sglang.srt.utils import add_prefix, make_layers



_ROPE_SCRATCH: dict = {}


def _rope_scratch(k: torch.Tensor) -> torch.Tensor:
    """Throw-away key for the decode-time q RoPE (see OuroAttention.forward). One buffer per
    shape, so each captured CUDA-graph batch size gets its own static tensor."""
    key = (tuple(k.shape), k.dtype, k.device)
    t = _ROPE_SCRATCH.get(key)
    if t is None:
        t = _ROPE_SCRATCH[key] = torch.empty_like(k, memory_format=torch.contiguous_format)
    return t

class OuroMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(f"Ouro expects hidden_act='silu', got {hidden_act!r}")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class OuroAttention(nn.Module):
    """One attention block, reused across all `total_ut_steps` loop steps.

    Projections are shared; only the KV destination differs per loop step, so we
    hold one RadixAttention per step with `layer_id = ut * num_hidden_layers +
    layer_id` -- the exact slot index HF's UniversalTransformerCache uses.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        num_layers = config.num_hidden_layers
        self.total_ut_steps = getattr(config, "total_ut_steps", 1) or 1
        self.attn = nn.ModuleList(
            [
                RadixAttention(
                    self.num_heads,
                    self.head_dim,
                    self.scaling,
                    num_kv_heads=self.num_kv_heads,
                    # slot = ut * L + layer, matching UniversalTransformerCache
                    layer_id=ut * num_layers + layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"attn.{ut}", prefix),
                )
                for ut in range(self.total_ut_steps)
            ]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        ut_step: int,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn = self.attn[ut_step]

        # CQ/TaSQ/LoopVQ quantize keys PRE-RoPE (CQ paper sec 3.2), but standard
        # serving caches post-RoPE K. Under SGLANG_VQ_PRE_ROPE the pool gets raw K
        # and the decode_attention *_prerope kernels re-apply RoPE at read time from
        # each cached row's own absolute position.
        if os.environ.get("SGLANG_VQ_PRE_ROPE") == "1":
            k_raw = k
            if forward_batch.forward_mode.is_decode_or_idle():
                # Only q is rotated here (the pool keeps raw K; decode rotates it per row), but the
                # fused RoPE op needs a key. A per-shape scratch replaces the per-slot k.clone():
                # same q result bit for bit, one copy kernel fewer per layer slot per step.
                q_rope, _ = self.rotary_emb(positions, q, _rope_scratch(k))
                attn_output = attn(
                    q_rope, k_raw, v, forward_batch, save_kv_cache=True,
                    cos_sin_cache=self.rotary_emb.cos_sin_cache,
                )
            else:
                # Fresh prefill attends to nothing pre-cached, so live compute
                # uses normal post-RoPE q/k; only the pool write differs.
                q_rope, k_rope = self.rotary_emb(positions, q, k.clone())
                attn_output = attn(
                    q_rope, k_rope, v, forward_batch, save_kv_cache=False,
                    cos_sin_cache=self.rotary_emb.cos_sin_cache,
                )
                # RadixAttention.forward() reshapes k/v to [T, H, D] before
                # touching the pool; writing directly skips that, so do it here.
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    attn,
                    forward_batch.out_cache_loc,
                    k_raw.view(-1, self.num_kv_heads, self.head_dim),
                    v.view(-1, self.num_kv_heads, self.head_dim),
                    is_decode=False,
                )
        else:
            q, k = self.rotary_emb(positions, q, k)
            attn_output = attn(q, k, v, forward_batch)

        output, _ = self.o_proj(attn_output)
        return output


class OuroDecoderLayer(nn.Module):
    """Sandwich RMSNorm: a norm on each side of both sublayers, then the residual.

    modeling_ouro.py:
        h = x + input_layernorm_2(attn(input_layernorm(x)))
        h = h + post_attention_layernorm_2(mlp(post_attention_layernorm(h)))

    The trailing norms mean the residual stream is NOT the usual
    pre-norm-plus-raw-sublayer, so the fused add_rms_norm residual path other
    SGLang models use does not apply here; the norms are kept explicit.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = OuroAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = OuroMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        ut_step: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            ut_step=ut_step,
        )
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OuroModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.total_ut_steps = getattr(config, "total_ut_steps", 1) or 1
        self.pp_group = get_pp_group()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: OuroDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Loaded for checkpoint fidelity; unused because this port always runs
        # the full loop (no adaptive exit), same as vLLM.
        self.early_exit_gate = nn.Linear(config.hidden_size, 1, bias=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        # The loop is OUTSIDE the layer stack, so slot order over a forward pass
        # is 0,1,...,L*T-1 -- the same order HF's cache sees.
        for ut_step in range(self.total_ut_steps):
            for i in range(self.start_layer, self.end_layer):
                hidden_states = self.layers[i](
                    positions, hidden_states, forward_batch, ut_step
                )
            # HF normalizes the hidden state at the end of EVERY loop step, and
            # the next step consumes the normalized value.
            hidden_states = self.norm(hidden_states)
        return hidden_states


class OuroForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = OuroModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if name.startswith("model.") is False and name.startswith("lm_head.") is False:
                # HF checkpoints prefix everything with "model." except lm_head.
                name = "model." + name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if name not in params_dict:
                    raise KeyError(f"unexpected checkpoint tensor {name!r}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        missing = set(params_dict) - loaded
        if getattr(self.config, "tie_word_embeddings", False):
            missing.discard("lm_head.weight")
        if missing:
            raise KeyError(f"missing checkpoint tensors: {sorted(missing)[:8]}")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()


EntryClass = OuroForCausalLM
