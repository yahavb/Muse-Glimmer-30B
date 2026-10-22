# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer-30B text decoder for vLLM Neuron (BF16, TP + SP).

Ported from the plugin's Qwen3 template (vllm_neuron/model/qwen3/model.py), with the
sliding-window decode handling of vllm_neuron/model/gpt_oss/model_bf16.py.

Differences from Qwen3 (each marked  # <-- MUSE):
  - Gated attention: out = o_proj(attn * sigmoid(gate_proj(x))). The gate sits between
    attention and o_proj, so neither the prefill o_proj nor the decode megakernel fuses
    the output projection.
  - Q/K RMSNorm has NO weight; Q is scaled by qk_scale_factor (3.87) after its norm.
    The whole softmax scale (3.87 / sqrt(head_dim)) is folded into the Q-norm gamma, so
    every kernel runs with scale 1.0 (flash_attention requires 1.0 with a sliding window).
  - Layers [local x3, global] repeat: local = sliding window 2048 + RoPE (theta 500000),
    global = full attention, NoPE (cos/sin None -> the kernels skip RoPE).
  - Four norms per layer, "centered" (x * (1 + w)): input, post-attention, pre-FF, post-FF.
    post_* norms use post_norm_eps (1e-8).
  - Embedding output goes through a weight-less RMSNorm; final norm is a standard RMSNorm.
  - Logits: softcap(logits * output_multiplier) with cap = final_logit_softcapping.
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from nkilib.core.utils.common_types import NormType
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec, paged_slot_indices
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
)

from .config import MuseGlimmerTextConfig

logger = logging.getLogger(__name__)


# ── norms ────────────────────────────────────────────────────────────────────


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """HF MuseGlimmer RMS core, in fp32: x * (mean(x^2) + eps)^-0.5."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class CenteredRMSNorm(nn.Module):  # <-- MUSE: HF MuseGlimmerTextCenteredRMSNorm, x * (1 + w)
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms(x, self.eps) * (1.0 + self.weight.float())).to(x.dtype)


class RMSNorm(nn.Module):  # <-- MUSE: HF MuseGlimmerRMSNorm (final norm), x * w
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (_rms(x, self.eps) * self.weight.float()).to(x.dtype)


# ── RoPE ─────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Default RoPE, rotate_half (contiguous halves) layout, like HF."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.head_dim = config.head_dim

    def forward(self, positions: torch.Tensor, device, dtype):
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device) / self.head_dim))
        freqs = (inv_freq[:, None] @ positions[None, :].float()).transpose(0, 1)  # [T, Dh/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, Dh]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


# ── attention ────────────────────────────────────────────────────────────────


class MuseGlimmerAttention(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_name = f"layers.{layer_idx}.self_attn"
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.eps = config.rms_norm_eps
        self.use_rope = config.has_rope(layer_idx)                                  # <-- MUSE
        self.sliding_window = config.sliding_window if config.is_sliding(layer_idx) else None
        # <-- MUSE: softmax scale folded into the Q-norm gamma; kernels run at scale 1.0
        self.q_gamma_value = config.qk_scale_factor * config.head_dim ** -0.5

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = self.num_key_value_heads // self.world_size
            self.num_kv_replicas = 1
        self.num_key_value_groups = self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank

        self.q_size = self.num_attention_heads_per_rank * self.head_dim
        self.kv_size = self.num_key_value_heads_per_rank * self.head_dim
        self.qkv_split_indices = [self.q_size, self.q_size + self.kv_size]
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.q_size + 2 * self.kv_size, dtype=self.dtype))
        self.gate_proj_weight = nn.Parameter(  # <-- MUSE: attention output gate, sharded like Q
            torch.empty(self.hidden_size, self.q_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(torch.empty(self.q_size, self.hidden_size, dtype=self.dtype))

        self.k_cache = None
        self.v_cache = None

        set_weight_loader(self.qkv_proj_weight, fused_qkv_weight_loader(
            q_size=self.q_size, kv_size=self.kv_size, shard_dim=1, num_shards=self.world_size,
            is_storage_transposed=True, num_kv_replicas=self.num_kv_replicas))
        set_weight_loader(self.gate_proj_weight, sharding_weight_loader(
            shard_dim=1, shard_size=self.q_size, num_shards=self.world_size, is_storage_transposed=True))
        set_weight_loader(self.o_proj_weight, sharding_weight_loader(
            shard_dim=0, shard_size=self.q_size, num_shards=self.world_size, is_storage_transposed=True))

    def _qk_gammas(self, device):
        # <-- MUSE: weight-less Q/K norm. K gamma 1; Q gamma = full softmax scale.
        q = torch.full((1, self.head_dim), self.q_gamma_value, dtype=self.dtype, device=device)
        k = torch.ones((1, self.head_dim), dtype=self.dtype, device=device)
        return q, k

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[self.layer_name]
        if md["max_query_len"] <= md["decode_token_threshold"]:
            return self.forward_decode(hidden_states, positions, position_embeddings, attn_metadata)
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)  # SP -> full sequence
        return self.forward_prefill(hidden_states, position_embeddings, attn_metadata)

    # ── prefill ──

    def forward_prefill(self, hidden_states, position_embeddings, attn_metadata):
        hidden_states = hidden_states.to(self.dtype)
        tokens = hidden_states.shape[0]
        q_gamma, k_gamma = self._qk_gammas(hidden_states.device)
        cos_cache = sin_cache = None
        if self.use_rope:
            cos, sin = position_embeddings
            cos_cache, sin_cache = cos.unsqueeze(0), sin.unsqueeze(0)

        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0), qkv_weights=self.qkv_proj_weight, bias=None,
            d_head=self.head_dim, cos_cache=cos_cache, sin_cache=sin_cache,
            num_q_heads=self.num_attention_heads_per_rank, num_kv_heads=self.num_key_value_heads_per_rank,
            qk_norm_pre_rope_q_norm=NormType.RMS_NORM, qk_norm_pre_rope_k_norm=NormType.RMS_NORM,
            qk_norm_pre_rope_eps=self.eps, qk_norm_pre_rope_q_gamma=q_gamma, qk_norm_pre_rope_k_gamma=k_gamma,
        ).squeeze(0)
        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)
        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        md = attn_metadata[self.layer_name]
        block_size = md["block_size"]
        self._write_kv(k, v, md["slot_mapping"], block_size)

        kv_segment_size = md.get("kv_segment_size")
        if kv_segment_size:
            attn_output = NF.segmented_attention(
                q, k_cache=self.k_cache, v_cache=self.v_cache, block_tables=md["block_table_tensor"],
                prior_tokens=md.get("cached_seq_len"), block_size=block_size,
                kv_segment_size=kv_segment_size, scale=1.0, sliding_window=self.sliding_window,
                tp_q=True, tp_out=True)  # [Nh, Dh, T]
        else:
            k = k.repeat_interleave(self.num_key_value_groups, dim=0)
            v = v.repeat_interleave(self.num_key_value_groups, dim=0)
            attn_output = NF.flash_attention(
                q.transpose(1, 2), k.transpose(1, 2), v, scale=1.0,
                sliding_window=self.sliding_window, tp_q=False, tp_out=True)  # [Nh, Dh, T]

        # <-- MUSE: gate before o_proj. gate [T, Nh*Dh] -> [Nh, Dh, T] to match attn_output
        gate = torch.sigmoid(hidden_states @ self.gate_proj_weight)
        gate = gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim).permute(1, 2, 0)
        attn_output = (attn_output * gate.to(attn_output.dtype)).unsqueeze(0)
        out = NF.o_proj(attn_output, self.o_proj_weight, None).squeeze(0)  # [T, H]
        if self.world_size > 1:
            out = self.tp_group.reduce_scatter(out, dim=0)  # back to SP layout
        return out.contiguous()

    def _write_kv(self, k, v, slot_mapping, block_size):
        nkh = self.num_key_value_heads_per_rank
        block_idx, pos_idx = paged_slot_indices(slot_mapping, block_size, self.k_cache.shape[0])
        heads = torch.arange(nkh, dtype=torch.long, device=k.device).repeat_interleave(slot_mapping.shape[0])
        index = (block_idx.repeat(nkh), heads, pos_idx.repeat(nkh))
        self.k_cache.index_put_(index, k.reshape(-1, self.head_dim).to(self.k_cache.dtype))
        self.v_cache.index_put_(index, v.reshape(-1, self.head_dim).to(self.v_cache.dtype))

    # ── decode ──

    def forward_decode(self, hidden_states, positions, position_embeddings, attn_metadata):
        md = attn_metadata[self.layer_name]
        slot_mapping = md["slot_mapping"]
        block_table = md["block_table_tensor"]
        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S = tokens // B
        assert tokens == B * S, f"decode tokens {tokens} not a multiple of batch {B}"
        hidden_states = hidden_states.to(self.dtype)
        X = hidden_states.view(B, S, hidden)
        q_gamma, k_gamma = self._qk_gammas(hidden_states.device)

        cos_k = sin_k = None
        if self.use_rope:
            cos, sin = position_embeddings
            half = self.head_dim // 2
            cos_k = cos[:, :half].view(B, S, half).permute(2, 0, 1).contiguous().to(self.dtype)
            sin_k = sin[:, :half].view(B, S, half).permute(2, 0, 1).contiguous().to(self.dtype)

        # In-kernel causal mask from positions; sliding-window start for local layers.
        # The runner trims SWA block tables and gives swa_kv_pos_offset (GPT-OSS pattern).
        pos_ids = positions.view(1, B * S)
        swa_start = None
        if self.sliding_window is not None:
            off = md.get("swa_kv_pos_offset")
            if off is not None:
                pos_ids = torch.clamp(pos_ids - off.view(B, 1).expand(B, S).reshape(1, B * S), min=0)
            swa_start = torch.clamp(pos_ids - self.sliding_window + 1, min=0)

        attn = NF.attention_decode(
            X=X, X_hidden_dim_actual=self.hidden_size, rmsnorm_X_enabled=False,
            W_qkv=self.qkv_proj_weight, bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=self.eps,
            rmsnorm_QK_pre_rope_W_Q=q_gamma, rmsnorm_QK_pre_rope_W_K=k_gamma,
            rmsnorm_QK_post_rope_enabled=False,
            cos=cos_k, sin=sin_k, rope_contiguous_layout=True,
            K_cache_transposed=False, active_blocks_table=block_table,
            K_cache=self.k_cache, V_cache=self.v_cache, attention_mask=None,
            pos_ids=pos_ids.view(B, S).to(torch.float32),
            swa_start_pos_ids=swa_start.view(B, S).to(torch.float32) if swa_start is not None else None,
            softmax_scale=1.0, sink=None,
            update_cache=True, kv_cache_update_idx=slot_mapping.view(B, S).to(torch.uint32),
            W_out=None, bias_out=None,  # <-- MUSE: no fused o_proj; gate comes first
            transposed_out=False, out_in_sb=False,
        )  # [B, Nh, Dh, S]

        gate = torch.sigmoid(hidden_states @ self.gate_proj_weight)  # [B*S, Nh*Dh]
        gate = gate.view(B, S, self.num_attention_heads_per_rank, self.head_dim).permute(0, 2, 3, 1)
        attn = attn * gate.to(attn.dtype)
        attn = attn.permute(0, 3, 1, 2).reshape(B * S, self.q_size)
        out = attn @ self.o_proj_weight  # [B*S, H] partial over heads
        if self.world_size > 1:
            self.tp_group.all_reduce(out)
        return out


# ── MLP ──────────────────────────────────────────────────────────────────────


class MuseGlimmerMLP(nn.Module):
    """SwiGLU (silu), no bias; intermediate sharded across TP."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        inter = config.intermediate_size // self.world_size
        dt = config.torch_dtype
        self.gate_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inter, dtype=dt))
        self.up_proj_weight = nn.Parameter(torch.empty(config.hidden_size, inter, dtype=dt))
        self.down_proj_weight = nn.Parameter(torch.empty(inter, config.hidden_size, dtype=dt))
        gate_up = sharding_weight_loader(shard_dim=1, shard_size=inter, num_shards=self.world_size,
                                         is_storage_transposed=True)
        set_weight_loader(self.gate_proj_weight, gate_up)
        set_weight_loader(self.up_proj_weight, gate_up)
        set_weight_loader(self.down_proj_weight, sharding_weight_loader(
            shard_dim=0, shard_size=inter, num_shards=self.world_size, is_storage_transposed=True))

    def forward(self, x, is_prefill: bool):
        if is_prefill and self.world_size > 1:
            x = self.tp_group.all_gather(x, dim=0)
        out = NF.mlp(x, self.gate_proj_weight, self.up_proj_weight, self.down_proj_weight)
        if self.world_size > 1:
            if is_prefill:
                out = self.tp_group.reduce_scatter(out, dim=0)
            else:
                self.tp_group.all_reduce(out)
        return out


# ── decoder layer + backbone ─────────────────────────────────────────────────


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig, layer_idx: int):
        super().__init__()
        h, dt = config.hidden_size, config.torch_dtype
        self.input_layernorm = CenteredRMSNorm(h, config.rms_norm_eps, dt)
        self.post_attention_layernorm = CenteredRMSNorm(h, config.post_norm_eps, dt)
        self.pre_feedforward_layernorm = CenteredRMSNorm(h, config.rms_norm_eps, dt)
        self.post_feedforward_layernorm = CenteredRMSNorm(h, config.post_norm_eps, dt)
        self.self_attn = MuseGlimmerAttention(config, layer_idx)
        self.mlp = MuseGlimmerMLP(config)

    def forward(self, hidden_states, positions, position_embeddings, attn_metadata, is_prefill):
        residual = hidden_states
        x = self.self_attn(self.input_layernorm(hidden_states), positions, position_embeddings, attn_metadata)
        hidden_states = residual + self.post_attention_layernorm(x)
        residual = hidden_states
        x = self.mlp(self.pre_feedforward_layernorm(hidden_states), is_prefill=is_prefill)
        return residual + self.post_feedforward_layernorm(x)


class MuseGlimmerTextModel(nn.Module):
    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size, embed_dim=config.hidden_size,
            dtype=config.torch_dtype, tp_group=self.tp_group.device_group)
        set_weight_loader(self.embed_tokens.weight, sharding_weight_loader(
            shard_dim=0, shard_size=self.embed_tokens.vocab_size_per_rank, num_shards=self.world_size,
            is_storage_transposed=False, pad_shard=True))
        self.layers = nn.ModuleList(
            [MuseGlimmerDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, config.torch_dtype)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(self, input_ids, positions, attn_metadata, rank=None, inputs_embeds=None, is_token_ids=None):
        md = attn_metadata["layers.0.self_attn"]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        sp = is_prefill  # sequence parallel in prefill, like the Qwen3 template
        h = self.embed_tokens(input_ids, scatter_tokens=sp, rank=rank)
        h = NF.merge_prompt_embeds(h, inputs_embeds, is_token_ids)
        h = _rms(h, self.config.rms_norm_eps).to(h.dtype)  # <-- MUSE: weight-less embedding norm
        pos_emb = self.rotary_emb(positions, device=h.device, dtype=h.dtype)
        for layer in self.layers:
            h = layer(h, positions, pos_emb, attn_metadata, is_prefill)
        h = self.norm(h)
        if sp and self.world_size > 1:
            h = self.tp_group.all_gather(h, dim=0)
        return h


class MuseGlimmerForCausalLM(nn.Module):
    """Text-only Muse-Glimmer for vLLM Neuron (phase 1: no vision tower)."""

    def __init__(self, config: MuseGlimmerTextConfig):
        super().__init__()
        self.config = config
        self.model = MuseGlimmerTextModel(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        nc = config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        self._gather_logits = nc is not None and (nc.max_logprobs != 0 or nc.debug_logits_dir is not None)
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config, tp_group=self.tp_group.device_group)
        set_weight_loader(self.lm_head.weight, sharding_weight_loader(
            shard_dim=0, shard_size=config.vocab_size // self.world_size, num_shards=self.world_size,
            is_storage_transposed=False))
        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(self.on_device_sampling_config, process_group=self.tp_group.device_group)

    @torch.no_grad()
    def forward(self, input_ids, positions, inputs_embeds=None, is_token_ids=None, attn_metadata=None,
                sampling_positions=None, sampling_params=None, spec_decode_metadata=None,
                logit_mask=None, rank=None, **kwargs):
        if spec_decode_metadata is not None:
            raise NotImplementedError("speculative decoding is not supported for Muse-Glimmer yet")
        positions = positions.to(torch.int32)
        md = attn_metadata["layers.0.self_attn"]
        is_prefill = md["max_query_len"] > md["decode_token_threshold"]
        T = input_ids.shape[0]
        if is_prefill and (T <= self.world_size or T % self.world_size != 0):
            raise ValueError(f"Prompt length ({T}) must be > and divisible by world_size ({self.world_size}) for SP.")
        h = self.model(input_ids, positions, attn_metadata=attn_metadata, rank=rank,
                       inputs_embeds=inputs_embeds, is_token_ids=is_token_ids)
        if not is_prefill and md["max_query_len"] == 1 and h.shape[0] == sampling_positions.shape[0]:
            h_logits = h
        else:
            h_logits = torch.index_select(h, dim=0, index=sampling_positions)
        logits = self.lm_head(h_logits)
        # <-- MUSE: cap * tanh(logits * multiplier / cap); elementwise, so vocab shards are fine
        cap = self.config.final_logit_softcapping
        logits = (torch.tanh(logits.float() * (self.config.output_multiplier / cap)) * cap).to(logits.dtype)

        if self.on_device_sampling_config is None:
            return logits
        sampled = self.sampler(logits, sampling_params, logit_mask=logit_mask, tp_rank=rank)
        gathered = self.tp_group.all_gather(logits, dim=1) if self._gather_logits else None
        return sampled, gathered

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        return cls(MuseGlimmerTextConfig.from_configs(hf_config, neuron_config))

    # ── KV cache ──

    def get_kv_spec(self):
        return KVSpec(layers=[
            LayerSpec(name=f"layers.{i}.self_attn",
                      num_kv_heads=l.self_attn.num_key_value_heads_per_rank,
                      head_size=l.self_attn.head_dim, dtype=l.self_attn.dtype,
                      sliding_window_size=l.self_attn.sliding_window, chunk_size=None)
            for i, l in enumerate(self.model.layers)])

    def bind_kv_cache(self, kv_caches):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise KeyError(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache, layer.self_attn.v_cache = kv_caches[name][0], kv_caches[name][1]

    # ── weights ──

    def load_weights(self, checkpoint_path, device, cache_dir):
        src = "model.language_model"  # HF MuseGlimmerForConditionalGeneration text prefix
        m = {
            "model.embed_tokens.weight": f"{src}.embed_tokens.weight",
            "model.norm.weight": f"{src}.norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(len(self.model.layers)):
            d, s = f"model.layers.{i}", f"{src}.layers.{i}"
            m[f"{d}.self_attn.qkv_proj_weight"] = [f"{s}.self_attn.q_proj.weight",
                                                    f"{s}.self_attn.k_proj.weight",
                                                    f"{s}.self_attn.v_proj.weight"]
            m[f"{d}.self_attn.gate_proj_weight"] = f"{s}.self_attn.gate_proj.weight"
            m[f"{d}.self_attn.o_proj_weight"] = f"{s}.self_attn.o_proj.weight"
            for n in ("input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                m[f"{d}.{n}.weight"] = f"{s}.{n}.weight"
            for n in ("gate_proj", "up_proj", "down_proj"):
                m[f"{d}.mlp.{n}_weight"] = f"{s}.mlp.{n}.weight"
        state = SafetensorsCheckpoint(checkpoint_path, cache_dir).load_sharded_pipelined(
            self.rank, self.world_size, self, m, device).state_dict
        for k, t in state.items():
            if t.dtype != self.config.torch_dtype:
                state[k] = t.to(self.config.torch_dtype)
        missing = set(dict(self.named_parameters())) - set(state)
        if missing:
            raise RuntimeError(f"checkpoint did not provide {len(missing)} parameters, e.g. {sorted(missing)[:4]}")
        self.load_state_dict(state, strict=False, assign=True)
