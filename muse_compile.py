"""Make Muse-Glimmer's vision path traceable with torch.compile(fullgraph=True).

The text decoder already compiles fullgraph with a StaticCache. The vision tower
breaks the graph in two ways, both fixed here with pure PyTorch (no kernels):

1. Grid metadata (cu_seqlens, window index, position ids, interpolation taps,
   pixel-shuffle index) is computed from grid_thw with .tolist()/data-dependent
   shapes. `precompute_vision_kwargs` runs those helpers once, eagerly, and the
   tower pops them from kwargs instead (transformers' export path).
2. Non-flash attention splits q/k/v per segment with lengths.tolist().
   `patch_vision_attention` replaces that with one SDPA call under a
   block-diagonal mask built from cu_seqlens with static-shape ops only.
"""
import time

import torch
import torch.nn.functional as F
from transformers import StaticCache
from transformers.cache_utils import StaticLayer
from transformers.models.muse_glimmer import modeling_muse_glimmer as mg
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
    get_vision_window_index,
)


def precompute_vision_kwargs(vision_tower, grid_thw):
    """Grid-dependent tensors for one fixed image size; pass as **kwargs to the tower."""
    pe = vision_tower.patch_embedder
    window_index, cu_window_seqlens = get_vision_window_index(
        grid_thw,
        spatial_merge_size=vision_tower.spatial_merge_size,
        window_size=vision_tower.window_size,
        patch_size=vision_tower.patch_size,
    )
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=pe.num_grid_per_side,
        mode=pe.interpolation_mode,
        align_corners=pe.interpolation_align_corners,
        spatial_merge_size=1,
        padding=pe.interpolation_padding,
    )
    return dict(
        cu_seqlens=get_vision_cu_seqlens(grid_thw),
        window_index=window_index,
        cu_window_seqlens=torch.as_tensor(cu_window_seqlens, dtype=torch.int32),
        position_ids=get_vision_position_ids(grid_thw, spatial_merge_size=vision_tower.spatial_merge_size),
        interp_indices=interp_indices,
        interp_weights=interp_weights,
        pixel_shuffle_index=mg.get_vision_pixel_shuffle_index(grid_thw, vision_tower.merge_size),
    )


def _block_diag_mask(cu_seqlens, seq_len, device):
    # segment id of each token = number of boundaries it has passed; static shapes only
    pos = torch.arange(seq_len, device=device)
    seg = (pos[:, None] >= cu_seqlens[None, 1:].to(device)).sum(-1)
    return seg[:, None] == seg[None, :]


def _vision_attention_forward(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
    seq_length = hidden_states.shape[0]
    q = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    k = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    v = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    cos, sin = position_embeddings
    q, k = mg.apply_rotary_pos_emb_vision(q, k, cos, sin)
    q, k, v = (t.transpose(2, 1) for t in (q, k, v))
    mask = _block_diag_mask(cu_seqlens, seq_length, hidden_states.device)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scaling)
    out = out.transpose(1, 2).reshape(seq_length, -1).contiguous()
    return self.proj(out)


def patch_vision_attention():
    mg.MuseGlimmerVisionAttention.forward = _vision_attention_forward


def make_static_cache(text_config, max_cache_len, dtype, device, tp_size=1):
    """StaticCache with a full-length StaticLayer for EVERY layer, sliding ones included.

    StaticSlidingWindowLayer tracks its length as a python int and its mask sizes
    change until the window fills -> one recompile per decode step. Full-length
    layers keep a tensor length and fixed mask sizes; the sliding-window mask still
    enforces the 2048 window. Cost: ~1KB/token per sliding layer at TP=1.

    Allocated up front: lazy allocation on the first update would make the first
    prefill trace an uninitialized cache and recompile on the next call.
    """
    cache = StaticCache(config=text_config, max_cache_len=max_cache_len)
    cache.layers = [StaticLayer(max_cache_len=max_cache_len) for _ in cache.layers]
    kv_heads = text_config.num_key_value_heads // tp_size  # colwise k/v_proj under TP
    cache.early_initialization(1, kv_heads, text_config.head_dim, dtype, device)
    return cache


def make_step(model, **compile_kwargs):
    """One forward over a StaticCache, returning last-position logits. Compiled fullgraph."""

    @torch.compile(fullgraph=True, dynamic=False, **compile_kwargs)
    def step(ids, cache, cache_position):
        return model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                     cache_position=cache_position, position_ids=cache_position.unsqueeze(0)).logits[:, -1]

    return step


def make_mm_prefill(model, **compile_kwargs):
    """Prefill from text ids + image features, compiled fullgraph.

    Same math as MuseGlimmerModel.forward: image placeholder ids embed as id 0, then
    the image rows replace them in order. masked_scatter is swapped for index_copy
    on host-computed positions (static shape; index_copy already lowers on neuron
    via the StaticCache update).
    """

    @torch.compile(fullgraph=True, dynamic=False, **compile_kwargs)
    def prefill(text_ids, image_pos, image_feats, cache, cache_position):
        embeds = model.get_input_embeddings()(text_ids)
        embeds = embeds[0].index_copy(0, image_pos, image_feats.to(embeds.dtype)).unsqueeze(0)
        return model(inputs_embeds=embeds, past_key_values=cache, use_cache=True, logits_to_keep=1,
                     cache_position=cache_position, position_ids=cache_position.unsqueeze(0)).logits[:, -1]

    return prefill


def split_image_tokens(input_ids, image_token_id):
    """(text_ids with image placeholders -> 0, sorted image positions). Host side."""
    mask = input_ids[0] == image_token_id
    text_ids = input_ids.masked_fill(mask.unsqueeze(0), 0)
    return text_ids, mask.nonzero().squeeze(-1)


def make_vision(model, pixel_values, grid_thw, compile=False, **compile_kwargs):
    """Image features [n_image_tokens, hidden] for one fixed image size.

    compile=False: stock vision math, eager. Grid metadata is computed on the host:
                   on device, eager repeat_interleave with tensor repeats (data-dependent
                   output size) fails to lower ('stablehlo.concatenate' type error).
    compile=True : patched attention + the same host metadata, fullgraph=True.
    """
    dev = pixel_values.device
    vk = {k: v.to(dev) for k, v in precompute_vision_kwargs(model.model.vision_tower, grid_thw.cpu()).items()}
    if not compile:
        return lambda: image_embeds(model, pixel_values, grid_thw, **vk)
    patch_vision_attention()
    # one compiled fn per process: same-size images reuse the same graph
    if "vision" not in _COMPILED:
        _COMPILED["vision"] = torch.compile(lambda pv, g, kw: image_embeds(model, pv, g, **kw),
                                            fullgraph=True, dynamic=False, **compile_kwargs)
    fn = _COMPILED["vision"]
    return lambda: fn(pixel_values, grid_thw, vk)


_COMPILED = {}


def greedy_generate(step, input_ids, cache, max_new_tokens, sync=lambda: None, prefill=None):
    """Greedy loop over two graphs: prefill (prompt length) and decode (1 token).

    Reuse the same `cache` across calls (it is reset here): a new cache object would
    fail dynamo's guards and recompile both graphs.
    `prefill(cache, cache_position) -> logits` overrides the text prefill (image prompts);
    input_ids then only supplies the prompt length.
    Returns (new_tokens[1, n], ttft_s, decode_s).
    """
    cache.reset()
    # clone under inference_mode so the prompt has the same dispatch keys as the
    # decode tokens (else one extra recompile)
    input_ids = input_ids.clone()
    prompt_len = input_ids.shape[1]
    dev = input_ids.device
    t0 = time.time()
    pos = torch.arange(prompt_len, device=dev)
    logits = prefill(cache, pos) if prefill is not None else step(input_ids, cache, pos)
    nxt = logits.argmax(-1, keepdim=True)
    sync()
    ttft = time.time() - t0
    tokens = [nxt]
    t1 = time.time()
    for i in range(max_new_tokens - 1):
        nxt = step(nxt, cache, torch.tensor([prompt_len + i], device=dev)).argmax(-1, keepdim=True)
        tokens.append(nxt)
    sync()
    return torch.cat(tokens, dim=-1), ttft, time.time() - t1


def image_embeds(model, pixel_values, grid_thw, **vision_kwargs):
    """get_image_features without the trailing per-image split (.tolist())."""
    m = model.model
    feats = m.vision_tower(pixel_values=pixel_values, grid_thw=grid_thw, **vision_kwargs).last_hidden_state
    return m.perception_emb_norm(m.vision_projection(m.vision_adapter(feats)))


def final_answer(text):
    """User-facing part of a decoded reply: text after the last 'to=user' channel tag."""
    return text.rsplit("to=user", 1)[-1].strip()


def feature_diff(ref, out):
    """(max_abs_diff, cosine) between two feature tensors, in fp32."""
    ref, out = ref.float().flatten(), out.float().flatten()
    cos = torch.nn.functional.cosine_similarity(ref, out, dim=0)
    return (ref - out).abs().max().item(), cos.item()


def matching_prefix(a, b):
    """Number of leading tokens two sequences share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
