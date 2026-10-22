"""Find torch.compile graph breaks in Muse-Glimmer on CPU, with a tiny random-weight model.

Same model code as the 30B, shrunk dims/layers so it runs on a laptop. Checks the
text decoder forward (prefill + decode with StaticCache) and the vision tower.
Device-specific gaps (ops the neuron backend can't lower) do NOT show up here.

    python scripts/graph_break_probe.py --config path/to/config.json
"""
import argparse
import json

import torch
import torch._dynamo as dynamo
from transformers import AutoConfig, AutoModelForImageTextToText, StaticCache

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--layers", type=int, default=4)
args = p.parse_args()

cfg_dict = json.load(open(args.config))
t = cfg_dict["text_config"]
L = args.layers
t.update(num_hidden_layers=L, hidden_size=256, intermediate_size=512,
         num_attention_heads=4, num_key_value_heads=2, head_dim=64,
         layer_types=t["layer_types"][:L], layer_rope_theta=t["layer_rope_theta"][:L],
         sliding_window=16)
v = cfg_dict["vision_config"]
v.update(num_hidden_layers=L, hidden_size=128, intermediate_size=256,
         num_attention_heads=4, layer_types=v["layer_types"][:L])
cfg_dict.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
config = AutoConfig.for_model(**cfg_dict)
config._attn_implementation = "sdpa"

torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(config).eval()


def report(name, fn, *a, **kw):
    dynamo.reset()
    ex = dynamo.explain(fn)(*a, **kw)
    print(f"\n=== {name}: graphs={ex.graph_count} breaks={ex.graph_break_count} ops={ex.op_count}")
    for b in ex.break_reasons:
        print("  -", b.reason.splitlines()[0][:200])
        for fr in b.user_stack[-3:]:
            print(f"      {fr.filename.split('site-packages/')[-1]}:{fr.lineno} {fr.name}")
    return ex.graph_break_count


def fullgraph(name, fn, *a, **kw):
    dynamo.reset()
    try:
        torch.compile(fn, fullgraph=True, backend="aot_eager")(*a, **kw)
        print(f"=== {name}: fullgraph=True OK")
        return True
    except Exception as e:
        print(f"=== {name}: fullgraph=True FAILED: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
        return False


PREFILL, MAX_LEN = 32, 64
ids = torch.randint(0, 1000, (1, PREFILL))


def new_cache():
    return StaticCache(config=config.text_config, max_cache_len=MAX_LEN)


def step(input_ids, cache, cache_position):
    return model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                 cache_position=cache_position, position_ids=cache_position.unsqueeze(0)).logits


with torch.inference_mode():
    # prefill
    report("text prefill", step, ids, new_cache(), torch.arange(PREFILL))
    fullgraph("text prefill", step, ids, new_cache(), torch.arange(PREFILL))

    # decode (one token, cache already holds the prompt)
    cache = new_cache()
    step(ids, cache, torch.arange(PREFILL))
    nxt = torch.randint(0, 1000, (1, 1))
    report("text decode", step, nxt, cache, torch.tensor([PREFILL]))
    fullgraph("text decode", step, nxt, cache, torch.tensor([PREFILL]))

    # vision tower: one 224x224 image through the processor-shaped path
    ps, tp = v["patch_size"], v["patch_temporal"]
    gh = gw = 224 // ps
    grid = torch.tensor([[1, gh, gw]])
    pix = torch.randn(gh * gw, 3 * tp * ps * ps)
    try:
        vis = lambda pv, g: model.model.get_image_features(pixel_values=pv, image_grid_thw=g)
        report("vision tower", vis, pix, grid)
        fullgraph("vision tower", vis, pix, grid)
    except Exception as e:
        print(f"=== vision tower: probe could not run: {type(e).__name__}: {str(e)[:300]}")

    # vision tower with muse_compile fixes: precomputed grid kwargs + masked SDPA.
    # 672x672 -> 48x48 patches -> several 32x32 windows, so window attention is exercised.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import muse_compile as mc
    gh = gw = 672 // ps
    grid = torch.tensor([[1, gh, gw]])
    pix = torch.randn(gh * gw, 3 * tp * ps * ps)
    ref = mc.image_embeds(model, pix, grid)  # original code path, eager
    mc.patch_vision_attention()
    vk = mc.precompute_vision_kwargs(model.model.vision_tower, grid)
    print(f"\nwindows: {len(vk['cu_window_seqlens']) - 1}, tokens: {gh * gw}")
    vis = lambda pv, g: mc.image_embeds(model, pv, g, **{k: x.clone() for k, x in vk.items()})
    report("vision tower (patched)", vis, pix, grid)
    fullgraph("vision tower (patched)", vis, pix, grid)
    dynamo.reset()
    out = torch.compile(vis, fullgraph=True, backend="aot_eager")(pix, grid)
    print(f"patched vs original max abs diff: {(out.float() - ref.float()).abs().max().item():.3e}")
