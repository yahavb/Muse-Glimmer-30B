"""CPU check of the image-prompt rungs on a tiny random-weight model:
  R4  vision eager (stock)  + compiled mm prefill + compiled decode
  R5  vision fullgraph      + compiled mm prefill + compiled decode
Both must give the same tokens as model.generate(pixel_values=...), with no
recompiles on the second (timed) call.

    python scripts/mm_loop_check.py --config path/to/config.json
"""
import argparse, json, os, sys
import torch
from torch._dynamo.utils import counters
from transformers import AutoConfig, AutoModelForImageTextToText

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import muse_compile as mc

p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args = p.parse_args()
cfg = json.load(open(args.config)); t = cfg["text_config"]
t.update(num_hidden_layers=4, hidden_size=256, intermediate_size=512, num_attention_heads=4,
         num_key_value_heads=2, head_dim=64, layer_types=t["layer_types"][:4],
         layer_rope_theta=t["layer_rope_theta"][:4], sliding_window=16)
v = cfg["vision_config"]
v.update(num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4, layer_types=v["layer_types"][:4])
cfg.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
config = AutoConfig.for_model(**cfg); config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(config).eval()
STOCK_ATTN = mc.mg.MuseGlimmerVisionAttention.forward  # R5/V5 patch it class-wide

# one 672x672 image -> 48x48 patches (4 windows) -> 576 image tokens after 2x2 merge
ps, tp, ms = v["patch_size"], v["patch_temporal"], v["merge_size"]
gh = gw = 672 // ps
grid = torch.tensor([[1, gh, gw]])
pix = torch.randn(gh * gw, 3 * tp * ps * ps)
n_img = gh * gw // ms**2
ids = torch.cat([torch.randint(0, 1000, (1, 5)), torch.full((1, n_img), config.image_token_id),
                 torch.randint(0, 1000, (1, 7))], dim=1)
N = 24

with torch.inference_mode():
    ref = model.generate(input_ids=ids, pixel_values=pix, image_grid_thw=grid,
                         max_new_tokens=N, min_new_tokens=N, do_sample=False)[:, ids.shape[1]:]
    text_ids, img_pos = mc.split_image_tokens(ids, config.image_token_id)
    step = mc.make_step(model, backend="aot_eager")
    mm_prefill = mc.make_mm_prefill(model, backend="aot_eager")
    cache = mc.make_static_cache(config.text_config, ids.shape[1] + N, model.dtype, ids.device)

    for rung, compile_vision in (("R4 vision eager", False), ("R5 vision fullgraph", True)):
        vision = mc.make_vision(model, pix, grid, compile=compile_vision, backend="aot_eager")
        prefill = lambda c, pos: mm_prefill(text_ids, img_pos, vision(), c, pos)
        out1, *_ = mc.greedy_generate(step, ids, cache, N, prefill=prefill)
        before = counters["stats"]["unique_graphs"]
        out2, *_ = mc.greedy_generate(step, ids, cache, N, prefill=prefill)
        print(f"{rung}: match generate warmup={torch.equal(out1, ref)} timed={torch.equal(out2, ref)} "
              f"| new graphs in timed run={counters['stats']['unique_graphs'] - before}")

# video: the real clip's grid (8 frames -> t=4, 20x26 patches -> 520 tokens), random pixels
vgrid = torch.tensor([[4, 20, 26]])
vpix = torch.randn(4 * 20 * 26, 3 * tp * ps * ps)
n_vid = 4 * 20 * 26 // ms**2
vids = torch.cat([torch.randint(0, 1000, (1, 5)), torch.full((1, n_vid), config.video_token_id),
                  torch.randint(0, 1000, (1, 7))], dim=1)
with torch.inference_mode():
    mc._COMPILED.clear()
    mc.mg.MuseGlimmerVisionAttention.forward = STOCK_ATTN  # stock per-segment attention for ref + V4
    ref = model.generate(input_ids=vids, pixel_values_videos=vpix, video_grid_thw=vgrid,
                         max_new_tokens=N, min_new_tokens=N, do_sample=False)[:, vids.shape[1]:]
    text_ids, vpos = mc.split_image_tokens(vids, config.video_token_id)
    cache = mc.make_static_cache(config.text_config, vids.shape[1] + N, model.dtype, vids.device)
    for rung, compile_vision in (("V4 video, vision eager", False), ("V5 video, vision fullgraph", True)):
        vision = mc.make_vision(model, vpix, vgrid, compile=compile_vision, backend="aot_eager")
        prefill = lambda c, pos: mm_prefill(text_ids, vpos, vision(), c, pos)
        out1, *_ = mc.greedy_generate(step, vids, cache, N, prefill=prefill)
        before = counters["stats"]["unique_graphs"]
        out2, *_ = mc.greedy_generate(step, vids, cache, N, prefill=prefill)
        print(f"{rung}: match generate warmup={torch.equal(out1, ref)} timed={torch.equal(out2, ref)} "
              f"| new graphs in timed run={counters['stats']['unique_graphs'] - before}")
