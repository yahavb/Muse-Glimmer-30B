"""CPU check of muse_compile.greedy_generate: same tokens as model.generate, and no
recompiles on the second (timed) call.

    python scripts/compiled_loop_check.py --config path/to/config.json
"""
import argparse, json, os, sys
import torch
import torch._dynamo as dynamo
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
v.update(num_hidden_layers=2, hidden_size=128, intermediate_size=256, num_attention_heads=4, layer_types=v["layer_types"][:2])
cfg.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
config = AutoConfig.for_model(**cfg); config._attn_implementation = "sdpa"
torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(config).eval()

N, ids = 40, torch.randint(0, 1000, (1, 24))  # 24 + 40 > sliding_window: exercises window wrap
with torch.inference_mode():
    ref = model.generate(input_ids=ids, max_new_tokens=N, do_sample=False, min_new_tokens=N)[:, ids.shape[1]:]
    step = mc.make_step(model, backend="aot_eager")
    cache = mc.make_static_cache(config.text_config, ids.shape[1] + N, model.dtype, ids.device)
    out1, *_ = mc.greedy_generate(step, ids, cache, N)
    frames_after_warmup = counters["stats"]["unique_graphs"]
    out2, *_ = mc.greedy_generate(step, ids, cache, N)
print("graphs compiled in warmup:", frames_after_warmup)
print("new graphs in timed run :", counters["stats"]["unique_graphs"] - frames_after_warmup)
print("match generate (warmup) :", torch.equal(out1, ref), "| timed:", torch.equal(out2, ref))
