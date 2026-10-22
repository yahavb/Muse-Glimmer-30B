"""Graph-break check for the text decoder under transformers TP (gloo on CPU).

    torchrun --nproc_per_node 2 scripts/tp_graph_break_probe.py --config path/to/config.json
"""
import argparse
import json
import os
import tempfile

import torch
import torch._dynamo as dynamo
from transformers import AutoConfig, AutoModelForImageTextToText, StaticCache

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
args = p.parse_args()

cfg_dict = json.load(open(args.config))
t = cfg_dict["text_config"]
t.update(num_hidden_layers=4, hidden_size=256, intermediate_size=512,
         num_attention_heads=4, num_key_value_heads=2, head_dim=64,
         layer_types=t["layer_types"][:4], layer_rope_theta=t["layer_rope_theta"][:4], sliding_window=16)
v = cfg_dict["vision_config"]
v.update(num_hidden_layers=2, hidden_size=128, intermediate_size=256, num_attention_heads=4,
         layer_types=v["layer_types"][:2])
cfg_dict.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
config = AutoConfig.for_model(**cfg_dict)

# On a Mac, transformers would pick MPS (no TP support); force the CPU path.
torch._C._get_accelerator = lambda: torch.device("cpu")
torch.distributed.init_process_group("gloo")
rank = torch.distributed.get_rank()
path = os.path.join(tempfile.gettempdir(), "muse_tiny_tp_probe")
if rank == 0:
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(config).save_pretrained(path)
torch.distributed.barrier()

model = AutoModelForImageTextToText.from_pretrained(path, tp_plan="auto", attn_implementation="sdpa").eval()
tcfg = model.config.text_config
PREFILL, MAX_LEN = 32, 64
ids = torch.randint(0, 1000, (1, PREFILL))


def step(input_ids, cache, cache_position):
    return model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1,
                 cache_position=cache_position, position_ids=cache_position.unsqueeze(0)).logits


def check(name, *a):
    dynamo.reset()
    ex = dynamo.explain(step)(*a)
    reasons = [b.reason.splitlines()[0][:160] for b in ex.break_reasons]
    dynamo.reset()
    try:
        torch.compile(step, fullgraph=True, backend="aot_eager")(*a)
        fg = "OK"
    except Exception as e:
        fg = f"FAILED {type(e).__name__}: {str(e).splitlines()[0][:200]}"
    if rank == 0:
        print(f"=== TP=2 {name}: graphs={ex.graph_count} breaks={ex.graph_break_count} fullgraph={fg}")
        for r in reasons:
            print("  -", r)


with torch.inference_mode():
    check("prefill", ids, StaticCache(config=tcfg, max_cache_len=MAX_LEN), torch.arange(PREFILL))
    cache = StaticCache(config=tcfg, max_cache_len=MAX_LEN)
    step(ids, cache, torch.arange(PREFILL))
    check("decode", torch.randint(0, 1000, (1, 1)), cache, torch.tensor([PREFILL]))
torch.distributed.destroy_process_group()
