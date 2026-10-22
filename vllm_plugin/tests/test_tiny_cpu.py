"""Tiny Muse-Glimmer through vLLM Neuron in CPU mode vs HF transformers, greedy.

Random weights, real architecture: [local, local, local, global] layers, gated attention,
weight-less Q/K norm with qk_scale_factor, NoPE global layers, centered norms, logit softcap.
sliding_window=16 and a 24-token prompt + 16 new tokens, so the window is crossed in both
prefill and decode. Weight names are the real checkpoint's (model.language_model.*).

    VLLM_NEURON_CPU_MODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python test_tiny_cpu.py --config config.json [--tp 1 2]
"""
import argparse
import json
import os
import tempfile

import torch

p = argparse.ArgumentParser()
p.add_argument("--config", required=True, help="real Muse-Glimmer config.json (dims are shrunk)")
p.add_argument("--tp", type=int, nargs="+", default=[1, 2])
p.add_argument("--new-tokens", type=int, default=16)
args = p.parse_args()

cfg = json.load(open(args.config))
t = cfg["text_config"]
L = 4
t.update(num_hidden_layers=L, hidden_size=512, intermediate_size=1024, num_attention_heads=4,
         num_key_value_heads=2, head_dim=128, vocab_size=4096, sliding_window=16,
         layer_types=t["layer_types"][:L], layer_rope_theta=t["layer_rope_theta"][:L],
         max_position_embeddings=256)
v = cfg["vision_config"]
v.update(num_hidden_layers=2, hidden_size=128, intermediate_size=256, num_attention_heads=4,
         layer_types=v["layer_types"][:2])
cfg.update(out_hidden_size=128 * v["merge_size"] ** 2, projector_hidden_size=256)
for k in ("bos_token_id", "eos_token_id"):
    t[k] = 1 if k == "bos_token_id" else 2
cfg["image_token_id"], cfg["video_token_id"] = 4000, 4001

from transformers import AutoConfig, AutoModelForImageTextToText  # noqa: E402

hf_cfg = AutoConfig.for_model(**cfg)
torch.manual_seed(0)
hf = AutoModelForImageTextToText.from_config(hf_cfg).to(torch.bfloat16).eval()
with torch.no_grad():  # random norms around 0 / 1 so every norm actually does something
    for name, prm in hf.named_parameters():
        if "layernorm" in name or name.endswith("norm.weight"):
            prm.copy_(torch.randn_like(prm) * 0.1 + (0.0 if "layernorm" in name else 1.0))
d = tempfile.mkdtemp(prefix="muse_tiny_")
hf.save_pretrained(d)
print("tiny checkpoint:", d, sorted(os.listdir(d)))

prompt = list(range(5, 29))  # 24 tokens > sliding_window
with torch.no_grad():
    ref = hf.generate(torch.tensor([prompt]), max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                      do_sample=False)[0, len(prompt):].tolist()
print("HF reference:", ref)

from vllm import LLM, SamplingParams  # noqa: E402

ok = True
for tp in args.tp:
    llm = LLM(model=d, max_num_seqs=1, max_model_len=128, block_size=16, tensor_parallel_size=tp,
              enforce_eager=True, enable_prefix_caching=False, skip_tokenizer_init=True,
              num_gpu_blocks_override=16,
              additional_config={"neuron_config": {"num_batched_tokens_buckets": [32, 128]}})
    out = llm.generate([{"prompt_token_ids": prompt}],
                       SamplingParams(temperature=0.0, max_tokens=args.new_tokens, ignore_eos=True))
    got = list(out[0].outputs[0].token_ids)
    n = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
    print(f"TP={tp}: vLLM {got}")
    print(f"TP={tp}: first {n}/{len(ref)} tokens match HF")
    ok &= n == len(ref)
    del llm
print("RESULT:", "PASS" if ok else "MISMATCH")
