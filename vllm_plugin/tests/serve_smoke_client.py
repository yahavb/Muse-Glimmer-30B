"""Smoke-test a running `vllm serve` (OpenAI API on localhost): correctness + timing.

Sends the ladder's text prompt with greedy decoding, streams the reply, and prints
TTFT, decode tokens/s and the text, so it can be compared with the HF/TorchNeuron rungs.

    python serve_smoke_client.py --url http://127.0.0.1:8000 --model meta-models/Muse-Glimmer-30B
"""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
p.add_argument("--prompt", default="Explain quantum computing in simple terms.")
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--runs", type=int, default=2, help="first run includes any lazy warmup")
args = p.parse_args()

body = {
    "model": args.model,
    "messages": [{"role": "system", "content": "Reasoning strength: low"},
                 {"role": "user", "content": args.prompt}],
    "temperature": 0.0,
    "max_tokens": args.max_tokens,
    "stream": True,
    "stream_options": {"include_usage": True},
}

for run in range(args.runs):
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    t_first, text, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    text.append(piece)
    t_end = time.time()
    n = usage.get("completion_tokens", 0)
    ttft = (t_first or t_end) - t0
    dec = t_end - (t_first or t_end)
    print(f"RUN {run}: prompt_tokens={usage.get('prompt_tokens')} completion_tokens={n} "
          f"TTFT={ttft:.3f}s decode={dec:.2f}s ({(n - 1) / dec if dec > 0 and n > 1 else 0:.1f} tok/s) "
          f"e2e={t_end - t0:.2f}s ({n / (t_end - t0):.1f} tok/s)")
print("TEXT:", "".join(text)[:1500])
