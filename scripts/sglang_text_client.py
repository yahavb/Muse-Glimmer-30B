"""Text-only checks against a running SGLang server (localhost), greedy, streamed.

  1. known answers: 17 * 23 -> 391, capital of France -> Paris (PASS / FAIL)
  2. free text: the same prompt twice; greedy, so the two answers must match

Prints TTFT, decode tok/s and the answer (reasoning, if the server splits it out, shown apart).

    python scripts/sglang_text_client.py --url http://127.0.0.1:30000 --model Muse-Glimmer-30B
"""
import argparse
import json
import sys
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30000")
p.add_argument("--model", default="Muse-Glimmer-30B")
p.add_argument("--max-tokens", type=int, default=256)
args = p.parse_args()

CASES = [("What is 17 * 23? Answer with the number only.", "391"),
         ("What is the capital of France? Answer in one word.", "paris")]
FREE = "Explain quantum computing in simple terms."


def ask(label, text):
    body = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens, "stream": True,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, n, content, reasoning = time.time(), None, 0, [], []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {})
            c, rc = delta.get("content"), delta.get("reasoning_content") or delta.get("reasoning")
            if c or rc:
                first = first or time.time()
                n += 1
                (content if c else reasoning).append(c or rc)
    t1 = time.time()
    ttft = (first or t1) - t0
    tps = (n - 1) / (t1 - first) if first and n > 1 and t1 > first else 0.0
    answer = "".join(content)
    print(f"── {label}: TTFT {ttft:.2f} s, {n} chunks, {tps:.1f} chunks/s, total {t1 - t0:.2f} s")
    if reasoning:
        print(f"   reasoning: {''.join(reasoning)[:400]!r}")
    print(f"   answer   : {answer[:800]!r}")
    return answer


failed = 0
for i, (q, want) in enumerate(CASES):
    got = ask(f"known answer {i + 1}", q)
    ok = want in got.lower()
    failed += not ok
    print(f"   {'PASS' if ok else 'FAIL'} (want {want!r})")
a, b = ask("free text (1)", FREE), ask("free text (2)", FREE)
same = a == b
failed += not same
print(f"── greedy repeat: {'PASS (identical)' if same else 'FAIL (answers differ)'}")
print(f"TEXT CLIENT: {'PASS' if not failed else f'{failed} FAILED'}")
sys.exit(1 if failed else 0)
