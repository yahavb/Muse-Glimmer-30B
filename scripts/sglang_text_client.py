"""Text-only checks and latency against a running OpenAI-compatible server (SGLang or vLLM), greedy, streamed.

  1. known answers: 17 * 23 -> 391, capital of France -> Paris (PASS / FAIL)
  2. latency: one free-text prompt, --runs times; greedy, so every answer must be identical

Token counts come from the server's usage (stream_options.include_usage). Per request it prints one TIMING line:
TTFT, decode tok/s ((completion tokens - 1) / (end - first token)) and end-to-end time; then the median over the
warm runs (all but the first). --dump writes each answer to a JSON file, to compare servers or builds token for token.

    python scripts/sglang_text_client.py --url http://127.0.0.1:30000 --model Muse-Glimmer-30B --runs 4
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30000")
p.add_argument("--model", default="Muse-Glimmer-30B")
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--runs", type=int, default=4)
p.add_argument("--label", default="")
p.add_argument("--dump", default="")
args = p.parse_args()

CASES = [("What is 17 * 23? Answer with the number only.", "391"),
         ("What is the capital of France? Answer in one word.", "paris")]
FREE = "Explain quantum computing in simple terms."


def ask(label, text):
    body = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, content, reasoning, usage = time.time(), None, [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            if not d.get("choices"):
                continue
            delta = d["choices"][0].get("delta", {})
            c, rc = delta.get("content"), delta.get("reasoning_content") or delta.get("reasoning")
            if c or rc:
                first = first or time.time()
                (content if c else reasoning).append(c or rc)
    t1 = time.time()
    n = int(usage.get("completion_tokens") or 0)
    ttft = (first or t1) - t0
    tps = (n - 1) / (t1 - first) if first and n > 1 and t1 > first else 0.0
    answer, thought = "".join(content), "".join(reasoning)
    print(f"TIMING {args.label} {label}: prompt_tokens={usage.get('prompt_tokens')} completion_tokens={n} "
          f"ttft_s={ttft:.3f} decode_tok_s={tps:.2f} e2e_s={t1 - t0:.2f}")
    if thought:
        print(f"   reasoning: {thought[:300]!r}")
    print(f"   answer   : {answer[:300]!r}")
    return {"label": label, "answer": answer, "reasoning": thought, "completion_tokens": n,
            "ttft_s": ttft, "decode_tok_s": tps, "e2e_s": t1 - t0}


failed, results = 0, []
for i, (q, want) in enumerate(CASES):
    r = ask(f"known{i + 1}", q)
    results.append(r)
    ok = want in r["answer"].lower()
    failed += not ok
    print(f"   {'PASS' if ok else 'FAIL'} (want {want!r})")
runs = [ask(f"free{i + 1}", FREE) for i in range(args.runs)]
results += runs
same = len({r["reasoning"] + "\x00" + r["answer"] for r in runs}) == 1
failed += not same
print(f"── greedy repeat over {len(runs)} runs: {'PASS (identical)' if same else 'FAIL (answers differ)'}")
warm = runs[1:] or runs
med = {k: statistics.median(r[k] for r in warm) for k in ("ttft_s", "decode_tok_s", "e2e_s")}
spread = {k: max(r[k] for r in warm) - min(r[k] for r in warm) for k in ("ttft_s", "decode_tok_s", "e2e_s")}
print(f"SUMMARY {args.label}: warm runs={len(warm)} completion_tokens={warm[0]['completion_tokens']} "
      f"ttft_s={med['ttft_s']:.3f} decode_tok_s={med['decode_tok_s']:.2f} e2e_s={med['e2e_s']:.2f} "
      f"(spread ttft {spread['ttft_s']:.3f} s, decode {spread['decode_tok_s']:.2f} tok/s, e2e {spread['e2e_s']:.2f} s)")
if args.dump:
    json.dump(results, open(args.dump, "w"), indent=1)
print(f"TEXT CLIENT: {'PASS' if not failed else f'{failed} FAILED'}")
sys.exit(1 if failed else 0)
