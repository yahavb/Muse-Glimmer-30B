"""Few-shot GSM8K over the OpenAI completions API, for any server (SGLang or vLLM): the same prompts, stops, answer
rule and scores as SGLang's `python3 -m sglang.test.few_shot_gsm8k` (5-shot from the first test questions, greedy,
512 new tokens, stop on "Question" / "Assistant:" / "<|separator|>", last integer of the answer), sent to
/v1/completions so both servers run the identical requests.

    python scripts/gsm8k_openai.py --url http://127.0.0.1:8000 --model Muse-Glimmer-30B --num-questions 100
"""
import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request

INVALID = -9999999
URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:30000")
p.add_argument("--model", default="Muse-Glimmer-30B")
p.add_argument("--num-questions", type=int, default=100)
p.add_argument("--num-shots", type=int, default=5)
p.add_argument("--max-new-tokens", type=int, default=512)
p.add_argument("--parallel", type=int, default=8)
p.add_argument("--data", default="/tmp/gsm8k_test.jsonl")
p.add_argument("--label", default="")
args = p.parse_args()

if not os.path.exists(args.data):
    urllib.request.urlretrieve(URL, args.data)
lines = [json.loads(x) for x in open(args.data)]


def example(i, answer):
    s = "Question: " + lines[i]["question"] + "\nAnswer:"
    return s + " " + lines[i]["answer"] if answer else s


def value(text):
    nums = re.findall(r"\d+", text.replace(",", ""))
    if not nums:
        return INVALID
    try:
        return ast.literal_eval(nums[-1])
    except SyntaxError:
        return INVALID


shots = "".join(example(i, True) + "\n\n" for i in range(args.num_shots))
labels = [value(lines[i]["answer"]) for i in range(args.num_questions)]


def ask(i):
    body = {"model": args.model, "prompt": shots + example(i, False), "max_tokens": args.max_new_tokens,
            "temperature": 0.0, "stop": ["Question", "Assistant:", "<|separator|>"]}
    req = urllib.request.Request(f"{args.url}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["text"], int(d.get("usage", {}).get("completion_tokens", 0)), \
        int(d.get("usage", {}).get("prompt_tokens", 0))


t0 = time.perf_counter()
with cf.ThreadPoolExecutor(args.parallel) as ex:
    out = list(ex.map(ask, range(args.num_questions)))
latency = time.perf_counter() - t0
preds = [value(text) for text, _, _ in out]
acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
invalid = sum(p == INVALID for p in preds) / len(preds)
tokens = sum(n for _, n, _ in out)
print(f"GSM8K {args.label}: questions={len(labels)} prompt_tokens_max={max(n for _, _, n in out)}")
print(f"Accuracy: {acc:.3f}")
print(f"Invalid: {invalid:.3f}")
print(f"Latency: {latency:.3f} s")
print(f"Output throughput: {tokens / latency:.3f} token/s")
print("GSM8K_PREDS", json.dumps(preds))
