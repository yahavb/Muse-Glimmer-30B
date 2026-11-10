#!/usr/bin/env python3
"""Quick POPE check on a fixed subset (same questions every run, for A/B comparisons).

Requests and scoring copy VLMEvalKit @fd3884b (the vllm-neuron Qwen3-VL recipe):
  - message = [image (JPEG base64), question text from the TSV], temperature 0, max_tokens 256
  - answer -> Yes / No / Unknown with VLMEvalKit's YOrN_Extraction rule
  - F1 / precision / recall with "Yes" as the positive class
Positive questions are the same in all three POPE splits, so recall here estimates the
full-run recall. Negatives come from the adversarial split (the hardest).

Usage:
  python3 pope_subset.py --url http://127.0.0.1:8000 --model Muse-Glimmer-30B --n-pos 400 --n-neg 400
"""

import argparse
import base64
import csv
import io
import json
import random
import re
import string
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

POPE_TSV = "https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv"


def yorn(output: str) -> str:  # VLMEvalKit YOrN_Extraction (punctuation stripped, word match)
    s = output.lower()
    s = re.sub(f"[{re.escape(string.punctuation)}]", " ", s)
    words = s.split()
    if "yes" in words and "no" not in words:
        return "Yes"
    if "yes" not in words and "no" in words:
        return "No"
    return "Unknown"


def load_rows(path_or_url: str):
    if path_or_url.startswith("http"):
        raw = urllib.request.urlopen(path_or_url, timeout=600).read().decode()
    else:
        raw = open(path_or_url).read()
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(io.StringIO(raw), delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Muse-Glimmer-30B")
    ap.add_argument("--tsv", default=POPE_TSV)
    ap.add_argument("--n-pos", type=int, default=400)
    ap.add_argument("--n-neg", type=int, default=400)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    rows = load_rows(args.tsv)
    by_index = {r["index"]: r for r in rows}

    def image_b64(r):  # VLMEvalKit stores some images as a reference to another row's index
        v = r["image"]
        return by_index[v]["image"] if v in by_index and len(v) < 32 else v

    rng = random.Random(0)
    pos = [r for r in rows if r["answer"] == "Yes"]
    neg = [r for r in rows if r["answer"] == "No" and "adversarial" in r.get("category", "")]
    rng.shuffle(pos)
    rng.shuffle(neg)
    picked = pos[: args.n_pos] + neg[: args.n_neg]
    print(f"POPE subset: {len(picked)} questions ({min(args.n_pos, len(pos))} yes, "
          f"{min(args.n_neg, len(neg))} adversarial no) from {len(rows)} rows, seed 0")

    sizes = Counter()
    try:
        from PIL import Image
        for r in picked:
            sizes[Image.open(io.BytesIO(base64.b64decode(image_b64(r)))).size] += 1
        print("image sizes (w x h):", ", ".join(f"{w}x{h}:{n}" for (w, h), n in sizes.most_common(6)))
    except Exception as e:  # sizes are informational only
        print("image sizes: n/a", e)

    def ask(r):
        body = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64(r)}},
                    {"type": "text", "text": r["question"]}]}]}
        req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                out = json.load(urllib.request.urlopen(req, timeout=600))
                msg = out["choices"][0]["message"]
                return r, msg.get("content") or "", out.get("usage", {}).get("completion_tokens", 0)
            except Exception as e:
                err = e
                time.sleep(5)
        return r, f"ERROR {err}", 0

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(args.nproc) as ex:
        for i, res in enumerate(ex.map(ask, picked), 1):
            results.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(picked)} done, {i / (time.time() - t0) * 60:.1f} q/min", flush=True)

    tp = fp = fn = tn = unk = empty = err = 0
    toks = []
    for r, ans, n in results:
        toks.append(n)
        pred = yorn(ans)
        empty += ans.strip() == ""
        err += ans.startswith("ERROR")
        unk += pred == "Unknown"
        if r["answer"] == "Yes":
            tp += pred == "Yes"
            fn += pred != "Yes"
        else:
            fp += pred == "Yes"
            tn += pred != "Yes"
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"POPE-SUBSET RESULT: F1={100 * f1:.2f} precision={100 * precision:.2f} recall={100 * recall:.2f} "
          f"acc={100 * (tp + tn) / len(results):.2f} | tp={tp} fp={fp} fn={fn} tn={tn} | "
          f"unknown={unk} empty={empty} errors={err} | completion tokens mean={sum(toks) / len(toks):.1f} "
          f"max={max(toks)} | {time.time() - t0:.0f} s")
    for r, ans, _ in results[:3] + [x for x in results if x[0]["answer"] == "Yes" and yorn(x[1]) != "Yes"][:5]:
        print(f"  [{r['answer']}] {r['question']!r} -> {ans[:100]!r}")


if __name__ == "__main__":
    main()
