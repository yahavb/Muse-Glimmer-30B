"""Check the vLLM Neuron Muse-Glimmer-30B tutorial by running its code blocks as written.

Used by muse-glimmer-30b-tutorial-job.yaml. The job clones the docs branch; this
script pulls the fenced code blocks out of the tutorial file, so what runs is
exactly what is published. Only two substitutions are made:
  /path/to/                    -> $MODELS_ROOT/   (where the job keeps the weights)
  <your Hugging Face token>    -> "$HF_TOKEN"     (the job's secret; never printed)
The job runs each block itself (bash or python3), as a reader would.

Commands:
  extract TUTORIAL OUTDIR        write NN.orig and NN.sh / NN.py per block; check the layout
  expect OUTPUT_FILE WORD...     check a block's printed output contains every word
  fields URL MODEL IMAGE         one image request: where the reasoning goes in the response
  combo URL MODEL IMAGE VIDEO    image + video + text in one prompt
  strength URL MODEL             reasoning strength: low default vs "Reasoning strength: high"
"""

import base64
import hashlib
import json
import os
import re
import sys
import urllib.request

# (language, start of first line) for each block, in order. A change in the
# tutorial that moves blocks fails here instead of running the wrong thing.
LAYOUT = [
    ("bash", "neuron-ls"),
    ("bash", "# The validated configuration"),
    ("bash", "git clone https://github.com/vllm-project/vllm-neuron.git"),
    ("bash", "hf download"),
    ("bash", "vllm serve /path/to/Muse-Glimmer-30B"),
    ("python", "import base64\nfrom openai import OpenAI"),  # online client
    ("python", "import base64\nimport os"),  # offline LLM
    ("bash", "pip install pytest"),
    ("bash", "MUSE_GLIMMER_MODEL_PATH=/path/to/Muse-Glimmer-30B"),
    ("bash", "VLLM_NEURON_CPU_MODE=1 pytest"),
]


def check(name, ok, detail=""):
    print(f"CHECK {'PASS' if ok else 'FAIL'}: {name}" + (f" | {detail}" if detail else ""), flush=True)
    return ok


def extract(path, outdir):
    text = open(path).read()
    print("TUTORIAL sha256:", hashlib.sha256(text.encode()).hexdigest())
    blocks = re.findall(r"^```(\w+)\n(.*?)^```", text, flags=re.S | re.M)
    print("TUTORIAL code blocks:", len(blocks))
    ok = len(blocks) == len(LAYOUT)
    for i, (lang, body) in enumerate(blocks, 1):
        want = LAYOUT[i - 1] if i <= len(LAYOUT) else ("?", "?")
        match = lang == want[0] and body.startswith(want[1])
        ok &= match
        print(f"  block {i:02d} [{lang}] {'ok' if match else 'UNEXPECTED'}: {body.splitlines()[0]}")
    if not ok:
        sys.exit("TUTORIAL block layout changed; update LAYOUT in scripts/tutorial_check.py")
    os.makedirs(outdir, exist_ok=True)
    root = os.environ["MODELS_ROOT"].rstrip("/") + "/"
    for i, (lang, body) in enumerate(blocks, 1):
        open(os.path.join(outdir, f"{i:02d}.orig"), "w").write(body)
        run = body.replace("/path/to/", root).replace("<your Hugging Face token>", '"$HF_TOKEN"')
        open(os.path.join(outdir, f"{i:02d}.{'sh' if lang == 'bash' else 'py'}"), "w").write(run)


def expect(path, words):
    out = open(path).read().lower()
    ok = True
    for w in words:
        ok &= check(f"{os.path.basename(path)} output mentions {w!r}", w.lower() in out)
    ok &= check(f"{os.path.basename(path)} output holds no to=self channel", "to=self" not in out)
    return ok


def _post(url, body):
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def fields(url, model, image):
    # The tutorial says the reasoning is in message.reasoning (vLLM 0.29 field name).
    b64 = base64.b64encode(open(image, "rb").read()).decode()
    r = _post(url, {"model": model, "max_tokens": 256, "temperature": 0.0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "Describe this image."}]}]})
    m = r["choices"][0]["message"]
    print("FIELDS message keys with text:", sorted(k for k, v in m.items() if v))
    print("FIELDS reasoning:", repr((m.get("reasoning") or "")[:300]))
    print("FIELDS content:", repr((m.get("content") or "")[:300]))
    return check("reasoning is in message.reasoning", bool(m.get("reasoning")))


def combo(url, model, image, video):
    # Supported-features table: image + video + text in one prompt.
    img = base64.b64encode(open(image, "rb").read()).decode()
    vid = base64.b64encode(open(video, "rb").read()).decode()
    r = _post(url, {"model": model, "max_tokens": 256, "temperature": 0.0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{vid}"}},
        {"type": "text", "text": "What shapes are in the image, and what animal is in the video?"}]}]})
    ans = (r["choices"][0]["message"].get("content") or "").lower()
    print("COMBO answer:", repr(ans[:400]))
    return check("image + video + text in one prompt: answer mentions circle and cat",
                 "circle" in ans and "cat" in ans)


def strength(url, model):
    # The tutorial says a system message "Reasoning strength: high" overrides the
    # server default (low). Same question both ways; high should reason longer.
    q = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?"
    lens = {}
    for level in ("default", "high"):
        msgs = [{"role": "user", "content": q}]
        if level == "high":
            msgs.insert(0, {"role": "system", "content": "Reasoning strength: high"})
        r = _post(url, {"model": model, "messages": msgs, "max_tokens": 1536, "temperature": 0.0})
        m = r["choices"][0]["message"]
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        lens[level] = len(reasoning)
        print(f"STRENGTH {level}: reasoning {len(reasoning)} chars, completion_tokens "
              f"{r['usage']['completion_tokens']}, finish={r['choices'][0]['finish_reason']}, "
              f"answer {(m.get('content') or '')[:200]!r}")
    return check("system message 'Reasoning strength: high' reasons longer than the low default",
                 lens["high"] > lens["default"], str(lens))


def cachehit(log, limit):
    # The tutorial says a later run with the same configuration loads from the
    # compile cache. vLLM logs "init engine (...) took N s (compilation: M s)".
    took = [float(m.group(1)) for m in re.finditer(r"init engine .*? took ([0-9.]+) s", open(log).read())]
    print("CACHEHIT init engine times (s):", took)
    return check(f"engine init under {limit} s (cache used)", bool(took) and took[-1] < float(limit))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "extract":
        extract(*args)
    elif cmd == "expect":
        sys.exit(0 if expect(args[0], args[1:]) else 1)
    elif cmd == "fields":
        sys.exit(0 if fields(*args) else 1)
    elif cmd == "cachehit":
        sys.exit(0 if cachehit(*args) else 1)
    elif cmd == "combo":
        sys.exit(0 if combo(*args) else 1)
    elif cmd == "strength":
        sys.exit(0 if strength(*args) else 1)
    else:
        sys.exit(f"unknown command {cmd}")
