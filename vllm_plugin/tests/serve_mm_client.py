"""Text + image checks against a running `vllm serve` (localhost only), one server session.

  1. text   — the ladder's text prompt (regression vs phase 1)
  2. image  — bee.jpg description + known-answer images (eval_images/cases.json): PASS/FAIL
  3. text   — the same text prompt again, after images: must match step 1 (greedy)
  4. video  — media/cat_sunglasses.mp4 description (the validated clip, V3-V5)
  5. hybrid — text, image, video in one more round: text must still match step 1

Greedy decoding, streamed; prints TTFT and decode tok/s per request.

    python serve_mm_client.py --url http://127.0.0.1:8000 --cases eval_images/cases.json
"""
import argparse
import base64
import json
import os
import time
import urllib.request

BEE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
TEXT_PROMPT = "Explain quantum computing in simple terms."

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8000")
p.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
p.add_argument("--cases", default="eval_images/cases.json")
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--video", default="media/cat_sunglasses.mp4")
args = p.parse_args()


def data_url(src):
    raw = urllib.request.urlopen(src).read() if src.startswith("http") else open(src, "rb").read()
    mime = "image/png" if src.endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def ask(label, text, image=None, video=None):
    content = [{"type": "text", "text": text}]
    if image:
        content.insert(0, {"type": "image_url", "image_url": {"url": data_url(image)}})
    if video:
        raw = open(video, "rb").read()
        content.insert(0, {"type": "video_url",
                           "video_url": {"url": f"data:video/mp4;base64,{base64.b64encode(raw).decode()}"}})
    body = {"model": args.model, "temperature": 0.0, "max_tokens": args.max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": "Reasoning strength: low"},
                         {"role": "user", "content": content}]}
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, t_first, parts, usage = time.time(), None, [], {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    d = ch.get("delta") or {}
                    piece = d.get("content") or d.get("reasoning_content")
                    if piece:
                        t_first = t_first or time.time()
                        parts.append(piece)
    except Exception as e:  # report and keep going: later checks still run
        print(f"REQ {label}: ERROR {type(e).__name__}: {e}")
        return ""
    t_end = time.time()
    n = usage.get("completion_tokens", 0)
    ttft = (t_first or t_end) - t0
    dec = t_end - (t_first or t_end)
    e2e = t_end - t0
    print(f"REQ {label}: prompt_tokens={usage.get('prompt_tokens')} completion_tokens={n} TTFT={ttft:.3f}s "
          f"decode={(n - 1) / dec if dec > 0 and n > 1 else 0:.1f} tok/s "
          f"e2e={e2e:.2f}s ({n / e2e if e2e > 0 else 0:.1f} tok/s)")
    return "".join(parts)


def final(text):
    return text.rsplit("to=user", 1)[-1].strip()


print("════════ 1. TEXT (regression) ════════")
text1 = ask("text-1", TEXT_PROMPT)
print("ANSWER text-1:", final(text1)[:600])

print("════════ 2. IMAGE ════════")
bee = ask("image-bee", "Describe this image.", BEE)
print("ANSWER image-bee:", final(bee)[:600])
case_dir = os.path.dirname(os.path.abspath(args.cases))
passed = cases = 0
for c in json.load(open(args.cases)):
    ans = final(ask(f"case-{c['id']}", c["prompt"], os.path.join(case_dir, c["image"])))
    ok = any(k.lower() in ans.lower() for k in c["expect"])
    cases += 1
    passed += ok
    print(f"CASE {c['id']}: {'PASS' if ok else 'FAIL'} | expect {c['expect']} | answer: {ans[:200]!r}")
print(f"CASES: {passed}/{cases} passed")

print("════════ 3. TEXT after images (hybrid) ════════")
text2 = ask("text-2", TEXT_PROMPT)
same = text1 == text2
print(f"TEXT REGRESSION: text-2 {'IDENTICAL to' if same else 'DIFFERS from'} text-1"
      + ("" if same else f" (first diff at char {next((i for i, (a, b) in enumerate(zip(text1, text2)) if a != b), min(len(text1), len(text2)))})"))

print("════════ 4. VIDEO ════════")
vid = final(ask("video-cat", "Describe this video.", video=args.video))
ok = all(k in vid.lower() for k in ("cat", "sunglasses"))
print(f"CASE video-cat: {'PASS' if ok else 'FAIL'} | expect cat + sunglasses | answer: {vid[:400]!r}")

print("════════ 5. HYBRID: text, image, video ════════")
text3 = ask("text-3", TEXT_PROMPT)
bee2 = final(ask("image-bee-2", "Describe this image.", BEE))
vid2 = final(ask("video-cat-2", "Describe this video.", video=args.video))
print(f"HYBRID text-3 {'IDENTICAL to' if text3 == text1 else 'DIFFERS from'} text-1 | "
      f"image-bee-2 {'IDENTICAL to' if bee2 == final(bee) else 'DIFFERS from'} image-bee | "
      f"video-cat-2 {'IDENTICAL to' if vid2 == vid else 'DIFFERS from'} video-cat")
