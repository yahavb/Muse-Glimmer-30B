# Deploy Muse-Glimmer-30B with vLLM on Trainium3 (text, image, video)

An OpenAI-compatible server on one trn3 node: vLLM Neuron (plugin 0.29.0.1.2.0, vLLM 0.29.0)
plus this repo's model plugin (`vllm_plugin/`, loaded automatically via `vllm.general_plugins`).

## What you get

| | |
|---|---|
| API | OpenAI chat completions, `http://<pod IP>:8000/v1` (or port-forward to `127.0.0.1:8000`) |
| Model name | `meta-models/Muse-Glimmer-30B` |
| Inputs | text; 1 image per prompt; 1 video per prompt |
| Images | resized to 448x448 (256 image tokens) |
| Video | 8 frames sampled evenly, resized to 480x368 (520 video tokens), real timestamps |
| Context | 2048 tokens per request, 1 request at a time |
| Hardware | `m-lnc2` claim (2 trn3 devices), TP=2 |
| First start | ~12 min (model download + compile) |

## Prerequisites

1. A trn3 node labeled `node-type: trn3-dev1` and the `m-lnc2` ResourceClaimTemplate.
2. Secret `hf-token` with key `HF_TOKEN`.

## Deploy

```bash
kubectl apply -f muse-glimmer-30b-vllm-deploy.yaml
kubectl rollout status deploy/muse-glimmer-30b-vllm --timeout=60m
kubectl port-forward deploy/muse-glimmer-30b-vllm 8000:8000
```

The server listens on `0.0.0.0:8000`: any pod in the cluster can call it at the pod's IP, with
no authentication. There is no Service or LoadBalancer, so nothing outside the cluster reaches it.
From a laptop, use the port-forward above. Change `GIT_REF` in the manifest to pin a commit.

## Call it

Text:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "meta-models/Muse-Glimmer-30B", "temperature": 0, "max_tokens": 256,
  "messages": [{"role": "system", "content": "Reasoning strength: low"},
               {"role": "user", "content": "Explain quantum computing in simple terms."}]}'
```

Image or video (Python, standard library only):

```python
import base64, json, urllib.request

def ask(text, image=None, video=None):
    content = [{"type": "text", "text": text}]
    if image:  # path to a .png / .jpg
        content.insert(0, {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()}})
    if video:  # path to an .mp4
        content.insert(0, {"type": "video_url", "video_url": {
            "url": "data:video/mp4;base64," + base64.b64encode(open(video, "rb").read()).decode()}})
    body = {"model": "meta-models/Muse-Glimmer-30B", "temperature": 0, "max_tokens": 256,
            "messages": [{"role": "system", "content": "Reasoning strength: low"},
                         {"role": "user", "content": content}]}
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"]
    return reply.rsplit("to=user", 1)[-1].strip()  # drop the model's reasoning channel

print(ask("Describe this image.", image="eval_images/shapes.png"))
print(ask("Describe this video.", video="media/cat_sunglasses.mp4"))
```

A fuller client with timing (TTFT, decode tok/s, end to end): `vllm_plugin/tests/serve_mm_client.py`.

## Serving other workloads in the cluster

Other pods call `http://<pod IP>:8000/v1` directly (`kubectl get pod -l app=muse-glimmer-30b-vllm -o wide`).
For a stable name, add a ClusterIP Service on port 8000. To limit which pods can call it, add a
NetworkPolicy for the caller's namespace/pods.

## Limits

- One request at a time (`--max-num-seqs 1`), as validated. Raising it needs more decode buckets
  and a re-validation.
- Decode runs at ~9.3 tok/s (plain-PyTorch paged attention); the bare compiled path ran 15.4.
- Fixed media sizes keep one compiled vision graph each; other sizes are resized to these.
