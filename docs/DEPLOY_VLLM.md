# Deploy Muse-Glimmer-30B with vLLM on Trainium3 (text, image, video)

An OpenAI-compatible server on one trn3 node: vLLM Neuron (plugin 0.29.0.1.2.0, vLLM 0.29.0)
plus this repo's model plugin (`vllm_plugin/`, loaded automatically via `vllm.general_plugins`).

## What you get

| | |
|---|---|
| API | OpenAI chat completions, `http://muse-glimmer-30b-vllm:8000/v1` in-cluster (or port-forward to `127.0.0.1:8000`) |
| Model name | `meta-models/Muse-Glimmer-30B` |
| Inputs | text; 1 image per prompt; 1 video per prompt |
| Images | resized to 448x448 (256 image tokens) |
| Video | 8 frames sampled evenly, resized to 480x368 (520 video tokens), real timestamps |
| Context | 2048 tokens per request, 1 request at a time |
| Hardware | `m-lnc2` claim (2 trn3 devices), TP=2 |
| First start | ~12 min (model download + compile) |

## Two stacks: bare TorchNeuron vs vLLM

| | Bare bring-up (R1-R5, V3-V5, profiles) | vLLM serving |
|---|---|---|
| Image | `public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04` | `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| Neuron PyTorch library | **`torch-neuronx`** (TorchNeuron) 2.14.3.0.1 | **`libtorch-neuronx-lite`** (pulled in by vllm-neuron) |
| torch | 2.14.0 | 2.13.0 |
| transformers | 5.20.2 | 5.17.0 |
| Serving | our scripts (`run_muse_glimmer.py`) | vLLM 0.29.0 + vllm-neuron 0.29.0.1.2.0 + this repo's `vllm_plugin/` |
| Compile | `torch.compile(fullgraph=True, backend="neuron")` | vllm-neuron's `torch.compile` (`fullgraph=True`, `VLLM_NEURON_BACKEND=neuron_native`) |
| Env | — | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` (keep `torch-neuronx` from loading if present) |
| SDK | 2.33 | 2.33 |

The model math in `vllm_plugin/` mirrors the bare path (plain PyTorch); the libraries under it differ.

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
no authentication. The manifest adds a ClusterIP Service only (no NodePort / LoadBalancer), so nothing outside the cluster reaches it.
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

Other pods call the ClusterIP Service in the manifest:
`http://muse-glimmer-30b-vllm.default.svc.cluster.local:8000/v1` (no authentication). It is not
reachable from outside the cluster. To limit which pods can call it, add a NetworkPolicy for the
caller's namespace/pods.

## Health, fault injection, NEFF cache

The liveness probe checks the devices, not just the process. Faults (`hang`, `crash`) can be
injected per rank. Weights and NEFFs live on the node (`/opt/muse-glimmer-30b`);
`NEFF_SOURCE=hostpath` loads NEFFs from there and never compiles. See `docs/RESILIENCE_VLLM.md`.

## Limits

- One request at a time (`--max-num-seqs 1`), as validated. Raising it needs more decode buckets
  and a re-validation.
- Decode runs at ~9.3 tok/s (plain-PyTorch paged attention); the bare compiled path ran 15.4.
- Fixed media sizes keep one compiled vision graph each; other sizes are resized to these.
