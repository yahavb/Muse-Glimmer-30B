# Muse-Glimmer-30B on AWS Trainium3 with native PyTorch (TorchNeuron)

Runs [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
(text, image and video prompts) on trn3 with
[TorchNeuron](https://pytorch.org/blog/introducing-torchneuron-native-pytorch-backend-for-aws-trainium/),
the native PyTorch backend for Trainium, from the public
[`neuron/torch-neuronx`](https://gallery.ecr.aws/neuron/torch-neuronx) image. Hugging Face
`transformers` model code, BF16, tensor parallel TP=2, `torch.compile(fullgraph=True)`.

Full numbers, methods and every failure along the way: [RESULTS.md](RESULTS.md). Raw logs: [logs/](logs/).

## The ladder (BF16, TP=2, greedy, 256 new tokens)

| Rung | Prompt | Text decoder | Vision tower | End to end | TTFT (vision part) |
|---|---|---|---|---|---|
| R1 | text | eager | — | 3.3 tok/s | — |
| R2 | text | fullgraph | — | 15.2 tok/s | 0.37 s |
| R3 | image 448x448 | eager | eager | 3.3 tok/s | — |
| R4 | image | fullgraph | eager | 15.2 tok/s | 0.30 s (0.17 s) |
| R5 | image | fullgraph | fullgraph | 15.3 tok/s | 0.19 s (0.07 s) |
| V3 | video, 8 frames | eager | eager | 3.2 tok/s | — |
| V4 | video | fullgraph | eager | 14.8 tok/s | 0.77 s (0.54 s) |
| V5 | video | fullgraph | fullgraph | 15.1 tok/s | 0.38 s (0.16 s) |

All rungs answer correctly (known-answer image cases 3/3; image and video descriptions checked).

## What it took to compile with `fullgraph=True`

- **Text decoder:** zero graph breaks with a `StaticCache`. The stock sliding-window cache layer
  keeps a python-int length and recompiles every decode step, so every layer uses a full-length
  `StaticLayer` and the sliding mask enforces the 2048 window. Two graphs total: prefill + decode.
- **Vision tower:** 20 graph breaks in stock code (grid bookkeeping with `.tolist()` and
  data-dependent shapes). Fixed in [`muse_compile.py`](muse_compile.py) with no kernels:
  grid metadata precomputed on the host, per-window attention replaced by one SDPA call under a
  block-diagonal mask (FP32-equivalent to stock to 2.6e-6).
- **Eager on Neuron:** stock eager vision fails to lower `repeat_interleave` with tensor repeats;
  the same host-side metadata fixes the eager rungs.
- **TP on Neuron:** initialize the process group without `device_id`, else `DeviceMesh` calls
  `split_group`, which the neuron backend does not support. Max TP=2 (2 KV heads).

## Run it (Kubernetes, trn3)

| Job | What |
|---|---|
| [`muse-glimmer-30b-video-job.yaml`](muse-glimmer-30b-video-job.yaml) | clones this repo; V3/V4/V5 rungs + vision-path Neuron profile |
| [`muse-glimmer-30b-vision-job.yaml`](muse-glimmer-30b-vision-job.yaml) | image rungs R3/R4/R5 + known-answer cases (code via ConfigMap) |
| [`muse-glimmer-30b-job.yaml`](muse-glimmer-30b-job.yaml) | text rungs R1/R2 (code via ConfigMap) |

```bash
kubectl apply -f muse-glimmer-30b-video-job.yaml
kubectl logs job/muse-glimmer-30b-video > /tmp/muse-glimmer-30b-video
```

Edit `nodeSelector`, the `resourceClaimTemplateName` and the PVC for your cluster. The jobs need an
`hf-token` secret.

## Files

| Path | Purpose |
|---|---|
| `run_muse_glimmer.py` | entrypoint: eager / compiled, text / image / video, TP via `torchrun` |
| `muse_compile.py` | fullgraph pieces: static cache, prefill/decode graphs, vision patch, greedy loop |
| `muse_inputs.py` | image/video loading (fixed size / frame count -> fixed graph shapes) |
| `profile_vision.py` | vision tower alone on one NeuronCore, for profiling |
| `scripts/neuron/profile_vision_job.sh` | 3-run profile: clean median, runtime trace, NEFF+NTFF pairs, parquet op naming |
| `scripts/*_probe.py`, `scripts/*_check.py`, `scripts/smoke_run.py` | CPU checks: graph breaks, recompiles, token equality, end-to-end smoke |
| `eval_images/` | synthetic known-answer images (made for this repo) |
| `media/cat_sunglasses.mp4` | the 1.3 s test clip |

## License

Apache-2.0, see [LICENSE](LICENSE). The model weights are not included and have their own license.
