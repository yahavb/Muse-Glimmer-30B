# Muse-Glimmer-30B on vLLM Neuron: TP x LNC latency sweep (trn3)

Which tensor-parallel degree (TP) and logical core size (LNC) gives the fastest response, using the resource claim template (RCT) ladder. All runs on the same trn3 instance type, same image and model commit.

**Fastest: TP=8, LNC=2 on `m-lnc2` (2 chips)**: 14.0 tok/s decode, 119 ms TTFT for text, 18.36 s for a 256-token answer.  
**Fastest on one chip: TP=8, LNC=1 on `s-lnc1`**: 11.4 tok/s, 22.50 s per 256-token answer.

## Results (measured run, after a warm-up run)

TTFT = time to first token. e2e = full 256-token answer. Decode speed is the same for every request in a run.

| TP | LNC | RCT | Chips | Decode tok/s | TTFT text | TTFT image | TTFT video | e2e text | e2e image | e2e video | Image cases | Video | Text after images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | `s-lnc1` | 1 | does not fit: weights do not fit (nrt_tensor_allocate status=4) | | | | | | | | | |
| 4 | 1 | `s-lnc1` | 1 | does not fit: no room for KV cache (budget 0.00 GiB) | | | | | | | | | |
| 8 | 1 | `s-lnc1` | 1 | 11.4 | 171 ms | 281 ms | 194 ms | 22.50 s | 22.61 s | 22.52 s | 3/3 | PASS | identical |
| 16 | 1 | `m-lnc1` | 2 | 12.0 | 115 ms | 252 ms | 144 ms | 21.32 s | 21.46 s | 21.35 s | 3/3 | PASS | identical |
| 2 | 2 | `s-lnc2` | 1 | 9.3 | 321 ms | 443 ms | 345 ms | 27.86 s | 27.98 s | 27.88 s | 3/3 | PASS | identical |
| 4 | 2 | `s-lnc2` | 1 | 10.9 | 194 ms | 315 ms | 223 ms | 23.56 s | 23.68 s | 23.58 s | 3/3 | PASS | identical |
| **8** | 2 | `m-lnc2` | 2 | **14.0** | 119 ms | 237 ms | 149 ms | 18.36 s | 18.47 s | 18.39 s | 3/3 | PASS | identical |
| 16 | 2 | `l-lnc2` | 4 | 13.7 | 91 ms | 208 ms | 121 ms | 18.67 s | 18.79 s | 18.70 s | 3/3 | PASS | identical |

## What it means

- **LNC=2 beats LNC=1 on the same number of chips.** On 2 chips: TP=8 LNC=2 decodes at 14.0 tok/s, TP=16 LNC=1 at 12.0.
- **Past 2 chips there is no gain.** TP=16 on 4 chips (`l-lnc2`) decodes at 13.7 tok/s, slightly slower than TP=8 on 2 chips; its first token is about 30 ms sooner (91 ms against 119 ms for text).
- **On one chip, TP=8 LNC=1 is the fastest**: 11.4 tok/s, 5% faster than TP=4 LNC=2 (10.9), and 23% faster than TP=2 LNC=2 (9.3).
- **TP=2 and TP=4 do not fit with LNC=1.** An LNC=1 core has half the device memory of an LNC=2 core: at TP=2 the weights (about 30 GB per rank) fail to load; at TP=4 they load but leave no memory for the KV cache.
- **Same answers everywhere.** Every configuration passes the 3 image cases and the video case, and gives the same text answer.
- Muse has 2 KV heads. For TP > 2, vLLM replicates each KV head over TP/2 ranks (QKVParallelLinear, `num_kv_head_replicas`), so no head is split.

## Inputs

The repo's client, unchanged: `vllm_plugin/tests/serve_mm_client.py --cases eval_images/cases.json --max-tokens 256`, greedy, streamed. Per run: text, image (bee), 3 image cases (count red circles, read a word, main color), text again (regression), video (cat with sunglasses), and text + image + video. Each job runs the client twice and records the second run (`raw/`), so compile and first-request effects are excluded.

Serving settings as `muse-glimmer-30b-vllm-job.yaml`: max-model-len 2048, max-num-seqs 1, max-num-batched-tokens 1024, 512 KV blocks, image and video limits 1, encoder buckets [1024, 2080]. `NEURON_LOGICAL_NC_CONFIG` equals the RCT's logical core size.

## Setup

- trn3 (`trn3-dev1.48xlarge`), EKS, DRA resource claims; one job per configuration; weights and NEFFs on the node cache.
- Image `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (SDK 2.33): vllm 0.29.0, torch 2.13.0, transformers 5.17.0, this repo's plugin.
- Repo commit `1a8bbd16` (branch `exp/vllm-phase2`).
- Each RCT is the smallest rung with enough logical cores: `s-*` 1 chip, `m-*` 2 chips, `l-*` 4 chips; LNC=2 gives 4 cores per chip, LNC=1 gives 8. TP=32 (`l-lnc1`) was not run: the vision encoder's 16 heads cannot be split over 32 ranks.

## Reproduce

```
python3 make_jobs.py jobs          # regenerates jobs/ from ../../muse-glimmer-30b-vllm-job.yaml
for f in jobs/*.yaml; do kubectl apply -f $f; done
kubectl logs job/muse-bench-tp8-lnc2 | sed -n '/BENCH RESULT/,/END BENCH RESULT/p'
```
