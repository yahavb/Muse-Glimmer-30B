# Running Muse-Glimmer-30B on AWS Trainium: native PyTorch, vLLM and SGLang

Muse-Glimmer-30B is a 30B vision-language model: text, image and video in, text out. This post shows three ways to
run it on Trainium, all with native pieces and no model rewrite into a separate framework:

1. **Native PyTorch** with TorchNeuron, the PyTorch backend for Trainium, on the Hugging Face model code.
2. **vLLM** with its Neuron plugin, `vllm-neuron`, where Muse-Glimmer is a contrib model.
3. **SGLang** with its Neuron plugin, `sglang-neuron`, where Muse-Glimmer is a new model package (text only for now).

Every number below was measured by a job in this repository; the source of each is linked.

## The model in one paragraph

52 decoder layers, hidden size 6656, 32 query heads and 2 KV heads of 128, SwiGLU MLP of 19968, vocabulary 202048.
Layers repeat [sliding, sliding, sliding, full]: sliding layers use a 2048-token window with RoPE, full layers use no
position encoding. It has a few details the serving stacks must get right: a weight-less RMSNorm after the
embedding, four "centered" norms per layer (`x * (1 + w)`), a gated attention output
(`o_proj(attn * sigmoid(gate_proj(x)))`), a weight-less Q/K norm with the softmax scale folded into Q, and soft-capped
logits. Every port here uses the same plain PyTorch math for these.

## 1. Native PyTorch (TorchNeuron)

TorchNeuron registers Trainium as a PyTorch device. The Hugging Face `transformers` code runs as is, eager or under
`torch.compile(backend="neuron", fullgraph=True)`.

**Image:** `public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04` (torch 2.14,
torch-neuronx 2.14.3, Neuron SDK 2.33).

**Run it:**

```bash
kubectl apply -f muse-glimmer-30b-video-job.yaml     # video rungs + profile; clones this repo
kubectl logs -f job/muse-glimmer-30b-video > /tmp/muse-glimmer-30b-video
```

`run_muse_glimmer.py` is the entrypoint (eager or compiled, text, image or video, TP through `torchrun`);
`muse_compile.py` holds what `fullgraph=True` needed:

- **Text decoder:** a `StaticCache` with a full-length layer per layer (the stock sliding cache recompiled every step);
  the sliding mask enforces the 2048 window. Two graphs: prefill and decode.
- **Vision tower:** 20 graph breaks in stock code, all from host-side grid bookkeeping. Precomputing that metadata on
  the host and running window attention as one SDPA call under a block-diagonal mask removes them.
- **Tensor parallel:** create the process group without `device_id` (otherwise `DeviceMesh` calls `split_group`, which
  the Neuron backend does not support). With 2 KV heads the stock TP plan stops at TP=2.

**Result (trn3, TP=2, greedy, 256 tokens; [RESULTS.md](../RESULTS.md)):** eager 3.3 tok/s, compiled 15.2 tok/s on text,
15.3 tok/s with an image, 15.1 tok/s with video.

This path is the reference: it is the upstream model code, so the serving ports below are checked against its math.

## 2. vLLM with vllm-neuron

vllm-neuron is vLLM's out-of-tree platform plugin for Trainium. A model joins as a contrib package that registers
itself in every vLLM process through vLLM's `vllm.general_plugins` entry point. Muse-Glimmer's is
`contrib/models/muse_glimmer` in [vllm-project/vllm-neuron](https://github.com/vllm-project/vllm-neuron), with a full
tutorial in that folder (`TUTORIAL.md`): text, image and video, one image or one video per prompt.

**Image:** `public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04`.

**Serve it** (inside the image, with the contrib package installed: `pip install -e contrib/models/muse_glimmer`):

```bash
export VLLM_NEURON_BACKEND=neuron_native NEURON_LOGICAL_NC_CONFIG=2
vllm serve meta-models/Muse-Glimmer-30B --served-model-name Muse-Glimmer-30B \
  --tensor-parallel-size 8 --max-model-len 2048 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --num-gpu-blocks-override 512 \
  --reasoning-parser muse_glimmer --default-chat-template-kwargs '{"reasoning_strength": "low"}'
```

Things that each cost a run to find:

- `VLLM_NEURON_BACKEND=neuron_native` is required on the 0.29 image.
- Set `NEURON_LOGICAL_NC_CONFIG` to the logical core size your devices are claimed with; vllm-neuron defaults to 1.
- `--reasoning-parser muse_glimmer` splits the model's reasoning channel from the answer (`message.reasoning` on 0.29).

**Make decode fast: the one-hot KV gather.** The contrib model reads each request's cached K/V pages with
`self.k_cache[block_table]`. On Trainium that gather is a dynamic DMA of many small descriptors, and it dominated the
decode step (section 4). [`patches/vllm-neuron-muse-glimmer-onehot-kv-gather.patch`](../patches/vllm-neuron-muse-glimmer-onehot-kv-gather.patch)
adds an opt-in replacement: a one-hot matmul over the cache that picks the same pages in the same order with plain
sequential reads. It applies to vllm-project/vllm-neuron main; turn it on with `MUSE_GLIMMER_KV_ONEHOT_GATHER=1`.

**Result (trn2, TP=8 on 2 devices; [results/sglang-text-trn2](../results/sglang-text-trn2/README.md)):**

| vLLM | Time to first token | Decode | 256-token answer | GSM8K (100) |
| --- | --- | --- | --- | --- |
| as shipped | 0.390 s | 12.38 tok/s | 20.98 s | 0.96 |
| one-hot KV gather | **0.193 s** | **59.81 tok/s** | **4.46 s** | 0.95 |

Measured on the vllm-neuron `muse-glimmer-30b` branch (vLLM 0.32), the same contrib model; the job is
[`muse-glimmer-30b-vllm-text-job.yaml`](../muse-glimmer-30b-vllm-text-job.yaml).

## 3. SGLang with sglang-neuron

[sglang-neuron](https://github.com/sgl-project/sglang-neuron) is SGLang's platform plugin for Trainium: SGLang's own
launcher, scheduler and OpenAI API, with the plugin supplying the device, KV pool, graph runner and models. It runs on
TorchNeuron. Its first release serves token ids only, so Muse-Glimmer runs text only there for now; SGLang already
knows the checkpoint (config, reasoning parser `muse`, `--language-model-only`).

The model is a new package, `sglang_neuron/models/muse_glimmer`, plus one catalog entry, on the fork
[yahavb/sglang-neuron, branch `muse-glimmer-text`](https://github.com/yahavb/sglang-neuron/tree/muse-glimmer-text)
(also as [`patches/sglang-neuron-muse-glimmer.patch`](../patches/sglang-neuron-muse-glimmer.patch)). The decoder is
the same plain PyTorch math as the vLLM port.

**Image:** `public.ecr.aws/neuron/pytorch-inference-sglang-neuron:v0.1.0-neuron-py314-sdk-2.33.0-ubuntu26.04`. Install
the fork over the image's plugin (`pip uninstall -y sglang-neuron; pip install --no-deps --no-build-isolation -e .`).

**Serve it:**

```bash
export SGLANG_PLATFORM=neuron MUSE_GLIMMER_KV_ONEHOT_GATHER=1
python3 -m sglang.launch_server --model-path meta-models/Muse-Glimmer-30B --served-model-name Muse-Glimmer-30B \
  --language-model-only --tp-size 8 --context-length 2048 --chunked-prefill-size 1024 \
  --max-running-requests 1 --max-total-tokens 16384 --mem-fraction-static 0.6 \
  --reasoning-parser muse --default-chat-template-kwargs '{"reasoning_strength": "low"}'
```

Things that each cost a run to find:

- The plugin refuses any environment variable starting with `SGLANG_NEURON_` that it does not read, so name your own
  variables differently.
- Install the fork over the image's plugin with an uninstall first, or both copies register SGLang hooks.
- Size the KV pool. By default the pool fills the memory left after the weights (305K tokens here), and the gather's
  cost grows with the pool: decode went from 6.27 to 11.18 tok/s with `--max-total-tokens 16384`, same answers.

**Result (trn2, TP=8 on 2 devices; [results/sglang-text-trn2](../results/sglang-text-trn2/README.md)):**

| SGLang | Time to first token | Decode | 256-token answer | GSM8K (100) |
| --- | --- | --- | --- | --- |
| first port, default pool | 0.642 s | 6.27 tok/s | 41.29 s | not run |
| pool of 16K tokens | 0.428 s | 11.18 tok/s | 23.24 s | 0.97* |
| + one-hot KV gather | **0.241 s** | **44.64 tok/s** | **5.95 s** | 0.98 |

\* measured with SGLang's own `python3 -m sglang.test.few_shot_gsm8k` (same prompts, native `/generate` API); the
other SGLang and vLLM scores use `scripts/gsm8k_openai.py`. The one-hot build scored 0.98 with both.

The job is [`muse-glimmer-30b-sglang-job.yaml`](../muse-glimmer-30b-sglang-job.yaml).

## 4. How the decode got 4-5x faster

The method: measure both servers on the same node with the same client, then profile, then change one thing at a
time behind a flag, and keep a change only if the answers hold.

1. **Same setup for both.** One trn2 node, the same 2-device claim, TP=8, context 2048, one request at a time, greedy,
   the same client ([`scripts/sglang_text_client.py`](../scripts/sglang_text_client.py)) and the same GSM8K prompts
   ([`scripts/gsm8k_openai.py`](../scripts/gsm8k_openai.py), SGLang's few-shot GSM8K over `/v1/completions`).
2. **System trace.** SGLang's `/start_profile` with the `NEURON` activity showed one decode program per token, back to
   back, with no host gaps, and its 106 collectives took 1.5 ms of a 161 ms step. So the time was inside the program.
3. **Device profile.** The plugin's serve-window profile (`SGLANG_NEURON_PROFILE_SERVE_RANKS`) and `neuron-explorer`
   put about half of each step on the KV page gather (`k_cache[block_table]`): GpSimd-driven dynamic DMA, plus copies
   and transposes on the other engines.
4. **The fix.** A one-hot matmul `onehot(block_table) @ cache` reads the whole cache with static DMA and runs a tiny
   matmul on the tensor engine. Each output element is a single `1 * x` term (the pool starts as zeros and holds only
   finite values), so it picks the same values.

Ideas that did not help, each measured on the same setup:

| Idea | Result |
| --- | --- |
| compile with `--model-type=transformer` | 6.16 tok/s against 6.27, and greedy repeats stopped matching |
| write K/V as rows of a `[slots, d]` view | no change (the write was already in place) |
| gather pages with `index_select` | 10.92 tok/s against 11.18 |
| attend over the whole pool with a mask (no gather) | 50.25 tok/s, but the answer depended on where pages sat in the pool |

## 5. What the three paths give you today

| Path | Hardware | Inputs | Decode (256 tokens) |
| --- | --- | --- | --- |
| Native PyTorch, `torch.compile` | trn3, TP=2 | text, image, video | 15.2 tok/s |
| vLLM + vllm-neuron, one-hot gather | trn2, TP=8 | text, image, video | 59.81 tok/s (text) |
| SGLang + sglang-neuron, one-hot gather | trn2, TP=8 | text | 44.64 tok/s |

The rows use different hardware and parallelism, so compare within a row's section, not across rows.

Answers stay correct along the way: the known-answer prompts (17 x 23 = 391, the capital of France) are identical
everywhere, GSM8K stays between 0.95 and 0.98 on 100 questions (one question is 0.01), and every kept build gives the same
answer to the same prompt every time.

## What is next

- Image and video on SGLang: sglang-neuron's graphs need an embeddings input and a vision encoder graph first.
- The one-hot gather reads the whole KV pool each step, so it suits small pools (one request at a time here). Large
  batches need a paged attention kernel instead.
- GSM8K on the full 1,319 questions, to separate the servers' accuracy beyond 100 questions.
