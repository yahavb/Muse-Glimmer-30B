# Muse-Glimmer-30B text serving on trn2: SGLang (sglang-neuron) vs vLLM Neuron

Same node (trn2.48xlarge), same claim (`m-lnc2`, 2 devices, LNC2), TP=8, context 2048, one request at a time,
greedy, the same client (`scripts/sglang_text_client.py`): 63-token prompt "Explain quantum computing in simple terms.",
256 tokens out, median of 3 warm runs (spread 0.00 s on every row).

| Server | Build | KV pool | TTFT | Decode | 256-token answer |
| --- | --- | --- | --- | --- | --- |
| vLLM Neuron (`muse-glimmer-30b-vllm-text-job.yaml`) | vllm-neuron `muse-glimmer-30b` fb32962b, DLC 0.29.0.1.2.0 | 512 x 32 = 16,384 tokens | 0.390 s | 12.39 tok/s | **20.98 s** |
| SGLang (`muse-glimmer-30b-sglang-job.yaml`) | sglang-neuron fork `muse-glimmer-text` 8515b53, image v0.1.0 | default, 305K tokens (7.56 GB/rank) | 0.642 s | 6.27 tok/s | 41.29 s |
| SGLang, `--max-total-tokens 16384` | sglang-neuron dafe0e1 | 16,384 tokens (0.41 GB/rank) | 0.428 s | 11.18 tok/s | **23.24 s** |

Answers: the two known-answer prompts (391, Paris) are identical across all three. The free-text answer is identical
between the two SGLang rows (byte for byte) and differs from vLLM from character 317 (bf16 numerics of two compiled
stacks; both answers coherent).

## Findings

- SGLang's decode is device-bound: one decode program per token, back to back (system trace, rank 0: 161.3 ms per
  execution, 161.3 ms start to start); its 106 collectives take 1.5 ms of it.
- The decode's cost grows with the KV pool: 305K tokens 6.27 tok/s, 16K tokens 11.18 tok/s, same output. Hypothesis
  (not profiled): the graph's K/V write (`index_put` on the whole per-layer cache) lowers to a copy of the cache.
- Refuted: compiling with `--model-type=transformer` (vLLM's) instead of the platform's generic type: 6.16 tok/s
  against 6.27, and the greedy repeat stopped being identical (`MUSE_GLIMMER_NEURON_MODEL_TYPE`, off by default).
