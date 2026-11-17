# Muse-Glimmer-30B text serving on trn2: SGLang (sglang-neuron) vs vLLM Neuron

Same node (trn2.48xlarge), same claim (`m-lnc2`, 2 devices, LNC2), TP=8, context 2048, one request at a time,
greedy, the same client (`scripts/sglang_text_client.py`): 63-token prompt "Explain quantum computing in simple terms.",
256 tokens out, median of 3 warm runs (spread 0.00 s on every row).

| Server | Build | KV pool | TTFT | Decode | 256-token answer |
| --- | --- | --- | --- | --- | --- |
| vLLM Neuron (`muse-glimmer-30b-vllm-text-job.yaml`) | vllm-neuron `muse-glimmer-30b` fb32962b, DLC 0.29.0.1.2.0 | 512 x 32 = 16,384 tokens | 0.390 s | 12.39 tok/s | **20.98 s** |
| SGLang (`muse-glimmer-30b-sglang-job.yaml`) | sglang-neuron fork `muse-glimmer-text` 8515b53, image v0.1.0 | default, 305K tokens (7.56 GB/rank) | 0.642 s | 6.27 tok/s | 41.29 s |
| SGLang, `--max-total-tokens 16384` | sglang-neuron dafe0e1 | 16,384 tokens (0.41 GB/rank) | 0.428 s | 11.18 tok/s | 23.24 s |
| SGLang, 16K pool + `MUSE_GLIMMER_KV_ONEHOT_GATHER=1` (the job's default) | sglang-neuron f607a28 | 16,384 tokens | 0.239 s | 44.64 tok/s | **5.95 s** |

Answers: the two known-answer prompts (391, Paris) are identical across all three. The free-text answer is identical
between the two SGLang rows (byte for byte) and differs from vLLM from character 317 (bf16 numerics of two compiled
stacks; both answers coherent).

GSM8K (first 100 questions, SGLang's `python3 -m sglang.test.few_shot_gsm8k`, 5-shot, greedy, the plugin's own
accuracy check): SGLang 16K pool **0.97**, with the one-hot gather **0.98**; 0 invalid answers on both.

The one-hot build's free-text answer is deterministic (identical over 4 runs) and differs from the gather build from
character 144 (both coherent; the gather build and vLLM also differ from each other, from character 317).

## Findings

- SGLang's decode is device-bound: one decode program per token, back to back (system trace, rank 0: 161.3 ms per
  execution, 161.3 ms start to start); its 106 collectives take 1.5 ms of it.
- The decode's cost grows with the KV pool: 305K tokens 6.27 tok/s, 16K tokens 11.18 tok/s, same output. The write
  that held: the cost is the KV gather, whose cost grows with the pool it reads from (below).
- The decode device profile (16K pool, sglang-neuron dafe0e1, rank 0, 2 steps of 91 ms) put the KV page gather
  (`self.k_cache[table]`, graph.py) at about 45 ms of GpSimd per step, plus its copies and transposes on the other
  engines: a dynamic DMA of small descriptors. The one-hot matmul over the pool (static reads, one 1 x term per
  output element) replaces it: 89.4 -> 22.4 ms per token.
- Refuted (dense decode, `MUSE_GLIMMER_DENSE_DECODE`): attending over the whole pool, the row's slots from the block
  table's inverse: 50.25 tok/s, but the answer depends on where the request's pages sit in the pool (the first run and
  the prefix-cache runs differ), so not kept.
- Refuted: the K/V write as rows of a `[slots, d]` view (`MUSE_GLIMMER_KV_ROW_WRITE`): 6.25 tok/s at the big pool, 11.11
  at 16K (no change); the warm-up's in-place check passed, so the write was not the pool-size cost.
- Refuted: the gather as an `index_select` of whole pages (`MUSE_GLIMMER_KV_ROW_GATHER`): 10.92 tok/s against 11.18.
- Refuted: compiling with `--model-type=transformer` (vLLM's) instead of the platform's generic type: 6.16 tok/s
  against 6.27, and the greedy repeat stopped being identical (`MUSE_GLIMMER_NEURON_MODEL_TYPE`, off by default).
