# Muse-Glimmer-30B on trn3 (TorchNeuron)

## R1 — eager baseline, text prompt (2026-10-21)

| Item | Value |
|---|---|
| Mode | eager (no torch.compile) |
| Parallelism | TP=2 (transformers `tp_plan="auto"`), 2 of 8 NeuronCores used |
| Claim | `m-lnc2` (2 trn3 chips, LNC2) |
| Image | `public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04` |
| Versions | torch 2.14.0, torch_neuronx 2.14.3.0.1, transformers 5.20.2, torchvision 0.29.0 |
| Prompt | "Explain quantum computing in simple terms." (text only, reasoning strength low) |
| Decoding | greedy, 256 new tokens, batch 1 |
| Warmup (first generate, per-op NEFF compile) | 1249.6 s |
| **Timed generate** | **256 tokens in 77.9 s = 3.3 tok/s** |
| Output | coherent, correct answer |


## fullgraph=True readiness (CPU, tiny random-weight model, same model code)

Checked with `scripts/graph_break_probe.py`, `scripts/tp_graph_break_probe.py`,
`scripts/compiled_loop_check.py` (torch 2.14.0, transformers 5.20.2, backend aot_eager).
CPU tracing only: ops the neuron backend can't lower will not show up here.

| Graph | Breaks | fullgraph=True |
|---|---|---|
| Text prefill (StaticCache) | 0 | OK |
| Text decode (StaticCache) | 0 | OK |
| Text prefill/decode, TP=2 (gloo) | 0 | OK |
| Vision tower, stock code | 20 | FAILS (data-dependent grid metadata, per-window `.tolist()` split) |
| Vision tower + `muse_compile.py` fixes | 0 | OK, output identical to stock eager |

Decode loop (`muse_compile.greedy_generate`): exactly 2 graphs (prefill + decode), 0 recompiles
on the timed run, tokens identical to `model.generate` greedy. Needed two fixes:
- Stock `StaticSlidingWindowLayer` keeps a python-int length -> recompiles every decode step.
  Use full-length `StaticLayer` for all layers; the sliding mask still enforces the 2048 window.
- Pre-allocate the cache (`early_initialization`) so prefill doesn't trace an empty cache.

### Memory estimate, TP=2 on trn3 LNC2 (~36 GB per NeuronCore)
- Weights per core: ~14.8B params = ~29.7 GB.
- KV per core: ~26.6 KB/token (52 layers x 1 KV head) -> 8K context = ~0.2 GB.
- Prefill activations at 8K tokens: ~0.1-0.2 GB per tensor; logits only for the last token.
- Headroom ~6 GB per core. Unknown until the device run: compiler scratch and runtime reserve.

## R2 — torch.compile fullgraph=True, text decoder, text prompt (2026-10-21)

Same setup as R1 (TP=2, `m-lnc2`, same image, prompt, greedy, 256 tokens). Text path only.
`muse_compile.make_step` + `greedy_generate`: 2 graphs (prefill, decode), full-length StaticCache.

| Item | R1 eager | R2 compiled |
|---|---|---|
| Warmup / compile | 1249.6 s | 640.4 s |
| TTFT (34-token prefill) | n/a | 0.371 s |
| Decode | n/a | 255 tokens in 16.5 s = 15.5 tok/s |
| **End to end, 256 tokens** | **77.9 s = 3.3 tok/s** | **16.9 s = 15.2 tok/s (4.6x)** |
| Output | coherent | coherent, same answer; text diverges slightly late (bf16 numerics) |

No graph breaks, no op fallback errors, no OOM on device.


## Image test method (R3-R5)

Job: `muse-glimmer-30b-vision-job.yaml`. Same TP=2, `m-lnc2`, greedy decoding.
- Main image: HF `documentation-images/bee.jpg`, resized to 448x448 -> 32x32 patches -> 256
  image tokens, prompt "Describe this image.", 256 new tokens. sha256 printed in the log.
- **Vision check (R5):** compiled vision features vs stock eager features on the same image,
  on device: max abs diff and cosine.
- **Token match:** leading tokens R4/R5 share with R3 on the main image.
- **Known-answer cases** (`eval_images/`, synthetic, generated for this repo, sha256 in the log):
  count red circles (3), read a word (TRAINIUM), name a color (green). PASS = expected keyword in
  the final answer. Every rung answers all cases.
- CPU smoke test of the full flow: `scripts/smoke_run.py` (tiny random model; answers are
  gibberish there, only flow, shapes, recompiles and token equality are checked).

## Image ladder, attempt 1 (2026-10-21) — R3/R4 failed, R5 passed

Ran with the code version before the checks were added (no vision check, cases or token match).

| Rung | Result |
|---|---|
| R3 eager | **FAILED** at the stock vision tower |
| R4 compile text, vision eager | **FAILED**, same error |
| R5 compile text + vision, fullgraph | **15.3 tok/s**, TTFT 0.192 s (vision 0.068 s), warmup 656 s, accurate description (bumblebee on pink cosmos) |

Failure: eager `torch.repeat_interleave(grid_thw[:, 1], counts)` in
`get_vision_interpolation_indices_and_weights` (grid bookkeeping, not vision math) fails to lower:
`'stablehlo.concatenate' op inferred type(s) 'tensor<2xi64>' are incompatible with return type(s)
'tensor<1xi64>'`. Data-dependent output size, which Neuron eager can't lower.
R5 never hits it: its grid metadata is precomputed on the host.

Fix for eager rungs: same host-precomputed grid metadata, stock vision math on device; R3 passes
the features to `generate` as `mm_encoder_outputs`. CPU check: tokens identical to stock
`generate(pixel_values=...)`.

## Image ladder, attempt 2 (2026-10-21) — all rungs ran

(sha256 `8b21ba78…c99a6d`), 448x448, 289 prompt tokens (256 image), 256 new tokens, greedy, TP=2.

| Rung | Text decoder | Vision tower | Warmup | TTFT (vision part) | Decode | End to end | Cases* |
|---|---|---|---|---|---|---|---|
| R3 | eager | eager | 1348.8 s | n/a | n/a | 3.3 tok/s | 3/3 |
| R4 | fullgraph | eager | 541.5 s | 0.585 s (0.460 s) | 15.4 tok/s | 15.0 tok/s | 1/3 (harness) |
| R5 | fullgraph | fullgraph | 259.9 s | 0.196 s (0.072 s) | 15.4 tok/s | 15.3 tok/s | 3/3 |

\* Known-answer cases, 128-token budget. R4's 2 FAILs are harness truncation, not wrong answers:
"The word shown in the image is **TRAIN" (cut at 128 tokens) and an empty answer (budget ran out
right after the answer channel opened). Fixed for the next run: budget 256, answers cut at `<|eot|>`.

- All three main-image descriptions are accurate and consistent (pink cosmos, bumblebee, spent blooms).
- **Vision check (R5):** compiled vs stock eager features on device: cosine 0.999109, max abs diff 3.5.
  On CPU the two paths are bit-identical, so the device gap is eager per-op vs fused-graph bf16 numerics.
- **Token match (greedy, main image):** R3 vs R4 first 9/256 identical (only the text decoder differs),
  R4 vs R5 first 24/256 (only vision differs), R3 vs R5 first 9/256. Compiling the text decoder
  moves numerics more than compiling vision. Divergence is wording, not content.
  **Superseded by attempt 3:** compiled rungs are not deterministic run to run, so this
  per-component attribution does not hold.
- Vision compile cuts the vision part of TTFT 6.4x (0.460 -> 0.072 s) and warmup 2.1x.

## Image ladder, attempt 3 (2026-10-22) — harness fix, all cases pass

answers cut at `<|eot|>`.

| Rung | Text decoder | Vision tower | Warmup | TTFT (vision part) | Decode | End to end | Cases |
|---|---|---|---|---|---|---|---|
| R3 | eager | eager | 1364.0 s | n/a | n/a | 3.3 tok/s | 3/3 |
| R4 | fullgraph | eager | 544.1 s | 0.297 s (0.172 s) | 15.4 tok/s | 15.2 tok/s | 3/3 |
| R5 | fullgraph | fullgraph | 259.6 s | 0.191 s (0.069 s) | 15.4 tok/s | 15.3 tok/s | 3/3 |

- Vision check (R5) unchanged: cosine 0.999109, max abs diff 3.5 — identical to attempt 2.
- R4's eager vision time varies run to run: 0.460 s (attempt 2) vs 0.172 s (attempt 3).

### Run-to-run determinism (main image, greedy, same code path)

| Rung | Attempt 2 vs attempt 3 |
|---|---|
| R3 eager | 256/256 tokens identical |
| R4 compiled text | 9/256 |
| R5 compiled text + vision | 29/256 |

Eager is deterministic across runs; the compiled text decoder is not. The vision features are
(same cosine and max diff both runs), so the variation is in the compiled text graphs. Next run adds
an in-process check (warmup vs timed tokens, same compiled graphs) to tell runtime
nondeterminism (e.g. TP all-reduce / accumulation order) from compile-to-compile differences.

## Video ladder, attempt 1 (2026-10-22) — V3 passed, V4/V5 crashed in the harness

480x368, 21 frames @ 16 fps, 1.3 s. 8 frames sampled evenly [0,3,6,9,11,14,17,20] with real
timestamps -> grid 4x20x26 -> 520 video tokens, 584-token prompt, greedy, 256 new tokens, TP=2.

| Rung | Text decoder | Vision tower | Warmup | End to end | Result |
|---|---|---|---|---|---|
| V3 | eager | eager | 1470.8 s | 3.2 tok/s | accurate: cream long-haired cat in tortoiseshell sunglasses, turquoise sea, beach; "almost still" clip |
| V4 | fullgraph | eager | 570.1 s | — | harness crash after warmup |
| V5 | fullgraph | fullgraph | 241.0 s | — | harness crash after warmup |

- V3 warmup vs timed: 256/256 tokens identical (eager deterministic in process too).
- V4/V5 crash: the new determinism check read the warmup tokens outside `inference_mode`
  ("Inference tensors do not track version counter") — neuron only; the CPU smoke test passed.
  Fixed. Both compiled rungs got through warmup, so compile itself worked for video.

## Video ladder, attempt 2 (2026-10-22) — V4/V5 (V3 from attempt 1)


| Rung | Text decoder | Vision tower | Warmup | TTFT (vision part) | Decode | End to end |
|---|---|---|---|---|---|---|
| V3 (attempt 1) | eager | eager | 1470.8 s | n/a | n/a | 3.2 tok/s |
| V4 | fullgraph | eager | 628.0 s | 0.768 s (0.543 s) | 15.4 tok/s | 14.8 tok/s |
| V5 | fullgraph | fullgraph | 241.5 s | 0.382 s (0.160 s) | 15.4 tok/s | 15.1 tok/s |

- All three descriptions are accurate and nearly the same (cream long-haired cat, tortoiseshell
  sunglasses reflecting the sky, collar, turquoise sea, beach/cliffs, almost still clip).
- Vision compile cuts video vision time 3.4x (0.543 -> 0.160 s) and warmup 2.6x.
- **Determinism in process:** V4 and V5 warmup vs timed 256/256 identical. So the compiled graphs are
  deterministic at runtime; the run-to-run variation seen on images (attempt 2 vs 3) comes from
  separate compiles (each pod compiles fresh), not from the runtime or TP all-reduce.
- **Vision check (V5):** cosine 0.996045, max abs diff 7.25 — clearly lower than the image check
  (0.999109). On CPU in bf16 the video gap is 0.99998, so the extra gap is device-specific. Answers
  are unaffected here, but this is the first place to look if video quality drops.
- Token match vs V3: V4 133/256 (only text compiled), V5 27/256; V4 vs V5 27/256. Compile-to-compile
  variation (above) means single-run token matches are noisy.

## Vision-path profile, attempt 1 (2026-10-22) — latency + system trace; no device pairs yet

Job `muse-glimmer-30b-video-job.yaml` @ 55271dd: V5 correctness gate, then
`scripts/neuron/profile_vision_job.sh` (vision tower alone, TP=1, one NeuronCore, video 4x20x26).

| Item | Result |
|---|---|
| V5 gate (TP=2) | accurate description, 14.8 tok/s, TTFT 0.676 s (vision 0.454 s), warmup 669.8 s |
| Vision NEFFs in the stage | **1** (9.0 MB) — the whole vision path is one graph |
| Compile (empty cache) | 65.8 s |
| **Clean vision latency, TP=1** | **median 263.13 ms** (min 263.07, max 264.53, n=10) |
| Runtime inspector on | median 263.50 ms (+0.1% overhead); `trace_info.pb` written |
| Vision check, TP=1 device vs stock eager CPU bf16 | cosine 0.994192, max abs diff 14.66 |
| Per-NEFF `capture --single-io` | **0 / 1**: `execution 1 failed - unknown error (1204)` |
| Device `.ntff` from the runtime inspector | none (`DEVICE_PROFILE=1` is sync-only) |

- Vision time inside V5 varies run to run: 0.160 s (video attempt 2) vs 0.454 s here — same code,
  fresh compile per pod. Compile-to-compile variation affects speed, not just tokens.
- Fix for the next run: `NEURON_RT_INSPECT_DEVICE_PROFILE=session` (required for native PyTorch)
  gives a device `.ntff` from the real run with real inputs; it is paired with the stage NEFF by
  content hash. `capture` also gets `--ignore-exec-errors`.

## Vision-path profile, attempt 2 (2026-10-22) — pairs captured, stage time reconciled

Commit 822a759. Artifacts: `s3://621547421844-ap-southeast-4/muse_glimmer/runs/vision_profile_822a75939600_20261022_061513*.tar.gz`.
Vision tower alone, TP=1, one NeuronCore, video grid 4x20x26 (2080 patches, 520 tokens).

**Stage time vs NEFF time.** The vision stage is ONE NEFF (id `63e4a86ff3348c02`, 8 MB).

| Measure | Value |
|---|---|
| Wall time per call, clean (no profiler), median of 10 | 262.95 ms |
| Device time of the one NEFF execution (single-io capture) | 262.05 ms |
| Host overhead per call (wall - device) | ~0.9 ms (0.3%) |
| Session trace: 4 executions (1 warmup + 3 timed) | 1054.0 ms total = 263.5 ms each |

So wall time is the NEFF: the whole optimization target is inside this graph, not launch/host.
Inside V5 (TP=2) the same vision path took 0.159 s of TTFT.

**Inside the NEFF (capture pair, the complete trace).** Class COMPUTE (engine-mix gate): tensor 63%,
vector 73%, DMA 69%, gpsimd 18% busy, no idle stall; 1.61M matmul instructions. MFU ~15% (session
summary-text) — engines are busy but matmuls run far below peak. DMA moved 69 GB in 262 ms, mostly
hardware-dynamic `[128 1]` columns between SBUF and VIRTUAL (25.5 GB + 17.2 GB).
Op attribution is coarse (fused graph): MATMUL/LDWEIGHTS tagged `muse_compile.py:75`
(attention output projection), softmax-like ops (ACTIVATE/TENSOR_TENSOR/TENSOR_REDUCE) at
`muse_compile.py:73` (the masked SDPA) in the session pair and at the vision MLP lines in the capture.
Rotary (`rotate_half` concat/negate + fp32 casts) shows up as small DMA copies.

**Caveats.** The session trace (4.75 GB) is truncated: 2.45M matmuls for 4 executions vs 1.61M for
one in the capture, and only 1.9 GB of DMA. Use it for execution times only; use the capture pair
for instruction stats. Vision check TP=1 device vs stock eager CPU bf16: cosine 0.994192.

**Hypothesis to A/B (not yet measured).** For video every frame is its own attention segment
(4 x 520 tokens), but the fullgraph patch runs ONE 2080x2080 masked SDPA per layer: ~4x the
attention matmuls and softmax work of per-segment attention, and 2080^2 x 16 heads of scores that
likely spill (the VIRTUAL traffic). Equal-size segments can run as a batched SDPA over
(segments, heads, 520, 64) with static shapes — no graph break, no kernel.

**Run timing (83.4 min end to end):** V5 rung 14.2 min; profile warmup 1.5, clean 1.1,
session device profile 7.6, system perfetto 15.1, per-NEFF capture 11.7, upload bundle 4.0,
parquet 18.9, archive 8.4 (4.7 GB session trace).

## vLLM serving, phase 1: text (2026-10-22)

`vllm serve` on trn3 with vLLM Neuron (plugin 0.29.0.1.2.0, vLLM 0.29.0), this repo's out-of-tree
model `vllm_plugin/` (registered via `vllm.general_plugins`). Job `muse-glimmer-30b-vllm-job.yaml`
@ 73ef0cf. TP=2, `m-lnc2`, same node, CPU and memory limits as the bare bring-up.

Model code: same plain-PyTorch math as the HF path validated above (no NKI kernels in the model);
only the KV cache differs (vLLM paged cache via block tables). Memory knobs matched to the bare
bring-up: `--max-num-seqs 1`, `--max-model-len 2048`, `--num-gpu-blocks-override 512`
(4 KV-cache groups x 64 blocks of 32 tokens per sequence, 2x), one prefill bucket (1024) and one
decode bucket (1).

| | Bare compiled (R2) | vLLM |
|---|---|---|
| Prompt | 34 tokens | 34 tokens |
| TTFT | 0.371 s | 0.312 s |
| Decode | 15.5 tok/s | 9.3 tok/s |
| Startup | 640 s warmup | 740 s load + compile + warmup |
| Answer | correct | correct, same explanation (bits as switches, qubits as spinning coins, superposition, entanglement) |

Decode is slower: each step reads the whole cached context through the block table in plain PyTorch.

## vLLM serving, phase 2: images (2026-10-22)

Job `muse-glimmer-30b-vllm-job.yaml` @ 3f1a32c (branch `exp/vllm-phase2`). Vision path = the
validated fullgraph vision graph (R5 / vision profile), replicated on each core, writing into
vLLM's encoder cache; pixels from the transformers processor at 448x448 (256 image tokens).

| Request | Prompt tokens | TTFT | Decode | Result |
|---|---|---|---|---|
| text-1 | 34 | 0.340 s | 9.3 tok/s | correct (same as phase 1) |
| bee.jpg | 289 | 0.783 s | 9.3 tok/s | correct (pink cosmos, bumblebee) |
| count red circles | 299 | 0.430 s | 9.3 tok/s | PASS ("3") |
| read word | 293 | 0.527 s | 9.3 tok/s | PASS ("TRAINIUM") |
| square color | 299 | 0.526 s | 9.3 tok/s | PASS ("green") |
| text-2 (after images) | 34 | 0.414 s | 9.3 tok/s | differs from text-1 at char 141 |

text-2 hit vLLM's prefix cache (hit rate 0% -> 2.6% at that request), a path the bring-up never
had; prefix caching is now off in the job.

## Stacks used: bare TorchNeuron vs vLLM: bare TorchNeuron vs vLLM

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
