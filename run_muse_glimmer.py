"""Run meta-models/Muse-Glimmer-30B on Trainium with TorchNeuron (native PyTorch).

Single device:  python run_muse_glimmer.py
Compiled:       python run_muse_glimmer.py --compile   (text only, fullgraph=True,
                StaticCache; one prefill graph + one decode graph)
Image prompt:   add --image URL  (resized to --image-size, fixed patch grid)
                with --compile: --vision eager (stock tower) | compile (fullgraph=True)
                --cases eval_images/cases.json adds known-answer image questions (PASS/FAIL)
Video prompt:   --video PATH  (--video-frames sampled evenly; fixed count -> fixed grid)
Tensor-parallel across NeuronCores (needed when one core can't hold ~60GB):
                torchrun --nproc_per_node 2 run_muse_glimmer.py --tp
                (max TP=2: the model has only 2 KV heads)
"""
import argparse
import time

import torch
import torch_neuronx  # noqa: F401  registers the 'neuron' device
from transformers import AutoModelForImageTextToText, AutoProcessor

import muse_compile as mc
import muse_inputs as mi

# MUSE_MODEL_ID: local override for CPU smoke tests (scripts/smoke_run.py)
import os
MODEL_ID = os.environ.get("MUSE_MODEL_ID", "meta-models/Muse-Glimmer-30B")

p = argparse.ArgumentParser()
p.add_argument("--prompt", default=None,
               help="default: quantum-computing question; 'Describe this image.' with --image")
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--compile", action="store_true", help="use torch.compile(backend='neuron')")
p.add_argument("--tp", action="store_true", help="tensor parallel via torchrun")
p.add_argument("--image", default=None, help="image URL or path; adds the vision encoder")
p.add_argument("--image-size", type=int, default=448, help="square resize; 448 -> 32x32 patches, 256 tokens")
p.add_argument("--vision", choices=["eager", "compile"], default="eager",
               help="with --compile and --image: vision tower eager or fullgraph=True")
p.add_argument("--cases", default=None, help="known-answer image cases JSON (image mode only)")
p.add_argument("--video", default=None, help="video file path; same vision tower, frames in pairs")
p.add_argument("--video-frames", type=int, default=8, help="frames sampled evenly (even: 2 per patch)")
p.add_argument("--eval-max-new-tokens", type=int, default=256,
               help="per case; reasoning comes first, 128 truncated some answers")
args = p.parse_args()
if args.image and args.video:
    p.error("use --image or --video, not both")
if args.prompt is None:
    args.prompt = ("Describe this image." if args.image else "Describe this video." if args.video
                   else "Explain quantum computing in simple terms.")

processor = AutoProcessor.from_pretrained(MODEL_ID)
load_kwargs = dict(dtype=torch.bfloat16)
if args.tp:
    import os
    from transformers.distributed.utils import get_distributed_backend
    # Init the process group ourselves WITHOUT device_id. transformers passes
    # device_id, which makes DeviceMesh use split_group(), and the neuron backend
    # doesn't support splitting. Unbound -> DeviceMesh falls back to new_group().
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend=get_distributed_backend("neuron"))
    load_kwargs["tp_plan"] = "auto"
else:
    load_kwargs["device_map"] = "neuron"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kwargs).eval()

def load_image(src):
    return mi.load_image(src, args.image_size)


def load_video(path):
    return mi.load_video(path, args.video_frames)


def build_inputs(prompt, image=None, video=None):
    return mi.build_inputs(processor, prompt, model.device, image=image, video=video)


def mm_parts(inp):
    """(pixel_values, grid_thw, placeholder_token_id, modality) of an image/video prompt, else None."""
    if "pixel_values" in inp:
        return inp["pixel_values"], inp["image_grid_thw"], model.config.image_token_id, "image"
    if "pixel_values_videos" in inp:
        return inp["pixel_values_videos"], inp["video_grid_thw"], model.config.video_token_id, "video"
    return None


inputs = build_inputs(args.prompt, load_image(args.image) if args.image else None,
                      load_video(args.video) if args.video else None)
mm = mm_parts(inputs)
cases = []
if args.cases and args.image:
    import json
    import os

    case_dir = os.path.dirname(os.path.abspath(args.cases))
    for c in json.load(open(args.cases)):
        c["inputs"] = build_inputs(c["prompt"], load_image(os.path.join(case_dir, c["image"])))
        cases.append(c)


def report_case(c, new_tokens):
    """Print one known-answer case: PASS if any expected keyword is in the final answer.

    Cut at the first end-of-turn token: the compiled loop always runs the full budget
    (eager generate stops there by itself).
    """
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    toks = new_tokens.tolist()
    cut = next((i for i, t in enumerate(toks) if t in eos), len(toks))
    answer = mc.final_answer(processor.decode(toks[:cut], skip_special_tokens=True))
    ok = any(k.lower() in answer.lower() for k in c["expect"])
    print(f"CASE {c['id']}: {'PASS' if ok else 'FAIL'} | expect {c['expect']} | answer: {answer[:200]!r}")
    return ok


is_rank0 = not args.tp or torch.distributed.get_rank() == 0
if is_rank0:
    mode = "eager" if not args.compile else ("compile text" + (f", vision {args.vision}" if mm else ""))
    print(f"RUN CONFIG: mode={mode} image={args.image} video={args.video} prompt_tokens={inputs['input_ids'].shape[1]}"
          + (f" {mm[3]}_grid_thw={mm[1].tolist()} {mm[3]}_tokens={int((inputs['input_ids'] == mm[2]).sum())}"
             if mm else ""))
# Greedy: keeps TP ranks picking the same token (independent sampling could diverge).
gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)


if args.compile:
    # Text only: fullgraph=True over a StaticCache (TorchNeuron has no dynamic shapes),
    # so one prefill graph + one decode graph, reused for every token.
    torch.compiler.set_default_backend("neuron")
    step = mc.make_step(model)
    ids = inputs["input_ids"]
    tp_size = torch.distributed.get_world_size() if args.tp else 1
    # One cache shape for every prompt -> the decode graph is shared by all cases.
    max_prompt = max([ids.shape[1]] + [c["inputs"]["input_ids"].shape[1] for c in cases])
    max_new = max([args.max_new_tokens] + ([args.eval_max_new_tokens] if cases else []))
    cache = mc.make_static_cache(model.config.text_config, max_prompt + max_new,
                                 model.dtype, model.device, tp_size=tp_size)
    prefill = None
    vision_times = []
    ref_feats = None
    if mm:
        mm_prefill = mc.make_mm_prefill(model)

        def make_prefill(inp):
            """Image/video prompt: vision graph (eager or fullgraph) feeds the compiled mm prefill graph."""
            pv, grid, token_id, _ = mm_parts(inp)
            text_ids, img_pos = mc.split_image_tokens(inp["input_ids"].cpu(), token_id)
            text_ids, img_pos = text_ids.to(model.device), img_pos.to(model.device)
            vision = mc.make_vision(model, pv, grid, compile=args.vision == "compile")

            def prefill(c, pos):
                tv = time.time()
                feats = vision()
                torch.neuron.synchronize()
                vision_times.append(time.time() - tv)
                return mm_prefill(text_ids, img_pos, feats, c, pos)
            return prefill

        if args.vision == "compile":
            # Reference for the vision check: stock eager tower, BEFORE the attention patch.
            with torch.inference_mode():
                ref_feats = mc.make_vision(model, mm[0], mm[1])().cpu()
        prefill = make_prefill(inputs)
    with torch.inference_mode():
        t0 = time.time()
        warm, _, _ = mc.greedy_generate(step, ids, cache, args.max_new_tokens,
                                        sync=torch.neuron.synchronize, prefill=prefill)
        if is_rank0:
            print(f"warmup (compile) took {time.time() - t0:.1f}s")
        out, ttft, decode_s = mc.greedy_generate(step, ids, cache, args.max_new_tokens,
                                                 sync=torch.neuron.synchronize, prefill=prefill)
        new_tokens = out[0].cpu()
        # read inside inference_mode: on neuron, touching these outside it raises
        warm_tokens = warm[0].cpu().tolist()
    if is_rank0:
        # same process, same compiled graphs: a mismatch is runtime nondeterminism
        same = mc.matching_prefix(warm_tokens, new_tokens.tolist())
        print(f"DETERMINISM (warmup vs timed, same graphs): first {same}/{len(new_tokens)} tokens identical")
    if is_rank0:
        n = len(new_tokens)
        print(processor.decode(new_tokens, skip_special_tokens=True))
        print(f"\nprefill {inputs['input_ids'].shape[1]} tokens: TTFT {ttft:.3f}s")
        if vision_times:
            print(f"vision tower ({args.vision}): {vision_times[-1]:.3f}s of TTFT")
        print(f"decode {n - 1} tokens in {decode_s:.1f}s ({(n - 1) / decode_s:.1f} tok/s)")
        print(f"{n} tokens in {ttft + decode_s:.1f}s ({n / (ttft + decode_s):.1f} tok/s)")
    if mm:
        with torch.inference_mode():
            if ref_feats is not None:
                feats = mc.make_vision(model, mm[0], mm[1], compile=True)().cpu()
                diff, cos = mc.feature_diff(ref_feats, feats)
                if is_rank0:
                    print(f"VISION CHECK (fullgraph vs stock eager, same {mm[3]}): max_abs_diff={diff:.3e} cosine={cos:.6f}")
            if is_rank0:
                print(f"MAIN_TOKENS: {new_tokens.tolist()}")
            passed = 0
            for c in cases:
                out, _, _ = mc.greedy_generate(step, c["inputs"]["input_ids"], cache, args.eval_max_new_tokens,
                                               sync=torch.neuron.synchronize, prefill=make_prefill(c["inputs"]))
                if is_rank0:
                    passed += report_case(c, out[0].cpu())
            if is_rank0 and cases:
                print(f"CASES: {passed}/{len(cases)} passed")
    raise SystemExit(0)

def eager_generate(inp, **kw):
    """model.generate; for image/video prompts the stock vision tower runs first (eager, grid
    metadata on host, see mc.make_vision) and its features go in as mm_encoder_outputs."""
    parts = mm_parts(inp)
    if parts is None:
        return model.generate(**inp, **kw)
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    pv, grid, _, modality = parts
    feats = mc.make_vision(model, pv, grid)()
    return model.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                          mm_encoder_outputs={modality: BaseModelOutputWithPooling(pooler_output=(feats,))}, **kw)


with torch.inference_mode():
    # Warmup compiles every op/graph shape; not timed.
    t0 = time.time()
    warm = eager_generate(inputs, **gen_kwargs)
    torch.neuron.synchronize()
    if is_rank0:
        print(f"warmup (compile) took {time.time() - t0:.1f}s")

    t0 = time.time()
    out = eager_generate(inputs, **gen_kwargs)
    torch.neuron.synchronize()
    dt = time.time() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:].cpu()
    if is_rank0:
        same = mc.matching_prefix(warm[0, inputs["input_ids"].shape[1]:].cpu().tolist(), new_tokens.tolist())
        print(f"DETERMINISM (warmup vs timed, same process): first {same}/{len(new_tokens)} tokens identical")

if is_rank0:
    print(processor.decode(new_tokens, skip_special_tokens=True))
    print(f"\n{len(new_tokens)} tokens in {dt:.1f}s ({len(new_tokens)/dt:.1f} tok/s)")

if mm:
    if is_rank0:
        print(f"MAIN_TOKENS: {new_tokens.tolist()}")
    passed = 0
    with torch.inference_mode():
        for c in cases:
            inp = c["inputs"]
            out = eager_generate(inp, max_new_tokens=args.eval_max_new_tokens, do_sample=False)
            if is_rank0:
                passed += report_case(c, out[0, inp["input_ids"].shape[1]:].cpu())
    if is_rank0 and cases:
        print(f"CASES: {passed}/{len(cases)} passed")
