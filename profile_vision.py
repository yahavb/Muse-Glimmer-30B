"""Vision tower only, on ONE NeuronCore (TP=1): the unit the vision-path profiles target.

Loads the full checkpoint on CPU, moves only the vision path (tower + adapter +
projection + norm, ~1.8B params / ~3.6GB bf16) to the device. No collectives, so
`neuron-explorer capture --single-io` works on every NEFF, and the process compiles
nothing but vision graphs, so its NEFF cache IS the vision path.

Prints the markers the 3-run harness parses:
  built=<s>       first call (compile or NEFF-cache load), not timed
  iter <i> <ms>   each timed call
  median=<ms>     median of --iters timed calls (the clean latency when no profiler is on)
  VISION CHECK    compiled device features vs stock eager on CPU (bf16), same frames

    python profile_vision.py --video media/cat_sunglasses.mp4 --vision compile --warmup 1 --iters 10
"""
import argparse
import os
import statistics
import time

import torch
import torch_neuronx  # noqa: F401  registers the 'neuron' device
from transformers import AutoModelForImageTextToText, AutoProcessor

import muse_compile as mc
import muse_inputs as mi

MODEL_ID = os.environ.get("MUSE_MODEL_ID", "meta-models/Muse-Glimmer-30B")

p = argparse.ArgumentParser()
p.add_argument("--video", default=None)
p.add_argument("--video-frames", type=int, default=8)
p.add_argument("--image", default=None)
p.add_argument("--image-size", type=int, default=448)
p.add_argument("--vision", choices=["eager", "compile"], default="compile")
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--iters", type=int, default=10)
p.add_argument("--check", action="store_true", help="compare against stock eager on CPU (slow, once)")
args = p.parse_args()
if bool(args.video) == bool(args.image):
    p.error("give exactly one of --video / --image")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()  # CPU
m = model.model
dev = torch.device(os.environ.get("MUSE_DEVICE", "neuron"))  # "cpu" only for smoke tests

prompt = "Describe this video." if args.video else "Describe this image."
inputs = mi.build_inputs(processor, prompt, "cpu",
                         image=mi.load_image(args.image, args.image_size) if args.image else None,
                         video=mi.load_video(args.video, args.video_frames) if args.video else None)
if args.video:
    pv, grid, modality = inputs["pixel_values_videos"], inputs["video_grid_thw"], "video"
else:
    pv, grid, modality = inputs["pixel_values"], inputs["image_grid_thw"], "image"
print(f"VISION CONFIG: {modality} grid_thw={grid.tolist()} patches={pv.shape[0]} "
      f"tokens={pv.shape[0] // model.config.vision_config.merge_size ** 2} mode={args.vision} TP=1")

ref = None
if args.check:
    # stock vision path, eager, CPU, bf16 -- before any attention patch
    with torch.inference_mode():
        t0 = time.time()
        ref = mc.make_vision(model, pv, grid)().float()
        print(f"reference (stock eager, CPU bf16) took {time.time() - t0:.1f}s")

for mod in (m.vision_tower, m.vision_adapter, m.vision_projection, m.perception_emb_norm):
    mod.to(dev)
torch.compiler.set_default_backend("neuron")
vision = mc.make_vision(model, pv.to(dev), grid.to(dev), compile=args.vision == "compile")

with torch.inference_mode():
    out = None
    for i in range(args.warmup):
        t0 = time.time()
        out = vision()
        torch.neuron.synchronize()
        if i == 0:
            print(f"built={time.time() - t0:.1f}s")
    times = []
    for i in range(args.iters):
        t0 = time.time()
        out = vision()
        torch.neuron.synchronize()
        times.append((time.time() - t0) * 1000)
        print(f"iter {i} {times[-1]:.2f}ms")
    if times:
        print(f"median={statistics.median(times):.2f}ms min={min(times):.2f}ms max={max(times):.2f}ms n={len(times)}")
    if ref is not None and out is not None:
        diff, cos = mc.feature_diff(ref, out.cpu())
        print(f"VISION CHECK (device {args.vision} vs stock eager CPU bf16): max_abs_diff={diff:.3e} cosine={cos:.6f}")
