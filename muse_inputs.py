"""Prompt inputs for Muse-Glimmer: image / video loading and chat-template tokenization.

Shared by run_muse_glimmer.py (full model) and profile_vision.py (vision tower only),
so both see byte-identical pixel values and the same patch grid.
"""
import hashlib
import io
import urllib.request

import numpy as np
from PIL import Image


def load_image(src, size):
    """Square-resized RGB image. Fixed size -> fixed patch grid -> one static vision graph."""
    data = urllib.request.urlopen(src).read() if src.startswith("http") else open(src, "rb").read()
    print(f"IMAGE {src} sha256={hashlib.sha256(data).hexdigest()}")
    return Image.open(io.BytesIO(data)).convert("RGB").resize((size, size))


def load_video(path, n_frames):
    """n_frames sampled evenly, plus metadata so the prompt gets the real timestamps."""
    import imageio_ffmpeg
    from transformers.video_utils import VideoMetadata

    print(f"VIDEO {path} sha256={hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
    reader = imageio_ffmpeg.read_frames(path)
    meta = next(reader)
    w, h = meta["size"]
    frames = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int).tolist()
    print(f"VIDEO {w}x{h} {len(frames)} frames @ {meta['fps']} fps, sampled {idx}")
    md = VideoMetadata(total_num_frames=len(frames), fps=meta["fps"], frames_indices=idx, width=w, height=h)
    return np.stack([frames[i] for i in idx]), md


def build_inputs(processor, prompt, device, image=None, video=None):
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content.insert(0, {"type": "image", "image": image})
    if video is not None:
        content.insert(0, {"type": "video", "video": "frames"})  # template only reads the type
    messages = [
        {"role": "system", "content": "Reasoning strength: low"},
        {"role": "user", "content": content},
    ]
    if video is None:
        return processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(device)
    # Video: pre-sampled frames + metadata. The template text already has BOS.
    frames, md = video
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return processor(text=text, videos=[frames], video_metadata=[md], do_sample_frames=False,
                     add_special_tokens=False, return_tensors="pt").to(device)
