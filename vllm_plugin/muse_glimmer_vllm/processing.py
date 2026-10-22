# SPDX-License-Identifier: Apache-2.0
"""Multimodal prompt processing for Muse-Glimmer on vLLM Neuron.

Pixels come from the transformers MuseGlimmerProcessor — the one the bring-up validated
(R3-R5) — not vLLM's own processor implementation. Every image is resized to
IMAGE_SIZE x IMAGE_SIZE first, as validated, so the vision graph has one shape.

vLLM's MuseGlimmerMultiModalProcessor is kept for the prompt side: it counts `<|patch|>`
placeholders, expands each to the block the HF processor returns and records the
embedding count. The transformers block for one 448x448 image is
`<|image_start|> <|patch|> x 256 <|image_end|>` with pixel_values [1024, 1176] — the same
tokens vLLM's prompt-update replacement produces.

Video is not enabled yet (limit 0).
"""

from functools import cached_property

import torch
from PIL import Image
from transformers import AutoProcessor

from vllm.model_executor.models import muse_glimmer as vmg
from vllm.multimodal.parse import ImageSize

from .vision import IMAGE_GRID, IMAGE_SIZE

IMAGE_TOKENS = IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2] // 4  # 2x2 merge -> 256


class _TransformersImageProcessor:
    """Adapter: vLLM's processor calls processor(text="<|image|>", images=[img], return_tensors=None)
    and reads output["input_ids"][0] (the expanded block) and output["pixel_values"] (one tensor
    per image). This returns those from the transformers processor."""

    def __init__(self, hf):
        self._hf = hf

    def __getattr__(self, name):
        return getattr(self._hf, name)

    def __call__(self, text=None, images=None, videos=None, return_tensors=None, **kwargs):
        if videos is not None:
            raise NotImplementedError("Muse-Glimmer video on vLLM Neuron is not enabled yet")
        if not images:
            return self._hf(text=text, return_tensors=return_tensors, **kwargs)
        (img,) = images
        img = (img if isinstance(img, Image.Image) else Image.fromarray(img)).convert("RGB")
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        out = self._hf(text=self._hf.image_token, images=[img], add_special_tokens=False, return_tensors="pt")
        return {"input_ids": [out["input_ids"][0].tolist()], "pixel_values": [out["pixel_values"]]}


class MuseGlimmerNeuronProcessingInfo(vmg.MuseGlimmerProcessingInfo):
    @cached_property
    def _transformers_processor(self):
        return AutoProcessor.from_pretrained(self.ctx.model_config.model)

    def get_hf_processor(self, **kwargs):
        return _TransformersImageProcessor(self._transformers_processor)

    def get_supported_mm_limits(self):
        return {"image": None, "video": 0}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": IMAGE_TOKENS, "video": 0}

    def get_image_size_with_most_features(self):
        return ImageSize(width=IMAGE_SIZE, height=IMAGE_SIZE)


class MuseGlimmerNeuronDummyInputs(vmg.MuseGlimmerDummyInputsBuilder):
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        return {"image": self._get_dummy_images(
            width=IMAGE_SIZE, height=IMAGE_SIZE, num_images=mm_counts.get("image", 0),
            overrides=mm_options.get("image"))}


MuseGlimmerNeuronProcessor = vmg.MuseGlimmerMultiModalProcessor
