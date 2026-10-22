# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for MuseGlimmerForConditionalGeneration (the HF architecture)."""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class MuseGlimmerForConditionalGeneration(nn.Module):
    """Factory: validates config, returns the implementation (phase 1: text only)."""

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> None:
        super().__init__()
        self._model = self.from_configs(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None) -> nn.Module:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(f"quantization={quantization!r} is not supported; Muse-Glimmer runs BF16")
        from .model import MuseGlimmerForCausalLM

        return MuseGlimmerForCausalLM.from_configs(hf_config, neuron_config)
