# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelRegistry entry for MuseGlimmerForConditionalGeneration (the HF architecture)."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class MuseGlimmerForConditionalGeneration(nn.Module):
    """Factory: validates config, returns the implementation (phase 1: text only).

    vLLM inspects the registered class when it builds ModelConfig (in the API process, before
    any Neuron worker exists). Its VllmModelForTextGeneration protocol (vLLM 0.29
    interfaces_base.py) requires: __init__ accepting `vllm_config`, `embed_input_ids`,
    `forward(input_ids, positions, ...)` and `compute_logits`; without them `--runner generate`
    is rejected. vllm-neuron's own models skip this because they are registered later, in the
    worker, so vLLM inspects its built-in GPU class instead. The Neuron runner never
    instantiates this factory — it calls `from_configs` — so the protocol methods only exist
    to satisfy the check.
    """

    is_text_generation_model = True

    def __init__(self, hf_config: PretrainedConfig | None = None, neuron_config: NeuronConfig | None = None,
                 *, vllm_config=None, prefix: str = "") -> None:
        super().__init__()
        if hf_config is None and vllm_config is not None:
            hf_config = vllm_config.model_config.hf_config
        self._model = self.from_configs(hf_config, neuron_config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._model.model.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        return self._model(input_ids=input_ids, positions=positions, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("vllm-neuron runs the LM head inside forward()")

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None = None) -> nn.Module:
        quantization = neuron_config.quantization if neuron_config else None
        if quantization not in (None, "bf16"):
            raise ValueError(f"quantization={quantization!r} is not supported; Muse-Glimmer runs BF16")
        from .model import MuseGlimmerForCausalLM

        return MuseGlimmerForCausalLM.from_configs(hf_config, neuron_config)
