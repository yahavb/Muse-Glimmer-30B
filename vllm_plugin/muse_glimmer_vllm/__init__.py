# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer for vLLM Neuron, as an out-of-tree vLLM plugin.

vLLM loads `vllm.general_plugins` entry points in every process (API server, engine,
workers), so `vllm serve meta-models/Muse-Glimmer-30B` finds the model without changing
vllm-neuron. The class is registered lazily by import path.
"""

ARCH = "MuseGlimmerForConditionalGeneration"


def register() -> None:
    from vllm import ModelRegistry

    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "muse_glimmer_vllm.factory:MuseGlimmerForConditionalGeneration")
