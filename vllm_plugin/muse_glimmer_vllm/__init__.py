# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer for vLLM Neuron, as an out-of-tree vLLM plugin.

vLLM loads `vllm.general_plugins` entry points in every process (API server, engine,
workers), so `vllm serve meta-models/Muse-Glimmer-30B` finds the model without changing
vllm-neuron.

vLLM itself ships a (GPU) MuseGlimmerForConditionalGeneration, so this registration must
OVERRIDE it — the same thing vllm-neuron does for its own models — or vLLM builds its GPU
class and the Neuron runner cannot construct it.
"""
import logging

ARCH = "MuseGlimmerForConditionalGeneration"
IMPL = "muse_glimmer_vllm.factory:MuseGlimmerForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return ARCH not in record.getMessage()


def register() -> None:
    from vllm import ModelRegistry

    log = logging.getLogger("vllm.model_executor.models.registry")
    f = _DropOverwriteWarning()
    log.addFilter(f)
    try:
        ModelRegistry.register_model(ARCH, IMPL)  # intentional override of vLLM's GPU class
    finally:
        log.removeFilter(f)
