# SPDX-License-Identifier: Apache-2.0
"""Muse-Glimmer for vLLM Neuron, as an out-of-tree vLLM plugin.

vLLM loads `vllm.general_plugins` entry points in every process (API server, engine,
workers), so `vllm serve meta-models/Muse-Glimmer-30B` finds the model without changing
vllm-neuron files.

Two registrations:
  1. vLLM ModelRegistry: OVERRIDES vLLM's own (GPU) MuseGlimmerForConditionalGeneration,
     the same thing vllm-neuron does for its own models.
  2. vllm-neuron's model list (vllm_neuron.model.registry.get_models): the runner sizes the
     encoder cache by looking the class up there (mm_utils); an unlisted model gets 0.
"""
import logging

ARCH = "MuseGlimmerForConditionalGeneration"
IMPL = "muse_glimmer_vllm.factory:MuseGlimmerForConditionalGeneration"


class _DropOverwriteWarning(logging.Filter):
    def filter(self, record):
        return ARCH not in record.getMessage()


def _extend_neuron_model_list() -> None:
    from vllm_neuron.model import registry

    if getattr(registry.get_models, "_muse_glimmer", False):
        return
    original = registry.get_models

    def get_models():
        from .factory import MuseGlimmerForConditionalGeneration

        return [m for m in original() if m[0] != ARCH] + [(ARCH, MuseGlimmerForConditionalGeneration)]

    get_models._muse_glimmer = True
    registry.get_models = get_models


def register() -> None:
    from vllm import ModelRegistry

    log = logging.getLogger("vllm.model_executor.models.registry")
    f = _DropOverwriteWarning()
    log.addFilter(f)
    try:
        ModelRegistry.register_model(ARCH, IMPL)  # intentional override of vLLM's GPU class
    finally:
        log.removeFilter(f)
    _extend_neuron_model_list()
