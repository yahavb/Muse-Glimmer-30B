# SPDX-License-Identifier: Apache-2.0
"""Device fault injection for testing the health probe (FAULT_INJECTION=1 only).

Same modes as the Wan2.2 endpoint's /debug/fail, but triggered by a file instead of an
HTTP route, so the server gets no new endpoint. Inside the pod:

    echo '{"rank": 1, "mode": "hang"}' > /tmp/muse_fault.json

Every Neuron worker checks the file before each execute_model (one stat per step).
  hang:  that rank stops before its next device work. The other rank blocks in the next
         collective of the NEFF, token counters freeze, and the health probe fails.
         This is the dead-chip case.
  crash: that rank exits at once (os._exit). vLLM's worker monitor sees the death, the
         engine shuts down and /health fails. This is the kill -9 case.
/tmp is in the container's own filesystem, so a container restart clears the fault.
"""
import json
import logging
import os
import time

FAULT_FILE = os.environ.get("FAULT_FILE", "/tmp/muse_fault.json")
MODES = ("hang", "crash")
logger = logging.getLogger(__name__)


def install() -> None:
    """Wrap NeuronWorker.execute_model. Called from the model constructor (workers only)."""
    if os.environ.get("FAULT_INJECTION", "0") != "1":
        return
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    if getattr(NeuronWorker.execute_model, "_muse_fault", False):
        return
    original = NeuronWorker.execute_model

    def execute_model(self, *args, **kwargs):
        if os.path.exists(FAULT_FILE):
            _maybe_fail(self.rank)
        return original(self, *args, **kwargs)

    execute_model._muse_fault = True
    NeuronWorker.execute_model = execute_model
    logger.warning("FAULT_INJECTION=1: workers read %s before each step", FAULT_FILE)


def _maybe_fail(rank: int) -> None:
    try:
        with open(FAULT_FILE) as f:
            req = json.load(f)
    except (OSError, ValueError):
        return
    if req.get("rank") != rank or req.get("mode") not in MODES:
        return
    if req["mode"] == "crash":
        logger.warning("FAULT INJECTION: rank %d exits (simulated crash)", rank)
        os._exit(1)
    logger.warning("FAULT INJECTION: rank %d hangs (simulated dead chip)", rank)
    while True:
        time.sleep(3600)
