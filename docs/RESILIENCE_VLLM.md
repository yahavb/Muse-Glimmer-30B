# Resilience: device health, fault injection and the NEFF cache (vLLM deployment)

Applies to `muse-glimmer-30b-vllm-deploy.yaml`: 2 ranks (TP=2) on 2 trn3 devices (`m-lnc2`).
Same design as the Wan2.2-I2V-A14B endpoint (`wan2-i2v-14b/docs/resilience.md`), adapted to
`vllm serve`, where we do not own the HTTP server or the main loop.

## What a dead chip looks like

Both ranks take part in every collective inside the NEFFs. When one rank stops, the other
blocks in the next collective. The symptom is a hang, not an error. vLLM's `/health` only
reports that the engine process is up, so it stays green.

## Proof of life (`scripts/vllm_health.py`, the liveness probe)

1. Busy: vLLM's token counters (`/metrics`) moved since the last probe. Every step ran
   collectives on both ranks, so both chips are alive.
2. Idle for `HEARTBEAT_S` with no proof: the probe sends a 1-token completion (the
   heartbeat). It runs the whole model on both ranks.
3. The probe fails when the last proof is older than `HEARTBEAT_S + HEALTH_GRACE_S`, or
   when `/health` fails (engine dead).

A busy server never gets a heartbeat; its requests are the proof. The heartbeat only runs
when nothing is running or waiting. It is an exec probe, so the pod opens no new port.

| Setting | Default | Meaning |
|---|---|---|
| `HEARTBEAT_S` | 60 | idle time without proof before a heartbeat |
| `HEALTH_GRACE_S` | 60 | extra time before the probe fails |
| `HEARTBEAT_TIMEOUT_S` | 20 | heartbeat request timeout (probe timeout is 30 s) |
| `FAULT_INJECTION` | 1 in the deploy file | enables the fault file below |

Probes: startupProbe `/health` (up to 60 min), readinessProbe `/health`, livenessProbe this
script every 30 s, restart after 3 failures. Expected time from a dead chip to a restart:
up to about 2 min for the proof to go stale, then about 1.5 min of failures.

## Simulating a device failure

With `FAULT_INJECTION=1`, write the fault file inside the pod:

```bash
kubectl exec deploy/muse-glimmer-30b-vllm -- sh -c 'echo "{\"rank\": 1, \"mode\": \"hang\"}" > /tmp/muse_fault.json'
```

1. `hang`: that rank stops before its next step (a request or a heartbeat). The other rank
   blocks in the collective, the counters freeze, and the probe fails. The dead-chip case.
2. `crash`: that rank exits at once. vLLM sees the worker die, the engine stops and
   `/health` fails. The `kill -9` case.

Each worker checks the file before every step (`vllm_plugin/muse_glimmer_vllm/faults.py`).
A container restart clears it (`/tmp` is the container's own filesystem). Watch with
`kubectl logs -f deploy/muse-glimmer-30b-vllm` and `kubectl get pod -w` (RESTARTS goes up).

Both are software fakes. Neuron Observer would not see them; `hang` matches its
`unreachable` fault by symptom. A `wrong` mode (a chip that computes badly) is not done
yet: it needs a known-answer heartbeat, and repeated requests are not yet bit-identical.

## NEFF cache on the node (`NEFF_SOURCE`)

vllm-neuron keeps compiled graphs in `$VLLM_CACHE_ROOT/neuron/compile_cache`, keyed by the
graph, input shapes, library and compiler versions, compiler flags and target. The deploy
file sets `VLLM_CACHE_ROOT=/node-cache/vllm_cache` on a hostPath (`/opt/muse-glimmer-30b`),
next to the HF weights (`/node-cache/hf_cache`), so both survive restarts.

1. `NEFF_SOURCE=compile` (default): a cache miss compiles and writes the NEFF to the host
   path. The first start compiles; later starts hit the cache.
2. `NEFF_SOURCE=hostpath`: load only. Sets `VLLM_NEURON_DISABLE_WARMUP_COMPILE=1`, so a
   miss stops startup with an error instead of compiling. Startup also stops if the cache
   directory is empty.

Switch with `kubectl set env deploy/muse-glimmer-30b-vllm NEFF_SOURCE=hostpath`.
Graph tracing still runs at startup (it computes the cache key); only `neuronx-cc` is skipped.
Any change to the model code, flags, buckets or image gives new keys, so `hostpath` then
fails until one `compile` start refills the cache. The cache is per node: the pod must
land on the node that compiled (the deploy pins `node-type: trn3-dev1`).
