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

Demo timing in the deploy file (script defaults in brackets):

| Setting | Deploy | Meaning |
|---|---|---|
| `HEARTBEAT_S` | 4 (60) | idle time without proof before a heartbeat |
| `HEALTH_GRACE_S` | 4 (60) | fail when the last proof is older than `HEARTBEAT_S + HEALTH_GRACE_S` |
| `HEARTBEAT_TIMEOUT_S` | 5 (20) | heartbeat request timeout |
| `FAULT_INJECTION` | 1 (0) | enables the fault file below |

Probes: livenessProbe runs this script every 5 s and restarts on the first failure;
`terminationGracePeriodSeconds: 5`. startupProbe and readinessProbe use `/health`.

Detection after a hang:
1. Idle: the next probe (within 5 s) sends a heartbeat, which times out after 5 s. Proof is
   then older than 8 s, so that probe fails: about 10 s.
2. Busy: token counters stop; the first probe that finds them frozen for over 8 s fails:
   8 to 13 s.

Right after a restart, kubelet can run one liveness probe before the startupProbe takes
over; with `failureThreshold: 1` that killed the new container once (seen 2026-10-24). The
script now passes while the server has never answered in this container.

Then about 5 s to stop the container and about 1.5 min to restart from the node cache.
Trade-off: one slow probe restarts the server. A step that legitimately takes over 8 s
with no token counted would too; raise `HEALTH_GRACE_S` outside demos.

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

The deploy file puts three caches on a hostPath (`/opt/muse-glimmer-30b`, mounted at
`/node-cache`), so they survive restarts:

1. HF weights: `HF_HOME=/node-cache/hf_cache`.
2. NEFFs: `VLLM_CACHE_ROOT=/node-cache/vllm_cache`. On the native route
   (libtorch-neuronx-lite) they land in `neuron/native_neff_cache`, keyed by graph,
   shapes, versions, compiler flags and target.
3. NKI kernel binaries: `NKI_COMPILE_CACHE_URL=/node-cache/nki_cache`. Their index lives
   under `VLLM_CACHE_ROOT`; with the binaries left in the container's `/var/tmp`, a restart
   found the index but not the files and the compile failed (`NCC_EVRF059`). Startup now
   drops index entries whose binary is missing.

The startup log prints `NEFF cache: ... has N NEFFs at start`. The same N after a restart
means every graph came from the cache. (No per-graph cache-hit line is logged at this
log level.)

1. `NEFF_SOURCE=compile` (default): a miss compiles and writes the NEFF to the node.
2. `NEFF_SOURCE=hostpath`: refuses to start when the node has no NEFFs. vllm-neuron
   0.29.0.1.2.0 has no fail-on-cache-miss flag (`VLLM_NEURON_DISABLE_WARMUP_COMPILE` is
   in newer docs, not in this image), so a miss would still compile.

Switch with `kubectl set env deploy/muse-glimmer-30b-vllm NEFF_SOURCE=hostpath`.
Graph tracing still runs at startup (it computes the cache key); only `neuronx-cc` is
skipped. Any change to model code, flags, buckets or image gives new keys. The cache is per
node; the deploy pins `node-type: trn3-dev1`.
