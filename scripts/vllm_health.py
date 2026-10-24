#!/usr/bin/env python3
"""Device-level liveness probe for the vLLM server (kubelet exec probe; exit 0 = healthy).

vLLM's /health only reports that the engine process is up. A rank stuck in a collective
(dead chip) leaves it green. This probe uses proof of life instead, like the Wan2.2
endpoint's heartbeat (docs/RESILIENCE_VLLM.md):

1. Busy: every step runs collectives on both ranks, so token counters that moved since
   the last probe prove both chips are alive.
2. Idle for HEARTBEAT_S without proof: send a 1-token completion (the heartbeat). It runs
   the full model on both ranks, the same collectives as a request.
3. Fail when the last proof is older than HEARTBEAT_S + HEALTH_GRACE_S, or /health fails.

Runs as an exec probe, so the pod opens no new port. State is kept in STATE_FILE between
probes. Standard library only.
"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = os.environ.get("SERVED_MODEL", "meta-models/Muse-Glimmer-30B")
HEARTBEAT_S = int(os.environ.get("HEARTBEAT_S", "60"))
HEALTH_GRACE_S = int(os.environ.get("HEALTH_GRACE_S", "60"))
HEARTBEAT_TIMEOUT_S = int(os.environ.get("HEARTBEAT_TIMEOUT_S", "20"))
STATE_FILE = os.environ.get("HEALTH_STATE_FILE", "/tmp/muse_health.json")


def get(path, timeout=2):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read().decode()


def counters(text):
    """Sum of vLLM token counters and request gauges from /metrics."""
    out = {"tokens": 0.0, "running": 0.0, "waiting": 0.0}
    names = {"vllm:prompt_tokens_total": "tokens", "vllm:generation_tokens_total": "tokens",
             "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting"}
    for line in text.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in names:
            out[names[name]] += float(line.rsplit(" ", 1)[1])
    return out


def heartbeat():
    body = json.dumps({"model": MODEL, "prompt": "Hi", "max_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=HEARTBEAT_TIMEOUT_S) as r:
        return r.status == 200 and json.load(r)["choices"]


def main():
    now = time.time()
    try:
        state = json.load(open(STATE_FILE))
    except (OSError, ValueError):
        state = {"last_ok": now, "tokens": -1.0}  # first probe after startup: warmup just proved it
    try:
        status, _ = get("/health")
        if status != 200:
            print(f"UNHEALTHY: /health {status}")
            return 1
        m = counters(get("/metrics")[1])
    except Exception as e:  # noqa: BLE001 - any failure to reach the server is a failed probe
        if not os.path.exists(STATE_FILE):
            # This container has never served (STATE_FILE is written only after the server
            # answered). The startupProbe owns this phase; a liveness probe that kubelet
            # runs right after a restart must not kill the new container.
            print(f"starting: server not up yet ({e})")
            return 0
        print(f"UNHEALTHY: server unreachable: {e}")
        return 1

    if m["tokens"] != state["tokens"]:
        state["last_ok"], how = now, "tokens moved"
    elif m["running"] == 0 and m["waiting"] == 0 and now - state["last_ok"] >= HEARTBEAT_S:
        try:
            heartbeat()
            state["last_ok"], how = time.time(), "heartbeat ok"
            m = counters(get("/metrics")[1])
        except Exception as e:  # noqa: BLE001
            how = f"heartbeat failed: {e}"
    else:
        how = "no new proof"
    state["tokens"] = m["tokens"]
    json.dump(state, open(STATE_FILE, "w"))

    age = time.time() - state["last_ok"]
    ok = age <= HEARTBEAT_S + HEALTH_GRACE_S
    print(f"{'healthy' if ok else 'UNHEALTHY'}: {how}, last proof {age:.0f} s ago, "
          f"running {m['running']:.0f}, waiting {m['waiting']:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
