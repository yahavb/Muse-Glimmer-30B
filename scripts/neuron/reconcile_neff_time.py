"""Reconcile measured wall time per call with device time per NEFF execution.

Input: the parquet dump of a device-profile (session) pair, and the log of the run that
produced it (profile_vision.py prints "built=<s>" for the first call and "iter <i> <ms>" per
timed call). The trace's ExecutionInfo table has one row per NEFF execution with start/end
timestamps (ns). The last N executions are the N timed calls; earlier ones are warmup.

For each timed call:  wall_ms = device_ms + host_ms   (host = launch, input prep, sync)
Also prints the gap between consecutive executions (idle device between calls).

Column names are discovered from the parquet header and printed; nothing is assumed silently.

Usage: python3 reconcile_neff_time.py <parquet-dir> <run-log>
"""
import os
import re
import sys

import duckdb


def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    for c in cands:
        for lc, orig in low.items():
            if c in lc:
                return orig
    return None


def main():
    pq, log = sys.argv[1], sys.argv[2]
    f = os.path.join(pq, "ExecutionInfo.parquet")
    if not os.path.exists(f):
        print(f"no ExecutionInfo.parquet in {pq}; tables: {sorted(os.listdir(pq))}")
        return 1
    con = duckdb.connect()
    cols = [d[0] for d in con.execute(f"SELECT * FROM read_parquet('{f}') LIMIT 0").description]
    c_idx = pick(cols, "execution_index")
    c_neff = pick(cols, "neff_name")
    c_start = pick(cols, "execution_start_ts", "start_ts")
    c_end = pick(cols, "execution_end_ts", "end_ts")
    print(f"ExecutionInfo columns: {cols}")
    print(f"  using index={c_idx} neff={c_neff} start={c_start} end={c_end} (ns)")
    if not (c_start and c_end):
        print("  start/end columns not found — cannot reconcile")
        return 1
    order = c_start
    rows = con.execute(
        f"SELECT {c_idx or 'NULL'}, {c_neff or 'NULL'}, {c_start}, {c_end} "
        f"FROM read_parquet('{f}') ORDER BY {order}").fetchall()
    if not rows:
        print("  ExecutionInfo is empty")
        return 1

    text = open(log).read()
    iters = [float(x) for x in re.findall(r"^iter \d+ ([0-9.]+)ms$", text, re.M)]
    built = re.findall(r"^built=([0-9.]+)s$", text, re.M)

    print(f"\nNEFF executions in trace: {len(rows)}   timed calls in log: {len(iters)}"
          f"   first-call (warmup) wall: {built[0] + ' s' if built else 'n/a'}")
    print(f"  {'#':>3} {'neff':<28} {'start_ms':>10} {'device_ms':>10} {'gap_before_ms':>14}")
    prev_end = None
    dev = []
    for k, (idx, neff, s, e) in enumerate(rows):
        d = (e - s) / 1e6
        gap = (s - prev_end) / 1e6 if prev_end is not None else float("nan")
        dev.append(d)
        print(f"  {k:>3} {str(neff)[:28]:<28} {(s - rows[0][2]) / 1e6:>10.2f} {d:>10.2f} {gap:>14.2f}")
        prev_end = e

    n = min(len(iters), len(rows))
    if n == 0:
        print("\nno timed calls to reconcile")
        return 0
    print(f"\nRECONCILE (last {n} executions = the {n} timed calls)")
    print(f"  {'call':>4} {'wall_ms':>9} {'device_ms':>10} {'host_ms':>9} {'device_share':>13}")
    for j in range(n):
        w, d = iters[len(iters) - n + j], dev[len(dev) - n + j]
        print(f"  {j:>4} {w:>9.2f} {d:>10.2f} {w - d:>9.2f} {100 * d / w:>12.1f}%")
    w = sum(iters[-n:]) / n
    d = sum(dev[-n:]) / n
    print(f"  mean {w:>9.2f} {d:>10.2f} {w - d:>9.2f} {100 * d / w:>12.1f}%")
    print("  (the profiled call carries profiler overhead; compare its device_ms with the clean median)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
