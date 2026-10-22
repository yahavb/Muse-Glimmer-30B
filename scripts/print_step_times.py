"""Print the job's step-timing table: seconds, minutes, share of the end-to-end wall time.

Input lines are "<seconds>\t<step>" in run order; "profile ..." lines are sub-steps of the
profile, which runs after the rungs. Usage: python3 print_step_times.py <tsv> <e2e_seconds>
"""
import sys

rows = []
for line in open(sys.argv[1]):
    sec, _, name = line.rstrip("\n").partition("\t")
    if name:
        rows.append((int(sec), name))
e2e = int(sys.argv[2]) or 1
w = max([len(n) for _, n in rows] + [10])
print(f"  {'step':<{w}} {'seconds':>8} {'minutes':>8} {'share':>6}")
for sec, name in rows:
    print(f"  {name:<{w}} {sec:>8} {sec / 60:>8.1f} {100 * sec / e2e:>5.1f}%")
covered = sum(s for s, _ in rows)
print(f"  {'(other / untimed)':<{w}} {e2e - covered:>8} {(e2e - covered) / 60:>8.1f} {100 * (e2e - covered) / e2e:>5.1f}%")
print(f"  {'END TO END':<{w}} {e2e:>8} {e2e / 60:>8.1f} {100.0:>5.1f}%")
