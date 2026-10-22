#!/bin/bash
# Vision-path profile for Muse-Glimmer-30B: 3-run methodology on ONE NeuronCore (TP=1).
#
#   [1/3] warmup: compile the vision graph into an EMPTY stage cache (+ correctness check)
#   [2/3] clean run, no profiler: median of $ITERS calls = the real vision latency
#   [3a]  runtime inspector (system + device profile) -> trace_info.pb bundle
#   [4]   per-NEFF: capture --single-io + summary-json for every NEFF of the stage,
#         ranking, top pairs preserved as pairs/rankNN_<id>_<MB>MB/{<id>.neff,<id>.ntff,summary.json}
#   [5]   instruction level: parquet per pair -> pq_dma_report / pq_pad_shapes name the op
#
# The stage compiles nothing but the vision path, so every NEFF it writes is a vision NEFF.
# NEFFs are keyed by CONTENT hash (sha1 of the .neff), so a .ntff is never paired with
# the wrong graph even when the compiler reuses a file name across cache dirs.
#
# Run from the repo root inside the job pod:  bash scripts/neuron/profile_vision_job.sh
# Env: VIDEO VIDEO_FRAMES ITERS MAX_PAIRS INGEST_TIMEOUT RESULTS ARCHIVE_DIR RUN_TAG
#      INGEST_FLOOR_BYTES / INGEST_QUIET_S (parquet completion rule; lower only for dry runs)
set -euo pipefail

VIDEO=${VIDEO:-media/cat_sunglasses.mp4}
VIDEO_FRAMES=${VIDEO_FRAMES:-8}
ITERS=${ITERS:-10}
MAX_PAIRS=${MAX_PAIRS:-3}
INGEST_TIMEOUT=${INGEST_TIMEOUT:-3600}
INGEST_FLOOR_BYTES=${INGEST_FLOOR_BYTES:-2097152}
INGEST_QUIET_S=${INGEST_QUIET_S:-180}
RESULTS=${RESULTS:-/tmp/results/vision_profile}
ARCHIVE_DIR=${ARCHIVE_DIR:-}
RUN_TAG=${RUN_TAG:-vision_profile_$(date +%Y%m%d_%H%M%S)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm -rf "$RESULTS"; mkdir -p "$RESULTS"
PROF=neuron-explorer
command -v "$PROF" >/dev/null || PROF=/opt/aws/neuron/bin/neuron-explorer
[ -x "$PROF" ] || command -v "$PROF" >/dev/null || PROF=neuron-profile
echo "PROFILER CLI: $PROF"
python3 -c "import duckdb" 2>/dev/null || python3 -m pip install -q duckdb

# ── archive on EXIT: one tar, copied once, verified by size (PVC is S3-FUSE) ──
archive() {
  local rc=$?
  set +e
  local tgz="/tmp/${RUN_TAG}.tar.gz"
  tar -czf "$tgz" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")" 2>/dev/null
  echo "ARCHIVE: $tgz $(wc -c < "$tgz") bytes (exit code $rc)"
  if [ -n "$ARCHIVE_DIR" ]; then
    mkdir -p "$ARCHIVE_DIR"
    for f in "$tgz" /tmp/${RUN_TAG}_upload_bundle.tar.gz; do
      [ -s "$f" ] || continue
      cp "$f" "$ARCHIVE_DIR/" && [ "$(wc -c < "$ARCHIVE_DIR/$(basename "$f")")" = "$(wc -c < "$f")" ] \
        && echo "  saved $ARCHIVE_DIR/$(basename "$f")" || echo "  COPY FAILED/short: $f"
    done
  fi
}
trap archive EXIT

# One stage = one empty cache: everything compiled from here on is the vision path.
export NEURON_COMPILE_CACHE_URL=/tmp/vision_neff/compile_cache
export TORCH_NEURONX_NEFF_CACHE_DIR=/tmp/vision_neff/neff_cache
export TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS=1
rm -rf /tmp/vision_neff; mkdir -p "$NEURON_COMPILE_CACHE_URL" "$TORCH_NEURONX_NEFF_CACHE_DIR"
touch /tmp/vision_stage_start; sleep 1

BENCH=(python3 profile_vision.py --video "$VIDEO" --video-frames "$VIDEO_FRAMES" --vision compile)
median_of() { grep -oE "median=[0-9.]+ms" "$1" | tail -1 || true; }

echo "════════ VISION PROFILE (TP=1, ${VIDEO}, ${VIDEO_FRAMES} frames) ════════"

echo "=== [1/3] WARMUP: compile into empty stage cache (not timed) + VISION CHECK vs CPU ==="
"${BENCH[@]}" --warmup 1 --iters 0 --check 2>&1 | tee "$RESULTS/warmup.txt" \
  | grep -E "VIDEO|VISION|built=|reference|Error|Traceback" || echo "WARMUP FAILED"

# Collect this stage's NEFFs by content hash into one flat dir.
NEFFS="$RESULTS/neffs"; mkdir -p "$NEFFS"
find /tmp /var/tmp -xdev -name '*.neff' -newer /tmp/vision_stage_start -not -path "$RESULTS/*" 2>/dev/null \
  > "$RESULTS/stage_neff_paths.txt" || true
while read -r f; do
  [ -s "$f" ] || continue
  id=$(sha1sum "$f" | cut -c1-16)
  [ -f "$NEFFS/$id.neff" ] || cp "$f" "$NEFFS/$id.neff"
  echo "$id $(wc -c < "$f") $f" >> "$RESULTS/neff_index.txt"
done < "$RESULTS/stage_neff_paths.txt"
N_NEFF=$(ls "$NEFFS"/*.neff 2>/dev/null | wc -l || true)
echo "vision-stage NEFFs: $N_NEFF unique (from $(wc -l < "$RESULTS/stage_neff_paths.txt") files)"
sort -k2 -n -r "$RESULTS/neff_index.txt" 2>/dev/null | head -10 | sed 's/^/  /' || true

echo "=== [2/3] CLEAN RUN (NO profiler): median of $ITERS = REAL vision latency ==="
"${BENCH[@]}" --warmup 1 --iters "$ITERS" 2>&1 | tee "$RESULTS/clean.txt" \
  | grep -E "built=|median=|Error|Traceback" || echo "CLEAN RUN FAILED"
T_CLEAN=$(median_of "$RESULTS/clean.txt")

echo "=== [3a] RUNTIME INSPECTOR (system + device profile) -> trace_info.pb ==="
SYSTRACE=/tmp/vision_systrace; rm -rf "$SYSTRACE"; mkdir -p "$SYSTRACE"
NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
NEURON_RT_INSPECT_OUTPUT_DIR="$SYSTRACE" \
  "${BENCH[@]}" --warmup 1 --iters 3 2>&1 | tee "$RESULTS/profiled.txt" \
  | grep -E "median=|Error|Traceback" || echo "PROFILED RUN FAILED"
T_PROF=$(median_of "$RESULTS/profiled.txt")
echo "  trace_info.pb count: $(find "$SYSTRACE" -name trace_info.pb | wc -l)"
echo "  .ntff / .neff      : $(find "$SYSTRACE" -name '*.ntff' | wc -l) / $(find "$SYSTRACE" -name '*.neff' | wc -l)"
RANKDIR=$(find "$SYSTRACE" -name trace_info.pb -printf '%s %h\n' 2>/dev/null | sort -n -r | head -1 | cut -d' ' -f2- || true)
echo "  bundle rank dir    : ${RANKDIR:-none}"

echo "=== [3b] system timeline -> perfetto (WHEN graphs ran) ==="
if [ -n "$RANKDIR" ]; then
  mkdir -p "$RESULTS/perfetto"
  ( cd "$RESULTS/perfetto" && timeout 900 "$PROF" view -d "$RANKDIR" --output-format perfetto >/dev/null 2>&1 ) || true
  find "$RESULTS/perfetto" -name '*.pftrace' -exec gzip -f {} \; 2>/dev/null || true
  ls -la "$RESULTS/perfetto" 2>/dev/null | sed 's/^/  /' || true
fi

echo "=== [4] per-NEFF capture --single-io + summary-json ($N_NEFF NEFFs) ==="
POUT="$RESULTS/perneff"; mkdir -p "$POUT"
OK=0
for neff in "$NEFFS"/*.neff; do
  [ -f "$neff" ] || continue
  id=$(basename "$neff" .neff)
  ( cd "$POUT" && NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1 timeout 900 \
      "$PROF" capture --single-io -n "$neff" -s "$POUT/$id.ntff" > "$POUT/$id.capture.log" 2>&1 ) \
    || { echo "  $id capture rc=$? (tail: $(tail -1 "$POUT/$id.capture.log" 2>/dev/null))"; continue; }
  [ -s "$POUT/$id.ntff" ] || { echo "  $id: no ntff"; continue; }
  timeout 900 "$PROF" view -n "$neff" -s "$POUT/$id.ntff" --output-format summary-json \
    > "$POUT/$id.json" 2>/dev/null || true
  [ -s "$POUT/$id.json" ] && OK=$((OK + 1))
done
echo "captured+viewed OK: $OK / $N_NEFF"
python3 "$HERE/profile_all_neffs.py" "$POUT" 2>&1 | tee "$RESULTS/perneff_breakdown.txt" || echo "aggregation failed"
python3 "$HERE/rank_neffs.py" "$POUT" 2>&1 | tee "$RESULTS/neff_ranking.txt" || echo "ranking failed"

# Preserve the heaviest pairs, each .neff + its .ntff + summary in ONE dir.
PAIRS="$RESULTS/pairs"; mkdir -p "$PAIRS"
i=0
while read -r id; do
  [ -n "$id" ] && [ -s "$POUT/$id.ntff" ] || continue
  i=$((i + 1))
  mb=$(( $(wc -c < "$NEFFS/$id.neff") / 1048576 ))
  d="$PAIRS/$(printf 'rank%02d' "$i")_${id}_${mb}MB"; mkdir -p "$d"
  cp "$NEFFS/$id.neff" "$POUT/$id.ntff" "$d/"
  cp "$POUT/$id.json" "$d/summary.json" 2>/dev/null || true
done < <(python3 "$HERE/top_neffs.py" "$POUT" "$MAX_PAIRS" 2>/dev/null || true)
echo "preserved pairs: $i"
ls -d "$PAIRS"/rank* 2>/dev/null | sed 's/^/  /' || true

TOP=$(ls -d "$PAIRS"/rank01_* 2>/dev/null | head -1 || true)
if [ -n "$TOP" ]; then
  id=$(basename "$TOP" | cut -d_ -f2)
  echo "═══ rank01 NEFF detail (summary-text) ═══"
  timeout 900 "$PROF" view -n "$TOP/$id.neff" -s "$TOP/$id.ntff" --output-format summary-text \
    2>/dev/null | tee "$RESULTS/rank01_summary.txt" | grep -E \
    '^\s*(total_time|[a-z_]*engine_active_time_percent|dma_active_time|hbm_(read|write)_bytes|mfu_estimated_percent|mbu_estimated_percent|matmul_instruction_count|transpose_flops|[a-z_]*dynamic_dma_packet_(percent|count)|static_dma_packet_count|dma_transfer_average_bytes)\s' \
    | sed 's/^/  [rank01] /' || echo "  summary-text failed"
fi

echo "=== upload bundle (Directory Upload for System Profile): rank dir + preserved pairs ==="
if [ -n "$RANKDIR" ]; then
  UB=/tmp/${RUN_TAG}_upload_bundle; rm -rf "$UB"; mkdir -p "$UB"
  cp "$RANKDIR"/* "$UB/" 2>/dev/null || true
  for d in "$PAIRS"/rank*; do [ -d "$d" ] && cp "$d"/*.neff "$d"/*.ntff "$UB/" 2>/dev/null || true; done
  echo "  inventory:"; for f in "$UB"/*; do echo "    $(wc -c < "$f") $(basename "$f")"; done
  tar -czf "/tmp/${RUN_TAG}_upload_bundle.tar.gz" -C "$UB" . && \
    echo "  /tmp/${RUN_TAG}_upload_bundle.tar.gz $(wc -c < "/tmp/${RUN_TAG}_upload_bundle.tar.gz") bytes"
fi

echo "=== [5] instruction level: parquet per pair -> name the op (max $MAX_PAIRS pairs) ==="
# Completion rule: the converter writes ~20KB of table headers at once, then parses for
# minutes, then idles on a server forever. So: wait for > 2MB AND 3 min with no growth,
# kill it, then REQUIRE a reader to find rows (pq_check.py) before reporting anything.
ingest() {  # $1=neff $2=ntff $3=outdir
  rm -rf "$3"; mkdir -p "$3"
  timeout "$INGEST_TIMEOUT" "$PROF" view -n "$1" -s "$2" --output-format parquet --output-file "$3" \
    --ignore-instruction-hierarchy --ignore-event-trace > "$3.log" 2>&1 &
  local pid=$! prev=-1 cur quiet=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    cur=$(du -sb "$3" 2>/dev/null | cut -f1)
    if [ "${cur:-0}" -gt "$INGEST_FLOOR_BYTES" ] && [ "$cur" = "$prev" ]; then
      quiet=$((quiet + 15)); [ "$quiet" -ge "$INGEST_QUIET_S" ] && break
    else
      quiet=0
    fi
    prev=$cur
  done
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  python3 "$HERE/pq_check.py" "$3"
}
for d in "$PAIRS"/rank*; do
  [ -d "$d" ] || continue
  id=$(basename "$d" | cut -d_ -f2)
  echo "########## $(basename "$d") ##########"
  if ingest "$d/$id.neff" "$d/$id.ntff" "$d/parquet"; then
    python3 "$HERE/pq_dma_report.py" "$d/parquet" "$(basename "$d")" 2>&1 | tee "$d/dma_report.txt" || true
    python3 "$HERE/pq_pad_shapes.py" "$d/parquet" "$(basename "$d")" 2>&1 | tee "$d/shapes.txt" || true
  else
    echo "  parquet ingest gave no readable rows (timeout ${INGEST_TIMEOUT}s?) — op naming skipped for this pair"
  fi
done

echo "════════ VISION PROFILE SUMMARY ════════"
echo "  clean    (no profiler, REAL latency) : ${T_CLEAN:-N/A}"
echo "  profiled (runtime inspector on)      : ${T_PROF:-N/A}"
grep -h "VISION CHECK\|built=" "$RESULTS/warmup.txt" | sed 's/^/  /' || true
echo "  NEFFs: $N_NEFF   captured+viewed: $OK   pairs: $i"
echo "  (trust engine SHARES from [4]; absolute single-io us are noisy run to run)"
