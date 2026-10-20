#!/usr/bin/env bash
# Minimal multi-GPU job queue for machines without a scheduler.
#   bash scripts/gpu_queue.sh JOBS_FILE "1 2 3"      (run from the repository root)
# JOBS_FILE has one job per line:  <run_name>|<python script and arguments, without --name/--device>
# e.g.  ft_galatea_s0|train_ft.py --head ot --mode learn --seed 0 --save_desc
# One worker per listed GPU repeatedly claims the next job that has neither results.json nor a claim,
# so the file can be appended to while the queue runs and a re-launch skips finished work.
# Failed jobs keep their claim (marked FAILED) and are not retried; delete $RES/claims/<name> to retry.
set -uo pipefail
set -f   # no glob expansion of job arguments (e.g. --train_glob .../t*)
JOBS=$(readlink -f "$1"); GPUS=($2)
cd "$(dirname "$0")/../src"
RES="${GALATEA_RESULTS:-$(cd .. && pwd)/results}"
LOGS="${GALATEA_LOGS:-$(cd .. && pwd)/logs}"
mkdir -p "$RES/claims" "$LOGS"

worker() {
  local g=$1 name cmd picked pcmd
  while true; do
    picked=""
    while IFS='|' read -r name cmd; do
      name=$(echo "$name" | xargs)
      [ -z "$name" ] && continue
      [[ $name == \#* ]] && continue
      [ -f "$RES/runs/$name/results.json" ] && continue
      if mkdir "$RES/claims/$name" 2>/dev/null; then picked=$name; pcmd=$cmd; break; fi
    done < "$JOBS"
    [ -z "$picked" ] && { echo "[gpu$g] no more jobs $(date)"; return; }
    echo "[gpu$g] start $picked $(date)"
    if CUDA_VISIBLE_DEVICES=$g python $pcmd --name "$picked" --device cuda:0 > "$LOGS/$picked.log" 2>&1; then
      echo "[gpu$g] done  $picked $(date)"
    else
      echo "[gpu$g] FAILED $picked $(date)"; touch "$RES/claims/$picked/FAILED"
    fi
  done
}
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "QUEUE EMPTY $(date)"
