#!/usr/bin/env bash
# Run stage-2 re-ranking for several stage-1 runs in parallel on a many-core CPU.
#   bash scripts/rerank_parallel.sh NPROC "stage1:methods" ["stage1:methods" ...]
# e.g. bash scripts/rerank_parallel.sh 24 galatea_s0:mnn,uot,fgw,lg galatea_s1:mnn,fgw
# Each (stage1, split, terrain) is an independent process (resumable partial files); a final pass per
# (stage1, split) merges the partials into results/rerank/<tag>.npy.
set -uo pipefail
cd "$(dirname "$0")/../src"
NPROC=$1; shift
DATA="${GALATEA_DATA:-$(cd .. && pwd)/data/lunarsim}"
TASKS=$(mktemp)
for spec in "$@"; do
  s1=${spec%%:*}; m=${spec#*:}
  for sp in val test; do
    for d in "$DATA/$sp"/t*; do echo "$s1 $sp $m $(basename "$d")"; done
  done
done > "$TASKS"
echo "$(wc -l < "$TASKS") partial tasks"
xargs -P "$NPROC" -L 1 bash -c 'python rerank.py --stage1 $0 --split $1 --methods $2 --alpha 0.3 --threads 1 --only $3 > /dev/null 2>&1 && echo "ok $0 $1 $2 $3" || echo "FAILED $0 $1 $2 $3"' < "$TASKS"
for spec in "$@"; do
  s1=${spec%%:*}; m=${spec#*:}
  for sp in val test; do python rerank.py --stage1 "$s1" --split "$sp" --methods "$m" --alpha 0.3 --threads 1 | tail -1; done
done
rm -f "$TASKS"
