#!/usr/bin/env bash
# Exact list of training runs reported in the paper (each ~11 min on a 2-core CPU).
# Skips runs whose results.json already exists, so it can be re-launched safely.
set -uo pipefail
cd "$(dirname "$0")/../src"
RES="${GALATEA_RESULTS:-$(cd .. && pwd)/results}"
DATA="${GALATEA_DATA:-$(cd .. && pwd)/data/lunarsim}"
COMMON="--steps 500 --P 24 --threads ${THREADS:-2} --eval_split val,test"
run() {
  name=$1; shift
  if [ -f "$RES/runs/$name/results.json" ]; then echo "skip $name"; return; fi
  echo "=== $name $(date)"
  python3 train_eval.py --name "$name" $COMMON "$@"
}
for s in 0 1 2; do
  run galatea_s$s --head ot --mode learn    --seed $s --save_desc
  run salad_s$s   --head ot --mode balanced --seed $s --save_desc
  run netvlad_s$s --head netvlad            --seed $s --save_desc
  run gem_s$s     --head gem                --seed $s --save_desc
  run lam0_s$s    --head ot --mode softmax  --seed $s
done
run lam0.25_s0 --head ot --mode fixed --lam0 0.25 --seed 0
run lam0.75_s0 --head ot --mode fixed --lam0 0.75 --seed 0
run galatea_nodust_s0 --head ot --mode learn --no_dustbin --seed 0
run galatea_fast_s0 --head ot --mode learn --scalar_lr_mult 25 --seed 0
run galatea_fixedillum_s0 --head ot --mode learn    --seed 0 --train_glob "$DATA/train_fixed/t*"
run salad_fixedillum_s0   --head ot --mode balanced --seed 0 --train_glob "$DATA/train_fixed/t*"
run galatea_L1_s0  --head ot --mode learn --iters 1  --seed 0
run galatea_L10_s0 --head ot --mode learn --iters 10 --seed 0
run galatea_m16_s0 --head ot --mode learn --m 16 --seed 0
run galatea_m32_s0 --head ot --mode learn --m 32 --seed 0
echo "ALL RUNS DONE $(date)"
