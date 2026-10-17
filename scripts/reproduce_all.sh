#!/usr/bin/env bash
# End-to-end reproduction of every number, table and figure in the paper.
# Total ~8-10 h on a 2-core CPU / 8 GB RAM (far less with a GPU). Steps are resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup.sh
cd src
# 1) Benchmark images (skip if you use the images shipped in data/lunarsim)
[ -f ../data/lunarsim/test/t1001/meta.json ] || python3 gen_dataset.py
[ -f ../data/lunarsim/train_fixed/t1/meta.json ] || FIXED_SUN=1 python3 gen_dataset.py
# 2) Frozen features (not shipped: several GB)
python3 extract.py
# 3) Zero-shot baselines
python3 zeroshot.py test val
# 4) Training runs
bash ../scripts/run_experiments.sh
# 5) Stage-2 re-ranking (validation for tuning, then test)
for sp in val test; do python3 rerank.py --stage1 galatea_s0 --split $sp --methods mnn,uot,fgw,lg --alpha 0.3; done
for s in 1 2; do for sp in val test; do python3 rerank.py --stage1 galatea_s$s --split $sp --methods mnn,fgw --alpha 0.3; done; done
for a in 0.0 0.6; do python3 rerank.py --stage1 galatea_s0 --split val --methods fgw --alpha $a; done
for sp in val test; do python3 rerank.py --stage1 salad_zs --split $sp --methods mnn; done
# 6) Analyses, timing, tables, figures, paper
python3 dustbin_analysis.py salad_s0 galatea_s0
python3 timing.py
python3 make_tables.py
python3 plot_percond.py
cd ../paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
