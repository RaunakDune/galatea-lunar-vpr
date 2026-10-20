# Reproducing the paper

## Compute

Frozen-backbone experiments were run on a 2-core CPU with 8 GB RAM and no GPU (and reproduced on GPUs);
joint backbone fine-tuning was run on NVIDIA Tesla P40 GPUs (see the GPU section below).

| Step | Command | Time (2 cores) |
|---|---|---|
| Setup | `bash scripts/setup.sh` | ~5 min (≈570 MB of weights) |
| Render benchmark (optional; images are shipped) | `python3 src/gen_dataset.py` and `FIXED_SUN=1 python3 src/gen_dataset.py` | ~45 min |
| Extract frozen features | `python3 src/extract.py` | ~1.5 h, ~7 GB on disk |
| Zero-shot baselines | `python3 src/zeroshot.py test val` | 1 min |
| 25 training runs | `bash scripts/run_experiments.sh` | ~11 min each |
| Re-ranking | see `scripts/reproduce_all.sh` | ~2.5 h |
| Tables, figures, paper | `python3 src/make_tables.py && python3 src/plot_percond.py` | 1 min |

The single command `bash scripts/reproduce_all.sh` runs every step in order. Each step skips work that
already exists.

## GPU: joint backbone fine-tuning and faster reproduction

Every script accepts a device: `DEV=cuda:0 python3 extract.py`, `python3 train_eval.py ... --device cuda:0`.
On a Tesla P40 feature extraction takes about 1 min per scene, a frozen-backbone run about 2 min and a
fine-tuning run (`train_ft.py`) about 25 min. The fine-tuning runs of the paper are listed in
`scripts/finetune_jobs.txt`; on a multi-GPU machine without a scheduler run them with

```bash
GALATEA_RESULTS=results/gpu bash scripts/gpu_queue.sh scripts/finetune_jobs.txt "0 1 2 3"
GALATEA_RESULTS=results/gpu bash scripts/rerank_parallel.sh 16 ft1e-4_galatea_s0:mnn,uot,fgw,lg \
    ft1e-4_galatea_s1:mnn,fgw ft1e-4_galatea_s2:mnn,fgw ft1e-4_netvlad_s0:mnn,fgw
```

`gpu_queue.sh` skips finished runs and can be re-launched safely. Fine-tuning is deterministic on a given
GPU (a repeated run reproduced the sweep run exactly). Fine-tuned backbone weights are not stored in the
repository (use `--save_backbone` to keep them); `results/gpu/` holds the metrics, re-ranking scores and logs.
Tested with PyTorch 2.5.1 + CUDA 12.4, which still supports Pascal GPUs (sm_60/61).

## Skipping training

`results/runs/*/head.pt` holds the trained aggregation heads and `results/runs/*/results.json` their metrics,
so the tables can be regenerated without retraining:

```bash
cd src && python3 make_tables.py && python3 plot_percond.py
cd ../paper && latexmk -pdf main.tex
```

To evaluate a shipped head on new images, load it with `aggregators.OTHead(...)`, using the arguments stored
in `results.json["info"]["args"]` (see `src/dustbin_analysis.py`). Then apply it to tokens from `extract.py`.

## Determinism

- Rendering is seeded per scene.
- Training is seeded per run (`--seed`).
- CPU PyTorch results can differ slightly across versions and thread counts. Expect seed-level variation of
  about ±1 point of R@1.

## Troubleshooting

- **Out of memory.** Evaluation uses memory-mapped features and training reads tokens lazily. Run one job at
  a time on 8 GB machines.
- **`cv2.USAC_MAGSAC` missing.** Install OpenCV 4.5 or newer.
- **Paths.** Set `GALATEA_DATA`, `GALATEA_RESULTS`, `GALATEA_WEIGHTS` or `GALATEA_EXT` to relocate the data,
  results, weights or third-party code (see `src/paths.py`).
- **Paper build.** The paper needs the `IEEEtran` class (TeX Live package `texlive-publishers`).
