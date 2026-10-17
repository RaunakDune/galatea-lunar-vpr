# Reproducing the paper

## Compute

Everything in the paper was run on a 2-core CPU with 8 GB RAM and no GPU.

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
