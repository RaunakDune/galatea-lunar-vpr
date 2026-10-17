# Results

| Path | Contents |
|---|---|
| `runs/<name>/results.json` | run arguments, learned λ and dust-bin score, overall and per-condition Recall@k on val and test, per-query hits |
| `runs/<name>/head.pt` | trained aggregation-head weights (load with the args in `results.json`) |
| `zeroshot_test_val.json` | zero-shot CosPlace / EigenPlaces / DINOv2-SALAD results |
| `rerank/*.npy` | stage-2 local scores for the top-10 candidates of each query (val and test) |
| `rerank_summary.json` | validation-tuned fusion parameters and test recall per local scorer |
| `dustbin_analysis.json` | dust-bin mass vs. sky fraction, cluster-mass statistics |
| `timing.json` | CPU run times |
| `percond_series.json`, `*.png` | data and images for the figures |

Run names: `galatea` (learned λ), `salad` (λ = 1), `lam0` / `lam0.25` / `lam0.75` (fixed λ), `netvlad`, `gem`;
the suffixes `_nodust`, `_L1`, `_L10`, `_m16`, `_m32`, `_fast` (25× lr on λ, z) and `_fixedillum`
(single-Sun training) mark ablations; `_s<k>` is the seed. The descriptor files (`desc_*.pt`) are not
committed. `train_eval.py --save_desc` regenerates them.
