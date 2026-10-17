# Data

`lunarsim/` contains the LunarSim-VPR benchmark: 14,000 rendered images plus one `meta.json` per scene. See
[`../docs/BENCHMARK.md`](../docs/BENCHMARK.md) for the full dataset card.

Feature files (`feats.pt`) are deliberately not committed (they total several GB). Generate them with
`python3 src/extract.py`.
