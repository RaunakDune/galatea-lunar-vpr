# Real Apollo imagery

Two small station-level tests of global retrieval on Apollo Hasselblad surface photographs. Neither has
metric poses: a retrieval is correct when it returns the right traverse station. Paper: Section VI-G and
Table VI; supplement: "Real-imagery protocols" and Table S-real.

## 1. Panoramas (real against real)

The image libraries of the [Apollo Lunar Surface Journal](https://apollojournals.org/alsj/) list the frames
of every assembled panorama and the station where it was taken.

| Step | Script | Output |
|---|---|---|
| Parse the six image-library pages (`images11.html` ... `images17.html`) | `src/apollo/parse_alsj.py` | per-mission pan lists |
| Keep ordinary 60 mm single-standpoint pans and label stations | `src/apollo/label_pans.py` | `data/apollo/apollo_pans.json` (79 pans, 1,663 frames, 47 stations) |
| Download the frames (ASU scans, `small` size) and crop the film frame | DESPINA `tools/download_apollo.py`, `scripts/process_apollo.py --size 512` | 1,640 frames (23 frames of AS17-142 are not served) |
| Describe and retrieve | `src/apollo/apollo_eval.py` | `apollo.pt` |
| Summarise with intervals over query pans | `src/apollo/apollo_report.py` | `results/apollo/apollo_panos_{center,full}_summary.json` |

A query's database is every frame of every other pan of the same mission. The paper scores Apollo 15 to 17
(958 queries, 41 pans, 19 stations; random 6.5%). `main_noLM` leaves out the queries taken at the LM.

## 2. Rendered map (real against rendered)

Apollo 15 frames against views rendered from the LROC NAC DTM `NAC_DTM_APOLLO15` (2 m).

| Step | Script | Notes |
|---|---|---|
| Render depth, soft edges and a shaded view per station and heading | `src/apollo/render_a15.py` | needs the DESPINA repository (`DESPINA_DIR`), the DTM and the DE421 / lunar PA kernels for the Sun direction |
| Generate DESPINA images | `src/apollo/gen_views_lm.py` | SDXL + DESPINA LoRA and ControlNets; fp16, about 25 s per image on an RTX 5080 (16 GB) |
| Retrieve | `src/apollo/apollo_r2r.py` | databases `despina:image_seed0.png,shaded:dem_sim.png` |
| Summarise | `src/apollo/apollo_r2r_report.py` | `results/apollo/r2r_a15_summary.json` |

Nine stations (1, 2, 4, 6a, 9, 9a, 10, ALSEP = Station 8, LM) x 12 headings = 108 views
(`results/apollo/a15_views.json`). Station 6 lies about 50 m outside the valid DTM area; the SEVA pans were
taken from the LM hatch. 261 queries from 14 pans; random 11.1%.

## Tables

`src/apollo/make_apollo_tables.py` writes `paper/tab_real.tex`, `paper/tab_real_ci.tex` and
`paper/numbers_apollo.tex` from the three summary files.
