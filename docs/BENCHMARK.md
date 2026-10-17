# LunarSim-VPR: dataset card

LunarSim-VPR is a procedurally generated benchmark for visual place recognition on the lunar surface under
controlled illumination change. All images are rendered by [`src/lunarsim.py`](../src/lunarsim.py) and the
splits are produced by [`src/gen_dataset.py`](../src/gen_dataset.py). Both are deterministic given the scene
seeds.

## Contents

| Split | Directory | Scene seeds | Places | Images | Purpose |
|---|---|---|---|---|---|
| test | `data/lunarsim/test/t1001..t1003` | 1001, 1002, 1003 | 3 × 150 | 3 × (150 db + 1,050 queries) | evaluation |
| val | `data/lunarsim/val/t901` | 901 | 100 | 100 db + 700 queries | hyper-parameter selection only |
| train | `data/lunarsim/train/t1..t5` | 1–5 | 5 × 240 | 5 × 960 (4 views per place) | training, Sun drawn per view |
| train_fixed | `data/lunarsim/train_fixed/t1..t5` | 1–5 | 5 × 240 | 5 × 960 | ablation: same poses, single Sun direction |

In total there are 14,000 grayscale PNG images (320 × 240, 8-bit), about 675 MB. The extracted features
(`feats.pt`) are not distributed. Regenerate them with `python3 src/extract.py`.

## Renderer

- **Terrain.** Each scene has two DEMs. The regional DEM is 16 m/px over 16.4 km and has fBm relief, craters
  of 60–2,500 m, and 4–7 massifs 200–700 m high at 4–7.5 km. The local DEM is 0.3 m/px over 614 m and has
  fBm relief, craters of 1.5–120 m, ejecta boulders and fresh-crater albedo halos; it is blended into the
  centre of the regional DEM. Crater diameters follow a cumulative N(>D) = c·D⁻² distribution. Each crater
  has a bowl, rim and ejecta profile whose depth/diameter ratio (0.04–0.20) depends on its degradation.
- **Photometry.** The lunar-Lambert function (McEwen 1991) with phase-dependent L(g), a Hapke-style
  opposition surge (B₀ = 1, h = 0.06), and a weak ambient term. Cast shadows come from both DEMs and are
  averaged over the 0.53° solar disk.
- **Camera.** A 70° HFOV pinhole camera mounted 1.35–1.7 m above the ground, with zero roll and pitch and
  the principal point at 30 % of the image height. Rendering uses column-wise height-field ray casting with
  lunar curvature, followed by auto-exposure, a soft-shoulder tone curve, gamma, shot/read noise and 8-bit
  quantisation.

## Protocol

- Places lie every 6 m along a smooth random traverse inside ±170 m.
- **Database:** rendered at Sun azimuth 135°, elevation 30°.
- **Queries:** each test or val place has one jittered pose (offset ≤ 3 m, heading noise N(0, 10°), random
  camera height). That *same pose* is rendered under the seven conditions below, so differences between
  conditions are paired.
- **Ground truth:** a database image is a true match if it is within 10 m and 60° of heading of the query.
- **Metric:** Recall@k (k = 1, 5, 10), reported overall and per condition.

| ID | Name in files | Sun az / el (deg) | Change vs. database |
|---|---|---|---|
| C0 | `C0_same` | 135 / 30 | viewpoint only |
| C1 | `C1_el15` | 135 / 15 | lower Sun |
| C2 | `C2_el6` | 135 / 6 | grazing Sun |
| C3 | `C3_el60` | 135 / 60 | high Sun, low contrast |
| C4 | `C4_az+90` | 225 / 30 | shadows rotate 90° |
| C5 | `C5_az+180` | 315 / 30 | shadows reverse |
| C6 | `C6_az+180_el8` | 315 / 8 | reverse + grazing |

Each training view is rendered with a Sun direction drawn from a per-scene pool of 16 random directions
(elevation 4–75°, any azimuth).

## File format

Each scene directory contains the PNG images and a `meta.json`:

```json
{
  "seed": 1001, "split": "test",
  "db": [{"file": "db_0000.png", "place": 0, "x": 12.3, "y": -40.1, "yaw": 87.5, "az": 135.0, "el": 30.0}, ...],
  "q":  [{"file": "q_C2_el6_0000.png", "place": 0, "cond": "C2_el6", "x": 13.1, "y": -41.0, "yaw": 80.2, "az": 135, "el": 6}, ...]
}
```

Coordinates are in metres in the scene frame. `yaw` is in degrees, counter-clockwise from +x. Training
entries carry `view` (0–3) instead of `cond`.

## Regenerating or extending

```bash
cd src
python3 gen_dataset.py                 # test, val, train (about 25 min on 2 cores)
FIXED_SUN=1 python3 gen_dataset.py     # train_fixed ablation
```

To add scenes or conditions, edit `TEST_CONDS` and the `specs` list in `gen_dataset.py`.

## Known limitations

- Near-field rocks smaller than the 0.3 m DEM spacing are not resolved.
- Multiple scattering is approximated by a constant ambient term.
- There is no rover self-shadowing, dust or lens flare.
- The sky is near-black.

Treat LunarSim as a controlled test bed, not a substitute for real lunar imagery.
