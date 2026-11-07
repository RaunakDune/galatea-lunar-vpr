"""Render the robustness variants of the LunarSim-VPR test queries (roll, pitch, viewpoint).

Every variant re-renders the 3 x 150 test queries under all 7 Sun conditions, starting from exactly the
jittered query poses of the benchmark (replayed from the generator's random stream) and changing one
factor. The database is never re-rendered: variants are evaluated against the original upright
database images. Output: <DATA>/robust/<variant>/t<seed>/{q_*.png, meta.json}.

Variants
  roll_R     camera rolled by R deg about its central viewing direction (R = 0 re-renders the benchmark
             queries with fresh noise and is the paired baseline of every sweep)
  derot_R    roll R, then rotated back to upright as a gravity (IMU) correction would do
  pitch_P    camera pitched by P deg (positive = up)
  offset_D   position offset of exactly D m from the database place (benchmark: uniform in a 3 m disk),
             in the same random direction as the benchmark jitter
  heading_A  heading changed by +-A deg (random sign per query) on top of the benchmark heading noise
  height_H   camera height H m (benchmark: 1.35-1.7 m; database: 1.5 m)

usage: python gen_robust.py [--variants roll_0,roll_45,...] [--procs 30]
"""
import os, sys, json, time, argparse, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from lunarsim import Terrain
from tilt import render_tilted
from gen_dataset import make_path, TEST_CONDS, DB_SUN
from PIL import Image

SEEDS = (1001, 1002, 1003)
N_PLACES = 150
VARIANTS = ([f'roll_{r}' for r in (0, 15, 30, 45, 90, 180)] +
            [f'derot_{r}' for r in (15, 30, 45, 90, 180)] +
            [f'pitch_{p}' for p in (-30, -15, 15, 30)] +
            [f'offset_{d}' for d in (0, 2, 4, 6, 8)] +
            [f'heading_{a}' for a in (15, 30, 45)] +
            [f'height_{h}' for h in (0.5, 1.0, 1.5, 2.5, 4.0)])
OUT = os.path.join(paths.DATA, 'robust')


def benchmark_poses(seed):
    """Replay gen_dataset.job's random stream to recover the database path and the query jitter
    (offset radius r, direction a, heading, camera height); verified against meta.json."""
    rng = np.random.default_rng(seed + 12345)
    P, Hd = make_path(rng, N_PLACES)
    for _ in range(N_PLACES):                 # noise draws of the database renders
        rng.normal(0, 0.012, (240, 320)); rng.normal(0, 1, (240, 320))
    jit = []
    for i in range(N_PLACES):
        r = 3.0 * np.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
        jit.append(dict(r=r, a=a, yaw=(Hd[i] + rng.normal(0, 10)) % 360, h=rng.uniform(1.35, 1.7)))
    return P, Hd, jit


def query_pose(variant, P, i, j, sign):
    kind, val = variant.split('_'); val = float(val)
    x, y = P[i, 0] + j['r'] * np.cos(j['a']), P[i, 1] + j['r'] * np.sin(j['a'])
    pose = dict(x=x, y=y, yaw=j['yaw'], cam_h=j['h'], roll=0.0, pitch=0.0, derotate=False)
    if kind == 'roll':
        pose['roll'] = val
    elif kind == 'derot':
        pose.update(roll=val, derotate=True)
    elif kind == 'pitch':
        pose['pitch'] = val
    elif kind == 'offset':
        pose.update(x=P[i, 0] + val * np.cos(j['a']), y=P[i, 1] + val * np.sin(j['a']))
    elif kind == 'heading':
        pose['yaw'] = (j['yaw'] + sign * val) % 360
    elif kind == 'height':
        pose['cam_h'] = val
    return pose


_T = {}


def job(spec):
    seed, cname, variants = spec
    t0 = time.time()
    if seed not in _T:
        _T.clear(); _T[seed] = Terrain(seed)
    T = _T[seed]
    P, Hd, jit = benchmark_poses(seed)
    meta0 = json.load(open(f'{paths.DATA}/test/t{seed}/meta.json'))
    q0 = [m for m in meta0['q'] if m['cond'] == cname]
    assert all(abs(m['x'] - (P[i, 0] + jit[i]['r'] * np.cos(jit[i]['a']))) < 1e-9 and abs(m['yaw'] - jit[i]['yaw']) < 1e-9
               for i, m in enumerate(q0)), 'pose replay mismatch'
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], N_PLACES)
    az, el = TEST_CONDS[cname]
    ci = list(TEST_CONDS).index(cname)
    out = []
    for v in variants:
        d = f'{OUT}/{v}/t{seed}'
        os.makedirs(d, exist_ok=True)
        vi = VARIANTS.index(v)
        recs = []
        for i, j in enumerate(jit):
            pose = query_pose(v, P, i, j, signs[i])
            fn = f'q_{cname}_{i:04d}.png'
            if not os.path.exists(f'{d}/{fn}'):
                rng = np.random.default_rng([seed, vi, ci, i])
                img = render_tilted(T, pose['x'], pose['y'], pose['yaw'], az, el, cam_h=pose['cam_h'],
                                    roll=pose['roll'], pitch=pose['pitch'], rng=rng, derotate=pose['derotate'])
                Image.fromarray(img).save(f'{d}/{fn}', compress_level=3)
            recs.append({'file': fn, 'place': i, 'cond': cname, 'x': pose['x'], 'y': pose['y'], 'yaw': pose['yaw'],
                         'cam_h': pose['cam_h'], 'roll': pose['roll'], 'pitch': pose['pitch'],
                         'derotate': pose['derotate'], 'az': az, 'el': el})
        json.dump(recs, open(f'{d}/q_{cname}.json', 'w'))
        out.append(v)
    return f't{seed} {cname}: {len(out)} variants in {time.time() - t0:.0f}s'


def finalise(variants):
    """Merge the per-condition query lists into one meta.json per variant and terrain."""
    for v in variants:
        for seed in SEEDS:
            d = f'{OUT}/{v}/t{seed}'
            meta0 = json.load(open(f'{paths.DATA}/test/t{seed}/meta.json'))
            q = []
            for cname in TEST_CONDS:
                q += json.load(open(f'{d}/q_{cname}.json'))
            json.dump({'seed': seed, 'split': 'test', 'variant': v, 'db_dir': f'{paths.DATA}/test/t{seed}',
                       'db': meta0['db'], 'q': q}, open(f'{d}/meta.json', 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--procs', type=int, default=30)
    ap.add_argument('--groups', type=int, default=2, help='split the variant list per (terrain, condition)')
    a = ap.parse_args()
    vs = a.variants.split(',')
    chunks = [vs[k::a.groups] for k in range(a.groups)]
    specs = [(s, c, ch) for s in SEEDS for c in TEST_CONDS for ch in chunks if ch]
    t0 = time.time()
    with Pool(a.procs, maxtasksperchild=4) as p:
        for r in p.imap_unordered(job, specs):
            print(r, f'[{time.time() - t0:.0f}s]', flush=True)
    finalise(vs)
    print('DONE', len(vs), 'variants', f'{time.time() - t0:.0f}s')
