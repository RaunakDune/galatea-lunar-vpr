"""Generate the LunarSim-VPR benchmark (train / val / test splits)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, json, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from lunarsim import Terrain, render
from PIL import Image

OUT = paths.DATA
DB_SUN = (135.0, 30.0)
TEST_CONDS = {'C0_same': (135, 30), 'C1_el15': (135, 15), 'C2_el6': (135, 6), 'C3_el60': (135, 60),
              'C4_az+90': (225, 30), 'C5_az+180': (315, 30), 'C6_az+180_el8': (315, 8)}

def make_path(rng, n_places, spacing=6.0, box=170.0):
    pts, heads = [], []
    x, y = rng.uniform(-60, 60, 2); h = rng.uniform(0, 2 * np.pi); w = 0.0
    sub = 0.5
    acc = 0.0
    while len(pts) < n_places:
        w = 0.92 * w + rng.normal(0, 0.012)
        # steer back inside the box
        if abs(x) > box or abs(y) > box:
            target = np.arctan2(-y, -x)
            dh = (target - h + np.pi) % (2 * np.pi) - np.pi
            w = 0.04 * np.sign(dh)
        h += w * sub
        x += sub * np.cos(h); y += sub * np.sin(h); acc += sub
        if acc >= spacing:
            acc = 0.0; pts.append((x, y)); heads.append(np.degrees(h) % 360)
    return np.array(pts), np.array(heads)

def save(img, path):
    Image.fromarray(img).save(path, optimize=False, compress_level=3)

def job(spec):
    split, seed, n_places = spec['split'], spec['seed'], spec['n_places']
    rng = np.random.default_rng(seed + 12345)
    t0 = time.time()
    T = Terrain(seed)
    d = f"{OUT}/{split}{'_fixed' if os.environ.get('FIXED_SUN') else ''}/t{seed}"
    os.makedirs(d, exist_ok=True)
    P, Hd = make_path(rng, n_places)
    meta = {'seed': seed, 'split': split, 'db': [], 'q': []}
    if split in ('test', 'val'):
        # database
        for i, ((x, y), h) in enumerate(zip(P, Hd)):
            fn = f"db_{i:04d}.png"
            save(render(T, x, y, h, *DB_SUN, rng=rng), f"{d}/{fn}")
            meta['db'].append({'file': fn, 'place': i, 'x': x, 'y': y, 'yaw': h, 'az': DB_SUN[0], 'el': DB_SUN[1]})
        # queries: fixed jittered poses shared by all conditions (paired design)
        jit = []
        for i in range(n_places):
            r = 3.0 * np.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
            jit.append((P[i, 0] + r * np.cos(a), P[i, 1] + r * np.sin(a), (Hd[i] + rng.normal(0, 10)) % 360,
                        rng.uniform(1.35, 1.7)))
        for cname, (az, el) in TEST_CONDS.items():
            for i, (x, y, h, ch) in enumerate(jit):
                fn = f"q_{cname}_{i:04d}.png"
                save(render(T, x, y, h, az, el, cam_h=ch, rng=rng), f"{d}/{fn}")
                meta['q'].append({'file': fn, 'place': i, 'cond': cname, 'x': x, 'y': y, 'yaw': h, 'az': az, 'el': el})
    else:
        # training: K=4 views per place, sun drawn from a per-terrain pool of 16 conditions
        pool = [(rng.uniform(0, 360), float(np.clip(rng.uniform(4, 75), 4, 75))) for _ in range(16)]
        if os.environ.get('FIXED_SUN'):          # ablation: single-illumination training set, same poses
            pool = [DB_SUN] * 16
        K = 4
        assign = []
        for i in range(n_places):
            for k in range(K):
                c = rng.integers(len(pool))
                r = 3.0 * np.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
                assign.append((c, i, k, P[i, 0] + r * np.cos(a), P[i, 1] + r * np.sin(a),
                               (Hd[i] + rng.normal(0, 10)) % 360, rng.uniform(1.35, 1.7)))
        assign.sort(key=lambda z: z[0])   # group by sun condition (shadow-map cache)
        for c, i, k, x, y, h, ch in assign:
            az, el = pool[c]
            fn = f"tr_{i:04d}_{k}.png"
            save(render(T, x, y, h, az, el, cam_h=ch, rng=rng), f"{d}/{fn}")
            meta['q'].append({'file': fn, 'place': i, 'view': k, 'x': x, 'y': y, 'yaw': h, 'az': az, 'el': el})
    json.dump(meta, open(f"{d}/meta.json", 'w'))
    return f"{split} t{seed}: {len(meta['db'])} db, {len(meta['q'])} imgs, {time.time() - t0:.0f}s"

if __name__ == '__main__':
    if os.environ.get('FIXED_SUN'):
        specs = [{'split': 'train', 'seed': s, 'n_places': 240} for s in (1, 2, 3, 4, 5)]
        with Pool(1) as p:
            for r in p.imap_unordered(job, specs):
                print(r, flush=True)
        sys.exit(0)
    specs = ([{'split': 'test', 'seed': s, 'n_places': 150} for s in (1001, 1002, 1003)] +
             [{'split': 'val', 'seed': 901, 'n_places': 100}] +
             [{'split': 'train', 'seed': s, 'n_places': 240} for s in (1, 2, 3, 4, 5)])
    with Pool(2) as p:
        for r in p.imap_unordered(job, specs):
            print(r, flush=True)
