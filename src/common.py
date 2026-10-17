import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import json, glob, numpy as np, torch

ROOT = paths.DATA
POS_R, POS_YAW = 10.0, 60.0
CONDS = ['C0_same', 'C1_el15', 'C2_el6', 'C3_el60', 'C4_az+90', 'C5_az+180', 'C6_az+180_el8']


def yawdiff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def load_eval(split):
    """Returns a list of terrain dicts with features + metadata for val/test."""
    out = []
    import os
    for d in sorted(glob.glob(f'{ROOT}/{split}/t*')):
        if not os.path.exists(f'{d}/feats.pt'):
            print('WARNING: no features for', d); continue
        F = torch.load(f'{d}/feats.pt', weights_only=False, mmap=True)
        meta = json.load(open(f'{d}/meta.json'))
        ndb = len(meta['db'])
        db = meta['db']; q = meta['q']
        dbxy = np.array([[m['x'], m['y']] for m in db]); dby = np.array([m['yaw'] for m in db])
        qxy = np.array([[m['x'], m['y']] for m in q]); qy = np.array([m['yaw'] for m in q])
        dist = np.linalg.norm(qxy[:, None] - dbxy[None], axis=-1)
        pos = (dist < POS_R) & (yawdiff(qy[:, None], dby[None]) < POS_YAW)
        out.append({'dir': d, 'F': F, 'ndb': ndb, 'meta': meta, 'pos': pos, 'dist': dist,
                    'qcond': np.array([m['cond'] for m in q]), 'qplace': np.array([m['place'] for m in q])})
    return out


def recall_from_sim(sim, pos, Ns=(1, 5, 10)):
    """sim: [nq, ndb] numpy; pos: bool [nq, ndb]. Returns dict N -> bool array [nq]."""
    order = np.argsort(-sim, axis=1)
    hit = np.take_along_axis(pos, order, axis=1)
    return {N: hit[:, :N].any(1) for N in Ns}, order


def summarize(hits_by_terrain, conds_by_terrain, Ns=(1, 5, 10)):
    res = {}
    allc = np.concatenate(conds_by_terrain)
    for N in Ns:
        h = np.concatenate([h[N] for h in hits_by_terrain])
        res[f'R@{N}'] = float(h.mean() * 100)
        for c in CONDS:
            res[f'R@{N}_{c}'] = float(h[allc == c].mean() * 100)
    return res


def bootstrap_ci(h, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(h)
    s = np.array([h[rng.integers(0, n, n)].mean() for _ in range(B)]) * 100
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))
