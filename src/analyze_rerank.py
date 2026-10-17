"""Compute recall after stage-2 re-ranking from a rerank .npy file.
Modes: 'local' (sort top-K by local score, ties by global) and 'fused' s_g + beta*s/(s+kappa)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_eval, CONDS

_cache = {}


def gt(split):
    if split not in _cache:
        T = load_eval(split)
        _cache[split] = {os.path.basename(t['dir']): (t['pos'], t['qcond'], t['ndb']) for t in T}
        for t in T:
            del t['F']
    return _cache[split]


def recall_after(store, split, method, mode='fused', beta=1.0, kappa=10.0, Ns=(1, 5, 10)):
    G = gt(split)
    hits = {N: [] for N in Ns}
    conds = []
    for name, S in store.items():
        pos, qcond, ndb = G[name]
        cand, gs = S['cand'], S['gsim']
        nq, K = cand.shape
        if method is None:
            score = gs
        else:
            ls = S[method].astype(float)
            if mode == 'local':
                score = ls + 1e-3 * gs
            else:
                score = gs + beta * ls / (ls + kappa)
        order = np.argsort(-score, 1)
        rer = np.take_along_axis(cand, order, 1)
        hp = np.take_along_axis(pos[:nq], rer, 1)
        for N in Ns:
            hits[N].append(hp[:, :N].any(1))
        conds.append(qcond[:nq])
    conds = np.concatenate(conds)
    out = {}
    for N in Ns:
        h = np.concatenate(hits[N])
        out[f'R@{N}'] = h.mean() * 100
        out[f'hits{N}'] = h
        for c in CONDS:
            out[f'R@{N}_{c}'] = h[conds == c].mean() * 100 if (conds == c).any() else np.nan
    return out


def tune(store, split, method, betas=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0), kappas=(2, 5, 10, 20, 50)):
    best = None
    for b in betas:
        for k in kappas:
            r = recall_after(store, split, method, 'fused', b, k)
            key = r['R@1'] + 0.5 * r['R@5']
            if best is None or key > best[0]:
                best = (key, b, k)
    return best[1], best[2]


if __name__ == '__main__':
    f = sys.argv[1]
    split = 'val' if '_val_' in f else 'test'
    store = np.load(f, allow_pickle=True).item()
    methods = [k for k in next(iter(store.values())).keys() if k not in ('cand', 'gsim')]
    r = recall_after(store, split, None)
    print('global      R@1 %.1f R@5 %.1f' % (r['R@1'], r['R@5']))
    for m in methods:
        rl = recall_after(store, split, m, 'local')
        rf = recall_after(store, split, m, 'fused', 0.3, 10)
        print('%-12s local R@1 %.1f R@5 %.1f | fused(0.3,10) R@1 %.1f R@5 %.1f' % (m, rl['R@1'], rl['R@5'], rf['R@1'], rf['R@5']))
