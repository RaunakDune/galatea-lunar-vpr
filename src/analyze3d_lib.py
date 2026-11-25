"""Per-query hits, summaries and place-level bootstrap CIs shared by analyze3d.py and analyze_clipper.py."""
import numpy as np
from analyze_rerank import gt

HARD = ('C2_el6', 'C5_az+180', 'C6_az+180_el8')


def hits(store, split, method, beta=None, kappa=None, mode='fused'):
    G = gt(split)
    h1, h5, cond = [], [], []
    for name in sorted(store):
        S = store[name]
        pos, qcond, _ = G[name]
        cand, gs = S['cand'], S['gsim']
        if method is None:
            score = gs
        else:
            ls = S[method].astype(float)
            score = ls + 1e-3 * gs if mode == 'local' else gs + beta * ls / (ls + kappa)
        order = np.argsort(-score, 1, kind='stable')
        hp = np.take_along_axis(pos[:len(cand)], np.take_along_axis(cand, order, 1), 1)
        h1.append(hp[:, 0]); h5.append(hp[:, :5].any(1)); cond.append(qcond[:len(cand)])
    return np.concatenate(h1), np.concatenate(h5), np.concatenate(cond)


def summ(h1, h5, cond, valid=None):
    v = np.ones(len(h1), bool) if valid is None else valid
    return {'R@1': 100 * h1[v].mean(), 'R@5': 100 * h5[v].mean(),
            'hard': 100 * h1[v & np.isin(cond, HARD)].mean(),
            **{c.split('_')[0]: 100 * h1[v & (cond == c)].mean() for c in np.unique(cond)}}


def paired_ci(a, b, valid=None, n=2000, seed=0):
    """b - a in R@1 points; queries ordered scene x condition x place (3 x 7 x 150)."""
    rng = np.random.default_rng(seed)
    v = np.ones(len(a), bool) if valid is None else valid
    k = np.arange(len(a)); pid = ((k // 1050) * 150 + k % 150)[v]
    d = (b[v].astype(float) - a[v].astype(float)) * 100
    per = np.bincount(pid, d, minlength=450); cnt = np.bincount(pid, minlength=450)
    bs = [per[i].sum() / cnt[i].sum() for i in (rng.integers(0, 450, 450) for _ in range(n))]
    return [round(float(d.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


