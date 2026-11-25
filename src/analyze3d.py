"""Compare the image-plane FGW structural term with the 3-D one (rerank3d.py).

Benchmark: for every scorer, (beta, kappa) of the fusion s_g + beta * s / (s + kappa) are tuned on the
validation scene (same grid as the paper's Table IV) and applied to the test scenes. The 3-D variant
(mode, alpha) is selected on validation as well (fused score without RANSAC, the criterion of tune()). Differences in R@1 carry a 95% bootstrap CI over the
450 test places (all queries of a place resampled together), as in make_tables.py.
Robustness: the same fusion parameters, applied to the heading, offset and pitch variants.
Writes <RESULTS>/fgw3d_summary.json.
usage: python analyze3d.py [--robust_dir DIR]
"""
import os, sys, json, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from analyze_rerank import gt, tune, recall_after

ap = argparse.ArgumentParser()
ap.add_argument('--stage1', default='galatea_s0')
ap.add_argument('--robust_dir', default=os.path.join(paths.RESULTS, 'robust'))
args = ap.parse_args()
RR = os.path.join(paths.RESULTS, 'rerank')
BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (5, 10, 20, 40, 80, 160, 320, 640)
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


r2 = lambda d: {k: round(float(x), 2) for k, x in d.items()}
sv = np.load(f'{RR}/{args.stage1}_val_fgw3d.npy', allow_pickle=True).item()
st = np.load(f'{RR}/{args.stage1}_test_fgw3d.npy', allow_pickle=True).item()
keys3d = sorted(k for k in next(iter(sv.values())) if k.startswith('fgw3d') and k in next(iter(st.values())))
out = {'stage1': args.stage1, 'val': {}, 'test': {}, 'robust': {}}
# 1. validation: tune fusion for every scorer, pick the 3-D variant by fused val R@1 (+0.5 R@5, as tune())
fus = {}
for k in ['fgw', 'fgw_ransac'] + keys3d:
    b, kap = tune(sv, 'val', k, BET, KAP)
    r = recall_after(sv, 'val', k, 'fused', b, kap)
    fus[k] = (b, kap)
    out['val'][k] = {'beta': b, 'kappa': kap, 'R@1': round(r['R@1'], 2), 'R@5': round(r['R@5'], 2)}
# one structure (mode, alpha) is selected on its fused validation score without RANSAC and reported both
# with and without RANSAC verification
plain = [k for k in keys3d if not k.endswith('_ransac')]
b0 = max(plain, key=lambda k: out['val'][k]['R@1'] + 0.5 * out['val'][k]['R@5'])
best = {'': b0, '_ransac': b0 + '_ransac'}
out['selected'] = best
# 2. test
g1, g5, cond = hits(st, 'test', None)
out['test']['global'] = r2(summ(g1, g5, cond))
H = {}
for k in ['fgw', 'fgw_ransac'] + keys3d:
    l1, l5, _ = hits(st, 'test', k, mode='local')
    f1, f5, _ = hits(st, 'test', k, *fus[k])
    H[k] = f1
    out['test'][k] = {'local': r2(summ(l1, l5, cond)), 'fused': r2(summ(f1, f5, cond)), 'beta_kappa': fus[k],
                      'dR1_fused_vs_global': paired_ci(g1, f1)}
for suf, b2 in (('', 'fgw'), ('_ransac', 'fgw_ransac')):
    for k in keys3d:
        if k.endswith('_ransac') == (suf == '_ransac'):
            out['test'][k]['dR1_fused_vs_2d'] = paired_ci(H[b2], H[k])
# 3. robustness variants
for f in sorted(glob.glob(f'{args.robust_dir}/rerank3d_{args.stage1}_*.pt')):
    v = os.path.basename(f)[len(f'rerank3d_{args.stage1}_'):-3]
    R = torch.load(f'{args.robust_dir}/{v}.pt', weights_only=False)
    S2 = torch.load(f'{args.robust_dir}/rerank_{args.stage1}_{v}.pt', weights_only=False)
    S3 = torch.load(f, weights_only=False)
    tns = sorted(R)
    cond = np.concatenate([R[t]['qcond'] for t in tns])
    valid = np.concatenate([R[t]['pos'].any(1) for t in tns])
    res, HH = {}, {}
    for k in ['global', 'fgw', 'fgw_ransac', best[''], best['_ransac']]:
        h1, h5 = [], []
        for t in tns:
            cand = R[t]['models'][args.stage1]['cand'].astype(int); g = R[t]['models'][args.stage1]['gsim']
            if k == 'global':
                sc = g
            else:
                sl = (S2[t] if k in S2[t] else S3[t])[k].astype(float)
                b, kap = fus[k]
                sc = g + b * sl / (sl + kap)
            order = np.argsort(-sc, 1, kind='stable')
            hc = np.take_along_axis(np.take_along_axis(R[t]['pos'], cand, 1), order, 1)
            h1.append(hc[:, 0]); h5.append(hc[:, :5].any(1))
        h1, h5 = np.concatenate(h1), np.concatenate(h5)
        HH[k] = h1
        res[k] = r2(summ(h1, h5, cond, valid))
    res['dR1_3d_vs_2d'] = paired_ci(HH['fgw'], HH[best['']], valid)
    res['dR1_3d_vs_2d_ransac'] = paired_ci(HH['fgw_ransac'], HH[best['_ransac']], valid)
    res['n_valid'] = int(valid.sum())
    out['robust'][v] = res
json.dump(out, open(f'{paths.RESULTS}/fgw3d_summary.json', 'w'), indent=1)
print('selected', best)
print('val  ', {k: (v['R@1'], v['beta'], v['kappa']) for k, v in out['val'].items()})
print(f"test global R@1 {out['test']['global']['R@1']:.1f}")
for k in ['fgw', 'fgw_ransac'] + keys3d:
    t = out['test'][k]
    print(f"test {k:22s} local {t['local']['R@1']:5.1f}  fused {t['fused']['R@1']:5.1f} hard {t['fused']['hard']:5.1f}"
          f"  vs2d {t.get('dR1_fused_vs_2d', '')}")
for v, r in out['robust'].items():
    print(f"{v:11s} glob {r['global']['R@1']:5.1f} 2d {r['fgw']['R@1']:5.1f} 3d {r[best['']]['R@1']:5.1f} {r['dR1_3d_vs_2d']}"
          f" | 2dR {r['fgw_ransac']['R@1']:5.1f} 3dR {r[best['_ransac']]['R@1']:5.1f} {r['dR1_3d_vs_2d_ransac']}")
