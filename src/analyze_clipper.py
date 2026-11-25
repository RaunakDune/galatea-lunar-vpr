"""CLIPPER as a stage-2 scorer on LunarSim-VPR (rerank_clipper.py output).

Four rows, as in the paper's re-ranking table: CLIPPER with image-plane or 3-D (oracle depth) distances,
each with and without MAGSAC++ verification. For every row the tolerance eps is selected on the
validation scene by the fused score (tune(): R@1 + 0.5 R@5, same (beta, kappa) grid as the paper), and
the fusion parameters of that eps are applied to the test scenes. Differences in test R@1 carry a 95%
bootstrap CI over the 450 test places, against the global ranking, image-plane FGW and MNN + MAGSAC++.
Writes <RESULTS>/clipper_summary.json.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from analyze_rerank import tune, recall_after
from analyze3d_lib import hits, summ, paired_ci

RR = os.path.join(paths.RESULTS, 'rerank')
BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (5, 10, 20, 40, 80, 160, 320, 640)
sv = np.load(f'{RR}/galatea_s0_val_clipper.npy', allow_pickle=True).item()
st = np.load(f'{RR}/galatea_s0_test_clipper.npy', allow_pickle=True).item()
tv, tt = next(iter(sv.values())), next(iter(st.values()))
out = {'val': {}, 'rows': {}}
P = {}
for k in sorted(set(tv) & set(tt)):
    if k.startswith('clipper') or k in ('fgw', 'fgw_ransac', 'mnn_ransac', 'mnn', 'lg', 'lg_ransac'):
        b, kap = tune(sv, 'val', k, BET, KAP)
        r = recall_after(sv, 'val', k, 'fused', b, kap)
        P[k] = (b, kap)
        out['val'][k] = {'beta': b, 'kappa': kap, 'R@1': round(r['R@1'], 2), 'R@5': round(r['R@5'], 2)}
crit = lambda k: out['val'][k]['R@1'] + 0.5 * out['val'][k]['R@5']
g1, g5, cond = hits(st, 'test', None)
H = {}
for k in ('fgw', 'mnn_ransac'):
    H[k] = hits(st, 'test', k, *P[k])[0]
for dim in ('2d', '3d'):
    for suf in ('', '_ransac'):
        cands = [k for k in out['val'] if k.startswith(f'clipper{dim}_') and k.endswith('_ransac') == (suf == '_ransac')]
        k = max(cands, key=crit)
        l1, l5, _ = hits(st, 'test', k, mode='local')
        f1, f5, _ = hits(st, 'test', k, *P[k])
        out['rows'][f'clipper{dim}{suf}'] = {'selected': k, 'beta_kappa': P[k],
                                             'local': {kk: round(float(v), 2) for kk, v in summ(l1, l5, cond).items()},
                                             'fused': {kk: round(float(v), 2) for kk, v in summ(f1, f5, cond).items()},
                                             'd_vs_global': paired_ci(g1, f1), 'd_vs_fgw': paired_ci(H['fgw'], f1),
                                             'd_vs_mnn_ransac': paired_ci(H['mnn_ransac'], f1)}
json.dump(out, open(f'{paths.RESULTS}/clipper_summary.json', 'w'), indent=1)
print(f"global {100 * g1.mean():.1f}  fgw {100 * H['fgw'].mean():.1f}  mnn_ransac {100 * H['mnn_ransac'].mean():.1f}")
for n, v in out['rows'].items():
    print(f"{n:18s} [{v['selected']}] local {v['local']['R@1']:.1f} fused {v['fused']['R@1']:.1f} R5 {v['fused']['R@5']:.1f} "
          f"hard {v['fused']['hard']:.1f} | vs glob {v['d_vs_global']} vs fgw {v['d_vs_fgw']} vs mnnR {v['d_vs_mnn_ransac']}")
