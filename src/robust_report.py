"""Summarise the robustness tests: Recall@k per variant and model, with and without fusion re-ranking.
Reads <RESULTS>/robust/<variant>.pt (robust_eval.py) and rerank_<stage1>_<variant>.pt (robust_rerank.py).
Writes <RESULTS>/robust/summary.json. Queries without any true match in the database (possible only for
the larger heading and offset changes) are excluded and counted.
"""
import os, sys, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

RB = os.path.join(paths.RESULTS, 'robust')
HARD = ('C2_el6', 'C5_az+180', 'C6_az+180_el8')
# fusion parameters tuned on the validation terrain for the frozen Galatea head (paper, Table IV)
FUSION = {'fgw': (1.0, 80.0), 'mnn_ransac': (8.0, 640.0)}
GROUPS = {'zs': ['salad_zs'], 'fr_gal': ['galatea_s0', 'galatea_s1', 'galatea_s2'],
          'fr_nv': ['netvlad_s0', 'netvlad_s1', 'netvlad_s2'],
          'ft_gal': ['rft_galatea_s0', 'rft_galatea_s1', 'rft_galatea_s2'],
          'ft_nv': ['rft_netvlad_s0', 'rft_netvlad_s1', 'rft_netvlad_s2']}


def metrics(hit1, hit5, cond, valid):
    m = {'R@1': 100 * hit1[valid].mean(), 'R@5': 100 * hit5[valid].mean()}
    m['C0'] = 100 * hit1[valid & (cond == 'C0_same')].mean()
    hv = valid & np.isin(cond, HARD)
    m['hard'] = 100 * hit1[hv].mean()
    for c in np.unique(cond):
        m[c.split('_')[0]] = 100 * hit1[valid & (cond == c)].mean()
    return m


def variant_summary(v):
    R = torch.load(f'{RB}/{v}.pt', weights_only=False)
    tns = sorted(R)
    cond = np.concatenate([R[t]['qcond'] for t in tns])
    valid = np.concatenate([R[t]['pos'].any(1) for t in tns])
    out = {'n_queries': int(len(valid)), 'n_no_positive': int((~valid).sum()), 'models': {}, 'valid': valid}
    for n in R[tns[0]]['models']:
        h = [np.take_along_axis(R[t]['pos'], R[t]['models'][n]['cand'].astype(int), 1) for t in tns]
        h = np.concatenate(h)
        out['models'][n] = metrics(h[:, 0], h[:, :5].any(1), cond, valid)
        out['models'][n]['hits'] = h[:, 0].astype(np.int8)
    rr = glob.glob(f'{RB}/rerank_*_{v}.pt')
    for f in rr:
        s1 = os.path.basename(f)[len('rerank_'):-len(f'_{v}.pt')]
        S = torch.load(f, weights_only=False)
        for meth, (beta, kappa) in FUSION.items():
            h1, h5 = [], []
            for t in tns:
                cand = R[t]['models'][s1]['cand'].astype(int); g = R[t]['models'][s1]['gsim']
                sl = S[t][meth]
                fused = g + beta * sl / (sl + kappa)
                order = np.argsort(-fused, 1, kind='stable')
                hc = np.take_along_axis(np.take_along_axis(R[t]['pos'], cand, 1), order, 1)
                h1.append(hc[:, 0]); h5.append(hc[:, :5].any(1))
            h1, h5 = np.concatenate(h1), np.concatenate(h5)
            key = f'{s1}+{meth}'
            out['models'][key] = metrics(h1, h5, cond, valid)
            out['models'][key]['hits'] = h1.astype(np.int8)
    return out


def paired_ci(a, b, valid, n=2000, seed=0):
    """Bootstrap CI of the paired difference in R@1 (b - a), resampling queries."""
    rng = np.random.default_rng(seed)
    a, b = a[valid].astype(float), b[valid].astype(float)
    d = b - a
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1) * 100
    return float(d.mean() * 100), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == '__main__':
    vs = sorted(os.path.basename(f)[:-3] for f in glob.glob(f'{RB}/*.pt')
                if not os.path.basename(f).startswith(('rerank_', 'partial_')) and not f.endswith('_sp.pt')
                and os.path.basename(f) != 'summary.pt')
    S = {v: variant_summary(v) for v in vs}
    base = S.get('roll_0')
    js = {}
    for v, s in S.items():
        js[v] = {'n_queries': s['n_queries'], 'n_no_positive': s['n_no_positive'], 'models': {}, 'groups': {}}
        valid = None
        for n, m in s['models'].items():
            js[v]['models'][n] = {k: round(float(x), 2) for k, x in m.items() if k != 'hits'}
            if base is not None and n in base['models'] and len(base['models'][n]['hits']) == len(m['hits']):
                ok = s['valid'] & base['valid']
                js[v]['models'][n]['dR1_vs_roll0'] = [round(x, 2) for x in paired_ci(base['models'][n]['hits'], m['hits'], ok)]
        for g, names in GROUPS.items():
            vals = [s['models'][n] for n in names if n in s['models']]
            if vals:
                js[v]['groups'][g] = {k: [round(float(np.mean([x[k] for x in vals])), 2),
                                          round(float(np.std([x[k] for x in vals])), 2)]
                                      for k in ('R@1', 'R@5', 'C0', 'hard')}
                js[v]['groups'][g]['n_seeds'] = len(vals)
    json.dump(js, open(f'{RB}/summary.json', 'w'), indent=1)
    for v in vs:
        g = js[v]['groups']
        line = '  '.join(f"{k}:{g[k]['R@1'][0]:.1f}" for k in g)
        fus = '  '.join(f"{n.split('+')[1]}:{m['R@1']:.1f}" for n, m in js[v]['models'].items() if '+' in n)
        print(f'{v:12s} {line}  | fused {fus}  | no-pos {js[v]["n_no_positive"]}')
