"""Recall on the Apollo panorama benchmark after stage-2 re-ranking (apollo_rerank.py scores).

For every model family of apollo_eval.py the top-10 frames are re-ordered by the fusion
s_g + beta * s / (s + kappa). There is no real validation set, so (beta, kappa) come from two sources:
  sim   the values tuned on the LunarSim validation scene for the frozen reference head (results/
        rerank_summary.json and the CLIPPER selection), transferred unchanged;
  lomo  leave one mission out: tuned on the queries of the other two of Apollo 15-17 (same grid and
        criterion as the simulation, R@1 + 0.5 R@5, averaged over the family's seeds).
Families with several seeds are averaged per query over seeds; R@1 and the paired difference against
the family's own global ranking are bootstrapped over query pans (2000 resamples), as apollo_report.py.
Writes <out_dir>/apollo_rerank_summary.json.
usage: python apollo_rerank_report.py --rr apollo_rr_mnn-fgw-lg.pt,apollo_rr_clipper16-clipper32.pt \
          --sim_params ../../results/rerank_summary.json [--clipper_params '{"clipper16": [1, 80]}']
"""
import os, re, sys, json, argparse, numpy as np, torch
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--res', default=os.path.expanduser('~/apollo_bench/apollo_center.pt'))
ap.add_argument('--out_dir', default=os.path.expanduser('~/apollo_bench'))
ap.add_argument('--rr', required=True)
ap.add_argument('--sim_params', required=True)
ap.add_argument('--extra_params', default='{}', help='JSON {scorer: [beta, kappa]} for scorers not in sim_params')
ap.add_argument('--families', default='salad_zs,gem,netvlad,salad,galatea,rft_netvlad,rft_galatea')
args = ap.parse_args()
BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (5, 10, 20, 40, 80, 160, 320, 640)

r = torch.load(args.res, weights_only=False)
pan, st, mi = r['pan'], r['station'], r['mission']
pos = (st[:, None] == st[None]) & (pan[:, None] != pan[None])
q = np.where(pos.any(1) & np.isin(mi, [15, 16, 17]))[0]          # the 'main' subset of apollo_report.py
S = {}
for f in args.rr.split(','):
    d = torch.load(os.path.join(args.out_dir, f), weights_only=False)
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(d['pairs'])}
    for k, v in d.items():
        if k not in ('pairs', 'K'):
            S[k] = (idx, v)
K = 10
sim_p = {k: (v['beta'], v['kappa']) for k, v in json.load(open(args.sim_params)).items()}
sim_p.update({k: tuple(v) for k, v in json.loads(args.extra_params).items()})
fam = defaultdict(list)
for n in r['models']:
    fam[re.sub(r'_s\d+$', '', n)].append(n)


def local(n, scorer):
    idx, v = S[scorer]
    c = r['models'][n]['cand'][q, :K]
    return np.array([[v[idx[(int(qi), int(cc))]] for cc in row] for qi, row in zip(q, c)], np.float64)


def hits(n, scorer=None, bk=None):
    c = r['models'][n]['cand'][q, :K]
    g = r['models'][n]['sim'][q, :K].astype(np.float64)
    if scorer is None:
        sc = g
    elif bk == 'local':
        sc = LOC[(n, scorer)] + 1e-3 * g                 # local score, ties broken by the global one
    else:
        sc = g + bk[0] * LOC[(n, scorer)] / (LOC[(n, scorer)] + bk[1])
    o = np.argsort(-sc, 1, kind='stable')
    h = np.take_along_axis(np.take_along_axis(pos[q], c, 1), o, 1)
    return h[:, 0].astype(float), h[:, :5].any(1).astype(float)


scorers = [s for s in S if s in sim_p]
LOC = {(n, s): local(n, s) for f in args.families.split(',') for n in fam[f] for s in scorers}
qpan, qmi = pan[q], mi[q]
upans = np.unique(qpan)
by = {p: np.where(qpan == p)[0] for p in upans}
rng = np.random.default_rng(0)
boot = [np.concatenate([by[p] for p in rng.choice(upans, len(upans))]) for _ in range(2000)]


def ci(x):
    s = np.array([x[b].mean() for b in boot]) * 100
    return [round(float(x.mean() * 100), 4), round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def lomo(names, s):
    """Per-query hit@1 with (beta, kappa) tuned on the other two missions."""
    h1 = np.zeros(len(q))
    chosen = {}
    for m in (15, 16, 17):
        tr, te = qmi != m, qmi == m
        best = None
        for b in BET:
            for k in KAP:
                hh = [hits(n, s, (b, k)) for n in names]
                key = np.mean([x[0][tr].mean() + 0.5 * x[1][tr].mean() for x in hh])
                if best is None or key > best[0]:
                    best = (key, b, k)
        h1[te] = np.mean([hits(n, s, best[1:])[0] for n in names], 0)[te]
        chosen[m] = best[1:]
    return h1, chosen


out = {'n_queries': int(len(q)), 'n_query_pans': int(len(upans)), 'K': K, 'params_sim': sim_p, 'families': {}}
for f in args.families.split(','):
    names = fam[f]
    g1 = np.mean([hits(n)[0] for n in names], 0)
    g5 = np.mean([hits(n)[1] for n in names], 0)
    oracle_top10 = np.mean([np.take_along_axis(pos[q], r['models'][n]['cand'][q, :K], 1).any(1) for n in names], 0)
    res = {'seeds': len(names), 'global_R1': ci(g1), 'global_R5': ci(g5), 'top10_contains_positive': ci(oracle_top10),
           'scorers': {}}
    for s in scorers:
        f1 = np.mean([hits(n, s, sim_p[s])[0] for n in names], 0)
        l1 = np.mean([hits(n, s, 'local')[0] for n in names], 0)
        lo1, ch = lomo(names, s)
        res['scorers'][s] = {'local_R1': ci(l1), 'fused_sim_R1': ci(f1), 'd_fused_sim': ci(f1 - g1),
                             'fused_lomo_R1': ci(lo1), 'd_fused_lomo': ci(lo1 - g1), 'lomo_params': ch}
    out['families'][f] = res
    print(f'{f:12s} global {res["global_R1"][0]:5.1f}  top10 {res["top10_contains_positive"][0]:5.1f}  | ' +
          '  '.join(f'{s}: {v["fused_sim_R1"][0]:.1f} ({v["d_fused_sim"][0]:+.1f} [{v["d_fused_sim"][1]:+.1f},{v["d_fused_sim"][2]:+.1f}]) '
                    f'lomo {v["fused_lomo_R1"][0]:.1f}' for s, v in res['scorers'].items()), flush=True)
json.dump(out, open(os.path.join(args.out_dir, 'apollo_rerank_summary.json'), 'w'), indent=1)
