"""Summarise apollo_r2r.py output: Recall@1/5 per model family and database type, 95% intervals
from resampling query pans, and paired differences between the DESPINA and shaded databases.

usage: python apollo_r2r_report.py r2r_a15.pt
"""
import sys, re, json, numpy as np, torch
from collections import defaultdict

r = torch.load(sys.argv[1], weights_only=False)
vst = np.array([v['station'] for v in r['views']])
qst, qpan = r['q_station'], np.array(r['q_pan'])
rng = np.random.default_rng(0)
upans = np.unique(qpan)
by = {p: np.where(qpan == p)[0] for p in upans}
boot = [np.concatenate([by[p] for p in rng.choice(upans, len(upans))]) for _ in range(2000)]


def ci(x):
    s = np.array([x[b].mean() for b in boot]) * 100
    return [float(x.mean() * 100), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


fam = defaultdict(list)
for n in r['db'][next(iter(r['db']))]:
    fam[re.sub(r'_s\d+$', '', n)].append(n)


def hits(db, names, k):
    return np.mean([(vst[r['db'][db][n]['cand'][:, :k]] == qst[:, None]).any(1) for n in names], 0)


out = {'n_queries': int(len(qst)), 'n_query_pans': int(len(upans)), 'n_stations': int(len(set(vst))),
       'n_views': int(len(vst)), 'chance_R1': float(np.mean([(vst == s).mean() for s in qst]) * 100),
       'families': {}}
for f, names in fam.items():
    e = {'seeds': len(names)}
    for db in r['db']:
        e[db] = {'R1': ci(hits(db, names, 1)), 'R5': ci(hits(db, names, 5))}
    if {'despina', 'shaded'} <= set(r['db']):
        e['despina_minus_shaded_R1'] = ci(hits('despina', names, 1) - hits('shaded', names, 1))
    out['families'][f] = e
    print(f'{f:22s} x{len(names)} ' + '  '.join(f'{db} R@1 {e[db]["R1"][0]:5.1f} [{e[db]["R1"][1]:4.1f},{e[db]["R1"][2]:4.1f}]'
                                                for db in r['db']))
print(f'chance {out["chance_R1"]:.1f}%, {out["n_queries"]} queries, {out["n_query_pans"]} pans')
json.dump(out, open(sys.argv[1].replace('.pt', '_summary.json'), 'w'), indent=1)
