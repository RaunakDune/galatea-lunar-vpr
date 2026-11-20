"""Summarise apollo_eval.py output.

Query subsets (a query is scored only when its station has another pan in its mission):
  main      : Apollo 15, 16, 17 (the missions with many stations; Apollo 11 has a single station and
              Apollo 12's queries are almost all LM pans, so both are near-trivial)
  main_noLM : the same, without queries from the LM station (the LM itself is visible in those)
  all       : every scored query except Apollo 11
Models with several seeds are averaged per query over seeds (the family mean), then R@1/R@5 are
bootstrapped over query pans (2000 resamples), with paired differences against a reference.

usage: python apollo_report.py apollo.pt [ref_model]
"""
import sys, re, json, numpy as np, torch
from collections import defaultdict

r = torch.load(sys.argv[1], weights_only=False)
ref = sys.argv[2] if len(sys.argv) > 2 else 'salad_zs'
pan, st, mi = r['pan'], r['station'], r['mission']
pos = (st[:, None] == st[None]) & (pan[:, None] != pan[None])
db = (mi[:, None] == mi[None]) & (pan[:, None] != pan[None])
scored = pos.any(1)
isLM = np.array([s.endswith('-LM') for s in st])
subsets = {'main': scored & np.isin(mi, [15, 16, 17]),
           'main_noLM': scored & np.isin(mi, [15, 16, 17]) & ~isLM,
           'all': scored & (mi != 11)}

fam = defaultdict(list)
for n in r['models']:
    fam[re.sub(r'_s\d+$', '', n)].append(n)

rng = np.random.default_rng(0)
out = {'subsets': {}}
for sname, sel in subsets.items():
    q = np.where(sel)[0]
    qpan = pan[q]
    upans = np.unique(qpan)
    by = {p: np.where(qpan == p)[0] for p in upans}
    boot = [np.concatenate([by[p] for p in rng.choice(upans, len(upans))]) for _ in range(2000)]
    chance = (pos[q].sum(1) / db[q].sum(1)).mean() * 100

    def hits(names, k):
        return np.mean([np.take_along_axis(pos[q], r['models'][n]['cand'][q, :k], 1).any(1) for n in names], 0)

    def ci(x):
        s = np.array([x[b].mean() for b in boot]) * 100
        return [float(x.mean() * 100), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]

    stations = sorted(set(st[q]))
    print(f'\n== {sname}: {len(q)} queries, {len(upans)} query pans, {len(stations)} query stations; '
          f'chance R@1 {chance:.1f}%')
    h_ref = hits(fam[ref] if ref in fam else [ref], 1)
    res = {'n_queries': int(len(q)), 'n_query_pans': int(len(upans)), 'n_query_stations': len(stations),
           'chance_R1': float(chance), 'families': {}}
    for f, names in fam.items():
        h1, h5 = hits(names, 1), hits(names, 5)
        a, b, d = ci(h1), ci(h5), ci(h1 - h_ref)
        per = {int(m): float(h1[mi[q] == m].mean() * 100) for m in sorted(set(mi[q]))}
        res['families'][f] = {'seeds': len(names), 'R1': a, 'R5': b, 'dR1_vs_ref': d, 'R1_mission': per}
        print(f'{f:22s} x{len(names)}  R@1 {a[0]:5.1f} [{a[1]:4.1f},{a[2]:4.1f}]  R@5 {b[0]:5.1f} '
              f'[{b[1]:4.1f},{b[2]:4.1f}]  d vs {ref}: {d[0]:+5.1f} [{d[1]:+.1f},{d[2]:+.1f}]  '
              + ' '.join(f'A{m}:{v:4.1f}' for m, v in per.items()))
    out['subsets'][sname] = res
json.dump(out, open(sys.argv[1].replace('.pt', '_summary.json'), 'w'), indent=1)
