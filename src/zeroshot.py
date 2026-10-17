"""Evaluate zero-shot terrestrial baselines (CosPlace, EigenPlaces, DINOv2-SALAD) on LunarSim."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import load_eval, recall_from_sim, summarize

out = {}
for split in sys.argv[1:] or ['test']:
    T = load_eval(split)
    for key, name in [('cosplace', 'CosPlace-R50'), ('eigen', 'EigenPlaces-R50'), ('salad_zs', 'DINOv2-SALAD')]:
        hits, conds = [], []
        for Tt in T:
            D = Tt['F'][key].float().numpy()
            h, _ = recall_from_sim(D[Tt['ndb']:] @ D[:Tt['ndb']].T, Tt['pos'])
            hits.append(h); conds.append(Tt['qcond'])
        r = summarize(hits, conds)
        for N in (1, 5, 10):
            r[f'hits_R{N}'] = np.concatenate([h[N] for h in hits]).astype(int).tolist()
        out[f'{split}/{name}'] = r
        print(split, name, 'R@1 %.1f R@5 %.1f R@10 %.1f' % (r['R@1'], r['R@5'], r['R@10']),
              ' per-cond R@1:', ' '.join('%.0f' % r[f'R@1_{c}'] for c in
                                       ['C0_same', 'C1_el15', 'C2_el6', 'C3_el60', 'C4_az+90', 'C5_az+180', 'C6_az+180_el8']))
os.makedirs(paths.RESULTS, exist_ok=True)
json.dump(out, open(paths.RESULTS + '/zeroshot_' + '_'.join(sys.argv[1:] or ['test']) + '.json', 'w'))
