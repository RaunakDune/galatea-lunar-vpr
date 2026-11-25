"""How well do the FGW structure matrices agree on true keypoint correspondences?

For the validation queries under the database Sun (C0) and their own database image, a pair of top-384
SuperPoint keypoints is a true correspondence if their oracle 3-D points (geom3d.py) are mutual nearest
neighbours within 0.3 m + 1% of the range, and closer than 2.5 km. For every structure mode the
distortion is the mean |C_q(i, i') - C_db(j, j')| over pairs of correspondences, divided by the mean of C_q.
Writes <RESULTS>/fgw3d_diag.json.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from rerank_lib import topk_idx, structure

d = f'{paths.DATA}/val/t901'
F = torch.load(f'{d}/feats.pt', weights_only=False, mmap=True)
P = torch.load(f'{paths.RESULTS}/geom3d/val_t901.pt')
meta = json.load(open(f'{d}/meta.json')); ndb = len(meta['db'])
modes = ['2d', '3d', '3dlog', '3dclip', '3dlogfix']
dist = {m: [] for m in modes}; ncorr, nkp = [], []
for qi, m in enumerate(meta['q']):
    if m['cond'] != 'C0_same':
        continue
    a, b = ndb + qi, m['place']
    ia, ib = topk_idx(F['sp'][a], 384), topk_idx(F['sp'][b], 384)
    xa, xb = P[a][ia].double(), P[b][ib].double()
    D = torch.cdist(xa, xb); j = D.argmin(1); i = torch.arange(len(j)); back = D.argmin(0)
    rng = torch.hypot(xa[:, 0] - m['x'], xa[:, 1] - m['y'])
    ok = (back[j] == i) & (D[i, j] < 0.3 + 0.01 * rng) & (rng < 2500)
    I, J = i[ok], j[ok]
    ncorr.append(len(I)); nkp.append(min(len(ia), len(ib)))
    if len(I) < 10:
        continue
    ka, kb = F['sp'][a]['kp'].float()[ia], F['sp'][b]['kp'].float()[ib]
    for mode in modes:
        if mode == '2d':
            Ca, Cb = structure(ka), structure(kb)
        else:
            Ca, Cb = structure((xa - xa.mean(0)).float(), mode), structure((xb - xb.mean(0)).float(), mode)
        t = torch.triu_indices(len(I), len(I), 1)
        e = (Ca[I][:, I] - Cb[J][:, J])[t[0], t[1]].abs()
        dist[mode].append((e.mean() / Ca.mean()).item())
out = {'n_queries': len(ncorr), 'median_keypoints': float(np.median(nkp)), 'median_true_corr': float(np.median(ncorr)),
       'median_corr_fraction': float(np.median(np.array(ncorr) / np.array(nkp))),
       'distortion_median': {k: round(float(np.median(v)), 4) for k, v in dist.items()}}
json.dump(out, open(f'{paths.RESULTS}/fgw3d_diag.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
