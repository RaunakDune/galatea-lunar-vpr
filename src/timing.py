"""Run-time measurements on this CPU (2 threads). Run when the machine is otherwise idle."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
torch.set_num_threads(2); torch.set_grad_enabled(False)
import nets
from aggregators import OTHead, NetVLADHead, GeMHead
from ot_utils import log_sinkhorn_dustbin, mutual_matches

res = {}
def tm(fn, n=20, warm=3):
    for _ in range(warm): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

x = torch.randn(1, 3, 210, 280)
dn = nets.DinoSalad()
res['backbone_ms'] = tm(lambda: dn.forward_tokens(x), n=10)
f, t = dn.forward_tokens(x)
for name, h in [('gem', GeMHead()), ('netvlad', NetVLADHead()), ('salad', OTHead(mode='balanced')), ('galatea', OTHead(mode='learn'))]:
    h.eval(); res[f'head_{name}_ms'] = tm(lambda: h((f, t)), n=50)
# local matching per pair (p = p' = 200 keypoints, typical)
p = 200
k1, k2 = torch.rand(p, 2) * 300, torch.rand(p, 2) * 300
d1 = torch.nn.functional.normalize(torch.randn(p, 256), dim=1); d2 = torch.nn.functional.normalize(torch.randn(p, 256), dim=1)
def uot():
    return mutual_matches(log_sinkhorn_dustbin(d1 @ d2.T / 0.05, 0.75 / 0.05))
def fgw():
    M = 1 - d1 @ d2.T
    C1 = torch.cdist(k1, k1); C1 /= C1.mean(); C2 = torch.cdist(k2, k2); C2 /= C2.mean()
    T = log_sinkhorn_dustbin(-M / 0.05, -0.25 / 0.05)
    for _ in range(5):
        P = T[:-1, :-1]; P = P / P.sum()
        G = (C1 ** 2 @ P.sum(1))[:, None] + (C2 ** 2 @ P.sum(0))[None, :] - 2 * C1 @ P @ C2.T
        T = log_sinkhorn_dustbin(-(0.7 * M + 0.3 * G) / 0.05, -0.25 / 0.05)
    return mutual_matches(T)
lg = nets.lightglue()
s = torch.tensor([[320., 240.]])
def lgf():
    return lg({'image0': {'keypoints': k1[None], 'descriptors': d1[None], 'image_size': s},
               'image1': {'keypoints': k2[None], 'descriptors': d2[None], 'image_size': s}})
res['pair_uot_ms'] = tm(uot, n=20); res['pair_fgw_ms'] = tm(fgw, n=20); res['pair_lightglue_ms'] = tm(lgf, n=20)
sp = nets.superpoint()
g = torch.rand(1, 1, 240, 320)
res['superpoint_ms'] = tm(lambda: sp({'image': g}), n=10)
print(json.dumps({k: round(v, 1) for k, v in res.items()}, indent=1))
json.dump(res, open(os.path.join(paths.RESULTS, 'timing.json'), 'w'), indent=1)
