"""Stage-2 re-ranking of the robustness variants (top-10 candidates of a frozen stage-1 head).

Uses the same scorers as rerank.py (rerank_lib) on SuperPoint keypoints: the variant's query keypoints
(from robust_eval.py) against the original database keypoints (feats.pt of the benchmark).
One job per (variant, terrain); run in parallel with --procs.
Output: <RESULTS>/robust/rerank_<stage1>_<variant>.pt
"""
import os, sys, glob, time, argparse, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from rerank_lib import topk_kp, ransac_inliers, mutual_matches, fgw_match, mnn_matches

ap = argparse.ArgumentParser()
ap.add_argument('--variants', required=True)
ap.add_argument('--stage1', default='galatea_s0')
ap.add_argument('--methods', default='mnn,fgw')
ap.add_argument('--maxkp', type=int, default=384)
ap.add_argument('--procs', type=int, default=30)
args = ap.parse_args()
OUT = os.path.join(paths.RESULTS, 'robust')
_DB = {}


def db_kps(tn):
    if tn not in _DB:
        F = torch.load(f'{paths.DATA}/test/{tn}/feats.pt', weights_only=False, mmap=True)
        ndb = 150
        _DB[tn] = [topk_kp(s, args.maxkp) for s in F['sp'][:ndb]]
    return _DB[tn]


def job(spec):
    v, tn = spec
    torch.set_num_threads(1)
    part = f'{OUT}/partial_rr_{args.stage1}_{v}_{tn}.npy'
    if os.path.exists(part):
        return part
    t0 = time.time()
    R = torch.load(f'{OUT}/{v}.pt', weights_only=False)[tn]
    sp = torch.load(f'{OUT}/{v}_sp.pt', weights_only=False)[tn]
    cand = R['models'][args.stage1]['cand'].astype(int)
    dbk = db_kps(tn)
    methods = args.methods.split(',')
    sc = {m: np.zeros(cand.shape, np.float32) for m in methods}
    sc.update({m + '_ransac': np.zeros(cand.shape, np.float32) for m in methods})
    for qi in range(len(cand)):
        kq, _, dq = topk_kp(sp[qi], args.maxkp)
        for r, c in enumerate(cand[qi]):
            kc, _, dc = dbk[c]
            for m in methods:
                mm = mnn_matches(dq, dc) if m == 'mnn' else mutual_matches(fgw_match(kq, dq, kc, dc))
                sc[m][qi, r] = len(mm)
                if len(mm):
                    sc[m + '_ransac'][qi, r] = ransac_inliers(kq[mm[:, 0]].numpy(), kc[mm[:, 1]].numpy())
    np.save(part, sc, allow_pickle=True)
    return f'{v} {tn} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    vs = args.variants.split(',')
    specs = [(v, os.path.basename(d)) for v in vs for d in sorted(glob.glob(f'{paths.DATA}/robust/{v}/t*'))
             if not os.path.exists(f'{OUT}/rerank_{args.stage1}_{v}.pt')]
    with Pool(args.procs) as p:
        for r in p.imap_unordered(job, specs):
            print(r, flush=True)
    for v in vs:
        parts = sorted(glob.glob(f'{OUT}/partial_rr_{args.stage1}_{v}_t*.npy'))
        if len(parts) == 3:
            torch.save({os.path.basename(f).rsplit('_', 1)[1][:-4]: np.load(f, allow_pickle=True).item() for f in parts},
                       f'{OUT}/rerank_{args.stage1}_{v}.pt')
    print('DONE')
