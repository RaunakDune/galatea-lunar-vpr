"""Partial FGW re-ranking with a 3-D structural term (oracle keypoint depth from geom3d.py).

Re-scores the same top-10 stage-1 candidates as the existing re-ranking files, so the only change from
the image-plane FGW is the structure matrices C1, C2 (rerank_lib.structure, modes 3d and 3dlog).

Benchmark splits: candidates from <RESULTS>/rerank/<stage1>_<split>_<tag>.npy; writes
  <RESULTS>/rerank/<stage1>_<split>_fgw3d.npy   with keys fgw<mode>_a<alpha>[_ransac] per terrain.
Robustness variants (--robust): candidates from <ROBUST>/<variant>.pt; writes
  <ROBUST>/rerank3d_<stage1>_<variant>.pt
usage: python rerank3d.py --split val --modes 3d,3dlog --alphas 0.3,0.6 [--procs 12]
       python rerank3d.py --robust heading_15,... --robust_dir DIR --modes 3dlog --alphas 0.3
"""
import os, sys, glob, time, argparse, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from rerank_lib import topk_kp, topk_idx, structure, fgw_match, mutual_matches, ransac_inliers

ap = argparse.ArgumentParser()
ap.add_argument('--stage1', default='galatea_s0')
ap.add_argument('--split', default='')
ap.add_argument('--base_tag', default='mnn-uot-fgw-lg_a0.3_c0.25')
ap.add_argument('--robust', default='')
ap.add_argument('--robust_dir', default=os.path.join(paths.RESULTS, 'robust'))
ap.add_argument('--modes', default='3d,3dlog')
ap.add_argument('--alphas', default='0.3')
ap.add_argument('--maxkp', type=int, default=384)
ap.add_argument('--eps', type=float, default=0.05)
ap.add_argument('--cbin', type=float, default=0.25)
ap.add_argument('--procs', type=int, default=12)
ap.add_argument('--limit', type=int, default=0)
args = ap.parse_args()
G3 = os.path.join(paths.RESULTS, 'geom3d')
RR = os.path.join(paths.RESULTS, 'rerank')
KEYS = [(m, float(a)) for m in args.modes.split(',') for a in args.alphas.split(',')]
LIM = f'_lim{args.limit}' if args.limit else ''


def prep(sp, P):
    idx = topk_idx(sp, args.maxkp)
    k, _, d = topk_kp(sp, args.maxkp)
    x = P[idx].double()
    x = (x - x.mean(0)).float()          # centre before float32 cdist (world coordinates are ~1e2 m)
    return k, d, {m: structure(x, m) for m in {m for m, _ in KEYS}}


def score(q, c):
    kq, dq, Cq = q
    kc, dc, Cc = c
    out = {}
    for m, a in KEYS:
        mm = mutual_matches(fgw_match(kq, dq, kc, dc, a, args.eps, cbin=args.cbin, C1=Cq[m], C2=Cc[m]))
        key = f'fgw{m}_a{a}'
        out[key] = len(mm)
        out[key + '_ransac'] = ransac_inliers(kq[mm[:, 0]].numpy(), kc[mm[:, 1]].numpy()) if len(mm) else 0
    return out


def run(cand, qitems, dbitems):
    nq = len(cand) if not args.limit else args.limit
    names = [f'fgw{m}_a{a}' + s for m, a in KEYS for s in ('', '_ransac')]
    sc = {n: np.zeros((nq, cand.shape[1]), np.float32) for n in names}
    for qi in range(nq):
        for r, c in enumerate(cand[qi]):
            for n, v in score(qitems[qi], dbitems[c]).items():
                sc[n][qi, r] = v
    return sc


def job_split(tn):
    torch.set_num_threads(1)
    tag = '-'.join(f'{m}a{a}' for m, a in KEYS)
    part = f'{RR}/partial3d_{args.stage1}_{args.split}_{tag}{LIM}_{tn}.npy'
    if os.path.exists(part):
        return part
    t0 = time.time()
    base = np.load(f'{RR}/{args.stage1}_{args.split}_{args.base_tag}.npy', allow_pickle=True).item()[tn]
    F = torch.load(f'{paths.DATA}/{args.split}/{tn}/feats.pt', weights_only=False, mmap=True)
    Pk = torch.load(f'{G3}/{args.split}_{tn}.pt')
    ndb = len(F['sp']) - len(base['cand'])
    items = [prep(s, p) for s, p in zip(F['sp'], Pk)]
    sc = run(base['cand'], items[ndb:], items[:ndb])
    np.save(part, sc, allow_pickle=True)
    return f'{args.split} {tn} {time.time() - t0:.0f}s'


def job_robust(spec):
    v, tn = spec
    torch.set_num_threads(1)
    tag = '-'.join(f'{m}a{a}' for m, a in KEYS)
    part = f'{args.robust_dir}/partial3d_{args.stage1}_{v}_{tag}_{tn}.npy'
    if os.path.exists(part):
        return part
    t0 = time.time()
    R = torch.load(f'{args.robust_dir}/{v}.pt', weights_only=False)[tn]
    sps = torch.load(f'{args.robust_dir}/{v}_sp.pt', weights_only=False)[tn]
    F = torch.load(f'{paths.DATA}/test/{tn}/feats.pt', weights_only=False, mmap=True)
    Pdb = torch.load(f'{G3}/test_{tn}.pt')
    Pq = torch.load(f'{G3}/robust_{v}_{tn}.pt')
    cand = R['models'][args.stage1]['cand'].astype(int)
    dbitems = [prep(F['sp'][i], Pdb[i]) for i in range(150)]
    qitems = [prep(s, p) for s, p in zip(sps, Pq)]
    sc = run(cand, qitems, dbitems)
    np.save(part, sc, allow_pickle=True)
    return f'{v} {tn} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    tag = '-'.join(f'{m}a{a}' for m, a in KEYS)
    if args.split:
        tns = [os.path.basename(d) for d in sorted(glob.glob(f'{paths.DATA}/{args.split}/t*'))]
        with Pool(min(args.procs, len(tns))) as p:
            for r in p.imap_unordered(job_split, tns):
                print(r, flush=True)
        base = np.load(f'{RR}/{args.stage1}_{args.split}_{args.base_tag}.npy', allow_pickle=True).item()
        out = f'{RR}/{args.stage1}_{args.split}_fgw3d.npy'
        store = np.load(out, allow_pickle=True).item() if os.path.exists(out) else {}
        for tn in tns:
            sc = np.load(f'{RR}/partial3d_{args.stage1}_{args.split}_{tag}{LIM}_{tn}.npy', allow_pickle=True).item()
            st = store.setdefault(tn, {k: base[tn][k] for k in base[tn]})
            st.update(sc)
        if not args.limit:
            np.save(out, store, allow_pickle=True)
        print('saved', out)
    if args.robust:
        vs = args.robust.split(',')
        specs = [(v, os.path.basename(d)) for v in vs for d in sorted(glob.glob(f'{paths.DATA}/robust/{v}/t*'))]
        with Pool(args.procs) as p:
            for r in p.imap_unordered(job_robust, specs):
                print(r, flush=True)
        for v in vs:
            out = f'{args.robust_dir}/rerank3d_{args.stage1}_{v}.pt'
            S = torch.load(out, weights_only=False) if os.path.exists(out) else {}
            for f in sorted(glob.glob(f'{args.robust_dir}/partial3d_{args.stage1}_{v}_{tag}_t*.npy')):
                S.setdefault(f.rsplit('_', 1)[1][:-4], {}).update(np.load(f, allow_pickle=True).item())
            torch.save(S, out)
            print('saved', out)
    print('DONE')
