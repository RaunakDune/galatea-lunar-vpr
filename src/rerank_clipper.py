"""CLIPPER as a stage-2 scorer (rerank_lib.clipper_matches), with image-plane or 3-D distances.

Re-scores the same top-10 stage-1 candidates as the existing re-ranking files. The score is the number
of associations CLIPPER selects (key clipper<2d|3d>_e<eps>), and the MAGSAC++ inliers among them
(_ransac). 3-D distances use the oracle keypoint points of geom3d.py.
Writes <RESULTS>/rerank/<stage1>_<split>_clipper.npy (base file keys plus the new ones), per terrain.
usage: python rerank_clipper.py --split val --eps2d 4,8,16 --eps3d 0.5,1,2 [--procs 12]
"""
import os, sys, glob, time, argparse, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from rerank_lib import topk_kp, topk_idx, clipper_matches, ransac_inliers

ap = argparse.ArgumentParser()
ap.add_argument('--stage1', default='galatea_s0')
ap.add_argument('--split', default='')
ap.add_argument('--base_tag', default='mnn-uot-fgw-lg_a0.3_c0.25')
ap.add_argument('--robust', default='')
ap.add_argument('--robust_dir', default=os.path.join(paths.RESULTS, 'robust'))
ap.add_argument('--eps2d', default='4,8,16')
ap.add_argument('--eps3d', default='0.5,1,2')
ap.add_argument('--maxkp', type=int, default=384)
ap.add_argument('--eps', type=float, default=0.05)
ap.add_argument('--cbin', type=float, default=0.25)
ap.add_argument('--procs', type=int, default=12)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--chunk', type=int, default=150, help='queries per parallel job')
args = ap.parse_args()
G3 = os.path.join(paths.RESULTS, 'geom3d')
RR = os.path.join(paths.RESULTS, 'rerank')
KEYS = [('2d', float(e)) for e in args.eps2d.split(',') if e] + [('3d', float(e)) for e in args.eps3d.split(',') if e]
LIM = f'_lim{args.limit}' if args.limit else ''


def prep(sp, P):
    idx = topk_idx(sp, args.maxkp)
    k, _, d = topk_kp(sp, args.maxkp)
    return k, d, P[idx]


def score(q, c):
    kq, dq, Pq = q
    kc, dc, Pc = c
    out = {}
    for m, e in KEYS:
        mm = clipper_matches(kq, dq, kc, dc, *((Pq, Pc) if m == '3d' else (None, None)), eps=e)
        key = f'clipper{m}_e{e:g}'
        out[key] = len(mm)
        out[key + '_ransac'] = ransac_inliers(kq[mm[:, 0]].numpy(), kc[mm[:, 1]].numpy()) if len(mm) else 0
    return out


def run(cand, qitems, dbitems, lo, hi):
    names = [f'clipper{m}_e{e:g}' + s for m, e in KEYS for s in ('', '_ransac')]
    sc = {n: np.zeros((hi - lo, cand.shape[1]), np.float32) for n in names}
    for qi in range(lo, hi):
        for r, c in enumerate(cand[qi]):
            for n, v in score(qitems[qi], dbitems[c]).items():
                sc[n][qi - lo, r] = v
    return sc


def merge(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def chunks(n):
    n = min(n, args.limit) if args.limit else n
    return [(lo, min(lo + args.chunk, n)) for lo in range(0, n, args.chunk)]


def job_split(spec):
    tn, lo, hi = spec
    torch.set_num_threads(1)
    tag = 'clipper-' + '-'.join(f'{m}e{e:g}' for m, e in KEYS)
    part = f'{RR}/partialclip_{args.stage1}_{args.split}_{tag}{LIM}_{tn}_{lo:05d}.npy'
    if os.path.exists(part):
        return part
    t0 = time.time()
    base = np.load(f'{RR}/{args.stage1}_{args.split}_{args.base_tag}.npy', allow_pickle=True).item()[tn]
    F = torch.load(f'{paths.DATA}/{args.split}/{tn}/feats.pt', weights_only=False, mmap=True)
    Pk = torch.load(f'{G3}/{args.split}_{tn}.pt')
    ndb = len(F['sp']) - len(base['cand'])
    dbitems = [prep(F['sp'][i], Pk[i]) for i in range(ndb)]
    qitems = {qi: prep(F['sp'][ndb + qi], Pk[ndb + qi]) for qi in range(lo, hi)}
    sc = run(base['cand'], qitems, dbitems, lo, hi)
    np.save(part, sc, allow_pickle=True)
    return f'{args.split} {tn} {lo}-{hi} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    tag = 'clipper-' + '-'.join(f'{m}e{e:g}' for m, e in KEYS)
    if args.split:
        tns = [os.path.basename(d) for d in sorted(glob.glob(f'{paths.DATA}/{args.split}/t*'))]
        nq = {tn: len(np.load(f'{RR}/{args.stage1}_{args.split}_{args.base_tag}.npy', allow_pickle=True).item()[tn]['cand'])
              for tn in tns}
        specs = [(tn, lo, hi) for tn in tns for lo, hi in chunks(nq[tn])]
        with Pool(args.procs) as p:
            for r in p.imap_unordered(job_split, specs):
                print(r, flush=True)
        base = np.load(f'{RR}/{args.stage1}_{args.split}_{args.base_tag}.npy', allow_pickle=True).item()
        out = f'{RR}/{args.stage1}_{args.split}_clipper.npy'
        store = np.load(out, allow_pickle=True).item() if os.path.exists(out) else {}
        for tn in tns:
            sc = merge([np.load(f'{RR}/partialclip_{args.stage1}_{args.split}_{tag}{LIM}_{tn}_{lo:05d}.npy',
                                allow_pickle=True).item() for lo, _ in chunks(nq[tn])])
            st = store.setdefault(tn, {k: base[tn][k] for k in base[tn]})
            st.update(sc)
        if not args.limit:
            np.save(out, store, allow_pickle=True)
            print('saved', out)
    print('DONE')
