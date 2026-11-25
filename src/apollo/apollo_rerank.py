"""Stage-2 re-ranking on the Apollo panorama benchmark (apollo_eval.py output).

Stage 1 is unchanged: every model's top-10 frames of other pans of the same mission. Stage 2 scores each
(query, candidate) pair with the local scorers of the simulation (rerank_lib) on SuperPoint keypoints.
Images get the crop of apollo_eval.py (grey, 3:4 band through the frame centre) and are resized to the
LunarSim resolution, 320 x 240, so the keypoint scorers see the same image size as in simulation.
A pair is scored once even when several models retrieve it.

  --stage sp     SuperPoint for all frames (one GPU)           -> <out_dir>/apollo_sp.pt
  --stage score  local scores for the union of top-K pairs (CPU) -> <out_dir>/apollo_rr_<methods>.pt
                 methods: mnn, fgw, lg, clipper<eps> (image-plane CLIPPER with tolerance eps pixels)
usage: python apollo_rerank.py --stage sp --res ~/apollo_bench/apollo_center.pt --img_dir ~/apollo_bench/processed
       python apollo_rerank.py --stage score --methods mnn,fgw,lg --procs 16
"""
import os, sys, time, argparse, numpy as np, torch
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['sp', 'score'])
ap.add_argument('--res', default=os.path.expanduser('~/apollo_bench/apollo_center.pt'))
ap.add_argument('--img_dir', default=os.path.expanduser('~/apollo_bench/processed'))
ap.add_argument('--out_dir', default=os.path.expanduser('~/apollo_bench'))
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--K', type=int, default=10)
ap.add_argument('--methods', default='mnn,fgw,lg')
ap.add_argument('--maxkp', type=int, default=384)
ap.add_argument('--procs', type=int, default=16)
ap.add_argument('--chunk', type=int, default=500)
args = ap.parse_args()
SIZE = (320, 240)


def load(name):
    im = Image.open(f'{args.img_dir}/{name}.png').convert('L')
    w, h = im.size
    hh = int(round(w * 3 / 4))
    top = max(0, (h - hh) // 2)
    return np.array(im.crop((0, top, w, top + hh)).resize(SIZE, Image.BICUBIC))


def query_set(r):
    pan, st, mi = r['pan'], r['station'], r['mission']
    pos = (st[:, None] == st[None]) & (pan[:, None] != pan[None])
    return np.where(pos.any(1) & (mi != 11))[0]          # scored queries of every reported subset


def pairs(r):
    q = query_set(r)
    P = set()
    for m in r['models'].values():
        for qi in q:
            for c in m['cand'][qi, :args.K]:
                P.add((int(qi), int(c)))
    return np.array(sorted(P), dtype=np.int32)


_SP = None
_LG = None


def score_chunk(spec):
    lo, hi = spec
    global _SP, _LG
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    from rerank_lib import topk_kp, ransac_inliers, mutual_matches, fgw_match, mnn_matches
    methods = args.methods.split(',')
    if _SP is None:
        _SP = torch.load(f'{args.out_dir}/apollo_sp.pt', weights_only=False)
    if 'lg' in methods and _LG is None:
        import nets
        _LG = nets.lightglue()
    if any(m.startswith('clipper') for m in methods):
        from rerank_lib import clipper_matches
    P = np.load(f'{args.out_dir}/apollo_pairs.npy')[lo:hi]
    out = {m: np.zeros(len(P), np.float32) for m in methods}
    out.update({m + '_ransac': np.zeros(len(P), np.float32) for m in methods})
    cache = {}

    def kp(i):
        if i not in cache:
            cache[i] = topk_kp(_SP['sp'][i], args.maxkp)
        return cache[i]
    for n, (qi, c) in enumerate(P):
        kq, _, dq = kp(qi)
        kc, _, dc = kp(c)
        for m in methods:
            if m == 'mnn':
                mm = mnn_matches(dq, dc)
            elif m == 'fgw':
                mm = mutual_matches(fgw_match(kq, dq, kc, dc))
            elif m == 'lg':
                s = torch.tensor([SIZE], dtype=torch.float32)
                o = _LG({'image0': {'keypoints': kq[None], 'descriptors': dq[None], 'image_size': s},
                         'image1': {'keypoints': kc[None], 'descriptors': dc[None], 'image_size': s}})
                mm = o['matches'][0].numpy()
            elif m.startswith('clipper'):            # clipper<eps>: image-plane CLIPPER, eps in pixels
                mm = clipper_matches(kq, dq, kc, dc, eps=float(m[len('clipper'):]))
            out[m][n] = len(mm)
            if len(mm):
                out[m + '_ransac'][n] = ransac_inliers(kq[mm[:, 0]].numpy(), kc[mm[:, 1]].numpy())
    return lo, out


if __name__ == '__main__':
    r = torch.load(args.res, weights_only=False)
    if args.stage == 'sp':
        import nets
        torch.set_grad_enabled(False)
        dev = torch.device(args.device)
        net = nets.superpoint().to(dev)
        sp = []
        t0 = time.time()
        for f in r['frames']:
            g = torch.from_numpy(load(f)).float().div(255.)[None, None].to(dev)
            o = net({'image': g})
            sp.append({'kp': o['keypoints'][0].half().cpu(), 'sc': o['keypoint_scores'][0].half().cpu(),
                       'desc': o['descriptors'][0].half().cpu()})
        torch.save({'frames': r['frames'], 'sp': sp, 'size': SIZE}, f'{args.out_dir}/apollo_sp.pt')
        P = pairs(r)
        np.save(f'{args.out_dir}/apollo_pairs.npy', P)
        print(f'SuperPoint for {len(sp)} frames in {time.time() - t0:.0f}s; median keypoints '
              f'{np.median([len(s["sc"]) for s in sp]):.0f}; {len(P)} pairs')
    else:
        P = np.load(f'{args.out_dir}/apollo_pairs.npy')
        specs = [(lo, min(lo + args.chunk, len(P))) for lo in range(0, len(P), args.chunk)]
        res = {}
        t0 = time.time()
        with Pool(args.procs) as p:
            for lo, o in p.imap_unordered(score_chunk, specs):
                res[lo] = o
                print(f'{lo} [{time.time() - t0:.0f}s]', flush=True)
        keys = res[0].keys()
        out = {'pairs': P, 'K': args.K, **{k: np.concatenate([res[lo][k] for lo, _ in specs]) for k in keys}}
        torch.save(out, f'{args.out_dir}/apollo_rr_{args.methods.replace(",", "-")}.pt')
        print('DONE', len(P), 'pairs', f'{time.time() - t0:.0f}s')
