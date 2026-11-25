"""Boulder segments as local evidence on the Apollo panoramas (exploratory; after LunarLoc).

  --stage seg    FastSAM (FastSAM-x, Ultralytics) on every frame, with the crop of apollo_eval.py (one
                 GPU). Masks are filtered as in LunarLoc, with the size limits in pixels because the
                 frames have no depth: no contact with a 5% border margin, centroid in the lower two thirds
                 of the frame, area 0.05-5% of the frame, eigenvalue ratio of the mask's 2-D covariance
                 <= 3 and compactness 4 pi A / P^2 >= 0.5.          -> <out_dir>/apollo_boulders.pt
  --stage score  CLIPPER on the boulder centroids (image plane, 320 x 240 coordinates, all-to-all
                 associations) for every pair scored by apollo_rerank.py; key boulder<eps>.
                                                                    -> <out_dir>/apollo_rr_boulders.pt
Fusion parameters (and eps) for these scores are tuned leave-one-mission-out by apollo_rerank_report.py.
"""
import os, sys, time, argparse, numpy as np, torch, cv2
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['seg', 'score'])
ap.add_argument('--res', default=os.path.expanduser('~/apollo_bench/apollo_center.pt'))
ap.add_argument('--img_dir', default=os.path.expanduser('~/apollo_bench/img'))
ap.add_argument('--out_dir', default=os.path.expanduser('~/apollo_bench'))
ap.add_argument('--weights', default=os.path.expanduser('~/fastsam/FastSAM-x.pt'))
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--eps', default='4,8,16,32')
ap.add_argument('--procs', type=int, default=16)
args = ap.parse_args()


def crop(name):
    im = Image.open(f'{args.img_dir}/{name}.png').convert('L')
    w, h = im.size
    hh = int(round(w * 3 / 4)); top = max(0, (h - hh) // 2)
    return np.array(im.crop((0, top, w, top + hh)))


def segments(masks, img):
    H, W = img.shape
    out = []
    mg = int(0.05 * min(H, W))
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        if xs.min() < mg or ys.min() < mg or xs.max() >= W - mg or ys.max() >= H - mg:
            continue
        a = len(xs) / (H * W)
        cy, cx = ys.mean(), xs.mean()
        if cy < H / 3 or not (5e-4 <= a <= 0.05):
            continue
        ev = np.linalg.eigvalsh(np.cov(np.stack([xs, ys])))
        elong = np.sqrt(ev[1] / max(ev[0], 1e-9))
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        P = sum(cv2.arcLength(c, True) for c in cs)
        comp = 4 * np.pi * len(xs) / max(P, 1) ** 2
        if elong > 3 or comp < 0.5:
            continue
        out.append([cx * 320 / W, cy * 240 / H, a, elong, comp, float(img[ys, xs].mean())])
    return np.array(out, np.float32).reshape(-1, 6)


def score_chunk(spec):
    lo, hi = spec
    torch.set_num_threads(1)
    from rerank_lib import clipper_matches
    B = torch.load(f'{args.out_dir}/apollo_boulders.pt', weights_only=False)['seg']
    P = np.load(f'{args.out_dir}/apollo_pairs.npy')[lo:hi]
    eps = [float(e) for e in args.eps.split(',')]
    out = {f'boulder{e:g}': np.zeros(len(P), np.float32) for e in eps}
    for n, (q, c) in enumerate(P):
        bq, bc = B[q], B[c]
        if len(bq) < 3 or len(bc) < 3:
            continue
        for e in eps:
            out[f'boulder{e:g}'][n] = len(clipper_matches(None, torch.ones(len(bq), 1), None, torch.ones(len(bc), 1),
                                                          P1=torch.from_numpy(bq[:, :2]), P2=torch.from_numpy(bc[:, :2]),
                                                          eps=e, topk=len(bc), simthr=-1))
    return lo, out


if __name__ == '__main__':
    r = torch.load(args.res, weights_only=False)
    if args.stage == 'seg':
        from ultralytics import FastSAM
        m = FastSAM(args.weights)
        seg, nraw = [], []
        t0 = time.time()
        for i, f in enumerate(r['frames']):
            img = crop(f)
            res = m(np.stack([img] * 3, -1), device=args.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose=False)[0]
            masks = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None else []
            seg.append(segments(masks, img)); nraw.append(len(masks))
            if i % 200 == 0:
                print(i, f'{time.time() - t0:.0f}s', flush=True)
        torch.save({'frames': r['frames'], 'seg': seg, 'n_raw': nraw}, f'{args.out_dir}/apollo_boulders.pt')
        print('masks per frame: median raw', np.median(nraw), 'kept', np.median([len(s) for s in seg]))
    else:
        P = np.load(f'{args.out_dir}/apollo_pairs.npy')
        specs = [(lo, min(lo + 2000, len(P))) for lo in range(0, len(P), 2000)]
        res = {}
        with Pool(args.procs) as p:
            for lo, o in p.imap_unordered(score_chunk, specs):
                res[lo] = o
        out = {'pairs': P, 'K': 10, **{k: np.concatenate([res[lo][k] for lo, _ in specs]) for k in res[0]}}
        torch.save(out, f'{args.out_dir}/apollo_rr_boulders.pt')
        print('DONE')
