"""Metric pose after retrieval on LunarSim-VPR (test scenes).

For every query, the top-1 database image after stage 2 (fusion of the frozen reference head with a
local scorer, (beta, kappa) from the validation scene) gives 2-D to 3-D correspondences: query keypoints
matched to database keypoints, whose 3-D positions come from the database image's rendered depth
(geom3d.py). The query camera pose then follows from PnP with RANSAC (OpenCV, EPnP + LM refinement) and
the known intrinsics. Errors are against the true query pose: camera-centre distance (m) and rotation
angle (deg). A query succeeds if both are within the thresholds (default 1 m, 5 deg); a query whose PnP
fails counts as a failure. Also reported: the same with the true place given (the database image of the
query's own place), which separates retrieval from pose estimation.

Camera model (lunarsim.render): pinhole, f = 160 / tan(35 deg) px, principal point (159.5, 72) with
the row of a pixel centre at r + 0.5; camera x right, y down, z forward; the camera looks along yaw
(world azimuth, counter-clockwise from +x) and is level.
Writes <RESULTS>/pose_summary.json.
usage: python pose_eval.py --matcher mnn [--procs 16]
"""
import os, sys, json, glob, time, argparse, numpy as np, torch, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from rerank_lib import topk_kp, topk_idx, mnn_matches
from lunarsim import Terrain
from geom3d import replay

ap = argparse.ArgumentParser()
ap.add_argument('--matchers', default='mnn,lg')
ap.add_argument('--scorer', default='mnn_ransac', help='local scorer of the fusion that picks the top-1')
ap.add_argument('--stage1_file', default='galatea_s0_test_mnn-uot-fgw-lg_a0.3_c0.25.npy')
ap.add_argument('--maxkp', type=int, default=384)
ap.add_argument('--tmax', type=float, default=1.0)
ap.add_argument('--rmax', type=float, default=5.0)
ap.add_argument('--procs', type=int, default=12)
ap.add_argument('--check', action='store_true', help='only verify the camera model by reprojection')
args = ap.parse_args()
W, H = 320, 240
F = (W / 2) / np.tan(np.radians(35.0))
K = np.array([[F, 0, W / 2 - 0.5], [0, F, H * 0.30], [0, 0, 1.0]])
MAXRANGE = 2500.0          # database points farther than this are sky placeholders (geom3d.SKY_RANGE)


def cam_pose(x, y, yaw, zc):
    """World-to-camera rotation R and camera centre c."""
    a = np.radians(yaw)
    fwd = np.array([np.cos(a), np.sin(a), 0.0]); right = np.array([np.sin(a), -np.cos(a), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R = np.stack([right, down, fwd])          # rows: camera axes in world coordinates
    return R, np.array([x, y, zc])


def pix(kp):
    """SuperPoint keypoint (col, row) -> image coordinates of the pinhole model."""
    k = kp.astype(np.float64).copy(); k[:, 1] += 0.5
    return k


_G = {}


def scene(tn):
    if tn not in _G:
        meta = json.load(open(f'{paths.DATA}/test/{tn}/meta.json'))
        Fe = torch.load(f'{paths.DATA}/test/{tn}/feats.pt', weights_only=False, mmap=True)
        P = torch.load(f'{paths.RESULTS}/geom3d/test_{tn}.pt')
        T = Terrain(meta['seed'])
        jit = replay(meta['seed'], len(meta['db']))
        ndb = len(meta['db'])
        poses = []
        for m in meta['db']:
            poses.append(cam_pose(m['x'], m['y'], m['yaw'], float(T.height(np.array([m['x']]), np.array([m['y']]))[0]) + 1.5))
        for m in meta['q']:
            x, y, yaw, ch = jit[m['place']]
            poses.append(cam_pose(x, y, yaw, float(T.height(np.array([x]), np.array([y]))[0]) + ch))
        kps = []
        for i, s in enumerate(Fe['sp']):
            idx = topk_idx(s, args.maxkp)
            k, _, d = topk_kp(s, args.maxkp)
            kps.append((k, d, P[i][idx].double().numpy()))
        _G.clear()
        _G[tn] = (meta, ndb, poses, kps)
    return _G[tn]


_LG = None


def match(m, kq, dq, kc, dc):
    global _LG
    if m == 'mnn':
        return mnn_matches(dq, dc)
    if _LG is None:
        import nets
        _LG = nets.lightglue()
    s = torch.tensor([[W, H]], dtype=torch.float32)
    with torch.no_grad():
        o = _LG({'image0': {'keypoints': kq[None], 'descriptors': dq[None], 'image_size': s},
                 'image1': {'keypoints': kc[None], 'descriptors': dc[None], 'image_size': s}})
    return o['matches'][0].numpy()


def solve(kq, Xc, ok_range):
    if ok_range.sum() < 6:
        return None
    o, rv, tv, inl = cv2.solvePnPRansac(Xc[ok_range], pix(kq[ok_range]), K, None, iterationsCount=2000,
                                        reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not o or inl is None or len(inl) < 6:
        return None
    rv, tv = cv2.solvePnPRefineLM(Xc[ok_range][inl[:, 0]], pix(kq[ok_range][inl[:, 0]]), K, None, rv, tv)
    R = cv2.Rodrigues(rv)[0]
    return R, (-R.T @ tv).ravel(), len(inl)


def err(Rq, cq, est):
    if est is None:
        return np.inf, np.inf, 0
    R, c, n = est
    dR = R @ Rq.T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return float(np.linalg.norm(c - cq)), float(ang), n


def job(spec):
    tn, lo, hi, top1 = spec
    torch.set_num_threads(1)
    meta, ndb, poses, kps = scene(tn)
    out = []
    for qi in range(lo, hi):
        kq, dq, _ = kps[ndb + qi]
        Rq, cq = poses[ndb + qi]
        rec = {'q': qi}
        for which, c in (('top1', int(top1[qi - lo])), ('true', int(meta['q'][qi]['place']))):
            kc, dc, Xc = kps[c]
            rng_c = np.linalg.norm(Xc - poses[c][1], axis=1)
            for m in args.matchers.split(','):
                mm = match(m, kq, dq, kc, dc)
                if len(mm) == 0:
                    rec[f'{which}_{m}'] = (np.inf, np.inf, 0); continue
                kqm, Xm = kq[mm[:, 0]].numpy(), Xc[mm[:, 1]]
                rec[f'{which}_{m}'] = err(Rq, cq, solve(kqm, Xm, rng_c[mm[:, 1]] < MAXRANGE))
        out.append(rec)
    return tn, lo, out


def check():
    """Reprojection of database keypoints' 3-D points into their own image with the true pose."""
    for tn in ('t1001',):
        meta, ndb, poses, kps = scene(tn)
        e = []
        for i in list(range(0, ndb, 10)) + list(range(ndb, ndb + 1050, 70)):
            k, _, X = kps[i]
            R, c = poses[i]
            ok = np.linalg.norm(X - c, axis=1) < MAXRANGE
            Xc = (X[ok] - c) @ R.T
            p = (Xc @ K.T); p = p[:, :2] / p[:, 2:]
            e.append(np.linalg.norm(p - pix(k.numpy()[ok]), axis=1))
        e = np.concatenate(e)
        print(f'{tn}: reprojection error of true 3-D points, median {np.median(e):.3f} px, 95% {np.percentile(e, 95):.3f} px')


if __name__ == '__main__':
    if args.check:
        check(); sys.exit()
    from analyze_rerank import tune
    RR = os.path.join(paths.RESULTS, 'rerank')
    sv = np.load(f'{RR}/{args.stage1_file.replace("_test_", "_val_")}', allow_pickle=True).item()
    st = np.load(f'{RR}/{args.stage1_file}', allow_pickle=True).item()
    b, k = tune(sv, 'val', args.scorer, (0.1, 0.25, 0.5, 1, 2, 4, 8), (5, 10, 20, 40, 80, 160, 320, 640))
    specs = []
    for tn in sorted(st):
        S = st[tn]
        ls = S[args.scorer].astype(float)
        fused = S['gsim'] + b * ls / (ls + k)
        top1 = S['cand'][np.arange(len(fused)), np.argmax(fused, 1)]
        g1 = S['cand'][:, 0]
        n = len(top1)
        for lo in range(0, n, 75):
            specs.append((tn, lo, min(lo + 75, n), top1[lo:lo + 75]))
    res = {}
    t0 = time.time()
    with Pool(args.procs) as p:
        for tn, lo, o in p.imap_unordered(job, specs):
            res.setdefault(tn, []).extend(o)
            print(tn, lo, f'{time.time() - t0:.0f}s', flush=True)
    from common import load_eval
    G = {os.path.basename(t['dir']): t for t in load_eval('test')}
    rows = []
    for tn in sorted(res):
        S = st[tn]; ls = S[args.scorer].astype(float); fused = S['gsim'] + b * ls / (ls + k)
        top1 = S['cand'][np.arange(len(fused)), np.argmax(fused, 1)]
        for r in sorted(res[tn], key=lambda r: r['q']):
            r['tn'] = tn; r['cond'] = G[tn]['qcond'][r['q']]; r['retrieved'] = bool(G[tn]['pos'][r['q'], top1[r['q']]])
            rows.append(r)
    summ = {'scorer': args.scorer, 'beta_kappa': [b, k], 'thresholds': [args.tmax, args.rmax], 'n': len(rows),
            'retrieval_R1': 100 * np.mean([r['retrieved'] for r in rows]), 'matchers': {}}
    HARD = ('C2_el6', 'C5_az+180', 'C6_az+180_el8')
    for m in args.matchers.split(','):
        d = {}
        for which in ('top1', 'true'):
            te = np.array([r[f'{which}_{m}'][0] for r in rows]); re_ = np.array([r[f'{which}_{m}'][1] for r in rows])
            ok = (te <= args.tmax) & (re_ <= args.rmax)
            cond = np.array([r['cond'] for r in rows]); retr = np.array([r['retrieved'] for r in rows])
            fin = np.isfinite(te)
            d[which] = {'success': 100 * ok.mean(), 'success_hard': 100 * ok[np.isin(cond, HARD)].mean(),
                        'success_C0': 100 * ok[cond == 'C0_same'].mean(),
                        'success_given_retrieved': 100 * ok[retr].mean() if which == 'top1' else None,
                        'pnp_fail': 100 * (~fin).mean(),
                        'median_t_success': float(np.median(te[ok])) if ok.any() else None,
                        'median_r_success': float(np.median(re_[ok])) if ok.any() else None,
                        'median_t_all': float(np.median(te)), 'median_r_all': float(np.median(re_)),
                        'success_by_cond': {c: 100 * ok[cond == c].mean() for c in sorted(set(cond))},
                        'success_at': {f'{t}m_{a}deg': 100 * ((te <= t) & (re_ <= a)).mean()
                                       for t, a in ((0.25, 2), (0.5, 5), (1, 5), (2, 10), (5, 10))}}
        summ['matchers'][m] = d
        print(m, json.dumps({w: {kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in d[w].items()
                                 if kk not in ('success_by_cond', 'success_at')} for w in d}))
    json.dump(summ, open(f'{paths.RESULTS}/pose_summary.json', 'w'), indent=1, default=float)
    torch.save(rows, f'{paths.RESULTS}/pose_rows.pt')
