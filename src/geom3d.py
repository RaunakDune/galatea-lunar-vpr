"""3-D positions of the SuperPoint keypoints of the LunarSim test, val and robustness images.

For every image the renderer's ray caster is re-run without shading (geometry only) at the image's
camera pose, which gives the world point seen by each pixel. The keypoint's 3-D point is the point at
its nearest pixel. This is oracle depth: a rover would get it from stereo (as in LunarLoc) with noise
that grows with range, so results with these points are an upper bound for a 3-D structural term.

Camera poses: database images use the stored pose and the default camera height (1.5 m). The test and
val query poses are stored without their camera height, so the generator's random stream is replayed
(as in gen_robust.benchmark_poses) and checked against meta.json. The robustness variants store their
full pose (including pitch). Pixels that see the sky (no ground hit) get a point at SKY_RANGE metres
along their ray at camera height, except that keypoints within 3 px of the skyline take the point of the
nearest ground pixel (snap()).

Output: <RESULTS>/geom3d/<split>_<terrain>.pt (list aligned with feats.pt 'sp') and
        <RESULTS>/geom3d/robust_<variant>_<terrain>.pt (list aligned with <variant>_sp.pt[terrain]).
Each entry is a float32 tensor [n_kp, 3] in metres (world frame) for all keypoints of that image.
usage: python geom3d.py [--splits test,val] [--robust heading_15,...] [--robust_dir DIR] [--procs 12]
"""
import os, sys, json, glob, time, argparse, numpy as np, torch
from multiprocessing import Pool
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from lunarsim import Terrain, R_MOON
from gen_dataset import make_path
import tilt

SKY_RANGE = 3000.0
OUT = os.path.join(paths.RESULTS, 'geom3d')


def geometry(T, x0, y0, yaw_deg, W=320, Hh=240, hfov_deg=70.0, cam_h=1.5, cy_frac=0.30, near=None, cx=None):
    """World point [Hh, W, 3] seen by every pixel: the ray caster of lunarsim.render without shading."""
    f = (W / 2) / np.tan(np.radians(hfov_deg) / 2)
    cx, cy = (W / 2 - 0.5) if cx is None else cx, Hh * cy_frac
    alpha = np.arctan((np.arange(W) - cx) / f)
    theta = np.radians(yaw_deg) - alpha
    zcam = float(T.height(np.array([x0]), np.array([y0]))[0]) + cam_h
    d = [0.8]
    while d[-1] < 12000:
        d.append(d[-1] * 1.0045 + 0.012)
    if near is not None:
        d = sorted(near) + d
    d = np.array(d)
    S = len(d)
    X = x0 + d[:, None] * np.cos(theta)[None, :]
    Y = y0 + d[:, None] * np.sin(theta)[None, :]
    Hs = T.height(X.ravel(), Y.ravel()).reshape(S, W) - d[:, None] ** 2 / (2 * R_MOON)
    zc = d[:, None] * np.cos(alpha)[None, :]
    v = cy - f * (Hs - zcam) / zc
    vmin = np.minimum.accumulate(v, axis=0)
    rows = np.arange(Hh) + 0.5
    hit_s = np.empty((Hh, W), int)
    for c in range(W):
        hit_s[:, c] = np.searchsorted(-vmin[:, c], -rows, side='left')
    sky = (hit_s >= S) | (hit_s == 0)
    s1 = np.clip(hit_s, 1, S - 1)
    cc = np.broadcast_to(np.arange(W)[None, :], (Hh, W))
    v1, v0 = v[s1, cc], v[s1 - 1, cc]
    t = np.clip((v0 - rows[:, None]) / np.maximum(v0 - v1, 1e-9), 0, 1)
    dh = d[s1 - 1] + t * (d[s1] - d[s1 - 1])
    dh = np.where(sky, SKY_RANGE, dh)
    th = theta[None, :].repeat(Hh, 0)
    Pz = Hs[s1 - 1, cc] + t * (Hs[s1, cc] - Hs[s1 - 1, cc])
    Pz = np.where(sky, zcam, Pz)
    return np.stack([x0 + dh * np.cos(th), y0 + dh * np.sin(th), Pz], -1), sky


def geometry_tilted(T, x0, y0, yaw_deg, cam_h=1.5, roll=0.0, pitch=0.0):
    """As tilt.render_tilted (ss = 1): wide upright geometry resampled through the homography, nearest
    neighbour so that no point is interpolated across a depth edge."""
    R = tilt.rotation(roll, pitch)
    if np.allclose(R, np.eye(3)):
        return geometry(T, x0, y0, yaw_deg, cam_h=cam_h)
    W, H, CX, CY, F = tilt.W, tilt.H, tilt.CX, tilt.CY, tilt.F
    uu, rr = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    border = np.zeros((H, W), bool); border[[0, -1], :] = True; border[:, [0, -1]] = True
    xs, ys, d = tilt._rays(R, uu[border], rr[border])
    m = 3
    kx = m - int(np.floor(xs.min() + CX)); ky = m - int(np.floor(ys.min() + CY - 0.5))
    cx_w, cy_w = CX + kx, CY - 0.5 + 0.5 + ky
    Ww = int(np.ceil(xs.max() + CX)) + kx + m + 1
    Hw = int(np.ceil(ys.max() + cy_w - 0.5)) + m + 1
    hfov_w = 2 * np.degrees(np.arctan(Ww / (2 * F)))
    dep = np.arctan2(d[1], np.hypot(d[0], d[2])).max()
    near = tilt.NEAR if (dep > 0 and cam_h / np.tan(dep) < 1.0) else None
    Pw, skyw = geometry(T, x0, y0, yaw_deg, W=Ww, Hh=Hw, hfov_deg=hfov_w, cam_h=cam_h, cy_frac=cy_w / Hw,
                        near=near, cx=cx_w)
    X, Y, _ = tilt._rays(R, uu, rr)
    cols, rows = cx_w + X, cy_w + Y - 0.5
    P = np.stack([ndimage.map_coordinates(Pw[..., k], [rows, cols], order=0, mode='nearest') for k in range(3)], -1)
    sky = ndimage.map_coordinates(skyw.astype(float), [rows, cols], order=0, mode='constant', cval=1.0) > 0.5
    return P, sky


def snap(P, sky, max_px=3.0):
    """Keypoints on the skyline often round to a sky pixel; give every sky pixel within max_px of the
    ground the 3-D point of its nearest ground pixel (the ridge it belongs to)."""
    if not sky.any() or sky.all():
        return P
    dist, (ri, ci) = ndimage.distance_transform_edt(sky, return_indices=True)
    ok = sky & (dist <= max_px)
    P = P.copy()
    P[ok] = P[ri[ok], ci[ok]]
    return P


def kp3d(P, kp):
    kp = kp.float().numpy()
    c = np.clip(np.round(kp[:, 0]).astype(int), 0, P.shape[1] - 1)
    r = np.clip(np.round(kp[:, 1]).astype(int), 0, P.shape[0] - 1)
    return torch.from_numpy(P[r, c].astype(np.float32))


def replay(seed, n_places):
    """Database path and query jitter of gen_dataset.job (val/test), including the query camera height."""
    rng = np.random.default_rng(seed + 12345)
    P, Hd = make_path(rng, n_places)
    for _ in range(n_places):
        rng.normal(0, 0.012, (240, 320)); rng.normal(0, 1, (240, 320))
    q = []
    for i in range(n_places):
        r = 3.0 * np.sqrt(rng.random()); a = rng.uniform(0, 2 * np.pi)
        q.append((P[i, 0] + r * np.cos(a), P[i, 1] + r * np.sin(a), (Hd[i] + rng.normal(0, 10)) % 360,
                  rng.uniform(1.35, 1.7)))
    return q


def job_split(spec):
    split, d = spec
    tn = os.path.basename(d)
    out = f'{OUT}/{split}_{tn}.pt'
    if os.path.exists(out):
        return f'skip {out}'
    t0 = time.time()
    meta = json.load(open(f'{d}/meta.json'))
    F = torch.load(f'{d}/feats.pt', weights_only=False, mmap=True)
    seed, ndb = meta['seed'], len(meta['db'])
    T = Terrain(seed)
    jit = replay(seed, ndb)
    for m in meta['q']:
        x, y, yaw, _ = jit[m['place']]
        assert abs(m['x'] - x) < 1e-6 and abs(m['y'] - y) < 1e-6 and abs(m['yaw'] - yaw) < 1e-6, 'replay mismatch'
    names = list(F['names'])
    assert names[:ndb] == [m['file'] for m in meta['db']] and names[ndb:] == [m['file'] for m in meta['q']]
    cache = {}
    res, nsky = [], 0
    for k, sp in enumerate(F['sp']):
        if k < ndb:
            m = meta['db'][k]; key = ('db', k)
            pose = (m['x'], m['y'], m['yaw'], 1.5)
        else:
            m = meta['q'][k - ndb]; key = ('q', m['place'])
            pose = jit[m['place']]
        if key not in cache:
            P, sky = geometry(T, pose[0], pose[1], pose[2], cam_h=pose[3])
            cache[key] = (snap(P, sky), sky & (ndimage.distance_transform_edt(sky) > 3.0))
        P, sky = cache[key]
        res.append(kp3d(P, sp['kp']))
        kp = sp['kp'].float().numpy()
        nsky += int(sky[np.clip(np.round(kp[:, 1]).astype(int), 0, 239), np.clip(np.round(kp[:, 0]).astype(int), 0, 319)].sum())
    torch.save(res, out)
    return f'{split} {tn}: {len(res)} images, {nsky} keypoints >3 px into the sky, {time.time() - t0:.0f}s'


def job_robust(spec):
    v, d, rdir = spec
    tn = os.path.basename(d)
    out = f'{OUT}/robust_{v}_{tn}.pt'
    if os.path.exists(out):
        return f'skip {out}'
    t0 = time.time()
    meta = json.load(open(f'{d}/meta.json'))
    sps = torch.load(f'{rdir}/{v}_sp.pt', weights_only=False)[tn]
    assert len(sps) == len(meta['q'])
    T = Terrain(meta['seed'])
    cache, res = {}, []
    for m, sp in zip(meta['q'], sps):
        key = (round(m['x'], 9), round(m['y'], 9), round(m['yaw'], 9), m['cam_h'], m['roll'], m['pitch'])
        if key not in cache:
            cache[key] = snap(*geometry_tilted(T, m['x'], m['y'], m['yaw'], m['cam_h'], m['roll'], m['pitch']))
        res.append(kp3d(cache[key], sp['kp']))
    torch.save(res, out)
    return f'robust {v} {tn}: {len(res)} images, {len(cache)} poses, {time.time() - t0:.0f}s'


def _run(fs):
    return fs[0](fs[1])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='test,val')
    ap.add_argument('--robust', default='')
    ap.add_argument('--robust_dir', default=os.path.join(paths.RESULTS, 'robust'), help='where <variant>_sp.pt live')
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    specs = [(job_split, (s, d)) for s in a.splits.split(',') if s for d in sorted(glob.glob(f'{paths.DATA}/{s}/t*'))]
    specs += [(job_robust, (v, d, a.robust_dir)) for v in a.robust.split(',') if v
              for d in sorted(glob.glob(f'{paths.DATA}/robust/{v}/t*'))]
    t0 = time.time()
    with Pool(a.procs) as p:
        for r in p.imap_unordered(_run, specs):
            print(r, f'[{time.time() - t0:.0f}s]', flush=True)
    print('DONE')
