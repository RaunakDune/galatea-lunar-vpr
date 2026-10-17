"""Stage-2 local re-ranking of the top-K stage-1 candidates.
Local scorers (all on SuperPoint keypoints):
  mnn        : mutual-nearest-neighbour descriptor matches (+RANSAC fundamental-matrix inliers)
  lg         : LightGlue (learned matcher, terrestrial weights) (+RANSAC)
  uot        : entropic OT with dust-bins on appearance only (training-free SuperGlue-style) (+RANSAC)
  fgw        : partial fused Gromov-Wasserstein: appearance + intra-image geometric structure,
               solved by entropic mirror descent with dust-bins (+RANSAC)
Writes results/rerank/<stage1>_<split>.npz with per-(query, candidate) local scores."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, json, time, argparse, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
from common import load_eval

p = argparse.ArgumentParser()
p.add_argument('--stage1', required=True, help='run name with desc_<split>.pt, or salad_zs')
p.add_argument('--split', default='test')
p.add_argument('--K', type=int, default=10)
p.add_argument('--methods', default='mnn,uot,fgw,lg')
p.add_argument('--maxkp', type=int, default=384)
p.add_argument('--alpha', type=float, default=0.3)
p.add_argument('--eps', type=float, default=0.05)
p.add_argument('--cbin', type=float, default=0.25)
p.add_argument('--threads', type=int, default=2)
p.add_argument('--limit', type=int, default=0, help='debug: only first N queries per terrain')
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.set_grad_enabled(False)
methods = args.methods.split(',')
OUT = os.path.join(paths.RESULTS, 'rerank')
os.makedirs(OUT, exist_ok=True)

if 'lg' in methods:
    import nets
    LG = nets.lightglue()


def topk_kp(sp, k):
    kp, sc, de = sp['kp'].float(), sp['sc'].float(), sp['desc'].float()
    if len(sc) > k:
        idx = torch.argsort(sc, descending=True)[:k]
        kp, sc, de = kp[idx], sc[idx], de[idx]
    return kp, sc, torch.nn.functional.normalize(de, dim=1)


def ransac_inliers(p1, p2):
    if len(p1) < 8:
        return 0
    try:
        F, mask = cv2.findFundamentalMat(p1.astype(np.float64), p2.astype(np.float64), cv2.USAC_MAGSAC, 1.5, 0.999, 2000)
    except cv2.error:
        return 0
    return 0 if mask is None else int(mask.sum())


def log_sinkhorn_dustbin(S, z, iters=50):
    """S: [p,q] similarity/eps; z: dust-bin score. Marginals: rows 1 (+ q for bin), cols 1 (+ p for bin)."""
    p_, q_ = S.shape
    Sa = torch.full((p_ + 1, q_ + 1), float(z))
    Sa[:p_, :q_] = S
    log_a = torch.zeros(p_ + 1); log_a[-1] = np.log(q_)
    log_b = torch.zeros(q_ + 1); log_b[-1] = np.log(p_)
    norm = -np.log(p_ + q_)
    log_a = log_a + norm; log_b = log_b + norm
    u = torch.zeros(p_ + 1); v = torch.zeros(q_ + 1)
    for _ in range(iters):
        u = log_a - torch.logsumexp(Sa + v[None], 1)
        v = log_b - torch.logsumexp(Sa + u[:, None], 0)
    return (Sa + u[:, None] + v[None] - norm).exp()          # rows of real points sum to 1


def mutual_matches(T, thr=0.2):
    Tc = T[:-1, :-1]
    if Tc.numel() == 0:
        return np.zeros((0, 2), int)
    m0 = Tc.argmax(1); m1 = Tc.argmax(0)
    i = torch.arange(Tc.shape[0])
    ok = (m1[m0] == i) & (Tc[i, m0] > thr)
    return torch.stack([i[ok], m0[ok]], 1).numpy()


def uot_match(d1, d2, eps=0.05, z=None):
    sim = d1 @ d2.T
    z = (0.75 / eps) if z is None else z        # a pair is preferred to the bin if cos > ~0.75
    return log_sinkhorn_dustbin(sim / eps, z)


def fgw_match(k1, d1, k2, d2, alpha=0.3, eps=0.05, outer=5, cbin=0.25):
    """Partial fused Gromov-Wasserstein (square loss) with dust-bins, entropic mirror descent."""
    M = 1 - d1 @ d2.T                                        # appearance cost in [0,2]
    C1 = torch.cdist(k1, k1); C1 = C1 / (C1.mean() + 1e-9)
    C2 = torch.cdist(k2, k2); C2 = C2 / (C2.mean() + 1e-9)
    z = -cbin / eps                                          # a pair beats the bin if fused cost < cbin
    T = log_sinkhorn_dustbin(-M / eps, z)
    for _ in range(outer):
        P = T[:-1, :-1]
        P = P / (P.sum() + 1e-9)                             # transported part, normalised
        pm, qm = P.sum(1), P.sum(0)
        G = (C1 ** 2 @ pm)[:, None] + (C2 ** 2 @ qm)[None, :] - 2 * C1 @ P @ C2.T
        Cf = (1 - alpha) * M + alpha * G
        T = log_sinkhorn_dustbin(-Cf / eps, z)
    return T


def lightglue_match(k1, d1, k2, d2, size=(320, 240)):
    s = torch.tensor([size], dtype=torch.float32)
    o = LG({'image0': {'keypoints': k1[None], 'descriptors': d1[None], 'image_size': s},
            'image1': {'keypoints': k2[None], 'descriptors': d2[None], 'image_size': s}})
    m = o['matches'][0]
    return m.numpy()


if args.stage1 == 'salad_zs':
    desc = None
else:
    desc = torch.load(f'{paths.RESULTS}/runs/{args.stage1}/desc_{args.split}.pt')

T_all = load_eval(args.split)
store = {}
t0 = time.time()
tag = f'{args.stage1}_{args.split}_{"-".join(methods)}' + (f'_a{args.alpha}_c{args.cbin}' if 'fgw' in methods else '') + (f'_lim{args.limit}' if args.limit else '')
for Tt in T_all:
    name = os.path.basename(Tt['dir'])
    part = f'{OUT}/partial_{tag}_{name}.npy'
    if os.path.exists(part):
        store[name] = np.load(part, allow_pickle=True).item(); print('resume', name); continue
    D = (Tt['F']['salad_zs'] if desc is None else desc[name]).float().numpy()
    ndb = Tt['ndb']
    sim = D[ndb:] @ D[:ndb].T
    cand = np.argsort(-sim, 1)[:, :args.K]
    nq = len(cand) if args.limit == 0 else args.limit
    sp = Tt['F']['sp']
    kps = [topk_kp(s, args.maxkp) for s in sp]
    sc = {m: np.zeros((nq, args.K), np.float32) for m in methods}
    sc.update({m + '_ransac': np.zeros((nq, args.K), np.float32) for m in methods})
    for qi in range(nq):
        kq, _, dq = kps[ndb + qi]
        for r, c in enumerate(cand[qi]):
            kc, _, dc = kps[c]
            for m in methods:
                if m == 'mnn':
                    s = dq @ dc.T
                    a, b = s.argmax(1), s.argmax(0)
                    i = torch.arange(len(a)); ok = (b[a] == i) & (s[i, a] > 0.7)
                    mm = torch.stack([i[ok], a[ok]], 1).numpy()
                elif m == 'uot':
                    mm = mutual_matches(uot_match(dq, dc, args.eps))
                elif m == 'fgw':
                    mm = mutual_matches(fgw_match(kq, dq, kc, dc, args.alpha, args.eps, cbin=args.cbin))
                elif m == 'lg':
                    mm = lightglue_match(kq, dq, kc, dc)
                sc[m][qi, r] = len(mm)
                if len(mm):
                    sc[m + '_ransac'][qi, r] = ransac_inliers(kq[mm[:, 0]].numpy(), kc[mm[:, 1]].numpy())
        if qi % 200 == 0:
            print(name, qi, f'{time.time() - t0:.0f}s', flush=True)
    store[name] = {'cand': cand[:nq], 'gsim': np.take_along_axis(sim, cand, 1)[:nq], **sc}
    np.save(part, store[name], allow_pickle=True)
np.save(f'{OUT}/{tag}.npy', store, allow_pickle=True)
print('done', tag, f'{time.time() - t0:.0f}s')
