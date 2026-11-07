"""Local scorers shared by rerank.py (paper experiments) and robust_rerank.py (robustness tests)."""
import numpy as np, torch, cv2


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


def mnn_matches(dq, dc, thr=0.7):
    s = dq @ dc.T
    a, b = s.argmax(1), s.argmax(0)
    i = torch.arange(len(a)); ok = (b[a] == i) & (s[i, a] > thr)
    return torch.stack([i[ok], a[ok]], 1).numpy()
