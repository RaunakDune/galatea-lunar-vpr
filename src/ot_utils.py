import numpy as np, torch


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

