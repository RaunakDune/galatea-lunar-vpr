"""Aggregation heads operating on frozen ViT tokens (f: [B,C,h,w], t: [B,C]).

GeMHead      : MLP + generalised-mean pooling + scene token.
NetVLADHead  : soft-assignment (softmax) VLAD on projected tokens (residuals to centroids).
OTHead       : SALAD-style optimal-transport assignment with dust-bin. The cluster-side
               (column) update of Sinkhorn is scaled by lam in [0,1]:
                   u <- lam * (log a - LSE_feat(S + v))       (clusters + dust-bin)
                   v <-        log b - LSE_clu (S + u)        (features, hard)
               lam = tau/(tau+eps) is the exact scaling exponent of the semi-relaxed
               (KL-penalised) unbalanced OT problem (Chizat et al. 2018).
               lam = 1  -> balanced OT (SALAD);  lam = 0 -> per-feature softmax (NetVLAD-like
               assignment with a dust-bin).  Galatea learns lam (mode='learn').
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def mlp_conv(cin, hid, cout, drop):
    return nn.Sequential(nn.Conv2d(cin, hid, 1), nn.Dropout(drop) if drop > 0 else nn.Identity(),
                         nn.ReLU(), nn.Conv2d(hid, cout, 1))


class TokenMLP(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, 512), nn.ReLU(), nn.Linear(512, cout))

    def forward(self, t):
        return F.normalize(self.net(t), dim=-1)


class GeMHead(nn.Module):
    def __init__(self, C=768, dim=1024, token_dim=256, drop=0.3):
        super().__init__()
        self.f = mlp_conv(C, 512, dim, drop)
        self.p = nn.Parameter(torch.tensor(3.0))
        self.tok = TokenMLP(C, token_dim)

    def forward(self, x):
        f, t = x
        z = F.relu(self.f(f)).clamp(min=1e-6).flatten(2)          # B,dim,n
        g = z.pow(self.p).mean(-1).pow(1 / self.p)
        return F.normalize(torch.cat([self.tok(t), F.normalize(g, dim=-1)], -1), dim=-1)


class NetVLADHead(nn.Module):
    def __init__(self, C=768, m=64, l=128, token_dim=256, drop=0.3):
        super().__init__()
        self.m, self.l = m, l
        self.f = mlp_conv(C, 512, l, drop)
        self.assign = nn.Conv1d(l, m, 1)
        self.centroids = nn.Parameter(torch.randn(m, l) * 0.1)
        self.tok = TokenMLP(C, token_dim)

    def forward(self, x):
        f, t = x
        z = F.normalize(self.f(f).flatten(2), dim=1)               # B,l,n
        a = F.softmax(self.assign(z), dim=1)                       # B,m,n
        V = torch.einsum('bmn,bln->bml', a, z) - a.sum(-1, keepdim=True) * self.centroids[None]
        V = F.normalize(V, dim=-1).flatten(1)
        return F.normalize(torch.cat([self.tok(t), V], -1), dim=-1)


def ot_assign(S, z, iters=3, lam=1.0, dustbin=True, mass_per_cluster=1.0):
    """S: [B,m,n] scores (clusters x features). Returns P [B,m,n] (dust-bin dropped).
    Marginals follow SALAD: features 1 each, clusters `mass_per_cluster` each, dust-bin the rest,
    all divided by (n+m) for numerical scale. Final update is on the feature side, so every
    feature's total mass (clusters + dust-bin) is exactly 1."""
    B, m, n = S.shape
    if dustbin:
        Sa = torch.cat([S, z.expand(B, 1, n)], 1)
        rest = max(n - mass_per_cluster * m, 1e-3)
        log_a = torch.full((m + 1,), math.log(mass_per_cluster), device=S.device)
        log_a[-1] = math.log(rest)
    else:
        Sa = S
        log_a = torch.full((m,), math.log(n / m), device=S.device)
    norm = -math.log(n + m)
    log_a = (log_a + norm).expand(B, -1)
    log_b = torch.full((B, n), norm, device=S.device)
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(iters):
        u = lam * (log_a - torch.logsumexp(Sa + v[:, None, :], dim=2))
        v = log_b - torch.logsumexp(Sa + u[:, :, None], dim=1)
    logP = Sa + u[:, :, None] + v[:, None, :] - norm
    P = logP.exp()
    return P[:, :m] if dustbin else P, P


class OTHead(nn.Module):
    """mode: 'balanced' (lam=1, SALAD), 'softmax' (lam=0), 'fixed' (lam=lam0), 'learn' (Galatea)."""

    def __init__(self, C=768, m=64, l=128, token_dim=256, drop=0.3, iters=3, mode='learn', lam0=0.5,
                 dustbin=True, mass_per_cluster=1.0):
        super().__init__()
        self.m, self.l, self.iters, self.mode, self.dustbin = m, l, iters, mode, dustbin
        self.mpc = mass_per_cluster
        self.f = mlp_conv(C, 512, l, drop)
        self.score = mlp_conv(C, 512, m, drop)
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        self.lam_logit = nn.Parameter(torch.tensor(math.log(lam0 / (1 - lam0))) if 0 < lam0 < 1 else torch.tensor(0.0))
        self.lam0 = lam0
        self.tok = TokenMLP(C, token_dim)

    def lam(self):
        if self.mode == 'balanced':
            return 1.0
        if self.mode == 'softmax':
            return 0.0
        if self.mode == 'fixed':
            return self.lam0
        return torch.sigmoid(self.lam_logit)

    def assignment(self, f):
        S = self.score(f).flatten(2)
        return ot_assign(S, self.dust_bin, self.iters, self.lam(), self.dustbin, self.mpc)

    def forward(self, x):
        f, t = x
        z = self.f(f).flatten(2)                                   # B,l,n
        P, _ = self.assignment(f)                                  # B,m,n
        V = torch.einsum('bmn,bln->bml', P, z)
        V = F.normalize(V, dim=-1).flatten(1)
        return F.normalize(torch.cat([self.tok(t), V], -1), dim=-1)


def load_salad_init(head, salad_state):
    """Initialise an OTHead from the pretrained (terrestrial) SALAD aggregator weights."""
    sd = {}
    for k, v in salad_state.items():
        k2 = (k.replace('cluster_features.', 'f.').replace('score.', 'score.')
              .replace('token_features.', 'tok.net.'))
        sd[k2] = v
    missing, unexpected = head.load_state_dict(sd, strict=False)
    return missing, unexpected
