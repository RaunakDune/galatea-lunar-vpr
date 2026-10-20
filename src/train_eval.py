"""Train an aggregation head on frozen tokens from the LunarSim training terrains and evaluate.
Usage: python train_eval.py --name galatea --head ot --mode learn [--seed 0] ..."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, json, glob, time, argparse, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from aggregators import GeMHead, NetVLADHead, OTHead, load_salad_init
from common import ROOT, load_eval, recall_from_sim, summarize

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--head', default='ot', choices=['ot', 'netvlad', 'gem'])
p.add_argument('--mode', default='learn')
p.add_argument('--lam0', type=float, default=0.5)
p.add_argument('--iters', type=int, default=3)
p.add_argument('--m', type=int, default=64)
p.add_argument('--l', type=int, default=128)
p.add_argument('--no_dustbin', action='store_true')
p.add_argument('--mpc', type=float, default=1.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=600)
p.add_argument('--lr', type=float, default=4e-4)
p.add_argument('--P', type=int, default=32)
p.add_argument('--init', default='none', choices=['none', 'salad'])
p.add_argument('--train_glob', default=f'{ROOT}/train/t*')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--scalar_lr_mult', type=float, default=1.0)
p.add_argument('--eval_split', default='test')
p.add_argument('--save_desc', action='store_true')
p.add_argument('--device', default='cpu', help="e.g. cuda:1")
args = p.parse_args()
torch.set_num_threads(args.threads)
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = torch.device(args.device)
OUT = f'{paths.RESULTS}/runs/{args.name}'
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data
lab, xy, terr = [], [], []
tdirs = sorted(glob.glob(args.train_glob))
TOK, CLS, gidx = [], [], []          # memory-mapped per-terrain tensors (no copy into RAM)
for ti, d in enumerate(tdirs):
    meta = json.load(open(f'{d}/meta.json'))
    Fd = torch.load(f'{d}/feats.pt', weights_only=False, mmap=True)
    TOK.append(Fd['tokens']); CLS.append(Fd['cls'])
    for j, mm in enumerate(meta['q']):
        lab.append(ti * 10000 + mm['place']); xy.append((mm['x'], mm['y'])); terr.append(ti); gidx.append((ti, j))


def fetch(idx):
    t = torch.stack([TOK[gidx[i][0]][gidx[i][1]] for i in idx]).float()
    c = torch.stack([CLS[gidx[i][0]][gidx[i][1]] for i in idx]).float()
    return t.to(DEV), c.to(DEV)


lab = np.array(lab); xy = np.array(xy); terr = np.array(terr)
places = np.unique(lab)
by_place = {pl: np.nonzero(lab == pl)[0] for pl in places}
place_xy = {pl: xy[by_place[pl]].mean(0) for pl in places}
print(f'train: {len(lab)} imgs, {len(places)} places', flush=True)

# ------------------------------------------------------------------ model
if args.head == 'gem':
    net = GeMHead()
elif args.head == 'netvlad':
    net = NetVLADHead(m=args.m, l=args.l)
else:
    net = OTHead(m=args.m, l=args.l, iters=args.iters, mode=args.mode, lam0=args.lam0,
                 dustbin=not args.no_dustbin, mass_per_cluster=args.mpc)
if args.init == 'salad':
    sd = torch.load(os.path.join(paths.WEIGHTS, 'dino_salad.ckpt'), map_location='cpu')
    ag = {k[len('aggregator.'):]: v for k, v in sd.items() if k.startswith('aggregator.')}
    print('init from SALAD:', load_salad_init(net, ag))
if args.head == 'netvlad':   # k-means initialisation of centroids, as in NetVLAD
    with torch.no_grad():
        idx = np.random.choice(len(lab), 200, replace=False)
        z = F.normalize(net.f(fetch(idx)[0].cpu()).flatten(2), dim=1).permute(0, 2, 1).reshape(-1, args.l)
        from sklearn.cluster import KMeans
        km = KMeans(args.m, n_init=1, random_state=args.seed).fit(z.numpy())
        c = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        net.centroids.data = c
        net.assign.weight.data = (2 * 10 * F.normalize(c, dim=1))[:, :, None]
        net.assign.bias.data = -10 * c.norm(dim=1)
net = net.to(DEV)
scal = [p_ for n_, p_ in net.named_parameters() if n_ in ('lam_logit', 'dust_bin')]
rest = [p_ for n_, p_ in net.named_parameters() if n_ not in ('lam_logit', 'dust_bin')]
groups = [{'params': rest, 'lr': args.lr}]
if scal:
    groups.append({'params': scal, 'lr': args.lr * args.scalar_lr_mult})
opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=args.steps, pct_start=0.1)


def ms_loss(D, labels, ign, alpha=1.0, beta=50.0, base=0.0, eps=0.1):
    """Multi-similarity loss with MS miner; ign marks pairs to ignore (nearby different places)."""
    S = D @ D.T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=D.device)
    posm = same & ~eye
    negm = ~same & ~ign
    loss, cnt = 0.0, 0
    for i in range(len(labels)):
        sp, sn = S[i][posm[i]], S[i][negm[i]]
        if len(sp) == 0 or len(sn) == 0:
            continue
        hn = sn[sn + eps > sp.min()]
        hp = sp[sp - eps < sn.max()]
        if len(hn) == 0 and len(hp) == 0:
            continue
        lp = torch.log1p(torch.exp(-alpha * (hp - base)).sum()) / alpha if len(hp) else 0.0
        ln = torch.log1p(torch.exp(beta * (hn - base)).sum()) / beta if len(hn) else 0.0
        loss = loss + lp + ln; cnt += 1
    return loss / max(cnt, 1)


rng = np.random.default_rng(args.seed)
t0 = time.time()
net.train()
for step in range(args.steps):
    pl = rng.choice(places, args.P, replace=False)
    idx = np.concatenate([by_place[q] for q in pl])
    L = torch.tensor(lab[idx], device=DEV)
    pxy = np.array([place_xy[lab[i]] for i in idx]); ptr = terr[idx]
    near = (np.linalg.norm(pxy[:, None] - pxy[None], axis=-1) < 15) & (ptr[:, None] == ptr[None])
    ign = torch.tensor(near, device=DEV) & (L[:, None] != L[None, :])
    D = net(fetch(idx))
    loss = ms_loss(D, L, ign)
    if not torch.is_tensor(loss):          # no informative pair mined in this batch: skip the update
        sched.step(); continue
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 50 == 0 or step == args.steps - 1:
        extra = f' lam={float(net.lam()):.3f} z={float(net.dust_bin):.2f}' if args.head == 'ot' else ''
        print(f'step {step} loss {float(loss):.4f}{extra} {time.time() - t0:.0f}s', flush=True)
del TOK, CLS
net.eval()
torch.save(net.state_dict(), f'{OUT}/head.pt')

# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def describe(Fd):
    out = []
    for i in range(0, len(Fd['tokens']), 64):
        out.append(net((Fd['tokens'][i:i + 64].float().to(DEV), Fd['cls'][i:i + 64].float().to(DEV))).cpu())
    return torch.cat(out)


res_all = {}
for split in args.eval_split.split(','):
    while any(not os.path.exists(f'{d}/feats.pt') for d in glob.glob(f'{ROOT}/{split}/t*')):
        time.sleep(20)
    T = load_eval(split)
    hits, conds, descs = [], [], {}
    t1 = time.time()
    for Tt in T:
        D = describe(Tt['F']).numpy()
        db, q = D[:Tt['ndb']], D[Tt['ndb']:]
        h, order = recall_from_sim(q @ db.T, Tt['pos'])
        hits.append(h); conds.append(Tt['qcond'])
        if args.save_desc:
            descs[os.path.basename(Tt['dir'])] = torch.tensor(D).half()
    dt = (time.time() - t1)
    r = summarize(hits, conds)
    r['hits_R1'] = np.concatenate([h[1] for h in hits]).astype(int).tolist()
    r['hits_R5'] = np.concatenate([h[5] for h in hits]).astype(int).tolist()
    r['hits_R10'] = np.concatenate([h[10] for h in hits]).astype(int).tolist()
    res_all[split] = r
    if args.save_desc:
        torch.save(descs, f'{OUT}/desc_{split}.pt')
    print(split, {k: round(v, 1) for k, v in r.items() if not k.startswith('hits') and '_' not in k},
          flush=True)
    print(split, 'per-cond R@1', {c.split('_')[0]: round(r[f'R@1_{c}'], 1)
                                  for c in ['C0_same', 'C1_el15', 'C2_el6', 'C3_el60', 'C4_az+90', 'C5_az+180',
                                            'C6_az+180_el8']}, flush=True)
info = {'args': vars(args), 'train_time_s': time.time() - t0,
        'n_params': sum(p.numel() for p in net.parameters()),
        'desc_dim': int(describe({'tokens': torch.zeros(1, 768, 15, 20), 'cls': torch.zeros(1, 768)}).shape[1])}
if args.head == 'ot':
    info['lam'] = float(net.lam()); info['dustbin'] = float(net.dust_bin)
json.dump({'info': info, 'results': res_all}, open(f'{OUT}/results.json', 'w'))
