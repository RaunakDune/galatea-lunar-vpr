"""Jointly fine-tune the last blocks of the DINOv2 backbone together with an aggregation head (GPU).

Same data, loss, sampler and evaluation protocol as train_eval.py; the only difference is that the last
--ft_blocks transformer blocks (and the final norm) are trained as well. The first 12-ft_blocks blocks stay
frozen, so their output is cached once per run, which makes training ~3x faster and is exactly equivalent
because no augmentation is used.

Usage: python train_ft.py --name ft_galatea_s0 --head ot --mode learn --device cuda:1 [--seed 0]"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os, json, glob, time, argparse, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import nets
from aggregators import GeMHead, NetVLADHead, OTHead
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
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--lr', type=float, default=4e-4, help='head learning rate')
p.add_argument('--bb_lr', type=float, default=3e-5, help='backbone learning rate')
p.add_argument('--ft_blocks', type=int, default=4)
p.add_argument('--P', type=int, default=24)
p.add_argument('--train_glob', default=f'{ROOT}/train/t*')
p.add_argument('--eval_split', default='val,test')
p.add_argument('--device', default='cuda:0')
p.add_argument('--save_desc', action='store_true')
p.add_argument('--save_backbone', action='store_true')
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
DEV = torch.device(args.device)
OUT = f'{paths.RESULTS}/runs/{args.name}'
os.makedirs(OUT, exist_ok=True)
DINO_SIZE = (210, 280)

# ------------------------------------------------------------------ backbone split
ds = nets.DinoSalad()
bb = ds.model.to(DEV)
NB = len(bb.blocks)
K0 = NB - args.ft_blocks                     # blocks [0, K0) frozen and cached
for prm in bb.parameters():
    prm.requires_grad_(False)
train_mods = list(bb.blocks[K0:]) + [bb.norm]
for mod in train_mods:
    for prm in mod.parameters():
        prm.requires_grad_(True)


@torch.no_grad()
def prefix(imgs):
    """Frozen part: images (list of uint8 HxW) -> tokens after block K0, fp16 on CPU."""
    x = nets.to_rgb_tensor(imgs, DINO_SIZE).to(DEV)
    z = bb.prepare_tokens_with_masks(x)
    for blk in bb.blocks[:K0]:
        z = blk(z)
    return z.half().cpu()


def suffix(z):
    """Trainable part: cached tokens -> (patch tokens [B,768,15,20], class token [B,768])."""
    for blk in bb.blocks[K0:]:
        z = blk(z)
    z = bb.norm(z)
    B = z.shape[0]
    f = z[:, 1:].reshape(B, DINO_SIZE[0] // 14, DINO_SIZE[1] // 14, -1).permute(0, 3, 1, 2)
    return f, z[:, 0]


def cache_dir(d, names, bs=64):
    out = []
    for i in range(0, len(names), bs):
        out.append(prefix([np.array(Image.open(f'{d}/{n}')) for n in names[i:i + bs]]))
    return torch.cat(out)


# ------------------------------------------------------------------ data
t0 = time.time()
lab, xy, terr, Z = [], [], [], []
for ti, d in enumerate(sorted(glob.glob(args.train_glob))):
    meta = json.load(open(f'{d}/meta.json'))
    assert len(meta['db']) == 0
    Z.append(cache_dir(d, [m['file'] for m in meta['q']]))
    for mm in meta['q']:
        lab.append(ti * 10000 + mm['place']); xy.append((mm['x'], mm['y'])); terr.append(ti)
Z = torch.cat(Z).pin_memory()
lab = np.array(lab); xy = np.array(xy); terr = np.array(terr)
places = np.unique(lab)
by_place = {pl: np.nonzero(lab == pl)[0] for pl in places}
place_xy = {pl: xy[by_place[pl]].mean(0) for pl in places}
print(f'train: {len(lab)} imgs, {len(places)} places, cached in {time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ head
if args.head == 'gem':
    head = GeMHead()
elif args.head == 'netvlad':
    head = NetVLADHead(m=args.m, l=args.l)
else:
    head = OTHead(m=args.m, l=args.l, iters=args.iters, mode=args.mode, lam0=args.lam0,
                  dustbin=not args.no_dustbin)
if args.head == 'netvlad':   # k-means initialisation on tokens of the (initial) backbone
    with torch.no_grad():
        idx = np.random.choice(len(lab), 200, replace=False)
        f, _ = suffix(Z[idx].float().to(DEV))
        z = F.normalize(head.f(f.cpu()).flatten(2), dim=1).permute(0, 2, 1).reshape(-1, args.l)
        from sklearn.cluster import KMeans
        km = KMeans(args.m, n_init=1, random_state=args.seed).fit(z.numpy())
        c = torch.tensor(km.cluster_centers_, dtype=torch.float32)
        head.centroids.data = c
        head.assign.weight.data = (2 * 10 * F.normalize(c, dim=1))[:, :, None]
        head.assign.bias.data = -10 * c.norm(dim=1)
head = head.to(DEV)
bb_params = [prm for mod in train_mods for prm in mod.parameters()]
groups = [{'params': list(head.parameters()), 'lr': args.lr}, {'params': bb_params, 'lr': args.bb_lr}]
opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in groups], total_steps=args.steps,
                                            pct_start=0.1)


def ms_loss(D, labels, ign, alpha=1.0, beta=50.0, base=0.0, eps=0.1):
    """Multi-similarity loss with MS miner (identical to train_eval.py)."""
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
t1 = time.time()
head.train(); bb.train()
for step in range(args.steps):
    pl = rng.choice(places, args.P, replace=False)
    idx = np.concatenate([by_place[q] for q in pl])
    L = torch.tensor(lab[idx], device=DEV)
    pxy = np.array([place_xy[lab[i]] for i in idx]); ptr = terr[idx]
    near = (np.linalg.norm(pxy[:, None] - pxy[None], axis=-1) < 15) & (ptr[:, None] == ptr[None])
    ign = torch.tensor(near, device=DEV) & (L[:, None] != L[None, :])
    D = head(suffix(Z[idx].to(DEV, non_blocking=True).float()))
    loss = ms_loss(D, L, ign)
    if not torch.is_tensor(loss):          # no informative pair mined in this batch: skip the update
        sched.step(); continue
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 50 == 0 or step == args.steps - 1:
        extra = f' lam={float(head.lam()):.3f} z={float(head.dust_bin):.2f}' if args.head == 'ot' else ''
        print(f'step {step} loss {float(loss):.4f}{extra} {time.time() - t1:.0f}s', flush=True)
train_time = time.time() - t1
del Z
head.eval(); bb.eval()
torch.save(head.state_dict(), f'{OUT}/head.pt')
if args.save_backbone:
    torch.save({k: v.half() for k, v in bb.state_dict().items()
                if k.startswith('norm') or any(k.startswith(f'blocks.{b}.') for b in range(K0, NB))},
               f'{OUT}/backbone_ft.pt')


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def describe_dir(d, names, bs=64):
    out = []
    for i in range(0, len(names), bs):
        z = prefix([np.array(Image.open(f'{d}/{n}')) for n in names[i:i + bs]]).float().to(DEV)
        out.append(head(suffix(z)).cpu())
    return torch.cat(out)


res_all = {}
for split in args.eval_split.split(','):
    T = load_eval(split)
    hits, conds, descs = [], [], {}
    for Tt in T:
        names = [m['file'] for m in Tt['meta']['db']] + [m['file'] for m in Tt['meta']['q']]
        D = describe_dir(Tt['dir'], names).numpy()
        db, q = D[:Tt['ndb']], D[Tt['ndb']:]
        h, _ = recall_from_sim(q @ db.T, Tt['pos'])
        hits.append(h); conds.append(Tt['qcond'])
        if args.save_desc:
            descs[os.path.basename(Tt['dir'])] = torch.tensor(D).half()
    r = summarize(hits, conds)
    for N in (1, 5, 10):
        r[f'hits_R{N}'] = np.concatenate([h[N] for h in hits]).astype(int).tolist()
    res_all[split] = r
    if args.save_desc:
        torch.save(descs, f'{OUT}/desc_{split}.pt')
    print(split, {k: round(v, 1) for k, v in r.items() if not k.startswith('hits') and '_' not in k}, flush=True)
    print(split, 'per-cond R@1', {c.split('_')[0]: round(r[f'R@1_{c}'], 1) for c in
                                  ['C0_same', 'C1_el15', 'C2_el6', 'C3_el60', 'C4_az+90', 'C5_az+180',
                                   'C6_az+180_el8']}, flush=True)

info = {'args': vars(args), 'train_time_s': train_time, 'gpu': torch.cuda.get_device_name(DEV),
        'n_params_head': sum(prm.numel() for prm in head.parameters()),
        'n_params_backbone_trained': sum(prm.numel() for prm in bb_params)}
with torch.no_grad():
    info['desc_dim'] = int(head((torch.zeros(1, 768, 15, 20, device=DEV), torch.zeros(1, 768, device=DEV))).shape[1])
if args.head == 'ot':
    info['lam'] = float(head.lam()); info['dustbin'] = float(head.dust_bin)
json.dump({'info': info, 'results': res_all}, open(f'{OUT}/results.json', 'w'))
print('DONE', args.name, flush=True)
