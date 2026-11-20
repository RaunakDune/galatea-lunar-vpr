"""Real-image place recognition on Apollo surface panoramas (station-level ground truth).

Every pan in pans_final.json was taken from one standpoint at a named traverse station. A query is
any frame of a pan; its database is every frame of every OTHER pan of the same mission, and a
retrieved frame is correct when it comes from the same station. Only queries whose station has at
least one other pan are scored, so every query has a positive. Pans of the same station differ in
photographer, time (often a different EVA, so different Sun elevation), standpoint (metres to tens of
metres) and heading coverage.

Descriptors use the same models as robust_eval.py (frozen DINOv2 prefix, frozen or fine-tuned suffix,
trained heads). Images are converted to grey, centre-cropped to the training aspect (3:4) and resized.

usage: python apollo_eval.py --img_dir ~/apollo_bench/processed --pans ~/apollo_meta/pans_final.json \
          --frozen_dir ~/galatea/results_gpu/runs --frozen galatea_s0,... \
          --ft_dir ~/galatea/results_robust/runs --ft rft_galatea_s0,... --device cuda:0 --out apollo.pt
"""
import os, sys, json, time, argparse, copy, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import nets
from aggregators import GeMHead, NetVLADHead, OTHead

ap = argparse.ArgumentParser()
ap.add_argument('--img_dir', required=True)
ap.add_argument('--pans', required=True)
ap.add_argument('--frozen_dir', required=True)
ap.add_argument('--frozen', default='')
ap.add_argument('--ft_dir', default='')
ap.add_argument('--ft', default='')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--crop', default='center', choices=['center', 'full'])
ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_grad_enabled(False)
DEV = torch.device(args.device)
DINO_SIZE = (210, 280) if args.crop == 'center' else (280, 280)

ds = nets.DinoSalad().to(DEV)
bb = ds.model
K0 = len(bb.blocks) - 4


def build_head(run_dir):
    a = json.load(open(f'{run_dir}/results.json'))['info']['args']
    if a['head'] == 'gem':
        h = GeMHead()
    elif a['head'] == 'netvlad':
        h = NetVLADHead(m=a['m'], l=a['l'])
    else:
        h = OTHead(m=a['m'], l=a['l'], iters=a['iters'], mode=a['mode'], lam0=a['lam0'],
                   dustbin=not a['no_dustbin'], mass_per_cluster=a.get('mpc', 1.0))
    h.load_state_dict(torch.load(f'{run_dir}/head.pt', map_location='cpu'))
    return h.to(DEV).eval()


frozen = {n: build_head(f'{args.frozen_dir}/{n}') for n in args.frozen.split(',') if n}
ft = {}
for n in [n for n in args.ft.split(',') if n]:
    suf = torch.nn.ModuleList([copy.deepcopy(b) for b in bb.blocks[K0:]] + [copy.deepcopy(bb.norm)])
    sd = torch.load(f'{args.ft_dir}/{n}/backbone_ft.pt', map_location='cpu')
    for k, mod in enumerate(suf[:-1]):
        mod.load_state_dict({kk.split('.', 2)[2]: v.float() for kk, v in sd.items() if kk.startswith(f'blocks.{K0 + k}.')})
    suf[-1].load_state_dict({kk[len('norm.'):]: v.float() for kk, v in sd.items() if kk.startswith('norm.')})
    ft[n] = (suf.to(DEV).eval(), build_head(f'{args.ft_dir}/{n}'))
print('models:', ['salad_zs', *frozen, *ft], flush=True)


def load(name):
    im = Image.open(f'{args.img_dir}/{name}.png').convert('L')
    w, h = im.size
    if args.crop == 'center':  # 3:4 (h:w) band through the frame centre
        hh = int(round(w * 3 / 4))
        top = max(0, (h - hh) // 2)
        im = im.crop((0, top, w, top + hh))
    return np.array(im.resize((DINO_SIZE[1] * 2, DINO_SIZE[0] * 2), Image.BICUBIC))


def tokens(z, blocks, norm):
    for blk in blocks:
        z = blk(z)
    z = norm(z)
    B = z.shape[0]
    return z[:, 1:].reshape(B, DINO_SIZE[0] // 14, DINO_SIZE[1] // 14, -1).permute(0, 3, 1, 2), z[:, 0]


def describe(names, bs=32):
    out = {k: [] for k in ['salad_zs', *frozen, *ft]}
    for i in range(0, len(names), bs):
        imgs = [load(n) for n in names[i:i + bs]]
        x = nets.to_rgb_tensor(imgs, DINO_SIZE).to(DEV)
        z = bb.prepare_tokens_with_masks(x)
        for blk in bb.blocks[:K0]:
            z = blk(z)
        f, t = tokens(z, bb.blocks[K0:], bb.norm)
        out['salad_zs'].append(ds.agg((f, t)).float().cpu())
        for n, h in frozen.items():
            out[n].append(h((f, t)).float().cpu())
        for n, (suf, h) in ft.items():
            out[n].append(h(tokens(z, suf[:-1], suf[-1])).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


pans = json.load(open(args.pans))
rows = []  # (frame, pan index, station, mission)
for k, p in enumerate(pans):
    for f in p['frames']:
        if os.path.exists(f'{args.img_dir}/{f}.png'):
            rows.append((f, k, p['station'], p['mission']))
names = [r[0] for r in rows]
pan = np.array([r[1] for r in rows]); st = np.array([r[2] for r in rows]); mi = np.array([r[3] for r in rows])
print(f'{len(rows)} frames from {len(set(pan))} pans', flush=True)
t0 = time.time()
D = describe(names)
print(f'described in {time.time() - t0:.0f}s', flush=True)

res = {'frames': names, 'pan': pan, 'station': st, 'mission': mi, 'models': {}}
for n, X in D.items():
    sim = X @ X.T
    # database of a query: frames of other pans, same mission
    allowed = (mi[:, None] == mi[None]) & (pan[:, None] != pan[None])
    sim = np.where(allowed, sim, -np.inf)
    cand = np.argsort(-sim, 1)[:, :25]
    res['models'][n] = {'cand': cand.astype(np.int32), 'sim': np.take_along_axis(sim, cand, 1).astype(np.float32)}
res['desc'] = {n: X.astype(np.float16) for n, X in D.items()}
torch.save(res, args.out)

# quick report
pos = (st[:, None] == st[None]) & (pan[:, None] != pan[None])
scored = pos.any(1)
for n, r in res['models'].items():
    hit = np.take_along_axis(pos, r['cand'], 1)
    print(f'{n:28s} R@1 {hit[scored, :1].any(1).mean() * 100:5.1f}  R@5 {hit[scored, :5].any(1).mean() * 100:5.1f}'
          f'  (n={scored.sum()})', flush=True)
