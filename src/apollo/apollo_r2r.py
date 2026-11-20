"""Real Apollo 15 photographs as queries against a database rendered from the LROC NAC DTM.

Queries: every frame of an Apollo 15 pan whose station is in the rendered database (Station 8 is the
ALSEP site; SEVA pans, taken from the LM hatch, are left out). Database: the rendered views of
render_a15.py, as DESPINA images (image_seed<k>.png), plain shaded DEM renders (dem_sim.png), or
both. A retrieval is correct when the top database view is at the query's station.

usage: python apollo_r2r.py --img_dir ~/apollo_bench/img --pans pans_final.json --views views_a15 \
         --db despina:image_seed0.png,shaded:dem_sim.png --frozen_dir ... --frozen ... --ft_dir ... --ft ... \
         --out r2r_a15.pt
"""
import os, sys, json, argparse, copy, re, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import nets
from aggregators import GeMHead, NetVLADHead, OTHead

ap = argparse.ArgumentParser()
ap.add_argument('--img_dir', required=True)
ap.add_argument('--pans', required=True)
ap.add_argument('--views', required=True)
ap.add_argument('--db', default='despina:image_seed0.png,shaded:dem_sim.png')
ap.add_argument('--frozen_dir', required=True)
ap.add_argument('--frozen', default='')
ap.add_argument('--ft_dir', default='')
ap.add_argument('--ft', default='')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
torch.set_grad_enabled(False)
DEV = torch.device(args.device)
DINO_SIZE = (210, 280)
ALIAS, SKIP = {'8': 'ALSEP'}, {'SEVA'}

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


def load(path):
    im = Image.open(path).convert('L')
    w, h = im.size
    hh = int(round(w * 3 / 4))
    top = max(0, (h - hh) // 2)
    return np.array(im.crop((0, top, w, top + hh)).resize((DINO_SIZE[1] * 2, DINO_SIZE[0] * 2), Image.BICUBIC))


def tokens(z, blocks, norm):
    for blk in blocks:
        z = blk(z)
    z = norm(z)
    B = z.shape[0]
    return z[:, 1:].reshape(B, DINO_SIZE[0] // 14, DINO_SIZE[1] // 14, -1).permute(0, 3, 1, 2), z[:, 0]


def describe(paths, bs=32):
    out = {k: [] for k in ['salad_zs', *frozen, *ft]}
    for i in range(0, len(paths), bs):
        x = nets.to_rgb_tensor([load(p) for p in paths[i:i + bs]], DINO_SIZE).to(DEV)
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


views = json.load(open(f'{args.views}/views.json'))
vst = np.array([v['station'] for v in views])
stations = set(vst)
q = []
for p in json.load(open(args.pans)):
    if p['mission'] != 15:
        continue
    s = p['station'].split('-', 1)[1]
    s = 'A15-' + ALIAS.get(s, s)
    if s.split('-', 1)[1] in SKIP or s not in stations:
        continue
    q += [(f, s, p['name']) for f in p['frames'] if os.path.exists(f'{args.img_dir}/{f}.png')]
qst = np.array([x[1] for x in q])
print(f'{len(q)} queries at {len(set(qst))} stations; database {len(views)} views at {len(stations)} stations',
      flush=True)
Dq = describe([f'{args.img_dir}/{x[0]}.png' for x in q])
res = {'q_frames': [x[0] for x in q], 'q_station': qst, 'q_pan': [x[2] for x in q], 'views': views, 'db': {}}
for spec in args.db.split(','):
    tag, fname = spec.split(':')
    Dd = describe([f'{args.views}/{v["name"]}/{fname}' for v in views])
    res['db'][tag] = {}
    for n in Dq:
        sim = Dq[n] @ Dd[n].T
        cand = np.argsort(-sim, 1)[:, :25]
        hit = vst[cand] == qst[:, None]
        res['db'][tag][n] = {'cand': cand.astype(np.int16), 'sim': np.take_along_axis(sim, cand, 1)}
        print(f'{tag:8s} {n:24s} R@1 {hit[:, 0].mean() * 100:5.1f}  R@5 {hit[:, :5].any(1).mean() * 100:5.1f}',
              flush=True)
chance = np.mean([(vst == s).mean() for s in qst]) * 100
res['chance_R1'] = chance
print(f'chance R@1 {chance:.1f}%')
torch.save(res, args.out)
