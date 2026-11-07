"""Stage-1 evaluation of the robustness variants written by gen_robust.py.

For every variant and test terrain, the (unchanged, upright) database images and the variant's query
images go through the same pipeline: the frozen DINOv2 prefix (blocks 0-7) is computed once per image,
then
  - the frozen suffix (blocks 8-11) gives the tokens for the zero-shot SALAD model and the frozen heads,
  - each fine-tuned run's own blocks 8-11 (backbone_ft.pt, from train_ft.py --save_backbone) and head
    give its descriptor.
SuperPoint keypoints of the queries are saved for robust_rerank.py.
Output: <RESULTS>/robust/<variant>.pt with, per terrain and model, the top-10 database candidates and
their similarities, plus query conditions and ground-truth positives.

usage: python robust_eval.py --variants roll_0,roll_45 --frozen_dir ~/galatea/results_gpu/runs \
           --frozen galatea_s0,... --ft rft_galatea_s0,... --device cuda:0
"""
import os, sys, json, glob, time, argparse, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths, nets
from aggregators import GeMHead, NetVLADHead, OTHead
from common import POS_R, POS_YAW, yawdiff
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--variants', required=True)
ap.add_argument('--frozen_dir', default=os.path.join(paths.RESULTS, 'runs'))
ap.add_argument('--frozen', default='galatea_s0,galatea_s1,galatea_s2,netvlad_s0,netvlad_s1,netvlad_s2')
ap.add_argument('--ft_dir', default=os.path.join(paths.RESULTS, 'runs'))
ap.add_argument('--ft', default='')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--no_sp', action='store_true')
ap.add_argument('--K', type=int, default=10)
args = ap.parse_args()
torch.set_grad_enabled(False)
DEV = torch.device(args.device)
DINO_SIZE = (210, 280)
OUT = os.path.join(paths.RESULTS, 'robust')
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ models
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
sp_net = None if args.no_sp else nets.superpoint().to(DEV)
print('models:', list(frozen), list(ft), flush=True)


def prefix(imgs):
    x = nets.to_rgb_tensor(imgs, DINO_SIZE).to(DEV)
    z = bb.prepare_tokens_with_masks(x)
    for blk in bb.blocks[:K0]:
        z = blk(z)
    return z


def tokens(z, blocks, norm):
    for blk in blocks:
        z = blk(z)
    z = norm(z)
    B = z.shape[0]
    return z[:, 1:].reshape(B, DINO_SIZE[0] // 14, DINO_SIZE[1] // 14, -1).permute(0, 3, 1, 2), z[:, 0]


def describe(d, names, bs=64, want_sp=False):
    out = {k: [] for k in ['salad_zs', *frozen, *ft]}
    sps = []
    for i in range(0, len(names), bs):
        imgs = [np.array(Image.open(f'{d}/{n}')) for n in names[i:i + bs]]
        z = prefix(imgs)
        f, t = tokens(z, bb.blocks[K0:], bb.norm)
        out['salad_zs'].append(ds.agg((f, t)).float().cpu())
        for n, h in frozen.items():
            out[n].append(h((f, t)).float().cpu())
        for n, (suf, h) in ft.items():
            out[n].append(h(tokens(z, suf[:-1], suf[-1])).float().cpu())
        if want_sp:
            g = torch.from_numpy(np.stack(imgs)).float().div(255.)[:, None].to(DEV)
            for k in range(g.shape[0]):
                o = sp_net({'image': g[k:k + 1]})
                sps.append({'kp': o['keypoints'][0].half().cpu(), 'sc': o['keypoint_scores'][0].half().cpu(),
                            'desc': o['descriptors'][0].half().cpu()})
    return {k: torch.cat(v).numpy() for k, v in out.items()}, sps


db_cache = {}
for v in args.variants.split(','):
    if os.path.exists(f'{OUT}/{v}.pt'):
        print('skip', v); continue
    t0 = time.time()
    res, sp_all = {}, {}
    for d in sorted(glob.glob(f'{paths.DATA}/robust/{v}/t*')):
        meta = json.load(open(f'{d}/meta.json'))
        tn = os.path.basename(d)
        if tn not in db_cache:
            db_cache[tn], _ = describe(meta['db_dir'], [m['file'] for m in meta['db']])
        Dq, sps = describe(d, [m['file'] for m in meta['q']], want_sp=not args.no_sp)
        db, q = meta['db'], meta['q']
        dbxy = np.array([[m['x'], m['y']] for m in db]); dby = np.array([m['yaw'] for m in db])
        qxy = np.array([[m['x'], m['y']] for m in q]); qy = np.array([m['yaw'] for m in q])
        pos = (np.linalg.norm(qxy[:, None] - dbxy[None], axis=-1) < POS_R) & (yawdiff(qy[:, None], dby[None]) < POS_YAW)
        r = {'qcond': np.array([m['cond'] for m in q]), 'pos': pos, 'models': {}}
        for n in Dq:
            sim = Dq[n] @ db_cache[tn][n].T
            cand = np.argsort(-sim, 1)[:, :args.K]
            r['models'][n] = {'cand': cand.astype(np.int16), 'gsim': np.take_along_axis(sim, cand, 1).astype(np.float32)}
        res[tn] = r
        sp_all[tn] = sps
    torch.save(res, f'{OUT}/{v}.pt')
    if not args.no_sp:
        torch.save(sp_all, f'{OUT}/{v}_sp.pt')
    r1 = {n: np.mean(np.concatenate([np.take_along_axis(res[t]['pos'], res[t]['models'][n]['cand'][:, :1].astype(int), 1)[:, 0]
                                     for t in res])) * 100 for n in res[next(iter(res))]['models']}
    print(v, f'{time.time() - t0:.0f}s', {n: round(x, 1) for n, x in r1.items()}, flush=True)
