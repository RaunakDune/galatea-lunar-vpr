"""Place recognition on the LunarLoc traverses (a second lunar simulator, built on the Lunar Autonomy
Challenge simulator by Thomas et al., IEEE Aerospace 2026).

Pairs follow LunarLoc's Table 5: straight line (1,2), (3,4), (5,6), (7,8); lighting change (9,10),
(11,12), (13,14); flipped start (1,15), (2,16), (3,17) ((4,18) is not in the release). Each pair is
evaluated in both directions: the frames of one traverse are the database, those of the other the queries.
A database frame is a positive if its camera is within 1 m and 30 deg of heading (the LunarSim-VPR
definition). On flipped pairs the routes run in opposite directions, so almost no frame has a positive
under this definition; there we also report a view-overlap definition: the ground seen by the two cameras
(1 to 15 m ahead, within the 55 deg crop) overlaps by at least 30% (intersection over union).

Images: FrontLeft grayscale, centre-cropped to 4:3 (960 x 720, 55 deg horizontal field of view) and
resized as for the Apollo frames (DINOv2 input 280 x 210); SuperPoint on the same crop at 320 x 240.
Stage 2 re-ranks each model's top 10 with MNN + MAGSAC++, partial FGW and image-plane CLIPPER, with the
fusion parameters of the LunarSim validation scene, and with CLIPPER on LunarLoc's own 3-D boulder
detections (boulder: tolerance and fusion parameters tuned on the straight-line pairs, which serve as the
development set; lighting and flipped pairs are held out).

  --stage desc   descriptors + SuperPoint per traverse (one GPU) -> <feat>/traverse_<k>.pt
  --stage eval   retrieval, re-ranking and metrics (CPU)         -> <out>/lunarloc_summary.json
"""
import os, sys, json, copy, time, argparse, numpy as np, torch
from multiprocessing import Pool
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ap = argparse.ArgumentParser()
ap.add_argument('--stage', required=True, choices=['desc', 'eval'])
ap.add_argument('--ext', default=os.path.expanduser('~/lunarloc/ext'))
ap.add_argument('--feat', default=os.path.expanduser('~/lunarloc/feat'))
ap.add_argument('--out', default=os.path.expanduser('~/lunarloc'))
ap.add_argument('--frozen_dir', default=os.path.expanduser('~/galatea/results_gpu/runs'))
ap.add_argument('--frozen', default='galatea_s0,galatea_s1,galatea_s2,netvlad_s0,netvlad_s1,netvlad_s2,gem_s0,gem_s1,gem_s2,'
                                     'salad_s0,salad_s1,salad_s2')
ap.add_argument('--ft_dir', default=os.path.expanduser('~/galatea/results_robust/runs'))
ap.add_argument('--ft', default='rft_galatea_s0,rft_galatea_s1,rft_galatea_s2,rft_netvlad_s0,rft_netvlad_s1,rft_netvlad_s2')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--sim_params', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'rerank_summary.json'))
ap.add_argument('--procs', type=int, default=16)
args = ap.parse_args()

PAIRS = {'straight': [(1, 2), (3, 4), (5, 6), (7, 8)], 'lighting': [(9, 10), (11, 12), (13, 14)],
         'flipped': [(1, 15), (2, 16), (3, 17)]}
CROP_HFOV = 2 * np.degrees(np.arctan((960 / 2) / ((1280 / 2) / np.tan(np.radians(35)))))
K = 10


def load_crop(path):
    im = Image.open(path).convert('L')
    w, h = im.size
    ww = int(round(h * 4 / 3))
    return im.crop(((w - ww) // 2, 0, (w - ww) // 2 + ww, h))


def stage_desc():
    import nets
    from aggregators import GeMHead, NetVLADHead, OTHead
    torch.set_grad_enabled(False)
    dev = torch.device(args.device)
    ds = nets.DinoSalad().to(dev); bb = ds.model; K0 = len(bb.blocks) - 4
    DS = (210, 280)

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
        return h.to(dev).eval()

    frozen = {n: build_head(f'{args.frozen_dir}/{n if os.path.isdir(f"{args.frozen_dir}/{n}") else "ft1e-4_" + n}')
              for n in args.frozen.split(',') if n}
    ft = {}
    for n in [n for n in args.ft.split(',') if n]:
        suf = torch.nn.ModuleList([copy.deepcopy(b) for b in bb.blocks[K0:]] + [copy.deepcopy(bb.norm)])
        sd = torch.load(f'{args.ft_dir}/{n}/backbone_ft.pt', map_location='cpu')
        for k, mod in enumerate(suf[:-1]):
            mod.load_state_dict({kk.split('.', 2)[2]: v.float() for kk, v in sd.items() if kk.startswith(f'blocks.{K0 + k}.')})
        suf[-1].load_state_dict({kk[len('norm.'):]: v.float() for kk, v in sd.items() if kk.startswith('norm.')})
        ft[n] = (suf.to(dev).eval(), build_head(f'{args.ft_dir}/{n}'))
    sp_net = nets.superpoint().to(dev)

    def tokens(z, blocks, norm):
        for blk in blocks:
            z = blk(z)
        z = norm(z)
        B = z.shape[0]
        return z[:, 1:].reshape(B, DS[0] // 14, DS[1] // 14, -1).permute(0, 3, 1, 2), z[:, 0]

    os.makedirs(args.feat, exist_ok=True)
    for d in sorted(os.listdir(args.ext), key=lambda s: int(s.split('_')[1])):
        out = f'{args.feat}/{d}.pt'
        if os.path.exists(out) or not os.path.exists(f'{args.ext}/{d}/meta.json'):
            continue
        meta = json.load(open(f'{args.ext}/{d}/meta.json'))
        D = {k: [] for k in ['salad_zs', *frozen, *ft]}
        sps = []
        names = [m['file'] for m in meta['images']]
        for i in range(0, len(names), 32):
            crops = [load_crop(f'{args.ext}/{d}/img/{n}') for n in names[i:i + 32]]
            imgs = [np.array(c.resize((DS[1] * 2, DS[0] * 2), Image.BICUBIC)) for c in crops]
            x = nets.to_rgb_tensor(imgs, DS).to(dev)
            z = bb.prepare_tokens_with_masks(x)
            for blk in bb.blocks[:K0]:
                z = blk(z)
            f, t = tokens(z, bb.blocks[K0:], bb.norm)
            D['salad_zs'].append(ds.agg((f, t)).float().cpu())
            for n, h in frozen.items():
                D[n].append(h((f, t)).float().cpu())
            for n, (suf, h) in ft.items():
                D[n].append(h(tokens(z, suf[:-1], suf[-1])).float().cpu())
            for c in crops:
                g = torch.from_numpy(np.array(c.resize((320, 240), Image.BICUBIC))).float().div(255.)[None, None].to(dev)
                o = sp_net({'image': g})
                sps.append({'kp': o['keypoints'][0].half().cpu(), 'sc': o['keypoint_scores'][0].half().cpu(),
                            'desc': o['descriptors'][0].half().cpu()})
        torch.save({'names': names, 'desc': {k: torch.cat(v).numpy() for k, v in D.items()}, 'sp': sps}, out)
        print(d, len(names), 'images', flush=True)


# ------------------------------------------------------------------------------------------------ eval
def footprint(x, y, yaw, step=0.25, rmin=1.0, rmax=15.0):
    """Grid cells (step m) of the ground seen by a camera: range rmin..rmax, within the crop's field of view."""
    r = np.arange(rmin, rmax, step / 2); a = np.radians(np.linspace(-CROP_HFOV / 2, CROP_HFOV / 2, 60)) + np.radians(yaw)
    rr, aa = np.meshgrid(r, a)
    return set(zip(np.floor((x + rr * np.cos(aa)) / step).astype(int).ravel(), np.floor((y + rr * np.sin(aa)) / step).astype(int).ravel()))


def positives(mq, md):
    qx = np.array([[m['x'], m['y']] for m in mq]); dx = np.array([[m['x'], m['y']] for m in md])
    qy = np.array([m['yaw'] for m in mq]); dy = np.array([m['yaw'] for m in md])
    dist = np.linalg.norm(qx[:, None] - dx[None], axis=-1)
    dyaw = np.abs((qy[:, None] - dy[None] + 180) % 360 - 180)
    pos = (dist <= 1.0) & (dyaw <= 30)
    fq = [footprint(m['x'], m['y'], m['yaw']) for m in mq]; fd = [footprint(m['x'], m['y'], m['yaw']) for m in md]
    ov = np.zeros(dist.shape, bool)
    for i in range(len(mq)):
        for j in np.where(dist[i] < 30)[0]:
            ov[i, j] = len(fq[i] & fd[j]) / len(fq[i] | fd[j]) >= 0.3
    return pos, ov


def local_scores(spec):
    """Local scores for (query traverse, db traverse, list of (qi, di)) pairs."""
    tq, td, prs = spec
    torch.set_num_threads(1)
    from rerank_lib import topk_kp, mnn_matches, mutual_matches, fgw_match, clipper_matches, ransac_inliers
    Fq, Fd = torch.load(f'{args.feat}/traverse_{tq}.pt', weights_only=False), torch.load(f'{args.feat}/traverse_{td}.pt', weights_only=False)
    Mq = json.load(open(f'{args.ext}/traverse_{tq}/meta.json'))['images']
    Md = json.load(open(f'{args.ext}/traverse_{td}/meta.json'))['images']
    out = {}
    for qi, di in prs:
        kq, _, dq = topk_kp(Fq['sp'][qi], 384)
        kd, _, dd = topk_kp(Fd['sp'][di], 384)
        s = {}
        mm = mnn_matches(dq, dd)
        s['mnn_ransac'] = ransac_inliers(kq[mm[:, 0]].numpy(), kd[mm[:, 1]].numpy()) if len(mm) else 0
        s['fgw'] = len(mutual_matches(fgw_match(kq, dq, kd, dd)))
        s['clipper16'] = len(clipper_matches(kq, dq, kd, dd, eps=16.0))
        bq, bd = np.array(Mq[qi]['boulders']), np.array(Md[di]['boulders'])
        for e in (0.05, 0.1, 0.2, 0.4):
            key = f'boulder_e{e:g}'
            if len(bq) < 3 or len(bd) < 3:
                s[key] = 0; continue
            s[key] = len(clipper_matches(None, torch.ones(len(bq), 1), None, torch.ones(len(bd), 1),
                                         P1=torch.from_numpy(bq), P2=torch.from_numpy(bd), eps=e, topk=len(bd), simthr=-1))
        out[(qi, di)] = s
    return tq, td, out


def stage_eval():
    meta = {k: json.load(open(f'{args.ext}/traverse_{k}/meta.json'))['images'] for kind in PAIRS for p in PAIRS[kind] for k in p}
    feat = {k: torch.load(f'{args.feat}/traverse_{k}.pt', weights_only=False) for k in meta}
    models = list(next(iter(feat.values()))['desc'])
    runs = []                                     # (kind, q traverse, db traverse)
    for kind, ps in PAIRS.items():
        for a, b in ps:
            runs += [(kind, b, a), (kind, a, b)]
    cand, pos, ovl = {}, {}, {}
    need = {}
    for kind, tq, td in runs:
        pos[(tq, td)], ovl[(tq, td)] = positives(meta[tq], meta[td])
        for n in models:
            Xq, Xd = feat[tq]['desc'][n], feat[td]['desc'][n]
            sim = Xq @ Xd.T
            c = np.argsort(-sim, 1)[:, :K]
            cand[(tq, td, n)] = (c, np.take_along_axis(sim, c, 1))
            need.setdefault((tq, td), set()).update((int(i), int(j)) for i in range(len(c)) for j in c[i])
    specs = []
    for (tq, td), prs in need.items():
        prs = sorted(prs)
        specs += [(tq, td, prs[i:i + 400]) for i in range(0, len(prs), 400)]
    LS = {}
    t0 = time.time()
    with Pool(args.procs) as p:
        for tq, td, o in p.imap_unordered(local_scores, specs):
            LS.setdefault((tq, td), {}).update(o)
            print(f'scored {tq}->{td} [{time.time() - t0:.0f}s]', flush=True)
    torch.save({'cand': cand, 'pos': pos, 'ovl': ovl, 'LS': LS, 'runs': runs, 'models': models}, f'{args.out}/lunarloc_eval.pt')
    report()


def report():
    import re
    from collections import defaultdict
    E = torch.load(f'{args.out}/lunarloc_eval.pt', weights_only=False)
    cand, pos, ovl, LS, runs, models = E['cand'], E['pos'], E['ovl'], E['LS'], E['runs'], E['models']
    sim_p = {k: (v['beta'], v['kappa']) for k, v in json.load(open(args.sim_params)).items()}
    sim_p['clipper16'] = (1, 80)
    fam = defaultdict(list)
    for n in models:
        fam[re.sub(r'_s\d+$', '', n)].append(n)
    BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (1, 2, 5, 10, 20, 40, 80)

    def hits(kind, n, P, scorer=None, bk=None, kinds=None):
        h1, h5, valid, block = [], [], [], []
        for k2, tq, td in runs:
            if k2 != kind:
                continue
            c, g = cand[(tq, td, n)]
            if scorer is None:
                sc = g
            else:
                l = np.array([[LS[(tq, td)][(int(i), int(j))][scorer] for j in c[i]] for i in range(len(c))], float)
                sc = g + bk[0] * l / (l + bk[1])
            o = np.argsort(-sc, 1, kind='stable')
            pp = P[(tq, td)]
            hc = np.take_along_axis(np.take_along_axis(pp, c, 1), o, 1)
            h1.append(hc[:, 0]); h5.append(hc[:, :5].any(1)); valid.append(pp.any(1))
            block.append([f'{tq}-{td}-{i // 10}' for i in range(len(c))])
        return np.concatenate(h1), np.concatenate(h5), np.concatenate(valid), np.concatenate(block)

    rng = np.random.default_rng(0)

    def ci(x, blocks):
        ub = np.unique(blocks); by = {b: np.where(blocks == b)[0] for b in ub}
        s = [x[np.concatenate([by[b] for b in rng.choice(ub, len(ub))])].mean() * 100 for _ in range(1000)]
        return [round(float(x.mean() * 100), 3), round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]

    # boulder CLIPPER: tolerance and fusion on the straight-line pairs (development set)
    best = None
    for e in (0.05, 0.1, 0.2, 0.4):
        for b in BET:
            for k in KAP:
                v = np.mean([hits('straight', n, pos, f'boulder_e{e:g}', (b, k))[0][hits('straight', n, pos)[2]].mean()
                             for n in fam['galatea']])
                if best is None or v > best[0]:
                    best = (v, e, b, k)
    sim_p['boulder'] = (best[2], best[3])
    bkey = f'boulder_e{best[1]:g}'
    out = {'pairs': PAIRS, 'crop_hfov_deg': CROP_HFOV, 'boulder_dev': {'eps': best[1], 'beta': best[2], 'kappa': best[3]},
           'kinds': {}}
    for kind in PAIRS:
        for defn, P in (('pose', pos), ('overlap', ovl)):
            if defn == 'overlap' and kind != 'flipped':
                continue
            res = {}
            nq = int(sum(len(pos[(tq, td)]) for k2, tq, td in runs if k2 == kind))
            nv = int(sum(P[(tq, td)].any(1).sum() for k2, tq, td in runs if k2 == kind))
            chance = [P[(tq, td)][P[(tq, td)].any(1)].sum(1) / P[(tq, td)].shape[1] for k2, tq, td in runs if k2 == kind]
            print(f'== {kind} ({defn}): {nv} of {nq} queries have a positive', flush=True)
            if nv == 0:
                out['kinds'][f'{kind}_{defn}'] = {'n_queries': nq, 'n_valid': 0}
                continue
            for f, names in fam.items():
                hh = [hits(kind, n, P) for n in names]
                valid, blocks = hh[0][2], hh[0][3]
                g1 = np.mean([h[0] for h in hh], 0)[valid]; g5 = np.mean([h[1] for h in hh], 0)[valid]
                r = {'R1': ci(g1, blocks[valid]), 'R5': ci(g5, blocks[valid]), 'fused': {}}
                for s, key in (('mnn_ransac', 'mnn_ransac'), ('fgw', 'fgw'), ('clipper16', 'clipper16'), ('boulder', bkey)):
                    f1 = np.mean([hits(kind, n, P, key, sim_p[s])[0] for n in names], 0)[valid]
                    r['fused'][s] = {'R1': ci(f1, blocks[valid]), 'd': ci(f1 - g1, blocks[valid])}
                res[f] = r
            out['kinds'][f'{kind}_{defn}'] = {'n_queries': nq, 'n_valid': nv,
                                              'chance_R1': float(100 * np.concatenate(chance).mean()), 'families': res}
            for f, r in res.items():
                print(f'{f:12s} R@1 {r["R1"][0]:5.1f} [{r["R1"][1]:.1f},{r["R1"][2]:.1f}]  ' +
                      '  '.join(f'{s}: {v["R1"][0]:.1f} ({v["d"][0]:+.1f})' for s, v in r['fused'].items()), flush=True)
    json.dump(out, open(f'{args.out}/lunarloc_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    stage_desc() if args.stage == 'desc' else stage_eval()
