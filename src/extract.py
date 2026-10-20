"""Extract frozen features for every image of every terrain directory.
Outputs (per terrain dir) feats.pt with:
  names, tokens (fp16 [N,768,15,20]), cls (fp16 [N,768]), salad_zs ([N,8448])
  and for val/test additionally: cosplace, eigen, sp (SuperPoint kps/desc/scores lists)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, json, glob, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import nets as models
from PIL import Image

torch.set_num_threads(int(os.environ.get('NT', 2)))
torch.set_grad_enabled(False)
DEV = torch.device(os.environ.get('DEV', 'cpu'))   # e.g. DEV=cuda:1
ROOT = paths.DATA
DINO_SIZE = (210, 280)   # 15 x 20 patches of 14 px


def load(d, names):
    return [np.array(Image.open(f'{d}/{n}')) for n in names]


def run(d, nets, bs=16):
    out_f = f'{d}/feats.pt'
    if os.path.exists(out_f):
        return 'skip'
    meta = json.load(open(f'{d}/meta.json'))
    names = [m['file'] for m in meta['db']] + [m['file'] for m in meta['q']]
    split = meta['split']
    R = {'names': names, 'tokens': [], 'cls': [], 'salad_zs': []}
    full = split in ('val', 'test')
    if full:
        R.update({'cosplace': [], 'eigen': [], 'sp': []})
    t0 = time.time()
    for i in range(0, len(names), bs):
        imgs = load(d, names[i:i + bs])
        x = models.to_rgb_tensor(imgs, DINO_SIZE).to(DEV)
        f, t = nets['dino'].forward_tokens(x)
        R['salad_zs'].append(nets['dino'].agg((f, t)).cpu())
        R['tokens'].append(f.half().cpu()); R['cls'].append(t.half().cpu())
        if full:
            xr = models.to_rgb_tensor(imgs).to(DEV)
            R['cosplace'].append(nets['cos'](xr).cpu()); R['eigen'].append(nets['eig'](xr).cpu())
            with torch.no_grad():
                g = torch.from_numpy(np.stack(imgs)).float().div(255.)[:, None].to(DEV)
                for k in range(g.shape[0]):
                    o = nets['sp']({'image': g[k:k + 1]})
                    R['sp'].append({'kp': o['keypoints'][0].half().cpu(), 'sc': o['keypoint_scores'][0].half().cpu(),
                                    'desc': o['descriptors'][0].half().cpu()})
    for k in ('tokens', 'cls', 'salad_zs', 'cosplace', 'eigen'):
        if k in R:
            R[k] = torch.cat(R[k])
    torch.save(R, out_f)
    return f'{d}: {len(names)} imgs in {time.time() - t0:.0f}s'


if __name__ == '__main__':
    nets = {'dino': models.DinoSalad(), 'cos': models.cosplace(), 'eig': models.eigenplaces(),
            'sp': models.superpoint()}
    nets = {k: v.to(DEV) for k, v in nets.items()}
    dirs = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob(f'{ROOT}/*/t*'))
    for d in dirs:
        if not os.path.exists(f'{d}/meta.json'):
            print('not ready', d); continue
        print(run(d, nets), flush=True)
