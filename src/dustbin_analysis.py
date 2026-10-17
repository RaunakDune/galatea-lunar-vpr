"""How much mass does the dust-bin absorb, and does it track uninformative content (sky)?
Compares a balanced (SALAD) head and a Galatea (lambda<1) head on the test images."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import os, sys, json, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from aggregators import OTHead
torch.set_grad_enabled(False); torch.set_num_threads(1)
ROOT = os.path.join(paths.DATA, 'test')


def head(run):
    a = json.load(open(f'{paths.RESULTS}/runs/{run}/results.json'))['info']['args']
    n = OTHead(m=a['m'], l=a['l'], iters=a['iters'], mode=a['mode'], lam0=a['lam0'],
               dustbin=not a['no_dustbin'], mass_per_cluster=a['mpc'])
    n.load_state_dict(torch.load(f'{paths.RESULTS}/runs/{run}/head.pt')); n.eval()
    return n


runs = sys.argv[1:] or ['salad_s0', 'galatea_s0']
H = {r: head(r) for r in runs}
out = {r: {'dust_frac': [], 'sky_frac': [], 'cluster_mass_cv': [], 'sky_tok_dust': [], 'gnd_tok_dust': []} for r in runs}
for d in sorted(glob.glob(f'{ROOT}/t*')):
    F = torch.load(f'{d}/feats.pt', weights_only=False, mmap=True)
    names = F['names']
    idx = np.arange(0, len(names), 3)                       # subsample for speed
    for i0 in range(0, len(idx), 64):
        ii = idx[i0:i0 + 64]
        f = F['tokens'][ii].float()
        imgs = [np.array(Image.open(f'{d}/{names[i]}')) for i in ii]
        # sky mask at token resolution (15x20): sky pixels are ~black
        sky = []
        for im in imgs:
            m = (im < 10).astype(float)
            m = torch.tensor(m)[None, None]
            sky.append(torch.nn.functional.adaptive_avg_pool2d(m, (15, 20)).flatten() > 0.5)
        sky = torch.stack(sky)
        for r, net in H.items():
            _, P = net.assignment(f)                        # B, m+1, n (rows: clusters+bin)
            n = P.shape[-1]
            dust = P[:, -1, :]                              # per-token dust mass (tokens sum to 1)
            out[r]['dust_frac'] += (dust.sum(1) / n).tolist()
            out[r]['sky_frac'] += sky.float().mean(1).tolist()
            cm = P[:, :-1, :].sum(2)
            out[r]['cluster_mass_cv'] += (cm.std(1) / cm.mean(1)).tolist()
            for b in range(len(ii)):
                if sky[b].any():
                    out[r]['sky_tok_dust'].append(float(dust[b][sky[b]].mean()))
                out[r]['gnd_tok_dust'].append(float(dust[b][~sky[b]].mean()))
res = {}
for r in runs:
    o = {k: np.array(v) for k, v in out[r].items()}
    c = np.corrcoef(o['sky_frac'], o['dust_frac'])[0, 1]
    res[r] = {'dust_frac_mean': float(o['dust_frac'].mean()), 'dust_frac_std': float(o['dust_frac'].std()),
              'dust_frac_min': float(o['dust_frac'].min()), 'dust_frac_max': float(o['dust_frac'].max()),
              'corr_sky_dust': float(c), 'sky_tok_dust': float(o['sky_tok_dust'].mean()),
              'gnd_tok_dust': float(o['gnd_tok_dust'].mean()), 'cluster_mass_cv': float(o['cluster_mass_cv'].mean()),
              'n_images': int(len(o['dust_frac']))}
    print(r, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res[r].items()}))
json.dump(res, open(os.path.join(paths.RESULTS, 'dustbin_analysis.json'), 'w'), indent=1)
