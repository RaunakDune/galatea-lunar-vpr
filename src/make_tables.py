"""Aggregate all results into LaTeX tables, a numbers file (macros used in the text), and figures.
Every number in the paper's experiments section is produced here from the raw result files."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CONDS

R = paths.RESULTS
P = paths.PAPER
SH = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
SHW = ['Czero', 'Cone', 'Ctwo', 'Cthree', 'Cfour', 'Cfive', 'Csix']
macros = {}


def M(name, val, fmt='{:.1f}'):
    macros[name] = fmt.format(val) if not isinstance(val, str) else val


RG = os.path.join(R, 'gpu')          # GPU runs (backbone fine-tuning, cross-hardware reproduction)


def load_run(name, split='test', root=None):
    f = f'{root or R}/runs/{name}/results.json'
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return j['results'].get(split), j['info']


def agg(prefix, seeds=(0, 1, 2), split='test', root=None):
    rs = [load_run(f'{prefix}_s{s}', split, root) for s in seeds]
    rs = [r for r in rs if r is not None and r[0] is not None]
    if not rs:
        return None
    keys = ['R@1', 'R@5', 'R@10'] + [f'R@1_{c}' for c in CONDS]
    out = {k: (float(np.mean([r[0][k] for r in rs])), float(np.std([r[0][k] for r in rs])), len(rs)) for k in keys}
    out['info'] = rs[0][1]
    out['lams'] = [r[1].get('lam') for r in rs]
    out['dust'] = [r[1].get('dustbin') for r in rs]
    out['hits'] = {N: np.mean([np.array(r[0][f'hits_R{N}']) for r in rs], 0) for N in (1, 5, 10)}
    out['n'] = len(rs)
    return out


def cell(m, s=None, n=1, bold=False):
    t = f'{m:.1f}' if (s is None or n < 2) else f'{m:.1f}{{\\scriptsize$\\pm${s:.1f}}}'
    return f'\\textbf{{{t}}}' if bold else t


def paired_boot(a, b, B=5000, seed=0):
    """95% CI of mean(a-b) (percentage points) by resampling queries."""
    rng = np.random.default_rng(seed)
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100
    n = len(d)
    s = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


# ------------------------------------------------------------------ main table
zs = json.load(open(f'{R}/zeroshot_test_val.json'))
rows = []
for key, lab, dim in [('CosPlace-R50', 'CosPlace-R50~\\cite{berton2022cosplace}', 512),
                      ('EigenPlaces-R50', 'EigenPlaces-R50~\\cite{berton2023eigenplaces}', 2048),
                      ('DINOv2-SALAD', 'DINOv2-SALAD~\\cite{izquierdo2024optimal}', 8448)]:
    r = zs[f'test/{key}']
    rows.append(('zs', lab, dim, {k: (r[k], None, 1) for k in r if k.startswith('R@') and ('_' not in k or k.startswith('R@1_'))},
                 {N: np.array(r[f'hits_R{N}']) for N in (1, 5, 10)}))
heads = {}
for pre, lab in [('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('lam0', 'OT, $\\lambda{=}0$'), ('salad', 'OT, $\\lambda{=}1$ (SALAD)'),
                 ('galatea', 'OT, learned $\\lambda$')]:
    a = agg(pre)
    if a:
        heads[pre] = a
        rows.append(('tr', lab, a['info']['desc_dim'], a, a['hits']))

ftheads = {}
for pre, lab in [('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('lam0', 'OT, $\\lambda{=}0$'), ('salad', 'OT, $\\lambda{=}1$ (SALAD)'),
                 ('galatea', 'OT, learned $\\lambda$')]:
    a = agg('ft1e-4_' + pre, root=RG)
    if a:
        ftheads[pre] = a
        rows.append(('ft', lab, a['info']['desc_dim'], a, a['hits']))

best = {k: max(r[3][k][0] for r in rows) for k in ['R@1', 'R@5', 'R@10'] + [f'R@1_{c}' for c in CONDS]}
L = ['\\begin{table*}[t]', '\\centering',
     '\\caption{Recall (\\%) on the LunarSim-VPR test set (3 unseen scenes, 3150 queries). Lunar-trained heads share the '
     'DINOv2 backbone and training protocol; \\emph{frozen}: only the head is trained; \\emph{fine-tuned}: the last four '
     'backbone blocks are trained jointly with the head. Mean$\\pm$std over 3 seeds. Per-condition columns give R@1 '
     '(conditions in \\cref{tab:conds}).}', '\\label{tab:main}', '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
     '\\begin{tabular}{@{}llr ccc ccccccc@{}}', '\\toprule',
     '& Method & Dim. & R@1 & R@5 & R@10 & ' + ' & '.join(SH) + ' \\\\', '\\midrule']
for i, (kind, lab, dim, d, _) in enumerate(rows):
    if i in (3, 8):
        L.append('\\midrule')
    grp = {0: '\\multirow{3}{*}{\\rotatebox{90}{\\scriptsize zero-shot}}',
           3: '\\multirow{5}{*}{\\rotatebox{90}{\\scriptsize frozen}}',
           8: '\\multirow{5}{*}{\\rotatebox{90}{\\scriptsize fine-tuned}}'}.get(i, '')
    cells = [cell(*d[k], bold=abs(d[k][0] - best[k]) < 0.05) for k in ['R@1', 'R@5', 'R@10']]
    cells += [cell(d[f'R@1_{c}'][0], bold=abs(d[f'R@1_{c}'][0] - best[f'R@1_{c}']) < 0.05) for c in CONDS]
    L.append(f'{grp} & {lab} & {dim} & ' + ' & '.join(cells) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
open(f'{P}/tab_main.tex', 'w').write('\n'.join(L))

for key, nm in [('DINOv2-SALAD', 'zsSalad'), ('CosPlace-R50', 'zsCos'), ('EigenPlaces-R50', 'zsEig')]:
    r = zs[f'test/{key}']
    for k in ['R@1', 'R@5', 'R@10']:
        M(nm + k.replace('@', '').replace('1', 'one').replace('5', 'five').replace('0', 'zero'), r[k])
    for c, s in zip(CONDS, SHW):
        M(nm + s, r[f'R@1_{c}'], '{:.0f}')
for pre, nm in [('gem', 'gem'), ('netvlad', 'nv'), ('salad', 'sal'), ('galatea', 'gal')]:
    if pre in heads:
        a = heads[pre]
        for k in ['R@1', 'R@5', 'R@10']:
            kk = k.replace('@', '').replace('1', 'one').replace('5', 'five').replace('0', 'zero')
            M(nm + kk, a[k][0]); M(nm + kk + 'sd', a[k][1])
        for c, s in zip(CONDS, SHW):
            M(nm + s, a[f'R@1_{c}'][0], '{:.0f}')
if 'galatea' in heads:
    M('galLam', np.mean(heads['galatea']['lams']), '{:.2f}')
    zsS = zs['test/DINOv2-SALAD']
    d, lo, hi = paired_boot(heads['galatea']['hits'][1], np.array(zsS['hits_R1']))
    M('gainZsMean', d); M('gainZsLo', lo); M('gainZsHi', hi)
    if 'salad' in heads:
        d, lo, hi = paired_boot(heads['galatea']['hits'][1], heads['salad']['hits'][1])
        M('galVsSalMean', d, '{:+.1f}'); M('galVsSalLo', lo, '{:+.1f}'); M('galVsSalHi', hi, '{:+.1f}')
    # relative drop C0 -> C6
    for nm, src in [('zsSalad', {c: zsS[f'R@1_{c}'] for c in CONDS}),
                    ('gal', {c: heads['galatea'][f'R@1_{c}'][0] for c in CONDS})]:
        hard = np.mean([src[c] for c in ['C2_el6', 'C5_az+180', 'C6_az+180_el8']])
        M(nm + 'Hard', hard)

# ------------------------------------------------------------------ ablations (lambda etc.)
abl = []
def addabl(label, prefix, seeds=(0, 1, 2)):
    a = agg(prefix, seeds)
    if a:
        abl.append((label, a))
    return a
addabl('$\\lambda=0$ (NetVLAD assignment + dust-bin)', 'lam0')
addabl('$\\lambda=0.25$ (fixed)', 'lam0.25', (0,))
addabl('$\\lambda$ learned (Galatea, init.\\ 0.5)', 'galatea')
addabl('$\\lambda$ learned, $25\\times$ lr on $\\lambda,z$', 'galatea_fast', (0, 1, 2))
addabl('$\\lambda=0.75$ (fixed)', 'lam0.75', (0,))
addabl('$\\lambda=1$ (SALAD, balanced)', 'salad')
abl2 = []
for label, prefix in [('$L=1$ Sinkhorn iteration', 'galatea_L1'),
                      ('$L=10$ iterations', 'galatea_L10'), ('$m=16$ clusters', 'galatea_m16'),
                      ('$m=32$ clusters', 'galatea_m32'),
                      ('single-illumination training', 'galatea_fixedillum')]:
    a = agg(prefix, (0,))
    if a:
        abl2.append((label, a))
ref0 = agg('galatea', (0,))
L = ['\\begin{table}[t]', '\\centering',
     '\\caption{Ablations of the Galatea head (test, \\%, seed~0; hard = mean R@1 over C2, C5, C6). '
     'Single-illumination: the training scenes re-rendered with the same poses under the database Sun only.}',
     '\\label{tab:ablation}', '\\footnotesize', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lccc@{}}',
     '\\toprule', 'Variant & R@1 & R@5 & hard \\\\', '\\midrule']
def hard(a):
    return np.mean([a[f'R@1_{c}'][0] for c in ['C2_el6', 'C5_az+180', 'C6_az+180_el8']])
if ref0:
    L.append(f'Frozen backbone, reference & {ref0["R@1"][0]:.1f} & {ref0["R@5"][0]:.1f} & {hard(ref0):.1f} \\\\')
    for label, a in abl2:
        L.append(f'\\quad {label} & {a["R@1"][0]:.1f} & {a["R@5"][0]:.1f} & {hard(a):.1f} \\\\')
    sfa = agg('salad_fixedillum', (0,))
    if sfa:
        L.append(f'\\quad SALAD, single-illumination & {sfa["R@1"][0]:.1f} & {sfa["R@5"][0]:.1f} & {hard(sfa):.1f} \\\\')
_f0 = agg('ft1e-4_galatea', (0,), root=RG)
if _f0:
    L.append('\\midrule')
    L.append(f'Fine-tuned backbone, reference & {_f0["R@1"][0]:.1f} & {_f0["R@5"][0]:.1f} & {hard(_f0):.1f} \\\\')
    for lab_, pre_ in [('single-illumination training', 'ft1e-4_galatea_fixedillum'), ('SALAD, single-illumination', 'ft1e-4_salad_fixedillum')]:
        a = agg(pre_, (0,), root=RG)
        if a:
            L.append(f'\\quad {lab_} & {a["R@1"][0]:.1f} & {a["R@5"][0]:.1f} & {hard(a):.1f} \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
open(f'{P}/tab_ablation.tex', 'w').write('\n'.join(L))
for label, a in abl + abl2:
    pass
lamtab = {lab: a for lab, a in abl}
for prefix, nm in [('lam0', 'lamZero'), ('galatea_fast', 'galFast'), ('galatea_fixedillum', 'fixIll'),
                   ('galatea_nodust', 'noDust'), ('galatea_L1', 'Lone'), ('galatea_L10', 'Lten'),
                   ('galatea_m16', 'mSixteen'), ('galatea_m32', 'mThirtytwo'), ('lam0.25', 'lamQuarter'), ('lam0.75', 'lamThreeq')]:
    a = agg(prefix, (0, 1, 2))
    if a:
        M(nm + 'Rone', a['R@1'][0]); M(nm + 'Hard', hard(a))
        if a['lams'][0] is not None:
            M(nm + 'Lam', np.mean(a['lams']), '{:.2f}')
if ref0:
    M('galSzeroRone', ref0['R@1'][0]); M('galSzeroHard', hard(ref0))
fi = agg('galatea_fixedillum', (0,))
if fi and ref0:
    M('fixIllDrop', ref0['R@1'][0] - fi['R@1'][0])

# ------------------------------------------------------------------ dust-bin analysis
if os.path.exists(f'{R}/dustbin_analysis.json'):
    db = json.load(open(f'{R}/dustbin_analysis.json'))
    if 'salad_s0' in db:
        M('salDust', 100 * db['salad_s0']['dust_frac_mean']); M('salCV', db['salad_s0']['cluster_mass_cv'], '{:.2f}')
        M('salSkyCorr', db['salad_s0']['corr_sky_dust'], '{:.2f}')
    if 'galatea_s0' in db:
        M('galDust', 100 * db['galatea_s0']['dust_frac_mean']); M('galCV', db['galatea_s0']['cluster_mass_cv'], '{:.1f}')

open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
print(f'{len(macros)} macros written'); print(open(f'{P}/numbers.tex').read())

# ------------------------------------------------------------------ re-ranking (stage 2)
from analyze_rerank import recall_after, tune
VALF = f'{R}/rerank/galatea_s0_val_mnn-uot-fgw-lg_a0.3_c0.25.npy'
TESTF = f'{R}/rerank/galatea_s0_test_mnn-uot-fgw-lg_a0.3_c0.25.npy'
if os.path.exists(VALF) and os.path.exists(TESTF):
    sv = np.load(VALF, allow_pickle=True).item()
    st = np.load(TESTF, allow_pickle=True).item()
    g = recall_after(st, 'test', None)
    rows = [('Global only (Galatea stage 1)', g, g, None)]
    M('rrGlobRone', g['R@1']); M('rrGlobRfive', g['R@5'])
    names = [('mnn', 'MNN'), ('uot', 'OT, appearance only'), ('lg', 'LightGlue~\\cite{lindenberger2023lightglue} (learned)'),
             ('fgw', 'Partial FGW (ours)')]
    BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (5, 10, 20, 40, 80, 160, 320, 640)
    rr = {}
    for m, lab in names:
        for suf, sl in [('', ''), ('_ransac', ' + MAGSAC++')]:
            key = m + suf
            b, k = tune(sv, 'val', key, BET, KAP)
            loc = recall_after(st, 'test', key, 'local')
            fus = recall_after(st, 'test', key, 'fused', b, k)
            rr[key] = (loc, fus, b, k)
            rows.append((lab + sl, loc, fus, (b, k)))
    bestf = max(r[2]['R@1'] for r in rows[1:])
    L = ['\\begin{table}[t]', '\\centering',
         '\\caption{Stage-2 re-ranking of the top-10 Galatea candidates (test, \\%). \\emph{Local}: sort by the local '
         'score only. \\emph{Fused}: \\cref{eq:fusion} with $(\\beta,\\kappa)$ chosen on the validation scene.}',
         '\\label{tab:rerank}', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}lcccc@{}}', '\\toprule', '& \\multicolumn{2}{c}{Local} & \\multicolumn{2}{c}{Fused} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}', 'Local scorer & R@1 & R@5 & R@1 & R@5 \\\\', '\\midrule']
    L.append(f'none (global ranking) & \\multicolumn{{2}}{{c}}{{--}} & {g["R@1"]:.1f} & {g["R@5"]:.1f} \\\\')
    for lab, loc, fus, bk in rows[1:]:
        bf = '\\textbf{%.1f}' % fus['R@1'] if abs(fus['R@1'] - bestf) < 0.05 else '%.1f' % fus['R@1']
        L.append(f'{lab} & {loc["R@1"]:.1f} & {loc["R@5"]:.1f} & {bf} & {fus["R@5"]:.1f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(f'{P}/tab_rerank.tex', 'w').write('\n'.join(L))
    for key, nm in [('fgw', 'rrFgw'), ('lg', 'rrLg'), ('uot', 'rrUot'), ('mnn', 'rrMnn'), ('lg_ransac', 'rrLgR'), ('fgw_ransac', 'rrFgwR')]:
        loc, fus, b, k = rr[key]
        M(nm + 'Rone', fus['R@1']); M(nm + 'Rfive', fus['R@5']); M(nm + 'LocRone', loc['R@1'])
        M(nm + 'Beta', b, '{:g}'); M(nm + 'Kappa', k, '{:g}')
        for c, s in zip(CONDS, SHW):
            M(nm + s, fus[f'R@1_{c}'], '{:.0f}')
        M(nm + 'Hard', np.mean([fus[f'R@1_{c}'] for c in ['C2_el6', 'C5_az+180', 'C6_az+180_el8']]))
    d, lo, hi = paired_boot(rr['fgw'][1]['hits1'], g['hits1'])
    M('rrGainMean', d); M('rrGainLo', lo); M('rrGainHi', hi)
    d, lo, hi = paired_boot(rr['fgw'][1]['hits1'], rr['lg'][1]['hits1'])
    M('fgwVsLgMean', d, '{:+.1f}'); M('fgwVsLgLo', lo, '{:+.1f}'); M('fgwVsLgHi', hi, '{:+.1f}')
    d, lo, hi = paired_boot(rr['fgw'][1]['hits1'], rr['uot'][1]['hits1'])
    M('fgwVsUotMean', d, '{:+.1f}'); M('fgwVsUotLo', lo, '{:+.1f}'); M('fgwVsUotHi', hi, '{:+.1f}')
    M('fullVsZs', rr['fgw'][1]['R@1'] - zs['test/DINOv2-SALAD']['R@1'])
    json.dump({k: {'local_R1': v[0]['R@1'], 'fused_R1': v[1]['R@1'], 'fused_R5': v[1]['R@5'], 'beta': v[2], 'kappa': v[3]}
               for k, v in rr.items()}, open(f'{R}/rerank_summary.json', 'w'), indent=1)
    open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
    print('rerank tables written')

# ------------------------------------------------------------------ re-ranking zero-shot SALAD (MNN)
ZV, ZT = f'{R}/rerank/salad_zs_val_mnn.npy', f'{R}/rerank/salad_zs_test_mnn.npy'
if os.path.exists(ZV) and os.path.exists(ZT):
    sv = np.load(ZV, allow_pickle=True).item(); st = np.load(ZT, allow_pickle=True).item()
    b, k = tune(sv, 'val', 'mnn_ransac', (0.1, 0.25, 0.5, 1, 2, 4, 8), (5, 10, 20, 40, 80, 160, 320, 640))
    r = recall_after(st, 'test', 'mnn_ransac', 'fused', b, k)
    M('zsRRRone', r['R@1']); M('zsRRRfive', r['R@5'])
    M('zsRRHard', np.mean([r[f'R@1_{c}'] for c in ['C2_el6', 'C5_az+180', 'C6_az+180_el8']]))
    open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
    print('zs rerank macros written')

# ------------------------------------------------------------------ series for the per-condition figure
series = {}
for key in ['CosPlace-R50', 'EigenPlaces-R50', 'DINOv2-SALAD']:
    series[key + ' (zero-shot)'] = [zs[f'test/{key}'][f'R@1_{c}'] for c in CONDS]
for pre, lab in [('netvlad', 'NetVLAD (frozen)'), ('galatea', 'Galatea (frozen)')]:
    a = agg(pre)
    if a:
        series[lab] = [a[f'R@1_{c}'][0] for c in CONDS]
if os.path.exists(TESTF):
    series['Galatea (frozen) + FGW fusion'] = [rr['fgw'][1][f'R@1_{c}'] for c in CONDS]
for pre, lab in [('netvlad', 'NetVLAD (fine-tuned)'), ('galatea', 'Galatea (fine-tuned)')]:
    a = agg('ft1e-4_' + pre, root=RG)
    if a:
        series[lab] = [a[f'R@1_{c}'][0] for c in CONDS]
json.dump(series, open(f'{R}/percond_series.json', 'w'), indent=1)

# ------------------------------------------------------------------ timing
if os.path.exists(f'{R}/timing.json'):
    T = json.load(open(f'{R}/timing.json'))
    for k, nm in [('backbone_ms', 'tBackbone'), ('head_gem_ms', 'tGem'), ('head_netvlad_ms', 'tNv'),
                  ('head_salad_ms', 'tSal'), ('head_galatea_ms', 'tGal'), ('pair_uot_ms', 'tUot'),
                  ('pair_fgw_ms', 'tFgw'), ('pair_lightglue_ms', 'tLg'), ('superpoint_ms', 'tSp')]:
        M(nm, T[k], '{:.1f}' if T[k] < 10 else '{:.0f}')
    M('tFgwK', T['pair_fgw_ms'] * 10, '{:.0f}'); M('tLgK', T['pair_lightglue_ms'] * 10, '{:.0f}')
    open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')

# ------------------------------------------------------------------ multi-seed re-ranking gains
BET = (0.1, 0.25, 0.5, 1, 2, 4, 8); KAP = (5, 10, 20, 40, 80, 160, 320, 640)
ms = {}
for m in ['mnn', 'mnn_ransac', 'fgw']:
    fu, gl = [], []
    for s in [0, 1, 2]:
        tag = 'mnn-uot-fgw-lg' if s == 0 else 'mnn-fgw'
        fv, ft = f'{R}/rerank/galatea_s{s}_val_{tag}_a0.3_c0.25.npy', f'{R}/rerank/galatea_s{s}_test_{tag}_a0.3_c0.25.npy'
        if not (os.path.exists(fv) and os.path.exists(ft)):
            continue
        sv = np.load(fv, allow_pickle=True).item(); st = np.load(ft, allow_pickle=True).item()
        b, k = tune(sv, 'val', m, BET, KAP)
        fu.append(recall_after(st, 'test', m, 'fused', b, k)['R@1']); gl.append(recall_after(st, 'test', None)['R@1'])
    if len(fu) == 3:
        g = np.array(fu) - np.array(gl)
        nm = {'mnn': 'msMnn', 'mnn_ransac': 'msMnnR', 'fgw': 'msFgw'}[m]
        M(nm + 'Rone', np.mean(fu)); M(nm + 'Ronesd', np.std(fu)); M(nm + 'Gain', g.mean()); M(nm + 'Gainsd', g.std())
open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')

# ------------------------------------------------------------------ extra ablation macros
fi = agg('galatea_fixedillum', (0,)); sf = agg('salad_fixedillum', (0,))
if fi:
    M('fixIllCzero', fi['R@1_C0_same'][0], '{:.0f}')
    M('fixIllHardVal', hard(fi))
if sf:
    M('salFixIllRone', sf['R@1'][0])
if ref0:
    M('galSzeroCzero', ref0['R@1_C0_same'][0], '{:.0f}')
open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')

# ------------------------------------------------------------------ joint backbone fine-tuning (GPU)
HARD = ['C2_el6', 'C5_az+180', 'C6_az+180_el8']
ftall = {}
for pre, nm in [('gem', 'Gem'), ('netvlad', 'Nv'), ('salad', 'Sal'), ('galatea', 'Gal'), ('lam0', 'LamZero')]:
    a = agg('ft1e-4_' + pre, root=RG)
    if a:
        ftall[pre] = a
        M('ft' + nm + 'Rone', a['R@1'][0]); M('ft' + nm + 'Ronesd', a['R@1'][1])
        M('ft' + nm + 'Rfive', a['R@5'][0]); M('ft' + nm + 'Hard', hard(a))
        fr = agg(pre)
        if fr:
            M('ft' + nm + 'Gain', a['R@1'][0] - fr['R@1'][0])
        if a['lams'][0] is not None and pre == 'galatea':
            M('ftGalLam', np.mean(a['lams']), '{:.2f}')
for x, y, nm in [('galatea', 'salad', 'ftGalVsSal'), ('netvlad', 'galatea', 'ftNvVsGal'), ('netvlad', 'salad', 'ftNvVsSal'),
                 ('lam0', 'galatea', 'ftLamZeroVsGal')]:
    if x in ftall and y in ftall:
        d, lo, hi = paired_boot(ftall[x]['hits'][1], ftall[y]['hits'][1])
        M(nm + 'Mean', d, '{:+.1f}'); M(nm + 'Lo', lo, '{:+.1f}'); M(nm + 'Hi', hi, '{:+.1f}')
if 'galatea' in ftall:
    d, lo, hi = paired_boot(ftall['galatea']['hits'][1], heads['galatea']['hits'][1])
    M('ftGainMean', d); M('ftGainLo', lo); M('ftGainHi', hi)
sweep = {}
for lr, nm in [('1e-5', 'A'), ('3e-5', 'B'), ('1e-4', 'C'), ('3e-4', 'D'), ('1e-3', 'E')]:
    r = load_run(f'ftsweep_galatea_lr{lr}_s0', 'val', RG)
    if r:
        sweep[lr] = r[0]['R@1']; M('ftSweep' + nm, r[0]['R@1'])
ffi = agg('ft1e-4_galatea_fixedillum', (0,), root=RG); fsi = agg('ft1e-4_salad_fixedillum', (0,), root=RG)
ft0 = agg('ft1e-4_galatea', (0,), root=RG)
if ffi and ft0:
    M('ftFixIllRone', ffi['R@1'][0]); M('ftFixIllHard', hard(ffi)); M('ftFixIllDrop', ft0['R@1'][0] - ffi['R@1'][0])
    M('ftGalSzeroRone', ft0['R@1'][0]); M('ftGalSzeroHard', hard(ft0))
    M('ftFixIllCzero', ffi['R@1_C0_same'][0], '{:.0f}'); M('ftGalSzeroCzero', ft0['R@1_C0_same'][0], '{:.0f}')
if fsi:
    M('ftSalFixIllRone', fsi['R@1'][0])

# fused re-ranking on fine-tuned stage 1
ftrr = {}
for m in ['mnn', 'mnn_ransac', 'fgw']:
    fu, gl, hs = [], [], []
    for s in [0, 1, 2]:
        tag = 'mnn-uot-fgw-lg' if s == 0 else 'mnn-fgw'
        fv = f'{RG}/rerank/ft1e-4_galatea_s{s}_val_{tag}_a0.3_c0.25.npy'
        ft_ = f'{RG}/rerank/ft1e-4_galatea_s{s}_test_{tag}_a0.3_c0.25.npy'
        if not (os.path.exists(fv) and os.path.exists(ft_)):
            continue
        sv = np.load(fv, allow_pickle=True).item(); st = np.load(ft_, allow_pickle=True).item()
        b, k = tune(sv, 'val', m, BET, KAP)
        rf = recall_after(st, 'test', m, 'fused', b, k)
        fu.append(rf['R@1']); gl.append(recall_after(st, 'test', None)['R@1'])
        hs.append(np.mean([rf[f'R@1_{c}'] for c in HARD]))
    if fu:
        nm = {'mnn': 'ftMnn', 'mnn_ransac': 'ftMnnR', 'fgw': 'ftFgw'}[m]
        g = np.array(fu) - np.array(gl)
        M(nm + 'Rone', np.mean(fu)); M(nm + 'Ronesd', np.std(fu)); M(nm + 'Gain', g.mean()); M(nm + 'Gainsd', g.std())
        M(nm + 'Hard', np.mean(hs)); M(nm + 'N', len(fu), '{:d}')
        ftrr[m] = (fu, gl)
fv = f'{RG}/rerank/ft1e-4_netvlad_s0_val_mnn-fgw_a0.3_c0.25.npy'; ft_ = fv.replace('_val_', '_test_')
if os.path.exists(fv) and os.path.exists(ft_):
    sv = np.load(fv, allow_pickle=True).item(); st = np.load(ft_, allow_pickle=True).item()
    for m, nm in [('fgw', 'ftNvFgw'), ('mnn_ransac', 'ftNvMnnR')]:
        b, k = tune(sv, 'val', m, BET, KAP)
        rf = recall_after(st, 'test', m, 'fused', b, k); g = recall_after(st, 'test', None)
        M(nm + 'Rone', rf['R@1']); M(nm + 'Rfive', rf['R@5']); M(nm + 'Gain', rf['R@1'] - g['R@1'])
        M(nm + 'Hard', np.mean([rf[f'R@1_{c}'] for c in HARD]))
    M('ftNvSzeroRone', g['R@1'])

# cross-hardware reproduction of the frozen-backbone runs (GPU, 5 seeds) -> supplement table
rep = []
for pre, lab in [('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('salad', 'SALAD ($\\lambda{=}1$)'),
                 ('galatea', 'Galatea, learned $\\lambda$'), ('lam0', '$\\lambda=0$')]:
    cpu = agg(pre); gpu = agg(pre, (0, 1, 2, 3, 4), root=RG)
    if cpu and gpu:
        rep.append((lab, cpu, gpu))
if rep:
    L = ['\\begin{table}[t]', '\\centering', '\\caption{Cross-hardware reproduction of the frozen-backbone heads '
         '(test R@1, \\%, mean$\\pm$std). CPU: the runs reported in the paper (3 seeds). GPU: the same code and '
         'protocol re-run on NVIDIA Tesla P40 GPUs (5 seeds).}', '\\label{tab:repro}', '\\footnotesize',
         '\\begin{tabular}{@{}lccc@{}}', '\\toprule', 'Head & CPU (3 seeds) & GPU (5 seeds) & GPU hard \\\\', '\\midrule']
    for lab, c, g in rep:
        L.append(f'{lab} & {cell(*c["R@1"])} & {cell(*g["R@1"])} & {hard(g):.1f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(f'{P}/tab_repro.tex', 'w').write('\n'.join(L))
if sweep:
    L = ['\\begin{table}[t]', '\\centering', '\\caption{Backbone learning rate for joint fine-tuning, selected on the '
         'validation scene (Galatea head, seed 0, R@1 \\%).}', '\\label{tab:lrsweep}', '\\footnotesize',
         '\\begin{tabular}{@{}l' + 'c' * len(sweep) + '@{}}', '\\toprule',
         'Backbone lr & ' + ' & '.join('$%s\\cdot10^{-%s}$' % (k.split('e-')[0], k.split('e-')[1]) for k in sweep) + ' \\\\',
         '\\midrule', 'Val.\\ R@1 & ' + ' & '.join(f'{v:.1f}' for v in sweep.values()) + ' \\\\',
         '\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(f'{P}/tab_lrsweep.tex', 'w').write('\n'.join(L))
open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
print('fine-tuning macros written')

# ------------------------------------------------------------------ assignment family: frozen vs fine-tuned (GPU, 3 seeds)
fam = [('$\\lambda=0$, dust-bin', 'lam0'), ('$\\lambda=0$, no dust-bin', 'lam0_nodust'), ('$\\lambda=0.25$', 'lam0.25'),
       ('learned $\\lambda$ (Galatea)', 'galatea'), ('$\\lambda=0.75$', 'lam0.75'), ('$\\lambda=1$ (SALAD), dust-bin', 'salad'),
       ('$\\lambda=1$, no dust-bin', 'salad_nodust'), ('NetVLAD (residual, $k$-means)', 'netvlad')]
L = ['\\begin{table}[t]', '\\centering',
     '\\caption{The assignment family with a frozen and a fine-tuned backbone (test R@1, \\%, mean$\\pm$std over 3 '
     'seeds; hard: mean R@1 over C2, C5, C6). All runs in this table were made on the same GPUs. The OT rows share the '
     'aggregation architecture and differ only in the assignment; NetVLAD aggregates residuals to $k$-means-initialised '
     'centroids and has no dust-bin.}', '\\label{tab:family}', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
     '\\begin{tabular}{@{}lcccc@{}}', '\\toprule', '& \\multicolumn{2}{c}{Frozen} & \\multicolumn{2}{c}{Fine-tuned} \\\\',
     '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}', 'Assignment & R@1 & hard & R@1 & hard \\\\', '\\midrule']
famres = {}
for lab, pre in fam:
    fr = agg(pre, root=RG); ft = agg('ft1e-4_' + pre, root=RG)
    famres[pre] = (fr, ft)
    c1 = f'{cell(*fr["R@1"])} & {hard(fr):.1f}' if fr else '-- & --'
    c2 = f'{cell(*ft["R@1"])} & {hard(ft):.1f}' if ft else '-- & --'
    if pre == 'netvlad':
        L.append('\\midrule')
    L.append(f'{lab} & {c1} & {c2} \\\\')
L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
open(f'{P}/tab_family.tex', 'w').write('\n'.join(L))
fnm = {'lam0': 'LamZero', 'lam0_nodust': 'LamZeroND', 'lam0.25': 'LamQ', 'galatea': 'Gal', 'lam0.75': 'LamTQ',
       'salad': 'Sal', 'salad_nodust': 'SalND', 'netvlad': 'Nv'}
for pre, (fr, ft) in famres.items():
    if fr:
        M('famFr' + fnm[pre], fr['R@1'][0]); M('famFr' + fnm[pre] + 'sd', fr['R@1'][1])
    if ft:
        M('famFt' + fnm[pre], ft['R@1'][0]); M('famFt' + fnm[pre] + 'sd', ft['R@1'][1])
for (x, y, nm) in [('lam0_nodust', 'lam0', 'NDvsD'), ('netvlad', 'lam0_nodust', 'NvVsND'), ('salad_nodust', 'salad', 'SalNDvsD')]:
    for k, idx in [('Fr', 0), ('Ft', 1)]:
        a, b = famres.get(x, (None, None))[idx], famres.get(y, (None, None))[idx]
        if a and b:
            d, lo, hi = paired_boot(a['hits'][1], b['hits'][1])
            M('fam' + k + nm + 'Mean', d, '{:+.1f}'); M('fam' + k + nm + 'Lo', lo, '{:+.1f}'); M('fam' + k + nm + 'Hi', hi, '{:+.1f}')
lamfr = [famres[p][0]['R@1'][0] for p in ['lam0', 'lam0.25', 'galatea', 'lam0.75', 'salad'] if famres.get(p) and famres[p][0]]
if len(lamfr) == 5:
    M('famFrLamRange', max(lamfr) - min(lamfr))
lamft = [famres[p][1]['R@1'][0] for p in ['lam0', 'lam0.25', 'galatea', 'lam0.75', 'salad'] if famres.get(p) and famres[p][1]]
if len(lamft) == 5:
    M('famFtLamRange', max(lamft) - min(lamft))

# ------------------------------------------------------------------ why fusion works: rescued / broken queries
def rescue_break(vf, tf, methods=('fgw', 'mnn_ransac', 'lg')):
    sv = np.load(vf, allow_pickle=True).item(); st = np.load(tf, allow_pickle=True).item()
    g = recall_after(st, 'test', None)['hits1']
    out = {}
    for m in methods:
        if m not in next(iter(st.values())):
            continue
        b, k = tune(sv, 'val', m, BET, KAP)
        for mode in ('local', 'fused'):
            h = recall_after(st, 'test', m, mode, b, k)['hits1']
            out[(m, mode)] = ((~g & h).mean() * 100, (g & ~h).mean() * 100)
    return out, st, g
rb = {}
for tag, vf in [('Fr', f'{R}/rerank/galatea_s0_val_mnn-uot-fgw-lg_a0.3_c0.25.npy'),
                ('Ft', f'{RG}/rerank/ft1e-4_galatea_s0_val_mnn-uot-fgw-lg_a0.3_c0.25.npy')]:
    tf = vf.replace('_val_', '_test_')
    if os.path.exists(vf) and os.path.exists(tf):
        out, st, g = rescue_break(vf, tf)
        rb[tag] = out
        for (m, mode), (r_, b_) in out.items():
            nm = 'rb' + tag + {'fgw': 'Fgw', 'mnn_ransac': 'MnnR', 'lg': 'Lg'}[m] + ('Loc' if mode == 'local' else 'Fus')
            M(nm + 'Res', r_); M(nm + 'Brk', b_)
        if tag == 'Fr':   # global margin of the local-only overrides (FGW)
            from analyze_rerank import gt
            G = gt('test'); mb, mr = [], []
            for name, S in st.items():
                pos = G[name][0]; cand = S['cand']; gs = S['gsim']; ls = S['fgw'].astype(float); nq = len(cand)
                loc = np.argmax(ls + 1e-3 * gs, 1); ar = np.arange(nq)
                gok = pos[ar, cand[:, 0]]; lok = pos[ar, cand[ar, loc]]; mg = gs[:, 0] - gs[ar, loc]
                mb += list(mg[gok & ~lok]); mr += list(mg[~gok & lok])
            M('marginBroken', np.median(mb), '{:.3f}'); M('marginRescued', np.median(mr), '{:.3f}')
if rb:
    L = ['\\begin{table}[t]', '\\centering',
         '\\caption{Why fusion works: share of test queries (\\%) whose top-1 result re-ranking \\emph{rescues} '
         '(global wrong $\\to$ correct) or \\emph{breaks} (global correct $\\to$ wrong), for local-only ranking '
         'and for the fused score, on Galatea seed~0.}', '\\label{tab:rescue}', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}llcccc@{}}', '\\toprule', '& & \\multicolumn{2}{c}{Local only} & \\multicolumn{2}{c}{Fused} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}', 'Stage 1 & Scorer & resc. & broken & resc. & broken \\\\', '\\midrule']
    for tag, lab in [('Fr', 'frozen'), ('Ft', 'fine-tuned')]:
        if tag not in rb:
            continue
        for i, (m, ml) in enumerate([('mnn_ransac', 'MNN+MAGSAC++'), ('lg', 'LightGlue'), ('fgw', 'partial FGW')]):
            lo_ = rb[tag][(m, 'local')]; fu_ = rb[tag][(m, 'fused')]
            L.append(f'{lab if i == 0 else ""} & {ml} & {lo_[0]:.1f} & {lo_[1]:.1f} & {fu_[0]:.1f} & {fu_[1]:.1f} \\\\')
        if tag == 'Fr':
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    open(f'{P}/tab_rescue.tex', 'w').write('\n'.join(L))

# ------------------------------------------------------------------ fusion on fine-tuned NetVLAD, 3 seeds
for m, nm in [('fgw', 'nvFgw'), ('mnn_ransac', 'nvMnnR')]:
    fu, gl, hs, r5 = [], [], [], []
    for s in [0, 1, 2]:
        fv = f'{RG}/rerank/ft1e-4_netvlad_s{s}_val_mnn-fgw_a0.3_c0.25.npy'; ft_ = fv.replace('_val_', '_test_')
        if not (os.path.exists(fv) and os.path.exists(ft_)):
            continue
        sv = np.load(fv, allow_pickle=True).item(); st = np.load(ft_, allow_pickle=True).item()
        b, k = tune(sv, 'val', m, BET, KAP)
        rf = recall_after(st, 'test', m, 'fused', b, k)
        fu.append(rf['R@1']); r5.append(rf['R@5']); gl.append(recall_after(st, 'test', None)['R@1'])
        hs.append(np.mean([rf[f'R@1_{c}'] for c in HARD]))
    if len(fu) == 3:
        g = np.array(fu) - np.array(gl)
        M('ms' + nm + 'Rone', np.mean(fu)); M('ms' + nm + 'Ronesd', np.std(fu)); M('ms' + nm + 'Rfive', np.mean(r5))
        M('ms' + nm + 'Gain', g.mean()); M('ms' + nm + 'Gainsd', g.std()); M('ms' + nm + 'Hard', np.mean(hs))
open(f'{P}/numbers.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(macros.items())) + '\n')
print('framework macros written')
