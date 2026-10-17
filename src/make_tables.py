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


def load_run(name, split='test'):
    f = f'{R}/runs/{name}/results.json'
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return j['results'].get(split), j['info']


def agg(prefix, seeds=(0, 1, 2), split='test'):
    rs = [load_run(f'{prefix}_s{s}', split) for s in seeds]
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
for pre, lab in [('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('salad', 'SALAD ($\\lambda{=}1$)'),
                 ('galatea', 'Galatea, learned $\\lambda$')]:
    a = agg(pre)
    if a:
        heads[pre] = a
        rows.append(('tr', lab, a['info']['desc_dim'], a, a['hits']))

best = {k: max(r[3][k][0] for r in rows) for k in ['R@1', 'R@5', 'R@10'] + [f'R@1_{c}' for c in CONDS]}
L = ['\\begin{table*}[t]', '\\centering',
     '\\caption{Recall (\\%) on the LunarSim-VPR test set (3 unseen scenes, 3150 queries). Lunar-trained heads share the '
     'frozen DINOv2 backbone and training protocol; mean$\\pm$std over 3 seeds. Per-condition columns give R@1 '
     '(conditions in \\cref{tab:conds}).}', '\\label{tab:main}', '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
     '\\begin{tabular}{@{}llr ccc ccccccc@{}}', '\\toprule',
     '& Method & Dim. & R@1 & R@5 & R@10 & ' + ' & '.join(SH) + ' \\\\', '\\midrule']
for i, (kind, lab, dim, d, _) in enumerate(rows):
    if i == 3:
        L.append('\\midrule')
    grp = '\\multirow{3}{*}{\\rotatebox{90}{\\scriptsize zero-shot}}' if i == 0 else (
        '\\multirow{4}{*}{\\rotatebox{90}{\\scriptsize lunar-trained}}' if i == 3 else '')
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
for label, prefix in [('no dust-bin', 'galatea_nodust'), ('$L=1$ Sinkhorn iteration', 'galatea_L1'),
                      ('$L=10$ iterations', 'galatea_L10'), ('$m=16$ clusters', 'galatea_m16'),
                      ('$m=32$ clusters', 'galatea_m32'),
                      ('single-illumination training', 'galatea_fixedillum')]:
    a = agg(prefix, (0,))
    if a:
        abl2.append((label, a))
ref0 = agg('galatea', (0,))
L = ['\\begin{table}[t]', '\\centering',
     '\\caption{Ablations of the stage-1 aggregator (test R@1/R@5, \\%; hard = mean R@1 over C2, C5, C6). '
     'Top: assignment relaxation $\\lambda$ (mean$\\pm$std over the seeds available; single-seed rows marked $^\\dagger$). '
     'Bottom: Galatea variants vs.\\ the seed-0 reference.}', '\\label{tab:ablation}', '\\footnotesize',
     '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lccc@{}}', '\\toprule', 'Variant & R@1 & R@5 & hard \\\\', '\\midrule']
def hard(a):
    return np.mean([a[f'R@1_{c}'][0] for c in ['C2_el6', 'C5_az+180', 'C6_az+180_el8']])
for label, a in abl:
    dag = '$^\\dagger$' if a['n'] == 1 else ''
    lam = f" ($\\bar\\lambda{{=}}{np.mean(a['lams']):.2f}$)" if 'learned' in label else ''
    L.append(f'{label}{lam}{dag} & {cell(*a["R@1"])} & {cell(*a["R@5"])} & {hard(a):.1f} \\\\')
if ref0 and abl2:
    L.append('\\midrule')
    L.append(f'Galatea (seed 0, reference) & {ref0["R@1"][0]:.1f} & {ref0["R@5"][0]:.1f} & {hard(ref0):.1f} \\\\')
    for label, a in abl2:
        L.append(f'\\quad {label} & {a["R@1"][0]:.1f} & {a["R@5"][0]:.1f} & {hard(a):.1f} \\\\')
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
for pre, lab in [('netvlad', 'NetVLAD (lunar)'), ('salad', 'SALAD (lunar)'), ('galatea', 'Galatea')]:
    a = agg(pre)
    if a:
        series[lab] = [a[f'R@1_{c}'][0] for c in CONDS]
if os.path.exists(TESTF):
    series['Galatea + FGW fusion'] = [rr['fgw'][1][f'R@1_{c}'] for c in CONDS]
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
