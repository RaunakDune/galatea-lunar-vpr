"""Tables and macros for the stage-2 extensions: CLIPPER, 3-D structural terms (oracle depth) and
re-ranking on the Apollo panoramas.

Reads results/{rerank_summary,clipper_summary,fgw3d_summary,fgw3d_diag}.json and
results/apollo/apollo_rerank_summary.json. Writes
  paper/numbers_stage2.tex   macros
  paper/tab_rerank.tex       adds the CLIPPER and 3-D rows to the table written by make_tables.py
                             (between '% stage2' markers, so it can be re-run) and re-bolds the best
                             fused R@1
  paper/tab_stage2_supp.tex  supplement: 3-D variants, robustness sets, Apollo re-ranking per head
usage: python src/make_tables_stage2.py   (after make_tables.py)
"""
import json, os, re
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R, P = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'paper')
cl = json.load(open(f'{R}/clipper_summary.json'))
f3 = json.load(open(f'{R}/fgw3d_summary.json'))
dg = json.load(open(f'{R}/fgw3d_diag.json'))
apo = json.load(open(f'{R}/apollo/apollo_rerank_summary.json'))
mac = {}
f1 = lambda x: f'{x:.1f}'
sg = lambda x: f'{x:+.1f}'.replace('-0.0', '0.0').replace('+0.0', '0.0')


def ci_mac(name, v):
    mac[name + 'Mean'], mac[name + 'Lo'], mac[name + 'Hi'] = sg(v[0]), sg(v[1]), sg(v[2])


# ---------------------------------------------------------------- CLIPPER (LunarSim test)
CL = {'clipper2d': 'clTwo', 'clipper2d_ransac': 'clTwoR', 'clipper3d': 'clThree', 'clipper3d_ransac': 'clThreeR'}
for k, nm in CL.items():
    v = cl['rows'][k]
    mac[nm + 'Rone'] = f1(v['fused']['R@1']); mac[nm + 'LocRone'] = f1(v['local']['R@1'])
    mac[nm + 'Hard'] = f1(v['fused']['hard'])
    mac[nm + 'Eps'] = re.search(r'_e([\d.]+)', v['selected']).group(1)
    ci_mac(nm + 'VsFgw', v['d_vs_fgw']); ci_mac(nm + 'VsMnnR', v['d_vs_mnn_ransac']); ci_mac(nm + 'VsGlob', v['d_vs_global'])

# ---------------------------------------------------------------- FGW with 3-D structure
sel = f3['selected']['']
mode, alpha = re.match(r'fgw(\w+?)_a([\d.]+)$', sel).groups()
t3 = f3['test'][sel]
mac['fgwThreeRone'] = f1(t3['fused']['R@1']); mac['fgwThreeLocRone'] = f1(t3['local']['R@1'])
mac['fgwThreeHard'] = f1(t3['fused']['hard']); mac['fgwThreeAlpha'] = alpha
ci_mac('fgwThreeVsTwo', t3['dR1_fused_vs_2d']); ci_mac('fgwThreeRVsTwoR', f3['test'][sel + '_ransac']['dR1_fused_vs_2d'])
plain = [k for k in f3['test'] if k.startswith('fgw3d') and not k.endswith('_ransac')]
bt = max(plain, key=lambda k: f3['test'][k]['fused']['R@1'])
mac['fgwThreeBestTestRone'] = f1(f3['test'][bt]['fused']['R@1']); ci_mac('fgwThreeBestTestVsTwo', f3['test'][bt]['dR1_fused_vs_2d'])
mac['fgwThreeNvar'] = len(plain)
rb = f3['robust']
off = [rb[f'offset_{d}']['dR1_3d_vs_2d'] for d in (0, 2, 4, 6, 8)]
mac['rbThreeOffLo'], mac['rbThreeOffHi'] = f1(-max(x[0] for x in off)), f1(-min(x[0] for x in off))
mac['rbThreeOffFourTwo'], mac['rbThreeOffFourThree'] = f1(rb['offset_4']['fgw']['R@1']), f1(rb['offset_4'][sel]['R@1'])
hp = [rb[v]['dR1_3d_vs_2d'][0] for v in rb if v.startswith(('heading', 'pitch'))]
mac['rbThreeHPMax'] = f1(max(abs(x) for x in hp))
mac['diagCorrFrac'] = f'{100 * dg["median_corr_fraction"]:.0f}'
mac['diagCorr'] = f'{dg["median_true_corr"]:.0f}'; mac['diagKp'] = f'{dg["median_keypoints"]:.0f}'
mac['diagDistTwo'] = f'{100 * dg["distortion_median"]["2d"]:.0f}'
mac['diagDistThree'] = f'{100 * dg["distortion_median"]["3d"]:.1f}'

# ---------------------------------------------------------------- Apollo panoramas
FAM = apo['families']
SC = ['mnn_ransac', 'lg_ransac', 'fgw', 'clipper16']
gains = [FAM[f]['scorers'][s]['d_fused_sim'][0] for f in FAM for s in SC]
mac['apGainLo'], mac['apGainHi'] = f1(min(gains)), f1(max(gains))
lo_all = [FAM[f]['scorers'][s]['d_fused_sim'][1] for f in FAM for s in SC]
mac['apGainCiLoMin'] = f1(min(lo_all))
lomo = [FAM[f]['scorers'][s]['d_fused_lomo'][0] for f in FAM for s in SC]
mac['apLomoLo'], mac['apLomoHi'] = f1(min(lomo)), f1(max(lomo))
z = FAM['salad_zs']
mac['apZsGlob'] = f1(z['global_R1'][0]); mac['apZsFgw'] = f1(z['scorers']['fgw']['fused_sim_R1'][0])
ci_mac('apZsFgwGain', z['scorers']['fgw']['d_fused_sim'])
mac['apZsTopTen'] = f1(z['top10_contains_positive'][0])
loc_gain = [FAM[f]['scorers'][s]['local_R1'][0] - FAM[f]['global_R1'][0] for f in FAM for s in SC]
mac['apLocGainLo'], mac['apLocGainHi'] = f1(min(loc_gain)), f1(max(loc_gain))
best = max(((f, s) for f in FAM for s in SC), key=lambda fs: FAM[fs[0]]['scorers'][fs[1]]['fused_sim_R1'][0])
mac['apBestRone'] = f1(FAM[best[0]]['scorers'][best[1]]['fused_sim_R1'][0])
spread = [max(FAM[f]['scorers'][s]['fused_sim_R1'][0] for s in SC) - min(FAM[f]['scorers'][s]['fused_sim_R1'][0] for s in SC)
          for f in FAM]
mac['apScorerSpread'] = f1(max(spread))
open(f'{P}/numbers_stage2.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(mac.items())) + '\n')

# ---------------------------------------------------------------- rows of the main re-ranking table
T = open(f'{P}/tab_rerank.tex').read()
T = re.sub(r'% stage2-begin.*?% stage2-end\n', '', T, flags=re.S)
T = re.sub(r'\\textbf\{([\d.]+)\}', r'\1', T)
row = lambda lab, loc, fus: f'{lab} & {loc["R@1"]:.1f} & {loc["R@5"]:.1f} & {fus["R@1"]:.1f} & {fus["R@5"]:.1f} \\\\'
new = ['% stage2-begin', r'\midrule',
       row(r'CLIPPER~\cite{lusk2021clipper}', cl['rows']['clipper2d']['local'], cl['rows']['clipper2d']['fused']),
       row(r'CLIPPER~\cite{lusk2021clipper} + MAGSAC++', cl['rows']['clipper2d_ransac']['local'], cl['rows']['clipper2d_ransac']['fused']),
       row(r'Partial FGW, 3-D$^\dagger$', t3['local'], t3['fused']),
       row(r'CLIPPER, 3-D$^\dagger$ + MAGSAC++', cl['rows']['clipper3d_ransac']['local'], cl['rows']['clipper3d_ransac']['fused']),
       '% stage2-end']
T = T.replace('\\bottomrule', '\n'.join(new) + '\n\\bottomrule', 1)
lines = T.split('\n')
vals = [(i, float(l.split('&')[3])) for i, l in enumerate(lines) if l.count('&') == 4 and re.fullmatch(r'[\d.]+', l.split('&')[3].strip())]
best_v = max(v for _, v in vals)
for i, v in vals:
    if abs(v - best_v) < 0.05:
        p = lines[i].split('&'); p[3] = ' \\textbf{%.1f} ' % v; lines[i] = '&'.join(p)
T = '\n'.join(lines)
if r'$^\dagger$Distances' not in T:
    T = T.replace(r'chosen on the validation scene.}', r'chosen on the validation scene, as are the CLIPPER tolerances. '
                  r'$^\dagger$Distances between oracle 3-D keypoint positions from the renderer.}')
open(f'{P}/tab_rerank.tex', 'w').write(T)

# ---------------------------------------------------------------- supplement tables
L = [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
     r'\caption{Partial FGW with 3-D structure matrices (oracle depth): fused R@1 (\%) on validation and test, and '
     r'the test difference to image-plane FGW (95\% CI over places). \emph{3d}: Euclidean distance, normalised to mean~1; '
     r'\emph{3dlog}: $\log(1+d)$, mean~1; \emph{3dclip}: $\min(d,200\,\mathrm{m})$ and \emph{3dlogfix}: $\log(1+d)$, '
     r'each divided by one fixed scale. Selected on validation: ' + mode + r', $\alpha=' + alpha + r'$.}',
     r'\label{tab:fgw3d}', r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
     r'Structure & $\alpha$ & Val. & Test & $\Delta$ vs.\ image plane \\', r'\midrule',
     f'image plane & 0.3 & {f3["val"]["fgw"]["R@1"]:.1f} & {f3["test"]["fgw"]["fused"]["R@1"]:.1f} & -- \\\\']
for k in sorted(plain, key=lambda k: (re.match(r'fgw(\w+?)_a', k).group(1), k)):
    m_, a_ = re.match(r'fgw(\w+?)_a([\d.]+)$', k).groups()
    d = f3['test'][k]['dR1_fused_vs_2d']
    L.append(f'{m_} & {a_} & {f3["val"][k]["R@1"]:.1f} & {f3["test"][k]["fused"]["R@1"]:.1f} & '
             f'{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] \\\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
VAR = [('heading', (15, 30, 45), r'heading $\pm{}^\circ$'), ('offset', (0, 2, 4, 6, 8), r'offset (m)'),
       ('pitch', (-30, -15, -5, 5, 15, 30), r'pitch ($^\circ$)')]
L += [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
      r'\caption{Robustness variants: R@1 (\%) of the frozen reference head before and after fusion with image-plane '
      r'and 3-D FGW (' + mode + r', $\alpha=' + alpha + r'$), and the 3-D minus image-plane difference (95\% CI over places).}',
      r'\label{tab:fgw3drob}', r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
      r'Variant & Global & FGW & FGW 3-D & $\Delta$ \\', r'\midrule']
for kind, vals_, lab in VAR:
    L.append(r'\multicolumn{5}{@{}l}{\emph{' + lab + r'}} \\')
    for v in vals_:
        e = rb[f'{kind}_{v}']; d = e['dR1_3d_vs_2d']
        L.append(f'\\quad {v} & {e["global"]["R@1"]:.1f} & {e["fgw"]["R@1"]:.1f} & {e[sel]["R@1"]:.1f} & '
                 f'{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] \\\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
NAMES = [('salad_zs', 'DINOv2-SALAD (zero-shot)'), ('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('salad', r'OT, $\lambda=1$'),
         ('galatea', r'OT, $\lambda\approx0.5$'), ('rft_netvlad', 'NetVLAD, fine-tuned'), ('rft_galatea', r'OT, fine-tuned')]
L += [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{2.5pt}',
      r'\caption{Apollo panoramas (A15--17, ' + str(apo['n_queries']) + r' queries): station-level R@1 (\%) of the '
      r'global ranking and after fusion of the top~10 with each local scorer, with $(\beta,\kappa)$ transferred from '
      r'the LunarSim validation scene. Last column: FGW fusion with $(\beta,\kappa)$ tuned on the other two missions '
      r'(leave one mission out). Top~10: queries whose top~10 contain the right station. Mean over seeds.}',
      r'\label{tab:apollorr}', r'\begin{tabular}{@{}lccccccc@{}}', r'\toprule',
      r'Model & Global & Top~10 & MNN+M. & LG+M. & FGW & CLIPPER & FGW (LOMO) \\', r'\midrule']
for f, lab in NAMES:
    e = FAM[f]
    L.append(f'{lab} & {e["global_R1"][0]:.1f} & {e["top10_contains_positive"][0]:.1f} & ' +
             ' & '.join(f'{e["scorers"][s]["fused_sim_R1"][0]:.1f}' for s in SC) +
             f' & {e["scorers"]["fgw"]["fused_lomo_R1"][0]:.1f} \\\\')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
open(f'{P}/tab_stage2_supp.tex', 'w').write('\n'.join(L))
print(len(mac), 'macros; tables written')
