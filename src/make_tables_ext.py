"""Tables and macros for the second simulator (LunarLoc traverses), metric pose after retrieval and boulder
segments as local evidence.

Reads results/lunarloc/lunarloc_v{1,2}_summary.json, results/pose_summary.json and
results/apollo/apollo_rerank_summary.json. Writes paper/numbers_ext.tex and paper/tab_ext_supp.tex.
usage: python src/make_tables_ext.py
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R, P = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'paper')
v1 = json.load(open(f'{R}/lunarloc/lunarloc_v1_summary.json'))['kinds']
v2 = json.load(open(f'{R}/lunarloc/lunarloc_v2_summary.json'))['kinds']
po = json.load(open(f'{R}/pose_summary.json'))
apo = json.load(open(f'{R}/apollo/apollo_rerank_summary.json'))['families']
f1 = lambda x: f'{x:.1f}'
mac = {}
FAMS = [('salad_zs', 'DINOv2-SALAD (zero-shot)'), ('gem', 'GeM'), ('netvlad', 'NetVLAD'), ('salad', r'OT, $\lambda=1$'),
        ('galatea', r'OT, $\lambda\approx0.5$'), ('rft_netvlad', 'NetVLAD, fine-tuned'), ('rft_galatea', 'OT, fine-tuned')]
FR = [f for f, _ in FAMS]

# ---------------------------------------------------------------- LunarLoc v2
S, L, F = v2['straight_pose'], v2['lighting_pose'], v2['flipped_overlap']
for tag, k in (('St', S), ('Li', L), ('Fl', F)):
    mac[f'll{tag}Nq'] = k['n_valid']; mac[f'll{tag}Chance'] = f1(k['chance_R1'])
    fam = k['families']
    mac[f'll{tag}Zs'] = f1(fam['salad_zs']['R1'][0]); mac[f'll{tag}Gal'] = f1(fam['galatea']['R1'][0])
    mac[f'll{tag}FtGal'] = f1(fam['rft_galatea']['R1'][0]); mac[f'll{tag}FtNv'] = f1(fam['rft_netvlad']['R1'][0])
    mac[f'll{tag}Nv'] = f1(fam['netvlad']['R1'][0]); mac[f'll{tag}Gem'] = f1(fam['gem']['R1'][0])
    for s, nm in (('mnn_ransac', 'Mnn'), ('fgw', 'Fgw'), ('clipper16', 'Clip')):
        g = [fam[f]['fused'][s]['d'][0] for f in FR]
        mac[f'll{tag}{nm}GainLo'], mac[f'll{tag}{nm}GainHi'] = f1(min(g)), f1(max(g))
mac['llFlNqAll'] = F['n_queries']
mac['llLiGalVsZs'] = f1(L['families']['galatea']['R1'][0] - L['families']['salad_zs']['R1'][0])
mac['llLiBestClip'] = f1(max(L['families'][f]['fused']['clipper16']['R1'][0] for f in FR))

# ---------------------------------------------------------------- LunarLoc v1 and Apollo: boulders
B1 = v1['repeat_pose']
mac['llRepNq'] = B1['n_valid']
bg = [B1['families'][f]['fused']['boulder_lopo']['d'][0] for f in FR]
kg = [B1['families'][f]['fused']['mnn_ransac']['d'][0] for f in FR]
mac['llRepBoulderLo'], mac['llRepBoulderHi'] = f1(min(bg)), f1(max(bg))
mac['llRepMnnLo'], mac['llRepMnnHi'] = f1(min(kg)), f1(max(kg))
ab = [apo[f]['scorers']['boulder16']['d_fused_lomo'][0] for f in FR]
mac['apBoulderLo'], mac['apBoulderHi'] = f1(min(ab)), f1(max(ab))

# ---------------------------------------------------------------- pose
for m, nm in (('mnn', 'Mnn'), ('lg', 'Lg')):
    t, u = po['matchers'][m]['top1'], po['matchers'][m]['true']
    mac[f'po{nm}Succ'] = f1(t['success']); mac[f'po{nm}SuccCzero'] = f1(t['success_C0'])
    mac[f'po{nm}SuccHard'] = f1(t['success_hard']); mac[f'po{nm}SuccRet'] = f1(t['success_given_retrieved'])
    mac[f'po{nm}TrueSucc'] = f1(u['success']); mac[f'po{nm}Fail'] = f1(t['pnp_fail'])
    mac[f'po{nm}MedT'] = f'{t["median_t_success"]:.2f}'; mac[f'po{nm}MedR'] = f'{t["median_r_success"]:.2f}'
mac['poRet'] = f1(po['retrieval_R1'])
open(f'{P}/numbers_ext.tex', 'w').write('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(mac.items())) + '\n')

# ---------------------------------------------------------------- supplement tables
T = [r'\begin{table*}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
     r'\caption{Second simulator: R@1 (\%) on the LunarLoc traverses~\cite{thomas2026lunarloc} (release v2.0.0), '
     r'before and after fusion of the top~10 with each local scorer ($(\beta,\kappa)$ and CLIPPER tolerance from the LunarSim '
     r'validation scene). Straight line and lighting change: within 1\,m and $30^\circ$ (random ' + f1(S['chance_R1']) + ' and '
     + f1(L['chance_R1']) + r'\%). Flipped start: no query has a positive under that definition, so a database frame counts when '
     r'the ground both cameras see overlaps by at least $30\%$ (random ' + f1(F['chance_R1']) + r'\%). '
     r'Mean over seeds; 95\% intervals in the result files.}',
     r'\label{tab:lunarloc}', r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
     r' & \multicolumn{2}{c}{Straight line (' + str(S['n_valid']) + r')} & \multicolumn{4}{c}{Lighting change (' + str(L['n_valid'])
     + r')} & \multicolumn{2}{c}{Flipped start (' + str(F['n_valid']) + r')} \\',
     r'\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-9}',
     r'Model & global & +MNN+M. & global & +MNN+M. & +FGW & +CLIPPER & global & +CLIPPER \\', r'\midrule']
for f, lab in FAMS:
    s, l, fl = S['families'][f], L['families'][f], F['families'][f]
    T.append(f'{lab} & {s["R1"][0]:.1f} & {s["fused"]["mnn_ransac"]["R1"][0]:.1f} & {l["R1"][0]:.1f} & '
             f'{l["fused"]["mnn_ransac"]["R1"][0]:.1f} & {l["fused"]["fgw"]["R1"][0]:.1f} & {l["fused"]["clipper16"]["R1"][0]:.1f} & '
             f'{fl["R1"][0]:.1f} & {fl["fused"]["clipper16"]["R1"][0]:.1f} \\\\')
T += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
T += [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
      r'\caption{Boulders as local evidence: R@1 (\%) before and after fusion. LunarLoc v1 repeat traverses ('
      + str(B1['n_valid']) + r' queries): LunarLoc\textquotesingle s own 3-D boulder detections matched with CLIPPER, tolerance '
      r'and $(\beta,\kappa)$ tuned leave one pair out; keypoints (MNN+MAGSAC++) for comparison. Apollo panoramas: FastSAM '
      r'boulder segments, CLIPPER on image-plane centroids, $(\beta,\kappa)$ tuned leave one mission out.}',
      r'\label{tab:boulders}', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
      r' & \multicolumn{3}{c}{LunarLoc v1} & \multicolumn{2}{c}{Apollo} \\', r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
      r'Model & global & +MNN+M. & +boulders & global & +boulders \\', r'\midrule']
for f, lab in FAMS:
    b = B1['families'][f]; a = apo[f]
    T.append(f'{lab} & {b["R1"][0]:.1f} & {b["fused"]["mnn_ransac"]["R1"][0]:.1f} & {b["fused"]["boulder_lopo"]["R1"][0]:.1f} & '
             f'{a["global_R1"][0]:.1f} & {a["scorers"]["boulder16"]["fused_lomo_R1"][0]:.1f} \\\\')
T += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
T += [r'\begin{table}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
      r'\caption{Metric pose after retrieval on the LunarSim test scenes (' + str(po['n']) + r' queries): PnP with RANSAC on '
      r'matches to the database image\textquotesingle s exact 3-D points. Success: within ' + f'{po["thresholds"][0]:g}' + r'\,m and '
      + f'{po["thresholds"][1]:g}' + r'$^\circ$ (\%). Top-1: image ranked first after MNN+MAGSAC++ fusion (R@1 '
      + f1(po['retrieval_R1']) + r'\%); true place: the database image of the query\textquotesingle s own place. '
      r'Median errors are over successful queries.}',
      r'\label{tab:pose}', r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule',
      r'Matcher & Image & all & $C_0$ & hard & PnP fails & $t$ (m) & $R$ ($^\circ$) \\', r'\midrule']
for m, lab in (('mnn', 'MNN'), ('lg', 'LightGlue')):
    for w, wl in (('top1', 'top-1'), ('true', 'true place')):
        x = po['matchers'][m][w]
        T.append(f'{lab if w == "top1" else ""} & {wl} & {x["success"]:.1f} & {x["success_C0"]:.1f} & {x["success_hard"]:.1f} & '
                 f'{x["pnp_fail"]:.1f} & {x["median_t_success"]:.2f} & {x["median_r_success"]:.2f} \\\\')
T += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
open(f'{P}/tab_ext_supp.tex', 'w').write('\n'.join(T))
print(len(mac), 'macros; tables written')
