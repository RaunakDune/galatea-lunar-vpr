"""Write the real-imagery table and macros for the paper from results/apollo/*.json.

Outputs paper/tab_real.tex (main paper), paper/tab_real_ci.tex (supplement, with intervals) and
paper/numbers_apollo.tex.

usage: python src/apollo/make_apollo_tables.py
"""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
R = os.path.join(ROOT, 'results', 'apollo')
P = os.path.join(ROOT, 'paper')
pano = json.load(open(f'{R}/apollo_panos_center_summary.json'))['subsets']
full = json.load(open(f'{R}/apollo_panos_full_summary.json'))['subsets']
r2r = json.load(open(f'{R}/r2r_a15_summary.json'))

ROWS = [('Earth-trained, zero-shot', None, None),
        ('DINOv2-SALAD', 'salad_zs', 'salad_zs'),
        ('Lunar-trained head, frozen backbone', None, None),
        ('GeM', 'gem', 'gem'), ('NetVLAD', 'netvlad', 'netvlad'), (r'OT, $\lambda=0$', 'lam0', 'lam0'),
        (r'OT, $\lambda=1$ (SALAD)', 'salad', 'salad'), (r'OT, $\lambda\approx0.5$', 'galatea', 'galatea'),
        ('Lunar-trained, backbone fine-tuned', None, None),
        ('NetVLAD', 'rft_netvlad', 'rft_netvlad'), (r'OT, $\lambda\approx0.5$', 'rft_galatea', 'rft_galatea')]


def f1(x):
    return f'{x:.1f}'


def ci(v):
    return f'{v[0]:.1f} {{\\scriptsize[{v[1]:.1f}, {v[2]:.1f}]}}'


SHORT = {'lam0', 'salad'}  # left out of the main-paper table (page limit); kept in the supplement


def table(with_ci):
    m = pano['main']
    L = [r'\begin{table}[t]', r'\centering', r'\setlength{\tabcolsep}{4pt}' if with_ci else r'\setlength{\tabcolsep}{5pt}',
         r'\caption{Station-level Recall@1 (\%) on real Apollo photographs: against other pans (random '
         + f1(m['chance_R1']) + r'\%) and against a rendered Apollo~15 map (random ' + f1(r2r['chance_R1']) + r'\%). '
         r'Mean over three seeds' + (r'; 95\% intervals resample query pans.' if with_ci else
                                    r'; more heads and intervals in the supplement.') + '}',
         r'\label{tab:real_ci}' if with_ci else r'\label{tab:real}',
         r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
         r' & Panoramas & \multicolumn{2}{c}{Rendered map (A15)} \\', r'\cmidrule(lr){3-4}',
         r'Model & (A15--17) & DESPINA & shaded \\', r'\midrule']
    for name, kp, kr in ROWS:
        if kp is None:
            L.append(r'\multicolumn{4}{@{}l}{\emph{' + name + r'}} \\')
            continue
        if not with_ci and kp in SHORT:
            continue
        a = m['families'][kp]['R1']
        d = r2r['families'][kr]['despina']['R1']
        s = r2r['families'][kr]['shaded']['R1']
        if with_ci:
            L.append(f'{name} & {ci(a)} & {ci(d)} & {ci(s)} \\\\')
        else:
            L.append(f'{name} & {f1(a[0])} & {f1(d[0])} & {f1(s[0])} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L) + '\n'


open(f'{P}/tab_real.tex', 'w').write(table(False))
open(f'{P}/tab_real_ci.tex', 'w').write(table(True))

mac = {}
m, mn, al = pano['main'], pano['main_noLM'], pano['all']
mac['reNq'] = m['n_queries']; mac['reNpans'] = m['n_query_pans']; mac['reNst'] = m['n_query_stations']
mac['reChance'] = f1(m['chance_R1'])
mac['reNoLMNq'] = mn['n_queries']; mac['reNoLMChance'] = f1(mn['chance_R1'])
for key, fam in [('Zs', 'salad_zs'), ('Gem', 'gem'), ('Nv', 'netvlad'), ('Gal', 'galatea'), ('Sal', 'salad'),
                 ('LamZero', 'lam0'), ('FtNv', 'rft_netvlad'), ('FtGal', 'rft_galatea'), ('SalFix', 'salad_fixedillum')]:
    e = m['families'][fam]
    mac[f're{key}'] = f1(e['R1'][0]); mac[f're{key}Lo'] = f1(e['R1'][1]); mac[f're{key}Hi'] = f1(e['R1'][2])
    d = e['dR1_vs_ref']
    mac[f're{key}VsZsMean'] = f'{d[0]:+.1f}'; mac[f're{key}VsZsLo'] = f'{d[1]:+.1f}'; mac[f're{key}VsZsHi'] = f'{d[2]:+.1f}'
    mac[f're{key}Full'] = f1(full['main']['families'][fam]['R1'][0])
    mac[f're{key}NoLM'] = f1(mn['families'][fam]['R1'][0])
mac['rrNq'] = r2r['n_queries']; mac['rrNpans'] = r2r['n_query_pans']; mac['rrNst'] = r2r['n_stations']
mac['rrNviews'] = r2r['n_views']; mac['rrChance'] = f1(r2r['chance_R1'])
for key, fam in [('Zs', 'salad_zs'), ('Nv', 'netvlad'), ('Gal', 'galatea'), ('FtNv', 'rft_netvlad'), ('FtGal', 'rft_galatea')]:
    e = r2r['families'][fam]
    for db, tag in [('despina', 'D'), ('shaded', 'S')]:
        v = e[db]['R1']
        mac[f'rr{key}{tag}'] = f1(v[0]); mac[f'rr{key}{tag}Lo'] = f1(v[1]); mac[f'rr{key}{tag}Hi'] = f1(v[2])
    if 'despina_minus_shaded_R1' in e:
        v = e['despina_minus_shaded_R1']
        mac[f'rr{key}DmS'] = f'{v[0]:+.1f}'; mac[f'rr{key}DmSLo'] = f'{v[1]:+.1f}'; mac[f'rr{key}DmSHi'] = f'{v[2]:+.1f}'
open(f'{P}/numbers_apollo.tex', 'w').write(
    '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in sorted(mac.items())) + '\n')
print(f'{len(mac)} macros; tables written')
