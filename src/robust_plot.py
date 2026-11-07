"""Figure for the robustness tests: Recall@1 (all 7 Sun conditions) against roll, pitch, position offset,
heading change and camera height. Reads <RESULTS>/robust/summary.json (robust_report.py)."""
import os, sys, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

RB = os.path.join(paths.RESULTS, 'robust')
S = json.load(open(f'{RB}/summary.json'))
METRIC = sys.argv[1] if len(sys.argv) > 1 else 'R@1'
OUTF = sys.argv[2] if len(sys.argv) > 2 else f'{RB}/robustness.pdf'

# categorical slots 1-4 of the reference palette (validated; CVD pairs get markers + line styles as
# secondary encoding), zero-shot reference in neutral grey
SERIES = [  # key, label, colour, marker, linestyle
    ('zs', 'DINOv2-SALAD, zero-shot', '#8a8a85', 'x', '--'),
    ('fr_gal', 'Galatea, frozen', '#2a78d6', 'o', '-'),
    ('galatea_s0+fgw', 'Galatea, frozen + FGW fusion', '#e87ba4', 'D', '-'),
    ('ft_gal', 'Galatea, fine-tuned', '#1baf7a', 's', '-'),
    ('ft_nv', 'NetVLAD, fine-tuned', '#eb6834', '^', '-'),
]
PANELS = [  # title, x label, [(x, variant)]
    ('Roll', 'roll (deg)', [(0, 'roll_0'), (5, 'roll_5'), (10, 'roll_10'), (15, 'roll_15'), (30, 'roll_30'), (45, 'roll_45'), (90, 'roll_90'), (180, 'roll_180')]),
    ('Pitch', 'pitch (deg, + = up)', [(-30, 'pitch_-30'), (-15, 'pitch_-15'), (-5, 'pitch_-5'), (0, 'roll_0'), (5, 'pitch_5'), (15, 'pitch_15'), (30, 'pitch_30')]),
    ('Position offset', 'distance from database place (m)', [(0, 'offset_0'), (2, 'offset_2'), (4, 'offset_4'), (6, 'offset_6'), (8, 'offset_8')]),
    ('Heading change', 'added heading change (deg)', [(0, 'roll_0'), (15, 'heading_15'), (30, 'heading_30'), (45, 'heading_45')]),
    ('Camera height', 'camera height (m; database 1.5)', [(0.5, 'height_0.5'), (1.0, 'height_1.0'), (1.5, 'height_1.5'), (2.5, 'height_2.5'), (4.0, 'height_4.0')]),
]


def value(v, key):
    if v not in S:
        return np.nan, 0.0
    if key in S[v]['groups']:
        m, s = S[v]['groups'][key][METRIC]
        return m, s
    if key in S[v]['models']:
        return S[v]['models'][key][METRIC], 0.0
    return np.nan, 0.0


plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#9a9a96', 'axes.labelcolor': '#3a3a38', 'xtick.color': '#5f6066',
                     'ytick.color': '#5f6066', 'axes.titlesize': 9, 'axes.titleweight': 'bold'})
fig, axes = plt.subplots(1, 6, figsize=(13.5, 2.7), gridspec_kw={'width_ratios': [1.25, 1.25, 1, 1, 1, 1]})
# roll: as rendered, and after gravity (IMU) correction
for ai, (title, xl, pts) in enumerate([PANELS[0], ('Roll, gravity-corrected', 'roll (deg)',
                                       [(0, 'roll_0')] + [(r, f'derot_{r}') for r in (15, 30, 45, 90, 180)])] + PANELS[1:]):
    ax = axes[ai]
    for key, lab, col, mk, ls in SERIES:
        xs = [p[0] for p in pts]; ms = np.array([value(p[1], key) for p in pts])
        ax.errorbar(xs, ms[:, 0], yerr=ms[:, 1], color=col, marker=mk, ms=4, lw=1.6, ls=ls, capsize=1.5,
                    label=lab, markeredgecolor='white' if mk != 'x' else col, markeredgewidth=0.6)
    ax.set_title(title, loc='left'); ax.set_xlabel(xl)
    ax.set_ylim(0, 100); ax.grid(axis='y', color='#e3e2de', lw=0.6); ax.set_axisbelow(True)
    if pts[0][1].startswith(('roll', 'derot')) or ai == 0:
        ax.set_xticks([0, 15, 30, 45, 90, 180])
        ax.set_xscale('function', functions=(lambda x: np.sqrt(np.maximum(x, 0)), lambda x: x ** 2))
    if ai == 0:
        ax.set_ylabel(f'{METRIC} (%), all Sun conditions')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUTF, bbox_inches='tight')
fig.savefig(OUTF.rsplit('.', 1)[0] + '.png', dpi=200, bbox_inches='tight')
print('wrote', OUTF)
