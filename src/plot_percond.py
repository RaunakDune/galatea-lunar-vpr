import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import paths
import os
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
S = json.load(open(os.path.join(paths.RESULTS, 'percond_series.json')))
labels = ['C0\nsame', 'C1\nel15', 'C2\nel6', 'C3\nel60', 'C4\naz+90', 'C5\naz+180', 'C6\naz+180\nel8']
style = {  # validated categorical palette (light): blue, orange, aqua, yellow, magenta, violet
    'CosPlace-R50 (zero-shot)': ('#eda100', '--', 'v'),
    'EigenPlaces-R50 (zero-shot)': ('#e87ba4', '--', '^'),
    'DINOv2-SALAD (zero-shot)': ('#eb6834', '--', 's'),
    'NetVLAD (lunar)': ('#1baf7a', '-', 'D'),
    'SALAD (lunar)': ('#6250d6', '-', 'P'),
    'Galatea': ('#2a78d6', '-', 'o'),
    'Galatea + FGW fusion': ('#0b0b0b', '-', '*'),
}
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=300)
x = range(len(labels))
for k, v in S.items():
    c, ls, mk = style.get(k, ('gray', '-', 'o'))
    ax.plot(x, v, ls, color=c, marker=mk, ms=4 if mk != '*' else 6, lw=1.4, label=k)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel('R@1 (%)'); ax.set_ylim(0, 100)
ax.grid(axis='y', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.legend(fontsize=5.2, ncol=2, frameon=False, loc='upper right', handlelength=2.2, columnspacing=0.8)
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(paths.PAPER, 'figures', 'percond.pdf')); plt.savefig(os.path.join(paths.RESULTS, 'percond.png'), dpi=200)
