"""[venv-xsage] figs/fig_worked_example_ml1m.{pdf,png}

One held-out case (Evening comedy / The Mask): situation space (left) +
rank ladders BACKBONE -> +X-SAGE (right) + a user-friendly EXPLANATION box
(why the recommendation changed: situation = contextual attributes, intent,
typical behaviour in that situation).

Data: D.space / D.reco.examples[0] embedded in demo_xsage_ml1m.html (no recompute).
Rank semantics: positions refer to the FIRST item of the watched category
(Cat-MRR semantics), not to the watched item itself.
"""
import re, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
HTML_PATH = CLEAN / "outputs_results/explain/demo_xsage_ml1m.html"
OUT_DIR = CLEAN / "paper/figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.edgecolor': '0.3', 'axes.linewidth': 0.8, 'pdf.fonttype': 42})

html = open(HTML_PATH, encoding='utf-8').read()
D = json.loads(re.search(r'const\s+D\s*=\s*(\{.*?\});', html, flags=re.S).group(1))
pts = D['space']['points']; c0 = D['space']['centroids'][0]
ex = D['reco']['examples'][0]
BLUE = '#2563eb'; RED = '#dc2626'; GREEN = '#15803d'

fig = plt.figure(figsize=(6.5, 4.75))
# 2 rows: top = space + ladders ; bottom = EXPLANATION box
gs = gridspec.GridSpec(2, 2, width_ratios=[0.28, 0.72], height_ratios=[0.64, 0.36],
                       wspace=0.10, hspace=0.06)

# ---- left: situation space, Evening-comedy region highlighted ----
axs = fig.add_subplot(gs[0, 0])
for x, y, s, b in pts:
    axs.scatter(x, y, s=6, color=(BLUE if s == 0 else '0.84'),
                alpha=(0.45 if s == 0 else 0.6), lw=0, zorder=(2 if s == 0 else 1))
axs.scatter(c0['x'], c0['y'], marker='*', s=220, color=BLUE,
            edgecolor='white', lw=1.0, zorder=4)
axs.annotate('Evening\ncomedy', xy=(c0['x'], c0['y']), xytext=(0.52, 0.86),
             fontsize=7.8, color=BLUE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))
axs.set_xticks([]); axs.set_yticks([])
axs.set_xlabel('situation space', fontsize=7.5, color='0.45', labelpad=3)
for sp in ['top', 'right']: axs.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: axs.spines[sp].set_color('0.7')

# ---- right: rank ladders ----
ax = fig.add_subplot(gs[0, 1])
ax.set_xlim(0, 1); ax.set_ylim(26, -6.5)     # inverted: rank 1 on top
ax.set_xticks([])
ax.set_yticks([1, 5, 10, 15, 20, 25]); ax.tick_params(labelsize=7.5, length=2, colors='0.4')
ax.set_ylabel('position in the recommended list', fontsize=8, color='0.35')
for sp in ['top', 'right', 'bottom']: ax.spines[sp].set_visible(False)
ax.spines['left'].set_color('0.7'); ax.spines['left'].set_bounds(1, 25)

xB, xX = 0.18, 0.60
ax.text(0.0, -5.6, 'request:  9 p.m. · weekend · after an Adventure movie',
        fontsize=7.8, color='0.35', va='center')
ax.text(0.0, -3.8, 'the user then watched:  Comedy — The Mask (1994)',
        fontsize=8.0, color='0.12', va='center', fontweight='bold')
ax.text(xB, -1.6, 'BACKBONE', fontsize=8, color='0.35', ha='center', fontweight='bold')
ax.text(xX, -1.6, '+X-SAGE', fontsize=8, color=BLUE, ha='center', fontweight='bold')

for xc in (xB, xX):
    ax.plot([xc, xc], [1, 25], color='0.85', lw=2.5, zorder=1, solid_capstyle='round')
ax.axhline(20, color='0.6', lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(0.995, 19.2, 'top-20 cutoff', fontsize=7, color='0.45', ha='right')

ax.scatter([xB], [1], s=30, color='0.6', zorder=3)
ax.text(xB + 0.035, 0.6, 'Independence Day', fontsize=7.2, color='0.45', va='center')
ax.text(xB + 0.035, 2.5, '(Action)', fontsize=6.8, color='0.55', va='center')
ax.scatter([xB], [23], s=110, color=RED, edgecolor='white', lw=1.2, zorder=4)
ax.text(xB + 0.035, 22.6, 'first Comedy — #23', fontsize=7.6, color=RED,
        va='center', fontweight='bold')
ax.text(xB + 0.035, 24.6, 'outside the top-20', fontsize=7.0, color=RED, va='center')

ax.scatter([xX], [1], s=130, color=GREEN, edgecolor='white', lw=1.2, zorder=4)
ax.text(xX + 0.035, 1, 'first Comedy — #1', fontsize=7.6, color=GREEN,
        va='center', fontweight='bold')
ax.text(xX + 0.035, 3.0, 'Forrest Gump (Comedy)', fontsize=7.0, color='0.45', va='center')

arr = FancyArrowPatch((xB + 0.02, 22.4), (xX - 0.02, 2.0),
                      connectionstyle='arc3,rad=-0.22',
                      arrowstyle='-|>', mutation_scale=14, color=BLUE, lw=2.0, zorder=5)
ax.add_patch(arr)
ax.text(0.395, 13.0, 'situational nudge\non Comedy', fontsize=7.5, color=BLUE,
        ha='center', va='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85))

# ---- bottom: EXPLANATION box (user-friendly, natural language) ----
axb = fig.add_subplot(gs[1, :])
axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis('off')
box = FancyBboxPatch((0.006, 0.04), 0.988, 0.92, transform=axb.transAxes,
                     boxstyle='round,pad=0.010,rounding_size=0.03',
                     linewidth=1.0, edgecolor=BLUE, facecolor='#eff4ff', zorder=1)
axb.add_patch(box)
axb.text(0.028, 0.855, 'EXPLANATION', fontsize=8.2, color=BLUE, fontweight='bold',
         va='center', zorder=2)
# who-said-what, in the user's own words: situation (context) + intent + norm + effect
lines = [
    ("It's a weekend evening (9 p.m.) and you just watched an Adventure movie.", '0.18'),
    ("Right now you might be in the mood for something lighter — so here's a", '0.18'),
    ("comedy you may enjoy tonight: Forrest Gump.", '0.18'),
]
y0 = 0.60
for i, (txt, col) in enumerate(lines):
    axb.text(0.028, y0 - i * 0.185, txt, fontsize=8.0, color=col, va='center', zorder=2)

plt.subplots_adjust(left=0.045, right=0.985, top=0.965, bottom=0.035)
fig.savefig(OUT_DIR / 'fig_worked_example_ml1m.pdf', bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_worked_example_ml1m.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('saved -> paper/figs/fig_worked_example_ml1m.{pdf,png}')
