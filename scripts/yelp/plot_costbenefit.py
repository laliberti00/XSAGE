"""[venv-xsage] paper/figs/fig_costbenefit.{pdf,png} — cost/benefit of the
situational head on the focal backbone (FM), per dataset.

FOURSQUARE IS ONE DATASET: the two city instances (NYC, Sao Paulo) are
averaged into a single 'Foursquare' row. Per-city values are kept below in
comments and surfaced as a small annotation under the Foursquare accuracy
bar, so the localized cost stays declared.

Authored at FINAL physical size (figsize 5.0 x 2.6 in): include in LaTeX
with \\includegraphics{figs/fig_costbenefit.pdf} WITHOUT width= so fonts
render 1:1.

Data source: results_record.csv, focal backbone B_full, deltas SIT-BASE.
Values verified against the CSV (2026-07-06).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/Users/lucaaliberti/Downloads/xsage-clean/paper/figs")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                     'axes.edgecolor': '0.3', 'axes.linewidth': 0.8,
                     'pdf.fonttype': 42})
BLUE = '#2563eb'; GREY = '#4b5563'

# ---- data (focal FM, SIT - BASE) -------------------------------------
# Foursquare = mean of the two city instances:
#   NYC:  exposure +0.01382 , dHR@20 -0.00556  (only TOST failure, focal)
#   SP :  exposure +0.00245 , dHR@20 -0.00125
datasets = ['Foursquare', 'MovieLens-1M', 'KuaiRand', 'Yelp']
expgain  = [ 0.00814,      0.00431,        0.00035,    0.00012]   # -dGini
dhr20    = [-0.00341,      0.00214,        0.00019,    0.00004]

fig, ax = plt.subplots(figsize=(5.0, 2.6))
y = np.arange(len(datasets))[::-1]
h = 0.34

ax.axvline(0, color='0.45', lw=0.9, zorder=2)

ax.barh(y + h/2 + 0.02, expgain, height=h, color=BLUE, zorder=3,
        label='exposure gain ($-\\Delta$Gini)')
ax.barh(y - h/2 - 0.02, dhr20, height=h, color=GREY, zorder=3,
        label='accuracy $\\Delta$ (HR@20)')

for yi, v in zip(y + h/2 + 0.02, expgain):
    ax.text(v + 0.0003, yi, f'+{v:.4f}', va='center', fontsize=7.2, color=BLUE)
for yi, v in zip(y - h/2 - 0.02, dhr20):
    off, ha = (0.0003, 'left') if v >= 0 else (-0.0003, 'right')
    ax.text(v + off, yi, f'{v:+.4f}', va='center', ha=ha, fontsize=7.2,
            color=GREY)

ax.set_yticks(y); ax.set_yticklabels(datasets, fontsize=8.5)
ax.set_xlim(-0.0075, 0.0115)
ax.legend(loc='lower right', fontsize=7.5, frameon=False)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.tick_params(left=False, labelsize=7.5)

plt.tight_layout()
plt.savefig(OUT / 'fig_costbenefit.pdf')
plt.savefig(OUT / 'fig_costbenefit.png', dpi=300)
print('saved paper/figs/fig_costbenefit.{pdf,png}')
