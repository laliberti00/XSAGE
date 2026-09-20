"""[venv-xsage] fig_costbenefit_grid.{pdf,png} — sintesi multi-backbone delle
Tabelle 1-2: per ogni backbone (subplot) e dataset, tre Δ (SIT-backbone):
M-CMRR (guadagno categoriale), HR@20 (accuracy), -Gini (esposizione).
Banda ±0.005 = margine di equivalenza pre-registrato. Foursquare = media NYC+SP.
Dati: results_record.csv (delta_l1). Nessun ricalcolo.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5,
                     'axes.edgecolor': '0.3', 'axes.linewidth': 0.7, 'pdf.fonttype': 42})

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
OUTS = [CLEAN / "paper/figs", Path("/Users/lucaaliberti/Desktop/MDPI_Article_Template/figs")]
for o in OUTS: o.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(CLEAN / "outputs_results/results_record.csv"); sit = d[d.method == "SIT"]
BK = [("B_blind", "BPR"), ("B_full", "FM"), ("EASE", "EASE"), ("DeepFM", "DeepFM"),
      ("AFM", "AFM"), ("FPMC", "FPMC"), ("SASRec", "SASRec")]
DS = [("ml1m", "ML-1M"), ("FSQ", "Foursquare"), ("yelp", "Yelp"), ("kuairand", "KuaiRand")]
BLUE = "#2563eb"; GREY = "#6b7280"; GREEN = "#15803d"

def dl(ds, bk, met):
    if ds == "FSQ": return np.mean([dl(x, bk, met) for x in ["nyc_tist", "saopaulo"]])
    r = sit[(sit.dataset == ds) & (sit.backbone == bk) & (sit.metric == met)]
    return float(r.delta_l1.iloc[0]) if len(r) else np.nan

fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.8), sharex=True); axes = axes.ravel()
y = np.arange(len(DS))[::-1]; h = 0.24
for ai, (bk, bl) in enumerate(BK):
    ax = axes[ai]
    cm = [dl(dd, bk, "macroCatMRR") for dd, _ in DS]
    gi = [-dl(dd, bk, "Gini") for dd, _ in DS]
    hr = [dl(dd, bk, "HR20") for dd, _ in DS]
    ax.barh(y + h + 0.02, cm, height=h, color=BLUE, zorder=3)
    ax.barh(y,           gi, height=h, color=GREEN, zorder=3)
    ax.barh(y - h - 0.02, hr, height=h, color=GREY, zorder=3)
    ax.axvspan(-0.005, 0.005, color='0.93', zorder=0)
    ax.axvline(0, color='0.4', lw=0.8, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in DS], fontsize=7.5)
    ax.set_title(bl, fontsize=9.5, fontweight='bold')
    ax.tick_params(labelsize=7)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axes[7].axis('off')
axes[7].legend(handles=[
    Patch(color=BLUE, label=r'categorical gain ($\Delta$M-CMRR)'),
    Patch(color=GREEN, label=r'exposure gain ($-\Delta$Gini)'),
    Patch(color=GREY, label=r'accuracy ($\Delta$HR@20)'),
    Patch(facecolor='0.93', label=r'equivalence ($\pm0.005$)')],
    loc='center', frameon=False, fontsize=8.5)
fig.supxlabel("change with respect to the backbone   (right = better)", fontsize=9)
plt.tight_layout()
for o in OUTS:
    fig.savefig(o / "fig_costbenefit_grid.pdf", bbox_inches='tight')
    fig.savefig(o / "fig_costbenefit_grid.png", dpi=300, bbox_inches='tight')
plt.close(fig)
print("saved fig_costbenefit_grid.{pdf,png} ->", [str(o) for o in OUTS])
