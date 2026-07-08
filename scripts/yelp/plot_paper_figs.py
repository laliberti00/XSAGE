"""[venv-xsage] Figure X-SAGE per pubblicazione Elsevier (cas-sc, single column).
Rigenera le 3 figure dai MEDESIMI JSON gia' prodotti (nessun dato ricalcolato):
  fig_situation_space_ml1m  — PCA scatter (core/boundary + centroidi)
  fig_inequity_lens_ml1m    — heatmap bias z-score per situazione (ordine lensKL desc)
  fig_transitions_ml1m      — matrice di transizione L3 (Markov)
Stile unificato: font sans 9pt, niente titoli interni (vanno in caption LaTeX),
inglese, pdf.fonttype=42. Output: paper/figs/{name}.{pdf,png} (PDF vettoriale + PNG 300dpi).
Uso:  python scripts/yelp/plot_paper_figs.py [city=ml1m]
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.edgecolor": "0.3", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
EXP = CLEAN / "outputs_results" / "explain"
FIGS = CLEAN / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# mappa nomi situazioni — identica in tutte le figure
NAMES = {0: "Evening comedy", 1: "Afternoon horror", 2: "Late-night action",
         3: "Afternoon fantasy", 4: "Weekend drama"}
# palette qualitativa 5-classi ancorata all'accent blue
PAL = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> paper/figs/{name}.pdf + .png")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    sp = json.load(open(EXP / f"situation_space_{city}.json"))
    pr = json.load(open(EXP / f"situation_profiles_{city}.json"))
    tr = json.load(open(EXP / f"situation_transitions_{city}.json"))
    K = pr["K"]; genres = pr["genres"]
    lensKL = {s["k"]: s["lens_KL"] for s in pr["situations"]}
    lab = lambda k: f"S{k} — {NAMES[k]}"   # 'S0 — Evening comedy'

    # ---- FIG 1: situation space (PCA scatter) ----
    pts = np.array(sp["points"]); var = sp["var"]
    x, y, sit, isb = pts[:, 0], pts[:, 1], pts[:, 2].astype(int), pts[:, 3].astype(int)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for k in range(K):
        m_core = (sit == k) & (isb == 0); m_bnd = (sit == k) & (isb == 1)
        ax.scatter(x[m_core], y[m_core], s=14, c=PAL[k], alpha=0.55,
                   edgecolors="none", label=lab(k))
        ax.scatter(x[m_bnd], y[m_bnd], s=14, facecolors="none",
                   edgecolors=PAL[k], linewidths=0.6, alpha=0.7)
    for c in sp["centroids"]:
        k = c["sit"]
        ax.scatter(c["x"], c["y"], s=260, c=PAL[k], marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, title="situations")
    leg._legend_box.align = "left"
    fig.tight_layout()
    save(fig, f"fig_situation_space_{city}")

    # ---- FIG 2: inequity lens (bias z-score heatmap) ----
    bz = np.array(pr["b_z"])                                   # [K x n_genres]
    order = sorted(range(K), key=lambda k: -lensKL[k])         # lensKL desc
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    im = ax.imshow(bz[order], cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(genres)))
    ax.set_xticklabels(genres, rotation=40, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{lab(k)} (KL={lensKL[k]:.2f})" for k in order])
    for yi, k in enumerate(order):
        top3 = sorted(range(len(genres)), key=lambda g: -bz[k, g])[:3]
        ax.scatter(top3, [yi] * len(top3), marker="^", s=28, color="black", zorder=3)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_edgecolor("0.5")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("situational bias (z-score)")
    cb.outline.set_linewidth(0.6)
    fig.tight_layout()
    save(fig, f"fig_inequity_lens_{city}")

    # ---- FIG 3: transitions L3 (Markov heatmap) ----
    T = np.array(tr["T"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([lab(k) for k in range(K)])
    ax.set_xlabel("next situation"); ax.set_ylabel("current situation")
    for i in range(K):
        for j in range(K):
            v = T[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.6 else "black",
                    fontweight="bold" if i == j else "normal")
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_edgecolor("0.5")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("P(next | current)")
    cb.outline.set_linewidth(0.6)
    fig.tight_layout()
    save(fig, f"fig_transitions_{city}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
