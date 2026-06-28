"""[venv-xsage] Figure di explainability/trustworthiness dai JSON già prodotti (ml1m).
Genera 3 figure publication-ready (PNG 200dpi + PDF) in outputs_results/explain/:
  fig1_situation_space  — spazio PCA: cluster nominati + regioni di incertezza (boundary)
  fig2_inequity_lens    — lente: distribuzione-categoria per situazione + pozzi (lensKL alto)
  fig3_transitions      — proiezione L3: matrice di transizione + persistenze (dove si muove)
Uso:  python scripts/yelp/plot_explain.py [city=ml1m]
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean"); EXP = CLEAN / "outputs_results" / "explain"
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})


def save(fig, name):
    fig.savefig(EXP / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(EXP / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  -> explain/{name}.png + .pdf")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    sp = json.load(open(EXP / f"situation_space_{city}.json"))
    pr = json.load(open(EXP / f"situation_profiles_{city}.json"))
    tr = json.load(open(EXP / f"situation_transitions_{city}.json"))
    K = pr["K"]; genres = pr["genres"]
    lab = {s["k"]: s["label"] for s in pr["situations"]}
    lensKL = {s["k"]: s["lens_KL"] for s in pr["situations"]}
    nm_f = EXP / f"situation_names_{city}.json"                # nomi curati (override) se presenti
    names = {int(k): v for k, v in json.load(open(nm_f)).items()} if nm_f.exists() else lab
    short = {k: f"S{k}: {names.get(k, lab[k])}" for k in range(K)}

    # ---- FIG 1: spazio delle situazioni (PCA) ----
    pts = np.array(sp["points"]); var = sp["var"]; bfrac = sp["bfrac"]
    x, y, sit, isb = pts[:, 0], pts[:, 1], pts[:, 2].astype(int), pts[:, 3].astype(int)
    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(K):
        m_core = (sit == k) & (isb == 0); m_bnd = (sit == k) & (isb == 1)
        ax.scatter(x[m_core], y[m_core], s=14, c=PAL[k], alpha=0.75, edgecolors="none", label=short[k])
        ax.scatter(x[m_bnd], y[m_bnd], s=16, facecolors="none", edgecolors=PAL[k], linewidths=0.6, alpha=0.5)
    for c in sp["centroids"]:
        k = c["sit"]; ax.scatter(c["x"], c["y"], s=320, c=PAL[k], marker="*", edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(f"S{k}", (c["x"], c["y"]), fontsize=12, fontweight="bold", ha="center", va="center", color="white", zorder=6)
    ax.set_xlabel(f"PC1 ({var[0]*100:.0f}% var)"); ax.set_ylabel(f"PC2 ({var[1]*100:.0f}% var)")
    ax.set_title(f"Spazio delle situazioni — {city} (★=centroide, ○=boundary/incertezza, {bfrac*100:.0f}% boundary)")
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False, title="situazioni")
    leg._legend_box.align = "left"
    bnd_handle = Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="grey", linestyle="none", label="boundary (incerto)")
    save(fig, f"fig1_situation_space_{city}")

    # ---- FIG 2: lente dei pozzi di iniquità (b_z per situazione + lensKL) ----
    bz = np.array(pr["b_z"])                                  # [K x n_genres]
    order = sorted(range(K), key=lambda k: -lensKL[k])        # sink (KL alto) in alto
    fig, ax = plt.subplots(figsize=(11, 0.7 * K + 2.2))
    vmax = np.abs(bz).max()
    im = ax.imshow(bz[order], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(genres))); ax.set_xticklabels(genres, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"S{k}  (lensKL={lensKL[k]:.2f})" + ("  ← pozzo" if lensKL[k] == max(lensKL.values()) else "") for k in order], fontsize=9)
    for yi, k in enumerate(order):
        top3 = sorted(range(len(genres)), key=lambda g: -bz[k, g])[:3]
        for g in top3:
            ax.text(g, yi, "▲", ha="center", va="center", fontsize=8, color="black")
    ax.set_title(f"Lente per-situazione: bias di categoria b̃ (z-score) — {city}\n▲ = genere favorito; lensKL alto = situazione che concentra (pozzo)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="bias z-score")
    save(fig, f"fig2_inequity_lens_{city}")

    # ---- FIG 3: proiezione L3 (matrice transizione + persistenze) ----
    T = np.array(tr["T"]); persist = {s["k"]: s["persist_pct"] for s in tr["situations"]}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k}" for k in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([short[k] for k in range(K)], fontsize=9)
    for i in range(K):
        for j in range(K):
            v = T[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.5 else "black", fontweight="bold" if i == j else "normal")
    ax.set_xlabel("situazione successiva"); ax.set_ylabel("situazione corrente")
    ax.set_title(f"Proiezione L3 — dove si muove la situazione (Markov) — {city}\ndiagonale = persistenza (es. S2 stabile {persist.get(2,0):.0f}%, S3 transitoria {persist.get(3,0):.0f}%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="P(succ | corr)")
    save(fig, f"fig3_transitions_{city}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
