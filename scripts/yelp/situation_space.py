"""[venv-xsage] Proiezione 2D (PCA) dello SPAZIO delle situazioni per la figura-paper.
Standardizza i descrittori v (test), PCA→2D, e per un campione di richieste emette:
x,y, situazione, core/boundary. Più i centroidi e i generi-top per situazione (per i nomi).
Stampa JSON. Uso:  python scripts/yelp/situation_space.py ml1m
"""
import json, sys
from pathlib import Path
import numpy as np
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
from sklearn.decomposition import PCA


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    nmac = D0["n_macros"]; cmt = D0["cmt"]
    i2m = {v: k for k, v in D0["ds"]["macro_to_idx"].items()}
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    _, kte, comp, isb = _assign(vte, fit.prototypes, eps)

    # standardizza + PCA (fit su train per stabilità, transform su test)
    mu, sd = vtr.mean(0), vtr.std(0) + 1e-8
    pca = PCA(n_components=2, random_state=0).fit((vtr - mu) / sd)
    P = pca.transform((vte - mu) / sd)
    # scala in [0,1]
    lo, hi = P.min(0), P.max(0); Pn = (P - lo) / (hi - lo + 1e-9)

    # campione per il plot
    idx = rng.permutation(len(Pn))[:1800]
    pts = [[round(float(Pn[i, 0]), 4), round(float(Pn[i, 1]), 4), int(kte[i]), int(bool(isb[i]))] for i in idx]
    cents = []
    for k in range(K):
        m = kte == k
        cents.append({"sit": k, "x": round(float(Pn[m, 0].mean()), 4), "y": round(float(Pn[m, 1].mean()), 4),
                      "top": [i2m[g] for g in sorted(range(nmac), key=lambda g: -b_z[k, g])[:2]]})
    print(json.dumps({"city": city, "K": int(K),
                      "var": [round(float(x), 3) for x in pca.explained_variance_ratio_],
                      "bfrac": round(float(isb.mean()), 3), "points": pts, "centroids": cents}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
