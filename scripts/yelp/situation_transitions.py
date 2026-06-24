"""[venv-xsage] L3 / PROIEZIONE: matrice di transizione fra situazioni T (catena di Markov sugli stati)
per la figura del grafo di flusso. Per ml-1m: T[i][j]=P(prossima situazione=j | attuale=i),
persistenza (diagonale), + nomi/posizioni (riuso dei centroidi PCA). Stampa JSON.
Uso:  python scripts/yelp/situation_transitions.py ml1m
"""
import json, sys
from pathlib import Path
import numpy as np
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    nmac = D0["n_macros"]; cmt = D0["cmt"]; i2m = {v: k for k, v in D0["ds"]["macro_to_idx"].items()}
    vtr, vva = D0["vs"]["train"], D0["vs"]["val"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z, cmt, K, nmac, alpha=ALPHA)

    # df_train è già ordinato per (u_idx, tempo) dal preprocess → transizioni consecutive intra-utente
    u = D0["ds"]["df_train"]["u_idx"].values.astype(np.int64)
    T = np.zeros((K, K))
    same = u[1:] == u[:-1]
    for a, b, s in zip(z[:-1], z[1:], same):
        if s: T[a, b] += 1
    rowsum = T.sum(1, keepdims=True)
    Tn = np.divide(T, np.maximum(rowsum, 1))
    size = np.bincount(z, minlength=K) / len(z)
    sits = [{"k": int(k), "size_pct": round(100 * size[k], 1),
             "persist_pct": round(100 * Tn[k, k], 1),
             "top": [i2m[g] for g in sorted(range(nmac), key=lambda g: -b_z[k, g])[:2]]} for k in range(K)]
    print(json.dumps({"city": city, "K": int(K),
                      "T": [[round(float(x), 3) for x in row] for row in Tn],
                      "situations": sits}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
