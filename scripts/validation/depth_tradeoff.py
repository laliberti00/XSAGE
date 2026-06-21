"""Diagnostic: choose context-tree DEPTH (L1a) as an explicit trade-off between
interpretability, descriptor geometry and stability — NOT a blind ARI optimum.

For depth ∈ {2..6}, other perception params fixed (γ=0.4, n=3, H=2, β=0.7), per
city, report three dimensions side by side:
  A. Interpretability — avg #leaves/tree, avg effective depth used.
  B. Geometry — silhouette/CH/DB on the HARD assignment at the city's
     internal-criteria K (from the previous K test), silhouette sampled 25k.
  C. Stability — ARI cross-seed (S=3) at the same K.

K is held FIXED per city (isolate depth): {istanbul 4, bangkok 4, nyc 6,
saopaulo 3, tokyo 3} (from K_internal_criteria; istanbul=4 = its silhouette pick,
geometry ambiguous). ε=0.02 (in-band, representative) only to fit; the hard
labels are core_label = nearest centroid, independent of ε.

Everything on TRAIN (consistent with the K test). Istanbul flatness (silhouette
range across K) reported separately to answer whether its ambiguous geometry is
depth-driven or intrinsic.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
    estimate_macro_transition, find_attractors, fit_contribution_functions)
from pipeline.step02_models.xsage.l2_comprehension import (
    adjusted_rand_score, fit_rough_kmeans)

GAMMA, N, H, BETA = 0.4, 3, 2, 0.7
DEPTHS = [2, 3, 4, 5, 6]
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
FIXED_K = {"istanbul": 4, "bangkok": 4, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 3}
EPS_FIT = 0.02
SEEDS = (42, 43, 44)
SIL_SAMPLE = 25000
SEED = 42
IST_FLAT_K = [3, 5, 7]          # K grid for Istanbul flatness probe


def build_v_and_trees(city: str, depth: int):
    ds = _load_city(city)
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    l0 = build_recent_window(ds["df_train"], ds["df_train"], m2i, n=N)
    contrib = fit_contribution_functions(ds["df_train"], m2i,
                                         attributes=DEFAULT_ATTRIBUTES,
                                         max_depth=depth, min_leaf=200)
    c = contrib.transform(ds["df_train"])
    W = estimate_macro_transition(ds["df_train"], m2i,
                                  transit_macros=["Travel & Transport"],
                                  transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)
    m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=GAMMA)
    e = compute_intent(m, W, attractors, H=H, beta=BETA, mode="hard")
    v = np.concatenate([c, e], axis=1).astype(np.float32)
    leaves = [contrib.trees[a].get_n_leaves() for a in DEFAULT_ATTRIBUTES]
    eff_d = [contrib.trees[a].get_depth() for a in DEFAULT_ATTRIBUTES]
    return v, float(np.mean(leaves)), float(np.mean(eff_d))


def geom(v, labels, sidx):
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan"), float("nan")
    sil = float(silhouette_score(v[sidx], labels[sidx], metric="euclidean"))
    return sil, float(calinski_harabasz_score(v, labels)), \
        float(davies_bouldin_score(v, labels))


def main():
    rng = np.random.default_rng(SEED)
    rows, ist_flat = [], []
    for city in CITIES:
        K = FIXED_K[city]
        sidx = None
        for depth in DEPTHS:
            v, n_leaves, eff_d = build_v_and_trees(city, depth)
            if sidx is None:
                n = v.shape[0]
                sidx = (rng.choice(n, SIL_SAMPLE, replace=False)
                        if n > SIL_SAMPLE else np.arange(n))
            fits = [fit_rough_kmeans(v, K=K, eps=EPS_FIT, seed=s, max_iter=80)
                    for s in SEEDS]
            labs = [f.core_label for f in fits]
            ari = float(np.mean([adjusted_rand_score(labs[i], labs[j])
                                 for i, j in combinations(range(len(SEEDS)), 2)]))
            sil, ch, db = geom(v, labs[0], sidx)
            rows.append({"city": city, "depth": depth, "K": K,
                         "n_foglie": round(n_leaves, 1),
                         "depth_effettivo": round(eff_d, 2),
                         "silhouette": round(sil, 4),
                         "CH": round(ch, 1), "DB": round(db, 4),
                         "ARI": round(ari, 4)})
            print(f"[{city}] depth={depth} K={K}  foglie={n_leaves:.1f} "
                  f"effD={eff_d:.2f}  sil={sil:.4f} CH={ch:.0f} DB={db:.4f}  "
                  f"ARI={ari:.3f}", flush=True)
            if city == "istanbul":
                sils = []
                for kk in IST_FLAT_K:
                    r = fit_rough_kmeans(v, K=kk, eps=0.0, seed=SEED, max_iter=80)
                    s, _, _ = geom(v, r.core_label, sidx)
                    sils.append(s)
                rng_sil = float(max(sils) - min(sils))
                ist_flat.append({"depth": depth, "sil_K3": round(sils[0], 4),
                                 "sil_K5": round(sils[1], 4),
                                 "sil_K7": round(sils[2], 4),
                                 "sil_range": round(rng_sil, 4)})
                print(f"    [istanbul flatness] depth={depth} "
                      f"sil(K3,5,7)={[round(x,3) for x in sils]} "
                      f"range={rng_sil:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = CLEAN / "outputs_results" / "validation" / "depth_tradeoff.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n\n=== Istanbul: la geometria si 'sblocca' a depth più bassi? ===")
    print("(sil_range piccolo = silhouette piatta su K = nessuna struttura netta)")
    for r in ist_flat:
        print(f"  depth={r['depth']}  sil(K3/5/7)={r['sil_K3']}/{r['sil_K5']}"
              f"/{r['sil_K7']}  range={r['sil_range']}")

    print("\n=== ARI: prezzo dell'interpretabilità (depth 6 → shallow) per città ===")
    for city in CITIES:
        sub = df[df.city == city].set_index("depth")
        a6 = sub.loc[6, "ARI"]; a3 = sub.loc[3, "ARI"]; a4 = sub.loc[4, "ARI"]
        print(f"  {city:<12} ARI: depth6={a6:.3f}  depth4={a4:.3f}  "
              f"depth3={a3:.3f}   Δ(6→3)={a6-a3:+.3f}")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
