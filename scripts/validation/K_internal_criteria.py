"""Diagnostic: choose K by INTERNAL cluster-quality criteria (not cross-seed
ARI, which is biased toward small K). On the SAME descriptors v=[c̃‖e] of the
overnight winning perception (γ=0.4, depth=6, n=3), for each city × K compute
Silhouette / Calinski-Harabasz / Davies-Bouldin on the HARD assignment.

Methodological notes (reported):
  * internal criteria assume HARD assignment → we use the nearest-centroid
    label (core_label = argmin distance), eps=0 fit, NOT the rough boundary.
  * same (Euclidean) distance as the rough k-means used everywhere else.
  * the descriptor mixes context-entropy and intent-distribution → criteria are
    an INDICATION, not a verdict; if the three disagree, geometry isn't
    spherical — flag it, don't force a K.
  * Silhouette is O(n²) → sampled (25k) where needed; flagged per row.
"""
import sys
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
from pipeline.step02_models.xsage.l2_comprehension import fit_rough_kmeans

# Winning perception from the overnight run — FIXED here (we isolate K).
GAMMA, DEPTH, N, H, BETA = 0.4, 6, 3, 2, 0.7
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_GRID = [3, 4, 5, 6, 7, 8, 9]
SIL_SAMPLE = 25000          # silhouette subsample cap
SEED = 42
DOSSIER_K = {"istanbul": 8, "bangkok": 4, "nyc_tist": 6, "saopaulo": 8, "tokyo_tist": 4}
STABILITY_K = 4             # what ARI-maximin chose for every city


def build_train_v(city: str) -> np.ndarray:
    ds = _load_city(city)
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    l0 = build_recent_window(ds["df_train"], ds["df_train"], m2i, n=N)
    contrib = fit_contribution_functions(ds["df_train"], m2i,
                                         attributes=DEFAULT_ATTRIBUTES,
                                         max_depth=DEPTH, min_leaf=200)
    c = contrib.transform(ds["df_train"])
    W = estimate_macro_transition(ds["df_train"], m2i,
                                  transit_macros=["Travel & Transport"],
                                  transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)
    m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=GAMMA)
    e = compute_intent(m, W, attractors, H=H, beta=BETA, mode="hard")
    return np.concatenate([c, e], axis=1).astype(np.float32)


def main():
    rows = []
    rng = np.random.default_rng(SEED)
    for city in CITIES:
        v = build_train_v(city)
        n = v.shape[0]
        print(f"\n=== {city}  (n={n}, D={v.shape[1]}) ===", flush=True)
        # fixed subsample indices for silhouette (same across K → comparable)
        sampled = n > SIL_SAMPLE
        sidx = (rng.choice(n, SIL_SAMPLE, replace=False) if sampled
                else np.arange(n))
        for K in K_GRID:
            r = fit_rough_kmeans(v, K=K, eps=0.0, seed=SEED, max_iter=80)
            labels = r.core_label
            n_eff = int(len(np.unique(labels)))
            if n_eff < 2:
                sil = ch = db = float("nan")
            else:
                sil = float(silhouette_score(v[sidx], labels[sidx],
                                             metric="euclidean"))
                ch = float(calinski_harabasz_score(v, labels))
                db = float(davies_bouldin_score(v, labels))
            rows.append({"city": city, "K": K, "n_eff": n_eff,
                         "silhouette": round(sil, 4),
                         "calinski_harabasz": round(ch, 1),
                         "davies_bouldin": round(db, 4),
                         "sil_sampled": sampled, "n_points": n})
            print(f"  K={K} n_eff={n_eff}  sil={sil:.4f}  CH={ch:.0f}  "
                  f"DB={db:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = CLEAN / "outputs_results" / "validation" / "K_internal_criteria.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # per-city preferred K by each criterion
    print("\n\n=== K preferito per criterio (per città) ===")
    print(f"{'city':<12} {'sil':>4} {'CH':>4} {'DB':>4}  {'concorda?':>10} "
          f"{'stab':>4} {'dossier':>7}")
    summary = []
    for city in CITIES:
        sub = df[df.city == city]
        k_sil = int(sub.loc[sub.silhouette.idxmax(), "K"])
        k_ch = int(sub.loc[sub.calinski_harabasz.idxmax(), "K"])
        k_db = int(sub.loc[sub.davies_bouldin.idxmin(), "K"])
        ks = [k_sil, k_ch, k_db]
        agree = "SÌ" if max(ks) - min(ks) <= 1 else "no"
        print(f"{city:<12} {k_sil:>4} {k_ch:>4} {k_db:>4}  {agree:>10} "
              f"{STABILITY_K:>4} {DOSSIER_K[city]:>7}")
        summary.append({"city": city, "K_sil": k_sil, "K_CH": k_ch,
                        "K_DB": k_db, "agree": agree})
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
