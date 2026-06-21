"""Final K selection — automatic two-branch rule on perception (γ=0.4, depth=3,
n=3). Internal criteria are RECOMPUTED at depth=3 (geometry changes with depth).

Rule, per city, on v=[c̃‖e], hard assignment, K ∈ {3..9}:
  1. silhouette / CH / DB per K (silhouette sampled 25k, declared).
  2. Regime: SHARP if max-silhouette ≥ sharpness threshold (data-calibrated as
     the midpoint of the largest gap in the cities' max-silhouette), else DIFFUSE.
  3. SHARP branch  → K* = silhouette-preferred K (primary); report CH/DB accord.
  4. DIFFUSE branch → K* = SMALLEST K with cross-seed ARI ≥ adequacy threshold
     (stability as a CONSTRAINT, not an objective → no coarsening collapse).
     If none meets it → argmax ARI (flagged).

Thresholds are calibrated from the data and their robustness is demonstrated.
Geometry: eps=0 hard fit. ARI: cross-seed S=3 at ε=0.02 (core_label). Train.
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

GAMMA, DEPTH, N, H, BETA = 0.4, 3, 3, 2, 0.7
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_GRID = [3, 4, 5, 6, 7, 8, 9]
EPS_FIT = 0.02
SEEDS = (42, 43, 44)
SIL_SAMPLE = 25000
SEED = 42
ADEQUACY_DEFAULT = 0.70
ADEQUACY_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]
DOSSIER_K = {"istanbul": 8, "bangkok": 4, "nyc_tist": 6, "saopaulo": 8, "tokyo_tist": 4}
DEPTH6_INTERNAL_K = {"istanbul": "amb", "bangkok": 4, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 3}


def build_v(city: str) -> np.ndarray:
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
    rng = np.random.default_rng(SEED)
    per = {}      # city -> DataFrame over K
    for city in CITIES:
        v = build_v(city)
        n = v.shape[0]
        sidx = rng.choice(n, SIL_SAMPLE, replace=False) if n > SIL_SAMPLE else np.arange(n)
        recs = []
        for K in K_GRID:
            rh = fit_rough_kmeans(v, K=K, eps=0.0, seed=SEED, max_iter=80)
            lab = rh.core_label
            if len(np.unique(lab)) < 2:
                sil = ch = db = float("nan")
            else:
                sil = float(silhouette_score(v[sidx], lab[sidx], metric="euclidean"))
                ch = float(calinski_harabasz_score(v, lab))
                db = float(davies_bouldin_score(v, lab))
            fits = [fit_rough_kmeans(v, K=K, eps=EPS_FIT, seed=s, max_iter=80)
                    for s in SEEDS]
            labs = [f.core_label for f in fits]
            ari = float(np.mean([adjusted_rand_score(labs[i], labs[j])
                                 for i, j in combinations(range(len(SEEDS)), 2)]))
            recs.append({"city": city, "K": K, "silhouette": round(sil, 4),
                         "CH": round(ch, 1), "DB": round(db, 4),
                         "ARI": round(ari, 4)})
            print(f"[{city}] K={K} sil={sil:.4f} CH={ch:.0f} DB={db:.4f} "
                  f"ARI={ari:.3f}", flush=True)
        per[city] = pd.DataFrame(recs)

    # --- peakedness + data-calibrated sharpness threshold ---
    max_sil = {c: float(per[c].silhouette.max()) for c in CITIES}
    sil_range = {c: float(per[c].silhouette.max() - per[c].silhouette.min())
                 for c in CITIES}
    order = sorted(CITIES, key=lambda c: max_sil[c])      # ascending
    vals = [max_sil[c] for c in order]
    gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
    gsize, gi = max(gaps)
    gap_lo, gap_hi = vals[gi], vals[gi + 1]
    sharp_thr = 0.5 * (gap_lo + gap_hi)
    print(f"\nmax-silhouette per città (asc): "
          f"{[(c, round(max_sil[c],3)) for c in order]}")
    print(f"largest gap = {gsize:.3f} tra [{gap_lo:.3f}, {gap_hi:.3f}] → "
          f"soglia nitidezza = {sharp_thr:.3f}")

    def regime(c):
        return "sharp" if max_sil[c] >= sharp_thr else "diffuse"

    def sharp_Kstar(c):
        d = per[c]
        k_sil = int(d.loc[d.silhouette.idxmax(), "K"])
        k_ch = int(d.loc[d.CH.idxmax(), "K"])
        k_db = int(d.loc[d.DB.idxmin(), "K"])
        accord = "sì" if max(k_sil, k_ch, k_db) - min(k_sil, k_ch, k_db) <= 1 else "no"
        return k_sil, k_ch, k_db, accord

    def diffuse_Kstar(c, thr):
        d = per[c].sort_values("K")
        ok = d[d.ARI >= thr]
        if len(ok):
            return int(ok.iloc[0].K), True
        return int(d.loc[d.ARI.idxmax(), "K"]), False   # fallback: argmax ARI

    # --- build final rule rows ---
    rows = []
    final = {}
    for c in CITIES:
        reg = regime(c)
        if reg == "sharp":
            k_sil, k_ch, k_db, accord = sharp_Kstar(c)
            kstar, branch = k_sil, "principale"
            note = f"sil={k_sil},CH={k_ch},DB={k_db},accordo={accord}"
        else:
            kstar, met = diffuse_Kstar(c, ADEQUACY_DEFAULT)
            branch = "fallback"
            note = (f"min K con ARI≥{ADEQUACY_DEFAULT}"
                    if met else f"nessun K≥{ADEQUACY_DEFAULT}→argmaxARI")
        final[c] = (kstar, branch)
        for _, r in per[c].iterrows():
            rows.append({**r.to_dict(), "max_sil": round(max_sil[c], 4),
                         "sil_range": round(sil_range[c], 4), "regime": reg,
                         "K_star": kstar, "branch": branch, "note": note})
    df = pd.DataFrame(rows)
    OUT = CLEAN / "outputs_results" / "validation"
    df.to_csv(OUT / "K_final_rule.csv", index=False)

    # --- robustness: sharpness threshold across the gap ---
    rob = []
    for t in np.round(np.linspace(gap_lo + 0.2 * gsize, gap_hi - 0.2 * gsize, 5), 4):
        sharp_set = sorted([c for c in CITIES if max_sil[c] >= t])
        diff_set = sorted([c for c in CITIES if max_sil[c] < t])
        rob.append({"kind": "nitidezza", "threshold": float(t),
                    "outcome": f"sharp={sharp_set} | diffuse={diff_set}"})
    # --- robustness: adequacy threshold (diffuse cities) ---
    diffuse_cities = [c for c in CITIES if regime(c) == "diffuse"]
    for c in diffuse_cities:
        for thr in ADEQUACY_GRID:
            k, met = diffuse_Kstar(c, thr)
            rob.append({"kind": f"adeguatezza/{c}", "threshold": thr,
                        "outcome": f"K*={k}" + ("" if met else " (argmaxARI, soglia non raggiunta)")})
    pd.DataFrame(rob).to_csv(OUT / "K_rule_robustness.csv", index=False)

    # --- report ---
    print("\n\n=== REGOLA A DUE RAMI — esito (depth=3) ===")
    print(f"{'city':<12}{'regime':<9}{'K*':<4}{'ramo':<12}{'depth6_int':<11}{'dossier':<8}")
    for c in CITIES:
        k, br = final[c]
        print(f"{c:<12}{regime(c):<9}{k:<4}{br:<12}"
              f"{str(DEPTH6_INTERNAL_K[c]):<11}{DOSSIER_K[c]:<8}")

    print("\n=== Robustezza soglia NITIDEZZA (classificazione città) ===")
    for r in rob:
        if r["kind"] == "nitidezza":
            print(f"  thr={r['threshold']:.3f}  {r['outcome']}")
    print("\n=== Robustezza soglia ADEGUATEZZA (K* città diffuse) ===")
    for r in rob:
        if r["kind"].startswith("adeguatezza"):
            print(f"  {r['kind']}  thr={r['threshold']}  {r['outcome']}")
    print(f"\n→ {OUT/'K_final_rule.csv'}\n→ {OUT/'K_rule_robustness.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
