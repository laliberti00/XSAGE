"""Sensitivity of n (recency-window length, L1b) — justify n=3 (min edge of the
grid) as an INNOCUOUS edge, not an un-bracketed downward optimum. Perception
fixed (γ=0.4, depth=3, H=2, β=0.7), K/ε at their final per-city values; vary
only n ∈ {2,3,5,7,10} (n=2 probes below the edge).

Measures per (city, n):
  - ARI cross-seed (S=3) at final K/ε  — n enters v → affects clustering.
  - Silhouette at final K (hard, eps=0, sampled 25k).
  - Intent proxy: mean normalised entropy of e over attractors (lower = more
    peaked/informative).
  - mean n_prior + share-with-history: how much recent history ACTUALLY enters
    (if it saturates by n=3, larger n adds nothing → data-driven support for n=3).
  - (optional) NYC X-SAGE R@20 across n, via the validated overnight machinery.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
    estimate_macro_transition, find_attractors, fit_contribution_functions)
from pipeline.step02_models.xsage.l2_comprehension import (
    adjusted_rand_score, fit_rough_kmeans)

GAMMA, DEPTH, H, BETA = 0.4, 3, 2, 0.7
N_GRID = [2, 3, 5, 7, 10]
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02,
             "saopaulo": 0.05, "tokyo_tist": 0.07}
SEEDS = (42, 43, 44)
SIL_SAMPLE = 25000
SEED = 42

_CTX = {}


def build(city, n):
    if city not in _CTX:
        ds = _load_city(city); m2i = ds["macro_to_idx"]
        contrib = fit_contribution_functions(ds["df_train"], m2i,
                                             attributes=DEFAULT_ATTRIBUTES,
                                             max_depth=DEPTH, min_leaf=200)
        c = contrib.transform(ds["df_train"])
        W = estimate_macro_transition(ds["df_train"], m2i,
                                      transit_macros=["Travel & Transport"],
                                      transit_mode="keep")
        A = find_attractors(W, exclude_indices=None)
        _CTX[city] = {"ds": ds, "m2i": m2i, "c": c, "W": W, "A": A,
                      "n_macros": ds["n_macros"]}
    x = _CTX[city]
    l0 = build_recent_window(x["ds"]["df_train"], x["ds"]["df_train"], x["m2i"], n=n)
    m = compute_profile(l0.recent_macro, l0.n_prior, x["n_macros"], gamma=GAMMA)
    e = compute_intent(m, x["W"], x["A"], H=H, beta=BETA, mode="hard")
    v = np.concatenate([x["c"], e], axis=1).astype(np.float32)
    # intent entropy over attractors (normalised); n_prior stats
    Acols = np.where(x["A"])[0]
    ea = e[:, Acols].astype(np.float64)
    ea = ea / np.maximum(ea.sum(1, keepdims=True), 1e-12)
    ent = -np.sum(np.where(ea > 0, ea * np.log(ea), 0.0), axis=1)
    ent_norm = float(np.mean(ent) / np.log(len(Acols)))
    npri = l0.n_prior
    return v, ent_norm, float(npri.mean()), float((npri > 0).mean())


def _nyc_r20():
    """Optional: NYC X-SAGE R@20 across n via the overnight machinery."""
    spec = importlib.util.spec_from_file_location(
        "ov", str(CLEAN / "scripts" / "overnight_selection.py"))
    ov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ov)
    from xsage import pipeline
    out = {}
    for n in N_GRID:
        perc = {"gamma": GAMMA, "depth": DEPTH, "n": n}
        prep, _, _ = ov.build_prep_for_report("nyc_tist", perc,
                                               K_FINAL["nyc_tist"], EPS_FINAL["nyc_tist"])
        blind_top, u = ov.blind_topk_test(prep)
        srows = ov.stage_b_stats_per_request(prep, blind_top, u)
        sinks, _g = ov.identify_sinks_per_request(srows)
        agg = pipeline.aggregate(pipeline.score_request_metrics(prep, sinks,
                                                                kappa=1.0, lam=1.0))
        out[n] = (float(agg["xsage"]["R20"]), float(agg["blind"]["R20"]), sinks)
        print(f"  [nyc R@20] n={n}  xsage={out[n][0]:.4f}  blind={out[n][1]:.4f}  "
              f"sinks={sinks}", flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for city in CITIES:
        K, eps = K_FINAL[city], EPS_FINAL[city]
        sidx = None
        for n in N_GRID:
            v, ent, npri_mean, hist_share = build(city, n)
            if sidx is None:
                N = v.shape[0]
                sidx = rng.choice(N, SIL_SAMPLE, replace=False) if N > SIL_SAMPLE else np.arange(N)
            # ARI cross-seed at final K/eps (train core labels)
            labs = [fit_rough_kmeans(v, K=K, eps=eps, seed=s, max_iter=80).core_label
                    for s in SEEDS]
            ari = float(np.mean([adjusted_rand_score(labs[i], labs[j])
                                 for i, j in combinations(range(len(SEEDS)), 2)]))
            # silhouette at final K (hard eps=0)
            rh = fit_rough_kmeans(v, K=K, eps=0.0, seed=SEED, max_iter=80)
            lab = rh.core_label
            sil = (float(silhouette_score(v[sidx], lab[sidx], metric="euclidean"))
                   if len(np.unique(lab)) > 1 else float("nan"))
            rows.append({"city": city, "n": n, "K": K, "eps": eps,
                         "ARI": round(ari, 4), "silhouette": round(sil, 4),
                         "intent_entropy": round(ent, 4),
                         "n_prior_mean": round(npri_mean, 3),
                         "history_share": round(hist_share, 4)})
            print(f"[{city}] n={n}  ARI={ari:.3f} sil={sil:.4f} "
                  f"e_entropy={ent:.3f}  n_prior_mean={npri_mean:.2f} "
                  f"hist={hist_share:.1%}", flush=True)

    r20 = {}
    try:
        print("\n--- optional: NYC X-SAGE R@20 across n ---")
        r20 = _nyc_r20()
    except Exception as ex:
        print(f"  (R@20 opzionale non eseguito: {ex})")
    for r in rows:
        if r["city"] == "nyc_tist" and r["n"] in r20:
            r["R20_xsage"] = round(r20[r["n"]][0], 4)
            r["R20_blind"] = round(r20[r["n"]][1], 4)

    df = pd.DataFrame(rows)
    out = CLEAN / "outputs_results" / "validation" / "n_sensitivity.csv"
    df.to_csv(out, index=False)

    print("\n\n=== Sintesi per città (ARI / silhouette / n_prior_mean su n) ===")
    for city in CITIES:
        sub = df[df.city == city].set_index("n")
        ari = " ".join(f"{n}:{sub.loc[n,'ARI']:.2f}" for n in N_GRID)
        sil = " ".join(f"{n}:{sub.loc[n,'silhouette']:.3f}" for n in N_GRID)
        npr = " ".join(f"{n}:{sub.loc[n,'n_prior_mean']:.2f}" for n in N_GRID)
        print(f"  {city:<12} ARI[{ari}]")
        print(f"  {'':<12} sil[{sil}]")
        print(f"  {'':<12} n_prior[{npr}]")
    print(f"\n→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
