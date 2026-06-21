"""Final ε selection — boundary-band [10%,30%] + plateau-maximin on ARI, on the
FINAL perception (γ=0.4, depth=3, n=3) with the FINAL per-city K. Closes L2.

Per city, K fixed at its final value, sweep ε ∈ {0.01..0.10}:
  1. boundary fraction (validation, mean seed) + cross-seed ARI (S=5).
  2. keep cells with boundary fraction ∈ [10%,30%].
  3. among in-band cells: plateau-maximin on ARI (in-band rook neighbourhood);
     tie-break: neigh mean ↓, then boundary closest to 20%.
  4. fallback (no in-band cell): ε whose boundary fraction is closest to 20%.
  5. confirm finalist with S=10.
Band-robustness re-check on {[5,35],[10,30],[15,25]} (free re-filter).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
    estimate_macro_transition, find_attractors, fit_contribution_functions)
from pipeline.step02_models.xsage.l2_comprehension import (
    _assign, adjusted_rand_score, fit_rough_kmeans)

GAMMA, DEPTH, N, H, BETA = 0.4, 3, 3, 2, 0.7
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
EPS_GRID = [0.010, 0.020, 0.030, 0.050, 0.070, 0.100]
SEEDS_SEL = (42, 43, 44, 45, 46)
SEEDS_CONFIRM = tuple(range(42, 52))
PRIMARY_BAND = (0.10, 0.30)
BANDS = [(0.05, 0.35), (0.10, 0.30), (0.15, 0.25)]
MAX_ITER = 80
# ε* from the overnight run (K different there) for the old-vs-new comparison
OVERNIGHT_EPS = {"istanbul": 0.01, "bangkok": 0.03, "nyc_tist": 0.02,
                 "saopaulo": 0.03, "tokyo_tist": 0.07}
OVERNIGHT_K = {"istanbul": 4, "bangkok": 4, "nyc_tist": 4, "saopaulo": 4, "tokyo_tist": 4}

_VC = {}


def build_v(city, split):
    key = (city, split)
    if key in _VC:
        return _VC[key]
    ds = _load_city(city)
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    if "contrib" not in _VC.get(("__shared__", city), {}):
        contrib = fit_contribution_functions(ds["df_train"], m2i,
                                             attributes=DEFAULT_ATTRIBUTES,
                                             max_depth=DEPTH, min_leaf=200)
        W = estimate_macro_transition(ds["df_train"], m2i,
                                      transit_macros=["Travel & Transport"],
                                      transit_mode="keep")
        attractors = find_attractors(W, exclude_indices=None)
        _VC[("__shared__", city)] = {"contrib": contrib, "W": W,
                                     "attractors": attractors, "ds": ds}
    sh = _VC[("__shared__", city)]
    if split == "train":
        tgt, hist = ds["df_train"], ds["df_train"]
    else:
        tgt, hist = ds["df_val"], ds["df_train"]
    l0 = build_recent_window(tgt, hist, m2i, n=N)
    c = sh["contrib"].transform(tgt)
    m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=GAMMA)
    e = compute_intent(m, sh["W"], sh["attractors"], H=H, beta=BETA, mode="hard")
    v = np.concatenate([c, e], axis=1).astype(np.float32)
    _VC[key] = v
    return v


def measure(city, K, eps, seeds):
    vtr, vva = build_v(city, "train"), build_v(city, "val")
    labels, bf = [], []
    for s in seeds:
        r = fit_rough_kmeans(vtr, K=K, eps=eps, seed=s, max_iter=MAX_ITER)
        _, k_va, _, isb_va = _assign(vva, r.prototypes, eps)
        labels.append(k_va.astype(np.int32)); bf.append(float(isb_va.mean()))
    aris = [adjusted_rand_score(labels[i], labels[j])
            for i, j in combinations(range(len(seeds)), 2)]
    return float(np.mean(aris)), float(np.min(aris)), float(np.mean(bf))


def rook(eps):
    i = EPS_GRID.index(eps)
    out = []
    if i > 0: out.append(EPS_GRID[i - 1])
    if i < len(EPS_GRID) - 1: out.append(EPS_GRID[i + 1])
    return out


def select_eps(recs, band):
    """recs: list of dict(eps, ari, bfrac). Returns (eps*, in_band)."""
    lo, hi = band
    ari = {r["eps"]: r["ari"] for r in recs}
    bf = {r["eps"]: r["bfrac"] for r in recs}
    in_band = {e for e in ari if lo <= bf[e] <= hi}
    if in_band:
        scored = []
        for e in in_band:
            nb = [e] + [x for x in rook(e) if x in in_band]
            P = min(ari[x] for x in nb)
            scored.append((P, np.mean([ari[x] for x in nb]),
                           -abs(bf[e] - 0.20), e))
        scored.sort(reverse=True)
        return scored[0][3], True
    e = min(bf, key=lambda x: abs(bf[x] - 0.20))      # fallback: closest to 20%
    return e, False


def main():
    rows, sel = [], {}
    for city in CITIES:
        K = K_FINAL[city]
        recs = []
        for eps in EPS_GRID:
            am, amin, bfrac = measure(city, K, eps, SEEDS_SEL)
            recs.append({"eps": eps, "ari": am, "ari_min": amin, "bfrac": bfrac})
            print(f"[{city}] K={K} ε={eps}  bfrac={bfrac:.1%}  ARI={am:.3f} "
                  f"(min {amin:.3f})", flush=True)
        estar, in_band = select_eps(recs, PRIMARY_BAND)
        cam, camin, cbf = measure(city, K, estar, SEEDS_CONFIRM)   # S=10 confirm
        sel[city] = {"eps": estar, "in_band": in_band, "bfrac": cbf,
                     "ari_confirm": cam}
        for r in recs:
            rows.append({"city": city, "K": K, **r,
                         "in_band_10_30": PRIMARY_BAND[0] <= r["bfrac"] <= PRIMARY_BAND[1],
                         "eps_star": estar, "selected": r["eps"] == estar})
        print(f"  → {city}: ε*={estar}  in_band={in_band}  "
              f"boundary(S10)={cbf:.1%}  ARI(S10)={cam:.3f}", flush=True)

    OUT = CLEAN / "outputs_results" / "validation"
    pd.DataFrame(rows).to_csv(OUT / "epsilon_final.csv", index=False)

    # band robustness (re-filter the computed cells)
    by_city = {c: [{"eps": r["eps"], "ari": r["ari"], "bfrac": r["bfrac"]}
                   for r in rows if r["city"] == c] for c in CITIES}
    rob = []
    for band in BANDS:
        for c in CITIES:
            e, ib = select_eps(by_city[c], band)
            rob.append({"band": f"{band[0]:.0%}-{band[1]:.0%}", "city": c,
                        "eps_star": e, "in_band": ib})
    pd.DataFrame(rob).to_csv(OUT / "epsilon_band_robustness.csv", index=False)

    print("\n\n=== ε* FINALE per città ===")
    print(f"{'city':<12}{'K':>3}{'ε*':>7}{'boundary':>10}{'in banda':>10}"
          f"{'ARI S10':>9}  {'ε notte (K=4)':>14}")
    for c in CITIES:
        s = sel[c]
        print(f"{c:<12}{K_FINAL[c]:>3}{s['eps']:>7}{s['bfrac']:>9.1%}"
              f"{('sì' if s['in_band'] else 'FALLBACK'):>10}{s['ari_confirm']:>9.3f}"
              f"  {OVERNIGHT_EPS[c]:>14}")

    print("\n=== Robustezza banda → ε* per città ===")
    piv = pd.DataFrame(rob).pivot(index="city", columns="band", values="eps_star")
    print(piv.to_string())
    print(f"\n→ {OUT/'epsilon_final.csv'}\n→ {OUT/'epsilon_band_robustness.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
