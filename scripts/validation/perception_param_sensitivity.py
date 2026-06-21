"""Phase 02 (Perception) — parameter sensitivity measured END-TO-END on the
cross-seed ARI of the rough-k-means situations.

The descriptor v = [c̃ ‖ e] is built with the SAME tested functions used to
produce the dossier fit.npz (imported from the old repo so the derived columns
cat_target / prev_geohash5 / intent_last_cat_idx are computed identically). We
vary ONE perception parameter at a time, hold (K, ε) at each city's selected
values, and measure the cross-seed ARI (Hubert-Arabie) over seeds {42,43,44}.

Output: outputs_results/validation/perception_param_sensitivity.csv
        (parameter, value, city, ari_mean, ari_min, K, eps, n_iters)
"""
import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
    estimate_macro_transition, find_attractors, fit_contribution_functions)
from pipeline.step02_models.xsage.l2_comprehension import (
    adjusted_rand_score, fit_rough_kmeans)

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")

# (K, ε) selected by the dossier sweep, per summary.json — held fixed here.
CITY_KEPS = {
    "istanbul": (8, 0.010), "bangkok": (4, 0.030), "nyc_tist": (6, 0.020),
    "saopaulo": (8, 0.020), "tokyo_tist": (4, 0.050),
}
DEFAULTS = dict(n=5, gamma=0.6, H=2, beta=0.7, depth=3)
SEEDS = (42, 43, 44)
MAX_ITER = 80


def build_train_v(ds, n, gamma, H, beta, depth):
    """Replicate orchestrator._build_perception for the TRAIN split only,
    exposing max_depth. Same functions, same order, hard/keep mode."""
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    l0 = build_recent_window(ds["df_train"], ds["df_train"], m2i, n=n)
    contrib = fit_contribution_functions(ds["df_train"], m2i,
                                         attributes=DEFAULT_ATTRIBUTES,
                                         max_depth=depth, min_leaf=200)
    c = contrib.transform(ds["df_train"])
    W = estimate_macro_transition(ds["df_train"], m2i,
                                  transit_macros=["Travel & Transport"],
                                  transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)
    m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=gamma)
    e = compute_intent(m, W, attractors, H=H, beta=beta, mode="hard")
    return np.concatenate([c, e], axis=1).astype(np.float32)


def ari_cross_seed(v, K, eps):
    labels, iters = [], []
    for s in SEEDS:
        r = fit_rough_kmeans(v, K=K, eps=eps, seed=s, max_iter=MAX_ITER)
        labels.append(r.core_label); iters.append(r.n_iters)
    aris = [adjusted_rand_score(labels[i], labels[j])
            for i, j in combinations(range(len(SEEDS)), 2)]
    return float(np.mean(aris)), float(np.min(aris)), int(np.mean(iters))


def sweep(cities, grids, probe=False):
    rows = []
    for city in cities:
        t0 = time.time()
        ds = _load_city(city)
        K, eps = CITY_KEPS[city]
        print(f"[{city}] K={K} eps={eps}  (load {time.time()-t0:.0f}s)", flush=True)
        for param, values in grids.items():
            for val in values:
                kw = dict(DEFAULTS); kw[param] = val
                tb = time.time()
                v = build_train_v(ds, **kw)
                am, amin, it = ari_cross_seed(v, K, eps)
                rows.append({"parameter": param, "value": val, "city": city,
                             "ari_mean": round(am, 4), "ari_min": round(amin, 4),
                             "K": K, "eps": eps, "n_iters": it, "B": v.shape[0],
                             "D": v.shape[1]})
                print(f"    {param}={val}: ARI={am:.3f} (min {amin:.3f}) "
                      f"[{time.time()-tb:.0f}s]", flush=True)
                if probe:
                    return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="one config, timing only")
    ap.add_argument("--cities", default="all")
    args = ap.parse_args()

    cities = (list(CITY_KEPS) if args.cities == "all"
              else args.cities.split(","))
    grids = {
        "depth": [2, 3, 4, 5],
        "gamma": [0.3, 0.5, 0.6, 0.7, 0.9],
        "beta":  [0.3, 0.5, 0.7, 0.9],
        "n":     [3, 5, 7, 10],
        "H":     [1, 2, 3],
    }
    if args.probe:
        rows = sweep(cities[:1], {"depth": [3]}, probe=True)
        print("PROBE done:", rows)
        return 0

    rows = sweep(cities, grids)
    df = pd.DataFrame(rows)
    OUT = CLEAN / "outputs_results" / "validation"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "perception_param_sensitivity.csv", index=False)
    print(f"\n→ {OUT/'perception_param_sensitivity.csv'}  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
