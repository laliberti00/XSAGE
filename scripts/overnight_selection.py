"""Overnight JOINT plateau-aware selection of the situation parameters
(K, ε, γ, depth, n) on VALIDATION, followed by a full metrics report on TEST.

LANCIA-E-DORMI: checkpointed, resumable, idempotent, resilient, logged.
Re-running after an interruption resumes from the cells already on disk.

Run:  python -m scripts.overnight_selection          (or ./run_overnight.sh)

It NEVER touches the test split during selection. Test is used only at the very
end, for the report metrics on the SELECTED parameters.

--------------------------------------------------------------------------- """
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN))     # clean xsage modules
sys.path.insert(0, str(OLD))       # old tested perception/clustering machinery

OUT = CLEAN / "outputs_results" / "overnight"
LOGDIR = CLEAN / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# FIXED DECISIONS (made by the user; do NOT re-litigate). Constants up top.
# --------------------------------------------------------------------------- #
# - Selection on VALIDATION only; TEST stays intact until the report step.
# - Plateau-aware MAXIMIN criterion (robust-max): each grid cell's value is the
#   mean cross-seed ARI; the plateau score P(c) = min over {c} ∪ rook-neighbours
#   (±1 step on exactly one axis) of that value. We maximise P. Tie-breaks:
#   neighbourhood mean → plateau width → parsimony (smaller depth/n) → boundary
#   fraction closest to 20%.
# - Perception params γ/depth/n are SHARED across cities; K/ε are PER-CITY.
# - H and β are already robust → FIXED, NOT swept.
H_FIXED = 2
BETA_FIXED = 0.7
# - Parameters are selected on the ARI ONLY (no fairness metric in the loop →
#   no circularity). Fairness/accuracy metrics are computed AFTER, on test.
# - Sink rule made COHERENT per-request: BOTH the LT_k numerator and the
#   LT_available baseline are averaged over REQUESTS (the old code averaged the
#   baseline over unique users — fixed here). Thresholds: kl_mult=1.5, lt_gap=0.05.
SINK_KL_MULT = 1.5
SINK_LT_GAP = 0.05
ALPHA_BIAS = 50.0          # Dirichlet pseudo-count for the Stage-B bias (z-scored)
SHORT_HEAD = 0.20          # top-20% popularity = head; rest = long-tail
K_TOP = 20
BATCH = 1024
MAX_ITER = 80

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]

# (K, ε) held at the dossier values during Stage A (perception scan).
DOSSIER_KEPS = {
    "istanbul": (8, 0.010), "bangkok": (4, 0.030), "nyc_tist": (6, 0.020),
    "saopaulo": (8, 0.020), "tokyo_tist": (4, 0.050),
}
# Dossier ari_seeds (single 42-43 pair) + sinks, for the old-vs-new SUMMARY.
DOSSIER_ARI = {"istanbul": 0.722, "bangkok": 0.998, "nyc_tist": 0.908,
               "saopaulo": 0.760, "tokyo_tist": 0.951}
DOSSIER_SINKS = {"istanbul": [0, 4], "bangkok": [2], "nyc_tist": [5],
                 "saopaulo": [5], "tokyo_tist": []}
DOSSIER_XSAGE = {  # point estimates from the dossier (per-user LT convention)
    "nyc_tist": {"R20": 0.0907, "LT20": 0.0974},
}

# Grids — EXTENDED (+1) to bracket the previous run's edge optimum
# (γ=0.7/depth=5/K=8 were at grid edges). ε extended UPWARD (not down): the
# boundary-band constraint below makes small-ε degenerate impossible, while the
# band [10%,30%] needs ε large enough to REACH ≥10% boundary on the denser
# perception → 0.07, 0.10 added as band-reachability insurance.
GAMMA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEPTH_GRID = [2, 3, 4, 5, 6]          # depth≥6 weakens the "shallow" rationale:
                                       # bracket-but-FLAG if the optimum lands ≥6
N_GRID = [3, 5, 7, 10]                 # 7 was interior already → unchanged
K_GRID = [4, 5, 6, 7, 8, 9]
EPS_GRID = [0.010, 0.020, 0.030, 0.050, 0.070, 0.100]

# Boundary-fraction band for ε / (K,ε) selection. ε selected on ARI ALONE is
# degenerate (ARI ↑ monotonically as ε ↓ → ε→min → boundary→0 → rough k-means
# DEGENERATES into plain k-means, boundary mechanism off). The band is a
# STRUCTURAL constraint (keeps the method in its "rough" regime, Lingras), NOT a
# fairness metric. Primary band = the dossier's [0.10, 0.30]; the other two are
# a robustness check that the selection does not hinge on the exact band.
BOUNDARY_BAND = (0.10, 0.30)
ROBUSTNESS_BANDS = [(0.05, 0.35), (0.10, 0.30), (0.15, 0.25)]

# Seeds: tiered honesty. S=3 broad scan, S=5 selection, S=10 finalist confirm.
SEEDS_A = (42, 43, 44)
SEEDS_B = (42, 43, 44, 45, 46)
SEEDS_CONFIRM = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)

# --------------------------------------------------------------------------- #
# Imports from the two code bases
# --------------------------------------------------------------------------- #
from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, compute_intent, compute_profile,
    estimate_macro_transition, find_attractors, fit_contribution_functions)
from pipeline.step02_models.xsage.l2_comprehension import (
    _assign, adjusted_rand_score, fit_rough_kmeans)

from xsage import data as D
from xsage.metrics import long_tail_groups, kl_divergence
from xsage.recommendation import fit_situation_biases_z, unified_combine_scores

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
TS = time.strftime("%Y%m%d_%H%M%S")
LOGFILE = LOGDIR / f"overnight_{TS}.log"
logger = logging.getLogger("overnight")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOGFILE); _fh.setFormatter(_fmt); logger.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); logger.addHandler(_sh)


def log(msg: str) -> None:
    logger.info(msg)


# --------------------------------------------------------------------------- #
# Checkpointing — append one row per atomic cell; resume by skipping done keys.
# Atomic unit is (stage, config, city): all its seeds are computed together,
# because cross-seed ARI is intrinsically a multi-seed quantity. This still
# gives full resumability at fine granularity.
# --------------------------------------------------------------------------- #
def load_done(path: Path, key_cols: list[str]) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty or any(c not in df.columns for c in key_cols):
        return set()
    return set(tuple(r) for r in df[key_cols].itertuples(index=False, name=None))


def append_row(path: Path, row: dict) -> None:
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


# --------------------------------------------------------------------------- #
# Perception descriptor v = [c̃ ‖ e] (uses the tested functions; exposes depth/n).
# Cached per (city, gamma, depth, n) within one process run.
# --------------------------------------------------------------------------- #
_DS_CACHE: dict = {}
_V_CACHE: dict = {}


def get_ds(city: str):
    if city not in _DS_CACHE:
        _DS_CACHE[city] = _load_city(city)
    return _DS_CACHE[city]


def build_v(city: str, gamma: float, depth: int, n: int,
            splits=("train", "val")) -> dict:
    """Return {split: v} plus shared objects, for the requested splits.
    train/val are enough for selection; 'test' is built only for the report."""
    key = (city, gamma, depth, n, splits)
    if key in _V_CACHE:
        return _V_CACHE[key]
    ds = get_ds(city)
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    contrib = fit_contribution_functions(ds["df_train"], m2i,
                                         attributes=DEFAULT_ATTRIBUTES,
                                         max_depth=depth, min_leaf=200)
    W = estimate_macro_transition(ds["df_train"], m2i,
                                  transit_macros=["Travel & Transport"],
                                  transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)

    def one(split):
        if split == "train":
            tgt, hist = ds["df_train"], ds["df_train"]
        elif split == "val":
            tgt, hist = ds["df_val"], ds["df_train"]
        else:
            tgt, hist = ds["df_test"], ds["history_for_test"]
        l0 = build_recent_window(tgt, hist, m2i, n=n)
        c = contrib.transform(tgt)
        m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=gamma)
        e = compute_intent(m, W, attractors, H=H_FIXED, beta=BETA_FIXED, mode="hard")
        return np.concatenate([c, e], axis=1).astype(np.float32)

    res = {"vs": {s: one(s) for s in splits}, "W": W, "attractors": attractors,
           "contrib": contrib}
    _V_CACHE[key] = res
    return res


def membership_from_assign(k_star, competing, is_boundary, K):
    B = len(k_star)
    mem = np.zeros((B, K), dtype=np.float32)
    core = ~is_boundary
    mem[core, k_star[core]] = 1.0
    bnd = np.where(is_boundary)[0]
    if bnd.size:
        T = competing[bnd].sum(axis=1).astype(np.float32)
        mem[bnd] = competing[bnd].astype(np.float32) / T[:, None]
    return mem


def ari_on_validation(city: str, gamma: float, depth: int, n: int,
                      K: int, eps: float, seeds) -> dict:
    """Fit rough k-means on TRAIN per seed; assign to VALIDATION; ARI across
    seeds on the validation core labels (selection on validation, not test)."""
    built = build_v(city, gamma, depth, n, splits=("train", "val"))
    vtr, vva = built["vs"]["train"], built["vs"]["val"]
    val_labels, iters, bfracs = [], [], []
    for s in seeds:
        r = fit_rough_kmeans(vtr, K=K, eps=eps, seed=s, max_iter=MAX_ITER)
        _, k_va, _, isb_va = _assign(vva, r.prototypes, eps)
        val_labels.append(k_va.astype(np.int32)); iters.append(r.n_iters)
        bfracs.append(float(isb_va.mean()))      # boundary fraction on VAL
    aris = [adjusted_rand_score(val_labels[i], val_labels[j])
            for i, j in combinations(range(len(seeds)), 2)]
    return {"ari_mean": float(np.mean(aris)), "ari_min": float(np.min(aris)),
            "bfrac": float(np.mean(bfracs)), "n_iters": int(np.mean(iters))}


# --------------------------------------------------------------------------- #
# STAGE A — perception scan (γ, depth, n) shared, K/ε at dossier values, S=3
# --------------------------------------------------------------------------- #
STAGE_A_CSV = OUT / "stage_a_ari.csv"


def stage_a() -> pd.DataFrame:
    key_cols = ["gamma", "depth", "n", "city"]
    done = load_done(STAGE_A_CSV, key_cols)
    configs = [(g, d, nn) for g in GAMMA_GRID for d in DEPTH_GRID for nn in N_GRID]
    total = len(configs) * len(CITIES)
    log(f"[Stage A] {len(configs)} perception configs × {len(CITIES)} cities "
        f"= {total} cells (S={len(SEEDS_A)}). {len(done)} already done.")
    idx = 0
    for (g, d, nn) in configs:
        for city in CITIES:
            idx += 1
            if (g, d, nn, city) in done:
                continue
            K, eps = DOSSIER_KEPS[city]
            t0 = time.time()
            try:
                res = ari_on_validation(city, g, d, nn, K, eps, SEEDS_A)
                row = {"gamma": g, "depth": d, "n": nn, "city": city,
                       "K": K, "eps": eps, **res, "secs": round(time.time() - t0, 1)}
                append_row(STAGE_A_CSV, row)
                log(f"[Stage A] cell {idx}/{total} — {city} γ={g} depth={d} n={nn} "
                    f"— ARI={res['ari_mean']:.3f} (min {res['ari_min']:.3f}) "
                    f"— {row['secs']:.0f}s")
            except Exception as e:
                log(f"[Stage A] ERROR cell {idx}/{total} {city} γ={g} d={d} n={nn}: "
                    f"{e}\n{traceback.format_exc()}")
        # free the per-config descriptor caches to bound memory
        _V_CACHE.clear()
    return pd.read_csv(STAGE_A_CSV)


def _rook_neighbours(value, grid):
    i = grid.index(value)
    out = []
    if i > 0: out.append(grid[i - 1])
    if i < len(grid) - 1: out.append(grid[i + 1])
    return out


def select_perception(df_a: pd.DataFrame) -> dict:
    """Plateau maximin over (γ, depth, n). Per-config value = MEAN ARI across
    cities; plateau score = min over the config's rook-neighbourhood."""
    # per-config aggregate across cities
    agg = (df_a.groupby(["gamma", "depth", "n"])
                .agg(ari_city_mean=("ari_mean", "mean"),
                     ari_city_min=("ari_mean", "min")).reset_index())
    val = {(r.gamma, r.depth, r.n): r.ari_city_mean for r in agg.itertuples()}

    def neigh(g, d, nn):
        cells = [(g, d, nn)]
        for gg in _rook_neighbours(g, GAMMA_GRID): cells.append((gg, d, nn))
        for dd in _rook_neighbours(d, DEPTH_GRID): cells.append((g, dd, nn))
        for n2 in _rook_neighbours(nn, N_GRID): cells.append((g, d, n2))
        return cells

    scored = []
    for (g, d, nn), v in val.items():
        cells = neigh(g, d, nn)
        vals = [val[c] for c in cells if c in val]
        P = float(np.min(vals))
        mean_n = float(np.mean(vals))
        width = int(sum(1 for x in vals if x >= max(vals) - 0.05))
        scored.append({"gamma": g, "depth": d, "n": nn, "P": P,
                       "neigh_mean": mean_n, "plateau_width": width,
                       "self_ari": v})
    sdf = pd.DataFrame(scored)
    # maximin, tie-breaks: neigh_mean ↓, width ↓, parsimony (depth+n ↑→worse), self
    sdf = sdf.sort_values(
        by=["P", "neigh_mean", "plateau_width", "depth", "n"],
        ascending=[False, False, False, True, True]).reset_index(drop=True)
    win = sdf.iloc[0]
    log(f"[Stage A] winner perception: γ={win.gamma} depth={int(win.depth)} "
        f"n={int(win.n)}  P(plateau)={win.P:.3f} neigh_mean={win.neigh_mean:.3f}")
    sdf.to_csv(OUT / "stage_a_plateau_scores.csv", index=False)
    return {"gamma": float(win.gamma), "depth": int(win.depth), "n": int(win.n),
            "P": float(win.P), "neigh_mean": float(win.neigh_mean)}


# --------------------------------------------------------------------------- #
# STAGE B — (K, ε) per city at the winning perception, S=5; then confirm the
# joint plateau by re-checking the 6 perception neighbours at the selected K/ε.
# --------------------------------------------------------------------------- #
STAGE_B_CSV = OUT / "stage_b_ari.csv"


def select_Keps(df_b: pd.DataFrame, city: str, band: tuple) -> dict:
    """Select (K*, ε*) for one city under a boundary-fraction band.

    Step 1: filter cells to boundary fraction ∈ band (structural constraint —
            keeps the method in its rough regime; ε on ARI alone is degenerate).
    Step 2: among IN-BAND cells, plateau-maximin on ARI over the in-band rook
            neighbourhood. Tie-breaks: neigh mean ↓, smaller K.
    Fallback: no in-band cell → the cell whose boundary fraction is closest to
              the band centre (dossier behaviour)."""
    sub = df_b[(df_b["city"] == city) & (df_b["kind"] == "Keps")]
    ari = {(int(r.K), float(r.eps)): float(r.ari_mean) for r in sub.itertuples()}
    bf = {(int(r.K), float(r.eps)): float(r.bfrac) for r in sub.itertuples()}
    lo, hi = band
    in_band = {c for c, b in bf.items() if lo <= b <= hi}
    if in_band:
        scored = []
        for (K, eps) in in_band:
            nb = [(K, eps)]
            for kk in _rook_neighbours(K, K_GRID): nb.append((kk, eps))
            for ee in _rook_neighbours(eps, EPS_GRID): nb.append((K, ee))
            vals = [ari[c] for c in nb if c in in_band]   # in-band neighbours only
            scored.append({"K": K, "eps": eps, "P": float(np.min(vals)),
                           "neigh_mean": float(np.mean(vals)), "ari": ari[(K, eps)],
                           "bfrac": bf[(K, eps)]})
        win = (pd.DataFrame(scored)
               .sort_values(by=["P", "neigh_mean", "K"],
                            ascending=[False, False, True]).iloc[0])
        return {"K": int(win.K), "eps": float(win.eps), "P": float(win.P),
                "ari": float(win.ari), "bfrac": float(win.bfrac), "in_band": True}
    # fallback: closest boundary fraction to the band centre
    centre = 0.5 * (lo + hi)
    (K, eps) = min(bf, key=lambda c: abs(bf[c] - centre))
    return {"K": K, "eps": eps, "P": ari[(K, eps)], "ari": ari[(K, eps)],
            "bfrac": bf[(K, eps)], "in_band": False}


def stage_b(perc: dict) -> pd.DataFrame:
    g, d, nn = perc["gamma"], perc["depth"], perc["n"]
    key_cols = ["city", "K", "eps", "kind"]
    done = load_done(STAGE_B_CSV, key_cols)
    # main K×ε sweep
    cells = [(city, K, eps) for city in CITIES for K in K_GRID for eps in EPS_GRID]
    log(f"[Stage B] {len(cells)} (city,K,ε) cells at perception "
        f"γ={g} depth={d} n={nn} (S={len(SEEDS_B)}). {len(done)} done.")
    idx = 0
    for (city, K, eps) in cells:
        idx += 1
        if (city, K, eps, "Keps") in done:
            continue
        t0 = time.time()
        try:
            res = ari_on_validation(city, g, d, nn, K, eps, SEEDS_B)
            append_row(STAGE_B_CSV, {"city": city, "K": K, "eps": eps,
                                     "kind": "Keps", "gamma": g, "depth": d,
                                     "n": nn, **res,
                                     "secs": round(time.time() - t0, 1)})
            log(f"[Stage B] cell {idx}/{len(cells)} — {city} K={K} ε={eps} "
                f"— ARI={res['ari_mean']:.3f} (min {res['ari_min']:.3f}) "
                f"— {time.time()-t0:.0f}s")
        except Exception as e:
            log(f"[Stage B] ERROR {city} K={K} ε={eps}: {e}\n{traceback.format_exc()}")
        _V_CACHE.clear()
    df_b = pd.read_csv(STAGE_B_CSV)

    # primary selection under the [10%,30%] band
    selected = {city: select_Keps(df_b, city, BOUNDARY_BAND) for city in CITIES}
    for city in CITIES:
        s = selected[city]
        log(f"[Stage B] {city}: K*={s['K']} ε*={s['eps']}  P={s['P']:.3f}  "
            f"boundary={s['bfrac']:.1%}  (in-band: {s['in_band']})")

    # band-robustness: re-select under alternative bands (FREE — re-filters the
    # already-computed cells, no new fits) and dump a band→(K*,ε*) table.
    rob_rows = []
    for band in ROBUSTNESS_BANDS:
        for city in CITIES:
            s = select_Keps(df_b, city, band)
            rob_rows.append({"band": f"{band[0]:.0%}-{band[1]:.0%}", "city": city,
                             "K": s["K"], "eps": s["eps"], "bfrac": round(s["bfrac"], 3),
                             "ari": round(s["ari"], 3), "in_band": s["in_band"]})
    pd.DataFrame(rob_rows).to_csv(OUT / "band_robustness.csv", index=False)
    log(f"[Stage B] band-robustness table → band_robustness.csv "
        f"({len(rob_rows)} rows)")

    # joint-plateau confirmation: re-check perception neighbours at (K*, ε*)
    done2 = load_done(STAGE_B_CSV, key_cols)
    perc_neigh = []
    for gg in _rook_neighbours(g, GAMMA_GRID): perc_neigh.append((gg, d, nn))
    for dd in _rook_neighbours(d, DEPTH_GRID): perc_neigh.append((g, dd, nn))
    for n2 in _rook_neighbours(nn, N_GRID): perc_neigh.append((g, d, n2))
    log(f"[Stage B] joint-plateau check: {len(perc_neigh)} perception neighbours "
        f"× {len(CITIES)} cities at selected (K*,ε*).")
    for city in CITIES:
        K, eps = selected[city]["K"], selected[city]["eps"]
        for (g2, d2, n2) in perc_neigh:
            tag = f"neigh_{g2}_{d2}_{n2}"
            if (city, K, eps, tag) in done2:
                continue
            t0 = time.time()
            try:
                res = ari_on_validation(city, g2, d2, n2, K, eps, SEEDS_B)
                append_row(STAGE_B_CSV, {"city": city, "K": K, "eps": eps,
                                         "kind": tag, "gamma": g2, "depth": d2,
                                         "n": n2, **res,
                                         "secs": round(time.time() - t0, 1)})
                log(f"[Stage B] joint-neigh {city} γ={g2} d={d2} n={n2} "
                    f"@K={K},ε={eps} — ARI={res['ari_mean']:.3f}")
            except Exception as e:
                log(f"[Stage B] ERROR joint-neigh {city}: {e}")
            _V_CACHE.clear()
    return selected


# --------------------------------------------------------------------------- #
# Finalist confirmation (S=10) on the selected cells, per city.
# --------------------------------------------------------------------------- #
CONFIRM_CSV = OUT / "finalist_confirm_ari.csv"


def confirm(perc: dict, selected: dict) -> dict:
    g, d, nn = perc["gamma"], perc["depth"], perc["n"]
    done = load_done(CONFIRM_CSV, ["city"])
    out = {}
    for city in CITIES:
        if (city,) in done:
            prev = pd.read_csv(CONFIRM_CSV)
            r = prev[prev.city == city].iloc[0]
            out[city] = {"ari_mean": float(r.ari_mean), "ari_min": float(r.ari_min)}
            continue
        K, eps = selected[city]["K"], selected[city]["eps"]
        try:
            res = ari_on_validation(city, g, d, nn, K, eps, SEEDS_CONFIRM)
            append_row(CONFIRM_CSV, {"city": city, "K": K, "eps": eps,
                                     "gamma": g, "depth": d, "n": nn, **res})
            out[city] = res
            log(f"[Confirm] {city} K={K} ε={eps} S=10 — ARI={res['ari_mean']:.3f} "
                f"(min {res['ari_min']:.3f})")
        except Exception as e:
            log(f"[Confirm] ERROR {city}: {e}")
        _V_CACHE.clear()
    return out


# --------------------------------------------------------------------------- #
# REPORT on TEST — situations from selected params; per-request-coherent sinks;
# all metrics under all conventions, for blind / context / X-SAGE.
# --------------------------------------------------------------------------- #
def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    nnz = x.size
    if nnz == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, nnz + 1)
    return float((2.0 * (idx * x).sum()) / (nnz * x.sum()) - (nnz + 1.0) / nnz)


def per_user_mean(per_req, u):
    uniq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uniq)); c = np.zeros(len(uniq))
    np.add.at(s, inv, per_req.astype(np.float64)); np.add.at(c, inv, 1)
    return float((s / c).mean())


def build_prep_for_report(city: str, perc: dict, K: int, eps: float, seed=42):
    """Mirror xsage.pipeline.prepare but with FRESH situations from the selected
    params (instead of the cached fit.npz). Returns the prep dict + extras."""
    g, d, nn = perc["gamma"], perc["depth"], perc["n"]
    ds = get_ds(city)
    n_items = ds["n_items"]; n_macros = ds["n_macros"]; m2i = ds["macro_to_idx"]
    built = build_v(city, g, d, nn, splits=("train", "test"))
    vtr, vte = built["vs"]["train"], built["vs"]["test"]

    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=MAX_ITER)
    z_train = fit.core_label.astype(np.int64)
    d_te, k_te, comp_te, isb_te = _assign(vte, fit.prototypes, eps)
    membership_test = membership_from_assign(k_te, comp_te, isb_te, K)
    z_test = k_te.astype(np.int32)

    cat_macro_train = (ds["df_train"]["cat_macro"].map(m2i)
                       .values.astype(np.int64))
    b_z = fit_situation_biases_z(z_train, cat_macro_train, K, n_macros,
                                 alpha=ALPHA_BIAS)
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]],
                       ignore_index=True)
    item_cat_macro = (df_all.groupby("i_idx")["cat_macro"].first().map(m2i)
                      .reindex(np.arange(n_items), fill_value=0)
                      .values.astype(np.int64))
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    transition_size = np.ones(len(z_test), dtype=np.float32)
    transition_size[isb_te] = 2.0          # match clean-pipeline γ convention
    gamma_req = np.where(isb_te, 1.0 / transition_size, 1.0).astype(np.float32)

    sb, sf = D.load_backbone_scores(city)
    excl = D.load_excluded_mask(city, n_items)
    prep = {"city": city, "ds": ds, "z_test": z_test, "isb_test": isb_te,
            "membership_test": membership_test, "K_sit": K, "b_z": b_z,
            "item_cat_macro": item_cat_macro, "G1_mask": G1_mask,
            "gamma": gamma_req, "scores_blind_full": sb, "scores_full_mmap": sf,
            "excluded": excl}
    return prep, fit, (d_te, k_te, comp_te, isb_te)


def blind_topk_test(prep):
    """Backbone (B_blind) top-K per test request, history-masked."""
    ds = prep["ds"]; df_test = ds["df_test"]; n_items = ds["n_items"]
    n_test = len(df_test); u_test = df_test["u_idx"].values.astype(np.int64)
    sb = prep["scores_blind_full"]; excl = prep["excluded"]
    top = np.zeros((n_test, K_TOP), dtype=np.int32)
    for bs in range(0, n_test, BATCH):
        be = min(n_test, bs + BATCH); u_b = u_test[bs:be]
        S = sb[u_b].astype(np.float32, copy=True)
        for j in range(be - bs):
            uu = int(u_b[j])
            cols = excl.indices[excl.indptr[uu]:excl.indptr[uu + 1]]
            if len(cols): S[j, cols] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        top[bs:be] = part
    return top, u_test


def stage_b_stats_per_request(prep, blind_top, u_test):
    """Per-situation KL_k, LT_k, available_LT_k — ALL per-request coherent.
    LT_k: long-tail share pooled over the requests' top-K (per-request).
    available_LT_k: long-tail share among each REQUEST user's not-yet-seen items,
                    averaged over REQUESTS (was per-user in the old code)."""
    ds = prep["ds"]; n_items = ds["n_items"]; G1 = prep["G1_mask"]
    z = prep["z_test"]; excl = prep["excluded"]
    K = prep["K_sit"]
    glob = blind_top.flatten()
    global_dist = np.bincount(glob, minlength=n_items).astype(np.float64)
    global_dist /= max(global_dist.sum(), 1.0)

    # per-user available LT (computed once per distinct user, then averaged
    # over REQUESTS via the request→user map → per-request coherent)
    avail_cache: dict[int, float] = {}

    def avail(u):
        if u not in avail_cache:
            seen = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            allowed = np.ones(n_items, dtype=bool); allowed[seen] = False
            avail_cache[u] = float(G1[allowed].mean()) if allowed.any() else 0.0
        return avail_cache[u]

    rows = []
    for k in range(K):
        mask = z == k
        n_req = int(mask.sum())
        if n_req == 0:
            rows.append({"situation": k, "n_requests": 0, "LT": None,
                         "KL": None, "available_LT": None})
            continue
        items = blind_top[mask].flatten()
        dist = np.bincount(items, minlength=n_items).astype(np.float64)
        dist /= max(dist.sum(), 1.0)
        lt = float(G1[items].mean())                         # per-request
        kl = kl_divergence(dist, global_dist)
        avail_req = float(np.mean([avail(int(u)) for u in u_test[mask]]))  # per-request
        rows.append({"situation": k, "n_requests": n_req, "LT": lt, "KL": kl,
                     "available_LT": avail_req})
    return rows


def identify_sinks_per_request(rows) -> list[int]:
    kls = np.array([r["KL"] for r in rows if r["KL"] is not None], dtype=np.float64)
    gmean = float(kls.mean()) if len(kls) else 0.0
    sinks = []
    for r in rows:
        if r["KL"] is None or gmean <= 0:
            continue
        if (r["KL"] >= SINK_KL_MULT * gmean and
                abs(r["LT"] - r["available_LT"]) >= SINK_LT_GAP):
            sinks.append(int(r["situation"]))
    return sorted(sinks), gmean


def report_metrics(perc: dict, selected: dict) -> None:
    from xsage import pipeline
    metric_rows, sink_rows = [], []
    for city in CITIES:
        K, eps = selected[city]["K"], selected[city]["eps"]
        log(f"[Report] {city} — situations from γ={perc['gamma']} "
            f"depth={perc['depth']} n={perc['n']}, K*={K} ε*={eps}")
        try:
            prep, fit, _ = build_prep_for_report(city, perc, K, eps)
            blind_top, u_test = blind_topk_test(prep)
            srows = stage_b_stats_per_request(prep, blind_top, u_test)
            sinks, gmean = identify_sinks_per_request(srows)
            for r in srows:
                sink_rows.append({"city": city, **r,
                                  "global_KL_mean": gmean,
                                  "is_sink": int(r["situation"] in sinks)})
            # score all three configs with the per-request-coherent sinks
            out = pipeline.score_request_metrics(prep, sinks, kappa=1.0, lam=1.0)
            u = out["__meta__"]["u_test"]
            touch = out["__meta__"]["touch_share"]
            G1 = prep["G1_mask"]; n_G1 = int(G1.sum())
            for cfg in ("blind", "full", "xsage"):
                topk = out[cfg]["topk"]; lt = out[cfg]["lt"]
                exposure = np.bincount(topk.flatten(),
                                       minlength=prep["ds"]["n_items"]).astype(float)
                cov = float(np.isin(np.where(G1)[0],
                                    np.unique(topk)).mean()) if n_G1 else 0.0
                # per-situation KL of THIS config's lists
                gdist = exposure / max(exposure.sum(), 1.0)
                kls = []
                for k in range(prep["K_sit"]):
                    m = prep["z_test"] == k
                    if not m.any(): continue
                    dd = np.bincount(topk[m].flatten(),
                                     minlength=prep["ds"]["n_items"]).astype(float)
                    dd /= max(dd.sum(), 1.0)
                    kls.append(kl_divergence(dd, gdist))
                metric_rows.append({
                    "city": city, "config": cfg,
                    "R20": per_user_mean(out[cfg]["hit"], u),
                    "N20": per_user_mean(out[cfg]["ndcg"], u),
                    "APL_per_request": float(lt.mean()),
                    "APL_per_user": per_user_mean(lt, u),
                    "LT_coverage": cov,
                    "KL_situation_mean": float(np.mean(kls)) if kls else 0.0,
                    "KL_situation_max": float(np.max(kls)) if kls else 0.0,
                    "Gini_exposure": gini(exposure),
                    "n_sinks": len(sinks), "touch_share": touch,
                })
            log(f"[Report] {city} sinks={sinks} touch={touch:.1%}")
        except Exception as e:
            log(f"[Report] ERROR {city}: {e}\n{traceback.format_exc()}")
        _V_CACHE.clear()
    pd.DataFrame(metric_rows).to_csv(OUT / "report_metrics.csv", index=False)
    pd.DataFrame(sink_rows).to_csv(OUT / "sinks_new.csv", index=False)
    log(f"[Report] wrote report_metrics.csv ({len(metric_rows)} rows) and "
        f"sinks_new.csv ({len(sink_rows)} rows)")


# --------------------------------------------------------------------------- #
# SUMMARY.md — auto-generated old-vs-new digest.
# --------------------------------------------------------------------------- #
def write_summary(perc, selected, confirm_ari):
    lines = ["# Overnight selection — SUMMARY", "",
             f"Run timestamp: {TS}", "",
             "## Parametri selezionati (plateau-aware, su validation)", "",
             f"- Perception SHARED: **γ={perc['gamma']}, depth={perc['depth']}, "
             f"n={perc['n']}** (H={H_FIXED}, β={BETA_FIXED} fissi)",
             f"- Plateau score P(perception) = {perc['P']:.3f} "
             f"(neigh mean {perc['neigh_mean']:.3f})", "",
             "## (K*, ε*) per città + boundary fraction + ARI vecchio vs nuovo", "",
             f"Banda boundary primaria: [{BOUNDARY_BAND[0]:.0%}, {BOUNDARY_BAND[1]:.0%}]",
             "",
             "| città | K vecchio→nuovo | ε vecchio→nuovo | boundary frac (in banda?) "
             "| ARI dossier | ARI nuovo (S=10) |",
             "|---|---|---|---|---|---|"]
    for c in CITIES:
        oldK, oldE = DOSSIER_KEPS[c]
        nK, nE = selected[c]["K"], selected[c]["eps"]
        bfr = selected[c]["bfrac"]; ib = "✅" if selected[c]["in_band"] else "⚠️ FALLBACK"
        an = confirm_ari.get(c, {}).get("ari_mean", float("nan"))
        lines.append(f"| {c} | {oldK}→{nK} | {oldE}→{nE} | {bfr:.1%} {ib} | "
                     f"{DOSSIER_ARI[c]:.3f} | {an:.3f} |")

    # band-robustness table
    lines += ["", "## Robustezza alla banda (band → (K*,ε*) per città)", "",
              "Se i (K*,ε*) NON cambiano tra le bande, la scelta è robusta "
              "(come il plateau di λ).", ""]
    try:
        rdf = pd.read_csv(OUT / "band_robustness.csv")
        bands = list(rdf["band"].unique())
        lines.append("| città | " + " | ".join(bands) + " |")
        lines.append("|---|" + "|".join(["---"] * len(bands)) + "|")
        for c in CITIES:
            cells = []
            for b in bands:
                row = rdf[(rdf.city == c) & (rdf.band == b)]
                if len(row):
                    r = row.iloc[0]
                    flag = "" if r.in_band else "*"
                    cells.append(f"K={int(r.K)},ε={r.eps}{flag}")
                else:
                    cells.append("—")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("(* = fallback fuori banda)")
    except Exception as e:
        lines.append(f"(band_robustness non disponibile: {e})")

    # automatic checks for the morning
    edge = []
    if perc["gamma"] == max(GAMMA_GRID): edge.append(f"γ={perc['gamma']} (bordo max)")
    if perc["depth"] == max(DEPTH_GRID): edge.append(f"depth={perc['depth']} (bordo max)")
    if perc["depth"] >= 6: edge.append(f"depth={perc['depth']} ≥6 (tensione interpretabilità — DISCUTERE)")
    for c in CITIES:
        if selected[c]["K"] in (min(K_GRID), max(K_GRID)):
            edge.append(f"{c}: K={selected[c]['K']} (bordo griglia)")
        if not selected[c]["in_band"]:
            edge.append(f"{c}: boundary FUORI banda (fallback)")
    lines += ["", "## ⚠️ Controlli automatici", ""]
    if edge:
        lines.append("Da verificare (ottimo al bordo / fuori banda / depth alta):")
        lines += [f"- {e}" for e in edge]
    else:
        lines.append("✅ Nessun parametro al bordo griglia; tutte le boundary "
                     "fraction in banda. Ottimo bracketato.")
    # sinks old vs new
    lines += ["", "## Sink + touch share (nuovi parametri, regola per-request)", ""]
    try:
        mdf = pd.read_csv(OUT / "report_metrics.csv")
        sdf = pd.read_csv(OUT / "sinks_new.csv")
        lines += ["| città | sink vecchi | sink nuovi | touch share nuovo |",
                  "|---|---|---|---|"]
        for c in CITIES:
            new_sinks = sorted(sdf[(sdf.city == c) & (sdf.is_sink == 1)]
                               ["situation"].astype(int).tolist())
            ts = mdf[(mdf.city == c) & (mdf.config == "xsage")]["touch_share"]
            tsv = float(ts.iloc[0]) if len(ts) else float("nan")
            lines.append(f"| {c} | {DOSSIER_SINKS[c]} | {new_sinks} | {tsv:.1%} |")
        # NYC dossier comparison
        lines += ["", "## Confronto col dossier (NYC X-SAGE, dove disponibile)", ""]
        nyc = mdf[(mdf.city == "nyc_tist") & (mdf.config == "xsage")]
        if len(nyc):
            r = nyc.iloc[0]
            lines += [f"- NYC X-SAGE R@20: dossier {DOSSIER_XSAGE['nyc_tist']['R20']} "
                      f"→ nuovo {r.R20:.4f}",
                      f"- NYC X-SAGE APL(per-user): dossier "
                      f"{DOSSIER_XSAGE['nyc_tist']['LT20']} → nuovo {r.APL_per_user:.4f} "
                      f"(per-request {r.APL_per_request:.4f})"]
    except Exception as e:
        lines.append(f"(report metrics non disponibili: {e})")
    lines += ["", "## File prodotti", "",
              "- stage_a_ari.csv, stage_a_plateau_scores.csv",
              "- stage_b_ari.csv (con colonna bfrac)", "- band_robustness.csv",
              "- finalist_confirm_ari.csv",
              "- selected_params.json", "- report_metrics.csv (TEST, 3×5)",
              "- sinks_new.csv", f"- logs/overnight_{TS}.log", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines))
    log("[Summary] wrote SUMMARY.md")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _apply_smoke():
    """Shrink grids/seeds/cities to validate the WHOLE flow end-to-end in a few
    minutes, exercising the exact same code paths. Writes to a separate dir."""
    global CITIES, GAMMA_GRID, DEPTH_GRID, N_GRID, K_GRID, EPS_GRID
    global SEEDS_A, SEEDS_B, SEEDS_CONFIRM, OUT
    global STAGE_A_CSV, STAGE_B_CSV, CONFIRM_CSV
    CITIES = ["nyc_tist", "saopaulo"]
    GAMMA_GRID = [0.6, 0.7]; DEPTH_GRID = [3]; N_GRID = [5]
    K_GRID = [5, 6]; EPS_GRID = [0.020, 0.050]   # ≥2 K and ≥2 ε → exercise band
    SEEDS_A = (42, 43); SEEDS_B = (42, 43); SEEDS_CONFIRM = (42, 43)
    OUT = CLEAN / "outputs_results" / "overnight_smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    STAGE_A_CSV = OUT / "stage_a_ari.csv"
    STAGE_B_CSV = OUT / "stage_b_ari.csv"
    CONFIRM_CSV = OUT / "finalist_confirm_ari.csv"
    log("*** SMOKE MODE: 1 perception config × 2 cities × S=2, minimal K/ε ***")


def _guard_schema():
    """If a previous run's CSVs predate the boundary-band schema (no 'bfrac'
    column), archive them so the new run starts clean instead of crashing on
    resume. Idempotent: no-op once the current-schema files are present."""
    if not STAGE_B_CSV.exists():
        return
    try:
        cols = list(pd.read_csv(STAGE_B_CSV, nrows=1).columns)
    except Exception:
        cols = []
    if "bfrac" in cols:
        return
    arch = OUT / f"superseded_{TS}"
    arch.mkdir(parents=True, exist_ok=True)
    for name in ["stage_a_ari.csv", "stage_b_ari.csv", "finalist_confirm_ari.csv",
                 "stage_a_plateau_scores.csv", "band_robustness.csv",
                 "selected_params.json", "report_metrics.csv", "sinks_new.csv",
                 "SUMMARY.md"]:
        p = OUT / name
        if p.exists():
            p.rename(arch / name)
    log(f"[guard] previous-run outputs (pre-band schema) archived → {arch}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="minimal grid end-to-end validation (few minutes)")
    args = ap.parse_args()

    t0 = time.time()
    log("=" * 70)
    log("OVERNIGHT JOINT SELECTION — start")
    log(f"log file: {LOGFILE}")
    if args.smoke:
        _apply_smoke()
    else:
        _guard_schema()
    log("=" * 70)

    df_a = stage_a()
    perc = select_perception(df_a)

    selected = stage_b(perc)
    confirm_ari = confirm(perc, selected)

    sel = {"perception_shared": {"gamma": perc["gamma"], "depth": perc["depth"],
                                 "n": perc["n"], "H": H_FIXED, "beta": BETA_FIXED},
           "plateau_score": perc["P"],
           "boundary_band": list(BOUNDARY_BAND),
           "per_city": {c: {"K": selected[c]["K"], "eps": selected[c]["eps"],
                            "boundary_fraction": round(selected[c]["bfrac"], 4),
                            "in_band": selected[c]["in_band"],
                            "ari_plateau": selected[c]["P"],
                            "ari_confirm_S10": confirm_ari.get(c, {}).get("ari_mean")}
                        for c in CITIES}}
    (OUT / "selected_params.json").write_text(json.dumps(sel, indent=2))
    log(f"[Select] wrote selected_params.json")

    report_metrics(perc, selected)
    write_summary(perc, selected, confirm_ari)

    log("=" * 70)
    log(f"OVERNIGHT JOINT SELECTION — done in {(time.time()-t0)/3600:.2f} h")
    log(f"Open first:  {OUT/'SUMMARY.md'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
