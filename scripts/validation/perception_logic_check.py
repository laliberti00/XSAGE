"""Phase 02 (Perception) — Part 1: does the CLEAN-repo logic match the Approach?

Each check exercises the clean xsage.l1_perception functions and prints a
minimal numeric verification on NYC. Data prep (derived columns) reuses the old
repo's _load_city; the LOGIC under test is the clean module.
"""
import sys
from pathlib import Path

import numpy as np

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN))   # clean xsage first
sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from xsage.l1_perception import (
    DEFAULT_ATTRIBUTES, fit_contribution_functions, estimate_macro_transition,
    find_attractors, compute_profile, compute_intent)
from xsage.l0_sensing import build_recent_window

CITY = "nyc_tist"


def main():
    ds = _load_city(CITY)
    m2i = ds["macro_to_idx"]; i2m = ds["idx_to_macro"]; n_macros = ds["n_macros"]
    df = ds["df_train"]
    print(f"=== Perception logic check on {CITY} (K_mac={n_macros}) ===\n")

    # --- L1a.1: one shallow tree per attribute, declared depth ---
    contrib = fit_contribution_functions(df, m2i, attributes=DEFAULT_ATTRIBUTES,
                                          max_depth=3, min_leaf=200)
    depths = {a: contrib.trees[a].get_depth() for a in DEFAULT_ATTRIBUTES}
    print("[L1a.1] one tree per attribute, max_depth=3")
    print(f"    A = {len(contrib.trees)} attributes = {list(DEFAULT_ATTRIBUTES)}")
    print(f"    tree depths = {depths}  (all <= 3: {all(d <= 3 for d in depths.values())})\n")

    # --- L1a.2: theta = 1 - H(p_leaf)/log2(K_mac), manual check on one leaf ---
    a0 = DEFAULT_ATTRIBUTES[0]
    col = df[a0].values.astype(np.int32)
    y = df["cat_target"].values.astype(np.int64)
    leaves = contrib.trees[a0].apply(col.reshape(-1, 1))
    some_leaf = int(np.bincount(leaves).argmax())     # most populated leaf
    mask = leaves == some_leaf
    counts = np.bincount(y[mask], minlength=n_macros).astype(np.float64)
    p = counts / counts.sum()
    H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    theta_manual = 1.0 - H / np.log2(n_macros)
    theta_code = contrib.leaf_theta[a0][some_leaf]
    print("[L1a.2] theta = 1 - H(p_leaf)/log2(K_mac)")
    print(f"    attr={a0} leaf={some_leaf} n={int(counts.sum())}  "
          f"H={H:.4f}  theta_manual={theta_manual:.6f}  theta_code={theta_code:.6f}  "
          f"match={abs(theta_manual - theta_code) < 1e-6}\n")

    # --- L1a.3: context state is a (B, A) vector, no scalar aggregation ---
    c = contrib.transform(df)
    print("[L1a.3] context state shape (B, A), no scalar collapse")
    print(f"    c.shape = {c.shape}  (A = {len(DEFAULT_ATTRIBUTES)})  "
          f"in [0,1]: {c.min():.3f}..{c.max():.3f}\n")

    # --- L1b.4: W add-one smoothing, rows sum to 1 ---
    W = estimate_macro_transition(df, m2i, add_one_smoothing=True)
    rowsums = W.sum(axis=1)
    print("[L1b.4] W: add-one smoothing, row-stochastic")
    print(f"    W.shape={W.shape}  rowsum in {rowsums.min():.6f}..{rowsums.max():.6f}  "
          f"(=1: {np.allclose(rowsums, 1.0)})  all entries >0 (smoothed): {(W > 0).all()}\n")

    # --- L1b.5: attractors = {c: indeg >= mean indeg} ---
    indeg = W.sum(axis=0)
    A = find_attractors(W)
    names = [i2m[i] for i in np.where(A)[0]]
    print("[L1b.5] attractors via indegree >= mean")
    print(f"    mean indeg={indeg.mean():.4f}")
    print(f"    attractors ({A.sum()}/{n_macros}): {names}")
    print(f"    manual match: {np.array_equal(A, indeg >= indeg.mean())}\n")

    # --- L1b.6: recency profile m with gamma decay; intent e formula ---
    l0 = build_recent_window(df, df, m2i, n=5)
    m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=0.6)
    e = compute_intent(m, W, A, H=2, beta=0.7)
    # manual intent for one non-empty row
    K = W.shape[0]
    acc = np.zeros_like(W)
    Wk = np.eye(K)
    for k in range(1, 3):
        Wk = Wk @ W
        acc += (0.7 ** k) * Wk
    raw = m @ acc
    raw = raw * A[None, :]
    e_manual = raw / np.where(raw.sum(1, keepdims=True) > 0, raw.sum(1, keepdims=True), 1.0)
    print("[L1b.6] profile m (gamma decay) + intent e = norm((m Sigma beta^k W^k) ⊙ 1[A])")
    print(f"    m rows sum to 1: {np.allclose(m.sum(1), 1.0)}")
    print(f"    e supported only on attractors: {np.all(e[:, ~A] == 0)}")
    print(f"    e matches manual formula: {np.allclose(e, e_manual, atol=1e-5)}\n")

    # --- L1b.7: gamma and beta are distinct params on distinct objects ---
    m_g = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=0.3)
    e_b = compute_intent(m, W, A, H=2, beta=0.3)
    print("[L1b.7] gamma (recency, in compute_profile) != beta (graph reach, in compute_intent)")
    print(f"    changing gamma 0.6->0.3 changes m: {not np.allclose(m, m_g)}")
    print(f"    changing beta  0.7->0.3 changes e: {not np.allclose(e, e_b)}")
    print(f"    gamma leaves W/e-graph untouched; beta leaves m untouched (distinct objects)\n")

    # --- L1b.8: mode hard only; soft/all absent ---
    import inspect
    sig = inspect.signature(compute_intent)
    has_mode = "mode" in sig.parameters
    print("[L1b.8] intent mode = hard only (Approach Eq.5); soft_topr/all removed")
    print(f"    compute_intent signature params: {list(sig.parameters)}")
    print(f"    no 'mode' parameter (hard hard-wired): {not has_mode}\n")

    print("=== all numeric checks above should read match=True / correct ===")


if __name__ == "__main__":
    main()
