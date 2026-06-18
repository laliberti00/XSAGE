"""Additive fidelity: combiner output − backbone == κ · m_sel · γ · Σ_k r_k · (b̃ + λ·b_LT),
exactly within float32 epsilon on touched rows; zero on untouched rows.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xsage.recommendation import unified_combine_scores


def test_additive_fidelity_synthetic():
    rng = np.random.default_rng(1)
    B, I, K_sit, M = 64, 200, 5, 8
    sB = rng.normal(size=(B, I)).astype(np.float32)
    membership = np.zeros((B, K_sit), dtype=np.float32)
    # half the rows: one-hot membership; the other half: split between 2 sit
    for b in range(B):
        if b % 2 == 0:
            membership[b, rng.integers(0, K_sit)] = 1.0
        else:
            k1, k2 = rng.choice(K_sit, 2, replace=False)
            membership[b, k1] = 0.5; membership[b, k2] = 0.5
    b_z = rng.normal(size=(K_sit, M)).astype(np.float32)
    item_cat_macro = rng.integers(0, M, I)
    bLT = rng.integers(0, 2, I).astype(np.float32)
    sink_core = (rng.uniform(size=B) > 0.5).astype(np.float32)
    gamma = np.where(sink_core > 0, 1.0, 0.5).astype(np.float32)
    kappa, lam = 1.3, 0.8

    s = unified_combine_scores(sB, membership, b_z, bLT, sink_core,
                                     item_cat_macro, kappa=kappa, lam=lam,
                                     gamma_per_request=gamma)
    delta = s - sB
    # Expected nudge for each (B, I)
    learned_per_macro = membership @ b_z              # (B, M)
    learned_per_item = learned_per_macro[:, item_cat_macro]
    pref = learned_per_item + lam * bLT[None, :]
    expected = (kappa * sink_core[:, None] * gamma[:, None] * pref).astype(np.float32)

    err = np.max(np.abs(delta - expected))
    print(f"  max |delta - expected| = {err:.2e}")
    assert err < 1e-5, f"additive fidelity violated, max err {err}"

    # On non-touched rows (sink_core==0) delta must be zero exactly
    mask0 = sink_core == 0
    if mask0.any():
        assert np.all(delta[mask0] == 0.0), "untouched rows are not zero"


if __name__ == "__main__":
    test_additive_fidelity_synthetic()
    print("OK")
