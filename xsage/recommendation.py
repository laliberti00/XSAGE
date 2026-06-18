"""Recommendation — UNIFIED additive combiner with selective gating.

This is the formula that produced every X-SAGE number in the paper. In the
round4-multicity tree the formula lived only in experiment scripts; here it
lives inside the pipeline as a first-class function.

    ŝ(u, i) = s_B(u, i) + κ · m_sel(req) · γ_S(v) · Σ_k r_k · b^(k)_{c(i)}

with the LEARNED + LONG-TAIL preference

    b^(k)_c = b̃^(k)_c  +  λ · b^LT_c

where
    b̃^(k)_c = z-score (per situation) of the shrunk-log-odds bias (α is the
                Dirichlet pseudo-count, default 50.0)
    b^LT_c  = {0, 1} long-tail indicator of c
    m_sel   = {0, 1} request-level selective gate (1 on core ∩ sink, 0 else)
    γ_S(v)  = 1 if core, 1/|T(v)| if boundary
    κ, λ    = scalar dosages

Naming: in round4 the smoothing pseudo-count was called ``lam``; here it is
``alpha`` (α) so it does not collide with λ, the long-tail dosage.

Matched-OFF guarantee: κ = 0 ⇒ ŝ = s_B exactly (structural short-circuit).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bias estimation
# ---------------------------------------------------------------------------

def fit_situation_biases_z(z_train: np.ndarray, cat_macro_train: np.ndarray,
                              K: int, n_macros: int,
                              alpha: float = 50.0) -> np.ndarray:
    """Per-situation per-macro bias, standardised (z-scored) within situation.

    Shrunk-log-odds with Bayesian smoothing toward the global macro
    distribution, then z-scored so κ is interpretable as "additive nudge of
    size ≈ κ standard deviations".

    Args:
        z_train:          (B_train,) int — situation label per training row.
        cat_macro_train:  (B_train,) int — macro of the training row's target.
        K:                number of situations.
        n_macros:         vocabulary size.
        alpha:            Dirichlet pseudo-count (was ``lam`` in round4).

    Returns:
        (K, n_macros) z-scored bias.
    """
    z_train = z_train.astype(np.int64)
    cat_macro_train = cat_macro_train.astype(np.int64)
    p_global = np.bincount(cat_macro_train, minlength=n_macros).astype(np.float64)
    p_global /= p_global.sum()
    counts_k = np.zeros((K, n_macros), dtype=np.float64)
    np.add.at(counts_k, (z_train, cat_macro_train), 1)
    smoothed = counts_k + alpha * p_global[None, :]
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    b = np.log(smoothed) - np.log(p_global + 1e-12)
    # z-score within situation
    mu = b.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((b - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------------------
# Unified combiner — the formula of the Approach
# ---------------------------------------------------------------------------

def unified_combine_scores(scores_B: np.ndarray,
                                membership_per_request: np.ndarray,
                                b_z: np.ndarray,
                                b_LT_mask: np.ndarray,
                                sink_core_mask: np.ndarray,
                                item_cat_macro: np.ndarray,
                                kappa: float,
                                lam: float,
                                gamma_per_request: np.ndarray) -> np.ndarray:
    """ŝ(u, i) = s_B + κ · m_sel(req) · γ_S(v) · Σ_k r_k · (b̃ + λ · b_LT).

    Args:
        scores_B:               (B, I) backbone scores.
        membership_per_request: (B, K) — r_k, soft membership of req to sit k.
        b_z:                    (K, n_macros) — z-scored learned bias b̃^(k)_c.
        b_LT_mask:              (n_items,) bool/float — {0,1} long-tail indicator
                                          for items (NOT macros).
        sink_core_mask:         (B,) bool/float — m_sel = 1 on core ∩ sink, 0 else.
        item_cat_macro:         (n_items,) int — each item's macro index.
        kappa:                  scalar κ.
        lam:                    scalar λ (long-tail dosage).
        gamma_per_request:      (B,) — γ_S(v): 1 on core, 1/|T(v)| on boundary.

    Returns:
        (B, n_items) modified scores.

    Matched-OFF: κ = 0 ⇒ returns ``scores_B`` exactly (structural).
    """
    if kappa == 0.0:
        return scores_B.astype(np.float32)
    n_macros = b_z.shape[1]
    # Σ_k r_k · b̃^{(k)}_c → (B, n_macros)
    learned_per_macro = membership_per_request.astype(np.float32) @ b_z
    # Project per item: (B, n_items)
    learned_per_item = learned_per_macro[:, item_cat_macro]
    # b_LT is per-item {0,1}; broadcast as (1, n_items)
    bLT_per_item = b_LT_mask.astype(np.float32)[None, :]
    # Combined preference (B, n_items)
    pref = learned_per_item + lam * bLT_per_item
    # Multiply by κ, γ_S(v), m_sel — all (B, 1)
    m_sel = sink_core_mask.astype(np.float32)[:, None]
    gamma = gamma_per_request.astype(np.float32)[:, None]
    nudge = (kappa * m_sel * gamma) * pref
    return (scores_B + nudge).astype(np.float32)
