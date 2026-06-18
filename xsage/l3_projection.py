"""L3 — projection (cleaned).

Estimates the situation transition matrix T_{kk'} = P̂(s_{t+1}=k' | s_t=k)
from per-user core-label sequences (eq.17). Boundary disambiguation (eq.18)
r̃_k ∝ r_k · T_{z_{prev}, k} is implemented as a thin helper consumed by the
combiner.

Change vs round4-multicity: ``boundary_disambiguate`` now ASSERTS that all
the rows it touches are boundary requests, so the "applied only to boundary"
gate is part of the function contract, not the caller's responsibility.
"""
from __future__ import annotations

import numpy as np


def estimate_transition(core_labels_per_user: list[np.ndarray],
                          K: int,
                          add_one_smoothing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((K, K), dtype=np.float64)
    for seq in core_labels_per_user:
        if len(seq) < 2: continue
        s = seq[:-1]; d = seq[1:]
        np.add.at(counts, (s, d), 1)
    raw = counts.copy()
    if add_one_smoothing:
        counts += 1.0
    T = counts / counts.sum(axis=1, keepdims=True)
    return T, raw


def predict_next_situation(T: np.ndarray, z_t: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmax(T[int(z)])) for z in z_t])


def time_only_prior(z_train: np.ndarray, hour_train: np.ndarray,
                     hour_test: np.ndarray) -> np.ndarray:
    Hs = 24
    K = int(z_train.max() + 1)
    counts = np.zeros((Hs, K), dtype=np.int64)
    np.add.at(counts, (hour_train, z_train), 1)
    most = counts.argmax(axis=1)
    return most[hour_test]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    f1s = []
    for k in range(K):
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def dynamic_fairness(T: np.ndarray, lt_per_situation: np.ndarray,
                       tau_max: int = 3) -> np.ndarray:
    K = T.shape[0]
    out = np.zeros((K, tau_max), dtype=np.float64)
    Tpow = np.eye(K, dtype=np.float64)
    for t in range(tau_max):
        Tpow = Tpow @ T
        out[:, t] = Tpow @ lt_per_situation
    return out


def boundary_disambiguate(r_membership: np.ndarray, T: np.ndarray,
                            z_prev: np.ndarray,
                            is_boundary: np.ndarray) -> np.ndarray:
    """Apply eq.18 ONLY to boundary requests: r̃_k = r_k · T_{z_prev, k},
    renormalised. Non-boundary rows are returned untouched.

    The is_boundary gate is part of the function contract (was external in
    round4-multicity).
    """
    assert r_membership.shape[0] == z_prev.shape[0] == is_boundary.shape[0], \
        "rows mismatch"
    out = r_membership.astype(np.float64).copy()
    apply = is_boundary & (z_prev >= 0)
    if apply.any():
        priors = T[z_prev[apply]]
        out[apply] = out[apply] * priors
        s = out[apply].sum(axis=1, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        out[apply] = out[apply] / s
    return out.astype(np.float32)
