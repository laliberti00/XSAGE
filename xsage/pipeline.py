"""High-level scoring pipeline for one city under the unified combiner.

Given:
  - parquet data (L0 output)
  - cached backbone scores (B_blind from FM, B_full from context-aware FM)
  - cached Stage-A fit (situations) and Stage-B per-situation stats

Produce per-request R@20, NDCG@20, LT@20 and per-item top-K for any
(λ, κ, kl_mult, lt_gap) configuration. Used by scripts/* and tests/*.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sps

from . import data as D
from .l1_perception import (estimate_macro_transition, find_attractors)
from .metrics import long_tail_groups
from .recommendation import (fit_situation_biases_z, unified_combine_scores)
from .sinks import (SituationStat, identify_sinks,
                       situation_stats_from_dataframe)


K_TOP = 20
SHORT_HEAD = 0.20
BATCH = 1024
DEFAULT_LAMBDA = 1.0
DEFAULT_KAPPA = 1.0


def prepare(city: str, data_root: Path | None = None,
                 alpha: float = 50.0) -> dict:
    """Load + build all the per-city tensors once, ready to feed the combiner
    at any (λ, κ, sink set)."""
    ds = D.load_city(city, data_root=data_root)
    fit = D.load_fit(city, data_root=data_root)
    z_train = fit["core_label_train"].astype(np.int64)
    z_test = fit["core_label_test"].astype(np.int32)
    isb_test = fit["is_boundary_test"].astype(bool)
    membership_test = fit["membership_test"].astype(np.float32)
    K_sit = membership_test.shape[1]
    macro_to_idx = ds["macro_to_idx"]; n_macros = ds["n_macros"]
    cat_macro_train = (ds["df_train"]["cat_macro"].map(macro_to_idx)
                              .values.astype(np.int64))
    b_z = fit_situation_biases_z(z_train, cat_macro_train, K_sit, n_macros,
                                       alpha=alpha)
    # Item → macro
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]],
                            ignore_index=True)
    item_cat_macro = (df_all.groupby("i_idx")["cat_macro"].first()
                            .map(macro_to_idx).reindex(np.arange(ds["n_items"]),
                                                            fill_value=0)
                            .values.astype(np.int64))
    # Long-tail mask (items)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(axis=0)).ravel()
    _, G1_mask = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    # γ per request
    transition_size = np.ones(len(z_test), dtype=np.float32)
    transition_size[isb_test] = 2.0
    gamma = np.where(isb_test, 1.0 / transition_size, 1.0).astype(np.float32)
    # Backbone scores
    sb, sf = D.load_backbone_scores(city, data_root=data_root)
    excl = D.load_excluded_mask(city, ds["n_items"], data_root=data_root)
    return {
        "city": city, "ds": ds, "fit": fit,
        "z_test": z_test, "isb_test": isb_test,
        "membership_test": membership_test, "K_sit": K_sit,
        "b_z": b_z, "item_cat_macro": item_cat_macro,
        "G1_mask": G1_mask, "gamma": gamma,
        "scores_blind_full": sb, "scores_full_mmap": sf,
        "excluded": excl,
    }


def resolve_sinks(city: str, prep: dict, data_root: Path | None = None,
                      kl_mult: float = 1.5, lt_gap: float = 0.05) -> list[int]:
    """Return the sink situation IDs for a given (kl_mult, lt_gap). Falls
    back to the cached verdict.json if Stage B per_situation.csv is missing.
    """
    try:
        df = D.load_stage_b_per_situation(city, data_root=data_root)
        stats = situation_stats_from_dataframe(df)
        return identify_sinks(stats, kl_mult=kl_mult, lt_gap=lt_gap)
    except FileNotFoundError:
        v = D.load_sinks_verdict(city, data_root=data_root)
        return [int(s["situation"]) for s in v.get("inequity_sinks", [])]


def score_request_metrics(prep: dict, sink_ids: list[int],
                              kappa: float = DEFAULT_KAPPA,
                              lam: float = DEFAULT_LAMBDA,
                              include_blind: bool = True,
                              include_full: bool = True) -> dict:
    """For every test request compute R@20, NDCG@20, LT@20 and the top-K
    items for the X-SAGE unified config (and optionally B_blind, B_full).

    Returns a dict of per-config arrays: keys = "blind", "full", "xsage";
    each has 'hit' (B,), 'ndcg' (B,), 'lt' (B,), 'topk' (B, K_top).
    """
    ds = prep["ds"]; df_test = ds["df_test"]; n_items = ds["n_items"]
    n_test = len(df_test)
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    z_test = prep["z_test"]; isb_test = prep["isb_test"]
    sb_full = prep["scores_blind_full"]; sf_mmap = prep["scores_full_mmap"]
    excl = prep["excluded"]
    G1_mask = prep["G1_mask"]

    # Selective gate m_sel = sink ∩ core (request-level)
    sink_mask = np.isin(z_test, sink_ids) if sink_ids else np.zeros(n_test, bool)
    core_sink = sink_mask & ~isb_test

    out = {}
    cfgs = ["xsage"]
    if include_blind: cfgs.insert(0, "blind")
    if include_full: cfgs.insert(1, "full")
    for c in cfgs:
        out[c] = {
            "hit": np.zeros(n_test, dtype=np.float32),
            "ndcg": np.zeros(n_test, dtype=np.float32),
            "lt": np.zeros(n_test, dtype=np.float32),
            "topk": np.zeros((n_test, K_TOP), dtype=np.int32),
        }

    for bs in range(0, n_test, BATCH):
        be = min(n_test, bs + BATCH)
        u_b = u_test[bs:be]; i_b = i_target[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        SCORES = {}
        if include_blind:
            SCORES["blind"] = sb.copy()
        if include_full:
            SCORES["full"] = np.array(sf_mmap[bs:be], dtype=np.float32, copy=True)
        # X-SAGE — unified combiner
        SCORES["xsage"] = unified_combine_scores(
            scores_B=sb,
            membership_per_request=prep["membership_test"][bs:be],
            b_z=prep["b_z"],
            b_LT_mask=G1_mask,
            sink_core_mask=core_sink[bs:be],
            item_cat_macro=prep["item_cat_macro"],
            kappa=kappa, lam=lam,
            gamma_per_request=prep["gamma"][bs:be],
        )

        for name, S in SCORES.items():
            for j in range(be - bs):
                uu = int(u_b[j])
                cols = excl.indices[excl.indptr[uu]:excl.indptr[uu + 1]]
                if len(cols):
                    S[j, cols] = -np.inf
            s_tgt = S[np.arange(be - bs), i_b]
            ranks = (S > s_tgt[:, None]).sum(axis=1) + 1
            out[name]["hit"][bs:be] = (ranks <= K_TOP).astype(np.float32)
            out[name]["ndcg"][bs:be] = np.where(
                ranks <= K_TOP,
                1.0 / np.log2(ranks.astype(np.float64) + 1),
                0.0).astype(np.float32)
            partition = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
            out[name]["topk"][bs:be] = partition
            out[name]["lt"][bs:be] = G1_mask[partition].mean(axis=1).astype(
                np.float32)

    out["__meta__"] = {
        "n_test": n_test, "sink_ids": sink_ids,
        "touch_share": float(core_sink.mean()),
        "u_test": u_test,
    }
    return out


def per_user_mean(per_req: np.ndarray, u_arr: np.ndarray) -> float:
    uniq, inv = np.unique(u_arr, return_inverse=True)
    n = len(uniq); s = np.zeros(n); c = np.zeros(n)
    np.add.at(s, inv, per_req.astype(np.float64))
    np.add.at(c, inv, 1)
    return float((s / c).mean())


def aggregate(out: dict) -> dict:
    """Aggregate metrics — per-user macro for ALL of R@20, NDCG@20, LT@20.

    Convention chosen to match the dossier base_table_union_SD (see brief
    targets: NYC X-SAGE R=0.0907 LT=0.0974). In the original
    union_consolidation.py LT@20 is also macro-averaged per user.
    """
    u = out["__meta__"]["u_test"]
    agg = {}
    for c, d in out.items():
        if c == "__meta__": continue
        agg[c] = {
            "R20": per_user_mean(d["hit"], u),
            "N20": per_user_mean(d["ndcg"], u),
            "LT20": per_user_mean(d["lt"], u),
        }
    agg["__meta__"] = out["__meta__"]
    return agg
