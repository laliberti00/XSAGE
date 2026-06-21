"""Lightweight per-city loader for the clean repo.

Reads:
    data_dir / <city> / {train,val,test}.parquet         (output of step01 L0)
    artefacts_dir / <city> / xsage / situations / fit.npz
    artefacts_dir / <city> / xsage / backbone / Bfull.scores.npy
    artefacts_dir / <city> / xsage / fairness / verdict.json
    artefacts_dir / <city> / cached_scores / FM.npy     (backbone B_blind)
    artefacts_dir / <city> / step01_preprocessing / excluded_mask.npz (CSR)

In the round4-multicity tree these live under:
    {repo}/data/processed/<city>/    and    {repo}/outputs/<city>/
Set ARTEFACTS_ROOT to that {repo} via env or argument; the loader will
resolve everything from there.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps


DEFAULT_DATA_ROOT = Path(os.environ.get("XSAGE_DATA_ROOT",
                                                "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis"))
DEFAULT_CITIES = ("istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist")

# Backbone scores live LOCALLY in the clean repo after the one-time
# checksum-verified import (see docs/00_data_backbone/). The loader reads them
# from here so the repo is standalone; it only falls back to DEFAULT_DATA_ROOT
# (the old repo) when a local copy is missing.
LOCAL_BACKBONE_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_city(city: str, data_root: Path | None = None) -> dict:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    proc = root / "data" / "processed" / city
    df_train = pd.read_parquet(proc / "df_train.parquet")
    df_val = pd.read_parquet(proc / "df_val.parquet")
    df_test = pd.read_parquet(proc / "df_test.parquet")

    macros = sorted(set(df_train["cat_macro"].unique()) |
                       set(df_val["cat_macro"].unique()) |
                       set(df_test["cat_macro"].unique()))
    macro_to_idx = {m: i for i, m in enumerate(macros)}
    idx_to_macro = {i: m for m, i in macro_to_idx.items()}
    n_macros = len(macros)

    # Item / user catalogues
    n_users = int(max(df_train["u_idx"].max(), df_val["u_idx"].max(),
                          df_test["u_idx"].max()) + 1)
    n_items = int(max(df_train["i_idx"].max(), df_val["i_idx"].max(),
                          df_test["i_idx"].max()) + 1)

    # URM cached by step01 (already binarised under the floor's protocol).
    # Reading the .npz is the source of truth for popularity + excluded_mask;
    # rebuilding from df would double-count any repeated (u, i) interactions.
    urm_train = sps.load_npz(proc / "URM_train.npz").tocsr()
    urm_val = sps.load_npz(proc / "URM_val.npz").tocsr()

    return {
        "city": city,
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "macro_to_idx": macro_to_idx, "idx_to_macro": idx_to_macro,
        "n_macros": n_macros, "n_users": n_users, "n_items": n_items,
        "urm_train": urm_train, "urm_val": urm_val,
    }


def load_fit(city: str, data_root: Path | None = None) -> dict:
    """Load the rough k-means fit + situation labels (Stage A output)."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    fit = np.load(root / "outputs" / city / "xsage" / "situations" / "fit.npz",
                    allow_pickle=True)
    return {k: np.asarray(fit[k]) for k in fit.keys()}


def load_backbone_scores(city: str,
                              data_root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (B_blind (n_users, n_items), B_full (n_test, n_items, mmap)).

    Reads the checksum-verified LOCAL copy under data/<city>/backbone/ to keep
    the repo standalone; falls back to the old repo only if a local file is
    absent.
    """
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT

    def _resolve(fn: str) -> Path:
        local = LOCAL_BACKBONE_ROOT / city / "backbone" / fn
        if local.exists():
            return local
        return root / "outputs" / city / "xsage" / "backbone" / fn

    blind = np.load(_resolve("FM.scores.npy"), mmap_mode="r")
    full = np.load(_resolve("Bfull.scores.npy"), mmap_mode="r")
    return blind, full


def load_excluded_mask(city: str, n_items: int,
                            data_root: Path | None = None) -> sps.csr_matrix:
    """(n_users, n_items) mask of items to exclude from ranking per user —
    URM_train ∪ URM_val (the floor's exclude_seen protocol)."""
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    p = root / "data" / "processed" / city
    mask = (sps.load_npz(p / "URM_train.npz")
              + sps.load_npz(p / "URM_val.npz")).tocsr()
    mask.data[:] = 1.0
    return mask


def load_sinks_verdict(city: str, data_root: Path | None = None) -> dict:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    p = root / "outputs" / city / "xsage" / "fairness" / "verdict.json"
    if not p.exists(): return {"inequity_sinks": []}
    return json.loads(p.read_text())


def load_stage_b_per_situation(city: str,
                                       data_root: Path | None = None) -> pd.DataFrame:
    root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
    p = root / "outputs" / city / "xsage" / "fairness" / "per_situation.csv"
    return pd.read_csv(p)
