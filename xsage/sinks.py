"""Inequity-sink identification — explicit two-threshold rule.

Definition (extracted from round4-multicity orchestrator.run_stage_b, where
it was hard-coded). A situation k is an INEQUITY SINK iff

    KL_k ≥ kl_mult · KL_global_mean       (fairness lens distortion)
    |LT_k − LT_available_k| ≥ lt_gap      (excess long-tail vs structural baseline)

Defaults match the round4 hard-coded values (1.5, 0.05). Both are now
parameters so the sweep over (kl_mult, lt_gap) is a one-call away (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SituationStat:
    """Per-situation summary computed once from the backbone's top-K lists."""
    situation: int
    KL: float                       # KL(top-K macro distribution || global)
    LT: float                       # long-tail rate in top-K of this situation
    LT_available: float             # structural baseline: LT rate among items
                                    # the typical user in this situation could
                                    # still receive (not yet seen)


def identify_sinks(situation_stats: Sequence[SituationStat],
                    kl_mult: float = 1.5,
                    lt_gap: float = 0.05) -> list[int]:
    """Return situation IDs that satisfy BOTH the KL-multiplier rule and the
    long-tail-excess rule.

    Args:
        situation_stats: per-situation (KL, LT, LT_available) measured on the
                              backbone's test top-K (Stage B output).
        kl_mult:        multiplier on the global mean KL (default 1.5).
        lt_gap:         absolute gap |LT − LT_available| (default 0.05).

    Returns:
        sorted list of int situation IDs that are flagged sinks.
    """
    kls = np.array([s.KL for s in situation_stats], dtype=np.float64)
    global_KL_mean = float(kls.mean()) if len(kls) else 0.0
    sinks: list[int] = []
    for s in situation_stats:
        if global_KL_mean <= 0:
            kl_ok = False
        else:
            kl_ok = s.KL >= kl_mult * global_KL_mean
        lt_ok = abs(s.LT - s.LT_available) >= lt_gap
        if kl_ok and lt_ok:
            sinks.append(int(s.situation))
    return sorted(sinks)


def situation_stats_from_dataframe(df: pd.DataFrame) -> list[SituationStat]:
    """Read a Stage-B per_situation.csv-style dataframe (rows for split=="all"
    are taken) into a list of SituationStat.

    Expected columns: situation, split, KL, LT, available_LT.
    """
    sub = df[df["split"] == "all"].copy()
    stats = []
    for _, r in sub.iterrows():
        if r["KL"] is None or r["available_LT"] is None or pd.isna(r["KL"]):
            continue
        stats.append(SituationStat(
            situation=int(r["situation"]),
            KL=float(r["KL"]),
            LT=float(r["LT"]),
            LT_available=float(r["available_LT"]),
        ))
    return stats
