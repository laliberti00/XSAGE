"""Causal split invariant: no test row's user has a TRAINING row dated at or
after the test row's timestamp (strict per-user temporal 80/10/10 split).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xsage import data as D


def _check_one(city: str):
    ds = D.load_city(city)
    df_train = ds["df_train"]; df_test = ds["df_test"]
    # Per user: max train time < min test time
    train_max = df_train.groupby("u_idx")["time_local"].max().rename("train_max")
    test_min = df_test.groupby("u_idx")["time_local"].min().rename("test_min")
    joined = pd.concat([train_max, test_min], axis=1, join="inner")
    bad = joined[joined["train_max"] >= joined["test_min"]]
    return len(bad), len(joined)


def test_causal_split_5_cities():
    failures = []
    for city in D.DEFAULT_CITIES:
        n_bad, n_users = _check_one(city)
        print(f"  {city:<14} {n_bad}/{n_users} users with train≥test (must be 0)")
        if n_bad > 0:
            failures.append((city, n_bad, n_users))
    assert not failures, f"causal split violated: {failures}"


if __name__ == "__main__":
    test_causal_split_5_cities()
    print("causal split OK")
