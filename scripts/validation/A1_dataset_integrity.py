"""A1 — Dataset integrity check (one row per city, PASS/FAIL).

Reads the parquet + cached URMs the clean repo actually uses. No fixes:
measures + verdict.

k-core convention: k-core=10 is enforced on the FULL interaction set
(train ∪ val ∪ test) BEFORE the per-user temporal 80/10/10 split, then a
cold-item drop removes a few val/test interactions whose item is absent from
train. So the correct k-core check is on the full union, NOT on train+val
(which by construction drops the per-user minimum below 10). The cold-drop can
legitimately lower the per-USER full-set minimum to 9 in a handful of users;
this is read from stats.json (cold_dropped) and treated as a documented note,
not a violation. The per-ITEM minimum stays at exactly 10.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from xsage import data as D


EXPECTED_NTEST = {
    "istanbul": 136220, "bangkok": 43556, "nyc_tist": 17702,
    "saopaulo": 23821, "tokyo_tist": 59561,
}
KCORE_EXPECTED = 10
DENSITY_RANGE = (1e-4, 1e-2)


def measure_one(city: str) -> dict:
    ds = D.load_city(city)
    df_train = ds["df_train"]; df_test = ds["df_test"]
    urm_train = ds["urm_train"]; urm_val = ds["urm_val"]
    n_items = ds["n_items"]
    n_users_train = int(df_train["u_idx"].nunique())
    n_users_test = int(df_test["u_idx"].nunique())
    n_int_train = int(len(df_train))
    n_int_test = int(len(df_test))
    n_test = len(df_test)
    density_train = float(urm_train.nnz) / (urm_train.shape[0] * urm_train.shape[1])

    # k-core on the FULL union (train ∪ val ∪ test) — the set k-core is
    # enforced on, pre-split. Binarise so we count distinct items per user.
    proc = D.DEFAULT_DATA_ROOT / "data" / "processed" / city
    urm_test = sps.load_npz(proc / "URM_test.npz").tocsr()
    full = (urm_train + urm_val + urm_test).tocsr()
    full.data[:] = 1.0
    uc = np.asarray(full.sum(axis=1)).ravel()
    ic = np.asarray(full.sum(axis=0)).ravel()
    nz_u = uc[uc > 0]; nz_i = ic[ic > 0]
    min_user_full = int(nz_u.min()) if len(nz_u) else 0
    min_item_full = int(nz_i.min()) if len(nz_i) else 0
    n_users_zero = int((uc == 0).sum())
    n_items_zero = int((ic == 0).sum())

    # k-core enforcement stage + cold-drop, straight from stats.json
    stats = json.loads((proc / "stats.json").read_text())
    cold = stats.get("cold_dropped", {})
    n_cold_dropped = int(cold.get("val_dropped", 0)) + int(cold.get("test_dropped", 0))

    K_mac = int(ds["n_macros"])

    return {
        "city": city,
        "n_users_train": n_users_train,
        "n_users_test": n_users_test,
        "n_items": n_items,
        "n_int_train": n_int_train,
        "n_int_test": n_int_test,
        "n_test": n_test,
        "URM_train_density": density_train,
        "kcore_user_min_full": min_user_full,
        "kcore_item_min_full": min_item_full,
        "n_cold_dropped": n_cold_dropped,
        "n_users_zero": n_users_zero,
        "n_items_zero": n_items_zero,
        "K_mac": K_mac,
    }


def verdict(row: dict) -> tuple[str, list[str], list[str]]:
    fails, notes = [], []
    city = row["city"]
    exp = EXPECTED_NTEST[city]
    # (a) n_test exact
    if row["n_test"] != exp:
        fails.append(f"n_test={row['n_test']} ≠ expected {exp}")
    # (b) k-core=10: item-min on full union must be exactly >=10; user-min<10
    #     is allowed only if explained by the documented cold-drop.
    if row["kcore_item_min_full"] < KCORE_EXPECTED:
        fails.append(f"item-min (full union) = {row['kcore_item_min_full']} "
                       f"< {KCORE_EXPECTED}")
    if row["kcore_user_min_full"] < KCORE_EXPECTED:
        if row["n_cold_dropped"] > 0:
            notes.append(f"user-min (full) = {row['kcore_user_min_full']} "
                           f"(cold-drop removed {row['n_cold_dropped']} interactions "
                           f"post-kcore; expected, not a violation)")
        else:
            fails.append(f"user-min (full) = {row['kcore_user_min_full']} "
                           f"< {KCORE_EXPECTED} with no cold-drop to explain it")
    # (c) anomalies
    if row["n_users_zero"] != 0:
        fails.append(f"{row['n_users_zero']} empty users")
    if row["n_items_zero"] != 0:
        fails.append(f"{row['n_items_zero']} empty items")
    if not (DENSITY_RANGE[0] <= row["URM_train_density"] <= DENSITY_RANGE[1]):
        fails.append(f"URM_train density {row['URM_train_density']:.2e} "
                       f"out of plausible range {DENSITY_RANGE}")
    if row["K_mac"] < 9:
        fails.append(f"K_mac={row['K_mac']} implausibly low")
    elif row["K_mac"] != 10:
        notes.append(f"K_mac={row['K_mac']} (city genuinely lacks some macro "
                       f"categories; coherent)")
    return ("PASS" if not fails else "FAIL"), fails, notes


def main():
    rows = []
    verdicts = []
    for city in EXPECTED_NTEST:
        r = measure_one(city)
        v, fails, notes = verdict(r)
        r["verdict"] = v
        r["fails"] = "; ".join(fails) if fails else ""
        r["notes"] = "; ".join(notes) if notes else ""
        rows.append(r)
        verdicts.append((city, v, fails, notes))

    df = pd.DataFrame(rows)
    OUT = ROOT / "outputs_results" / "validation"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "A1_dataset_integrity.csv", index=False)

    # Compact stdout
    print("\n=== A1 Dataset integrity ===\n")
    short = df[["city", "n_users_train", "n_items", "n_int_train",
                     "n_test", "URM_train_density",
                     "kcore_user_min_full", "kcore_item_min_full",
                     "n_cold_dropped", "n_users_zero", "n_items_zero",
                     "K_mac", "verdict"]]
    print(short.to_string(index=False))

    print("\n--- per-city verdicts ---")
    for c, v, f, n in verdicts:
        print(f"  {c:<14} {v}")
        for msg in f:
            print(f"       FAIL: {msg}")
        for msg in n:
            print(f"       note: {msg}")

    n_pass = sum(1 for _, v, _, _ in verdicts if v == "PASS")
    print(f"\nOverall: {n_pass}/{len(verdicts)} cities PASS")
    print(f"→ {OUT/'A1_dataset_integrity.csv'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
