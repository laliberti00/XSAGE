"""Amazon CDs&Vinyl (McAuley 2023) → schema X-SAGE (NO geo, come ml-1m).
interazione = review; macro = genere (categories[1] del meta); contesto temporale (timestamp ms →
ora/giorno/weekend/mese); split temporale per-utente 80/10/10; k-core=20.
Output: data/processed/amazoncd/ df_{train,val,test}.parquet + URM_{train,val}.npz + _cornac_in.npz.
Uso:  python scripts/amazon/preprocess_amazon.py [kcore=20]
"""
import json, gzip, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps

RAW = Path("/Users/lucaaliberti/Downloads/xsage-clean/data/amazoncd")
OUT = Path("/Users/lucaaliberti/Downloads/xsage-clean/data/processed/amazoncd")
KCORE = int(sys.argv[1]) if len(sys.argv) > 1 else 10   # protocollo UNIFORME (come ml-1m)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g2 = {}
    for ln in gzip.open(RAW / "meta_CDs_and_Vinyl.jsonl.gz", "rt", encoding="utf-8"):
        m = json.loads(ln); c = m.get("categories") or []
        mac = c[1] if len(c) >= 2 else (c[0] if c else m.get("main_category"))
        if mac: g2[m["parent_asin"]] = mac
    print(f"[1] meta: {len(g2)} item con genere", flush=True)

    rows = []; n = 0
    for ln in open(RAW / "CDs_and_Vinyl.jsonl"):
        n += 1
        if n % 3000000 == 0: print(f"    ...{n//1000000}M review", flush=True)
        r = json.loads(ln); pa = r.get("parent_asin")
        if pa in g2:
            rows.append((r["user_id"], pa, r["timestamp"], g2[pa]))
    df = pd.DataFrame(rows, columns=["user", "item", "ts", "cat_macro"])
    print(f"[2] review con genere: {len(df)}  utenti={df.user.nunique()}  item={df.item.nunique()}", flush=True)

    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        m = df.user.isin(uc[uc >= KCORE].index) & df.item.isin(ic[ic >= KCORE].index)
        if m.all(): break
        df = df[m]
    print(f"[3] dopo k-core={KCORE}: {len(df)}  utenti={df.user.nunique()}  item={df.item.nunique()}", flush=True)

    uu = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.item.unique()))}
    df["u_idx"] = df.user.map(uu).astype(np.int64); df["i_idx"] = df.item.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {mm: k for k, mm in enumerate(macros)}

    df["time_local"] = pd.to_datetime(df.ts, unit="ms")
    df["c_hour"] = df.time_local.dt.hour.astype(np.int64)
    df["c_dow"] = df.time_local.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = df.time_local.dt.month.astype(np.int64)
    df = df.sort_values(["u_idx", "ts"], kind="stable").reset_index(drop=True)
    r = df.groupby("u_idx").cumcount(); nu = df.groupby("u_idx")["u_idx"].transform("size")
    frac = r / nu
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)

    cols = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
            "c_isweekend", "c_month", "intent_last_cat_idx"]
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][cols].reset_index(drop=True)
        d.to_parquet(OUT / f"df_{sp}.parquet"); print(f"[5] df_{sp}: {len(d)}", flush=True)

    def urm(s):
        d = df[df.split.isin(s)]
        return sps.csr_matrix((np.ones(len(d)), (d.u_idx, d.i_idx)), shape=(n_users, n_items))
    sps.save_npz(OUT / "URM_train.npz", urm(["train"])); sps.save_npz(OUT / "URM_val.npz", urm(["val"]))

    tr = df[df.split == "train"]
    test_users = np.array(sorted(df[df.split == "test"].u_idx.unique()), dtype=np.int64)
    np.savez(OUT / "_cornac_in.npz", train_u=tr.u_idx.values.astype(np.int64),
             train_i=tr.i_idx.values.astype(np.int64), n_users=np.int64(n_users),
             n_items=np.int64(n_items), test_users=test_users)
    print(f"\n[OK] amazoncd → {OUT}")
    print(f"  utenti={n_users} item={n_items} macro={len(macros)}")
    print(f"  review/utente media={df.groupby('u_idx').size().mean():.1f} min={df.groupby('u_idx').size().min()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
