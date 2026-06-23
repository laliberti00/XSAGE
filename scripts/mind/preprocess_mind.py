"""MIND-large → schema X-SAGE (step 01). Porta MIND al setup attuale (decisione utente):
interazione = CLICK (impressions label=1), k-core=10, split temporale per-utente 80/10/10.

Scostamenti documentati da Foursquare:
- NIENTE geohash (MIND non ha geolocalizzazione) → c̃ usa 5 attributi temporali+intento.
- categoria nativa = news.category (14 macro reali).
- utenti poco profondi (mediana 3 click) → k-core=10 tiene solo i ~105K utenti profondi.

Output in data/processed/mind/: df_{train,val,test}.parquet + URM_{train,val}.npz.
Colonne df: u_idx, i_idx, cat_macro, time_local, c_hour, c_dow, c_isweekend, c_month, intent_last_cat_idx.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

MINDDIR = Path("/Users/lucaaliberti/Downloads")
TRAIN_B = MINDDIR / "MINDlarge_train" / "behaviors.tsv"
DEV_B = MINDDIR / "MINDlarge_dev" / "behaviors.tsv"
NEWS = MINDDIR / "MINDlarge_train" / "news.tsv"
OUT = Path("/Users/lucaaliberti/Downloads/xsage-clean/data/processed/mind")
KCORE = 10


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. news -> category
    cat = {}
    for ln in open(NEWS, encoding="utf-8", errors="replace"):
        p = ln.split("\t")
        if len(p) > 2: cat[p[0]] = p[1]
    print(f"[1] news con categoria: {len(cat)}", flush=True)

    # 2. stream click (label=1)
    U, I, T = [], [], []
    for F in (TRAIN_B, DEV_B):
        for ln in open(F, encoding="utf-8", errors="replace"):
            p = ln.rstrip("\n").split("\t")
            if len(p) < 5: continue
            u, t = p[1], p[2]
            for it in p[4].split():
                if it.endswith("-1"):
                    nid = it[:-2]
                    if nid in cat:
                        U.append(u); I.append(nid); T.append(t)
    df = pd.DataFrame({"user": U, "item": I, "time_str": T})
    df["time_local"] = pd.to_datetime(df["time_str"], format="%m/%d/%Y %I:%M:%S %p")
    df["cat_macro"] = df["item"].map(cat)
    df = df.drop(columns="time_str")
    print(f"[2] click grezzi: {len(df)}  utenti={df.user.nunique()}  item={df.item.nunique()}", flush=True)

    # 3. k-core=10 iterativo
    while True:
        uc = df["user"].value_counts(); ic = df["item"].value_counts()
        keep_u = uc[uc >= KCORE].index; keep_i = ic[ic >= KCORE].index
        m = df["user"].isin(keep_u) & df["item"].isin(keep_i)
        if m.all(): break
        df = df[m]
    print(f"[3] dopo k-core={KCORE}: {len(df)} click  utenti={df.user.nunique()}  item={df.item.nunique()}", flush=True)

    # 4. remap indici contigui
    uu = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.item.unique()))}
    df["u_idx"] = df.user.map(uu).astype(np.int64)
    df["i_idx"] = df.item.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {m: k for k, m in enumerate(macros)}

    # 5. split temporale per-utente 80/10/10
    df = df.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    df["rank"] = df.groupby("u_idx").cumcount()
    df["n_u"] = df.groupby("u_idx")["u_idx"].transform("size")
    frac = df["rank"] / df["n_u"]
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))

    # 6. contesto + intento (intent_last_cat_idx = macro click precedente; sentinel = n_macros)
    df["c_hour"] = df.time_local.dt.hour.astype(np.int64)
    df["c_dow"] = df.time_local.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = df.time_local.dt.month.astype(np.int64)
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = (df.groupby("u_idx")["_cmi"].shift(1)
                                 .fillna(len(macros)).astype(np.int64))

    cols = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
            "c_isweekend", "c_month", "intent_last_cat_idx"]
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][cols].reset_index(drop=True)
        d.to_parquet(OUT / f"df_{sp}.parquet")
        print(f"[5] df_{sp}: {len(d)} righe", flush=True)

    # 7. URM train/val (binari, user×item)
    def urm(sp_set):
        d = df[df.split.isin(sp_set)]
        return sps.csr_matrix((np.ones(len(d)), (d.u_idx, d.i_idx)), shape=(n_users, n_items))
    sps.save_npz(OUT / "URM_train.npz", urm(["train"]))
    sps.save_npz(OUT / "URM_val.npz", urm(["val"]))

    print(f"\n[OK] MIND processato → {OUT}")
    print(f"  utenti={n_users}  item={n_items}  macro={len(macros)} ({macros})")
    print(f"  click/utente: media={df.groupby('u_idx').size().mean():.1f} "
          f"min={df.groupby('u_idx').size().min()}")
    print(f"  NB: niente prev_geohash5 → attributi c̃ = [c_hour,c_dow,c_isweekend,c_month,intent_last_cat_idx]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
