"""Porta il RAW Foursquare TSMC2014 (NYC o TKY) allo schema X-SAGE, protocollo UNIFORME (k-core10),
con GEO. Macro = categoria ROOT (10) mappata dalla tassonomia legacy (fine→root). Stessa pipeline
di yelp/ml-1m. Output: data/processed/<tsmc_nyc|tsmc_tky>/ df_* + URM_* + _cornac_in.npz.

Uso:  python scripts/foursquare/preprocess_tsmc.py tsmc_nyc   (oppure tsmc_tky)
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps

RAW = Path("/Users/lucaaliberti/Downloads/dataset_tsmc2014")
TAX = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/config/foursquare_legacy_taxonomy.json")
OUTROOT = Path("/Users/lucaaliberti/Downloads/xsage-clean/data/processed")
CITY = sys.argv[1] if len(sys.argv) > 1 else "tsmc_nyc"
FILE = {"tsmc_nyc": "dataset_TSMC2014_NYC.txt", "tsmc_tky": "dataset_TSMC2014_TKY.txt"}[CITY]
KCORE = 10
GH_PREC = 6
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat, lon, prec=GH_PREC):
    lat_r = [-90.0, 90.0]; lon_r = [-180.0, 180.0]; gh = []; bits = [16, 8, 4, 2, 1]
    bit = 0; ch = 0; even = True
    while len(gh) < prec:
        if even:
            mid = (lon_r[0] + lon_r[1]) / 2
            if lon > mid: ch |= bits[bit]; lon_r[0] = mid
            else: lon_r[1] = mid
        else:
            mid = (lat_r[0] + lat_r[1]) / 2
            if lat > mid: ch |= bits[bit]; lat_r[0] = mid
            else: lat_r[1] = mid
        even = not even
        if bit < 4: bit += 1
        else: gh.append(_B32[ch]); bit = 0; ch = 0
    return "".join(gh)


def build_fine2root():
    """mappa ogni categoria-id (a qualsiasi profondità) → nome del root."""
    tree = json.load(open(TAX)); m = {}
    def walk(node, root):
        m[node["id"]] = root
        for ch in node.get("categories", []): walk(ch, root)
    for top in tree:
        for ch in top.get("categories", []): walk(ch, top["name"])
        m[top["id"]] = top["name"]
    return m


def main():
    out = OUTROOT / CITY; out.mkdir(parents=True, exist_ok=True)
    f2r = build_fine2root(); print(f"[0] tassonomia: {len(f2r)} categorie → root", flush=True)
    cols = ["user", "venue", "catid", "catname", "lat", "lon", "tzoff", "utc"]
    df = pd.read_csv(RAW / FILE, sep="\t", header=None, names=cols, encoding="latin-1")
    print(f"[1] {CITY}: {len(df)} check-in raw", flush=True)
    df["cat_macro"] = df.catid.map(f2r)
    df = df.dropna(subset=["cat_macro"]).copy()
    print(f"[2] con root-macro: {len(df)}  macro distinte={df.cat_macro.nunique()}", flush=True)

    # tempo locale (UTC + offset minuti) per il contesto; UTC per l'ordinamento
    df["utc_dt"] = pd.to_datetime(df.utc, format="%a %b %d %H:%M:%S %z %Y")
    df["local"] = df.utc_dt + pd.to_timedelta(df.tzoff, unit="m")
    df["geohash"] = [geohash(a, b) for a, b in zip(df.lat.values, df.lon.values)]

    # k-core10
    while True:
        uc = df.user.value_counts(); ic = df.venue.value_counts()
        m = df.user.isin(uc[uc >= KCORE].index) & df.venue.isin(ic[ic >= KCORE].index)
        if m.all(): break
        df = df[m]
    print(f"[3] dopo k-core={KCORE}: {len(df)}  utenti={df.user.nunique()}  venue={df.venue.nunique()}", flush=True)

    uu = {u: i for i, u in enumerate(sorted(df.user.unique()))}
    ii = {it: i for i, it in enumerate(sorted(df.venue.unique()))}
    df["u_idx"] = df.user.map(uu).astype(np.int64); df["i_idx"] = df.venue.map(ii).astype(np.int64)
    n_users, n_items = len(uu), len(ii)
    macros = sorted(df.cat_macro.unique()); m2i = {mm: k for k, mm in enumerate(macros)}

    df = df.sort_values(["u_idx", "utc_dt"], kind="stable").reset_index(drop=True)
    df["time_local"] = df.local.dt.tz_localize(None)
    df["c_hour"] = df.local.dt.hour.astype(np.int64)
    df["c_dow"] = df.local.dt.dayofweek.astype(np.int64)
    df["c_isweekend"] = (df.c_dow >= 5).astype(np.int64)
    df["c_month"] = df.local.dt.month.astype(np.int64)
    df["prev_geohash5"] = df.groupby("u_idx")["geohash"].shift(1).fillna("none").astype(str)
    df["_cmi"] = df.cat_macro.map(m2i).astype(np.int64)
    df["intent_last_cat_idx"] = df.groupby("u_idx")["_cmi"].shift(1).fillna(len(macros)).astype(np.int64)
    r = df.groupby("u_idx").cumcount(); nu = df.groupby("u_idx")["u_idx"].transform("size")
    frac = r / nu
    df["split"] = np.where(frac < 0.8, "train", np.where(frac < 0.9, "val", "test"))

    keep = ["u_idx", "i_idx", "cat_macro", "time_local", "c_hour", "c_dow",
            "c_isweekend", "c_month", "prev_geohash5", "intent_last_cat_idx"]
    for sp in ("train", "val", "test"):
        d = df[df.split == sp][keep].reset_index(drop=True)
        d.to_parquet(out / f"df_{sp}.parquet"); print(f"[5] df_{sp}: {len(d)}", flush=True)

    def urm(s):
        d = df[df.split.isin(s)]
        return sps.csr_matrix((np.ones(len(d)), (d.u_idx, d.i_idx)), shape=(n_users, n_items))
    sps.save_npz(out / "URM_train.npz", urm(["train"])); sps.save_npz(out / "URM_val.npz", urm(["val"]))

    tr = df[df.split == "train"]
    test_users = np.array(sorted(df[df.split == "test"].u_idx.unique()), dtype=np.int64)
    np.savez(out / "_cornac_in.npz", train_u=tr.u_idx.values.astype(np.int64),
             train_i=tr.i_idx.values.astype(np.int64), n_users=np.int64(n_users),
             n_items=np.int64(n_items), test_users=test_users)
    print(f"\n[OK] {CITY} → {out}")
    print(f"  utenti={n_users} venue={n_items} macro={len(macros)} ({macros})")
    print(f"  check-in/utente media={df.groupby('u_idx').size().mean():.1f} min={df.groupby('u_idx').size().min()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
