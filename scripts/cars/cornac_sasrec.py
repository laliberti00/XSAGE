"""[venv-cornac] Baseline SEQUENZIALE SOTA — SASRec (Kang & McAuley 2018) nel NOSTRO task top-N.
Allena SASRec (Cornac) sulle sequenze train, poi per OGNI richiesta-test scora il next-item data la
storia (causale) e calcola le NOSTRE metriche (Cat-MRR/R@20/LT/Coverage/Gini) con la NOSTRA excl mask.
Salva outputs_results/cars_<city>.csv (append, method=SASRec). Anti-circolare: storia = solo passato.

Uso:  python scripts/cars/cornac_sasrec.py <city> [device=cpu] [epochs=20]
"""
import sys, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sps
from cornac.data import SequentialDataset
from cornac.models import SASRec

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
K_TOP = 20


def gini(x):
    x = np.sort(np.asarray(x, np.float64)); n = len(x); s = x.sum()
    return 0.0 if s <= 0 else float((2*np.sum(np.arange(1, n+1)*x)/(n*s)) - (n+1)/n)


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    P = CLEAN / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet"); dva = pd.read_parquet(P / "df_val.parquet"); dte = pd.read_parquet(P / "df_test.parquet")
    for d, s in [(dtr, "train"), (dva, "val"), (dte, "test")]: d["split"] = s
    alld = pd.concat([dtr, dva, dte], ignore_index=True)
    m2i = {m: k for k, m in enumerate(sorted(alld.cat_macro.unique()))}
    n_items = int(alld.i_idx.max()) + 1; n_users = int(alld.u_idx.max()) + 1
    icm = (alld.groupby("i_idx")["cat_macro"].first().map(m2i).reindex(np.arange(n_items), fill_value=0).values.astype(np.int64))
    # long-tail G1 (fuori dal top-20% per popolarità su train+val)
    ut = sps.load_npz(P / "URM_train.npz"); uv = sps.load_npz(P / "URM_val.npz")
    pop = np.asarray((ut + uv).sum(0)).ravel()
    thr = np.quantile(pop[pop > 0], 0.80) if (pop > 0).any() else 0
    G1 = (pop <= thr).astype(np.float64)
    excl = (ut + uv).tocsr(); excl.data[:] = 1.0

    # SASRec: USIT train (user, session=user, item, timestamp)
    dtr_s = dtr.sort_values(["u_idx", "time_local"], kind="stable")
    tcol = (dtr_s.groupby("u_idx").cumcount() + 1).values
    usit = list(zip(dtr_s.u_idx.astype(str), dtr_s.u_idx.astype(str), dtr_s.i_idx.astype(str), tcol))
    ds = SequentialDataset.build(usit, fmt="USIT")
    uid, iid = dict(ds.uid_map), dict(ds.iid_map)
    inv = np.full(len(iid), -1, np.int64)
    for si, c in iid.items(): inv[int(c)] = int(si)            # cornac-item-idx → nostro i_idx
    print(f"[{city}] SASRec fit (dev={dev}, ep={epochs}) users={len(uid)} items={len(iid)}", flush=True)
    m = SASRec(embedding_dim=64, n_epochs=epochs, max_len=50, num_blocks=2, num_heads=1,
               device=dev, seed=42, batch_size=256, verbose=False)
    m.fit(ds)

    # eval per-richiesta: storia causale, score next-item, NOSTRE metriche
    alld = alld.sort_values(["u_idx", "time_local"], kind="stable").reset_index(drop=True)
    hist = defaultdict(list)                                    # u_idx → [cornac item idx]
    cm = []; hit = []; lt = []; expo = np.zeros(n_items); uu = []
    for r in alld.itertuples():
        u = r.u_idx; i = r.i_idx
        if r.split == "test":
            uc = uid.get(str(u))
            if uc is not None and hist[u]:
                sc = np.asarray(m.score(uc, hist[u]), np.float32)   # (n_cornac_items,)
                S = np.full(n_items, -np.inf, np.float32); S[inv] = sc
                cc = excl.indices[excl.indptr[u]:excl.indptr[u+1]]
                if len(cc): S[cc] = -np.inf
                part = np.argpartition(-S, K_TOP-1)[:K_TOP]; order = np.argsort(-S[part]); topk = part[order]
                macros = icm[topk]; tmac = icm[i]
                match = np.where(macros == tmac)[0]
                cm.append(1.0/(match[0]+1) if len(match) else 0.0)
                s_t = S[i]; hit.append(int((S > s_t).sum()+1 <= K_TOP))
                lt.append(G1[topk].mean()); np.add.at(expo, topk, 1.0); uu.append(u)
        ic = iid.get(str(i))
        if ic is not None: hist[u].append(int(ic))

    cm = np.array(cm); hit = np.array(hit); lt = np.array(lt); uu = np.array(uu)
    uq, ix = np.unique(uu, return_inverse=True); ss = np.zeros(len(uq)); cc2 = np.zeros(len(uq))
    np.add.at(ss, ix, hit); np.add.at(cc2, ix, 1)
    row = {"city": city, "method": "SASRec", "CatMRR": round(float(cm.mean()), 5),
           "R20": round(float((ss/cc2).mean()), 5), "LT20": round(float(lt.mean()), 5),
           "Coverage": round(float((expo > 0).mean()), 5), "Gini": round(gini(expo), 5)}
    print("  ", row, flush=True)
    OUT = CLEAN / "outputs_results" / f"cars_{city}.csv"
    df = pd.DataFrame([row])
    if OUT.exists():
        old = pd.read_csv(OUT); old = old[old.method != "SASRec"]; df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT, index=False); print(f"→ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
