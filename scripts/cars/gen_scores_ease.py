"""[venv-cornac] Genera la matrice score per-utente di EASE^R (Steck 2019, WWW) come BACKBONE
citabile, da montare nella battery uniforme (XTRA_BACKBONES=EASE). EASE è statico/per-utente
(come B_blind): s_u = r_u · B, stesso vettore per tutte le richieste dell'utente.

Anti-circolare: B appreso su TRAIN (come B_blind/B_full). Score = r_u(train) · B.
Allineamento: M[u_idx, i_idx] sui NOSTRI indici → l'hook fa M[uv[idx]] (per-utente).

Uso:  python scripts/cars/gen_scores_ease.py <city> [lamb=500]
Out:  data/<city>/backbone/EASE.scores_user.npy   [n_users x n_items] float16
Modello: cornac EASE^R [Steck 2019]; framework: Cornac [Salah et al. 2020, JMLR].
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cornac.data import Dataset
from cornac.models import EASE

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "nyc_tist"
    lamb = float(sys.argv[2]) if len(sys.argv) > 2 else 500.0
    P = CLEAN / "data" / "processed" / city
    dtr = pd.read_parquet(P / "df_train.parquet")
    alld = pd.concat([dtr, pd.read_parquet(P / "df_val.parquet"), pd.read_parquet(P / "df_test.parquet")], ignore_index=True)
    n_users = int(alld.u_idx.max()) + 1
    n_items = int(alld.i_idx.max()) + 1
    print(f"[{city}] EASE lamb={lamb} | n_users={n_users} n_items={n_items} | train_int={len(dtr)}", flush=True)

    data = list(zip(dtr.u_idx.astype(str), dtr.i_idx.astype(str), np.ones(len(dtr), np.float32)))
    ts = Dataset.from_uir(data)
    m = EASE(lamb=lamb, verbose=False)
    m.fit(ts)

    # mappe cornac → nostri indici
    uid = dict(ts.uid_map)                      # "our_u"(str) → cornac_user_idx
    iid = dict(ts.iid_map)                       # "our_i"(str) → cornac_item_idx
    inv_i = np.full(len(iid), -1, np.int64)
    for si, c in iid.items():
        inv_i[int(c)] = int(si)                  # cornac_item_idx → our i_idx

    M = np.zeros((n_users, n_items), np.float16)
    for su, uc in uid.items():
        sc = np.asarray(m.score(int(uc)), np.float32)   # (n_cornac_items,)
        M[int(su), inv_i] = sc.astype(np.float16)
    out = CLEAN / "data" / city / "backbone"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "EASE.scores_user.npy", M)
    print(f"[{city}] -> {out/'EASE.scores_user.npy'}  shape={M.shape}  (utenti scorati={len(uid)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
