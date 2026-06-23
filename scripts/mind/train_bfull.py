"""[venv-xsage] SCAFFOLD — allena B_full (ContextAwareFM, repo OLD) su MIND/ml1m per il confronto
B3 (SIT vs context-aware). Adattamenti: NIENTE geo (n_geo=0 → solo sentinel), NIENTE fine
(n_fine=1 → tutti 0). Feature costruite DIRETTAMENTE dalle colonne intere del df (bypassa
_build_request_features che vuole stringhe+geo+fine). Salva data/<city>/backbone/Bfull.scores.npy
[n_test x n_items] (sostituisce lo stub).

⚠️ SCAFFOLD: la parte di SCORING è assemblata da score_full_catalogue — va verificata con una
run di prova prima di fidarsi dei numeri. Training = train_b_full del repo OLD (invariato).

Uso:  python scripts/mind/train_bfull.py <city> [n_epochs]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import torch

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from xsage import data as D
from pipeline.step02_models.xsage.backbone_full import (ContextAwareFM, FeatureSpec, train_b_full)

EMB_D, EPOCHS_DEF, BATCH_SCORE = 32, 12, 512


def feats_from_df(df, icm, n_macros):
    """Feature intere dirette (no geo, no fine). intent_last: remap a 0=none, 1..n_macros."""
    il = df["intent_last_cat_idx"].values.astype(np.int64)
    il = np.where(il >= n_macros, 0, il + 1)          # 0=none, 1..n_macros
    i_idx = df["i_idx"].values.astype(np.int64)
    return {
        "u_idx": df["u_idx"].values.astype(np.int64),
        "i_idx": i_idx,
        "macro_idx": icm[i_idx].astype(np.int64),
        "fine_idx": np.zeros(len(df), np.int64),       # fine disabilitato
        "c_hour": df["c_hour"].values.astype(np.int64),
        "c_dow": df["c_dow"].values.astype(np.int64),
        "c_isw": df["c_isweekend"].values.astype(np.int64),
        "c_month": (df["c_month"].values - df["c_month"].values.min()).astype(np.int64),
        "prev_geo_idx": np.zeros(len(df), np.int64),   # geo disabilitato (sentinel)
        "intent_last_idx": il,
    }


def score_test(model, feats_te, spec, icm, device):
    """Bfull [n_test x n_items] via score_full_catalogue. ⚠️ DA VERIFICARE con run di prova."""
    offs = spec.offsets(); I = spec.n_items
    items = torch.arange(I, device=device)
    item_off = items + offs["item"]
    macro_off = torch.from_numpy(icm.astype(np.int64)).to(device) + offs["macro"]
    fine_off = torch.zeros(I, dtype=torch.long, device=device) + offs["fine"]
    n = len(feats_te["u_idx"]); out = np.zeros((n, I), np.float32)
    # context_idx (B,7): user, hour, dow, isw, month, prev_geo, intent_last (offset)
    cols = [("u_idx", "user"), ("c_hour", "hour"), ("c_dow", "dow"), ("c_isw", "isw"),
            ("c_month", "month"), ("prev_geo_idx", "prev_geo"), ("intent_last_idx", "intent_last")]
    model.eval()
    with torch.no_grad():
        for bs in range(0, n, BATCH_SCORE):
            be = min(n, bs + BATCH_SCORE)
            ctx = torch.stack([torch.from_numpy(feats_te[c][bs:be]).to(device) + offs[o]
                               for c, o in cols], dim=-1)        # (B,7)
            S = model.score_full_catalogue(ctx, item_off, macro_off, fine_off)  # (B,I)
            out[bs:be] = S.cpu().numpy()
    return out


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "mind"
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS_DEF
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ds = D.load_city(city, data_root=str(CLEAN))
    n_users, n_items, n_macros = int(ds["n_users"]), int(ds["n_items"]), int(ds["n_macros"])
    m2i = ds["macro_to_idx"]
    import pandas as pd
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (df_all.groupby("i_idx")["cat_macro"].first().map(m2i)
           .reindex(np.arange(n_items), fill_value=0).values.astype(np.int64))

    spec = FeatureSpec(n_users=n_users, n_items=n_items, n_macros=n_macros, n_fine=1,
                       n_geo=0, n_intent_last=n_macros)
    model = ContextAwareFM(spec, d=EMB_D).to(device)
    print(f"[{city}] B_full: {n_users} users, {n_items} items, {n_macros} macro; device={device}", flush=True)

    tv = pd.concat([ds["df_train"], ds["df_val"]], ignore_index=True)
    feats_tv = feats_from_df(tv, icm, n_macros)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.0
    hist = train_b_full(model, feats_tv, mask, icm, np.zeros(n_items, np.int64),
                        device, n_epochs=n_epochs, verbose=True)
    print(f"[{city}] training done (loss finale={hist.get('history',[{}])[-1].get('loss','?') if isinstance(hist,dict) else '?'})", flush=True)

    feats_te = feats_from_df(ds["df_test"], icm, n_macros)
    scores = score_test(model, feats_te, spec, icm, device)
    out = CLEAN / "data" / city / "backbone" / "Bfull.scores.npy"
    np.save(out, scores)
    print(f"[{city}] B_full scores [n_test x n_items]={scores.shape} → {out}")
    print("  ⚠️ SCAFFOLD: verificare lo scoring con un sanity (es. ranking del target ragionevole) "
          "prima di usarlo nel confronto B3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
