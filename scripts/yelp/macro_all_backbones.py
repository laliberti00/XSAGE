"""[venv-xsage] macro-Cat-MRR (equità-qualità categoriale) BASE→SIT per TUTTI i 7 backbone.
Usa le matrici score TEST già su disco + il κ* per SIT già selezionato dalla battery (colonna
kstar nel CSV, seed 42). Nessun training, nessuna ri-selezione: solo ri-scoring del breakdown
per-categoria. Auto-validazione: confronta il BASE micro ricalcolato col CSV (deve combaciare).

Uso:  python scripts/yelp/macro_all_backbones.py <city>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
import torch
from train_bfull import feats_from_df, score_test
from pipeline.step02_models.xsage.backbone_full import ContextAwareFM, FeatureSpec, train_b_full

K_TOP = 20
BK = ["B_blind", "B_full", "EASE", "SASRec", "FPMC", "DeepFM", "AFM"]


def bfull_test_scores(ds, icm, excl, nmac, dev):
    """Riallena B_full (seed 42) replicando la selezione della battery: griglia (32,64)x5e-3,
    checkpoint a 3/6/9 epoche, miglior config su val (micro Cat-MRR). Bfull.scores.npy su disco
    è un artefatto vecchio per-utente → NON usato."""
    import copy
    nU = int(ds["n_users"]); nItems = int(ds["n_items"])
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icmF = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(nItems), fill_value=0).values.astype(np.int64))
    ftr = feats_from_df(ds["df_train"], icmF, nmac); fva = feats_from_df(ds["df_val"], icmF, nmac); fte = feats_from_df(ds["df_test"], icmF, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); iv = ds["df_val"]["i_idx"].values.astype(np.int64)
    spec = FeatureSpec(n_users=nU, n_items=nItems, n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac)
    best = (-1, None)
    for emb in (32, 64):
        torch.manual_seed(42); mdl = ContextAwareFM(spec, d=emb).to(dev)
        for _ in range(3):                                   # 3 checkpoint da 3 epoche = 9
            train_b_full(mdl, ftr, mask, icmF, np.zeros(nItems, np.int64), dev, lr=5e-3, n_epochs=3, verbose=False)
            sv = score_test(mdl, fva, spec, icmF, dev)
            vm = cat_mrr_per_req((lambda idx, S=sv: S[idx]), None, 0.0, np.ones(len(uv), np.float32), uv, iv, icm, excl, nItems).mean()
            if vm > best[0]: best = (vm, copy.deepcopy(mdl.state_dict()), emb)
    mdl = ContextAwareFM(spec, d=best[2]).to(dev); mdl.load_state_dict(best[1])
    return score_test(mdl, fte, spec, icmF, dev)             # [n_test x n_items]


def load_test_matrix(city, bk, sb, ute, bdir, bft):
    if bk == "B_blind":
        return (lambda idx: sb[ute[idx]]), "sb (per-utente)"
    if bk == "B_full":
        return (lambda idx: bft[idx]), "B_full (riallenato seed42)"
    fu = bdir / f"{bk}.scores_user.npy"
    if fu.exists():
        M = np.load(fu, mmap_mode="r"); return (lambda idx: M[ute[idx]]), f"{bk}.scores_user"
    Mt = np.load(bdir / f"{bk}.scores_test.npy", mmap_mode="r"); return (lambda idx: Mt[idx]), f"{bk}.scores_test"


def cat_mrr_per_req(loader, nudge_item, kappa, gam, u_te, i_te, icm, excl, nI):
    """Cat-MRR per richiesta da matrice score. nudge_item=None per BASE."""
    n = len(u_te); tm = icm[i_te]; cm = np.zeros(n)
    for bs in range(0, n, 1024):
        be = min(n, bs + 1024); idx = np.arange(bs, be)
        S = np.asarray(loader(idx)).astype(np.float32, copy=True)
        if nudge_item is not None and kappa > 0:
            S = S + kappa * gam[idx][:, None].astype(np.float32) * nudge_item[idx][:, icm]
        for j in range(be - bs):
            u = int(u_te[bs + j]); cc = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1); topk = np.take_along_axis(part, order, 1)
        mac = icm[topk]; match = mac == tm[idx, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[bs:be] = np.where(first > 0, 1. / np.maximum(first, 1), 0.)
    return cm


def macro(cm, tm, nmac):
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).any()]
    return float(np.mean(per))


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    nI = int(ds["n_items"]); bdir = CLEAN / "data" / city / "backbone"
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    mem_te = membership_from_assign(kte, compte, isbte, K); gam_te = (1.0 / np.maximum(compte.sum(1), 1).astype(np.float32))
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
    tm = icm[ite]; nudge = mem_te.astype(np.float32) @ b_z                        # [n_test x nmac]
    cnt = np.bincount(tm, minlength=nmac); sat = cnt.max() / cnt.sum()

    csv = pd.read_csv(CLEAN / "outputs_results" / f"battery_bfull_{city}.csv")
    ks = csv[(csv.seed == 42) & (csv.method == "SIT")].set_index("backbone")["kstar"].to_dict()
    base_csv = csv[(csv.seed == 42) & (csv.method == "BASE")].set_index("backbone")["CatMRR"].to_dict()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[{city}] rialleno B_full (seed42, selezione su val)...", flush=True)
    bft = bfull_test_scores(ds, icm, excl, nmac, dev)

    print(f"\n########## {city}  (saturazione dominante={sat:.3f}, K={K}) ##########")
    print(f"{'backbone':9} {'BASEmic':>8} {'(CSVchk)':>9} {'BASEmac':>8} | {'SITmic':>8} {'SITmac':>8} | {'Δmacro':>9} {'κ*':>5}  src")
    out = []
    for bk in BK:
        loader, src = load_test_matrix(city, bk, sb, ute, bdir, bft)
        kap = float(ks.get(bk, 0.0))
        cmB = cat_mrr_per_req(loader, None, 0.0, gam_te, ute, ite, icm, excl, nI)
        cmS = cat_mrr_per_req(loader, nudge, kap, gam_te, ute, ite, icm, excl, nI)
        bmic, bmac = cmB.mean(), macro(cmB, tm, nmac); smic, smac = cmS.mean(), macro(cmS, tm, nmac)
        chk = base_csv.get(bk, np.nan); ok = "ok" if abs(bmic - chk) < 0.01 else f"DIFF({chk:.3f})"
        print(f"{bk:9} {bmic:8.4f} {ok:>9} {bmac:8.4f} | {smic:8.4f} {smac:8.4f} | {smac-bmac:+9.4f} {kap:5.2f}  {src}")
        out.append(dict(city=city, backbone=bk, sat=round(sat, 4), kstar=kap,
                        BASE_micro=round(bmic, 5), BASE_macro=round(bmac, 5),
                        SIT_micro=round(smic, 5), SIT_macro=round(smac, 5), dMacro=round(smac - bmac, 5),
                        BASE_micro_csv=round(float(chk), 5) if not np.isnan(chk) else None))
    pd.DataFrame(out).to_csv(CLEAN / "outputs_results" / f"macro_allbk_{city}.csv", index=False)
    print(f"-> outputs_results/macro_allbk_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
