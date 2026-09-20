"""[venv-xsage] E-1 — La situazione vale di piu' quando l'utente si conosce di meno?

PRE-REGISTRATO nel brief E-1. Griglia, bracci, guardia e regola di decisione fissati prima dei
numeri. NON e' un probe diagnostico: e' l'esperimento che fonda il capitolo privacy.

MECCANISMO (zero riaddestramento). EASE e' lineare nel vettore di interazioni: s_u = r_u . B.
Troncare r_u alle ultime n interazioni e' un prodotto piu' corto. B resta appresa sull'intero
training set (scoping dichiarato: modello addestrato offline su dati consentiti, ma al momento
della raccomandazione si conosce solo la finestra recente).

B NON e' salvata su disco (c'e' solo il prodotto r_u.B in float16): viene ricostruita in forma
chiusa (Steck 2019) con lamb=500 e **posB=True**, il default di cornac che azzera le entrate
negative — senza quel clipping la ricostruzione NON riproduce i punteggi pubblicati.
Verifica eseguita su ml1m: corr 0.9999999694, max|diff| 0.00195 < risoluzione float16 (0.0039).

STRUTTURA TEMPORALE: lo split e' temporale per-utente, quindi train+val precedono interamente il
test. Le "ultime n prima di t" sono percio' le stesse per tutte le richieste di test di un utente
-> il troncamento e' PER-UTENTE e S^(n) resta (n_utenti x n_item).

PARITA' INFORMATIVA (vincolo non negoziabile del brief): il descrittore usa la STESSA n del
backbone, via build_descriptor(n=...), e attinge allo stesso pool (train+val).

TRE BRACCI: BASE(n) · PRIOR(n) = bias di categoria globale non situazionale (UNI_mean, media di
b_z sulle situazioni: stessa forma additiva, stessa scala, un solo vettore) · SIT(n).
La quantita' decisiva e' Delta_sit(n) = SIT(n) - PRIOR(n).
GUARDIA G1: braccio con etichette di situazione PERMUTATE fra le richieste (stessa marginale).

Uso:  python -m scripts.exp.e1_truncation [ml1m ...] [--no-kappa-resel]
Out:  outputs_results/exp_e1/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import ALPHA, build_descriptor, membership_from_assign          # noqa: E402
from eval_kappa import KAPPA_GRID, SEED, select_K, select_eps                   # noqa: E402
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage.recommendation import fit_situation_biases_z                          # noqa: E402

OUT = CLEAN / "outputs_results" / "exp_e1"
CACHE = Path("/private/tmp/claude-501/-Users-lucaaliberti-Downloads-xsage-clean/"
             "39f635c3-ccf8-4202-8c8d-2c274d55280b/scratchpad")
N_GRID = [1, 2, 3, 5, 8, 13, 21, None]        # None = all (nessun troncamento)
SEEDS = [42, 43, 44, 45, 46]
LAMB, KTOP, BATCH, BOOT = 500.0, 20, 1024, 600
ANCHOR_BLIND = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
DECISIVE = "ml1m"
ARMS = ["BASE", "PRIOR", "SIT", "PERM"]


def nlab(n): return "all" if n is None else str(n)


# ----------------------------------------------------------------- EASE in forma chiusa
def ease_B(city, nI, dtr):
    f = CACHE / f"B_ease_{city}.npy"
    if f.exists():
        B = np.load(f)
        if B.shape == (nI, nI): return B
    print(f"[{city}] ricostruzione B di EASE (lamb={LAMB}, posB=True)...", flush=True)
    X = sps.csr_matrix((np.ones(len(dtr), np.float32),
                        (dtr.u_idx.values, dtr.i_idx.values)), shape=(int(dtr.u_idx.max()) + 1, nI))
    X.data[:] = 1.0
    G = (X.T @ X).toarray().astype(np.float64); G[np.diag_indices(nI)] += LAMB
    P = np.linalg.inv(G); B = P / (-np.diag(P)); B[np.diag_indices(nI)] = 0.0
    B[B < 0] = 0.0                                    # posB=True (default cornac)
    B = B.astype(np.float32); CACHE.mkdir(parents=True, exist_ok=True); np.save(f, B)
    return B


def truncated_scores(hist_df, nU, nI, B, n):
    """S^(n) = R^(n) . B con R^(n) = ultime n interazioni (train+val) per utente.
    n=None -> tutte. Il pool coincide con quello del descrittore -> parita' informativa."""
    d = hist_df.sort_values(["u_idx", "time_local"], kind="stable")
    if n is not None:
        d = d.groupby("u_idx", sort=False).tail(n)
    R = sps.csr_matrix((np.ones(len(d), np.float32),
                        (d.u_idx.values, d.i_idx.values)), shape=(nU, nI))
    R.data[:] = 1.0
    return np.asarray(R @ B, dtype=np.float32)


# ------------------------------------------------------------------------- valutazione
def eval_arm(S, df, dmac, gam, icm, excl, kappa):
    """Cat-MRR@20 (stessa convenzione di eval_kappa.cat_mrr, quella del gate) + HR@20 + NDCG@20."""
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64)
    tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); hit = np.zeros(n); ndcg = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); ub = u[bs:be]
        Sb = S[ub].astype(np.float32, copy=True)
        if dmac is not None and kappa > 0:
            Sb = Sb + kappa * gam[bs:be][:, None].astype(np.float32) * dmac[bs:be][:, icm]
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(ub[j])]:excl.indptr[int(ub[j]) + 1]]
            if len(cc): Sb[j, cc] = -np.inf
        part = np.argpartition(-Sb, KTOP - 1, axis=1)[:, :KTOP]
        order = np.argsort(-np.take_along_axis(Sb, part, 1), axis=1)
        tk = np.take_along_axis(part, order, 1)
        macros = icm[tk]
        match = macros == tm[bs:be, None]; has = match.any(1)
        first = np.where(has, match.argmax(1) + 1, 0)
        cm[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        pos = (tk == i_t[bs:be, None])
        h = pos.any(1); rk = np.where(h, pos.argmax(1) + 1, 0)
        hit[bs:be] = h.astype(float)
        ndcg[bs:be] = np.where(h, 1.0 / np.log2(np.maximum(rk, 1) + 1.0), 0.0)
    return cm, hit, ndcg, tm


def macro_avg(cm, tm, nmac, msupp=20):
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).sum() >= msupp]
    return float(np.mean(per)) if per else np.nan


# ------------------------------------------------------------------------------- run
def run(city, do_resel=True):
    OUT.mkdir(parents=True, exist_ok=True)
    Pdir = CLEAN / "data" / "processed" / city
    dtr = pd.read_parquet(Pdir / "df_train.parquet")
    dva = pd.read_parquet(Pdir / "df_val.parquet")
    alld = pd.concat([dtr, dva, pd.read_parquet(Pdir / "df_test.parquet")], ignore_index=True)
    nU = int(alld.u_idx.max()) + 1; nI = int(alld.i_idx.max()) + 1
    B = ease_B(city, nI, dtr)
    hist_tv = pd.concat([dtr[["u_idx", "i_idx", "time_local"]],
                         dva[["u_idx", "i_idx", "time_local"]]], ignore_index=True)
    hist_tr = dtr[["u_idx", "i_idx", "time_local"]]
    Msaved = np.load(CLEAN / "data" / city / "backbone" / "EASE.scores_user.npy", mmap_mode="r")

    bt = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")
    def kstar(bk, meth, seed):
        r = bt[(bt.seed == seed) & (bt.backbone == bk) & (bt.method == meth)]["kstar"]
        if not len(r): r = bt[(bt.seed == 42) & (bt.backbone == bk) & (bt.method == meth)]["kstar"]
        return float(r.iloc[0])

    rows = []; resel = []; anchors = []
    for n in N_GRID:
        S = truncated_scores(hist_tv, nU, nI, B, n)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            D0 = build_descriptor(city, splits=("train", "val", "test"), n=n)   # PARITA': stessa n
            ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; cmt = D0["cmt"]
            vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
            K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
            fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
            b_z = fit_situation_biases_z(fit.core_label.astype(np.int64), cmt, K, nmac, alpha=ALPHA)

            def asg(v):
                _, k, comp, isb = _assign(v, fit.prototypes, eps)
                return (membership_from_assign(k, comp, isb, K).astype(np.float32),
                        (1.0 / np.maximum(comp.sum(1), 1)).astype(np.float32))
            mem, gam = asg(vte); mem_v, gam_v = asg(vva)
            dft = ds["df_test"]; dfv = ds["df_val"]; nreq = len(dft)
            nudge = mem @ b_z
            prior = np.ascontiguousarray(np.broadcast_to(b_z.mean(0), (nreq, nmac)).astype(np.float32))
            pr = np.random.default_rng(1000 + seed).permutation(nreq)               # G1
            kap = kstar("EASE", "SIT", seed)
            dm = {"BASE": None, "PRIOR": prior, "SIT": nudge, "PERM": nudge[pr]}
            for arm in ARMS:
                cm, hit, ndcg, tm = eval_arm(S, dft, dm[arm], gam, icm, excl,
                                             0.0 if arm == "BASE" else kap)
                rows.append(dict(dataset=city, n=nlab(n), seed=seed, arm=arm, kappa=kap,
                                 K=K, eps=eps, n_req=nreq, n_users=int(dft.u_idx.nunique()),
                                 macroCatMRR=macro_avg(cm, tm, nmac), CatMRR=float(cm.mean()),
                                 HR20=float(hit.mean()), NDCG20=float(ndcg.mean())))
            # --- analisi SECONDARIA: kappa ri-selezionato su VALIDATION per ogni n (etichettata)
            if do_resel:
                Sv = S
                nudge_v = mem_v @ b_z
                prior_v = np.ascontiguousarray(np.broadcast_to(b_z.mean(0), (len(dfv), nmac)).astype(np.float32))
                for arm, dv, dt in (("SIT", nudge_v, nudge), ("PRIOR", prior_v, prior)):
                    best = (-1.0, None)
                    for kk in KAPPA_GRID:
                        cmv, _, _, _ = eval_arm(Sv, dfv, dv, gam_v, icm, excl, kk)
                        if cmv.mean() > best[0]: best = (float(cmv.mean()), kk)
                    cm, hit, ndcg, tm = eval_arm(S, dft, dt, gam, icm, excl, best[1])
                    resel.append(dict(dataset=city, n=nlab(n), seed=seed, arm=arm,
                                      kappa_reselected=best[1], val_CatMRR=round(best[0], 5),
                                      macroCatMRR=macro_avg(cm, tm, nmac), CatMRR=float(cm.mean()),
                                      HR20=float(hit.mean()), NDCG20=float(ndcg.mean())))
            # --- ancoraggi (solo n=all, seed 42)
            if n is None and seed == 42:
                for lab, SS in (("EASE_salvata_PUBBLICATA", Msaved),
                                ("EASE_ricostruita_pool_TRAIN_only", truncated_scores(hist_tr, nU, nI, B, None)),
                                ("EASE_ricostruita_pool_TRAIN+VAL", S)):
                    cmB, _, _, tm = eval_arm(SS, dft, None, gam, icm, excl, 0.0)
                    cmS, _, _, _ = eval_arm(SS, dft, nudge, gam, icm, excl, kap)
                    anchors.append(dict(dataset=city, sorgente=lab, BASE_CatMRR=round(float(cmB.mean()), 6),
                                        SIT_CatMRR=round(float(cmS.mean()), 6),
                                        BASE_macro=round(macro_avg(cmB, tm, nmac), 5),
                                        SIT_macro=round(macro_avg(cmS, tm, nmac), 5)))
                cmb, _, _, _ = eval_arm(D0["sb"], dft, nudge, gam, icm, excl,
                                        kstar("B_blind", "SIT", 42))
                anchors.append(dict(dataset=city, sorgente="GATE_B_blind (riferimento harness)",
                                    BASE_CatMRR=np.nan, SIT_CatMRR=round(float(cmb.mean()), 5),
                                    BASE_macro=np.nan, SIT_macro=ANCHOR_BLIND.get(city, np.nan)))
            print(f"  [{city}] n={nlab(n):>3} seed={seed} K={K} eps={eps}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"e1_curve_{city}.csv", index=False)
    pd.DataFrame([r for r in rows if r["arm"] == "PERM"]).to_csv(
        OUT / f"e1_permutation_guard_{city}.csv", index=False)
    if resel: pd.DataFrame(resel).to_csv(OUT / f"e1_kappa_reselected_{city}.csv", index=False)
    pd.DataFrame(anchors).to_csv(OUT / f"e1_anchors_{city}.csv", index=False)
    print(f"[{city}] -> e1_curve_{city}.csv ({len(rows)} righe)", flush=True)
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for c in (args or [DECISIVE]):
        print(f"\n===== E-1 · {c.upper()} =====", flush=True)
        run(c, do_resel="--no-kappa-resel" not in sys.argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
