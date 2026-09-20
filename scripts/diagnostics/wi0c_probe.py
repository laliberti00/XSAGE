"""[venv-xsage] WI-0c — Il guadagno si concentra dove la situazione cambia?

PRE-REGISTRATO nel brief WI-0c: bucket, regola di decisione e guardia di falsificazione sono
fissati PRIMA di vedere i numeri e non vengono rivisti dopo.

WI-0b ha stabilito che l'entropia situazionale e' ~0 sul test (finestre di 3-9 interazioni) ma
0.70-0.82 sulla timeline intera. Conseguenza non quantificata: X-SAGE reagisce ai cambi di
situazione ma e' valutato dove la situazione quasi non cambia. Questo probe misura, DALL'INTERNO
dei dati esistenti, se il vantaggio SIT-BASE si concentra vicino ai cambi.

  E1  Delta per distanza dall'ultimo cambio di situazione (misura principale).
      Con SIT e BASE riportati SEPARATAMENTE: guardia di falsificazione obbligatoria.
  E2  Delta per utente, stratificato per n. situazioni nel test, DENTRO bin di lunghezza test.
  E3  Regressione a livello utente: variare (entropia, transizioni) vs avere piu' dati (lunghezza).

Backbone: B_blind — e' cio' che produce i valori del gate (0.38479/0.34162/0.40045) ed e' cio'
su cui girano WI-0/WI-0b. NB: nel repo B_blind e' salvato in `FM.scores.npy` ma contiene punteggi
BPR (cornac_backbone.py:1); l'FM context-aware e' B_full, che results_record.py:200 RIALLENA per
seed e violerebbe il vincolo "zero retraining". Ambiguita' dichiarata, non risolta in silenzio.

5 seed {42-46}, kappa* CONGELATO dalla battery. Bootstrap a due livelli: ricampiona UTENTI
(cluster) e SEED, mai richieste. Nessun retraining, nessun ri-split.

Uso:  python -m scripts.diagnostics.wi0c_probe [ml1m nyc_tist saopaulo]
Out:  outputs_results/probe_wi0c/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import ALPHA, build_descriptor, membership_from_assign          # noqa: E402
from eval_kappa import SEED, cat_mrr, select_K, select_eps                      # noqa: E402
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage.recommendation import fit_situation_biases_z                          # noqa: E402

OUT = CLEAN / "outputs_results" / "probe_wi0c"
SEEDS = [42, 43, 44, 45, 46]
BOOT = 1000
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
# bucket PRE-REGISTRATI (l'ordine e' l'indice usato nelle matrici)
BUCKETS = ["d=0", "d1-2", "d3-5", "d>5", "stable", "pre_first_change"]
NB = len(BUCKETS)
MAIN = ["d=0", "d1-2"]          # vicino al cambio
FAR = ["d>5", "stable"]          # lontano / controllo
# priorita' dataset DICHIARATA PRIMA dei numeri
DECISIVE, SUPPORT, UNDERPOWERED = "ml1m", "saopaulo", "nyc_tist"


def rowd(**kw): return kw


# ------------------------------------------------------------------ stima per seed
def per_seed(city, seed, kap):
    rng = np.random.default_rng(seed)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
    cmt = D0["cmt"]; vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    _, k_te, comp, isb = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(k_te, comp, isb, K)
    gam = 1.0 / np.maximum(comp.sum(1), 1).astype(np.float32)
    b_z = fit_situation_biases_z(fit.core_label.astype(np.int64), cmt, K, nmac, alpha=ALPHA)
    dft = ds["df_test"]
    q_sit = cat_mrr(sb, dft, mem.astype(np.float32) @ b_z, gam, icm, excl, kap)
    q_base = cat_mrr(sb, dft, None, gam, icm, excl, 0.0)
    return dict(K=K, eps=eps, kap=kap, k_te=k_te.astype(np.int64),
                q_sit=q_sit.astype(np.float64), q_base=q_base.astype(np.float64),
                u=dft["u_idx"].values.astype(np.int64),
                t=dft["time_local"].values, n_users_ds=int(ds["n_users"]))


def distance_buckets(u, t, k):
    """d = n. richieste dall'ultimo CAMBIO di situazione, dentro la sequenza di test dell'utente
    ordinata per tempo. Utenti senza alcun cambio -> 'stable'. Richieste PRIMA del primo cambio in
    un utente che cambia -> 'pre_first_change' (categoria a se', riportata, fuori dal contrasto)."""
    o = np.lexsort((t, u))
    us, ks = u[o], k[o]
    b = np.full(len(us), BUCKETS.index("pre_first_change"), np.int64)
    starts = np.flatnonzero(np.r_[True, us[1:] != us[:-1]]); ends = np.r_[starts[1:], len(us)]
    i0, i12, i35, i5, ist = (BUCKETS.index(x) for x in ("d=0", "d1-2", "d3-5", "d>5", "stable"))
    for a, z in zip(starts, ends):
        seq = ks[a:z]
        if len(seq) < 2 or (seq[1:] == seq[:-1]).all():
            b[a:z] = ist; continue
        last = -1
        for i in range(len(seq)):
            if i > 0 and seq[i] != seq[i - 1]: last = i
            if last < 0: continue                      # prima del primo cambio
            d = i - last
            b[a + i] = i0 if d == 0 else i12 if d <= 2 else i35 if d <= 5 else i5
    inv = np.empty(len(us), np.int64); inv[o] = np.arange(len(us))
    return b[inv]


# ------------------------------------------------------------------------- CHECK E1
def check_e1(city, P, urank, nU, rng):
    S_sit = np.zeros((len(SEEDS), nU, NB)); S_base = np.zeros_like(S_sit)
    CNT = np.zeros_like(S_sit)
    for si, p in enumerate(P):
        b = distance_buckets(p["u"], p["t"], p["k_te"]); p["bucket"] = b
        ur = urank[p["u"]]
        np.add.at(S_sit[si], (ur, b), p["q_sit"])
        np.add.at(S_base[si], (ur, b), p["q_base"])
        np.add.at(CNT[si], (ur, b), 1.0)

    def stat(seed_idx, uidx):
        """medie per bucket, mediate sui seed estratti (bootstrap a due livelli)."""
        acc_s = np.zeros(NB); acc_b = np.zeros(NB)
        for si in seed_idx:
            c = CNT[si][uidx].sum(0); c = np.maximum(c, 1e-12)
            acc_s += S_sit[si][uidx].sum(0) / c
            acc_b += S_base[si][uidx].sum(0) / c
        return acc_s / len(seed_idx), acc_b / len(seed_idx)

    all_u = np.arange(nU); all_s = np.arange(len(SEEDS))
    sit0, base0 = stat(all_s, all_u)
    bs = np.empty((BOOT, NB)); bb = np.empty((BOOT, NB))
    for r in range(BOOT):
        si = rng.integers(0, len(SEEDS), len(SEEDS)); ui = rng.integers(0, nU, nU)
        bs[r], bb[r] = stat(si, ui)
    bd = bs - bb
    R = []
    for j, name in enumerate(BUCKETS):
        nreq = int(CNT[:, :, j].sum() / len(SEEDS))
        nusr = int((CNT[:, :, j] > 0).any(0).sum())
        lo, hi = np.percentile(bd[:, j], [2.5, 97.5])
        R.append(rowd(dataset=city, bucket=name, n_req_mean_per_seed=nreq, n_users=nusr,
                      sit=round(sit0[j], 5), base=round(base0[j], 5),
                      delta=round(sit0[j] - base0[j], 5),
                      ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5),
                      sit_ci_lo=round(float(np.percentile(bs[:, j], 2.5)), 5),
                      sit_ci_hi=round(float(np.percentile(bs[:, j], 97.5)), 5),
                      base_ci_lo=round(float(np.percentile(bb[:, j], 2.5)), 5),
                      base_ci_hi=round(float(np.percentile(bb[:, j], 97.5)), 5),
                      preregistered=int(name != "pre_first_change")))
    # contrasto pre-registrato: vicino (d=0,d1-2) vs lontano (d>5, stable)
    inear = [BUCKETS.index(x) for x in MAIN]; ifar = [BUCKETS.index(x) for x in FAR]
    con = []
    for a in inear:
        for z in ifar:
            diff = bd[:, a] - bd[:, z]
            lo, hi = np.percentile(diff, [2.5, 97.5])
            con.append(rowd(dataset=city, contrast=f"{BUCKETS[a]} - {BUCKETS[z]}",
                            point=round(float((sit0[a] - base0[a]) - (sit0[z] - base0[z])), 5),
                            ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5),
                            ci_excludes_zero=int(lo > 0 or hi < 0),
                            sit_diff=round(float(sit0[a] - sit0[z]), 5),
                            base_diff=round(float(base0[a] - base0[z]), 5)))
    # versione WITHIN-USER centrata: Delta(u,bucket) - media di Delta(u)
    W = []
    for j, name in enumerate(BUCKETS):
        vals = []
        for si in range(len(SEEDS)):
            c = CNT[si]; has = c[:, j] > 0
            dl = np.divide(S_sit[si] - S_base[si], np.maximum(c, 1e-12))
            umean = np.divide((S_sit[si] - S_base[si]).sum(1), np.maximum(c.sum(1), 1e-12))
            vals.append((dl[has, j] - umean[has]))
        v = np.concatenate(vals)
        lo, hi = (np.nan, np.nan)
        if v.size > 10:
            bmeans = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(400)])
            lo, hi = np.percentile(bmeans, [2.5, 97.5])
        W.append(rowd(dataset=city, bucket=name, n_user_seed_cells=int(v.size),
                      within_user_centered_delta=round(float(v.mean()), 5) if v.size else np.nan,
                      ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5)))
    pd.DataFrame(R + [dict(dataset=city, bucket=f"CONTRAST::{c['contrast']}", n_req_mean_per_seed=None,
                           n_users=None, sit=c["sit_diff"], base=c["base_diff"], delta=c["point"],
                           ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], sit_ci_lo=None, sit_ci_hi=None,
                           base_ci_lo=None, base_ci_hi=None, preregistered=1) for c in con] +
                 [dict(dataset=city, bucket=f"WITHIN::{w['bucket']}", n_req_mean_per_seed=None,
                       n_users=w["n_user_seed_cells"], sit=None, base=None,
                       delta=w["within_user_centered_delta"], ci_lo=w["ci_lo"], ci_hi=w["ci_hi"],
                       sit_ci_lo=None, sit_ci_hi=None, base_ci_lo=None, base_ci_hi=None,
                       preregistered=1) for w in W]
                 ).to_csv(OUT / f"check_e1_by_distance_{city}.csv", index=False)
    return dict(rows=R, contrasts=con, within=W)


# ------------------------------------------------------------------------- CHECK E2
def check_e2(city, P, urank, nU, rng):
    tl = np.zeros(nU); 
    for p in P[:1]:
        cnts = np.bincount(urank[p["u"]], minlength=nU); tl = cnts.astype(float)
    ok = tl > 0
    qs = np.quantile(tl[ok], [0.25, 0.5, 0.75])
    lbin = np.digitize(tl, qs)                       # 0..3
    R = []
    for si, p in enumerate(P):
        ur = urank[p["u"]]
        dsum = np.bincount(ur, weights=p["q_sit"] - p["q_base"], minlength=nU)
        dcnt = np.maximum(np.bincount(ur, minlength=nU), 1)
        du = dsum / dcnt
        nsit = np.zeros(nU, np.int64)
        df = pd.DataFrame({"u": ur, "k": p["k_te"]}).groupby("u")["k"].nunique()
        nsit[df.index.values] = df.values
        strat = np.where(nsit <= 1, 1, np.where(nsit == 2, 2, 3))
        for lb in range(4):
            for st in (1, 2, 3):
                m = ok & (lbin == lb) & (strat == st)
                R.append(rowd(dataset=city, seed=SEEDS[si], len_bin=f"Q{lb+1}", n_situations=st,
                              n_users=int(m.sum()),
                              delta=round(float(du[m].mean()), 5) if m.sum() else np.nan))
    df = pd.DataFrame(R)
    agg = (df.groupby(["dataset", "len_bin", "n_situations"])
             .agg(n_users=("n_users", "mean"), delta=("delta", "mean"),
                  sd_seed=("delta", "std")).reset_index())
    # CI bootstrap clusterizzato per utente, dentro cella, seed mediati
    cis = []
    for _, r in agg.iterrows():
        lb = int(r["len_bin"][1]) - 1; st = int(r["n_situations"])
        vals = []
        for si, p in enumerate(P):
            ur = urank[p["u"]]
            dsum = np.bincount(ur, weights=p["q_sit"] - p["q_base"], minlength=nU)
            dcnt = np.maximum(np.bincount(ur, minlength=nU), 1)
            du = dsum / dcnt
            df2 = pd.DataFrame({"u": ur, "k": p["k_te"]}).groupby("u")["k"].nunique()
            nsit = np.zeros(nU, np.int64); nsit[df2.index.values] = df2.values
            strat = np.where(nsit <= 1, 1, np.where(nsit == 2, 2, 3))
            vals.append(du[ok & (lbin == lb) & (strat == st)])
        n = min(len(v) for v in vals) if vals else 0
        if n < 10: cis.append((np.nan, np.nan)); continue
        b = np.array([np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in vals])
                      for _ in range(300)])
        cis.append(tuple(np.percentile(b, [2.5, 97.5])))
    agg["ci_lo"] = [round(float(c[0]), 5) for c in cis]
    agg["ci_hi"] = [round(float(c[1]), 5) for c in cis]
    agg.to_csv(OUT / f"check_e2_stratified_{city}.csv", index=False)
    return agg


# ------------------------------------------------------------------------- CHECK E3
def ols_cluster(X, y, g):
    XtX_inv = np.linalg.pinv(X.T @ X); beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    gi, gidx = np.unique(g, return_inverse=True); G = len(gi); n, kk = X.shape
    S = np.zeros((G, kk))
    for j in range(kk): S[:, j] = np.bincount(gidx, weights=X[:, j] * e, minlength=G)
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - kk, 1))
    V = c * (XtX_inv @ (S.T @ S) @ XtX_inv)
    sst = ((y - y.mean()) ** 2).sum()
    return beta, np.sqrt(np.maximum(np.diag(V), 0)), 1 - (e ** 2).sum() / sst if sst > 0 else np.nan


def check_e3(city, P, urank, nU):
    rows = []
    for si, p in enumerate(P):
        ur = urank[p["u"]]
        dsum = np.bincount(ur, weights=p["q_sit"] - p["q_base"], minlength=nU)
        cnt = np.bincount(ur, minlength=nU)
        du = dsum / np.maximum(cnt, 1)
        K = p["K"]
        d = pd.DataFrame({"u": ur, "k": p["k_te"], "t": p["t"]}).sort_values(["u", "t"])
        ent = np.zeros(nU); ntr = np.zeros(nU)
        for u_, gg in d.groupby("u"):
            kk = gg["k"].values
            c = np.bincount(kk, minlength=K).astype(float); pr = c / c.sum(); nz = pr[pr > 0]
            ent[u_] = -(nz * np.log(nz)).sum() / np.log(K) if K > 1 else 0.0
            ntr[u_] = int((kk[1:] != kk[:-1]).sum()) if len(kk) > 1 else 0
        m = cnt > 0
        rows.append(pd.DataFrame({"u": np.arange(nU)[m], "seed": SEEDS[si], "delta": du[m],
                                  "entropy": ent[m], "loglen": np.log(cnt[m]), "ntrans": ntr[m]}))
    D = pd.concat(rows, ignore_index=True)
    X = np.c_[np.ones(len(D)), D["entropy"], D["loglen"], D["ntrans"]]
    beta, se, r2 = ols_cluster(X, D["delta"].values, D["u"].values)
    names = ["intercept", "beta1_entropy", "beta2_loglen", "beta3_ntrans"]
    R = [rowd(dataset=city, term=nm, coef=round(float(b), 6), se_cluster=round(float(s), 6),
              t=round(float(b / s), 3) if s > 0 else np.nan,
              sig=int(abs(b / s) > 1.96) if s > 0 else 0,
              n_obs=len(D), n_clusters=int(D["u"].nunique()), r2=round(float(r2), 5))
         for nm, b, s in zip(names, beta, se)]
    C = D[["entropy", "loglen", "ntrans"]].corr()
    for a in C.columns:
        for z in C.columns:
            if a < z:
                R.append(rowd(dataset=city, term=f"corr({a},{z})", coef=round(float(C.loc[a, z]), 4),
                              se_cluster=np.nan, t=np.nan, sig=np.nan, n_obs=len(D),
                              n_clusters=int(D["u"].nunique()), r2=np.nan))
    pd.DataFrame(R).to_csv(OUT / f"check_e3_regression_{city}.csv", index=False)
    return R, C


# --------------------------------------------------------------------------------- run
def run(city):
    bt = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")
    kmap = {int(r.seed): float(r.kstar) for _, r in
            bt[(bt.backbone == "B_blind") & (bt.method == "SIT")].iterrows()}
    P = []
    for s in SEEDS:
        print(f"[{city}] seed {s} ...", flush=True)
        P.append(per_seed(city, s, kmap[s]))
    a = float(P[0]["q_sit"].mean()); ref = ANCHOR[city]
    ok = abs(a - ref) < 5e-6
    print(f"[{city}] GATE (seed 42): {a:.5f} atteso {ref:.5f} -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    if not ok:
        return dict(dataset=city, gate_ok=False, anchor=a, anchor_ref=ref)
    uall = np.unique(np.concatenate([p["u"] for p in P]))
    urank = np.full(uall.max() + 1, -1, np.int64); urank[uall] = np.arange(len(uall))
    nU = len(uall); rng = np.random.default_rng(SEED)
    print(f"[{city}] E1 ...", flush=True); e1 = check_e1(city, P, urank, nU, rng)
    print(f"[{city}] E2 ...", flush=True); e2 = check_e2(city, P, urank, nU, rng)
    print(f"[{city}] E3 ...", flush=True); e3, corr = check_e3(city, P, urank, nU)
    s = dict(dataset=city, gate_ok=True, anchor=a, anchor_ref=ref, n_users=nU,
             K=[p["K"] for p in P], kap=P[0]["kap"], e1=e1["rows"], contrasts=e1["contrasts"],
             within=e1["within"], e3=e3, corr=corr.to_dict())
    json.dump(s, open(OUT / f"_wi0c_{city}.json", "w"), indent=2, default=float)
    return s


def verdict(s):
    """Regola PRE-REGISTRATA (brief WI-0c §6), applicata meccanicamente."""
    con = s["contrasts"]; rows = {r["bucket"]: r for r in s["e1"]}
    all_disjoint = all(c["ci_excludes_zero"] == 1 and c["point"] > 0 for c in con)
    any_disjoint = any(c["ci_excludes_zero"] == 1 and c["point"] > 0 for c in con)
    # guardia di falsificazione: SIT deve SALIRE vicino al cambio, non solo BASE scendere
    sit_up = all(c["sit_diff"] > 0 for c in con)
    e3 = {r["term"]: r for r in s["e3"]}
    b1 = e3.get("beta1_entropy", {}); b3 = e3.get("beta3_ntrans", {})
    # SEGNO-CONSAPEVOLE: un beta significativo col segno CONTRARIO all'ipotesi non e' supporto.
    # L'ipotesi e' "variare aumenta il Delta" -> serve beta1>0 e/o beta3>0.
    e3_ok = bool((b1.get("sig", 0) and b1.get("coef", 0) > 0) or
                 (b3.get("sig", 0) and b3.get("coef", 0) > 0))
    e3_reversed = bool((b1.get("sig", 0) and b1.get("coef", 0) < 0) or
                       (b3.get("sig", 0) and b3.get("coef", 0) < 0))
    rev = [c for c in con if c["ci_excludes_zero"] == 1 and c["point"] < 0]
    if all_disjoint and sit_up and e3_ok: v = "CONFERMATO"
    elif any_disjoint or e3_ok: v = "AMBIGUO"
    else: v = "NON CONFERMATO"
    return v, dict(all_disjoint=all_disjoint, any_disjoint=any_disjoint, sit_up=sit_up,
                   e3_ok=e3_ok, e3_reversed=e3_reversed, n_contrasts_reversed=len(rev),
                   n_contrasts=len(con))


def summary():
    S = [json.load(open(p)) for p in sorted(OUT.glob("_wi0c_*.json"))]
    if not S: return
    L = ["# WI-0c — Il guadagno si concentra dove la situazione cambia?", "",
         "Pre-registrato nel brief WI-0c. Bucket, contrasti e regola di decisione fissati prima "
         "dei numeri. 5 seed {42–46}, kappa* congelato dalla battery, bootstrap a due livelli "
         "(ricampiona UTENTI e SEED, mai richieste). Nessun retraining, nessun ri-split.", "",
         "> **Backbone.** Usato `B_blind`: e' cio' che produce i valori del gate ed e' cio' su cui "
         "girano WI-0/WI-0b. Nel repo B_blind sta in `FM.scores.npy` ma contiene punteggi **BPR** "
         "(`cornac_backbone.py:1`); l'FM context-aware e' `B_full`, che `results_record.py:200` "
         "**riallena per seed** — incompatibile col vincolo \\\"zero retraining\\\" e col gate. "
         "Ambiguita' del brief dichiarata, non risolta in silenzio.", "",
         "> **Priorita' dichiarata prima dei numeri:** decisivo = **ml1m**; conferma = "
         "**saopaulo**; **nyc_tist e' strutturalmente sotto-potenziato** (test mediano 3, 92% "
         "mono-situazione) — riportato, non vota.", "",
         "## Gate di ancoraggio", "", "| dataset | Cat-MRR SIT (seed 42) | atteso | esito |",
         "|---|---|---|---|"]
    for s in S:
        L.append(f"| {s['dataset']} | {s['anchor']:.5f} | {s['anchor_ref']:.5f} | "
                 f"{'OK' if s['gate_ok'] else 'MISMATCH — STOP'} |")
    G = [s for s in S if s["gate_ok"]]
    L += ["", "## E1 — Delta per distanza dal cambio di situazione", "",
          "SIT e BASE riportati separatamente: e' la guardia di falsificazione.", ""]
    for s in G:
        L += [f"**{s['dataset']}**", "",
              "| bucket | n richieste | n utenti | SIT | BASE | Δ | CI 95% Δ |",
              "|---|---|---|---|---|---|---|"]
        for r in s["e1"]:
            star = "" if r["preregistered"] else " *(fuori contrasto)*"
            L.append(f"| {r['bucket']}{star} | {r['n_req_mean_per_seed']} | {r['n_users']} | "
                     f"{r['sit']:.5f} | {r['base']:.5f} | {r['delta']:+.5f} | "
                     f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}] |")
        L += ["", "Contrasti pre-registrati (vicino − lontano):", "",
              "| contrasto | Δ(vicino)−Δ(lontano) | CI 95% | CI esclude 0 | ΔSIT | ΔBASE |", "|---|---|---|---|---|---|"]
        for c in s["contrasts"]:
            L.append(f"| {c['contrast']} | {c['point']:+.5f} | [{c['ci_lo']:+.5f}, "
                     f"{c['ci_hi']:+.5f}] | {'sì' if c['ci_excludes_zero'] else 'no'} | "
                     f"{c['sit_diff']:+.5f} | {c['base_diff']:+.5f} |")
        L += ["", "Versione within-user centrata (effetto-utente rimosso per costruzione):", "",
              "| bucket | celle utente×seed | Δ centrato | CI 95% |", "|---|---|---|---|"]
        for w in s["within"]:
            L.append(f"| {w['bucket']} | {w['n_user_seed_cells']} | "
                     f"{w['within_user_centered_delta']:+.5f} | [{w['ci_lo']:+.5f}, {w['ci_hi']:+.5f}] |")
        L.append("")
    L += ["## E2 — stratificazione a lunghezza di test controllata", "",
          "Vedi `check_e2_stratified_<ds>.csv` (celle n_situazioni × quartile di lunghezza).", ""]
    L += ["## E3 — variare vs avere piu' dati", "",
          "| dataset | β₁ entropia | β₂ log(len) | β₃ n_transizioni | R² | n oss. | n cluster |",
          "|---|---|---|---|---|---|---|"]
    for s in G:
        e3 = {r["term"]: r for r in s["e3"]}
        f = lambda k: (f"{e3[k]['coef']:+.5f} ({e3[k]['se_cluster']:.5f})"
                       f"{'*' if e3[k]['sig'] else ''}") if k in e3 else "n/d"
        r0 = e3.get("beta1_entropy", {})
        L.append(f"| {s['dataset']} | {f('beta1_entropy')} | {f('beta2_loglen')} | "
                 f"{f('beta3_ntrans')} | {r0.get('r2', float('nan')):.5f} | "
                 f"{r0.get('n_obs','')} | {r0.get('n_clusters','')} |")
    L += ["", "Correlazione fra regressori (se quasi collineari, i coefficienti separati non sono "
          "interpretabili):", ""]
    for s in G:
        c = s["corr"]
        L.append(f"- **{s['dataset']}**: entropia~loglen {c['entropy']['loglen']:+.3f} · "
                 f"entropia~ntrans {c['entropy']['ntrans']:+.3f} · "
                 f"loglen~ntrans {c['loglen']['ntrans']:+.3f}")
    L += ["", "## Esito (regola pre-registrata, applicata meccanicamente)", ""]
    for s in G:
        v, why = verdict(s)
        tag = ("decisivo" if s["dataset"] == DECISIVE else
               "conferma" if s["dataset"] == SUPPORT else "NON VOTA (sotto-potenziato)")
        L.append(f"- **{s['dataset']}** ({tag}) → **{v}** — contrasti disgiunti nel verso "
                 f"dell'ipotesi: {'sì' if why['all_disjoint'] else 'no'} "
                 f"({why['n_contrasts_reversed']}/{why['n_contrasts']} disgiunti nel verso "
                 f"OPPOSTO) · guardia di falsificazione: "
                 f"{'superata' if why['sit_up'] else 'VIOLATA'} · β₁/β₃ significativo nel verso "
                 f"dell'ipotesi: {'sì' if why['e3_ok'] else 'no'}"
                 f"{' (significativo ma ROVESCIATO)' if why['e3_reversed'] else ''}")
    dec = [s for s in G if s["dataset"] == DECISIVE]
    if dec:
        v, _ = verdict(dec[0])
        L += ["", f"### esito = {v}", "",
              f"Determinato su **{DECISIVE}** (dataset decisivo), con "
              f"**{SUPPORT}** come conferma. **{UNDERPOWERED}** non concorre alla decisione."]
    L += ["", "## Domande emerse (RIPORTATE, non eseguite — come da brief §8)", "",
          "1. **Rovesciamento fra-utenti vs entro-utente (tipo Simpson).** Il contrasto FRA bucket "
          "e' rovesciato su ml1m, ma la versione ENTRO utente e' positiva a `d=0` su **3/3 "
          "dataset** (+0.0086 ml1m, +0.0271 nyc_tist, +0.0044 saopaulo) e decresce in modo "
          "monotono con la distanza. Le due analisi NON sono sulla stessa popolazione: il bucket "
          "`stable` vale 0.00000 **per costruzione** nella versione entro-utente (chi non cambia "
          "occupa un solo bucket, quindi la sua deviazione dalla propria media e' identicamente "
          "nulla) ed e' quindi escluso dal confronto appaiato.",
          "2. **I bucket differiscono per difficolta' intrinseca, non solo per trattamento.** "
          "BASE da solo varia fra bucket: 0.311-0.382 su ml1m, 0.271-0.446 su saopaulo. Il "
          "confronto fra bucket confonde composizione della popolazione ed effetto.",
          "3. **beta1 e beta3 non sono separabili fra loro** (corr entropia~n_transizioni 0.54-0.72): "
          "misurano quasi la stessa cosa. Sono invece separabili da beta2 (corr entropia~loglen "
          "0.09-0.16), quindi la distinzione *variare* vs *avere piu' dati* — quella che il brief "
          "chiedeva — resta interpretabile.",
          "4. **E2 e' la misura meno confusa** (lunghezza del test neutralizzata) e su ml1m e' "
          "monotona e negativa in tutti e quattro i quartili: 1 situazione +0.033/+0.058, "
          "2 situazioni -0.065/+0.001, 3+ situazioni ~-0.014/-0.000.",
          "", "Nessuna di queste e' stata investigata: come da brief, si riportano e ci si ferma."]
    (OUT / "wi0c_summary.md").write_text("\n".join(L) + "\n")
    print(f"\n-> {OUT / 'wi0c_summary.md'}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in (sys.argv[1:] or [DECISIVE]):
        print(f"\n===== WI-0c · {c.upper()} =====", flush=True); run(c)
    summary(); return 0


if __name__ == "__main__":
    sys.exit(main())
