"""[venv-xsage] WI-0 — VIABILITY PROBE: tre check go/no-go su artefatti gia' in cache.

NESSUN riaddestramento, NESSUNA modifica al metodo, NESSUN tocco a data/.
Ground truth = percorso BATTERY (s_hat = s_B + k*gamma*b_tilde[cat]), cioe' eval_kappa.cat_mrr.
Impianto comune replicato da scripts/yelp/macro_avg.py (che NON viene modificato).

  CHECK A  l'effetto-situazione sopravvive al controllo per utente?
           componenti di varianza (ANOVA sbilanciata, stimatore di Searle) + SS sequenziali
           in ENTRAMBI gli ordini + spread intra-utente + matrice segmento x situazione.
  CHECK B  il segnale situazionale e' separabile dalla preferenza statica?
           R2/coseno b_sit ~ b_user, flip-rate argmax e top-1, identificabilita' vs contesto.
  CHECK C  la situazione e' un'etichetta persistente?
           entropia/dominanza per utente, persistenza temporale, baseline demografica ml1m.

Il probe RIPORTA NUMERI. Nessun giudizio di merito: le decisioni sono pre-registrate.

Uso:  python -m scripts.diagnostics.viability_probe [ml1m ...]
Out:  outputs_results/diagnostics/check{A,B,C}_*_<ds>.csv + WI0_SUMMARY.md
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
from eval_kappa import BATCH, KAPPA_GRID, SEED, cat_mrr, select_K, select_eps  # noqa: E402
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage.recommendation import fit_situation_biases_z                        # noqa: E402
from sklearn.linear_model import LinearRegression                              # noqa: E402

OUT = CLEAN / "outputs_results" / "diagnostics"
MIN_CELL = 20          # sotto questa soglia la cella e' marcata low_support (mai mediata in silenzio)
MIN_SEQ = 5            # richieste di test minime per entrare nell'analisi di sequenza (CHECK C)
UGF_TOP = 0.05         # split active/inactive alla Li et al. 2021 (top 5% per attivita')


# ----------------------------------------------------------------------------- util
def _rows(dataset, check, block, key, subkey, metric, value, n=None, low=0):
    return dict(dataset=dataset, check=check, block=block, key=key, subkey=subkey,
                metric=metric, value=value, n=n, low_support=low)


def quart(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0: return (np.nan,) * 3
    return tuple(float(x) for x in np.percentile(a, [25, 50, 75]))


def top1_macro(sb_full, df, dmac, gamma, icm, excl, kappa):
    """Macro-categoria dell'item in TOP-1. Stesso batching/mascheramento di eval_kappa.cat_mrr."""
    u = df["u_idx"].values.astype(np.int64); n = len(df)
    out = np.zeros(n, np.int64)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        S = sb_full[u_b].astype(np.float32, copy=True)
        if dmac is not None:
            S = S + kappa * gamma[bs:be][:, None].astype(np.float32) * dmac[bs:be][:, icm]
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        out[bs:be] = icm[np.argmax(S, axis=1)]
    return out


# ------------------------------------------------------------------- componenti varianza
def oneway(y, g):
    """ANOVA a una via su disegno SBILANCIATO. sigma2_between con il correttore di Searle
    n0 = (N - sum(n_i^2)/N)/(G-1); ICC = s2_b/(s2_b+s2_e). eta2 = quota di SS spiegata."""
    _, inv = np.unique(g, return_inverse=True)
    G = int(inv.max()) + 1; N = len(y)
    cnt = np.bincount(inv, minlength=G).astype(np.float64)
    grand = float(y.mean())
    gmean = np.bincount(inv, weights=y, minlength=G) / np.maximum(cnt, 1)
    ss_b = float((cnt * (gmean - grand) ** 2).sum())
    ss_t = float(((y - grand) ** 2).sum())
    ss_w = ss_t - ss_b
    df_b, df_w = G - 1, N - G
    ms_b = ss_b / df_b if df_b > 0 else np.nan
    ms_w = ss_w / df_w if df_w > 0 else np.nan
    n0 = (N - (cnt ** 2).sum() / N) / df_b if df_b > 0 else np.nan
    s2_b = max((ms_b - ms_w) / n0, 0.0) if (n0 and n0 > 0 and np.isfinite(ms_b)) else np.nan
    icc = s2_b / (s2_b + ms_w) if np.isfinite(s2_b) and (s2_b + ms_w) > 0 else np.nan
    return dict(G=G, N=N, ss_between=ss_b, ss_total=ss_t, eta2=ss_b / ss_t if ss_t > 0 else np.nan,
                sigma2_between=s2_b, sigma2_resid=ms_w, icc=icc, n0=n0)


def additive_sse(y, ga, gb, iters=3000, tol=1e-13):
    """SSE del modello additivo y ~ a[ga] + b[gb], via Gauss-Seidel sulle equazioni normali.
    NIENTE centratura e niente intercetta fissa: l'ambiguita' additiva (a+c, b-c) non tocca i
    valori fittati. Partendo da b=0 la prima mezza-iterazione riproduce ESATTAMENTE il modello
    a sola A, quindi la SSE decresce in modo monotono e resta <= min(SSE_A, SSE_B).
    (La versione precedente, con mu fisso e centratura NON pesata, non era la proiezione ai
    minimi quadrati su disegno sbilanciato e poteva restituire SS incrementali negative.)"""
    _, ia = np.unique(ga, return_inverse=True); na = int(ia.max()) + 1
    _, ib = np.unique(gb, return_inverse=True); nb = int(ib.max()) + 1
    ca = np.maximum(np.bincount(ia, minlength=na).astype(np.float64), 1)
    cb = np.maximum(np.bincount(ib, minlength=nb).astype(np.float64), 1)
    a = np.zeros(na); b = np.zeros(nb); prev = np.inf; sse = np.inf
    for _ in range(iters):
        a = np.bincount(ia, weights=y - b[ib], minlength=na) / ca
        b = np.bincount(ib, weights=y - a[ia], minlength=nb) / cb
        sse = float(((y - a[ia] - b[ib]) ** 2).sum())
        if prev - sse < tol * max(abs(prev), 1.0): break
        prev = sse
    return sse


# ------------------------------------------------------------------------ impianto comune
def build_common(city):
    print(f"[{city}] descrittore + selezione K/eps (replica macro_avg.py)...", flush=True)
    rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]
    sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng)
    eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)

    def assign(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps)
        mem = membership_from_assign(k, comp, isb, K)
        gam = 1.0 / np.maximum(comp.sum(1), 1).astype(np.float32)
        return mem, gam, k.astype(np.int64), isb
    mem_va, gam_va, k_va, _ = assign(vva)
    mem_te, gam_te, k_te, isb_te = assign(vte)
    _, k_tr, _, _ = _assign(vtr, fit.prototypes, eps); k_tr = k_tr.astype(np.int64)

    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    n_users = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    dfv, dft = ds["df_val"], ds["df_test"]
    uv = dfv["u_idx"].values.astype(np.int64); ute = dft["u_idx"].values.astype(np.int64)
    dmac = {"SIT": (mem_va.astype(np.float32) @ b_z, mem_te.astype(np.float32) @ b_z),
            "Steck-b": (b_z_user[uv], b_z_user[ute])}

    print(f"[{city}] K={K} eps={eps}; selezione kappa* su VALIDATION...", flush=True)
    kstar = {}
    for m, (dv, _) in dmac.items():
        best = (-1.0, None)
        for kap in KAPPA_GRID:
            vm = float(cat_mrr(sb, dfv, dv, gam_va, icm, excl, kap).mean())
            if vm > best[0]: best = (vm, kap)
        kstar[m] = best[1]

    print(f"[{city}] kappa*: SIT={kstar['SIT']} Steck-b={kstar['Steck-b']}; scoring TEST...", flush=True)
    q_sit = cat_mrr(sb, dft, dmac["SIT"][1], gam_te, icm, excl, kstar["SIT"])
    q_base = cat_mrr(sb, dft, None, gam_te, icm, excl, 0.0)
    return dict(city=city, ds=ds, dft=dft, dfv=dfv, nmac=nmac, icm=icm, excl=excl, sb=sb,
                K=K, eps=eps, vtr=vtr, vva=vva, k_tr=k_tr, k_va=k_va,
                vte=vte, mem_te=mem_te, gam_te=gam_te, k_te=k_te, isb_te=isb_te,
                b_z=b_z, b_z_user=b_z_user, ute=ute, kstar=kstar,
                nudge_te=dmac["SIT"][1], stb_te=dmac["Steck-b"][1],
                q_sit=q_sit, q_base=q_base, n_users=n_users)


# ------------------------------------------------------------------------------- CHECK A
def check_A(C):
    city = C["city"]; q = C["q_sit"].astype(np.float64); u = C["ute"]; k = C["k_te"]
    K = C["K"]; R = []
    print(f"[{city}] CHECK A — componenti di varianza...", flush=True)

    ou, ok = oneway(q, u), oneway(q, k)
    sse_u = ou["ss_total"] - ou["ss_between"]
    sse_k = ok["ss_total"] - ok["ss_between"]
    sse_uk = additive_sse(q, u, k)
    ss_t = ou["ss_total"]
    # guardia: la SSE additiva non puo' superare quella dei modelli a un solo fattore
    slack = 1e-9 * max(ss_t, 1.0)
    assert sse_uk <= sse_u + slack and sse_uk <= sse_k + slack, (
        f"backfitting non convergente: sse_uk={sse_uk} sse_u={sse_u} sse_k={sse_k}")
    seq = {"ss_user_first": ou["ss_between"] / ss_t,
           "ss_sit_given_user": (sse_u - sse_uk) / ss_t,
           "ss_sit_first": ok["ss_between"] / ss_t,
           "ss_user_given_sit": (sse_k - sse_uk) / ss_t,
           "ss_resid": sse_uk / ss_t}
    for fac, o in (("user", ou), ("situation", ok)):
        for met in ("eta2", "icc", "sigma2_between", "sigma2_resid", "G", "n0"):
            R.append(_rows(city, "A", "variance_components", fac, "", met, float(o[met]), o["N"]))
    for met, v in seq.items():
        R.append(_rows(city, "A", "variance_components", "twoway", "", met, float(v), len(q)))
    R.append(_rows(city, "A", "variance_components", "twoway", "", "K", K, len(q)))

    # A.2 — spread intra-utente sulle celle utente x situazione effettivamente osservate
    print(f"[{city}] CHECK A — spread intra-utente...", flush=True)
    d = pd.DataFrame({"u": u, "k": k, "q": q})
    cell = d.groupby(["u", "k"], sort=False)["q"].mean().reset_index()
    per_u = cell.groupby("u")["q"].agg(["max", "min", "count"])
    multi = per_u[per_u["count"] >= 2]
    spread = (multi["max"] - multi["min"]).values
    n_u = int(per_u.shape[0]); n_multi = int(multi.shape[0])
    q1, med, q3 = quart(spread)
    for met, v, nn in (("share_users_multi_situation", n_multi / n_u if n_u else np.nan, n_u),
                       ("spread_p25", q1, n_multi), ("spread_median", med, n_multi),
                       ("spread_p75", q3, n_multi), ("spread_mean", float(np.mean(spread)) if spread.size else np.nan, n_multi),
                       ("n_users_test", n_u, n_u), ("n_users_multi", n_multi, n_multi)):
        R.append(_rows(city, "A", "user_spread", met, "", "value", float(v), nn))

    # A.3 — matrice segmento x situazione, due segmentazioni
    print(f"[{city}] CHECK A — matrice segmento x situazione...", flush=True)
    tr_cnt = C["ds"]["df_train"].groupby("u_idx").size()
    act = tr_cnt.reindex(np.arange(C["n_users"]), fill_value=0).values.astype(np.float64)
    a_u = act[u]
    qs = np.quantile(act[act > 0], [0.2, 0.4, 0.6, 0.8]) if (act > 0).any() else np.zeros(4)
    seg_quint = np.searchsorted(qs, a_u, side="right")
    thr = np.quantile(act[act > 0], 1.0 - UGF_TOP) if (act > 0).any() else np.inf
    seg_ugf = np.where(a_u >= thr, 1, 0)
    for name, seg, lab in (("activity_quintile", seg_quint, [f"Q{i+1}" for i in range(5)]),
                           ("ugf", seg_ugf, ["inactive", "active"])):
        dd = pd.DataFrame({"s": seg, "k": k, "q": q})
        g = dd.groupby(["s", "k"])["q"].agg(["mean", "size"]).reset_index()
        for _, row in g.iterrows():
            nn = int(row["size"]); low = int(nn < MIN_CELL)
            R.append(_rows(city, "A", "segment_matrix", name, lab[int(row["s"])],
                           f"sit_{int(row['k'])}", float(row["mean"]), nn, low))
        ok_cells = g[g["size"] >= MIN_CELL]
        for s in sorted(dd["s"].unique()):
            sub = ok_cells[ok_cells["s"] == s]
            worst = int(sub.loc[sub["mean"].idxmin(), "k"]) if len(sub) else -1
            R.append(_rows(city, "A", "worst_situation", name, lab[int(s)], "worst_sit_k",
                           worst, int((dd["s"] == s).sum()), int(len(sub) == 0)))
    pd.DataFrame(R).to_csv(OUT / f"checkA_variance_{city}.csv", index=False)
    return {"ss_sit_first": seq["ss_sit_first"], "ss_sit_given_user": seq["ss_sit_given_user"],
            "ss_user_first": seq["ss_user_first"], "ss_user_given_sit": seq["ss_user_given_sit"],
            "icc_user": ou["icc"], "icc_sit": ok["icc"],
            "share_users_multi": n_multi / n_u if n_u else np.nan, "spread_median": med,
            "n_req": len(q), "n_users": n_u}


# ------------------------------------------------------------------------------- CHECK B
def check_B(C):
    city = C["city"]; nmac = C["nmac"]; R = []
    x = C["b_z_user"][C["ute"]].astype(np.float64)      # (n_req, nmac) preferenza statica utente
    y = C["b_z"][C["k_te"]].astype(np.float64)          # (n_req, nmac) bias della situazione (hard)
    print(f"[{city}] CHECK B — ridondanza situazione ~ utente...", flush=True)
    xm = x.mean(1, keepdims=True); ym = y.mean(1, keepdims=True)
    xc = x - xm; yc = y - ym
    cov = (xc * yc).sum(1); vx = (xc ** 2).sum(1); vy = (yc ** 2).sum(1)
    ok = (vx > 1e-12) & (vy > 1e-12)
    r2 = np.full(len(x), np.nan); r2[ok] = cov[ok] ** 2 / (vx[ok] * vy[ok])
    nx = np.linalg.norm(x, axis=1); ny = np.linalg.norm(y, axis=1)
    okc = (nx > 1e-12) & (ny > 1e-12)
    cos = np.full(len(x), np.nan); cos[okc] = (x * y).sum(1)[okc] / (nx[okc] * ny[okc])
    q1, med, q3 = quart(r2[ok]); c1, cm_, c3 = quart(cos[okc])
    # aggregato: un solo OLS su tutte le coppie (richiesta, macro) impilate
    xf = xc[ok].ravel(); yf = yc[ok].ravel()
    r2_pool = float((xf @ yf) ** 2 / ((xf @ xf) * (yf @ yf))) if xf.size else np.nan
    for met, v, nn in (("r2_p25", q1, int(ok.sum())), ("r2_median", med, int(ok.sum())),
                       ("r2_p75", q3, int(ok.sum())), ("r2_pooled", r2_pool, int(ok.sum())),
                       ("cos_p25", c1, int(okc.sum())), ("cos_median", cm_, int(okc.sum())),
                       ("cos_p75", c3, int(okc.sum())), ("n_macros", nmac, len(x))):
        R.append(_rows(city, "B", "redundancy", met, "", "value", float(v), nn))

    print(f"[{city}] CHECK B — flip-rate...", flush=True)
    flip_arg = float((y.argmax(1) != x.argmax(1)).mean())
    t1_base = top1_macro(C["sb"], C["dft"], None, C["gam_te"], C["icm"], C["excl"], 0.0)
    t1_sit = top1_macro(C["sb"], C["dft"], C["nudge_te"], C["gam_te"], C["icm"], C["excl"], C["kstar"]["SIT"])
    flip_top1 = float((t1_sit != t1_base).mean())
    for met, v in (("flip_argmax_sit_vs_user", flip_arg), ("flip_top1_cat_sit_vs_base", flip_top1)):
        R.append(_rows(city, "B", "flip_rate", met, "", "value", v, len(x)))

    print(f"[{city}] CHECK B — identificabilita' vs contesto...", flush=True)
    A = C["vte"].shape[1] - nmac                        # blocco contesto c_tilde
    ctx = C["vte"][:, :A].astype(np.float64)
    tm = C["icm"][C["dft"]["i_idx"].values.astype(np.int64)]
    nud_true = C["nudge_te"][np.arange(len(tm)), tm].astype(np.float64)
    nud_norm = np.linalg.norm(C["nudge_te"].astype(np.float64), axis=1)
    r2_ctx = float(LinearRegression().fit(ctx, nud_true).score(ctx, nud_true))
    R.append(_rows(city, "B", "identifiability", "r2_nudge_true_on_ctx", "", "value", r2_ctx, len(tm)))
    R.append(_rows(city, "B", "identifiability", "n_ctx_cols", "", "value", A, len(tm)))
    r2_norm = float(LinearRegression().fit(ctx, nud_norm).score(ctx, nud_norm))
    R.append(_rows(city, "B", "identifiability", "r2_nudge_norm_on_ctx", "", "value", r2_norm, len(tm)))
    for j in range(A):
        cj = ctx[:, j]
        cc = float(np.corrcoef(cj, nud_norm)[0, 1]) if cj.std() > 1e-12 else np.nan
        R.append(_rows(city, "B", "identifiability", "pearson_ctx_vs_nudge_norm", f"c{j}", "value", cc, len(tm)))
    pd.DataFrame(R).to_csv(OUT / f"checkB_identifiability_{city}.csv", index=False)
    return {"r2_median": med, "r2_pooled": r2_pool, "cos_median": cm_,
            "flip_argmax": flip_arg, "flip_top1": flip_top1, "r2_ctx": r2_ctx, "n_req": len(x)}


# ------------------------------------------------------------------------------- CHECK C
def ml1m_demographics():
    """Ricostruisce u_idx replicando preprocess_ml1m.py:20-52 (k-core=10 sui film con genere,
    poi sorted() LESSICOGRAFICO degli id-utente superstiti) e unisce users.dat."""
    ML = CLEAN / "data" / "ml-1m"
    gen = {}
    for ln in open(ML / "movies.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 3:
            g = p[2].split("|")[0].strip()
            if g and g != "(no genres listed)": gen[p[0]] = g
    rows = []
    for ln in open(ML / "ratings.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 4 and p[1] in gen: rows.append((p[0], p[1]))
    df = pd.DataFrame(rows, columns=["user", "item"])
    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        m = df.user.isin(uc[uc >= 10].index) & df.item.isin(ic[ic >= 10].index)
        if m.all(): break
        df = df[m]
    uu = {us: i for i, us in enumerate(sorted(df.user.unique()))}
    dem = {}
    for ln in open(ML / "users.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 4: dem[p[0]] = (p[1], int(p[2]), int(p[3]))
    out = [(i, dem[us][0], dem[us][1], dem[us][2]) for us, i in uu.items() if us in dem]
    return pd.DataFrame(out, columns=["u_idx", "gender", "age", "occupation"])


def check_C(C):
    city = C["city"]; K = C["K"]; R = []
    print(f"[{city}] CHECK C — entropia e dominanza per utente...", flush=True)
    dft = C["dft"]
    d = pd.DataFrame({"u": C["ute"], "k": C["k_te"], "t": dft["time_local"].values})
    cnt = d.groupby("u").size()
    keep = cnt[cnt >= MIN_SEQ].index
    sub = d[d["u"].isin(keep)]
    ent, dom, dis = [], [], []
    for _, g in sub.groupby("u"):
        c = np.bincount(g["k"].values, minlength=K).astype(np.float64)
        p = c / c.sum(); nz = p[p > 0]
        ent.append(float(-(nz * np.log(nz)).sum() / np.log(K)) if K > 1 else 0.0)
        dom.append(float(p.max())); dis.append(int((c > 0).sum()))
    ent = np.array(ent); dom = np.array(dom); dis = np.array(dis, dtype=float)
    n_seq = len(ent)
    for name, arr in (("entropy_norm", ent), ("dominant_share", dom), ("n_distinct_sit", dis)):
        q1, med, q3 = quart(arr)
        for met, v in (("p25", q1), ("median", med), ("p75", q3),
                       ("mean", float(arr.mean()) if arr.size else np.nan)):
            R.append(_rows(city, "C", "sequence", name, met, "value", float(v), n_seq))
    share80 = float((dom > 0.80).mean()) if n_seq else np.nan
    R.append(_rows(city, "C", "sequence", "share_users_dominant_gt80", "", "value", share80, n_seq))
    R.append(_rows(city, "C", "sequence", "n_users_ge5_req", "", "value", n_seq, n_seq))
    R.append(_rows(city, "C", "sequence", "K", "", "value", K, n_seq))

    print(f"[{city}] CHECK C — persistenza temporale...", flush=True)
    s = sub.sort_values(["u", "t"], kind="stable")
    ku = s["k"].values; uu_ = s["u"].values
    same_u = uu_[1:] == uu_[:-1]
    pers = float((ku[1:][same_u] == ku[:-1][same_u]).mean()) if same_u.any() else np.nan
    R.append(_rows(city, "C", "persistence", "P_same_situation_t_to_t1_TEST", "", "value",
                   pers, int(same_u.sum())))
    tj = CLEAN / "outputs_results" / "explain" / f"situation_transitions_{city}.json"
    if tj.exists():
        T = np.array(json.load(open(tj))["T"], dtype=np.float64)
        sizes = np.bincount(C["k_te"], minlength=len(T)).astype(np.float64)
        w = sizes / sizes.sum() if sizes.sum() else np.zeros(len(T))
        diag_ref = float((np.diag(T) * w[:len(T)]).sum())
        R.append(_rows(city, "C", "persistence", "diag_weighted_TRAIN_reference", "", "value",
                       diag_ref, len(T)))
        for i in range(len(T)):
            R.append(_rows(city, "C", "persistence", "diag_TRAIN", f"sit_{i}", "value",
                           float(T[i, i]), int(sizes[i]) if i < len(sizes) else 0))

    maj = {}
    if city == "ml1m":
        print(f"[{city}] CHECK C — sanity demografica...", flush=True)
        dm = ml1m_demographics()
        R.append(_rows(city, "C", "demographics", "n_users_reconstructed", "", "value",
                       len(dm), C["n_users"], int(len(dm) != C["n_users"])))
        for col in ("gender", "age", "occupation"):
            vc = dm[col].value_counts(normalize=True).sort_values(ascending=False)
            maj[col] = float(vc.iloc[0])
            R.append(_rows(city, "C", "demographics", f"{col}_majority_baseline", str(vc.index[0]),
                           "value", float(vc.iloc[0]), len(dm)))
            R.append(_rows(city, "C", "demographics", f"{col}_n_classes", "", "value",
                           int(dm[col].nunique()), len(dm)))
            for lev, sh in vc.items():
                R.append(_rows(city, "C", "demographics", f"{col}_distribution", str(lev),
                               "share", float(sh), int(round(sh * len(dm)))))
    pd.DataFrame(R).to_csv(OUT / f"checkC_stability_{city}.csv", index=False)
    return {"entropy_median": float(np.median(ent)) if n_seq else np.nan,
            "dominant_median": float(np.median(dom)) if n_seq else np.nan,
            "share_dom_gt80": share80, "persistence_test": pers,
            "n_users_ge5": n_seq, "majority": maj}


# ---------------------------------------------------------------------------------- main
def run(city):
    C = build_common(city)
    anchor = float(C["q_sit"].mean())
    mv = CLEAN / "outputs_results" / f"macro_avg_{city}.csv"
    ref = np.nan
    if mv.exists():
        t = pd.read_csv(mv); r = t[t.method == "SIT"]["micro"]
        if len(r): ref = float(r.iloc[0])
    ok = bool(np.isfinite(ref) and abs(anchor - ref) < 5e-4)
    print(f"\n[{city}] ANCORAGGIO: Cat-MRR micro SIT@k* = {anchor:.5f}  "
          f"(macro_avg_{city}.csv = {ref:.5f})  -> {'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        print(f"[{city}] ATTENZIONE: il clustering non e' stato replicato; i numeri sotto NON "
              f"sono confrontabili con la batteria.", flush=True)

    A = check_A(C); B = check_B(C)
    Cc = check_C(C) if city == "ml1m" else None
    s = dict(dataset=city, K=C["K"], eps=C["eps"], kstar_SIT=C["kstar"]["SIT"],
             anchor_catmrr_sit=anchor, anchor_ref=ref, anchor_ok=ok,
             base_catmrr=float(C["q_base"].mean()), A=A, B=B, C=Cc)
    json.dump(s, open(OUT / f"_wi0_{city}.json", "w"), indent=2, default=float)
    return s


def write_summary():
    js = sorted(OUT.glob("_wi0_*.json"))
    if not js: return
    S = [json.load(open(p)) for p in js]
    L = ["# WI-0 — Viability probe: risultati", "",
         "Numeri prodotti da `scripts/diagnostics/viability_probe.py` su artefatti in cache "
         "(seed 42, percorso battery, nessun riaddestramento). **Nessun giudizio di merito.**", ""]
    L += ["## Ancoraggio (gate di correttezza)", "",
          "| dataset | K | eps | kappa* | Cat-MRR micro SIT | riferimento macro_avg | esito |",
          "|---|---|---|---|---|---|---|"]
    for s in S:
        L.append(f"| {s['dataset']} | {s['K']} | {s['eps']} | {s['kstar_SIT']} | "
                 f"{s['anchor_catmrr_sit']:.5f} | {s['anchor_ref']:.5f} | "
                 f"{'OK' if s['anchor_ok'] else 'MISMATCH'} |")
    L += ["", "## A — quota di varianza (SS sequenziali, entrambi gli ordini)", "",
          "| dataset | sit per prima | sit dato utente | utente per primo | utente data sit | "
          "ICC utente | ICC sit | n richieste | n utenti |", "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        a = s["A"]
        L.append(f"| {s['dataset']} | {a['ss_sit_first']:.1%} | {a['ss_sit_given_user']:.1%} | "
                 f"{a['ss_user_first']:.1%} | {a['ss_user_given_sit']:.1%} | {a['icc_user']:.4f} | "
                 f"{a['icc_sit']:.4f} | {a['n_req']} | {a['n_users']} |")
    L += ["", "| dataset | utenti con >=2 situazioni | spread mediano intra-utente |", "|---|---|---|"]
    for s in S:
        a = s["A"]
        L.append(f"| {s['dataset']} | {a['share_users_multi']:.1%} | {a['spread_median']:.5f} |")
    L += ["", "## B — separabilita' dalla preferenza statica", "",
          "| dataset | R2 mediano sit~utente | R2 aggregato | coseno mediano | flip argmax | "
          "flip top-1 | R2 nudge~contesto |", "|---|---|---|---|---|---|---|"]
    for s in S:
        b = s["B"]
        L.append(f"| {s['dataset']} | {b['r2_median']:.4f} | {b['r2_pooled']:.4f} | "
                 f"{b['cos_median']:+.4f} | {b['flip_argmax']:.1%} | {b['flip_top1']:.1%} | "
                 f"{b['r2_ctx']:.4f} |")
    cs = [s for s in S if s.get("C")]
    if cs:
        L += ["", "## C — persistenza dell'etichetta", "",
              "| dataset | entropia mediana | dominante mediana | % utenti dominante >80% | "
              "P(z_t+1=z_t) test | n utenti >=5 req |", "|---|---|---|---|---|---|"]
        for s in cs:
            c = s["C"]
            L.append(f"| {s['dataset']} | {c['entropy_median']:.4f} | {c['dominant_median']:.4f} | "
                     f"{c['share_dom_gt80']:.1%} | {c['persistence_test']:.4f} | {c['n_users_ge5']} |")
        for s in cs:
            if s["C"]["majority"]:
                L += ["", f"Baseline di maggioranza ({s['dataset']}): " +
                      " · ".join(f"{k} {v:.1%}" for k, v in s["C"]["majority"].items())]
    L += ["", "## Le tre risposte", ""]
    for s in S:
        a, b, c = s["A"], s["B"], s.get("C")
        L.append(f"**{s['dataset']}** — "
                 f"A) quota di varianza da situazione = {a['ss_sit_first']:.1%} da sola, "
                 f"{a['ss_sit_given_user']:.1%} al netto dell'utente (utente = {a['ss_user_first']:.1%}). "
                 f"B) R2 mediano situazione~utente = {b['r2_median']:.3f}; flip-rate argmax = "
                 f"{b['flip_argmax']:.1%}, top-1 = {b['flip_top1']:.1%}. ")
        if c:
            L[-1] += (f"C) entropia mediana = {c['entropy_median']:.3f}; utenti con dominante >80% = "
                      f"{c['share_dom_gt80']:.1%}; baseline gender = "
                      f"{c['majority'].get('gender', float('nan')):.1%}.")
        else:
            L[-1] += "C) non calcolato (ml1m-only)."
        L.append("")
    (OUT / "WI0_SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"\n-> {OUT / 'WI0_SUMMARY.md'}", flush=True)


def main():
    cities = sys.argv[1:] or ["ml1m"]
    OUT.mkdir(parents=True, exist_ok=True)
    for c in cities:
        print(f"\n===== WI-0 · {c.upper()} =====", flush=True)
        run(c)
    write_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
