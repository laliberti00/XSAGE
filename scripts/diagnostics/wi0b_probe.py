"""[venv-xsage] WI-0b — PROBE DI DISAMBIGUAZIONE (follow-up di WI-0).

WI-0 ha lasciato due risultati confusi fra loro: CHECK A dice che la situazione al netto
dell'utente spiega ~0% della varianza; CHECK C dice che l'utente mediano occupa UNA sola
situazione nel test. Se la situazione e' quasi una funzione deterministica dell'utente, non
esiste variazione within-user da attribuirle e lo "0%" e' il caso degenere, non una misura.

  CHECK A2  decomposizione RISTRETTA agli utenti multi-situazione + test appaiato within-user
            + controllo di Simpson per situazione + complemento mono-situazione.
  CHECK C2  entropia su train / val / test / timeline intera + versione a finestra scorrevole
            + geometria dello split (lunghezze e verifica temporale per-utente).
  CHECK D   curva di minimizzazione: demografia predetta da (a) situazione corrente,
            (b) istogramma situazioni sulla timeline, (c) storico item grezzo.

Nessun retraining. Stesso clustering della batteria (gate di ancoraggio ripetuto).
Il probe RIPORTA NUMERI: nessun giudizio di merito.

Uso:  python -m scripts.diagnostics.wi0b_probe [ml1m ...] [--no-d]
Out:  outputs_results/probe_wi0b/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "diagnostics"))
from viability_probe import additive_sse, build_common, oneway, quart   # noqa: E402
from sklearn.linear_model import LogisticRegression                    # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score    # noqa: E402
from sklearn.preprocessing import StandardScaler                       # noqa: E402
from sklearn.pipeline import make_pipeline                             # noqa: E402
import scipy.sparse as sps                                             # noqa: E402

OUT = CLEAN / "outputs_results" / "probe_wi0b"
SEED = 42
B_PERM = 500        # permutazioni within-user per il null dello spread
B_BOOT = 2000       # bootstrap CI
WINDOWS = (20, 50)  # finestre scorrevoli per l'entropia
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}


def row(ds, check, block, key, subkey, metric, value, n=None, note=""):
    return dict(dataset=ds, check=check, block=block, key=key, subkey=subkey,
                metric=metric, value=value, n=n, note=note)


def boot_median_ci(a, B=B_BOOT, seed=SEED):
    a = np.asarray(a, float)
    if a.size < 3: return (np.nan, np.nan)
    rng = np.random.default_rng(seed); n = a.size
    m = np.array([np.median(a[rng.integers(0, n, n)]) for _ in range(B)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def norm_entropy(codes, K):
    c = np.bincount(codes, minlength=K).astype(np.float64)
    p = c / c.sum(); nz = p[p > 0]
    return (float(-(nz * np.log(nz)).sum() / np.log(K)) if K > 1 else 0.0,
            float(p.max()), int((c > 0).sum()))


# =============================================================== CHECK A2
def check_a2(C):
    city = C["city"]; K = C["K"]; R = []; PS = []
    q = C["q_sit"].astype(np.float64); qb = C["q_base"].astype(np.float64)
    u = C["ute"].astype(np.int64); k = C["k_te"].astype(np.int64)
    print(f"[{city}] A2 — popolazione multi-situazione...", flush=True)

    d = pd.DataFrame({"u": u, "k": k, "q": q, "qb": qb})
    ndist = d.groupby("u")["k"].nunique()
    multi_u = set(ndist[ndist >= 2].index); mono_u = set(ndist[ndist < 2].index)
    m_mask = d["u"].isin(multi_u).values
    n_all_u = int(ndist.shape[0]); n_all_r = len(d)
    R.append(row(city, "A2", "population", "n_users_total", "", "value", n_all_u, n_all_u))
    R.append(row(city, "A2", "population", "n_users_multi", "", "value", len(multi_u), n_all_u))
    R.append(row(city, "A2", "population", "share_users_multi", "", "value",
                 len(multi_u) / n_all_u, n_all_u))
    R.append(row(city, "A2", "population", "n_req_multi", "", "value", int(m_mask.sum()), n_all_r))
    R.append(row(city, "A2", "population", "share_req_multi", "", "value",
                 float(m_mask.mean()), n_all_r))

    # --- decomposizione su ristretta e su complemento
    for lab, msk in (("multi_situation", m_mask), ("mono_situation", ~m_mask)):
        yy = q[msk]; uu = u[msk]; kk = k[msk]
        if yy.size < 10 or len(np.unique(kk)) < 2:
            R.append(row(city, "A2", f"variance_{lab}", "ss_sit_given_user", "", "value",
                         0.0, int(yy.size), "degenere per costruzione (situazione = f(utente))"))
            continue
        ou, ok = oneway(yy, uu), oneway(yy, kk)
        sse_u = ou["ss_total"] - ou["ss_between"]; sse_k = ok["ss_total"] - ok["ss_between"]
        sse_uk = additive_sse(yy, uu, kk); ss_t = ou["ss_total"]
        for met, v in (("ss_user_alone", ou["ss_between"] / ss_t),
                       ("ss_sit_alone", ok["ss_between"] / ss_t),
                       ("ss_sit_given_user", (sse_u - sse_uk) / ss_t),
                       ("ss_user_given_sit", (sse_k - sse_uk) / ss_t),
                       ("ss_residual", sse_uk / ss_t),
                       ("icc_user", ou["icc"]), ("icc_sit", ok["icc"])):
            R.append(row(city, "A2", f"variance_{lab}", met, "", "value", float(v), int(yy.size)))

    # --- test appaiato within-user: d(u,k) = media di cella meno media dell'utente
    print(f"[{city}] A2 — test appaiato within-user...", flush=True)
    dm = d[m_mask]
    cell = dm.groupby(["u", "k"], sort=False).agg(q=("q", "mean"), qb=("qb", "mean"),
                                                  n=("q", "size")).reset_index()
    umean = cell.groupby("u")["q"].transform("mean")     # media NON pesata sulle celle dell'utente
    cell["dev"] = cell["q"] - umean
    for kk in range(K):
        sub = cell[cell["k"] == kk]["dev"].values
        if sub.size < 8:
            PS.append(dict(dataset=city, situation=kk, n_users=int(sub.size), dev_median=np.nan,
                           wilcoxon_p=np.nan, ci_lo=np.nan, ci_hi=np.nan, note="n<8"))
            continue
        try: w = float(stats.wilcoxon(sub, zero_method="wilcox").pvalue)
        except Exception: w = np.nan
        lo, hi = boot_median_ci(sub)
        PS.append(dict(dataset=city, situation=kk, n_users=int(sub.size),
                       dev_median=float(np.median(sub)), wilcoxon_p=w, ci_lo=lo, ci_hi=hi, note=""))
    # Kruskal-Wallis: la deviazione within-user dipende dalla situazione?
    groups = [cell[cell["k"] == kk]["dev"].values for kk in range(K)
              if (cell["k"] == kk).sum() >= 8]
    if len(groups) >= 2:
        kw = stats.kruskal(*groups)
        R.append(row(city, "A2", "within_user", "kruskal_H_dev_by_situation", "", "value",
                     float(kw.statistic), len(cell)))
        R.append(row(city, "A2", "within_user", "kruskal_p", "", "value", float(kw.pvalue), len(cell)))

    # --- spread osservato vs null per permutazione DENTRO l'utente
    print(f"[{city}] A2 — null per permutazione within-user ({B_PERM} perm)...", flush=True)
    o = np.argsort(dm["u"].values, kind="stable")
    us = dm["u"].values[o]; ks = dm["k"].values[o]; qs = dm["q"].values[o]
    uq, ucode = np.unique(us, return_inverse=True)
    cid = ucode * K + ks
    cnt = np.bincount(cid, minlength=len(uq) * K).astype(np.float64)
    have = cnt > 0
    cell_u = (np.arange(len(uq) * K) // K)[have]

    def med_spread(vals):
        s = np.bincount(cid, weights=vals, minlength=len(uq) * K)[have] / cnt[have]
        df_ = pd.DataFrame({"u": cell_u, "m": s}).groupby("u")["m"].agg(["max", "min"])
        return float(np.median((df_["max"] - df_["min"]).values))
    obs = med_spread(qs)
    rng = np.random.default_rng(SEED)
    null = np.empty(B_PERM)
    for b in range(B_PERM):
        idx = np.lexsort((rng.random(len(us)), us))
        null[b] = med_spread(qs[idx])
    p_perm = float(((null >= obs).sum() + 1) / (B_PERM + 1))
    for met, v in (("spread_median_observed", obs), ("spread_median_null_mean", float(null.mean())),
                   ("spread_median_null_p95", float(np.percentile(null, 95))),
                   ("perm_p_onesided", p_perm), ("n_perm", B_PERM)):
        R.append(row(city, "A2", "spread_permutation", met, "", "value", float(v), len(uq)))

    # --- Simpson: Delta(SIT-BASE) per situazione, su tutte le richieste
    print(f"[{city}] A2 — controllo di Simpson per situazione...", flush=True)
    agg = d.groupby("k").agg(n=("q", "size"), sit=("q", "mean"), base=("qb", "mean")).reset_index()
    for _, r in agg.iterrows():
        R.append(row(city, "A2", "simpson_by_situation", f"sit_{int(r['k'])}", "", "delta_sit_base",
                     float(r["sit"] - r["base"]), int(r["n"])))
    R.append(row(city, "A2", "simpson_by_situation", "AGGREGATE", "", "delta_sit_base",
                 float(q.mean() - qb.mean()), len(q)))
    R.append(row(city, "A2", "simpson_by_situation", "n_situations_positive", "", "value",
                 int((agg["sit"] > agg["base"]).sum()), int(len(agg))))
    pd.DataFrame(R).to_csv(OUT / f"check_a2_restricted_{city}.csv", index=False)
    pd.DataFrame(PS).to_csv(OUT / f"check_a2_per_situation_{city}.csv", index=False)

    vm = {r["key"]: r["value"] for r in R if r["block"] == "variance_multi_situation"}
    return dict(share_users_multi=len(multi_u) / n_all_u, share_req_multi=float(m_mask.mean()),
                n_users_multi=len(multi_u),
                ss_sit_given_user=vm.get("ss_sit_given_user", np.nan),
                ss_user_alone=vm.get("ss_user_alone", np.nan),
                spread_obs=obs, spread_null=float(null.mean()), perm_p=p_perm,
                kruskal_p=float(kw.pvalue) if len(groups) >= 2 else np.nan,
                n_sit_positive=int((agg["sit"] > agg["base"]).sum()), n_sit=int(len(agg)),
                delta_aggregate=float(q.mean() - qb.mean()))


# =============================================================== CHECK C2
def check_c2(C):
    city = C["city"]; K = C["K"]; R = []; W = []
    ds = C["ds"]
    print(f"[{city}] C2 — entropia su quattro orizzonti...", flush=True)
    parts = {"train": (ds["df_train"], C["k_tr"]), "val": (ds["df_val"], C["k_va"]),
             "test": (C["dft"], C["k_te"])}
    allu = np.concatenate([p[0]["u_idx"].values.astype(np.int64) for p in parts.values()])
    allk = np.concatenate([p[1] for p in parts.values()])
    allt = np.concatenate([p[0]["time_local"].values for p in parts.values()])
    parts["all"] = None

    stats_h = {}
    for h in ("train", "val", "test", "all"):
        if h == "all": uu, kk = allu, allk
        else: uu = parts[h][0]["u_idx"].values.astype(np.int64); kk = parts[h][1]
        dd = pd.DataFrame({"u": uu, "k": kk})
        cnt = dd.groupby("u").size(); keep = cnt[cnt >= 5].index
        ent, dom, dis = [], [], []
        for _, g in dd[dd["u"].isin(keep)].groupby("u"):
            e, p, n = norm_entropy(g["k"].values, K); ent.append(e); dom.append(p); dis.append(n)
        ent = np.array(ent); dom = np.array(dom); dis = np.array(dis, float)
        n_u = len(ent)
        stats_h[h] = dict(ent_med=float(np.median(ent)) if n_u else np.nan,
                          dom_med=float(np.median(dom)) if n_u else np.nan,
                          share80=float((dom > .8).mean()) if n_u else np.nan,
                          dis_med=float(np.median(dis)) if n_u else np.nan, n_users=n_u)
        for name, arr in (("entropy_norm", ent), ("dominant_share", dom), ("n_distinct_sit", dis)):
            q1, med, q3 = quart(arr)
            for met, v in (("p25", q1), ("median", med), ("p75", q3)):
                R.append(row(city, "C2", "horizon", h, name, met, float(v), n_u))
        R.append(row(city, "C2", "horizon", h, "share_users_dominant_gt80", "value",
                     stats_h[h]["share80"], n_u))
        R.append(row(city, "C2", "horizon", h, "n_users_ge5", "value", n_u, n_u))

    # geometria dello split
    print(f"[{city}] C2 — geometria dello split...", flush=True)
    len_tl = pd.Series(allu).value_counts()
    len_te = C["dft"].groupby("u_idx").size()
    R.append(row(city, "C2", "split_geometry", "median_timeline_len", "", "value",
                 float(len_tl.median()), int(len_tl.shape[0])))
    R.append(row(city, "C2", "split_geometry", "median_test_len", "", "value",
                 float(len_te.median()), int(len_te.shape[0])))
    R.append(row(city, "C2", "split_geometry", "median_test_share", "", "value",
                 float((len_te / len_tl.reindex(len_te.index)).median()), int(len_te.shape[0])))
    tr_max = ds["df_train"].groupby("u_idx")["time_local"].max()
    te_min = C["dft"].groupby("u_idx")["time_local"].min()
    common = tr_max.index.intersection(te_min.index)
    viol = int((te_min.loc[common] < tr_max.loc[common]).sum())      # violazione STRETTA
    ties = int((te_min.loc[common] == tr_max.loc[common]).sum())     # pareggi di timestamp
    R.append(row(city, "C2", "split_geometry", "users_test_strictly_before_train_end", "", "value",
                 viol, int(len(common)), "violazione stretta = fuga di futuro; 0 = split causale"))
    R.append(row(city, "C2", "split_geometry", "users_timestamp_tie_train_test", "", "value",
                 ties, int(len(common)),
                 "pareggio esatto sul confine: timestamp a risoluzione di secondo, non leakage"))

    # finestra scorrevole sulla timeline intera
    print(f"[{city}] C2 — entropia a finestra scorrevole...", flush=True)
    o = np.lexsort((allt, allu)); us = allu[o]; ks = allk[o]
    bounds = np.flatnonzero(np.r_[True, us[1:] != us[:-1]])
    ends = np.r_[bounds[1:], len(us)]
    for Nw in WINDOWS:
        vals = []
        for a, b in zip(bounds, ends):
            seq = ks[a:b]
            if len(seq) < Nw: continue
            es = [norm_entropy(seq[i:i + Nw], K)[0] for i in range(0, len(seq) - Nw + 1, max(Nw // 2, 1))]
            if es: vals.append(float(np.mean(es)))
        vals = np.array(vals)
        q1, med, q3 = quart(vals)
        for met, v in (("p25", q1), ("median", med), ("p75", q3),
                       ("n_users_eligible", len(vals))):
            W.append(row(city, "C2", "sliding_window", f"N{Nw}", "entropy_norm", met,
                         float(v), len(vals)))
        stats_h[f"win{Nw}"] = float(med)
    pd.DataFrame(R).to_csv(OUT / f"check_c2_entropy_horizons_{city}.csv", index=False)
    pd.DataFrame(W).to_csv(OUT / f"check_c2_window_{city}.csv", index=False)
    return dict(h=stats_h, median_test_len=float(len_te.median()),
                median_tl_len=float(len_tl.median()), split_violations=viol, split_ties=ties)


# =============================================================== CHECK D
def check_d(C):
    city = C["city"]; K = C["K"]; R = []
    if city != "ml1m": return None
    print(f"[{city}] D — curva di minimizzazione...", flush=True)
    sys.path.insert(0, str(CLEAN / "scripts" / "diagnostics"))
    from viability_probe import ml1m_demographics
    dm = ml1m_demographics().set_index("u_idx").sort_index()
    ds = C["ds"]
    allu = np.concatenate([ds["df_train"]["u_idx"].values, ds["df_val"]["u_idx"].values,
                           C["dft"]["u_idx"].values]).astype(np.int64)
    allk = np.concatenate([C["k_tr"], C["k_va"], C["k_te"]])
    alli = np.concatenate([ds["df_train"]["i_idx"].values, ds["df_val"]["i_idx"].values,
                           C["dft"]["i_idx"].values]).astype(np.int64)
    nU = int(ds["n_users"]); nI = int(ds["n_items"])

    hist = np.zeros((nU, K), np.float64)
    np.add.at(hist, (allu, allk), 1.0)
    tot = hist.sum(1, keepdims=True)
    Xb = hist / np.maximum(tot, 1)                       # (b) istogramma situazioni
    cur = hist.argmax(1)                                  # (a) situazione modale
    Xa = np.zeros((nU, K), np.float64); Xa[np.arange(nU), cur] = 1.0
    Xc = sps.csr_matrix((np.ones(len(allu)), (allu, alli)), shape=(nU, nI))  # (c) bag-of-items
    Xc.data[:] = 1.0

    idx = dm.index.values
    feats = {"a_situation_label": Xa[idx], "b_situation_histogram": Xb[idx], "c_item_history": Xc[idx]}
    for tgt in ("gender", "age", "occupation"):
        y = dm[tgt].values
        vc = pd.Series(y).value_counts()
        maj = float(vc.iloc[0] / len(y))
        nsp = int(min(5, vc.min()))
        if nsp < 2:
            R.append(row(city, "D", tgt, "SKIPPED", "", "value", np.nan, len(y),
                         f"classe piu' rara ha {int(vc.min())} membri"))
            continue
        cv = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=SEED)
        R.append(row(city, "D", tgt, "majority_baseline", "", "value", maj, len(y)))
        R.append(row(city, "D", tgt, "n_classes", "", "value", int(vc.shape[0]), len(y)))
        R.append(row(city, "D", tgt, "cv_folds", "", "value", nsp, len(y)))
        for name, X in feats.items():
            clf = (make_pipeline(StandardScaler(with_mean=not sps.issparse(X)),
                                 LogisticRegression(max_iter=2000, random_state=SEED))
                   if not sps.issparse(X) else
                   LogisticRegression(max_iter=2000, random_state=SEED))
            pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=1)
            acc = float(accuracy_score(y, pred)); bal = float(balanced_accuracy_score(y, pred))
            R.append(row(city, "D", tgt, name, "accuracy", "value", acc, len(y),
                         f"baseline_maggioranza={maj:.4f}"))
            R.append(row(city, "D", tgt, name, "balanced_accuracy", "value", bal, len(y),
                         f"baseline_bilanciata={1.0/vc.shape[0]:.4f}"))
            R.append(row(city, "D", tgt, name, "acc_minus_majority", "value", acc - maj, len(y)))
            print(f"    {tgt:11} {name:22} acc={acc:.4f} (maj={maj:.4f}, delta={acc-maj:+.4f}) "
                  f"bal={bal:.4f}", flush=True)
    pd.DataFrame(R).to_csv(OUT / f"check_d_minimization_{city}.csv", index=False)
    df = pd.DataFrame(R)
    out = {}
    for tgt in ("gender", "age", "occupation"):
        t = df[(df.block == tgt)]
        if t.empty: continue
        mj = t[t.key == "majority_baseline"]["value"]
        out[tgt] = dict(majority=float(mj.iloc[0]) if len(mj) else np.nan,
                        **{k: float(t[(t.key == k) & (t.subkey == "accuracy")]["value"].iloc[0])
                           for k in feats if len(t[(t.key == k) & (t.subkey == "accuracy")])})
    return out


# =============================================================== main
def run(city, do_d=True):
    C = build_common(city)
    a = float(C["q_sit"].mean()); ref = ANCHOR.get(city, np.nan)
    ok = bool(np.isfinite(ref) and abs(a - ref) < 5e-6)
    print(f"\n[{city}] GATE: Cat-MRR SIT@k* = {a:.5f} (atteso {ref:.5f}) -> "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    if not ok:
        print(f"[{city}] STOP: il gate non replica bit-per-bit. Nessun check eseguito.", flush=True)
        return dict(dataset=city, gate_ok=False, anchor=a, anchor_ref=ref)
    A = check_a2(C); Cc = check_c2(C)
    D = check_d(C) if (do_d and city == "ml1m") else None
    s = dict(dataset=city, gate_ok=True, anchor=a, anchor_ref=ref, K=C["K"], A2=A, C2=Cc, D=D)
    json.dump(s, open(OUT / f"_wi0b_{city}.json", "w"), indent=2, default=float)
    return s


def summary():
    S = [json.load(open(p)) for p in sorted(OUT.glob("_wi0b_*.json"))]
    if not S: return
    L = ["# WI-0b — Probe di disambiguazione", "",
         "Follow-up di WI-0. Diagnostica pura su artefatti in cache, seed 42, nessun retraining.",
         "**Nessun giudizio di merito: i numeri, e la regola di lettura pre-registrata.**", "",
         "## Gate di ancoraggio", "",
         "| dataset | Cat-MRR SIT | atteso | esito |", "|---|---|---|---|"]
    for s in S:
        L.append(f"| {s['dataset']} | {s['anchor']:.5f} | {s['anchor_ref']:.5f} | "
                 f"{'OK' if s['gate_ok'] else 'MISMATCH — STOP'} |")
    G = [s for s in S if s["gate_ok"]]
    L += ["", "## CHECK A2 — ristretto agli utenti multi-situazione", "",
          "| dataset | utenti multi | richieste multi | ss_sit_given_user | ss_user_alone | "
          "spread oss. | spread null | p perm. | p Kruskal |", "|---|---|---|---|---|---|---|---|---|"]
    for s in G:
        a = s["A2"]
        L.append(f"| {s['dataset']} | {a['share_users_multi']:.1%} ({a['n_users_multi']}) | "
                 f"{a['share_req_multi']:.1%} | {a['ss_sit_given_user']:.2%} | "
                 f"{a['ss_user_alone']:.1%} | {a['spread_obs']:.5f} | {a['spread_null']:.5f} | "
                 f"{a['perm_p']:.4f} | {a['kruskal_p']:.2e} |")
    L += ["", "Controllo di Simpson — situazioni con Δ(SIT−BASE) > 0 e Δ aggregato:", "",
          "| dataset | situazioni positive | Δ aggregato |", "|---|---|---|"]
    for s in G:
        a = s["A2"]
        L.append(f"| {s['dataset']} | {a['n_sit_positive']}/{a['n_sit']} | {a['delta_aggregate']:+.5f} |")
    L += ["", "## CHECK C2 — entropia per orizzonte", "",
          "| dataset | train | val | test | timeline intera | finestra 20 | finestra 50 | "
          "len test mediana | len timeline mediana |", "|---|---|---|---|---|---|---|---|---|"]
    for s in G:
        h = s["C2"]["h"]
        L.append(f"| {s['dataset']} | {h['train']['ent_med']:.4f} | {h['val']['ent_med']:.4f} | "
                 f"{h['test']['ent_med']:.4f} | {h['all']['ent_med']:.4f} | "
                 f"{h.get('win20', float('nan')):.4f} | {h.get('win50', float('nan')):.4f} | "
                 f"{s['C2']['median_test_len']:.0f} | {s['C2']['median_tl_len']:.0f} |")
    L += ["", "Quota di utenti con situazione dominante >80%, per orizzonte:", "",
          "| dataset | train | val | test | timeline intera | violazioni STRETTE | pareggi timestamp |",
          "|---|---|---|---|---|---|---|"]
    for s in G:
        h = s["C2"]["h"]
        L.append(f"| {s['dataset']} | {h['train']['share80']:.1%} | {h['val']['share80']:.1%} | "
                 f"{h['test']['share80']:.1%} | {h['all']['share80']:.1%} | "
                 f"{s['C2']['split_violations']} | {s['C2'].get('split_ties', 'n/d')} |")
    dd = [s for s in G if s.get("D")]
    if dd:
        L += ["", "## CHECK D — curva di minimizzazione (ml1m)", "",
              "Ogni accuratezza va letta **contro la sua baseline di maggioranza**.", "",
              "| target | baseline | (a) etichetta situazione | (b) istogramma situazioni | "
              "(c) storico item |", "|---|---|---|---|---|"]
        for s in dd:
            for tgt, v in s["D"].items():
                L.append(f"| {tgt} | {v['majority']:.1%} | "
                         f"{v.get('a_situation_label', float('nan')):.1%} | "
                         f"{v.get('b_situation_histogram', float('nan')):.1%} | "
                         f"{v.get('c_item_history', float('nan')):.1%} |")
    L += ["", "## Esito vs pre-registrazione", ""]
    for s in G:
        a = s["A2"]; h = s["C2"]["h"]
        sig = (a["perm_p"] < 0.05) or (a["kruskal_p"] < 0.05)
        subst = a["ss_sit_given_user"] >= 0.01
        if subst and sig: ea = "caso migliore — effetto within-user sostanziale e significativo"
        elif sig: ea = "ambiguo — significativo ma sotto l'1% di varianza"
        else: ea = "caso peggiore — null vero anche dove misurabile"
        gap = h["all"]["ent_med"] - h["test"]["ent_med"]
        if gap > 0.10: ec = "caso migliore — artefatto dello split (entropia molto piu' alta sull'orizzonte lungo)"
        elif gap > 0.02: ec = "ambiguo — divario presente ma modesto"
        else: ec = "caso peggiore — etichetta stabile su ogni orizzonte"
        L += [f"- **{s['dataset']} · A2**: {ea} "
              f"(ss_sit_given_user={a['ss_sit_given_user']:.2%}, p_perm={a['perm_p']:.4f}, "
              f"p_kruskal={a['kruskal_p']:.2e})",
              f"- **{s['dataset']} · C2**: {ec} "
              f"(entropia test={h['test']['ent_med']:.4f} vs timeline={h['all']['ent_med']:.4f}, "
              f"divario={gap:+.4f})"]
    L += ["", "## Domande emerse (RIPORTATE, non eseguite — come da brief)", "",
          "1. **Il val ha la stessa compressione del test** (entropia mediana 0.0000 su tutti e tre "
          "gli orizzonti di validazione). Poiche' kappa* e' selezionato SU VAL, la selezione avviene "
          "su una fetta in cui la situazione e' quasi costante per utente. Non e' un errore di "
          "protocollo — val e test sono simmetrici, quindi la selezione resta anti-circolare — ma "
          "e' una domanda aperta su quanto sia informativa.",
          "2. **ml1m ha 120 utenti con pareggio esatto di timestamp** al confine train/test "
          "(violazioni strette = 0). Non e' leakage: i timestamp MovieLens hanno risoluzione al "
          "secondo. Nota collaterale: `tests/test_causal_split.py` copre solo `D.DEFAULT_CITIES` "
          "(le 5 citta' Foursquare), quindi lo split causale di ml1m non e' mai stato nella suite.",
          "3. **nyc_tist e' sotto-potenziato per A2 per costruzione**: solo 311 utenti "
          "multi-situazione (7,6%). Il suo esito non-significativo non e' un voto contrario; "
          "servirebbe un disegno diverso per misurarlo su quel dominio."]
    (OUT / "wi0b_summary.md").write_text("\n".join(L) + "\n")
    print(f"\n-> {OUT / 'wi0b_summary.md'}", flush=True)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_d = "--no-d" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    for c in (args or ["ml1m"]):
        print(f"\n===== WI-0b · {c.upper()} =====", flush=True)
        run(c, do_d=do_d)
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
