"""[venv-xsage] Costruisce results_record.csv per le tabelle del paper.
Per ogni (dataset, backbone, metric, method): mean + sd_seed (5 seed) + bootstrap CI per-richiesta
(B=1500). Per Δ=SIT-BASE: significativita' (t cross-seed se sd_seed>0, altrimenti bootstrap-Δ),
Holm sui confronti multipli, sig in {'','*','**','***'}; TOST (±0.005) su R@20.
Metriche: CatMRR, CatNDCG, R20, NDCG20, Coverage, Gini, LT20, macroCatMRR.
Ricalcola TUTTI i 5 seed (clustering per-seed; B_full ri-allenato per-seed; backbone cachati riusati);
bootstrap sugli array per-richiesta del seed 42. Anti-circolare: κ* preso dal CSV battery per (seed,backbone).
Uso:  python scripts/yelp/results_record.py [city ...]
Out:  outputs_results/results_record.csv  (append per città)
"""
import sys, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(CLEAN / "scripts" / "cars"))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
from train_bfull import feats_from_df, score_test
from pipeline.step02_models.xsage.backbone_full import ContextAwareFM, FeatureSpec, train_b_full

KTOP, BOOT, TOST_M = 20, 1500, 0.005
BK = ["B_blind", "B_full", "EASE", "DeepFM", "AFM", "FPMC", "SASRec"]
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["CatMRR", "CatNDCG", "R20", "NDCG20", "Coverage", "Gini", "LT20", "macroCatMRR"]


def gini(c):
    x = np.sort(c.astype(np.float64)); n = len(x); s = x.sum()
    return 0. if s <= 0 else float((2 * np.sum(np.arange(1, n + 1) * x) / (n * s)) - (n + 1) / n)


def js_rows(p, q, e=1e-9):
    p = p + e; p = p / p.sum(1, keepdims=True); q = q + e; q = q / q.sum(1, keepdims=True)
    m = .5 * (p + q); kl = lambda a, b: np.sum(a * np.log2(a / b), 1); return .5 * kl(p, m) + .5 * kl(q, m)


def per_request_eval(scores_fn, nudge, kappa, gam, u, i_t, icm, excl, G1, pur, nmac):
    """Ritorna array per-richiesta: cm,cn,ht,idc,js,lt + topk[n,KTOP] + u."""
    n = len(u); tm = icm[i_t]; nI = len(icm)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); idc = np.zeros(n); jsv = np.zeros(n); lt = np.zeros(n)
    topk = np.zeros((n, KTOP), np.int64)
    for bs in range(0, n, 1024):
        be = min(n, bs + 1024); idx = np.arange(bs, be)
        S = np.asarray(scores_fn(idx)).astype(np.float32, copy=True)
        if nudge is not None and kappa > 0:
            S = S + kappa * gam[idx][:, None].astype(np.float32) * nudge[idx][:, icm]
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u[bs + j])]:excl.indptr[int(u[bs + j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, KTOP - 1, axis=1)[:, :KTOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1); tk = np.take_along_axis(part, order, 1)
        topk[idx] = tk; mac = icm[tk]; match = mac == tm[idx, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[idx] = np.where(first > 0, 1. / np.maximum(first, 1), 0.); cn[idx] = np.where(first > 0, 1. / np.log2(np.maximum(first, 1) + 1.), 0.)
        s_t = S[np.arange(be - bs), i_t[idx]]; rank = (S > s_t[:, None]).sum(1) + 1
        ht[idx] = (rank <= KTOP); idc[idx] = np.where(rank <= KTOP, 1. / np.log2(rank + 1.), 0.)
        lt[idx] = G1[tk].mean(1)
        q = np.zeros((be - bs, nmac)); rows = np.repeat(np.arange(be - bs), KTOP); np.add.at(q, (rows, mac.ravel()), 1.)
        jsv[idx] = js_rows(pur[idx], q)
    return dict(cm=cm, cn=cn, ht=ht, idc=idc, js=jsv, lt=lt, topk=topk, u=u, tm=tm)


def pu_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq)); np.add.at(s, inv, v); np.add.at(c, inv, 1); return s / c


def macro_of(cm, tm, nmac):
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).any()]; return float(np.mean(per))


def metrics_from(e, nmac, nI):
    expo = np.bincount(e["topk"].ravel(), minlength=nI).astype(float)
    return {"CatMRR": e["cm"].mean(), "CatNDCG": e["cn"].mean(),
            "R20": pu_mean(e["ht"], e["u"]).mean(), "NDCG20": pu_mean(e["idc"], e["u"]).mean(),
            "JS": e["js"].mean(), "LT20": e["lt"].mean(),
            "Coverage": float((expo > 0).mean()), "Gini": gini(expo),
            "macroCatMRR": macro_of(e["cm"], e["tm"], nmac)}


def boot_ci_p(eB, eS, metric, nmac, nI, rng):
    """CI (su SIT) e p del Δ=SIT-BASE via bootstrap per-richiesta (utenti per R20/NDCG20)."""
    n = len(eB["u"])
    if metric in ("R20", "NDCG20"):
        key = "ht" if metric == "R20" else "idc"
        uq, inv = np.unique(eB["u"], return_inverse=True); nu = len(uq)
        sB = np.zeros(nu); cB = np.zeros(nu); np.add.at(sB, inv, eB[key]); np.add.at(cB, inv, 1); pB = sB / cB
        sS = np.zeros(nu); np.add.at(sS, inv, eS[key]); pS = sS / cB
        bs_s = np.empty(BOOT); bs_d = np.empty(BOOT)
        for b in range(BOOT):
            ix = rng.integers(0, nu, nu); bs_s[b] = pS[ix].mean(); bs_d[b] = pS[ix].mean() - pB[ix].mean()
    elif metric in ("Coverage", "Gini"):
        tkB, tkS = eB["topk"], eS["topk"]; bs_s = np.empty(BOOT); bs_d = np.empty(BOOT)
        for b in range(BOOT):
            ix = rng.integers(0, n, n)
            eB_ = np.bincount(tkB[ix].ravel(), minlength=nI).astype(float); eS_ = np.bincount(tkS[ix].ravel(), minlength=nI).astype(float)
            vS = (float((eS_ > 0).mean()) if metric == "Coverage" else gini(eS_))
            vB = (float((eB_ > 0).mean()) if metric == "Coverage" else gini(eB_))
            bs_s[b] = vS; bs_d[b] = vS - vB
    else:
        mp = {"CatMRR": "cm", "CatNDCG": "cn", "LT20": "lt"}
        if metric == "macroCatMRR":
            aB, aS, tm = eB["cm"], eS["cm"], eB["tm"]; bs_s = np.empty(BOOT); bs_d = np.empty(BOOT)
            for b in range(BOOT):
                ix = rng.integers(0, n, n); vS = macro_of(aS[ix], tm[ix], nmac); vB = macro_of(aB[ix], tm[ix], nmac)
                bs_s[b] = vS; bs_d[b] = vS - vB
        else:
            k = mp[metric]; aB, aS = eB[k], eS[k]; bs_s = np.empty(BOOT); bs_d = np.empty(BOOT)
            for b in range(BOOT):
                ix = rng.integers(0, n, n); bs_s[b] = aS[ix].mean(); bs_d[b] = aS[ix].mean() - aB[ix].mean()
    lo, hi = np.percentile(bs_s, [2.5, 97.5]); p = 2. * min((bs_d <= 0).mean(), (bs_d >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.))


def bfull_scores(ds, icm, excl, nmac, dev, seed):
    nU = int(ds["n_users"]); nI = int(ds["n_items"])
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icmF = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    ftr = feats_from_df(ds["df_train"], icmF, nmac); fva = feats_from_df(ds["df_val"], icmF, nmac); fte = feats_from_df(ds["df_test"], icmF, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); iv = ds["df_val"]["i_idx"].values.astype(np.int64)
    G1d = np.zeros(nI, np.float32); purd = np.full((len(uv), nmac), 1.0 / nmac, np.float32)
    spec = FeatureSpec(n_users=nU, n_items=nI, n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac); best = (-1, None, 64)
    for emb in (32, 64):
        torch.manual_seed(seed); mdl = ContextAwareFM(spec, d=emb).to(dev)
        for _ in range(3):
            train_b_full(mdl, ftr, mask, icmF, np.zeros(nI, np.int64), dev, lr=5e-3, n_epochs=3, verbose=False)
            sv = score_test(mdl, fva, spec, icmF, dev)
            vm = per_request_eval((lambda idx, S=sv: S[idx]), None, 0., np.ones(len(uv), np.float32), uv, iv, icm, excl, G1d, purd, nmac)["cm"].mean()
            if vm > best[0]: best = (vm, copy.deepcopy(mdl.state_dict()), emb)
    mdl = ContextAwareFM(spec, d=best[2]).to(dev); mdl.load_state_dict(best[1])
    return score_test(mdl, fte, spec, icmF, dev)


def run_city(city, dev):
    csvb = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")
    bdir = CLEAN / "data" / city / "backbone"
    per_seed = {(bk, m, met): [] for bk in BK for m in ("BASE", "SIT") for met in METRICS}
    seed42 = {}
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]; G1 = D0["G1"].astype(np.float32)
        nI = int(ds["n_items"])
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
        fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
        _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
        mem_te = membership_from_assign(kte, compte, isbte, K); gam_te = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
        b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); nudge = mem_te.astype(np.float32) @ b_z
        dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
        nU2 = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
        Pu = np.zeros((nU2, nmac)); np.add.at(Pu, (umac, cmt), 1.); Pu[Pu.sum(1) == 0] = 1.; Pu /= Pu.sum(1, keepdims=True); purt = Pu[ute]
        bft = bfull_scores(ds, icm, excl, nmac, dev, seed)
        print(f"  [{city}] seed {seed}: K={K} eps={eps} bfull ok", flush=True)
        for bk in BK:
            if bk == "B_blind": sfn = (lambda idx, u=ute: sb[u[idx]])
            elif bk == "B_full": sfn = (lambda idx: bft[idx])
            else:
                fu = bdir / f"{bk}.scores_user.npy"
                if fu.exists(): M = np.load(fu, mmap_mode="r"); sfn = (lambda idx, M=M, u=ute: M[u[idx]])
                else: Mt = np.load(bdir / f"{bk}.scores_test.npy", mmap_mode="r"); sfn = (lambda idx, Mt=Mt: Mt[idx])
            kap = float(csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"].iloc[0])
            eB = per_request_eval(sfn, None, 0., gam_te, ute, ite, icm, excl, G1, purt, nmac)
            eS = per_request_eval(sfn, nudge, kap, gam_te, ute, ite, icm, excl, G1, purt, nmac)
            mB = metrics_from(eB, nmac, nI); mS = metrics_from(eS, nmac, nI)
            for met in METRICS:
                per_seed[(bk, "BASE", met)].append(mB[met]); per_seed[(bk, "SIT", met)].append(mS[met])
            if seed == 42: seed42[bk] = (eB, eS, kap)
    # ---- aggrega + bootstrap + stats ----
    rng = np.random.default_rng(2024); nmac = D0["n_macros"]; nI = int(ds["n_items"])
    rows = []; pending = []  # (rowidx_sit, p_raw)
    for bk in BK:
        eB, eS, kap = seed42[bk]
        m42B = metrics_from(eB, nmac, nI); m42S = metrics_from(eS, nmac, nI)   # stime seed-42 (su cui il CI è centrato)
        for met in METRICS:
            vB = np.array(per_seed[(bk, "BASE", met)]); vS = np.array(per_seed[(bk, "SIT", met)])
            lo, hi, p = boot_ci_p(eB, eS, met, nmac, nI, rng)   # CI su SIT (seed 42) + p del Δ bootstrap
            d = float(vS.mean() - vB.mean()); sdd = (vS - vB).std(ddof=1)
            if sdd > 1e-9: p_use = float(stats.ttest_rel(vS, vB).pvalue)
            else: p_use = p
            tost = 1 if (met == "R20" and lo > -TOST_M and hi < TOST_M) else (0 if met == "R20" else "")
            rows.append(dict(dataset=city, backbone=bk, metric=met, method="BASE",
                             mean=round(float(vB.mean()), 5), sd_seed=round(float(vB.std(ddof=1)), 5),
                             mean_s42=round(m42B[met], 5), ci_lo="", ci_hi="", delta="", p_raw="", p_holm="", sig="", tost_equiv=""))
            ridx = len(rows)
            rows.append(dict(dataset=city, backbone=bk, metric=met, method="SIT",
                             mean=round(float(vS.mean()), 5), sd_seed=round(float(vS.std(ddof=1)), 5),
                             mean_s42=round(m42S[met], 5), ci_lo=round(lo, 5), ci_hi=round(hi, 5), delta=round(d, 5),
                             p_raw=round(p_use, 6), p_holm="", sig="", tost_equiv=tost))
            pending.append((ridx, p_use))
    # Holm su tutti i Δ
    ps = sorted(pending, key=lambda x: x[1]); mtot = len(ps); prev = 0
    for rank, (ridx, p) in enumerate(ps):
        ph = min(1.0, max(prev, (mtot - rank) * p)); prev = ph
        rows[ridx]["p_holm"] = round(ph, 6)
        rows[ridx]["sig"] = "***" if ph < .001 else "**" if ph < .01 else "*" if ph < .05 else ""
    return rows


def main():
    cities = sys.argv[1:] or ["nyc_tist", "saopaulo", "ml1m"]
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = CLEAN / "outputs_results" / "results_record.csv"
    for city in cities:
        print(f"=== {city} ===", flush=True)
        rows = run_city(city, dev); df = pd.DataFrame(rows)
        if out.exists():
            old = pd.read_csv(out); old = old[old.dataset != city]; df = pd.concat([old, df], ignore_index=True)
        df.to_csv(out, index=False); print(f"-> {out} ({city} scritto, {len(rows)} righe)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
