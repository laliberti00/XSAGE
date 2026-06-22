"""FOUNDATIONAL ablation: is INTENT constitutive of the situation construct?
Build situations on v=[c̃‖e] (FULL, current) vs v=[c̃] (CTX, context-only). K/ε
RE-SELECTED for CTX with the SAME criterion (internal silhouette + |A|+2 ceiling;
ε boundary-band). Everything else identical (backbone, bias b̃^(k), combiner, κ=0.25).

Win criterion (fixed BEFORE): FULL>CTX significant on Cat-MRR ⇒ intent constitutive.
FULL≈CTX ⇒ intent adds no discriminative power. FULL<CTX ⇒ intent adds noise.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("ov", str(CLEAN / "scripts" / "overnight_selection.py"))
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
from pipeline.step02_models.xsage.l2_comprehension import _assign, adjusted_rand_score, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FULL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FULL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
N_CTX = len(ov.DEFAULT_ATTRIBUTES)
KAPPA, K_TOP, BATCH = 0.25, 20, 1024
K_RANGE = [3, 4, 5, 6, 7, 8, 9]; EPS_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
BAND = (0.10, 0.30); SEEDS = (42, 43, 44); SIL_N, SEED, BOOT = 20000, 42, 1500
ALPHA = 50.0


def select_K(v_tr, ceil, rng):
    n = len(v_tr); sidx = rng.choice(n, SIL_N, replace=False) if n > SIL_N else np.arange(n)
    best_k, best_s = K_RANGE[0], -1
    for K in K_RANGE:
        r = fit_rough_kmeans(v_tr, K=K, eps=0.0, seed=SEED, max_iter=60)
        lab = r.core_label
        if len(np.unique(lab)) < 2: continue
        s = silhouette_score(v_tr[sidx], lab[sidx])
        if s > best_s: best_s, best_k = s, K
    return min(best_k, ceil)


def select_eps(v_tr, v_va, K):
    cells = []
    for eps in EPS_GRID:
        labs, bf = [], []
        for s in SEEDS:
            r = fit_rough_kmeans(v_tr, K=K, eps=eps, seed=s, max_iter=80)
            _, k_va, _, isb = _assign(v_va, r.prototypes, eps)
            labs.append(k_va); bf.append(float(isb.mean()))
        ari = float(np.mean([adjusted_rand_score(labs[i], labs[j]) for i, j in combinations(range(3), 2)]))
        cells.append((eps, ari, float(np.mean(bf))))
    inb = [(e, a, b) for e, a, b in cells if BAND[0] <= b <= BAND[1]]
    if inb:
        return max(inb, key=lambda x: x[1])[0]
    return min(cells, key=lambda x: abs(x[2] - 0.20))[0]


def build_view(v_tr, v_te, K, eps, cmt, nmac):
    fit = fit_rough_kmeans(v_tr, K=K, eps=eps, seed=SEED, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)
    _, k_te, comp, isb = _assign(v_te, fit.prototypes, eps)
    mem = ov.membership_from_assign(k_te, comp, isb, K)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    gamma = (1.0 / np.maximum(comp.sum(1), 1)).astype(np.float32)
    return k_te.astype(np.int64), mem, b_z, gamma


def score(prep, mem, b_z, gamma, cfg):
    ds = prep["ds"]; df = ds["df_test"]; n = len(df); icm = prep["item_cat_macro"]
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]
    sb_full = prep["scores_blind_full"]; excl = prep["excluded"]; b_bar = b_z.mean(0)
    cmrr = np.zeros(n); cndcg = np.zeros(n); hit = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        if cfg == "BASE":
            S = sb
        else:
            d = (mem[bs:be].astype(np.float32) @ b_z) if cfg == "SIT" else b_bar[None, :]
            S = sb + KAPPA * gamma[bs:be][:, None].astype(np.float32) * d[:, icm] if cfg == "SIT" \
                else sb + KAPPA * gamma[bs:be][:, None].astype(np.float32) * b_bar[icm][None, :]
        S = S.copy()
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        macros = icm[np.take_along_axis(part, order, 1)]
        match = macros == tm[bs:be, None]; has = match.any(1)
        first = np.where(has, match.argmax(1) + 1, 0)
        cmrr[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        cndcg[bs:be] = np.where(first > 0, 1.0 / np.log2(np.maximum(first, 1) + 1.0), 0.0)
        s_tgt = S[np.arange(be - bs), i_t[bs:be]]
        hit[bs:be] = ((S > s_tgt[:, None]).sum(1) + 1 <= K_TOP)
    return cmrr, cndcg, hit, u


def pu(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v.astype(np.float64)); np.add.at(c, inv, 1); return float((s / c).mean())


def main():
    rng = np.random.default_rng(SEED)
    rows, summ = [], []
    for city in CITIES:
        prep, fit_f, _ = ov.build_prep_for_report(city, PERC, K_FULL[city], EPS_FULL[city])
        ds = prep["ds"]; nmac = ds["n_macros"]; m2i = ds["macro_to_idx"]
        cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
        built = ov.build_v(city, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train", "val", "test"))
        vtr, vva, vte = built["vs"]["train"], built["vs"]["val"], built["vs"]["test"]
        nA = int(built["attractors"].sum()); ceil = nA + 2
        views = {}
        # FULL (given K/ε)
        views["FULL"] = (K_FULL[city], EPS_FULL[city], vtr, vte)
        # CTX (re-selected)
        ctr, cva, cte = vtr[:, :N_CTX], vva[:, :N_CTX], vte[:, :N_CTX]
        Kc = select_K(ctr, ceil, rng); ec = select_eps(ctr, cva, Kc)
        views["CTX"] = (Kc, ec, ctr, cte)
        print(f"[{city}] |A|={nA} ceil={ceil}  FULL K={K_FULL[city]} eps={EPS_FULL[city]}  "
              f"CTX K={Kc} eps={ec}", flush=True)
        res = {}
        for vn, (K, eps, v_tr, v_te) in views.items():
            z_te, mem, b_z, gamma = build_view(v_tr, v_te, K, eps, cmt, nmac)
            sc = {cf: score(prep, mem, b_z, gamma, cf) for cf in ["BASE", "SIT", "UNI_mean"]}
            res[vn] = {"z": z_te, "sc": sc, "K": K, "eps": eps}
        u = res["FULL"]["sc"]["BASE"][3]; n = len(u)
        ari_fc = float(adjusted_rand_score(res["FULL"]["z"], res["CTX"]["z"]))
        for vn in ["FULL", "CTX"]:
            sm = res[vn]["sc"]["SIT"]; um = res[vn]["sc"]["UNI_mean"]
            d_spec = sm[0] - um[0]
            bs_sp = np.array([d_spec[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
            rows.append({"city": city, "view": vn, "K": res[vn]["K"], "eps": res[vn]["eps"],
                         "CatMRR_SIT": round(float(sm[0].mean()), 5),
                         "CatNDCG_SIT": round(float(sm[1].mean()), 5),
                         "R20_SIT": round(pu(sm[2], u), 5),
                         "SITvsUNImean": round(float(d_spec.mean()), 5),
                         "spec_sig": "SI" if np.percentile(bs_sp, 2.5) > 0 else "no",
                         "ARI_full_ctx": round(ari_fc, 4)})
        # FULL vs CTX on Cat-MRR (SIT), bootstrap
        df_mrr = res["FULL"]["sc"]["SIT"][0] - res["CTX"]["sc"]["SIT"][0]
        bs_ = np.array([df_mrr[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
        lo, hi = np.percentile(bs_, [2.5, 97.5])
        verdict = "FULL>CTX (intento costitutivo)" if lo > 0 else "FULL<CTX (intento dannoso)" if hi < 0 else "FULL≈CTX (intento neutro)"
        summ.append({"city": city, "dCatMRR_FULL_CTX": round(float(df_mrr.mean()), 5),
                     "CI_lo": round(float(lo), 5), "CI_hi": round(float(hi), 5),
                     "ARI_full_ctx": round(ari_fc, 4), "verdict": verdict,
                     "K_full": K_FULL[city], "K_ctx": res["CTX"]["K"]})
        print(f"  → ΔCatMRR(FULL−CTX)={df_mrr.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  ARI={ari_fc:.3f}  {verdict}", flush=True)

    pd.DataFrame(rows).to_csv(CLEAN / "outputs_results" / "validation" / "intent_ablation.csv", index=False)
    print("\n=== FULL vs CTX su Cat-MRR (SIT, κ=0.25, bootstrap CI) ===")
    sd = pd.DataFrame(summ)
    for r in sd.itertuples():
        print(f"  {r.city:<11} K {r.K_full}→{r.K_ctx}  ΔCatMRR={r.dCatMRR_FULL_CTX:+.5f} "
              f"[{r.CI_lo:+.5f},{r.CI_hi:+.5f}]  ARI(full,ctx)={r.ARI_full_ctx:.3f}  → {r.verdict}")
    print("\n=== Specificità SIT>UNI_mean in entrambe le viste ===")
    dr = pd.DataFrame(rows)
    for city in CITIES:
        for vn in ["FULL", "CTX"]:
            r = dr[(dr.city == city) & (dr.view == vn)].iloc[0]
            print(f"  {city:<11} {vn:<5} CatMRR={r.CatMRR_SIT:.4f} R20={r.R20_SIT:.4f} "
                  f"SIT-UNImean={r.SITvsUNImean:+.4f}({r.spec_sig})")
    print("\n→ intent_ablation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
