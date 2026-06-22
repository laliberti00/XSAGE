"""São Paulo: l'intento dannoso (FULL<CTX) è ARTEFATTO di K=3/ε=.05 o PROPRIETÀ?
Ri-seleziona K/ε INDIPENDENTEMENTE per FULL [c̃‖e] e CTX [c̃] con i criteri INTERNI
del paper (silhouette/CH/DB + tetto |A|+2; ε boundary-band). NIENTE Cat-MRR nella
selezione (anti-circolare). Poi riconfronta al K/ε ottimale di ciascuna vista.
Caratterizza la ridondanza intento↔contesto (R², |corr|) a São Paulo vs le altre 4.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("ov", str(CLEAN / "scripts" / "overnight_selection.py"))
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
ia_spec = importlib.util.spec_from_file_location("ia", str(CLEAN / "scripts" / "validation" / "intent_ablation.py"))
ia = importlib.util.module_from_spec(ia_spec); ia_spec.loader.exec_module(ia)
from pipeline.step02_models.xsage.l2_comprehension import fit_rough_kmeans

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_USED = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_USED = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
N_CTX = ia.N_CTX; K_RANGE = ia.K_RANGE; SEED, BOOT, SIL_N = 42, 2000, 20000


def internal_curve(v_tr, ceil, sidx):
    """silhouette/CH/DB per K (criteri INTERNI, no Cat-MRR). K* = argmax silhouette ≤ ceil."""
    rows = []
    for K in K_RANGE:
        r = fit_rough_kmeans(v_tr, K=K, eps=0.0, seed=SEED, max_iter=60)
        lab = r.core_label; sub = v_tr[sidx]; ls = lab[sidx]
        if len(np.unique(ls)) < 2:
            rows.append((K, np.nan, np.nan, np.nan)); continue
        rows.append((K, float(silhouette_score(sub, ls)),
                     float(calinski_harabasz_score(sub, ls)),
                     float(davies_bouldin_score(sub, ls))))
    cand = [(K, s) for K, s, _, _ in rows if not np.isnan(s) and K <= ceil]
    kstar = max(cand, key=lambda x: x[1])[0] if cand else K_RANGE[0]
    return rows, kstar


def redundancy(c, e):
    """quanto e è spiegato/correlato da c̃. R² (c̃→e, var-weighted) e |corr| media tra blocchi."""
    r2 = float(LinearRegression().fit(c, e).score(c, e))
    C = np.corrcoef(np.hstack([c, e]).T)
    cc = C[:c.shape[1], c.shape[1]:]
    return r2, float(np.nanmean(np.abs(cc)))


def main():
    rng = np.random.default_rng(SEED)
    kcurve, summ, redun = [], [], []

    # ---------- São Paulo: diagnosi profonda ----------
    city = "saopaulo"
    prep, _, _ = ov.build_prep_for_report(city, PERC, K_USED[city], EPS_USED[city])
    ds = prep["ds"]; nmac = ds["n_macros"]; m2i = ds["macro_to_idx"]
    cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
    built = ov.build_v(city, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train", "val", "test"))
    vtr, vva, vte = built["vs"]["train"], built["vs"]["val"], built["vs"]["test"]
    nA = int(built["attractors"].sum()); ceil = nA + 2
    n_tr = len(vtr); sidx = rng.choice(n_tr, SIL_N, replace=False) if n_tr > SIL_N else np.arange(n_tr)
    views = {"FULL": (vtr, vva, vte), "CTX": (vtr[:, :N_CTX], vva[:, :N_CTX], vte[:, :N_CTX])}
    opt = {}
    for vn, (v_tr, v_va, v_te) in views.items():
        rows, kstar = internal_curve(v_tr, ceil, sidx)
        for K, sil, ch, db in rows:
            kcurve.append({"city": city, "view": vn, "K": K, "silhouette": round(sil, 4) if not np.isnan(sil) else None,
                           "calinski_harabasz": round(ch, 1) if not np.isnan(ch) else None,
                           "davies_bouldin": round(db, 4) if not np.isnan(db) else None,
                           "ceil_A2": ceil, "chosen": "*" if K == kstar else ""})
        eps_star = ia.select_eps(v_tr, v_va, kstar)
        opt[vn] = (kstar, eps_star, v_tr, v_te)
        print(f"[saopaulo] {vn}: K*={kstar} (ceil|A|+2={ceil}, K_used={K_USED[city]})  ε*={eps_star}", flush=True)

    # confronto al K/ε ottimale di CIASCUNA vista
    res = {}
    for vn, (K, eps, v_tr, v_te) in opt.items():
        z_te, mem, b_z, gamma = ia.build_view(v_tr, v_te, K, eps, cmt, nmac)
        sc = {cf: ia.score(prep, mem, b_z, gamma, cf) for cf in ["SIT"]}
        res[vn] = sc["SIT"]
    u = res["FULL"][3]; n = len(u)
    dmrr = res["FULL"][0] - res["CTX"][0]
    bs = np.array([dmrr[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "FULL≥CTX → ARTEFATTO (K=3 strozzava)" if lo > -1e-4 else "FULL<CTX → PROPRIETÀ (danno reale)"
    # NDCG too
    dndcg = res["FULL"][1].mean() - res["CTX"][1].mean()
    summ.append({"city": city, "K_full_opt": opt["FULL"][0], "eps_full_opt": opt["FULL"][1],
                 "K_ctx_opt": opt["CTX"][0], "eps_ctx_opt": opt["CTX"][1],
                 "CatMRR_FULL": round(float(res["FULL"][0].mean()), 5), "CatMRR_CTX": round(float(res["CTX"][0].mean()), 5),
                 "dCatMRR": round(float(dmrr.mean()), 5), "CI": f"[{lo:+.5f},{hi:+.5f}]",
                 "dCatNDCG": round(float(dndcg), 5), "verdict": verdict})
    print(f"[saopaulo] @ottimale FULL(K={opt['FULL'][0]}) vs CTX(K={opt['CTX'][0]}): "
          f"ΔCatMRR={dmrr.mean():+.5f} [{lo:+.5f},{hi:+.5f}] → {verdict}", flush=True)

    # ---------- ridondanza intento↔contesto + sanity K_used vs K*_FULL (tutte le città) ----------
    for c2 in CITIES:
        b2 = ov.build_v(c2, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train",))
        v2 = b2["vs"]["train"]; nA2 = int(b2["attractors"].sum()); ceil2 = nA2 + 2
        r2, mc = redundancy(v2[:, :N_CTX], v2[:, N_CTX:])
        n2 = len(v2); s2 = rng.choice(n2, SIL_N, replace=False) if n2 > SIL_N else np.arange(n2)
        _, kstar_full = internal_curve(v2, ceil2, s2)
        redun.append({"city": c2, "R2_e_from_c": round(r2, 4), "mean_abs_corr": round(mc, 4),
                      "K_used": K_USED[c2], "Kstar_FULL_internal": kstar_full, "ceil_A2": ceil2,
                      "K_match": "OK" if kstar_full == K_USED[c2] else f"diff({kstar_full}vs{K_USED[c2]})"})
        print(f"[{c2}] R²(e|c̃)={r2:.3f} |corr|={mc:.3f}  K_used={K_USED[c2]} K*_FULL_interno={kstar_full}", flush=True)

    OUT = CLEAN / "outputs_results" / "validation"
    pd.DataFrame(kcurve).to_csv(OUT / "saopaulo_diagnosis.csv", index=False)
    pd.DataFrame(redun).to_csv(OUT / "intent_redundancy.csv", index=False)

    print("\n=== (1) CRITERI INTERNI São Paulo — K* per vista (no Cat-MRR) ===")
    print(pd.DataFrame(kcurve).to_string(index=False))
    print("\n=== (2) Confronto al K/ε OTTIMALE di ciascuna vista ===")
    print(pd.DataFrame(summ).to_string(index=False))
    print("\n=== (3) Ridondanza intento↔contesto + (4) sanity K_used vs K*_FULL ===")
    print(pd.DataFrame(redun).to_string(index=False))
    print("  (R² alto = e ridondante col contesto → aggiungerlo rumoreggia)")
    print("\n→ saopaulo_diagnosis.csv, intent_redundancy.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
