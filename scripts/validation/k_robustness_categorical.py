"""Robustness of the categorical gain vs the recommendation cutoff K.
Is ΔCat-MRR/NDCG(SIT-BASE) at κ=0.25 a PLATEAU (holds over K), a WINDOW (only some
K), or FRAGILE (sign flip / only K=20)? Same rigour (bootstrap 95% CI).

K here = list cutoff @K (NOT the situational K). Grid {5,10,20,50,100}. κ=0.25 fixed.
We compute the FIRST correct-macro rank (in the ordered top-100) and the item rank
ONCE per config, then derive every @K metric by thresholding.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("ov", str(CLEAN / "scripts" / "overnight_selection.py"))
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
RUDDER = {"saopaulo": 0.191, "nyc_tist": 0.216, "bangkok": 0.079, "istanbul": 0.043, "tokyo_tist": 0.009}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
KAPPA = 0.25
K_GRID = [5, 10, 20, 50, 100]
TOPN, BATCH, SEED, BOOT = 100, 1024, 42, 1500


def per_user_rate(rank, u, K):
    ok = (rank <= K).astype(np.float64)
    uq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, ok); np.add.at(c, inv, 1)
    return float((s / c).mean())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for city in CITIES:
        prep, fit, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        ds = prep["ds"]; df = ds["df_test"]; n = len(df); icm = prep["item_cat_macro"]
        u = df["u_idx"].values.astype(np.int64); i_tgt = df["i_idx"].values.astype(np.int64)
        tm = icm[i_tgt]
        sb_full = prep["scores_blind_full"]; excl = prep["excluded"]
        mem = prep["membership_test"]; b_z = prep["b_z"]; gamma = prep["gamma"]; b_bar = b_z.mean(0)
        nud = {"BASE": None,
               "SIT": lambda bs, be: gamma[bs:be][:, None].astype(np.float32) *
                      (mem[bs:be].astype(np.float32) @ b_z)[:, icm],
               "UNI_mean": lambda bs, be: gamma[bs:be][:, None].astype(np.float32) * b_bar[icm][None, :]}
        first = {cf: np.full(n, 999, np.int32) for cf in nud}   # first correct-macro rank (1-based)
        irank = {cf: np.zeros(n, np.int32) for cf in nud}        # true-item rank
        for bs in range(0, n, BATCH):
            be = min(n, bs + BATCH); u_b = u[bs:be]
            sb = sb_full[u_b].astype(np.float32, copy=True)
            for cf, fn in nud.items():
                S = sb if cf == "BASE" else sb + KAPPA * fn(bs, be)
                S = S.copy()
                for j in range(be - bs):
                    cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
                    if len(cc): S[j, cc] = -np.inf
                part = np.argpartition(-S, TOPN - 1, axis=1)[:, :TOPN]
                order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
                macros = icm[np.take_along_axis(part, order, 1)]    # (B,100) ordered
                match = macros == tm[bs:be, None]
                has = match.any(1)
                first[cf][bs:be] = np.where(has, match.argmax(1) + 1, 999)
                s_tgt = S[np.arange(be - bs), i_tgt[bs:be]]
                irank[cf][bs:be] = (S > s_tgt[:, None]).sum(1) + 1
        print(f"[{city}] scored", flush=True)

        # derive @K metrics + bootstrap (resample requests once per iter, all K)
        def mrr(f, K): return np.where(f <= K, 1.0 / np.maximum(f, 1), 0.0)
        def ndcg(f, K): return np.where(f <= K, 1.0 / np.log2(np.maximum(f, 1) + 1.0), 0.0)
        boots = {(K, m): np.zeros(BOOT) for K in K_GRID for m in ["mrr", "ndcg", "mrr_um"]}
        for t in range(BOOT):
            idx = rng.integers(0, n, n)
            for K in K_GRID:
                boots[(K, "mrr")][t] = mrr(first["SIT"][idx], K).mean() - mrr(first["BASE"][idx], K).mean()
                boots[(K, "ndcg")][t] = ndcg(first["SIT"][idx], K).mean() - ndcg(first["BASE"][idx], K).mean()
                boots[(K, "mrr_um")][t] = mrr(first["SIT"][idx], K).mean() - mrr(first["UNI_mean"][idx], K).mean()
        for K in K_GRID:
            dmrr = mrr(first["SIT"], K).mean() - mrr(first["BASE"], K).mean()
            dndcg = ndcg(first["SIT"], K).mean() - ndcg(first["BASE"], K).mean()
            dhit = float((first["SIT"] <= K).mean() - (first["BASE"] <= K).mean())
            ci_l, ci_h = np.percentile(boots[(K, "mrr")], [2.5, 97.5])
            nl, nh = np.percentile(boots[(K, "ndcg")], [2.5, 97.5])
            ul, uh = np.percentile(boots[(K, "mrr_um")], [2.5, 97.5])
            rB = per_user_rate(irank["BASE"], u, K); rS = per_user_rate(irank["SIT"], u, K)
            rows.append({"city": city, "K": K, "rudder": RUDDER[city],
                         "dCatMRR": round(float(dmrr), 5), "MRR_CIlo": round(float(ci_l), 5),
                         "MRR_CIhi": round(float(ci_h), 5), "MRR_sig": "SI" if ci_l > 0 else "no",
                         "dCatNDCG": round(float(dndcg), 5), "NDCG_sig": "SI" if nl > 0 else "no",
                         "dCatHit": round(dhit, 5),
                         "R@K_BASE": round(rB, 5), "R@K_SIT": round(rS, 5),
                         "dR_pct": round((rS - rB) / rB * 100 if rB else 0, 2),
                         "dMRR_SITvsUNImean": round(float(mrr(first["SIT"], K).mean() - mrr(first["UNI_mean"], K).mean()), 5),
                         "UM_sig": "SI" if ul > 0 else "no"})
    dfo = pd.DataFrame(rows)
    dfo.to_csv(CLEAN / "outputs_results" / "validation" / "k_robustness_categorical.csv", index=False)

    print("\n=== CURVA ΔCat-MRR(SIT−BASE) vs K [CI 95%] + costo item + specificità ===")
    for city in CITIES:
        print(f"\n  {city} (timone={RUDDER[city]})")
        for r in dfo[dfo.city == city].itertuples():
            sig = "✅" if r.MRR_sig == "SI" else "❌"
            um = "✅spec" if r.UM_sig == "SI" else "—"
            print(f"    @{r.K:<3} ΔMRR={r.dCatMRR:+.4f}[{r.MRR_CIlo:+.4f},{r.MRR_CIhi:+.4f}]{sig} "
                  f"ΔNDCG={r.dCatNDCG:+.4f}({r.NDCG_sig}) ΔHit={r.dCatHit:+.4f}  "
                  f"R@K {r.dR_pct:+.1f}%  SIT-UNImean={r.dMRR_SITvsUNImean:+.4f}{um}")

    print("\n=== VERDETTO per città (robusto/finestra/fragile su Cat-MRR) ===")
    for city in CITIES:
        s = dfo[dfo.city == city]
        sigs = (s.MRR_sig == "SI").values
        signs = np.sign(s.dCatMRR.values)
        if sigs.all() and (signs > 0).all():
            v = "ROBUSTO (signif. e >0 su tutto K)"
        elif (signs < 0).any():
            v = "FRAGILE (cambia segno)"
        elif sigs.any():
            ks = s.K.values[sigs]
            v = f"FINESTRA (signif. @K={list(ks)})"
        else:
            v = "non significativo a nessun K"
        print(f"  {city:<11} {v}")
    print("\n→ k_robustness_categorical.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
