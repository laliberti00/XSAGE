"""DECISIVE: does the SITUATIONAL correction win on MINIMAL INTERVENTION — same
Δfairness with FEWER lists touched and MORE accuracy preserved than uniform?

Three configs (same backbone, λ-style nudges, κ sweep):
  SIT      : ŝ = s_B + κ·γ·Σ_k r_k·b̃^(k)_{c(i)}      (per-situation direction)
  UNI_glob : ŝ = s_B + κ·b^LT_i                        (flat long-tail push, CPFair-ish)
  UNI_mean : ŝ = s_B + κ·γ·b̄_{c(i)},  b̄=mean_k b̃^(k)  (same mechanism, mean direction)
SIT vs UNI_mean isolates pure situational specificity.

Key metric: lists_touched = fraction of test requests whose top-20 SET changed vs
BASE (κ=0). Compared at MATCHED Δfairness (interpolate κ to base_LT×{1.5,2.0}).
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
KAPPAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
CONFIGS = ["SIT", "UNI_glob", "UNI_mean"]
K_TOP, BATCH = 20, 1024
TARGETS = [1.5, 2.0]      # base_LT × these
SEED, BOOT = 42, 400


def gini(x):
    x = np.sort(np.asarray(x, float)); n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def per_user_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v.astype(np.float64)); np.add.at(c, inv, 1)
    return float((s / c).mean())


def score_city(prep):
    ds = prep["ds"]; df = ds["df_test"]; n_items = ds["n_items"]; n = len(df)
    u_test = df["u_idx"].values.astype(np.int64)
    i_tgt = df["i_idx"].values.astype(np.int64)
    sb_full = prep["scores_blind_full"]; excl = prep["excluded"]; G1 = prep["G1_mask"]
    mem = prep["membership_test"]; b_z = prep["b_z"]; icm = prep["item_cat_macro"]
    gamma = prep["gamma"]; b_bar = b_z.mean(0)
    res = {cf: {kp: {"hit": np.zeros(n, np.float32), "ndcg": np.zeros(n, np.float32),
                     "lt": np.zeros(n, np.float32), "changed": np.zeros(n, bool),
                     "topk": np.zeros((n, K_TOP), np.int32)} for kp in KAPPAS} for cf in CONFIGS}
    base_top = np.zeros((n, K_TOP), np.int32)
    sit_dlt_by_sit = {}                      # per-situation ΔLT at κ=1 (SIT)
    z = prep["z_test"]
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u_test[bs:be]; i_b = i_tgt[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        g = gamma[bs:be].astype(np.float32)[:, None]
        nud = {"SIT": g * (mem[bs:be].astype(np.float32) @ b_z)[:, icm],
               "UNI_mean": g * b_bar[icm][None, :],
               "UNI_glob": G1.astype(np.float32)[None, :].repeat(be - bs, 0)}
        # base top-K (κ=0)
        Sb = sb.copy()
        for j in range(be - bs):
            c = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(c): Sb[j, c] = -np.inf
        bt = np.argpartition(-Sb, K_TOP - 1, axis=1)[:, :K_TOP]
        base_top[bs:be] = bt; bt_sorted = np.sort(bt, axis=1)
        s_tgt0 = Sb[np.arange(be - bs), i_b]
        for cf in CONFIGS:
            for kp in KAPPAS:
                S = Sb if kp == 0.0 else sb + kp * nud[cf]
                if kp != 0.0:
                    S = S.copy()
                    for j in range(be - bs):
                        c = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
                        if len(c): S[j, c] = -np.inf
                s_tgt = S[np.arange(be - bs), i_b]
                ranks = (S > s_tgt[:, None]).sum(1) + 1
                res[cf][kp]["hit"][bs:be] = (ranks <= K_TOP)
                res[cf][kp]["ndcg"][bs:be] = np.where(ranks <= K_TOP, 1.0 / np.log2(ranks + 1.0), 0.0)
                part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
                res[cf][kp]["topk"][bs:be] = part
                res[cf][kp]["lt"][bs:be] = G1[part].mean(1)
                res[cf][kp]["changed"][bs:be] = ~(np.sort(part, axis=1) == bt_sorted).all(1)
    return res, u_test, G1, n_items, z


def main():
    rng = np.random.default_rng(SEED)
    rows, conc_rows, boot_rows = [], [], []
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        res, u, G1, n_items, z = score_city(prep)
        nG1 = int(G1.sum())
        base_lt = float(res["SIT"][0.0]["lt"].mean())
        base_R = per_user_mean(res["SIT"][0.0]["hit"], u)
        for cf in CONFIGS:
            for kp in KAPPAS:
                d = res[cf][kp]
                exposure = np.bincount(d["topk"].flatten(), minlength=n_items).astype(float)
                cov = float(np.isin(np.where(G1)[0], np.unique(d["topk"])).mean()) if nG1 else 0.0
                rows.append({"city": city, "config": cf, "kappa": kp,
                             "LT_req": round(float(d["lt"].mean()), 5),
                             "LT_cov": round(cov, 5), "Gini": round(gini(exposure), 5),
                             "R20": round(per_user_mean(d["hit"], u), 5),
                             "NDCG20": round(per_user_mean(d["ndcg"], u), 5),
                             "lists_touched": round(float(d["changed"].mean()), 5),
                             "rudder": RUDDER[city]})
        # per-situation ΔLT concentration at κ=1 (SIT)
        dlt = res["SIT"][1.0]["lt"] - res["SIT"][0.0]["lt"]
        for k in range(K_FINAL[city]):
            m = z == k
            if m.any():
                conc_rows.append({"city": city, "situation": k, "n_req": int(m.sum()),
                                  "mean_dLT": round(float(dlt[m].mean()), 5),
                                  "share_of_total_dLT": round(float(dlt[m].sum() / max(dlt.sum(), 1e-9)), 4)})
        # significance: SIT vs UNI_mean at κ=1, bootstrap requests
        n = len(u)
        sit_lt = res["SIT"][1.0]["lt"]; um_lt = res["UNI_mean"][1.0]["lt"]
        sit_ch = res["SIT"][1.0]["changed"].astype(float); um_ch = res["UNI_mean"][1.0]["changed"].astype(float)
        sit_h = res["SIT"][1.0]["hit"]; um_h = res["UNI_mean"][1.0]["hit"]
        dch, dh = [], []
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            dch.append(sit_ch[idx].mean() - um_ch[idx].mean())
            dh.append(sit_h[idx].mean() - um_h[idx].mean())
        boot_rows.append({"city": city,
                          "d_lists_SIT_minus_UNImean": round(float(sit_ch.mean() - um_ch.mean()), 4),
                          "CI": f"[{np.percentile(dch,2.5):+.4f},{np.percentile(dch,97.5):+.4f}]",
                          "d_hit": round(float(sit_h.mean() - um_h.mean()), 4)})
        print(f"[{city}] base_LT={base_lt:.4f} base_R={base_R:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CLEAN / "outputs_results" / "validation" / "minimal_intervention.csv", index=False)
    pd.DataFrame(conc_rows).to_csv(CLEAN / "outputs_results" / "validation" / "fairness_concentration.csv", index=False)

    # match at target Δfairness (base_LT × {1.5, 2.0}) — interpolate κ → lists & R
    print("\n=== CONFRONTO a parità di Δfairness (LT@20 target = base×1.5 e ×2.0) ===")
    for city in CITIES:
        sub = df[df.city == city]
        base_lt = float(sub[(sub.config == "SIT") & (sub.kappa == 0)].LT_req.iloc[0])
        base_R = float(sub[(sub.config == "SIT") & (sub.kappa == 0)].R20.iloc[0])
        print(f"\n  {city} (timone={RUDDER[city]}, base LT={base_lt:.4f} R={base_R:.4f})")
        for tgt in TARGETS:
            tl = base_lt * tgt
            print(f"    target LT={tl:.4f} (×{tgt}):")
            for cf in CONFIGS:
                s = sub[sub.config == cf].sort_values("LT_req")
                lt = s.LT_req.values
                if tl <= lt.max():
                    touched = float(np.interp(tl, lt, s.lists_touched.values))
                    R = float(np.interp(tl, lt, s.R20.values))
                    print(f"      {cf:<9} liste_toccate={touched:.1%}  R@20={R:.4f} "
                          f"({(R-base_R)/base_R*100:+.1f}%)")
                else:
                    print(f"      {cf:<9} non raggiunge il target (LT max {lt.max():.4f})")

    print("\n=== CONCENTRAZIONE del guadagno fairness per situazione (κ=1, SIT) ===")
    cdf = pd.DataFrame(conc_rows)
    for city in CITIES:
        s = cdf[cdf.city == city].sort_values("share_of_total_dLT", ascending=False)
        shares = s.share_of_total_dLT.values
        print(f"  {city:<11} share ΔLT per sit (ord.): {[f'{x:.0%}' for x in shares]}  "
              f"(top sit = {shares[0]:.0%})")

    print("\n=== Significatività SIT vs UNI_mean (κ=1, bootstrap 400) ===")
    for r in boot_rows:
        print(f"  {r['city']:<11} Δliste(SIT−UNImean)={r['d_lists_SIT_minus_UNImean']:+.4f} "
              f"{r['CI']}  Δhit={r['d_hit']:+.4f}")
    print("\n→ minimal_intervention.csv, fairness_concentration.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
