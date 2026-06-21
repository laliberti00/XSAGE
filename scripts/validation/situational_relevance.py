"""LOCAL test: is the situational layer a CATEGORY-relevance enhancer? Does
adding the per-situation direction make the recommender hit the TRUE next-item's
MACRO more often than the situation-blind backbone? (The right axis for SIT: its
lever is category-direction, not long-tail.) Params FIXED.

Configs: BASE (κ=0) · SIT (s_B + κ·γ·Σ_k r_k·b̃^(k)) · UNI_mean (s_B + κ·γ·b̄).
NB: b̃^(k) is LEARNED from TRAINING next-item macros per situation → behaviour-
based, not the (weakly-predictive) intent e. So Cat-Hit gain ⇒ the situation's
learned category tendency generalises to the TRUE test macro (the M_C check).

Metrics per request (ordered top-20):
  Cat-Hit@{5,10,20}: true next-macro present among top-k items' macros?
  Cat-MRR / Cat-NDCG: position of FIRST item with the true macro.
  Item control: R@20, NDCG@20 on the true ITEM (do-no-harm on item accuracy).
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
KAPPAS = [0.0, 0.25, 0.5, 1.0, 2.0]
CONFIGS = ["SIT", "UNI_mean"]      # BASE == κ=0 of either
K_TOP, BATCH = 20, 1024
SEED, BOOT = 42, 400


def per_user_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v.astype(np.float64)); np.add.at(c, inv, 1)
    return float((s / c).mean())


def score_city(prep):
    ds = prep["ds"]; df = ds["df_test"]; n_items = ds["n_items"]; n = len(df)
    u_test = df["u_idx"].values.astype(np.int64)
    i_tgt = df["i_idx"].values.astype(np.int64)
    sb_full = prep["scores_blind_full"]; excl = prep["excluded"]
    mem = prep["membership_test"]; b_z = prep["b_z"]; icm = prep["item_cat_macro"]
    gamma = prep["gamma"]; b_bar = b_z.mean(0)
    true_macro = icm[i_tgt]                                  # true next-item macro
    store = {cf: {kp: {"chit5": np.zeros(n), "chit10": np.zeros(n), "chit20": np.zeros(n),
                       "cmrr": np.zeros(n), "cndcg": np.zeros(n),
                       "hit": np.zeros(n), "ndcg": np.zeros(n)} for kp in KAPPAS}
             for cf in CONFIGS}
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u_test[bs:be]; i_b = i_tgt[bs:be]
        tm = true_macro[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        g = gamma[bs:be].astype(np.float32)[:, None]
        nud = {"SIT": g * (mem[bs:be].astype(np.float32) @ b_z)[:, icm],
               "UNI_mean": g * b_bar[icm][None, :]}
        for cf in CONFIGS:
            for kp in KAPPAS:
                S = sb if kp == 0.0 else sb + kp * nud[cf]
                S = S.copy()
                for j in range(be - bs):
                    c = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
                    if len(c): S[j, c] = -np.inf
                # ordered top-20
                part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
                order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
                top = np.take_along_axis(part, order, 1)             # (B,20) ordered
                macros = icm[top]                                    # (B,20)
                match = macros == tm[:, None]                        # (B,20)
                d = store[cf][kp]
                d["chit5"][bs:be] = match[:, :5].any(1)
                d["chit10"][bs:be] = match[:, :10].any(1)
                d["chit20"][bs:be] = match.any(1)
                first = np.where(match.any(1), match.argmax(1) + 1, 0)   # 1-based, 0 if none
                d["cmrr"][bs:be] = np.where(first > 0, 1.0 / first, 0.0)
                d["cndcg"][bs:be] = np.where(first > 0, 1.0 / np.log2(first + 1.0), 0.0)
                # item-level
                s_tgt = S[np.arange(be - bs), i_b]
                ranks = (S > s_tgt[:, None]).sum(1) + 1
                d["hit"][bs:be] = (ranks <= K_TOP)
                d["ndcg"][bs:be] = np.where(ranks <= K_TOP, 1.0 / np.log2(ranks + 1.0), 0.0)
    return store, u_test


def main():
    rng = np.random.default_rng(SEED)
    rows, boot_rows = [], []
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        store, u = score_city(prep)
        base = store["SIT"][0.0]
        for cf in CONFIGS:
            for kp in KAPPAS:
                d = store[cf][kp]
                rows.append({"city": city, "config": cf, "kappa": kp,
                             "CatHit5": round(float(d["chit5"].mean()), 5),
                             "CatHit10": round(float(d["chit10"].mean()), 5),
                             "CatHit20": round(float(d["chit20"].mean()), 5),
                             "CatMRR": round(float(d["cmrr"].mean()), 5),
                             "CatNDCG": round(float(d["cndcg"].mean()), 5),
                             "R20": round(per_user_mean(d["hit"], u), 5),
                             "NDCG20": round(per_user_mean(d["ndcg"], u), 5),
                             "rudder": RUDDER[city]})
        # bootstrap ΔCatHit10 (SIT−BASE) at κ=0.5 and κ=1
        for kp in [0.5, 1.0]:
            dd = store["SIT"][kp]["chit10"] - base["chit10"]; nrq = len(u)
            bs_ = [dd[rng.integers(0, nrq, nrq)].mean() for _ in range(BOOT)]
            boot_rows.append({"city": city, "kappa": kp,
                              "dCatHit10": round(float(dd.mean()), 5),
                              "CI": f"[{np.percentile(bs_,2.5):+.4f},{np.percentile(bs_,97.5):+.4f}]"})
        print(f"[{city}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CLEAN / "outputs_results" / "validation" / "situational_relevance.csv", index=False)

    print("\n=== CURVA κ: Cat-Hit@10 / Cat-MRR / R@20 (SIT) + BASE ===")
    for city in CITIES:
        sub = df[(df.city == city) & (df.config == "SIT")]
        b = sub[sub.kappa == 0].iloc[0]
        print(f"\n  {city} (timone={RUDDER[city]}) BASE: CatHit10={b.CatHit10:.4f} "
              f"CatMRR={b.CatMRR:.4f} R20={b.R20:.4f}")
        for r in sub.itertuples():
            dch = (r.CatHit10 - b.CatHit10) / b.CatHit10 * 100 if b.CatHit10 else 0
            dR = (r.R20 - b.R20) / b.R20 * 100 if b.R20 else 0
            print(f"    κ={r.kappa:<4} CatHit10={r.CatHit10:.4f}({dch:+.1f}%) "
                  f"CatMRR={r.CatMRR:.4f} CatNDCG={r.CatNDCG:.4f}  "
                  f"R20={r.R20:.4f}({dR:+.1f}%)")

    print("\n=== SIT vs UNI_mean su Cat-Hit@10 (la specificità serve?) ===")
    for city in CITIES:
        for kp in [0.5, 1.0]:
            s = df[(df.city == city) & (df.config == "SIT") & (df.kappa == kp)].CatHit10.iloc[0]
            m = df[(df.city == city) & (df.config == "UNI_mean") & (df.kappa == kp)].CatHit10.iloc[0]
            print(f"  {city:<11} κ={kp}: SIT={s:.4f}  UNI_mean={m:.4f}  Δ={s-m:+.4f}")

    print("\n=== Significatività ΔCat-Hit@10 (SIT−BASE), bootstrap 400 ===")
    for r in boot_rows:
        print(f"  {r['city']:<11} κ={r['kappa']}: ΔCatHit10={r['dCatHit10']:+.5f} {r['CI']}")
    print("\n→ situational_relevance.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
