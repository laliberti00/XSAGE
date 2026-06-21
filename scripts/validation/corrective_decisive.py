"""DECISIVE test: does the SITUATIONAL correction (per-situation bias b^(k),
applied EVERYWHERE — no gate, no long-tail term) improve / leave unchanged /
hurt accuracy & fairness vs the bare backbone, on all 5 cities?

ŝ = s_B + κ · γ_S(v) · Σ_k r_k · b̃^(k)_{c(i)}      (m_sel≡1, λ=0)
κ sweep {0,0.25,0.5,1,2}; κ=0 must equal BASE exactly (do-no-harm check).
Metrics on TEST per (city, κ): R@20, NDCG@20 (per-user), LT@20 per-request (APL),
Long-tail Coverage, Gini. Bootstrap CI on the Δ (corr−base) at κ=1.
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
K_TOP, BATCH = 20, 1024
BOOT, SEED = 500, 42


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0 or x.sum() == 0: return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def per_user_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v.astype(np.float64)); np.add.at(c, inv, 1)
    return float((s / c).mean())


def score_city(prep):
    """Return per-κ dict of per-request hit/ndcg/lt + topk, computing the
    situational nudge once per batch and reusing across κ."""
    ds = prep["ds"]; df_test = ds["df_test"]; n_items = ds["n_items"]
    n_test = len(df_test)
    u_test = df_test["u_idx"].values.astype(np.int64)
    i_target = df_test["i_idx"].values.astype(np.int64)
    sb_full = prep["scores_blind_full"]; excl = prep["excluded"]
    G1 = prep["G1_mask"]; mem = prep["membership_test"]; b_z = prep["b_z"]
    icm = prep["item_cat_macro"]; gamma = prep["gamma"]
    out = {k: {"hit": np.zeros(n_test, np.float32), "ndcg": np.zeros(n_test, np.float32),
               "lt": np.zeros(n_test, np.float32),
               "topk": np.zeros((n_test, K_TOP), np.int32)} for k in KAPPAS}
    for bs in range(0, n_test, BATCH):
        be = min(n_test, bs + BATCH)
        u_b = u_test[bs:be]; i_b = i_target[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        # situational nudge unit (λ=0, m_sel≡1): γ · (membership·b_z)_{macro(i)}
        learned = (mem[bs:be].astype(np.float32) @ b_z)[:, icm]       # (B, I)
        nudge = gamma[bs:be].astype(np.float32)[:, None] * learned
        for kp in KAPPAS:
            S = sb if kp == 0.0 else sb + kp * nudge
            S = S.copy()
            for j in range(be - bs):
                cols = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
                if len(cols): S[j, cols] = -np.inf
            s_tgt = S[np.arange(be - bs), i_b]
            ranks = (S > s_tgt[:, None]).sum(1) + 1
            out[kp]["hit"][bs:be] = (ranks <= K_TOP)
            out[kp]["ndcg"][bs:be] = np.where(ranks <= K_TOP, 1.0 / np.log2(ranks + 1.0), 0.0)
            part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
            out[kp]["topk"][bs:be] = part
            out[kp]["lt"][bs:be] = G1[part].mean(1)
    return out, u_test, G1, n_items


def main():
    rng = np.random.default_rng(SEED)
    rows, boot_rows = [], []
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        out, u, G1, n_items = score_city(prep)
        nG1 = int(G1.sum())
        base = out[0.0]
        for kp in KAPPAS:
            d = out[kp]
            exposure = np.bincount(d["topk"].flatten(), minlength=n_items).astype(float)
            cov = float(np.isin(np.where(G1)[0], np.unique(d["topk"])).mean()) if nG1 else 0.0
            rows.append({"city": city, "kappa": kp,
                         "R20": round(per_user_mean(d["hit"], u), 5),
                         "NDCG20": round(per_user_mean(d["ndcg"], u), 5),
                         "LT_req": round(float(d["lt"].mean()), 5),
                         "LT_coverage": round(cov, 5),
                         "Gini": round(gini(exposure), 5),
                         "rudder": RUDDER[city]})
        # do-no-harm check
        same = np.allclose(base["hit"], out[0.0]["hit"])
        # bootstrap Δ at κ=1.0 on per-request lt and hit
        kp = 1.0; n = len(u)
        d_lt = out[kp]["lt"] - base["lt"]; d_hit = out[kp]["hit"] - base["hit"]
        bl, bh = [], []
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            bl.append(d_lt[idx].mean()); bh.append(d_hit[idx].mean())
        boot_rows.append({"city": city,
                          "dLT_mean": round(float(d_lt.mean()), 5),
                          "dLT_CI": f"[{np.percentile(bl,2.5):+.4f},{np.percentile(bl,97.5):+.4f}]",
                          "dHit_mean": round(float(d_hit.mean()), 5),
                          "dHit_CI": f"[{np.percentile(bh,2.5):+.4f},{np.percentile(bh,97.5):+.4f}]"})
        print(f"[{city}] do-no-harm κ=0≡BASE: {same}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(CLEAN / "outputs_results" / "validation" / "corrective_decisive.csv", index=False)

    print("\n=== CURVA κ per città (R@20 / NDCG / LT_req / LC / Gini) ===")
    for city in CITIES:
        print(f"\n  {city}  (timone={RUDDER[city]})")
        sub = df[df.city == city]
        b = sub[sub.kappa == 0.0].iloc[0]
        for r in sub.itertuples():
            dR = (r.R20 - b.R20) / b.R20 * 100 if b.R20 else 0
            dLT = (r.LT_req - b.LT_req) / b.LT_req * 100 if b.LT_req else 0
            print(f"    κ={r.kappa:<4} R={r.R20:.4f}({dR:+.1f}%) N={r.NDCG20:.4f} "
                  f"LT={r.LT_req:.4f}({dLT:+.0f}%) LC={r.LT_coverage:.3f} Gini={r.Gini:.4f}")

    print("\n=== Significatività Δ(κ=1 − BASE), bootstrap 500 sulle richieste ===")
    for r in boot_rows:
        print(f"  {r['city']:<11} ΔLT={r['dLT_mean']:+.4f} {r['dLT_CI']}   "
              f"ΔHit={r['dHit_mean']:+.4f} {r['dHit_CI']}")
    print(f"\n→ corrective_decisive.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
