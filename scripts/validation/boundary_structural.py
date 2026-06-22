"""Fix the 2 structural code↔paper gaps, deciding ON THE NUMBERS (Cat-MRR, κ=0.25):
  A) γ_S: replace the hardcoded 0.5 with the declared γ_S = 1/|T| (|T| = boundary
     competing-set size). Diagnose |T| distribution; measure effect.
  B) L3: wire eq.18 boundary disambiguation r̃_k ∝ r_k·T_{z_prev,k} (causal z_prev,
     T from TRAIN situation sequences); measure if it improves/inert/hurts.
  + interaction (L3 + γ_S=1/|T|) vs current (no L3, γ_S=0.5).

CORE config: m_sel≡1, λ=0, κ=0.25 → ŝ = s_B + 0.25·γ_S·Σ_k r̃_k·b̃^(k)_{c(i)}.
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
from xsage.l3_projection import estimate_transition, boundary_disambiguate
from pipeline.step02_models.xsage.l2_comprehension import _assign

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
KAPPA = 0.25
K_TOP, BATCH, SEED, BOOT = 20, 1024, 42, 1500


def per_user_mean(v, u):
    uq, inv = np.unique(u, return_inverse=True)
    s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v.astype(np.float64)); np.add.at(c, inv, 1)
    return float((s / c).mean())


def causal_prev_situation(ds, z_train, z_val, z_test):
    """z_prev per test request = situation of the user's immediately-preceding
    interaction in the full causal timeline (train∪val∪earlier-test)."""
    def frame(df, z, tidx):
        return pd.DataFrame({"u": df["u_idx"].values,
                             "t": pd.to_datetime(df["time_local"].values),
                             "sit": z, "tidx": tidx})
    n_te = len(ds["df_test"])
    a = frame(ds["df_train"], z_train, np.full(len(z_train), -1))
    b = frame(ds["df_val"], z_val, np.full(len(z_val), -1))
    c = frame(ds["df_test"], z_test, np.arange(n_te))
    allr = pd.concat([a, b, c], ignore_index=True).sort_values(["u", "t"], kind="stable").reset_index(drop=True)
    allr["prev"] = allr.groupby("u", sort=False)["sit"].shift(1)
    tr = allr[allr.tidx >= 0]
    prev = np.full(n_te, -1, np.int64)
    prev[tr.tidx.values] = tr["prev"].fillna(-1).astype(np.int64).values
    return prev


def score(prep, mem, gamma):
    ds = prep["ds"]; df = ds["df_test"]; n = len(df); icm = prep["item_cat_macro"]
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64)
    tm = icm[i_t]; sb_full = prep["scores_blind_full"]; excl = prep["excluded"]; b_z = prep["b_z"]
    cmrr = np.zeros(n); cndcg = np.zeros(n); hit = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        sb = sb_full[u_b].astype(np.float32, copy=True)
        nud = gamma[bs:be][:, None].astype(np.float32) * (mem[bs:be].astype(np.float32) @ b_z)[:, icm]
        S = (sb + KAPPA * nud).copy()
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


def main():
    rng = np.random.default_rng(SEED)
    rows, tdist = [], []
    for city in CITIES:
        prep, fit, asg = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        d_te, k_te, comp_te, isb_te = asg
        K = K_FINAL[city]; eps = EPS_FINAL[city]
        Tsize = comp_te.sum(1).astype(np.int64)                  # |T| per request
        mem = prep["membership_test"]
        # --- A: |T| distribution on boundary ---
        bnd = Tsize[isb_te]
        frac2 = float((bnd == 2).mean()) if len(bnd) else 0.0
        frac3 = float((bnd >= 3).mean()) if len(bnd) else 0.0
        tdist.append({"city": city, "n_boundary": int(isb_te.sum()),
                      "boundary_frac": round(float(isb_te.mean()), 4),
                      "T2_share": round(frac2, 4), "T3plus_share": round(frac3, 4),
                      "T_max": int(bnd.max()) if len(bnd) else 0})
        gamma_05 = np.where(isb_te, 0.5, 1.0).astype(np.float32)
        gamma_1T = (1.0 / np.maximum(Tsize, 1)).astype(np.float32)
        # --- B: L3 wire (causal z_prev, T from train) ---
        built = ov.build_v(city, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train", "val"))
        z_train = fit.core_label.astype(np.int64)
        z_val = _assign(built["vs"]["val"], fit.prototypes, eps)[1].astype(np.int64)
        # per-user TRAIN situation sequences (time order)
        dtr = ds_sorted = prep["ds"]["df_train"]
        ordtr = np.argsort(prep["ds"]["df_train"]["time_local"].values, kind="stable")
        seqs = {}
        uu = prep["ds"]["df_train"]["u_idx"].values
        for idx in ordtr:
            seqs.setdefault(int(uu[idx]), []).append(int(z_train[idx]))
        T, _ = estimate_transition([np.array(s) for s in seqs.values()], K)
        z_prev = causal_prev_situation(prep["ds"], z_train, z_val, k_te.astype(np.int64))
        mem_l3 = boundary_disambiguate(mem, T, z_prev, isb_te)

        configs = {"base": (mem, gamma_05), "gS_1T": (mem, gamma_1T),
                   "L3": (mem_l3, gamma_05), "L3_gS1T": (mem_l3, gamma_1T)}
        res = {}
        for cf, (m, g) in configs.items():
            res[cf] = score(prep, m, g)
        u = res["base"][3]; n = len(u)
        bcm, bcn, bh = res["base"][0], res["base"][1], res["base"][2]
        for cf in configs:
            cm, cn, h, _ = res[cf]
            dmrr = cm - bcm
            bsd = np.array([dmrr[rng.integers(0, n, n)].mean() for _ in range(BOOT)]) if cf != "base" else np.zeros(BOOT)
            lo, hi = np.percentile(bsd, [2.5, 97.5])
            rows.append({"city": city, "config": cf,
                         "CatMRR": round(float(cm.mean()), 5),
                         "dMRR_vs_base": round(float(dmrr.mean()), 5),
                         "CI_lo": round(float(lo), 5), "CI_hi": round(float(hi), 5),
                         "sig": "—" if cf == "base" else ("SI" if (lo > 0 or hi < 0) else "no"),
                         "CatNDCG": round(float(cn.mean()), 5),
                         "R20": round(per_user_mean(h, u), 5),
                         "T2_share": round(frac2, 3), "T3plus_share": round(frac3, 3)})
        print(f"[{city}] |T|: bnd={isb_te.mean():.1%} |T|=2 {frac2:.0%} |T|>=3 {frac3:.0%} (max {bnd.max() if len(bnd) else 0})", flush=True)

    df = pd.DataFrame(rows); td = pd.DataFrame(tdist)
    df.to_csv(CLEAN / "outputs_results" / "validation" / "boundary_structural.csv", index=False)

    print("\n=== PARTE A — distribuzione |T| sui boundary ===")
    print(td.to_string(index=False))
    print("\n=== Effetto sul Cat-MRR (Δ vs base, bootstrap CI 95%), R@20, CatNDCG ===")
    for city in CITIES:
        print(f"\n  {city}")
        for r in df[df.city == city].itertuples():
            tag = "" if r.config == "base" else f" Δ={r.dMRR_vs_base:+.5f}[{r.CI_lo:+.5f},{r.CI_hi:+.5f}] {r.sig}"
            print(f"    {r.config:<9} CatMRR={r.CatMRR:.5f}{tag}  NDCG={r.CatNDCG:.4f} R20={r.R20:.4f}")
    print("\n→ boundary_structural.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
