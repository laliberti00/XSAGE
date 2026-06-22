"""Validate L3 (Endsley Projection) on ITS axis: do situations evolve in a
structured / predictable / readable way? (NOT re-ranking — that was inert.)
Causal: T from TRAIN sequences, eval on causal TEST transitions, no future leak.

ASSE 1 structure: row-entropy vs log2 K; KL(T_i || marginal); diagonal (persistence).
ASSE 2 predictivity: predict situation_{t+1} from situation_t via argmax T, vs
        marginal (most frequent) and persistence (stay). Top-1/Top-2 + lift + CI.
ASSE 3 interpretability: characterise the strongest transitions by readable traits
        (typical hour/weekend, dominant intent macro) → derived label.
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
from xsage.l3_projection import estimate_transition
from pipeline.step02_models.xsage.l2_comprehension import _assign

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
N_CTX = len(ov.DEFAULT_ATTRIBUTES)
SEED, BOOT = 42, 1500


def causal_prev(ds, z_tr, z_va, z_te):
    def fr(df, z, tidx):
        return pd.DataFrame({"u": df["u_idx"].values,
                             "t": pd.to_datetime(df["time_local"].values), "sit": z, "tidx": tidx})
    nte = len(ds["df_test"])
    allr = pd.concat([fr(ds["df_train"], z_tr, np.full(len(z_tr), -1)),
                      fr(ds["df_val"], z_va, np.full(len(z_va), -1)),
                      fr(ds["df_test"], z_te, np.arange(nte))],
                     ignore_index=True).sort_values(["u", "t"], kind="stable").reset_index(drop=True)
    allr["prev"] = allr.groupby("u", sort=False)["sit"].shift(1)
    tr = allr[allr.tidx >= 0]
    prev = np.full(nte, -1, np.int64)
    prev[tr.tidx.values] = tr["prev"].fillna(-1).astype(np.int64).values
    return prev


def main():
    rng = np.random.default_rng(SEED)
    s1, s2, s3 = [], [], []
    for city in CITIES:
        prep, fit, asg = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        K = K_FINAL[city]; eps = EPS_FINAL[city]; ds = prep["ds"]
        z_te = asg[1].astype(np.int64); z_tr = fit.core_label.astype(np.int64)
        built = ov.build_v(city, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train", "val"))
        e_tr = built["vs"]["train"][:, N_CTX:]; attr = built["attractors"]
        i2m = ds["idx_to_macro"]; m2i = ds["macro_to_idx"]
        z_va = _assign(built["vs"]["val"], fit.prototypes, eps)[1].astype(np.int64)
        # per-user TRAIN situation sequences (time order)
        ordtr = np.argsort(ds["df_train"]["time_local"].values, kind="stable")
        uu = ds["df_train"]["u_idx"].values; seqs = {}
        for idx in ordtr:
            seqs.setdefault(int(uu[idx]), []).append(int(z_tr[idx]))
        T, _ = estimate_transition([np.array(s) for s in seqs.values()], K)
        marg = np.bincount(z_tr, minlength=K).astype(float); marg /= marg.sum()

        # ---- ASSE 1: structure ----
        logK = np.log2(K)
        rowH = np.array([-np.sum(T[i][T[i] > 0] * np.log2(T[i][T[i] > 0])) for i in range(K)])
        klrow = np.array([float(np.sum(T[i][T[i] > 0] * np.log2(T[i][T[i] > 0] / np.maximum(marg[T[i] > 0], 1e-12)))) for i in range(K)])
        diag = np.diag(T)
        s1.append({"city": city, "K": K, "rowH_norm": round(float(rowH.mean() / logK), 4),
                   "KL_vs_marginal": round(float(klrow.mean()), 4),
                   "diag_persistence": round(float(diag.mean()), 4),
                   "diag_lift_vs_marg": round(float((diag - marg).mean()), 4)})

        # ---- ASSE 2: predictivity (causal) ----
        zp = causal_prev(ds, z_tr, z_va, z_te)
        ok = zp >= 0; zt = z_te[ok]; zpp = zp[ok]; n = len(zt)
        proj_pred = T[zpp].argmax(1)
        top2 = np.argsort(-T[zpp], axis=1)[:, :2]
        proj1 = (proj_pred == zt); proj2 = (top2 == zt[:, None]).any(1)
        marg_pred = int(marg.argmax()); marg_ok = (zt == marg_pred)
        pers_ok = (zpp == zt)
        def bootci(arr):
            b = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
            return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
        lm = proj1.astype(float) - marg_ok.astype(float)
        lp = proj1.astype(float) - pers_ok.astype(float)
        lm_lo, lm_hi = bootci(lm); lp_lo, lp_hi = bootci(lp)
        s2.append({"city": city, "n_pairs": n,
                   "proj_top1": round(float(proj1.mean()), 4), "proj_top2": round(float(proj2.mean()), 4),
                   "marginal_acc": round(float(marg_ok.mean()), 4), "persistence_acc": round(float(pers_ok.mean()), 4),
                   "lift_vs_marginal": round(float(lm.mean()), 4), "lm_CI": f"[{lm_lo:+.4f},{lm_hi:+.4f}]", "lm_sig": "SI" if lm_lo > 0 else "no",
                   "lift_vs_persist": round(float(lp.mean()), 4), "lp_CI": f"[{lp_lo:+.4f},{lp_hi:+.4f}]", "lp_sig": "SI" if (lp_lo > 0 or lp_hi < 0) else "no"})

        # ---- ASSE 3: interpretability ----
        prof = {}
        for s in range(K):
            m = z_tr == s
            if not m.any(): continue
            dfm = ds["df_train"][m]
            ek = e_tr[m].mean(0); ek = ek * attr
            prof[s] = {"hour": int(round(float(dfm["c_hour"].mean()))),
                       "we": round(float(dfm["c_isweekend"].mean()), 2),
                       "intent": i2m[int(ek.argmax())] if ek.max() > 0 else "—",
                       "next": dfm["cat_macro"].mode().iloc[0] if len(dfm) else "—"}
        # strongest off-diagonal transitions by lift T_ij/marg_j
        cand = []
        for i in range(K):
            for j in range(K):
                if i != j and marg[j] > 0 and i in prof and j in prof:
                    cand.append((T[i, j] / marg[j], T[i, j], i, j))
        cand.sort(reverse=True)
        for lift, w, i, j in cand[:4]:
            o, d = prof[i], prof[j]
            lab = (f"[h~{o['hour']},we{o['we']},{o['intent']}] → [h~{d['hour']},we{d['we']},{d['intent']}]")
            s3.append({"city": city, "from": i, "to": j, "T_ij": round(float(w), 3),
                       "lift_vs_marg": round(float(lift), 2),
                       "from_traits": f"h{o['hour']}/we{o['we']}/int:{o['intent']}/next:{o['next']}",
                       "to_traits": f"h{d['hour']}/we{d['we']}/int:{d['intent']}/next:{d['next']}",
                       "label": lab})
        print(f"[{city}] done", flush=True)

    OUT = CLEAN / "outputs_results" / "validation"
    pd.DataFrame(s1).to_csv(OUT / "situation_transition_structure.csv", index=False)
    pd.DataFrame(s2).to_csv(OUT / "situation_prediction.csv", index=False)
    pd.DataFrame(s3).to_csv(OUT / "transition_interpretability.csv", index=False)

    print("\n=== ASSE 1 — STRUTTURA di T ===")
    print(pd.DataFrame(s1).to_string(index=False))
    print("  (rowH_norm: 0=deterministico,1=uniforme/rumore; KL_vs_marginal alto=struttura; diag=persistenza)")
    print("\n=== ASSE 2 — PREDITTIVITÀ (causale) ===")
    for r in s2:
        print(f"  {r['city']:<11} proj@1={r['proj_top1']:.3f} @2={r['proj_top2']:.3f}  "
              f"marg={r['marginal_acc']:.3f} pers={r['persistence_acc']:.3f}  "
              f"lift_vs_marg={r['lift_vs_marginal']:+.3f}{r['lm_CI']}{r['lm_sig']}  "
              f"lift_vs_pers={r['lift_vs_persist']:+.3f}{r['lp_CI']}{r['lp_sig']}")
    print("\n=== ASSE 3 — TRANSIZIONI FORTI (etichetta derivata) ===")
    for city in CITIES:
        print(f"  {city}:")
        for r in [x for x in s3 if x["city"] == city]:
            print(f"    T={r['T_ij']:.2f} (×{r['lift_vs_marg']:.1f} marg)  {r['label']}")
    print("\n→ situation_transition_structure.csv, situation_prediction.csv, transition_interpretability.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
