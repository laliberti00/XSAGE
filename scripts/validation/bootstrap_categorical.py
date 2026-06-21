"""DECISIVE significance test for the ONE positive thread: at κ=0.25 SIT sharpens
category rank. Same rigour as the negative bootstraps (≥1000 resamples, 95% CI).
Two metrics: diagnostic Cat-MRR/NDCG AND Steck Calibration (RecSys'18) re-targeted
on the situation, measured with Jensen-Shannon.

Steck calibration, situational:
  p(g|s) = expected category mix in situation s = normalised macro distribution of
           the TRAIN next-items assigned to situation s (z_train from the fit). Anti-
           circular: estimated on TRAIN, never test.
  q(g|u,s) = macro distribution of the top-K recommended list (BASE vs SIT).
  Miscalibration = JS( p(g|s) ‖ q(g|u,s) ) per request. ΔMiscal=SIT−BASE (<0=better).

Fixed κ=0.25 (the enhancer point, chosen BEFORE). Configs: BASE(κ=0), SIT, UNI_mean.
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
K_TOP, BATCH, SEED, BOOT = 20, 1024, 42, 2000


def js(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def ci(boot, lo=2.5, hi=97.5):
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


def main():
    rng = np.random.default_rng(SEED)
    out_rows = []
    for city in CITIES:
        prep, fit, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        ds = prep["ds"]; df = ds["df_test"]; n_items = ds["n_items"]; nmac = ds["n_macros"]
        m2i = ds["macro_to_idx"]; icm = prep["item_cat_macro"]
        n = len(df); u = df["u_idx"].values.astype(np.int64); i_tgt = df["i_idx"].values.astype(np.int64)
        tm = icm[i_tgt]; z = prep["z_test"]
        sb_full = prep["scores_blind_full"]; excl = prep["excluded"]
        mem = prep["membership_test"]; b_z = prep["b_z"]; gamma = prep["gamma"]; b_bar = b_z.mean(0)
        # p(g|s) from TRAIN next-item macros per situation (anti-circular)
        z_train = fit.core_label.astype(np.int64)
        cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
        K = K_FINAL[city]; p_gs = np.zeros((K, nmac))
        for s in range(K):
            c = np.bincount(cmt[z_train == s], minlength=nmac).astype(float) + 1e-6
            p_gs[s] = c / c.sum()

        nudges = {"BASE": None,
                  "SIT": lambda bs, be: gamma[bs:be][:, None].astype(np.float32) *
                         (mem[bs:be].astype(np.float32) @ b_z)[:, icm],
                  "UNI_mean": lambda bs, be: gamma[bs:be][:, None].astype(np.float32) * b_bar[icm][None, :]}
        per = {cf: {"cmrr": np.zeros(n), "cndcg": np.zeros(n), "miscal": np.zeros(n)} for cf in nudges}
        for bs in range(0, n, BATCH):
            be = min(n, bs + BATCH); u_b = u[bs:be]
            sb = sb_full[u_b].astype(np.float32, copy=True)
            for cf, fn in nudges.items():
                S = sb if cf == "BASE" else sb + KAPPA * fn(bs, be)
                S = S.copy()
                for j in range(be - bs):
                    cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
                    if len(cc): S[j, cc] = -np.inf
                part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
                order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
                top = np.take_along_axis(part, order, 1); macros = icm[top]
                match = macros == tm[bs:be, None]
                first = np.where(match.any(1), match.argmax(1) + 1, 0)
                per[cf]["cmrr"][bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
                per[cf]["cndcg"][bs:be] = np.where(first > 0, 1.0 / np.log2(np.maximum(first, 1) + 1.0), 0.0)
                # q(g|u,s) = top-K macro hist; miscal = JS(p(g|s)||q)
                zb = z[bs:be]
                for r in range(be - bs):
                    q = np.bincount(macros[r], minlength=nmac).astype(float)
                    per[cf]["miscal"][bs + r] = js(p_gs[zb[r]], q)
            print(f"  [{city}] batch {bs//BATCH+1}", end="\r", flush=True)
        # bootstrap deltas
        def boot_delta(a, b):
            d = a - b
            bsd = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
            lo, hi = ci(bsd)
            return float(d.mean()), lo, hi
        defs = [("Cat-MRR_SIT-BASE", per["SIT"]["cmrr"], per["BASE"]["cmrr"], "pos"),
                ("Cat-NDCG_SIT-BASE", per["SIT"]["cndcg"], per["BASE"]["cndcg"], "pos"),
                ("Miscal-JS_SIT-BASE", per["SIT"]["miscal"], per["BASE"]["miscal"], "neg"),
                ("Cat-MRR_SIT-UNImean", per["SIT"]["cmrr"], per["UNI_mean"]["cmrr"], "pos")]
        for name, a, b, direction in defs:
            mean, lo, hi = boot_delta(a, b)
            sig = (lo > 0) if direction == "pos" else (hi < 0)
            out_rows.append({"city": city, "metric": name, "delta": round(mean, 5),
                             "CI_low": round(lo, 5), "CI_high": round(hi, 5),
                             "significativo": "SI" if sig else "no",
                             "rudder": RUDDER[city]})
        print(f"[{city}] done                       ", flush=True)

    dfo = pd.DataFrame(out_rows)
    dfo.to_csv(CLEAN / "outputs_results" / "validation" / "bootstrap_categorical.csv", index=False)

    print("\n=== BOOTSTRAP (2000 resample, CI 95%) — Δ(SIT−BASE) a κ=0.25 ===")
    for metric in ["Cat-MRR_SIT-BASE", "Cat-NDCG_SIT-BASE", "Miscal-JS_SIT-BASE"]:
        print(f"\n  {metric}  ({'positivo=meglio' if 'Miscal' not in metric else 'NEGATIVO=meglio'})")
        for city in CITIES:
            r = dfo[(dfo.city == city) & (dfo.metric == metric)].iloc[0]
            flag = "✅ SIGNIF" if r.significativo == "SI" else "  n.s."
            print(f"    {city:<11}(tim={r.rudder:.2f}) Δ={r.delta:+.5f} "
                  f"[{r.CI_low:+.5f},{r.CI_high:+.5f}]  {flag}")

    print("\n=== Specificità: Δ(SIT−UNImean) su Cat-MRR (la situazione-specificità aiuta il rango?) ===")
    for city in CITIES:
        r = dfo[(dfo.city == city) & (dfo.metric == "Cat-MRR_SIT-UNImean")].iloc[0]
        flag = "✅ SIT>UNImean" if r.significativo == "SI" else ("SIT<UNImean" if r.CI_high < 0 else "n.s.")
        print(f"  {city:<11} Δ={r.delta:+.5f} [{r.CI_low:+.5f},{r.CI_high:+.5f}]  {flag}")

    print("\n=== CONCORDANZA Cat-MRR vs Miscalibration (per città) ===")
    for city in CITIES:
        mrr = dfo[(dfo.city == city) & (dfo.metric == "Cat-MRR_SIT-BASE")].significativo.iloc[0]
        mis = dfo[(dfo.city == city) & (dfo.metric == "Miscal-JS_SIT-BASE")].significativo.iloc[0]
        verdict = ("ROBUSTO (entrambe)" if mrr == "SI" and mis == "SI"
                   else "solo MRR" if mrr == "SI" else "solo Miscal" if mis == "SI" else "nessuna")
        print(f"  {city:<11} MRR={mrr}  Miscal={mis}  → {verdict}")
    print("\n→ bootstrap_categorical.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
