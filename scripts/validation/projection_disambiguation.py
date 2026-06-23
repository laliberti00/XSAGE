"""Replica del B1 eq.18 (boundary disambiguation) nel SETTING ATTUALE: 5 città TIST2015,
K/ε finali. Domanda: la projection aiuta ancora a DISAMBIGUARE le situazioni boundary?

Metrica (come B1 OLD): per le richieste boundary con z_prev valido, predici la situazione
SUCCESSIVA dell'utente z_{t+1} via argmax della membership, SENZA feedback (argmax r =
prototipo più vicino) vs CON eq.18 (argmax r̃, r̃_k ∝ r_k·T_{z_prev,k}). Macro-F1 + ΔF1,
bootstrap CI. Anti-circolare: T da train+val, z_prev causale, target da sequenza reale.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
spec = importlib.util.spec_from_file_location("ov", str(CLEAN / "scripts" / "overnight_selection.py"))
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)
from xsage.l3_projection import boundary_disambiguate, estimate_transition, macro_f1
from pipeline.step02_models.xsage.l2_comprehension import _assign

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
N_CTX = len(ov.DEFAULT_ATTRIBUTES)
SEED, BOOT = 42, 1500


def frame(df, z):
    return pd.DataFrame({"u": df["u_idx"].values,
                         "t": pd.to_datetime(df["time_local"].values), "z": np.asarray(z)})


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for city in CITIES:
        K = K_FINAL[city]; eps = EPS_FINAL[city]
        prep, fit, asg = ov.build_prep_for_report(city, PERC, K, eps)
        ds = prep["ds"]
        z_te = np.asarray(asg[1]).astype(np.int64)
        z_tr = fit.core_label.astype(np.int64)
        isb = np.asarray(prep["isb_test"]).astype(bool)
        mem = np.asarray(prep["membership_test"]).astype(np.float64)
        built = ov.build_v(city, PERC["gamma"], PERC["depth"], PERC["n"], splits=("train", "val"))
        z_va = _assign(built["vs"]["val"], fit.prototypes, eps)[1].astype(np.int64)

        # T da train+val (sequenze per-utente, ordine temporale)
        allr = pd.concat([frame(ds["df_train"], z_tr), frame(ds["df_val"], z_va)],
                         ignore_index=True).sort_values(["u", "t"], kind="stable")
        seqs = [g["z"].values for _, g in allr.groupby("u", sort=False)]
        T, _ = estimate_transition(seqs, K)

        # z_prev causale (timeline train∪val∪test)
        nte = len(ds["df_test"])
        full = pd.concat([frame(ds["df_train"], z_tr).assign(tix=-1),
                          frame(ds["df_val"], z_va).assign(tix=-1),
                          frame(ds["df_test"], z_te).assign(tix=np.arange(nte))],
                         ignore_index=True).sort_values(["u", "t"], kind="stable")
        full["prev"] = full.groupby("u", sort=False)["z"].shift(1)
        z_prev = np.full(nte, -1, np.int64)
        te = full[full.tix >= 0]
        z_prev[te.tix.values] = te["prev"].fillna(-1).astype(np.int64).values

        # next-situation target z_{t+1} (stesso utente)
        tf = frame(ds["df_test"], z_te).assign(tix=np.arange(nte)).sort_values(["u", "t"], kind="stable")
        tf["nz"] = tf.groupby("u", sort=False)["z"].shift(-1)
        next_z = np.full(nte, -1, np.int64)
        next_z[tf.tix.values] = tf["nz"].fillna(-1).astype(np.int64).values

        # eq.18
        r_tilde = boundary_disambiguate(mem, T, z_prev, isb)
        pred_no = mem.argmax(1).astype(np.int64)
        pred_fb = np.asarray(r_tilde).argmax(1).astype(np.int64)

        valid = isb & (z_prev >= 0) & (next_z >= 0)
        nb = int(valid.sum())
        yt = next_z[valid]; pn = pred_no[valid]; pf = pred_fb[valid]
        pp = z_prev[valid].astype(np.int64)   # baseline persistenza: predici z_prev
        f1_no = macro_f1(yt, pn, K); f1_fb = macro_f1(yt, pf, K); f1_pp = macro_f1(yt, pp, K)

        def bootci(predA, predB):
            db = np.empty(BOOT)
            for b in range(BOOT):
                s = rng.integers(0, nb, nb)
                db[b] = macro_f1(yt[s], predA[s], K) - macro_f1(yt[s], predB[s], K)
            lo, hi = np.percentile(db, [2.5, 97.5])
            p = 2.0 * min((db <= 0).mean(), (db >= 0).mean())
            return lo, hi, float(min(p, 1.0)), ("SI+" if lo > 0 else "SI-" if hi < 0 else "no")
        # (1) eq.18 vs argmax geometrico grezzo  (2) eq.18 vs persistenza pura
        lo1, hi1, p1, s1 = bootci(pf, pn)
        lo2, hi2, p2, s2 = bootci(pf, pp)
        changed = float((pn != pf).mean())
        rows.append({"city": city, "K": K, "boundary_frac": round(float(isb.mean()), 4),
                     "n_boundary_valid": nb, "changed_by_eq18": round(changed, 4),
                     "F1_no_fb": round(f1_no, 5), "F1_persist": round(f1_pp, 5),
                     "F1_with_fb": round(f1_fb, 5),
                     "dF1_vs_nofb": round(f1_fb - f1_no, 5), "CI_vs_nofb": f"[{lo1:+.4f},{hi1:+.4f}]", "sig_vs_nofb": s1,
                     "dF1_vs_persist": round(f1_fb - f1_pp, 5), "CI_vs_persist": f"[{lo2:+.4f},{hi2:+.4f}]", "sig_vs_persist": s2})
        print(f"[{city}] nb={nb} | F1 geom={f1_no:.4f} persist={f1_pp:.4f} eq18={f1_fb:.4f} | "
              f"Δvs_geom={f1_fb-f1_no:+.4f}{s1}  Δvs_persist={f1_fb-f1_pp:+.4f}[{lo2:+.4f},{hi2:+.4f}]{s2}", flush=True)

    out = CLEAN / "outputs_results" / "validation" / "projection_disambiguation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\n=== eq.18 boundary disambiguation — 5 città TIST2015 ===")
    print(pd.DataFrame(rows)[["city", "n_boundary_valid", "F1_no_fb", "F1_persist", "F1_with_fb",
                              "dF1_vs_nofb", "sig_vs_nofb", "dF1_vs_persist", "CI_vs_persist", "sig_vs_persist"]].to_string(index=False))
    print(f"\n-> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
