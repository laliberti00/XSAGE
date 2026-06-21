"""Validate the two sink thresholds (1.5×mean-KL, 0.05 LT-gap) STRUCTURALLY —
no fairness in the loop (anti-circularity). Situation params FIXED (γ=0.4,
depth=3, n=3, final per-city K/ε). Reuses the validated overnight machinery.

Rule (from sinks.py / identify_sinks_per_request):
  sink(k) ⇔ KL_k ≥ kl_mult · mean_j(KL_j)  AND  |LT_k − LT_available_k| ≥ lt_gap
  combined with AND; KL ref = MEAN of per-situation KLs.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))

spec = importlib.util.spec_from_file_location(
    "ov", str(CLEAN / "scripts" / "overnight_selection.py"))
ov = importlib.util.module_from_spec(spec); spec.loader.exec_module(ov)

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02,
             "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
KL_GRID = [1.2, 1.5, 2.0, 2.5]
LT_GRID = [0.02, 0.05, 0.10]
ROB_KL = [1.3, 1.4, 1.5, 1.6, 1.7]
ROB_LT = [0.04, 0.05, 0.06]


def sinks_at(srows, gmean, kl_mult, lt_gap):
    return sorted([int(r["situation"]) for r in srows
                   if r["KL"] is not None and gmean > 0
                   and r["KL"] >= kl_mult * gmean
                   and abs(r["LT"] - r["available_LT"]) >= lt_gap])


def touch_of(prep, sinks):
    z = prep["z_test"]; isb = prep["isb_test"]
    m = np.isin(z, sinks) if sinks else np.zeros(len(z), bool)
    return float((m & ~isb).mean())


def main():
    land, sweep, rob = [], [], []
    inspect = {}
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        blind_top, u = ov.blind_topk_test(prep)
        srows = ov.stage_b_stats_per_request(prep, blind_top, u)
        kls = [r["KL"] for r in srows if r["KL"] is not None]
        gmean = float(np.mean(kls)) if kls else 0.0
        base_sinks = sinks_at(srows, gmean, 1.5, 0.05)

        z = prep["z_test"]; icm = prep["item_cat_macro"]; G1 = prep["G1_mask"]
        i2m = prep["ds"]["idx_to_macro"]
        for r in srows:
            k = r["situation"]
            klr = (r["KL"] / gmean) if (r["KL"] is not None and gmean > 0) else float("nan")
            ltgap = (abs(r["LT"] - r["available_LT"])
                     if r["LT"] is not None else float("nan"))
            land.append({"city": city, "situation": k, "n_req": r["n_requests"],
                         "KL": round(r["KL"], 4) if r["KL"] is not None else None,
                         "KL_ratio": round(klr, 3),
                         "LT": round(r["LT"], 4) if r["LT"] is not None else None,
                         "available_LT": round(r["available_LT"], 4) if r["available_LT"] is not None else None,
                         "LT_gap": round(ltgap, 4),
                         "is_sink_1.5_0.05": int(k in base_sinks)})
            # inspection: top-2 recommended macros + share, for this situation
            mask = z == k
            if mask.any():
                items = blind_top[mask].flatten()
                md = np.bincount(icm[items], minlength=len(i2m)).astype(float)
                md /= md.sum()
                top = np.argsort(md)[::-1][:2]
                inspect.setdefault(city, []).append(
                    (k, k in base_sinks, round(klr, 2), round(ltgap, 3),
                     [(i2m[int(t)], round(float(md[t]), 2)) for t in top]))

        for km in KL_GRID:
            for lg in LT_GRID:
                s = sinks_at(srows, gmean, km, lg)
                sweep.append({"city": city, "kl_mult": km, "lt_gap": lg,
                              "n_sink": len(s), "sink_ids": str(s),
                              "touch_share": round(touch_of(prep, s), 4)})
        for km in ROB_KL:
            for lg in ROB_LT:
                s = sinks_at(srows, gmean, km, lg)
                rob.append({"city": city, "kl_mult": km, "lt_gap": lg,
                            "sink_ids": str(s), "touch_share": round(touch_of(prep, s), 4)})
        print(f"[{city}] K={K_FINAL[city]} sinks@(1.5,0.05)={base_sinks}  "
              f"gmean_KL={gmean:.4f}", flush=True)

    OUT = CLEAN / "outputs_results" / "validation"
    pd.DataFrame(land).to_csv(OUT / "sink_landscape.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT / "sink_threshold_sweep.csv", index=False)
    pd.DataFrame(rob).to_csv(OUT / "sink_threshold_robustness.csv", index=False)

    # --- landscape gap analysis on both axes (all situations pooled) ---
    dl = pd.DataFrame(land)
    print("\n=== PAESAGGIO — KL_ratio di tutte le situazioni (ordinato) ===")
    s = dl.dropna(subset=["KL_ratio"]).sort_values("KL_ratio")
    for _, r in s.iterrows():
        flag = "  <== SINK" if r["is_sink_1.5_0.05"] else ""
        print(f"  {r['city']:<11} sit{int(r['situation'])}  KL_ratio={r['KL_ratio']:.2f}"
              f"  LT_gap={r['LT_gap']:.3f}{flag}")
    print(f"\n  soglia KL_ratio = 1.5  → quante sopra: "
          f"{int((dl['KL_ratio']>=1.5).sum())}/{len(dl)}")
    print(f"=== LT_gap: min={dl['LT_gap'].min():.3f} max={dl['LT_gap'].max():.3f}"
          f"  (soglia 0.05) → sotto 0.05: {int((dl['LT_gap']<0.05).sum())}/{len(dl)} ===")

    print("\n=== SWEEP 2D (n_sink / touch per città) ===")
    sw = pd.DataFrame(sweep)
    for city in CITIES:
        print(f"\n  {city}:")
        sub = sw[sw.city == city]
        piv = sub.pivot(index="kl_mult", columns="lt_gap", values="n_sink")
        print("    n_sink per (kl_mult × lt_gap):")
        print(piv.to_string().replace("\n", "\n    "))

    print("\n=== ROBUSTEZZA attorno a (1.5, 0.05) — sink_ids ===")
    rb = pd.DataFrame(rob)
    for city in CITIES:
        sub = rb[rb.city == city]
        uniq = sub["sink_ids"].unique()
        print(f"  {city:<11} set di sink nell'intorno: {list(uniq)}  "
              f"({'STABILE' if len(uniq)==1 else 'variabile'})")

    print("\n=== ISPEZIONE (validità di costrutto) — top-2 macro raccomandate ===")
    for city in ["nyc_tist", "saopaulo", "istanbul"]:
        print(f"\n  {city}:")
        for (k, issink, klr, ltg, tops) in inspect.get(city, []):
            tag = "SINK    " if issink else "non-sink"
            print(f"    sit{k} [{tag}] KLr={klr} LTgap={ltg}  top-macro={tops}")
    print(f"\n→ {OUT/'sink_landscape.csv'}\n→ {OUT/'sink_threshold_sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
