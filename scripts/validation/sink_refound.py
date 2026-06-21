"""Re-found the sink rule on an ABSOLUTE exposure-deficit measure (vs the old
relative-intra-city KL that was non-defensible). Situation params FIXED
(γ=0.4, depth=3, n=3, final per-city K/ε). Reuses the validated overnight code.

Per situation k we compute candidate deficits:
  D1 = available_LT_k − LT_k            (deficit vs availability; avail ≈ const)
  D2 = global_LT − LT_k                 (deficit vs corpus-avg exposure; SIGNED,
                                         fixed within-city absolute reference)
  D3 = KL(sit item dist ‖ global item dist)   (absolute KL — old KL, abs-thresholded)
  D4 = KL(sit MACRO dist ‖ UNIFORM macro)     (fair-reference concentration —
                                         the Istanbul test: captures uniform skew?)
LT_k = recommended long-tail rate; available_LT and KL come from the per-request
Stage-B stats. Activation criteria compared: outlier (mean+c·sd, pooled) and
absolute threshold (pooled percentiles).
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
from xsage.metrics import kl_divergence

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
PERC = {"gamma": 0.4, "depth": 3, "n": 3}
DEFS = ["D1", "D2", "D3", "D4"]
C_GRID = [1.0, 1.5, 2.0]
PCTL = [70, 80, 90]


def touch_of(prep, sinks):
    z = prep["z_test"]; isb = prep["isb_test"]
    m = np.isin(z, sinks) if sinks else np.zeros(len(z), bool)
    return float((m & ~isb).mean())


def main():
    land, preps = [], {}
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, PERC, K_FINAL[city], EPS_FINAL[city])
        blind_top, u = ov.blind_topk_test(prep)
        srows = ov.stage_b_stats_per_request(prep, blind_top, u)
        z = prep["z_test"]; icm = prep["item_cat_macro"]; G1 = prep["G1_mask"]
        i2m = prep["ds"]["idx_to_macro"]; n_items = prep["ds"]["n_items"]
        nmac = prep["ds"]["n_macros"]
        global_LT = float(G1[blind_top.flatten()].mean())
        preps[city] = prep
        for r in srows:
            k = r["situation"]
            if r["LT"] is None:
                continue
            mask = z == k
            items = blind_top[mask].flatten()
            macro_d = np.bincount(icm[items], minlength=nmac).astype(float)
            macro_d /= max(macro_d.sum(), 1.0)
            top1 = int(np.argmax(macro_d))
            D1 = r["available_LT"] - r["LT"]
            D2 = global_LT - r["LT"]
            D3 = r["KL"]
            D4 = kl_divergence(macro_d, np.ones(nmac) / nmac)
            land.append({"city": city, "situation": k, "n_req": r["n_requests"],
                         "LT": round(r["LT"], 4), "available_LT": round(r["available_LT"], 4),
                         "global_LT": round(global_LT, 4),
                         "D1": round(D1, 4), "D2": round(D2, 4), "D3": round(D3, 4),
                         "D4": round(D4, 4), "top_macro": i2m[top1],
                         "top_macro_share": round(float(macro_d[top1]), 3)})
        print(f"[{city}] global_LT={global_LT:.3f}  K={K_FINAL[city]}", flush=True)
    df = pd.DataFrame(land)
    OUT = CLEAN / "outputs_results" / "validation"
    df.to_csv(OUT / "sink_refound_deficits.csv", index=False)

    # pooled stats per deficit (ABSOLUTE: pooled over all 24 situations)
    pooled = {d: (float(df[d].mean()), float(df[d].std())) for d in DEFS}

    # --- distributions: sorted + Istanbul rank per deficit ---
    print("\n=== DISTRIBUZIONE DEI DEFICIT (ordinata) + dove cade ISTANBUL ===")
    for d in DEFS:
        s = df.sort_values(d, ascending=False)[["city", "situation", d]]
        ist = [f"sit{int(r.situation)}={getattr(r,d):.3f}" for r in
               df[df.city == "istanbul"].itertuples()]
        vals = s[d].tolist()
        gaps = [(round(vals[i] - vals[i+1], 3), i) for i in range(len(vals)-1)]
        gmax, gi = max(gaps)
        print(f"\n  {d}: pooled mean={pooled[d][0]:.3f} sd={pooled[d][1]:.3f}; "
              f"max gap={gmax:.3f} dopo rank {gi+1}/{len(vals)}")
        print("    top-6:", [f"{r.city[:3]}s{int(r.situation)}:{getattr(r,d):.3f}"
                              for r in s.head(6).itertuples()])
        print("    Istanbul:", ist)

    # --- activation: outlier (pooled mean+c·sd) and threshold (pooled pctl) ---
    act, rob = [], []
    for d in DEFS:
        mu, sd = pooled[d]
        for c in C_GRID:
            thr = mu + c * sd
            for city in CITIES:
                sub = df[df.city == city]
                s = sorted([int(r.situation) for r in sub.itertuples() if getattr(r, d) > thr])
                act.append({"deficit": d, "criterion": f"outlier_c={c}", "param": c,
                            "thr": round(thr, 4), "city": city, "n_sink": len(s),
                            "sink_ids": str(s), "touch_share": round(touch_of(preps[city], s), 4)})
        for p in PCTL:
            thr = float(np.percentile(df[d], p))
            for city in CITIES:
                sub = df[df.city == city]
                s = sorted([int(r.situation) for r in sub.itertuples() if getattr(r, d) > thr])
                act.append({"deficit": d, "criterion": f"pctl_{p}", "param": p,
                            "thr": round(thr, 4), "city": city, "n_sink": len(s),
                            "sink_ids": str(s), "touch_share": round(touch_of(preps[city], s), 4)})
    pd.DataFrame(act).to_csv(OUT / "sink_refound_activation.csv", index=False)

    # robustness of the OUTLIER criterion on D2 and D4 (the two candidates)
    for d in ["D2", "D4"]:
        mu, sd = pooled[d]
        for city in CITIES:
            sets = []
            for c in [1.0, 1.25, 1.5, 1.75, 2.0]:
                thr = mu + c * sd
                s = sorted([int(r.situation) for r in df[df.city == city].itertuples()
                            if getattr(r, d) > thr])
                sets.append(str(s))
            rob.append({"deficit": d, "city": city,
                        "sets_over_c": str(sets), "stable": len(set(sets)) == 1})
    pd.DataFrame(rob).to_csv(OUT / "sink_refound_robustness.csv", index=False)

    print("\n=== ATTIVAZIONE — outlier c=1.5 (n_sink per città) ===")
    for d in DEFS:
        row = {r["city"]: r["sink_ids"] for r in act
               if r["deficit"] == d and r["criterion"] == "outlier_c=1.5"}
        print(f"  {d}: {row}")

    print("\n=== ROBUSTEZZA outlier (D2, D4) su c∈{1.0..2.0} ===")
    for r in rob:
        print(f"  {r['deficit']} {r['city']:<11} {'STABILE' if r['stable'] else 'variabile'}  {r['sets_over_c']}")

    print("\n=== ISPEZIONE: D2 e D4 per situazione (sink? con outlier c=1.5) ===")
    for d in ["D2", "D4"]:
        mu, sd = pooled[d]; thr = mu + 1.5 * sd
        print(f"\n  --- {d} (soglia outlier c=1.5 = {thr:.3f}) ---")
        for city in CITIES:
            for r in df[df.city == city].itertuples():
                val = getattr(r, d)
                tag = "SINK" if val > thr else "    "
                print(f"    {tag} {city[:3]} sit{int(r.situation)} {d}={val:.3f}  "
                      f"LT={r.LT:.3f} topmacro={r.top_macro}({r.top_macro_share})")
    print(f"\n→ {OUT/'sink_refound_deficits.csv'}\n→ {OUT/'sink_refound_activation.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
