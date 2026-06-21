"""Test the hypothesis "the backbone already follows the situational intent".
Full-distribution (not top-1), WITH a non-situational reference (D2), and mapped
to WHERE it fails (D3, linked to the rudder strength). Params FIXED.

D1: div(e_k, exposed_k) full (KL+L1) + mass-overlap + top-3 overlap.
D2: vs baseline that IGNORES the situation:
      - marginal exposed (backbone exposure pooled over ALL situations of the city)
      - global popularity (top-K most popular items, same for everyone)
    Active alignment ⇔ div(e_k, exposed_k) < div(e_k, baseline).
D3: rank by div_backbone; correlate city-mean div with the rudder coherence gap.
"""
import importlib.util
import sys
from itertools import combinations
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
GAMMA, DEPTH, N = 0.4, 3, 3
N_CTX = len(ov.DEFAULT_ATTRIBUTES)
K_TOP = 20


def norm(p):
    p = np.asarray(p, float); return p / max(p.sum(), 1e-12)


def L1(p, q):
    return float(np.abs(norm(p) - norm(q)).sum())


def overlap(p, q):
    return float(np.minimum(norm(p), norm(q)).sum())          # shared mass ∈ [0,1]


def top3(p):
    return set(np.argsort(p)[::-1][:3].tolist())


def cosv(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    rows = []
    rudder = {}
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, {"gamma": GAMMA, "depth": DEPTH, "n": N},
                                              K_FINAL[city], EPS_FINAL[city])
        blind_top, u = ov.blind_topk_test(prep)
        built = ov.build_v(city, GAMMA, DEPTH, N, splits=("test",))
        e_test = built["vs"]["test"][:, N_CTX:]
        z = prep["z_test"]; icm = prep["item_cat_macro"]; nmac = prep["ds"]["n_macros"]
        i2m = prep["ds"]["idx_to_macro"]; b_z = prep["b_z"]; K = K_FINAL[city]

        # non-situational baselines (same for all situations of the city)
        marginal = np.bincount(icm[blind_top.flatten()], minlength=nmac).astype(float)
        pop = np.asarray((prep["ds"]["urm_train"] + prep["ds"]["urm_val"]).sum(0)).ravel()
        pop_topk = np.argsort(pop)[::-1][:K_TOP]
        pop_macro = np.bincount(icm[pop_topk], minlength=nmac).astype(float)

        ek = {}
        for k in range(K):
            mask = z == k
            if not mask.any():
                continue
            e_k = e_test[mask].mean(0); ek[k] = e_k
            exposed = np.bincount(icm[blind_top[mask].flatten()], minlength=nmac).astype(float)
            rows.append({
                "city": city, "situation": k, "n_req": int(mask.sum()),
                "div_backbone_L1": round(L1(e_k, exposed), 4),
                "div_backbone_KL": round(kl_divergence(e_k, exposed), 4),
                "overlap_backbone": round(overlap(e_k, exposed), 4),
                "div_marginal_L1": round(L1(e_k, marginal), 4),
                "div_pop_L1": round(L1(e_k, pop_macro), 4),
                "top3_overlap": round(len(top3(e_k) & top3(exposed)) / 3, 3),
                "improve_vs_marginal": round(L1(e_k, marginal) - L1(e_k, exposed), 4),
                "improve_vs_pop": round(L1(e_k, pop_macro) - L1(e_k, exposed), 4),
            })
        # rudder coherence gap (own vs others intent) — same as signal_diagnosis
        own = [cosv(b_z[k], ek[k]) for k in ek]
        oth = [np.mean([cosv(b_z[k], ek[j]) for j in ek if j != k]) for k in ek]
        rudder[city] = round(float(np.mean(own) - np.mean(oth)), 3)
        print(f"[{city}] rudder_gap={rudder[city]}", flush=True)

    df = pd.DataFrame(rows)
    df["rudder_strength"] = df["city"].map(rudder)
    OUT = CLEAN / "outputs_results" / "validation"
    df.to_csv(OUT / "backbone_intent_alignment.csv", index=False)

    print("\n=== D1 — quanto il backbone segue l'intento (distribuzione intera) ===")
    print(f"  div_backbone_L1: mean={df.div_backbone_L1.mean():.3f} "
          f"sd={df.div_backbone_L1.std():.3f}  range=[{df.div_backbone_L1.min():.3f},"
          f"{df.div_backbone_L1.max():.3f}]")
    print(f"  overlap di massa (1−L1/2 ~): mean={df.overlap_backbone.mean():.3f}  "
          f"range=[{df.overlap_backbone.min():.3f},{df.overlap_backbone.max():.3f}]")
    print(f"  top-3 overlap: mean={df.top3_overlap.mean():.3f}  "
          f"(quante situazioni con top3>=2/3: {(df.top3_overlap>=0.66).sum()}/{len(df)})")

    print("\n=== D2 — backbone vs baseline che IGNORA la situazione ===")
    print(f"  div(e, exposed)  media = {df.div_backbone_L1.mean():.3f}")
    print(f"  div(e, marginal) media = {df.div_marginal_L1.mean():.3f}  "
          f"(esposizione mediata su tutte le situazioni)")
    print(f"  div(e, pop)      media = {df.div_pop_L1.mean():.3f}  (popolarità globale)")
    print(f"  → miglioramento vs marginal: mean={df.improve_vs_marginal.mean():+.4f}  "
          f"(>0 in {(df.improve_vs_marginal>0).sum()}/{len(df)} situazioni)")
    print(f"  → miglioramento vs pop:      mean={df.improve_vs_pop.mean():+.4f}  "
          f"(>0 in {(df.improve_vs_pop>0).sum()}/{len(df)} situazioni)")

    print("\n=== D3 — DOVE il backbone segue meno ↔ forza del timone ===")
    g = df.groupby("city").agg(div=("div_backbone_L1", "mean"),
                               rud=("rudder_strength", "first")).reset_index()
    g = g.sort_values("div", ascending=False)
    print(f"  {'città':<12}{'div_backbone (segue meno→alto)':>32}{'timone_gap':>12}")
    for r in g.itertuples():
        print(f"  {r.city:<12}{r.div:>32.3f}{r.rud:>12.3f}")
    corr = float(np.corrcoef(g["div"], g["rud"])[0, 1])
    print(f"\n  correlazione (div_backbone vs forza timone) = {corr:+.3f}")
    print("  attesa storia coerente: corr POSITIVA (segue meno dove il timone è forte)")
    print(f"\n→ {OUT/'backbone_intent_alignment.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
