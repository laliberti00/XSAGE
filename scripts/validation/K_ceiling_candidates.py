"""Derive a UNIFORM ceiling on K from an a-priori structural statistic, to
handle bangkok=8 (at the grid edge, geometry wants >9) WITHOUT an ad-hoc fix.

Per city: K* (from the two-branch internal rule on depth=3) vs candidate ceilings
  - |A| = #attractors (indeg ≥ mean on W) — PRIMARY, conceptual: the intent
    vector lives on the attractor simplex, so the number of distinguishable
    situations is bounded by the number of attractor directions.
  - K_mac = #active macro categories.
  - log10(n_users) — weaker statistical candidate, for contrast.
  - (5th) |A| is perception-invariant (depends only on W, not depth/γ/n).

A ceiling is justified only if it (i) holds for the 4 SOLID cities with a small
margin, (ii) is exceeded by bangkok alone, (iii) is consistent (not noisy).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, str(OLD))

from pipeline.step02_models.xsage.orchestrator import _load_city
from pipeline.step02_models.xsage.l1_perception import (
    estimate_macro_transition, find_attractors)

# K* from the established two-branch rule (depth=3). Solid = all but bangkok.
KSTAR = {"istanbul": 5, "bangkok": 8, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
SOLID = ["istanbul", "nyc_tist", "saopaulo", "tokyo_tist"]   # bangkok = the edge case
CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]


def main():
    rows = []
    for city in CITIES:
        ds = _load_city(city)
        m2i = ds["macro_to_idx"]
        W = estimate_macro_transition(ds["df_train"], m2i)
        A = find_attractors(W)
        nA = int(A.sum())
        K_mac = int(ds["n_macros"])
        n_users = int(ds["df_train"]["u_idx"].nunique())
        rows.append({"city": city, "K_star": KSTAR[city], "n_attractors": nA,
                     "K_mac": K_mac, "n_users": n_users,
                     "log10_users": round(float(np.log10(n_users)), 2),
                     "attractors": [ds["idx_to_macro"][i] for i in np.where(A)[0]]})
    df = pd.DataFrame(rows)

    # candidate ceilings and exceedance
    cand = {"|A|": "n_attractors", "K_mac": "K_mac", "log10_users": "log10_users"}
    for name, col in cand.items():
        df[f"exceed_{name}"] = df["K_star"] - df[col]      # >0 means K* exceeds S

    OUT = CLEAN / "outputs_results" / "validation"
    df.drop(columns=["attractors"]).to_csv(OUT / "K_ceiling_candidates.csv", index=False)

    print("=== Statistiche città + K* ===")
    print(df[["city", "K_star", "n_attractors", "K_mac", "n_users",
              "log10_users"]].to_string(index=False))
    print("\nattrattori per città:")
    for r in rows:
        print(f"  {r['city']:<12} |A|={r['n_attractors']}  {r['attractors']}")

    print("\n=== K* − candidato  (>0 = K* SFORA il candidato) ===")
    print(f"{'city':<12}{'K*':>3}  {'|A|':>4}{'K*-|A|':>7}  {'K_mac':>5}{'K*-Kmac':>8}"
          f"  {'logU':>5}{'K*-logU':>8}")
    for r in df.itertuples():
        print(f"{r.city:<12}{r.K_star:>3}  {r.n_attractors:>4}{r.K_star-r.n_attractors:>+7}"
              f"  {r.K_mac:>5}{r.K_star-r.K_mac:>+8}"
              f"  {r.log10_users:>5}{r.K_star-r.log10_users:>+8.2f}")

    # consistency per candidate: do the 4 SOLID satisfy K* <= S (+margin)?
    print("\n=== Coerenza per candidato (4 città solide) ===")
    for name, col in cand.items():
        solid_ok = {c: (df.loc[df.city == c, "K_star"].iat[0]
                        <= df.loc[df.city == c, col].iat[0]) for c in SOLID}
        margins = {c: float(df.loc[df.city == c, col].iat[0]
                            - df.loc[df.city == c, "K_star"].iat[0]) for c in SOLID}
        bkk = df[df.city == "bangkok"]
        bkk_exceed = float(bkk["K_star"].iat[0] - bkk[col].iat[0])
        print(f"\n  candidato {name}:")
        print(f"    solide K*≤S: {solid_ok}")
        print(f"    margini S−K* (solide): {margins}")
        print(f"    bangkok K*−S = {bkk_exceed:+.2f}  "
              f"({'SFORA → spiega bangkok' if bkk_exceed > 0 else 'NON sfora'})")

    # ceiling forms with the winning candidate (|A|), robustness of the margin
    print("\n=== Forma del tetto K ≤ |A| + m  → chi viene limitato ===")
    for m in [0, 1, 2]:
        limited = []
        for r in df.itertuples():
            ceil = r.n_attractors + m
            if r.K_star > ceil:
                limited.append(f"{r.city}({r.K_star}→{ceil})")
        print(f"  m={m}: tetto=|A|+{m} → limita: {limited if limited else 'nessuno'}")

    print(f"\n→ {OUT/'K_ceiling_candidates.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
