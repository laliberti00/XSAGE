"""Decide WHICH situational signal the gate should rest on. The long-tail signal
was FLAT across situations (thin gate, ambiguous). Here we test an INTERNAL
signal: intent↔exposure mismatch — and whether the per-situation correction
DIRECTIONS diverge (rudder). Params FIXED (γ=0.4, depth=3, n=3, final K/ε).

PART 1 (Q1 — does mismatch discriminate? = save-the-gate):
  M_A = L1( e_k , exposed_macro_k )    intent vs backbone exposure
  M_B = L1( true_next_macro_k , exposed_macro_k )   real behaviour vs exposure (causal)
  M_C = L1( e_k , true_next_macro_k )  intent vs real behaviour (control: is intent predictive?)
PART 2 (Q2 — do directions b^(k) diverge & align with intent? = rudder):
  dispersion of b^(k) across situations; cos(b^(k), e_k) vs cos(b^(k), e_{j≠k}).
All distributions over the city's macro space; divergence = L1 (Σ|p−q|), declared.
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

CITIES = ["istanbul", "bangkok", "nyc_tist", "saopaulo", "tokyo_tist"]
K_FINAL = {"istanbul": 5, "bangkok": 6, "nyc_tist": 6, "saopaulo": 3, "tokyo_tist": 4}
EPS_FINAL = {"istanbul": 0.01, "bangkok": 0.01, "nyc_tist": 0.02, "saopaulo": 0.05, "tokyo_tist": 0.07}
GAMMA, DEPTH, N = 0.4, 3, 3
N_CTX = len(ov.DEFAULT_ATTRIBUTES)
C_GRID = [1.0, 1.5, 2.0]


def L1(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    return float(np.abs(p - q).sum())


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def touch_of(prep, sinks):
    z = prep["z_test"]; isb = prep["isb_test"]
    m = np.isin(z, sinks) if sinks else np.zeros(len(z), bool)
    return float((m & ~isb).mean())


def main():
    rows, preps, ek_all, bz_all = [], {}, {}, {}
    for city in CITIES:
        prep, _, _ = ov.build_prep_for_report(city, {"gamma": GAMMA, "depth": DEPTH, "n": N},
                                              K_FINAL[city], EPS_FINAL[city])
        blind_top, u = ov.blind_topk_test(prep)
        built = ov.build_v(city, GAMMA, DEPTH, N, splits=("test",))
        v_test = built["vs"]["test"]
        e_test = v_test[:, N_CTX:]                      # per-request intent over macros
        z = prep["z_test"]; icm = prep["item_cat_macro"]; nmac = prep["ds"]["n_macros"]
        i2m = prep["ds"]["idx_to_macro"]; m2i = prep["ds"]["macro_to_idx"]
        true_macro_req = (prep["ds"]["df_test"]["cat_macro"].map(m2i).values).astype(int)
        b_z = prep["b_z"]; bz_all[city] = b_z; preps[city] = prep
        K = K_FINAL[city]; ek_city = {}
        for k in range(K):
            mask = z == k
            if not mask.any():
                continue
            e_k = e_test[mask].mean(axis=0)
            ek_city[k] = e_k
            exposed = np.bincount(icm[blind_top[mask].flatten()], minlength=nmac).astype(float)
            truem = np.bincount(true_macro_req[mask], minlength=nmac).astype(float)
            M_A, M_B, M_C = L1(e_k, exposed), L1(truem, exposed), L1(e_k, truem)
            rows.append({"city": city, "situation": k, "n_req": int(mask.sum()),
                         "M_A": round(M_A, 4), "M_B": round(M_B, 4), "M_C": round(M_C, 4),
                         "intent_top": i2m[int(np.argmax(e_k))],
                         "exposed_top": i2m[int(np.argmax(exposed))],
                         "true_top": i2m[int(np.argmax(truem))]})
        ek_all[city] = ek_city
        print(f"[{city}] K={K} done", flush=True)
    df = pd.DataFrame(rows)
    OUT = CLEAN / "outputs_results" / "validation"

    # ---- PART 1 activation (outlier pooled mean+c·sd) ----
    DEFS = ["M_A", "M_B", "M_C"]
    pooled = {d: (float(df[d].mean()), float(df[d].std())) for d in DEFS}
    act = []
    for d in DEFS:
        mu, sd = pooled[d]
        for c in C_GRID:
            thr = mu + c * sd
            for city in CITIES:
                s = sorted([int(r.situation) for r in df[df.city == city].itertuples()
                            if getattr(r, d) > thr])
                act.append({"deficit": d, "c": c, "thr": round(thr, 4), "city": city,
                            "n_sink": len(s), "sink_ids": str(s),
                            "touch": round(touch_of(preps[city], s), 4)})
    actdf = pd.DataFrame(act)
    for r in rows:  # attach selected flags at c=1.5 for the main def later
        pass
    df.to_csv(OUT / "signal_mismatch.csv", index=False)
    actdf.to_csv(OUT / "signal_mismatch_activation.csv", index=False)

    print("\n=== PART 1 — distribuzione mismatch (ordinata) + spread + Istanbul ===")
    for d in DEFS:
        s = df.sort_values(d, ascending=False)
        vals = s[d].tolist()
        gaps = [(round(vals[i]-vals[i+1], 3), i) for i in range(len(vals)-1)]
        gmax, gi = max(gaps)
        ist = [f"s{int(r.situation)}={getattr(r,d):.3f}" for r in df[df.city=="istanbul"].itertuples()]
        print(f"\n  {d}: mean={pooled[d][0]:.3f} sd={pooled[d][1]:.3f}  max-gap={gmax:.3f} (rank {gi+1}/{len(vals)})")
        print("    top-6:", [f"{r.city[:3]}{int(r.situation)}:{getattr(r,d):.3f}" for r in s.head(6).itertuples()])
        print("    Istanbul:", ist, f"-> sd_intra={np.std([float(x.split('=')[1]) for x in ist]):.3f}")

    print("\n=== PART 1 — attivazione outlier c=1.5 (sink per città, touch) ===")
    for d in DEFS:
        line = {r["city"]: (r["sink_ids"], r["touch"]) for r in act if r["deficit"]==d and r["c"]==1.5}
        print(f"  {d}: " + "  ".join(f"{c}:{line[c][0]}(t={line[c][1]:.0%})" for c in CITIES))
    print("\n  robustezza (sink set su c∈{1,1.5,2}) per la def con più struttura:")
    for d in DEFS:
        for city in CITIES:
            sets = [next(r["sink_ids"] for r in act if r["deficit"]==d and r["c"]==c and r["city"]==city) for c in C_GRID]
            if any(s != "[]" for s in sets):
                print(f"    {d} {city:<11} {sets}  {'STABILE' if len(set(sets))==1 else 'variabile'}")

    # ---- PART 2: direction divergence + coherence with intent ----
    drows = []
    print("\n=== PART 2 — divergenza direzioni b^(k) + coerenza con intento ===")
    for city in CITIES:
        b_z = bz_all[city]; K = b_z.shape[0]; ek = ek_all[city]
        # (e) dispersion: mean pairwise L2 between situation bias vectors
        pdists = [float(np.linalg.norm(b_z[i]-b_z[j])) for i, j in combinations(range(K), 2)]
        disp = float(np.mean(pdists)) if pdists else 0.0
        # (f) coherence: cos(b_k, e_k) vs mean_{j!=k} cos(b_k, e_j)
        own, other = [], []
        for k in range(K):
            if k not in ek: continue
            own.append(cos(b_z[k], ek[k]))
            oj = [cos(b_z[k], ek[j]) for j in ek if j != k]
            other.append(float(np.mean(oj)) if oj else 0.0)
        own_m, other_m = float(np.mean(own)), float(np.mean(other))
        drows.append({"city": city, "K": K, "bias_dispersion_meanL2": round(disp, 3),
                      "cos_own_intent": round(own_m, 3), "cos_other_intent": round(other_m, 3),
                      "coherence_gap": round(own_m - other_m, 3)})
        print(f"  {city:<11} dispersione b^(k)={disp:.2f}  cos(b,own e)={own_m:+.3f}  "
              f"cos(b,other e)={other_m:+.3f}  gap={own_m-other_m:+.3f}")
    pd.DataFrame(drows).to_csv(OUT / "signal_direction_divergence.csv", index=False)

    print("\n=== ISPEZIONE costrutto — M_B (real vs exposure) alto vs basso ===")
    for city in ["saopaulo", "istanbul", "tokyo_tist"]:
        print(f"  {city}:")
        for r in df[df.city == city].sort_values("M_B", ascending=False).itertuples():
            print(f"    sit{int(r.situation)} M_A={r.M_A:.2f} M_B={r.M_B:.2f} M_C={r.M_C:.2f}  "
                  f"intent={r.intent_top} exposed={r.exposed_top} true={r.true_top}")
    print(f"\n→ {OUT/'signal_mismatch.csv'}\n→ {OUT/'signal_direction_divergence.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
