"""[venv-xsage] FASE A — chiusura parametri percezione su VALIDATION (Cat-MRR + plateau).
Criterio: val Cat-MRR di SIT (κ=0.5 fisso durante lo sweep; κ ri-selezionato in Fase B).
- γ, n, depth: coordinate-descent su val (ri-seleziona). K/ε ri-selezionati per OGNI config.
- β, H, α: sweep di sensibilità (mostra piattezza → fissi-motivati).
Scrive outputs_results/params/<city>.json (letto in automatico da build_descriptor in Fase B)
+ outputs_results/param_closure_<city>.csv (le curve). Anti-circolare: MAI sul test.

Uso:  python scripts/ml1m/close_params.py ml1m [smoke]
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
try: from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x
from mind_prep import build_descriptor, membership_from_assign, GAMMA, DEPTH, N, BETA, H as H0, ALPHA
from eval_kappa import select_K, select_eps, cat_mrr
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

KAPPA_FIX, SEED = 0.5, 42
GRID = {"depth": [2, 3, 4], "n": [2, 3, 5], "gamma": [0.3, 0.4, 0.5, 0.6], "beta": [0.5, 0.7, 0.9], "H": [1, 2, 3]}
ALPHA_GRID = [10.0, 50.0, 100.0, 200.0]
if len(sys.argv) > 2 and sys.argv[2] == "smoke":
    GRID = {"depth": [3], "n": [3], "gamma": [0.4, 0.5], "beta": [0.7], "H": [2]}; ALPHA_GRID = [50.0, 100.0]


def val_mrr(city, p, rng):
    """Costruisce descrittore con i parametri p, ri-seleziona K/ε, score SIT su VAL (κ fisso)."""
    D0 = build_descriptor(city, splits=("train", "val"), gamma=p["gamma"], depth=p["depth"],
                          n=p["n"], beta=p["beta"], h_hops=p["H"])
    vtr, vva = D0["vs"]["train"], D0["vs"]["val"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
    cmt = D0["cmt"]; nmac = D0["n_macros"]; ds = D0["ds"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    _, k_va, comp, isb = _assign(vva, fit.prototypes, eps)
    mem = membership_from_assign(k_va, comp, isb, K); gam = (1. / np.maximum(comp.sum(1), 1).astype(np.float32))
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=p["alpha"]); dmac = mem.astype(np.float32) @ b_z
    return float(cat_mrr(sb, ds["df_val"], dmac, gam, icm, excl, KAPPA_FIX).mean()), K, eps


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    cur = {"gamma": GAMMA, "depth": DEPTH, "n": N, "beta": BETA, "H": H0, "alpha": ALPHA}
    rows = []
    print(f"=== FASE A {city}: chiusura parametri su VAL (κ={KAPPA_FIX}) ===", flush=True)
    print(f"  default: {cur}", flush=True)
    # coordinate descent su descrittore: depth, n, gamma, beta, H
    for param in ["depth", "n", "gamma", "beta", "H"]:
        best = (-1, cur[param], None, None)
        for v in tqdm(GRID[param], desc=f"sweep {param}"):
            p = dict(cur); p[param] = v
            m, K, eps = val_mrr(city, p, rng)
            rows.append({"phase": "select" if param in ("depth", "n", "gamma") else "sensitivity",
                         "param": param, "value": v, "val_CatMRR": round(m, 5), "K": K, "eps": eps})
            print(f"    {param}={v}: val Cat-MRR={m:.5f} (K={K} ε={eps})", flush=True)
            if m > best[0]: best = (m, v, K, eps)
        cur[param] = best[1]
        print(f"  → {param}* = {best[1]} (val Cat-MRR={best[0]:.5f})", flush=True)
    # α sweep (al descrittore finale)
    print(f"  sweep α (descrittore finale {cur})...", flush=True)
    bestA = (-1, cur["alpha"])
    for a in tqdm(ALPHA_GRID, desc="sweep alpha"):
        p = dict(cur); p["alpha"] = a; m, K, eps = val_mrr(city, p, rng)
        rows.append({"phase": "sensitivity", "param": "alpha", "value": a, "val_CatMRR": round(m, 5), "K": K, "eps": eps})
        print(f"    α={a}: val Cat-MRR={m:.5f}", flush=True)
        if m > bestA[0]: bestA = (m, a)
    cur["alpha"] = bestA[1]

    OUT = CLEAN / "outputs_results"; (OUT / "params").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"param_closure_{city}.csv", index=False)
    json.dump(cur, open(OUT / "params" / f"{city}.json", "w"), indent=2)
    print(f"\n===== PARAMETRI CHIUSI {city} =====")
    print(f"  {cur}")
    print(f"  (γ/n/depth selezionati su val; β/H/α = curva di sensibilità → vedi param_closure_{city}.csv)")
    # piattezza: range val Cat-MRR per i sensitivity-param
    df = pd.DataFrame(rows)
    for pr in ["beta", "H", "alpha"]:
        s = df[df.param == pr]["val_CatMRR"]
        if len(s): print(f"  sensibilità {pr}: range val Cat-MRR = {s.max()-s.min():.5f} ({'PIATTO' if s.max()-s.min()<0.003 else 'NON piatto → finding'})")
    print(f"\n→ params/{city}.json (letto in automatico dalla Fase B) + param_closure_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
