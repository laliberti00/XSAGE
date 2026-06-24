"""[venv-xsage] Costo computazionale del MODULO PLUGGABLE X-SAGE (per la sezione efficienza).
Cronometra ogni stadio del FIT (descrittore L0+L1, selezione K/ε, rough k-means L2, bias) e
l'INFERENZA (scoring/re-ranking del test col combiner), + conta i "parametri" del modulo.
Stampa una tabella. Uso:  python scripts/yelp/profile_cost.py ml1m
"""
import sys, time
from pathlib import Path
import numpy as np
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, cat_mrr, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    t = {}
    t0 = time.perf_counter()
    D0 = build_descriptor(city, splits=("train", "val", "test")); t["descrittore (L0+L1)"] = time.perf_counter() - t0
    nmac = D0["n_macros"]; cmt = D0["cmt"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    n_test = len(D0["ds"]["df_test"])

    t0 = time.perf_counter(); K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); t["selezione K"] = time.perf_counter() - t0
    t0 = time.perf_counter(); eps = select_eps(vtr, vva, K); t["selezione ε"] = time.perf_counter() - t0
    t0 = time.perf_counter(); fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); t["rough k-means (L2)"] = time.perf_counter() - t0
    z = fit.core_label.astype(np.int64)
    t0 = time.perf_counter(); b_z = fit_situation_biases_z(z, cmt, K, nmac, alpha=ALPHA); t["bias situazioni b̃"] = time.perf_counter() - t0

    # inferenza: assegnazione test + scoring/re-ranking col combiner
    t0 = time.perf_counter()
    _, kte, comp, isb = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(kte, comp, isb, K); gam = 1.0 / np.maximum(comp.sum(1), 1)
    dmac = mem.astype(np.float32) @ b_z
    _ = cat_mrr(sb, D0["ds"]["df_test"], dmac, gam, icm, excl, 0.5)
    t["inferenza (assegna+re-rank test)"] = time.perf_counter() - t0

    fit_total = sum(v for k, v in t.items() if "inferenza" not in k)
    dim = vtr.shape[1]
    params = K * nmac + K * dim          # bias + prototipi (il "modello" X-SAGE)
    print(f"\n===== COSTO MODULO PLUGGABLE X-SAGE — {city.upper()} =====")
    print(f"  dati: train={len(vtr)} test={n_test} | K={K} ε={eps} macro={nmac} dim(v)={dim}")
    print(f"  {'stadio':<36} {'secondi':>9}")
    for k, v in t.items(): print(f"  {k:<36} {v:>9.2f}")
    print(f"  {'-'*46}")
    print(f"  {'FIT totale (pluggable)':<36} {fit_total:>9.2f}")
    print(f"  inferenza per richiesta: {1000*t['inferenza (assegna+re-rank test)']/max(n_test,1):.3f} ms")
    print(f"  parametri del modulo: {params:,}  (K·macro + K·dim) — vs B_full ≈ (n_users+n_items+ctx)·d")
    print(f"\n  Confronto (osservato, stessa macchina): BPR fit ~secondi; B_full (FM torch+MPS) ~1-2 min/training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
