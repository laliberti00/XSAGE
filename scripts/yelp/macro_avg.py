"""[venv-xsage] Cat-MRR MICRO vs MACRO-averaged (per-categoria) su B_blind, anti-saturazione.
Per BASE / SIT@κ* / Steck-b@κ* (κ* selezionato su VAL, come la batteria):
- micro = media sulle richieste; macro = media delle medie per-categoria (le minoritarie pesano uguale).
- quota macro dominante (saturazione) + su quante categorie SIT>BASE.
Uso:  python scripts/yelp/macro_avg.py <city>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, cat_mrr, KAPPA_GRID, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z


def macro_avg(cm, tm, nmac):
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).any()]
    return float(np.mean(per)), per


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)

    def assign(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps)
        return membership_from_assign(k, comp, isb, K), (1.0 / np.maximum(comp.sum(1), 1).astype(np.float32))
    mem_va, gam_va = assign(vva); mem_te, gam_te = assign(vte)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    n_users = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); ute = ds["df_test"]["u_idx"].values.astype(np.int64)
    dfv, dft = ds["df_val"], ds["df_test"]
    dmac = {"SIT": (mem_va.astype(np.float32) @ b_z, mem_te.astype(np.float32) @ b_z),
            "Steck-b": (b_z_user[uv], b_z_user[ute])}

    # κ* su VAL (micro Cat-MRR), per metodo
    kstar = {}
    for m, (dv, _) in dmac.items():
        best = (-1, None)
        for kap in KAPPA_GRID:
            vm = cat_mrr(sb, dfv, dv, gam_va, icm, excl, kap).mean()
            if vm > best[0]: best = (vm, kap)
        kstar[m] = best[1]

    tm = icm[dft["i_idx"].values.astype(np.int64)]                  # macro vera per richiesta (test)
    cm = {"BASE": cat_mrr(sb, dft, None, gam_te, icm, excl, 0.0),
          "SIT": cat_mrr(sb, dft, dmac["SIT"][1], gam_te, icm, excl, kstar["SIT"]),
          "Steck-b": cat_mrr(sb, dft, dmac["Steck-b"][1], gam_te, icm, excl, kstar["Steck-b"])}

    # saturazione
    cnt = np.bincount(tm, minlength=nmac); share = cnt / cnt.sum()
    dom = int(share.argmax())
    print(f"\n===== {city.upper()} — Cat-MRR MICRO vs MACRO (B_blind, test) =====")
    print(f"  K={K} ε={eps}  macro attive={int((cnt>0).sum())}/{nmac}")
    print(f"  saturazione: macro dominante #{dom} = {share.max():.1%} delle richieste; top-3 = {np.sort(share)[::-1][:3].sum():.1%}")
    print(f"  κ*(val): SIT={kstar['SIT']}  Steck-b={kstar['Steck-b']}")
    print(f"\n  {'metodo':<9} {'MICRO':>9} {'MACRO':>9}")
    base_mi = cm["BASE"].mean(); base_ma, base_per = macro_avg(cm["BASE"], tm, nmac)
    for m in ["BASE", "SIT", "Steck-b"]:
        mi = cm[m].mean(); ma, _ = macro_avg(cm[m], tm, nmac)
        print(f"  {m:<9} {mi:>9.5f} {ma:>9.5f}")
    sit_ma, sit_per = macro_avg(cm["SIT"], tm, nmac)
    print(f"\n  Δ(SIT−BASE):  micro={cm['SIT'].mean()-base_mi:+.5f}   MACRO={sit_ma-base_ma:+.5f}")
    # per-categoria: dove SIT aiuta/danneggia
    active = [c for c in range(nmac) if (tm == c).any()]
    deltas = [(c, cm["SIT"][tm == c].mean() - cm["BASE"][tm == c].mean(), int((tm == c).sum())) for c in active]
    wins = sum(1 for _, dd, _ in deltas if dd > 0)
    print(f"  SIT>BASE su {wins}/{len(active)} categorie")
    print(f"  per-categoria Δ(SIT−BASE) [cat: Δ (n_req, quota)]:")
    for c, dd, nrq in sorted(deltas, key=lambda x: -x[2]):
        print(f"    macro {c:>2}: {dd:+.4f}  (n={nrq}, {nrq/cnt.sum():.1%})")
    # CSV per la matrice allineata
    rows = [{"city": city, "method": m, "dominant_share": round(float(share.max()), 4),
             "macro_active": int((cnt > 0).sum()),
             "micro": round(cm[m].mean(), 5), "macro": round(macro_avg(cm[m], tm, nmac), 5)}
            for m in ["BASE", "SIT", "Steck-b"]]
    rows.append({"city": city, "method": "SIT_minus_BASE", "dominant_share": round(float(share.max()), 4),
                 "macro_active": int((cnt > 0).sum()),
                 "micro": round(cm["SIT"].mean() - cm["BASE"].mean(), 5),
                 "macro": round(macro_avg(cm["SIT"], tm, nmac) - macro_avg(cm["BASE"], tm, nmac), 5)})
    pd.DataFrame(rows).to_csv(CLEAN / "outputs_results" / f"macro_avg_{city}.csv", index=False)
    print(f"\n→ outputs_results/macro_avg_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
