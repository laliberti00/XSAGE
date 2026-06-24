"""[venv-xsage] ABLAZIONE DI NEUTRALITÀ (O9) — il valore della situazione è STRUTTURALE o cucito nelle feature?
4 celle su B_blind: full / raw_ctx / raw_int / raw_both (interruttori in build_descriptor).
Per ognuna seleziona K/ε, fitta, e misura:
  (a) LENTE  = dispersione KL fra situazioni (p(macro|z_train) vs globale); ratio max/media.
  (b) WIN SIT = Cat-MRR micro + MACRO-averaged + #categorie SIT>BASE (κ* su val).
Lettura: se sotto raw_both LENTE e MACRO-win SOPRAVVIVONO → valore STRUTTURALE (stato neutro).
         se COLLASSANO → era la category-informatività nelle feature.
Anti-circolare: K/ε/κ da train/val, mai dal test.

Uso:  python scripts/yelp/neutrality_ablation.py ml1m
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
try: from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, cat_mrr, KAPPA_GRID, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

MODES = [("full", False, False), ("raw_ctx", True, False),
         ("raw_int", False, True), ("raw_both", True, True)]


def macro_avg(cm, tm, nmac):
    return float(np.mean([cm[tm == c].mean() for c in range(nmac) if (tm == c).any()]))


def lens_ratio(z_tr, cmt, K, nmac):
    """Dispersione della lente: KL(p(macro|z) ‖ p(macro globale)) per situazione; ratio max/media."""
    cnt = np.zeros((K, nmac))
    for zi, ci in zip(z_tr, cmt): cnt[zi, ci] += 1
    glob = cnt.sum(0); glob = glob / max(glob.sum(), 1)
    KL = np.zeros(K)
    for k in range(K):
        p = cnt[k] / max(cnt[k].sum(), 1); mask = p > 0
        KL[k] = float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(glob[mask], 1e-12))))
    mean = max(KL.mean(), 1e-12)
    return float(KL.max()), float(KL.max() / mean)


def run_mode(city, name, rc, ri, rng):
    D0 = build_descriptor(city, splits=("train", "val", "test"), raw_context=rc, raw_intent=ri)
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)

    def assign(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps)
        return membership_from_assign(k, comp, isb, K), (1.0 / np.maximum(comp.sum(1), 1).astype(np.float32)), float(isb.mean())
    mem_va, gam_va, _ = assign(vva); mem_te, gam_te, bfrac = assign(vte)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    n_users = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); ute = ds["df_test"]["u_idx"].values.astype(np.int64)
    dfv, dft = ds["df_val"], ds["df_test"]
    dmac = {"SIT": (mem_va.astype(np.float32) @ b_z, mem_te.astype(np.float32) @ b_z),
            "Steck-b": (b_z_user[uv], b_z_user[ute])}

    kstar = {}
    for m, (dv, _) in dmac.items():
        best = (-1, None)
        for kap in tqdm(KAPPA_GRID, desc=f"{name}:κ {m}", leave=False):
            vm = cat_mrr(sb, dfv, dv, gam_va, icm, excl, kap).mean()
            if vm > best[0]: best = (vm, kap)
        kstar[m] = best[1]

    tm = icm[dft["i_idx"].values.astype(np.int64)]
    base = cat_mrr(sb, dft, None, gam_te, icm, excl, 0.0)
    sit = cat_mrr(sb, dft, dmac["SIT"][1], gam_te, icm, excl, kstar["SIT"])
    active = [c for c in range(nmac) if (tm == c).any()]
    wins = sum(1 for c in active if sit[tm == c].mean() > base[tm == c].mean())
    klmax, klratio = lens_ratio(z_tr, cmt, K, nmac)
    return {"mode": name, "K": K, "eps": eps, "bfrac": round(bfrac, 3),
            "lens_KLmax": round(klmax, 4), "lens_KLratio": round(klratio, 3),
            "kstar_SIT": kstar["SIT"], "dim_v": vtr.shape[1],
            "BASE_micro": round(base.mean(), 5), "SIT_micro": round(sit.mean(), 5),
            "BASE_macro": round(macro_avg(base, tm, nmac), 5), "SIT_macro": round(macro_avg(sit, tm, nmac), 5),
            "dMicro": round(sit.mean() - base.mean(), 5),
            "dMacro": round(macro_avg(sit, tm, nmac) - macro_avg(base, tm, nmac), 5),
            "cats_SIT_gt_BASE": f"{wins}/{len(active)}"}


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    print(f"=== ABLAZIONE DI NEUTRALITÀ — {city} (B_blind) ===", flush=True)
    rows = []
    for name, rc, ri in tqdm(MODES, desc="modi"):
        print(f"\n--- modo {name} (raw_context={rc}, raw_intent={ri}) ---", flush=True)
        r = run_mode(city, name, rc, ri, rng); rows.append(r)
        print(f"  K={r['K']} ε={r['eps']} dim_v={r['dim_v']} bfrac={r['bfrac']} | "
              f"lente KLratio={r['lens_KLratio']} | "
              f"SIT micro {r['SIT_micro']} (Δ{r['dMicro']:+}) MACRO {r['SIT_macro']} (Δ{r['dMacro']:+}) | "
              f"cat {r['cats_SIT_gt_BASE']}", flush=True)
    df = pd.DataFrame(rows)
    OUT = CLEAN / "outputs_results"; df.to_csv(OUT / f"neutrality_ablation_{city}.csv", index=False)
    print(f"\n===== {city.upper()} — ABLAZIONE DI NEUTRALITÀ =====")
    cols = ["mode", "dim_v", "K", "bfrac", "lens_KLratio", "dMicro", "dMacro", "cats_SIT_gt_BASE"]
    print(df[cols].to_string(index=False))
    full = df[df["mode"] == "full"].iloc[0]; rb = df[df["mode"] == "raw_both"].iloc[0]
    print("\n  LETTURA (full vs raw_both):")
    print(f"    lente KLratio: {full['lens_KLratio']} → {rb['lens_KLratio']}  ({'SOPRAVVIVE' if rb['lens_KLratio'] >= 0.6*full['lens_KLratio'] else 'COLLASSA'})")
    print(f"    macro-win Δ:   {full['dMacro']:+} → {rb['dMacro']:+}  ({'SOPRAVVIVE' if rb['dMacro'] > 0 else 'COLLASSA'})")
    verdict = ("STRUTTURALE (lo stato neutro funziona)" if (rb["dMacro"] > 0 and rb["lens_KLratio"] >= 0.6*full["lens_KLratio"])
               else "FEATURE-COUPLED (il valore era la category-informatività)")
    print(f"    → VALORE: {verdict}")
    print(f"\n→ outputs_results/neutrality_ablation_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
