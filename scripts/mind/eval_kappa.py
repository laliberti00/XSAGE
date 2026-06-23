"""[venv-xsage] Selezione κ ANTI-CIRCOLARE (su VALIDATION) per SIT e Steck-b separatamente,
poi confronto su TEST al κ scelto da ciascuno. + curva di robustezza completa su test.
Risponde: SIT batte Steck-b su un RANGE di κ, o solo al κ=0.25 ereditato da Foursquare?

K/ε selezionati col processo Foursquare (silhouette + banda). κ selezionato su val (Cat-MRR).
Uso:  python scripts/mind/eval_kappa.py <city>     Output: outputs_results/baselines/<city>_kappa.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

K_TOP, BATCH, ALPHA, BOOT, SEED, SIL_N = 20, 1024, 50.0, 1500, 42, 20000
K_RANGE = [3, 4, 5, 6, 7, 8, 9]; EPS_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]; BAND = (0.10, 0.30)
KAPPA_GRID = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]


def select_K(vtr, ceil, rng):
    n = len(vtr); sidx = rng.choice(n, SIL_N, replace=False) if n > SIL_N else np.arange(n)
    best_k, best_s = K_RANGE[0], -1
    for K in K_RANGE:
        r = fit_rough_kmeans(vtr, K=K, eps=0.0, seed=SEED, max_iter=60); lab = r.core_label
        if len(np.unique(lab)) < 2: continue
        s = silhouette_score(vtr[sidx], lab[sidx])
        if s > best_s: best_s, best_k = s, K
    return min(best_k, ceil)


def select_eps(vtr, vva, K):
    cells = []
    for eps in EPS_GRID:
        r = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
        _, _, _, isb = _assign(vva, r.prototypes, eps); cells.append((eps, float(isb.mean())))
    inb = [(e, b) for e, b in cells if BAND[0] <= b <= BAND[1]]
    return (min(inb, key=lambda x: abs(x[1] - 0.20)) if inb else min(cells, key=lambda x: abs(x[1] - 0.20)))[0]


def cat_mrr(sb_full, df, dmac, gamma, icm, excl, kappa):
    """Cat-MRR per richiesta. dmac: [n_rows x n_macros] o None (BASE)."""
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        S = sb_full[u_b].astype(np.float32, copy=True)
        if dmac is not None:
            S = S + kappa * gamma[bs:be][:, None].astype(np.float32) * dmac[bs:be][:, icm]
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        macros = icm[np.take_along_axis(part, order, 1)]
        match = macros == tm[bs:be, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    return cm


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    rng = np.random.default_rng(SEED)
    print(f"[{city}] descrittore + selezione K/ε...", flush=True)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    ceil = int(D0["attractors"].sum()) + 2
    K = select_K(vtr, ceil, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)

    def assign(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps)
        return membership_from_assign(k, comp, isb, K), (1.0 / np.maximum(comp.sum(1), 1).astype(np.float32))
    mem_va, gam_va = assign(vva); mem_te, gam_te = assign(vte)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); b_bar = b_z.mean(0)
    n_users = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); ute = ds["df_test"]["u_idx"].values.astype(np.int64)
    print(f"[{city}] K={K} ε={eps}; selezione κ su VALIDATION...", flush=True)

    # dmac per metodo, per split
    def dmac(method, split):
        mem = mem_va if split == "val" else mem_te; u = uv if split == "val" else ute
        if method == "SIT": return mem.astype(np.float32) @ b_z
        if method == "Steck-b": return b_z_user[u]
        if method == "UNI_mean": return np.broadcast_to(b_bar, (len(u), nmac))
        return None
    gam = {"val": gam_va, "test": gam_te}; dfv = ds["df_val"]; dft = ds["df_test"]

    # curva val+test + selezione κ* (val-argmax Cat-MRR) per SIT, Steck-b, UNI_mean
    rows = []; kstar = {}
    for method in ["SIT", "Steck-b", "UNI_mean"]:
        dv = dmac(method, "val"); dt = dmac(method, "test")
        best = (-1, None)
        for kap in KAPPA_GRID:
            vm = cat_mrr(sb, dfv, dv, gam["val"], icm, excl, kap).mean()
            tm = cat_mrr(sb, dft, dt, gam["test"], icm, excl, kap).mean()
            rows.append({"city": city, "method": method, "kappa": kap,
                         "val_CatMRR": round(float(vm), 5), "test_CatMRR": round(float(tm), 5)})
            if vm > best[0]: best = (vm, kap)
        kstar[method] = best[1]
    base_v = cat_mrr(sb, dfv, None, gam["val"], icm, excl, 0).mean()
    base_t = cat_mrr(sb, dft, None, gam["test"], icm, excl, 0).mean()
    rows.append({"city": city, "method": "BASE", "kappa": 0.0,
                 "val_CatMRR": round(float(base_v), 5), "test_CatMRR": round(float(base_t), 5)})

    # confronto TEST al κ* selezionato su val (per-metodo), bootstrap SIT vs Steck-b
    sit_cm = cat_mrr(sb, dft, dmac("SIT", "test"), gam["test"], icm, excl, kstar["SIT"])
    stb_cm = cat_mrr(sb, dft, dmac("Steck-b", "test"), gam["test"], icm, excl, kstar["Steck-b"])
    d = sit_cm - stb_cm; n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "SIT VINCE" if lo > 0 else "SIT perde" if hi < 0 else "pari"

    OUT = CLEAN / "outputs_results" / "baselines"; OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{city}_kappa.csv", index=False)
    print(f"\n===== {city.upper()} — κ selezionato su VAL (Cat-MRR), valutato su TEST =====")
    print(f"  K={K} ε={eps}")
    print(f"  κ*(val): SIT={kstar['SIT']}  Steck-b={kstar['Steck-b']}  UNI_mean={kstar['UNI_mean']}")
    print(f"  TEST Cat-MRR @κ*: SIT={sit_cm.mean():.5f}  Steck-b={stb_cm.mean():.5f}  BASE={base_t:.5f}")
    print(f"  SIT − Steck-b (κ* per ciascuno) = {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  → {verdict}")
    print("\n  Curva TEST Cat-MRR per κ (robustezza):")
    piv = pd.DataFrame(rows)
    for method in ["SIT", "Steck-b"]:
        sub = piv[piv.method == method].sort_values("kappa")
        s = "  ".join(f"{k:.2f}:{v:.4f}" for k, v in zip(sub.kappa, sub.test_CatMRR))
        print(f"    {method:<9} {s}")
    # in quanti κ SIT>Steck-b sul test?
    wins = sum(1 for kap in KAPPA_GRID
               if piv[(piv.method == 'SIT') & (piv.kappa == kap)].test_CatMRR.iloc[0]
               > piv[(piv.method == 'Steck-b') & (piv.kappa == kap)].test_CatMRR.iloc[0])
    print(f"  → SIT>Steck-b (a pari κ) su test in {wins}/{len(KAPPA_GRID)} κ")
    print(f"\n→ outputs_results/baselines/{city}_kappa.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
