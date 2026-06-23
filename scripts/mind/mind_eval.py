"""[venv-xsage] Valutazione MIND coi parametri ATTUALI (placeholder K=6,ε=0.02) — per CAPIRE
il comportamento, NON per fare claim. Accuracy (BASE vs SIT, bootstrap) + Fairness (globale +
LENTE per-situazione: LT/KL per situazione, sink). K/ε non selezionati → esplorativo.
"""
import sys
from pathlib import Path
import numpy as np

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
from mind_prep import build_mind_prep, K_TOP, BATCH

BOOT = 1500


def gini(x):
    x = np.sort(np.asarray(x, np.float64)); n = len(x); s = x.sum()
    return 0.0 if s <= 0 else float((2 * np.sum(np.arange(1, n + 1) * x) / (n * s)) - (n + 1) / n)


def score_full(prep, cfg, kappa=0.25):
    ds = prep["ds"]; df = ds["df_test"]; icm = prep["icm"]; excl = prep["excl"]; sb_full = prep["sb"]
    G1 = prep["G1"]; n_items = len(icm); n_macros = prep["n_macros"]
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]
    n = len(df); cmrr = np.zeros(n); hit = np.zeros(n); lt = np.zeros(n)
    expo = np.zeros(n_items)                       # esposizione item (per Gini)
    z = prep["mem"].argmax(1)                       # situazione per richiesta
    sit_mac = np.zeros((prep["K"], n_macros))       # macro servite per situazione (per la lente)
    mem = prep["mem"]; b_z = prep["b_z"]; gamma = prep["gamma"]
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        S = sb_full[u_b].astype(np.float32, copy=True)
        if cfg == "SIT":
            d = (mem[bs:be].astype(np.float32) @ b_z)[:, icm]
            S = S + kappa * gamma[bs:be][:, None].astype(np.float32) * d
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        topk = np.take_along_axis(part, order, 1)
        macros = icm[topk]
        match = macros == tm[bs:be, None]; has = match.any(1)
        first = np.where(has, match.argmax(1) + 1, 0)
        cmrr[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        s_tgt = S[np.arange(be - bs), i_t[bs:be]]
        hit[bs:be] = ((S > s_tgt[:, None]).sum(1) + 1 <= K_TOP)
        lt[bs:be] = G1[topk].mean(1)
        np.add.at(expo, topk.ravel(), 1.0)
        zb = z[bs:be]
        for j in range(be - bs):
            np.add.at(sit_mac[zb[j]], macros[j], 1.0)
    return dict(cmrr=cmrr, hit=hit, lt=lt, u=u, expo=expo, z=z, sit_mac=sit_mac)


def pu(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v); np.add.at(c, inv, 1); return float((s / c).mean())


def main():
    rng = np.random.default_rng(42)
    city = sys.argv[1] if len(sys.argv)>1 else "mind"
    K, eps = 6, 0.02
    print(f"[eval {city}] build prep K={K} eps={eps} (placeholder)...", flush=True)
    prep = build_mind_prep(K, eps, city=city)
    B = score_full(prep, "BASE"); Sx = score_full(prep, "SIT", 0.25)

    print("\n===== ACCURACY (placeholder, esplorativo) =====")
    print(f"  BASE Cat-MRR={B['cmrr'].mean():.5f}  R@20={pu(B['hit'],B['u']):.5f}")
    print(f"  SIT  Cat-MRR={Sx['cmrr'].mean():.5f}  R@20={pu(Sx['hit'],Sx['u']):.5f}")
    d = Sx["cmrr"] - B["cmrr"]; nL = len(d)
    bsd = np.array([d[rng.integers(0, nL, nL)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bsd, [2.5, 97.5])
    print(f"  ΔCat-MRR(SIT−BASE)={d.mean():+.5f} [{lo:+.5f},{hi:+.5f}] "
          f"{'SI+' if lo>0 else 'SI-' if hi<0 else 'no'}")

    print("\n===== FAIRNESS GLOBALE =====")
    print(f"  LT@20:  BASE={B['lt'].mean():.4f}  SIT={Sx['lt'].mean():.4f}  Δ={Sx['lt'].mean()-B['lt'].mean():+.4f}")
    print(f"  Gini esposizione item: BASE={gini(B['expo']):.4f}  SIT={gini(Sx['expo']):.4f}")

    print("\n===== LENTE per-situazione (sul backbone BASE) =====")
    sm = B["sit_mac"]; glob = sm.sum(0); glob = glob / max(glob.sum(), 1)
    nreq = np.bincount(B["z"], minlength=K)
    KLk = np.zeros(K); LTk = np.zeros(K)
    # LT per situazione
    for k in range(K):
        m = B["z"] == k
        LTk[k] = B["lt"][m].mean() if m.any() else 0.0
        p = sm[k] / max(sm[k].sum(), 1); mask = p > 0
        KLk[k] = float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(glob[mask], 1e-12))))
    klr = KLk / max(KLk.mean(), 1e-12)
    print(f"  {'sit':>4} {'n_req':>8} {'LT':>7} {'KL':>7} {'KL_ratio':>9} {'sink≥1.5?':>9}")
    for k in range(K):
        print(f"  {k:>4} {nreq[k]:>8} {LTk[k]:>7.4f} {KLk[k]:>7.4f} {klr[k]:>9.3f} {'SINK' if klr[k]>=1.5 else '':>9}")
    print(f"  → KL varia {klr.min():.2f}×–{klr.max():.2f}× tra situazioni (lente: rivela disparità?)")
    print("\n(K/ε placeholder, fuori-banda boundary → esplorativo, NESSUN claim)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
