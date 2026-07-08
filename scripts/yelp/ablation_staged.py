"""[venv-xsage] Ablation FUSIONE A STADI del descrittore v=[c‖e].
Confronta la fusione ANTICIPATA (concatenazione, 'joint') con due fusioni A STADI:
  staged-A (ctx→intent): clusterizzo prima sul CONTESTO (livello 1, hard), poi raffino
                          per INTENTO dentro ogni regime (livello 2, soft/boundary);
  staged-B (intent→ctx): l'ordine opposto.
Tutto a valle (b̃ per situazione → nudge → macro-Cat-MRR@20) è IDENTICO al metodo attuale:
cambia solo il passo di clustering ⇒ confronto mele-con-mele. Anti-circolarità preservata:
i cluster si formano dalla storia causale (t'<t), b̃ è solo su train, il film scelto NON entra
nella formazione dei cluster. Selezione K/ε AUTO per livello (silhouette + banda-boundary).
Regola di decisione = quella pre-registrata (EXPERIMENT_PLAN.md): Δ=staged−joint superiore ⟺
Δ>0 ∧ CI-bootstrap per-richiesta esclude 0 ∧ 5/5 seed; equivalenza ⟺ TOST ±0.005.
Uso: python scripts/yelp/ablation_staged.py [city=ml1m ...]
"""
import sys
from pathlib import Path
import numpy as np
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
for p in [str(CLEAN / "scripts" / "mind"), str(CLEAN), "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis", str(CLEAN / "scripts" / "yelp")]:
    sys.path.insert(0, p)
import pandas as pd
import results_record as rr
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

VARIANTS = ["joint", "stagedA", "stagedB"]      # A: ctx→intent · B: intent→ctx
BAND, BOOT = 0.005, 1500
MIN_SPLIT = 200                                  # regimi di livello-1 più piccoli non si spezzano


def macro_boot(cm_s, cm_j, tm, nmac, rng, B=BOOT):
    """CI bootstrap della DIFFERENZA per-richiesta macro(staged)-macro(joint): ricampiona le RICHIESTE (paired)."""
    n = len(tm); d = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        d[b] = rr.macro_msupp(cm_s[ix], tm[ix], nmac, rr.MSUPP) - rr.macro_msupp(cm_j[ix], tm[ix], nmac, rr.MSUPP)
    lo, hi = np.percentile(d, [2.5, 97.5]); return float(lo), float(hi)


def cluster_joint(vtr, vva, vte, ceil, rng, seed):
    """Fusione anticipata: un solo rough-k-means su v=[c‖e]. Ritorna (z_tr, mem_te, gam_te, K)."""
    K = select_K(vtr, ceil, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(kte, compte, isbte, K)
    gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
    return fit.core_label.astype(np.int64), mem.astype(np.float32), gam, K


def cluster_staged(vtr, vva, vte, A, order, ceil, rng, seed):
    """Fusione a stadi. order='ctx_intent' → livello1=contesto, livello2=intento (e viceversa).
    Livello-1 HARD (regime), livello-2 SOFT/boundary dentro il regime. Ritorna (z_tr, mem_te, gam_te, L)."""
    b1, b2 = (slice(0, A), slice(A, None)) if order == "ctx_intent" else (slice(A, None), slice(0, A))
    vt1, vv1, ve1 = vtr[:, b1], vva[:, b1], vte[:, b1]
    vt2, vv2, ve2 = vtr[:, b2], vva[:, b2], vte[:, b2]
    # ---- livello 1: regime grossolano (hard) ----
    K1 = select_K(vt1, ceil, rng); eps1 = select_eps(vt1, vv1, K1)
    fit1 = fit_rough_kmeans(vt1, K=K1, eps=eps1, seed=seed, max_iter=80)
    z1_tr = fit1.core_label.astype(np.int64)
    _, k1_va, _, _ = _assign(vv1, fit1.prototypes, eps1)
    _, k1_te, _, _ = _assign(ve1, fit1.prototypes, eps1)
    # ---- livello 2: raffinamento per l'altro blocco, dentro ogni regime ----
    leaf_of_train = np.empty(len(vtr), dtype=np.int64)
    reg = {}; base = 0
    for g in range(K1):
        mtr = (z1_tr == g); n_g = int(mtr.sum())
        if n_g < MIN_SPLIT:                                  # regime piccolo → foglia unica (no split)
            leaf_of_train[mtr] = base; reg[g] = (base, 1, None, None); base += 1; continue
        mva = (k1_va == g)
        vt2g = vt2[mtr]; vv2g = vv2[mva] if int(mva.sum()) > 0 else vt2g
        K2 = select_K(vt2g, ceil, rng); eps2 = select_eps(vt2g, vv2g, K2)
        fit2 = fit_rough_kmeans(vt2g, K=K2, eps=eps2, seed=seed, max_iter=80)
        leaf_of_train[mtr] = base + fit2.core_label.astype(np.int64)
        reg[g] = (base, K2, fit2, eps2); base += K2
    L = base
    # ---- membership sulle foglie (test): livello-1 hard sceglie il regime, livello-2 soft dentro ----
    mem_te = np.zeros((len(vte), L), dtype=np.float32)
    comp_te = np.zeros((len(vte), L), dtype=np.float32)
    for g in range(K1):
        mte = (k1_te == g)
        if not mte.any(): continue
        b0, K2, fit2, eps2 = reg[g]
        if fit2 is None:                                     # foglia unica
            mem_te[mte, b0] = 1.0; comp_te[mte, b0] = 1.0; continue
        _, k2, comp2, isb2 = _assign(ve2[mte], fit2.prototypes, eps2)
        sub_mem = membership_from_assign(k2, comp2, isb2, K2).astype(np.float32)
        rows = np.where(mte)[0]; cols = np.arange(b0, b0 + K2)
        mem_te[np.ix_(rows, cols)] = sub_mem
        comp_te[np.ix_(rows, cols)] = comp2.astype(np.float32)
    gam_te = 1.0 / np.maximum(comp_te.sum(1), 1).astype(np.float32)
    return leaf_of_train, mem_te, gam_te, L


def run(city, bk="B_blind"):
    csvb = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")

    def kget(seed):
        r = csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        if not len(r): r = csvb[(csvb.seed == 42) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        return float(r.iloc[0])

    macro = {v: [] for v in VARIANTS}; Ls = {v: [] for v in VARIANTS}
    cm42 = {}; tm42 = None
    for seed in rr.SEEDS:
        rng = np.random.default_rng(seed)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
        G1 = D0["G1"].astype(np.float32); ceil = int(D0["attractors"].sum()) + 2
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        A = vtr.shape[1] - nmac
        dft = D0["ds"]["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
        sfn = (lambda idx, u=ute: sb[u[idx]]); kap = kget(seed)
        clust = {
            "joint":  lambda: cluster_joint(vtr, vva, vte, ceil, rng, seed),
            "stagedA": lambda: cluster_staged(vtr, vva, vte, A, "ctx_intent", ceil, rng, seed),
            "stagedB": lambda: cluster_staged(vtr, vva, vte, A, "intent_ctx", ceil, rng, seed),
        }
        for v in VARIANTS:
            z_tr, mem_te, gam, L = clust[v]()
            b_z = fit_situation_biases_z(z_tr, cmt, L, nmac, alpha=ALPHA); nudge = mem_te @ b_z
            eS = rr.per_request_eval(sfn, nudge, kap, gam, ute, ite, icm, excl, G1, nmac)
            macro[v].append(rr.macro_msupp(eS["cm"], eS["tm"], nmac, rr.MSUPP)); Ls[v].append(L)
            if seed == 42: cm42[v] = eS["cm"].copy(); tm42 = eS["tm"].copy()
        print(f"  [{city}] seed {seed} ok  (L: joint={Ls['joint'][-1]} A={Ls['stagedA'][-1]} B={Ls['stagedB'][-1]})", flush=True)

    macro = {v: np.array(macro[v]) for v in VARIANTS}
    print(f"\n### {city} — FUSIONE A STADI vs concatenazione (macro-Cat-MRR@20, backbone {bk}, 5 seed)")
    print(f"  joint(concat) = {macro['joint'].mean():.5f} ± {macro['joint'].std(ddof=1):.5f}   [foglie medie {np.mean(Ls['joint']):.1f}]")
    for v, name in [("stagedA", "staged-A  ctx→intent"), ("stagedB", "staged-B  intent→ctx")]:
        d = macro[v] - macro["joint"]
        rng = np.random.default_rng(2024)
        lo, hi = macro_boot(cm42[v], cm42["joint"], tm42, D0["n_macros"], rng)
        conc = int((d > 0).sum()) == len(rr.SEEDS); ci0 = lo > 0
        superior = (d.mean() > 0) and conc and ci0
        equiv = (lo > -BAND) and (hi < BAND)
        verdict = "staged > joint" if superior else ("staged ≈ joint (equivalenza, TOST)" if equiv else "inconcludente")
        print(f"  {name}: {macro[v].mean():.5f} ± {macro[v].std(ddof=1):.5f}   [foglie medie {np.mean(Ls[v]):.1f}]")
        print(f"      Δ=staged−joint = {d.mean():+.5f} ± {d.std(ddof=1):.5f}  | seeds Δ>0 = {int((d>0).sum())}/5"
              f"  | CI-boot(seed42) = [{lo:+.5f}, {hi:+.5f}]  →  VERDETTO: {verdict}")
    return 0


def main():
    cities = sys.argv[1:] or ["ml1m"]
    print("=== ABLATION FUSIONE A STADI (regola pre-registrata: EXPERIMENT_PLAN.md) ===")
    for c in cities:
        run(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
