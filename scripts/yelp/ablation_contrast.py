"""[venv-xsage] Ablation-CONTRASTO (B6): test statistico di 'joint > best-single-component'.
Requisiti (obbligatori per validita'):
 (1) salva gli array PER-RICHIESTA di macro-Cat-MRR delle 3 varianti (ctx/joint/intent);
 (2) bootstrap sulla DIFFERENZA per-richiesta di macro-Cat-MRR (ricampiona le RICHIESTE, paired),
     non sul confronto di due medie aggregate;
 (3) contrasto = joint - best-half, best = argmax(ctx,intent) per QUEL dataset.
Regola di decisione PRE-REGISTRATA (vedi EXPERIMENT_PLAN.md): joint > best  <=>  Delta=joint-best supera
la banda +/-0.005 (mean 5-seed > 0.005) E 5/5 seed concordi (Delta>0) E CI-bootstrap esclude 0. Altrimenti
equivalenza (joint ~ best) sotto la stessa banda.
Uso: python scripts/yelp/ablation_contrast.py [city ...]
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

VARIANTS = ["ctx", "joint", "intent"]
BAND, BOOT = 0.005, 1500


def macro_boot(cm_j, cm_b, tm, nmac, rng, B=BOOT):
    """CI bootstrap della DIFFERENZA per-richiesta macro(joint)-macro(best): ricampiona le RICHIESTE (paired)."""
    n = len(tm); d = np.empty(B)
    for b in range(B):
        ix = rng.integers(0, n, n)
        d[b] = rr.macro_msupp(cm_j[ix], tm[ix], nmac, rr.MSUPP) - rr.macro_msupp(cm_b[ix], tm[ix], nmac, rr.MSUPP)
    lo, hi = np.percentile(d, [2.5, 97.5]); return float(lo), float(hi)


def run(city, bk="B_blind"):
    csvb = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")

    def kget(seed):
        r = csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        if not len(r): r = csvb[(csvb.seed == 42) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        return float(r.iloc[0])

    macro = {v: [] for v in VARIANTS}; cm42 = {}; tm42 = None
    for seed in rr.SEEDS:
        rng = np.random.default_rng(seed)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
        G1 = D0["G1"].astype(np.float32)
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        A = vtr.shape[1] - nmac
        dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
        sfn = (lambda idx, u=ute: sb[u[idx]]); kap = kget(seed)
        sl = {"ctx": slice(0, A), "joint": slice(None), "intent": slice(A, None)}
        for v in VARIANTS:
            vt, vv, ve = vtr[:, sl[v]], vva[:, sl[v]], vte[:, sl[v]]
            K = select_K(vt, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vt, vv, K)
            fit = fit_rough_kmeans(vt, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
            _, kte, compte, isbte = _assign(ve, fit.prototypes, eps)
            mem = membership_from_assign(kte, compte, isbte, K); gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
            b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); nudge = mem.astype(np.float32) @ b_z
            eS = rr.per_request_eval(sfn, nudge, kap, gam, ute, ite, icm, excl, G1, nmac)
            macro[v].append(rr.macro_msupp(eS["cm"], eS["tm"], nmac, rr.MSUPP))
            if seed == 42: cm42[v] = eS["cm"].copy(); tm42 = eS["tm"].copy()   # (1) salva array per-richiesta
        print(f"  [{city}] seed {seed} ok", flush=True)
    macro = {v: np.array(macro[v]) for v in VARIANTS}
    # (3) best-half = argmax(ctx, intent) per QUESTO dataset (sui mean 5-seed)
    best = "ctx" if macro["ctx"].mean() >= macro["intent"].mean() else "intent"
    dseed = macro["joint"] - macro[best]                     # Delta per-seed
    rng = np.random.default_rng(2024)
    lo, hi = macro_boot(cm42["joint"], cm42[best], tm42, nmac, rng)   # (2) bootstrap sulla diff per-richiesta
    passband = dseed.mean() > BAND; conc = int((dseed > 0).sum()) == len(rr.SEEDS); ci0 = lo > 0
    verdict = "joint > best" if (passband and conc and ci0) else "EQUIVALENZA (joint ~ best)"
    print(f"\n### {city} — contrasto joint - best(={best})   [best-half = argmax(ctx,intent)]")
    print(f"  macro 5-seed: ctx={macro['ctx'].mean():.5f} joint={macro['joint'].mean():.5f} intent={macro['intent'].mean():.5f}")
    print(f"  Δ=joint−best = {dseed.mean():+.5f} ± {dseed.std(ddof=1):.5f}  | seeds Δ>0 = {int((dseed>0).sum())}/5")
    print(f"  CI bootstrap per-richiesta (seed42) = [{lo:+.5f}, {hi:+.5f}]  (esclude 0: {ci0})")
    print(f"  banda ±{BAND}: mean>banda={passband} · 5/5={conc} · CI≠0={ci0}  →  VERDETTO: {verdict}")
    return verdict


def main():
    cities = sys.argv[1:] or ["ml1m", "nyc_tist", "saopaulo"]
    print("=== ABLATION-CONTRASTO B6 (regola pre-registrata: EXPERIMENT_PLAN.md) ===")
    for c in cities:
        run(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
