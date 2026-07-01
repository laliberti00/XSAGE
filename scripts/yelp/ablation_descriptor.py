"""[venv-xsage] Ablation della rappresentazione: clusterizza le situazioni su
  joint = [c̃ ‖ e]   ·   ctx = c̃ (solo contesto-informativita')   ·   intent = e (solo intento)
e misura la faro macro-Cat-MRR di SIT vs BASE (backbone B_blind, veloce). Risponde: la situazione e'
guidata dal contesto, dall'intento, o serve il joint? — senza toccare il modello (stessa pipeline,
descrittore affettato). Uso: python scripts/yelp/ablation_descriptor.py [city ...]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
for p in [str(CLEAN / "scripts" / "mind"), str(CLEAN), "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis", str(CLEAN / "scripts" / "yelp")]:
    sys.path.insert(0, p)
import results_record as rr
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

VARIANTS = ["joint", "ctx", "intent"]


def run(city, bk="B_blind"):
    csvb = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")

    def kget(seed):
        r = csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        if not len(r): r = csvb[(csvb.seed == 42) & (csvb.backbone == bk) & (csvb.method == "SIT")]["kstar"]
        return float(r.iloc[0])

    per = {v: [] for v in VARIANTS}; base = []; Ks = {v: [] for v in VARIANTS}
    for seed in rr.SEEDS:
        rng = np.random.default_rng(seed)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
        G1 = D0["G1"].astype(np.float32); nI = int(ds["n_items"])
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        A = vtr.shape[1] - nmac   # dimensioni di contesto (il resto = intento, K categorie)
        dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
        sfn = (lambda idx, u=ute: sb[u[idx]]); kap = kget(seed)
        sl = {"joint": slice(None), "ctx": slice(0, A), "intent": slice(A, None)}
        # BASE (κ=0) una volta
        eB = rr.per_request_eval(sfn, None, 0., np.ones(len(ute), np.float32), ute, ite, icm, excl, G1, nmac)
        base.append(rr.macro_msupp(eB["cm"], eB["tm"], nmac, rr.MSUPP))
        for v in VARIANTS:
            vt, vv, ve = vtr[:, sl[v]], vva[:, sl[v]], vte[:, sl[v]]
            K = select_K(vt, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vt, vv, K)
            fit = fit_rough_kmeans(vt, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
            _, kte, compte, isbte = _assign(ve, fit.prototypes, eps)
            mem = membership_from_assign(kte, compte, isbte, K); gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
            b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); nudge = mem.astype(np.float32) @ b_z
            eS = rr.per_request_eval(sfn, nudge, kap, gam, ute, ite, icm, excl, G1, nmac)
            per[v].append(rr.macro_msupp(eS["cm"], eS["tm"], nmac, rr.MSUPP)); Ks[v].append(K)
        print(f"  [{city}] seed {seed} ok (K joint/ctx/intent = {Ks['joint'][-1]}/{Ks['ctx'][-1]}/{Ks['intent'][-1]})", flush=True)
    base = np.array(base)
    print(f"\n### {city} (B_blind) — macro-Cat-MRR, BASE={base.mean():.5f}")
    print(f"{'variante':8s} {'SIT (mean±sd)':18s} {'ΔL1=SIT−BASE':14s} {'seeds Δ>0':9s} {'K medio':7s}")
    for v in VARIANTS:
        a = np.array(per[v]); d = a - base
        print(f"{v:8s} {a.mean():.5f}±{a.std(ddof=1):.5f}   {d.mean():+.5f}       {int((d>0).sum())}/5      {np.mean(Ks[v]):.1f}")
    return {v: (np.array(per[v]) - base) for v in VARIANTS}


def main():
    cities = sys.argv[1:] or ["ml1m", "nyc_tist", "saopaulo"]
    for c in cities:
        run(c)
    print("\n(joint = [c̃‖e]; ctx = solo c̃; intent = solo e. Se ctx≈joint e intent crolla → situazione guidata dal contesto.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
