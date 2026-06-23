"""[venv-xsage] Prep MIND + smoke run (BASE vs SIT). Replica build_prep_for_report MA con:
dati dal clean repo (D.load_city('mind', data_root='.')), attributi MIND SENZA geohash,
transit disattivato. Conferma che X-SAGE gira end-to-end su MIND.
K/ε qui sono PLACEHOLDER (selezione anti-circolare = passo successivo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from pipeline.step02_models.xsage.l0_sensing import build_recent_window
from pipeline.step02_models.xsage.l1_perception import (compute_intent, compute_profile,
                                                        estimate_macro_transition,
                                                        fit_contribution_functions, find_attractors)
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage import data as D
from xsage.metrics import long_tail_groups
from xsage.recommendation import fit_situation_biases_z

MIND_ATTRS = ("c_hour", "c_dow", "c_isweekend", "c_month", "intent_last_cat_idx")  # NO geohash
H, BETA, ALPHA, SHORT_HEAD, MAX_ITER = 2, 0.7, 50.0, 0.20, 80
GAMMA, DEPTH, N = 0.4, 3, 3
K_TOP, BATCH = 20, 1024


def membership_from_assign(k_star, comp, isb, K):
    B = len(k_star); mem = np.zeros((B, K), np.float32)
    core = ~isb; mem[core, k_star[core]] = 1.0
    bnd = np.where(isb)[0]
    if bnd.size:
        T = comp[bnd].sum(1).astype(np.float32); mem[bnd] = comp[bnd].astype(np.float32) / T[:, None]
    return mem


def build_descriptor(city="mind", splits=("train", "val", "test")):
    """Costruisce v=[c̃‖e] per gli split richiesti + oggetti condivisi. NIENTE K/ε fissato
    (così l'eval può selezionarli). Riusato sia dal prep sia dall'eval baseline."""
    ds = D.load_city(city, data_root=str(CLEAN))
    m2i = ds["macro_to_idx"]; n_macros = ds["n_macros"]; n_items = ds["n_items"]
    for k in ("df_train", "df_val", "df_test"):
        ds[k] = ds[k].copy()
        ds[k]["cat_target"] = ds[k]["cat_macro"].map(m2i).astype(np.int64)
        ds[k]["user_id"] = ds[k]["u_idx"]
    contrib = fit_contribution_functions(ds["df_train"], m2i, attributes=MIND_ATTRS,
                                         max_depth=DEPTH, min_leaf=200)
    W = estimate_macro_transition(ds["df_train"], m2i, transit_macros=[], transit_mode="keep")
    attractors = find_attractors(W, exclude_indices=None)
    hist = {"train": ds["df_train"], "val": ds["df_train"],
            "test": pd.concat([ds["df_train"], ds["df_val"]], ignore_index=True)}

    def build(split):
        tgt = ds[f"df_{split}"]
        l0 = build_recent_window(tgt, hist[split], m2i, n=N)
        c = contrib.transform(tgt)
        m = compute_profile(l0.recent_macro, l0.n_prior, n_macros, gamma=GAMMA)
        e = compute_intent(m, W, attractors, H=H, beta=BETA, mode="hard")
        return np.concatenate([c, e], axis=1).astype(np.float32)

    vs = {s: build(s) for s in splits}
    cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (df_all.groupby("i_idx")["cat_macro"].first().map(m2i)
           .reindex(np.arange(n_items), fill_value=0).values.astype(np.int64))
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(0)).ravel()
    _, G1 = long_tail_groups(pop, short_head_share=SHORT_HEAD)
    sb, _ = D.load_backbone_scores(city, data_root=str(CLEAN))
    excl = D.load_excluded_mask(city, n_items, data_root=str(CLEAN))
    return dict(ds=ds, vs=vs, m2i=m2i, n_macros=n_macros, n_items=n_items, cmt=cmt,
                icm=icm, G1=G1, sb=sb, excl=excl, attractors=attractors)


def build_mind_prep(K, eps, seed=42, city="mind"):
    D0 = build_descriptor(city, splits=("train", "test"))
    ds = D0["ds"]; m2i = D0["m2i"]; n_macros = D0["n_macros"]; n_items = D0["n_items"]
    vtr = D0["vs"]["train"]; vte = D0["vs"]["test"]
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=MAX_ITER)
    z_train = fit.core_label.astype(np.int64)
    _, k_te, comp_te, isb_te = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(k_te, comp_te, isb_te, K)
    b_z = fit_situation_biases_z(z_train, D0["cmt"], K, n_macros, alpha=ALPHA)
    gamma = (1.0 / np.maximum(comp_te.sum(1), 1).astype(np.float32))
    return dict(ds=ds, K=K, mem=mem, b_z=b_z, icm=D0["icm"], G1=D0["G1"], gamma=gamma,
                sb=D0["sb"], excl=D0["excl"], isb=isb_te, n_macros=n_macros,
                bfrac=float(isb_te.mean()))


def score(prep, cfg, kappa=0.25):
    ds = prep["ds"]; df = ds["df_test"]; icm = prep["icm"]; excl = prep["excl"]; sb_full = prep["sb"]
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]
    n = len(df); cmrr = np.zeros(n); hit = np.zeros(n); mem = prep["mem"]; b_z = prep["b_z"]; gamma = prep["gamma"]
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
        macros = icm[np.take_along_axis(part, order, 1)]
        match = macros == tm[bs:be, None]; has = match.any(1)
        first = np.where(has, match.argmax(1) + 1, 0)
        cmrr[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        s_tgt = S[np.arange(be - bs), i_t[bs:be]]
        hit[bs:be] = ((S > s_tgt[:, None]).sum(1) + 1 <= K_TOP)
    # R@20 per-utente
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, hit); np.add.at(c, inv, 1)
    return float(cmrr.mean()), float((s / c).mean())


def main():
    city = sys.argv[1] if len(sys.argv)>1 else "mind"
    K, eps = 6, 0.02  # PLACEHOLDER (selezione anti-circolare = passo successivo)
    print(f"[smoke] {city} build_mind_prep K={K} eps={eps} (placeholder)...", flush=True)
    prep = build_mind_prep(K, eps, city=city)
    print(f"  situazioni K={K}, boundary_frac={prep['bfrac']:.1%}, macro={prep['n_macros']}", flush=True)
    base_mrr, base_r = score(prep, "BASE")
    sit_mrr, sit_r = score(prep, "SIT", kappa=0.25)
    print("\n=== SMOKE MIND (esecuzione end-to-end) ===")
    print(f"  BASE : Cat-MRR={base_mrr:.5f}  R@20={base_r:.5f}")
    print(f"  SIT  : Cat-MRR={sit_mrr:.5f}  R@20={sit_r:.5f}")
    print(f"  ΔCat-MRR(SIT−BASE) = {sit_mrr-base_mrr:+.5f}   ΔR@20 = {sit_r-base_r:+.5f}")
    print("  (K/ε placeholder; niente claim — solo conferma che il pipeline GIRA su MIND)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
