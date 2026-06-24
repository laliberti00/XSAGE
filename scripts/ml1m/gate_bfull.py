"""[venv-xsage] GATE B_full — decide se SIT è competitivo vs un context-aware END-TO-END (B_full)
sui domini profondi. Disciplina: train impara, VAL sceglie iperparametri, TEST misura UNA volta.

SIT/Steck-b/UNI: κ selezionato su VAL (per-metodo). B_full: iperparametri (emb,lr) selezionati su VAL.
Confronto su TEST con bootstrap+Holm. Soglia PRE-REGISTRATA (margine fissato prima): vedi MARGIN.

PARITÀ FEATURE (dichiarata): SIT e B_full ricevono gli STESSI attributi di contesto
(c_hour,c_dow,c_isweekend,c_month,intent_last). Asimmetria onesta: SIT costruisce in più il vettore
intento propagato e (dagli stessi input grezzi); B_full è un FM end-to-end (user×item×contesto) → più
capacità ma stesso contesto. Backbone BASE/B_blind = BPR (coerente con gli altri esperimenti ml-1m).

Uso:  python scripts/ml1m/gate_bfull.py ml1m
Output: outputs_results/gate_bfull_ml1m.csv   —  NESSUN commit (l'utente vede prima i numeri).
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x
from mind_prep import build_descriptor, membership_from_assign
from eval_kappa import select_K, select_eps, cat_mrr, KAPPA_GRID
from train_bfull import feats_from_df, score_test
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from pipeline.step02_models.xsage.backbone_full import (ContextAwareFM, FeatureSpec, train_b_full)
from xsage.recommendation import fit_situation_biases_z

K_TOP, BATCH, ALPHA, BOOT, SEED = 20, 1024, 50.0, 1500, 42
MARGIN = 0.01                       # soglia PRE-REGISTRATA (PASS-debole)
BF_GRID = [(32, 5e-3), (32, 1e-2), (64, 5e-3), (64, 1e-2)]   # (emb_dim, lr)
BF_EPOCHS, BF_CKPT, VAL_SUB = 15, 3, 12000   # epoche, ogni quante valutare val, subsample val per tuning
N_CAND, ALPHA_STECK, LAM = 60, 0.01, 0.99


def metrics_rows(provider, df, icm, excl, n_macros):
    """provider(u_b, idx) -> [B, n_items] scores. Ritorna per-richiesta: cmrr, cndcg, hit(R@20), indcg(NDCG@20)."""
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); idc = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); idx = np.arange(bs, be); u_b = u[idx]
        S = provider(u_b, idx).astype(np.float32, copy=True)
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        macros = icm[np.take_along_axis(part, order, 1)]
        match = macros == tm[idx, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[idx] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        cn[idx] = np.where(first > 0, 1.0 / np.log2(np.maximum(first, 1) + 1.0), 0.0)
        s_tgt = S[np.arange(be - bs), i_t[idx]]; rank = (S > s_tgt[:, None]).sum(1) + 1
        ht[idx] = (rank <= K_TOP).astype(float)
        idc[idx] = np.where(rank <= K_TOP, 1.0 / np.log2(rank + 1.0), 0.0)
    return cm, cn, ht, idc


def pu(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v); np.add.at(c, inv, 1); return float((s / c).mean())


def boot_p(a, b, rng):
    d = a - b; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in tqdm(range(BOOT), desc="bootstrap", leave=False)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def holm(pv):
    idx = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * pv[i]); adj[i] = min(run, 1.0)
    return adj


def greedy_steck(sb_full, df, p_user, icm, excl, nmac, lam):
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); idc = np.zeros(n)
    for r in tqdm(range(n), desc="Steck-a greedy", leave=False):
        uu = int(u[r]); s = sb_full[uu].astype(np.float64).copy()
        cc = excl.indices[excl.indptr[uu]:excl.indptr[uu + 1]]
        if len(cc): s[cc] = -np.inf
        cand = np.argpartition(-s, N_CAND - 1)[:N_CAND]; cs = s[cand]; fin = np.isfinite(cs)
        cand = cand[fin]; cs = cs[fin]
        if len(cand) == 0: continue
        rel = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12); cmac = icm[cand]; p = p_user[uu]
        chosen = []; counts = np.zeros(nmac); avail = np.ones(len(cand), bool)
        for _ in range(min(K_TOP, len(cand))):
            L = len(chosen); q = (counts[None, :] + np.eye(nmac)[cmac]) / (L + 1.0)
            qs = (1 - ALPHA_STECK) * q + ALPHA_STECK * p[None, :]
            kl = np.sum(p[None, :] * np.log2((p[None, :] + 1e-12) / (qs + 1e-12)), axis=1)
            obj = (1 - lam) * rel - lam * kl; obj[~avail] = -np.inf
            pk = int(np.argmax(obj)); chosen.append(pk); avail[pk] = False; counts[cmac[pk]] += 1
        items = cand[np.array(chosen)]; mac = icm[items]; mt = mac == tm[r]
        if mt.any(): pos = int(mt.argmax()) + 1; cm[r] = 1.0 / pos; cn[r] = 1.0 / np.log2(pos + 1.0)
        rank_t = int((s > s[i_t[r]]).sum()) + 1  # rank item nel backbone (approssimazione per R@20 di Steck-a)
        ht[r] = float((items == i_t[r]).any())
        if ht[r]: idc[r] = 1.0 / np.log2(np.where(items == i_t[r])[0][0] + 2.0)
    return cm, cn, ht, idc


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    rng = np.random.default_rng(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== GATE B_full su {city} (device={dev}) ===", flush=True)

    # ---------- descrittore + situazioni + κ* (val) ----------
    print("[1] descrittore + selezione K/ε...", flush=True)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)

    def assign(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps)
        return membership_from_assign(k, comp, isb, K), (1.0 / np.maximum(comp.sum(1), 1).astype(np.float32))
    mem_va, gam_va = assign(vva); mem_te, gam_te = assign(vte)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); b_bar = b_z.mean(0)
    n_users = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    Pu = np.zeros((n_users, nmac)); np.add.at(Pu, (umac, cmt), 1.0); Pu[Pu.sum(1) == 0] = 1.0; Pu /= Pu.sum(1, keepdims=True)
    uv = ds["df_val"]["u_idx"].values.astype(np.int64); ute = ds["df_test"]["u_idx"].values.astype(np.int64)
    print(f"    K={K} ε={eps}", flush=True)

    def dmac(method, split):
        mem = mem_va if split == "val" else mem_te; u = uv if split == "val" else ute
        if method == "SIT": return mem.astype(np.float32) @ b_z
        if method == "Steck-b": return b_z_user[u]
        if method == "UNI_mean": return np.broadcast_to(b_bar, (len(u), nmac))
        return None
    print("[2] selezione κ* su VAL (SIT/Steck-b/UNI)...", flush=True)
    kstar = {}
    for method in tqdm(["SIT", "Steck-b", "UNI_mean"], desc="κ-tuning"):
        dv = dmac(method, "val"); best = (-1, None)
        for kap in KAPPA_GRID:
            vm = cat_mrr(sb, ds["df_val"], dv, gam_va, icm, excl, kap).mean()
            if vm > best[0]: best = (vm, kap)
        kstar[method] = best[1]
    print(f"    κ*: { {k: kstar[k] for k in kstar} }", flush=True)

    # ---------- B_full: tuning iperparametri su VAL ----------
    print("[3] B_full: tuning (emb,lr) su VAL...", flush=True)
    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm_full = (df_all.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"])
                .reindex(np.arange(ds["n_items"]), fill_value=0).values.astype(np.int64))
    feats_tr = feats_from_df(ds["df_train"], icm_full, nmac)
    feats_va = feats_from_df(ds["df_val"], icm_full, nmac)
    feats_te = feats_from_df(ds["df_test"], icm_full, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.0
    val_sub = rng.choice(len(ds["df_val"]), min(VAL_SUB, len(ds["df_val"])), replace=False)
    df_val_sub = ds["df_val"].iloc[val_sub].reset_index(drop=True)
    feats_va_sub = {k: v[val_sub] for k, v in feats_va.items()}

    best = {"val": -1, "state": None, "cfg": None}
    for (emb, lr) in tqdm(BF_GRID, desc="B_full grid"):
        torch.manual_seed(SEED)
        spec = FeatureSpec(n_users=n_users, n_items=ds["n_items"], n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac)
        model = ContextAwareFM(spec, d=emb).to(dev)
        for start in range(0, BF_EPOCHS, BF_CKPT):
            train_b_full(model, feats_tr, mask, icm_full, np.zeros(ds["n_items"], np.int64),
                         dev, lr=lr, n_epochs=BF_CKPT, verbose=False)
            sc = score_test(model, feats_va_sub, spec, icm_full, dev)
            vm = metrics_rows(lambda ub, idx: sc[idx], df_val_sub, icm, excl, nmac)[0].mean()
            ep = start + BF_CKPT
            print(f"    B_full emb={emb} lr={lr} ep={ep}: val Cat-MRR={vm:.5f}", flush=True)
            if vm > best["val"]:
                best = {"val": vm, "state": copy.deepcopy(model.state_dict()), "cfg": (emb, lr, ep), "spec": spec}
    print(f"    B_full best (val): emb,lr,ep={best['cfg']}  valCatMRR={best['val']:.5f}", flush=True)

    # ---------- TEST: una volta sola ----------
    print("[4] TEST (una volta sola)...", flush=True)
    bf_model = ContextAwareFM(best["spec"], d=best["cfg"][0]).to(dev); bf_model.load_state_dict(best["state"])
    bf_scores = score_test(bf_model, feats_te, best["spec"], icm_full, dev)  # [n_test, n_items]

    dft = ds["df_test"]
    ev = {}
    ev["BASE"] = metrics_rows(lambda ub, idx: sb[ub], dft, icm, excl, nmac)
    for method in ["SIT", "Steck-b", "UNI_mean"]:
        dm = dmac(method, "test"); kap = kstar[method]
        ev[f"{method}@κ*"] = metrics_rows(
            lambda ub, idx, dm=dm, kap=kap: sb[ub] + kap * gam_te[idx][:, None].astype(np.float32) * dm[idx][:, icm],
            dft, icm, excl, nmac)
    ev["Steck-a"] = greedy_steck(sb, dft, Pu, icm, excl, nmac, LAM)
    ev["B_full"] = metrics_rows(lambda ub, idx: bf_scores[idx], dft, icm, excl, nmac)

    u = dft["u_idx"].values
    rows = []
    for name, (cm, cn, ht, idc) in ev.items():
        rows.append({"city": city, "method": name, "CatMRR": round(float(cm.mean()), 5),
                     "CatNDCG": round(float(cn.mean()), 5), "R20": round(pu(ht, u), 5),
                     "NDCG20": round(pu(idc, u), 5)})
    # confronti vs SIT (Holm)
    sit = ev["SIT@κ*"]
    comps = {"B_full": ev["B_full"], "Steck-b@κ*": ev["Steck-b@κ*"], "Steck-a": ev["Steck-a"], "BASE": ev["BASE"]}
    diffs = {k: boot_p(sit[0], v[0], rng) for k, v in comps.items()}
    hp = holm(np.array([v[3] for v in diffs.values()]))

    OUT = CLEAN / "outputs_results"; pd.DataFrame(rows).to_csv(OUT / f"gate_bfull_{city}.csv", index=False)
    print(f"\n===== GATE B_full {city.upper()} — TEST (K={K}, ε={eps}, κ* val) =====")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nConfronti vs SIT@κ* (Δ CatMRR, bootstrap+Holm):")
    for (k, (m, lo, hi, p)), ph in zip(diffs.items(), hp):
        print(f"  SIT − {k:<11}: {m:+.5f} [{lo:+.5f},{hi:+.5f}] p_holm={ph:.4f}")
    # verdetto pre-registrato vs B_full
    dC = ev["SIT@κ*"][0].mean() - ev["B_full"][0].mean()      # SIT - B_full (Cat-MRR)
    dR = pu(ev["SIT@κ*"][2], u) - pu(ev["B_full"][2], u)       # SIT - B_full (R@20)
    if dC >= 0: verdict = "PASS-FORTE (SIT ≥ B_full su Cat-MRR)"
    elif (-dC) <= MARGIN and dR >= 0: verdict = f"PASS-DEBOLE (entro {MARGIN} su CatMRR e SIT≥B_full su R@20)"
    else: verdict = f"FAIL (B_full batte SIT di {-dC:+.4f} > {MARGIN}; angolo ottimizzatore cade)"
    print(f"\n  ΔCatMRR(SIT−B_full)={dC:+.5f}  ΔR20(SIT−B_full)={dR:+.5f}")
    print(f"  >>> VERDETTO (soglia pre-registrata MARGIN={MARGIN}): {verdict}")
    print(f"\n→ outputs_results/gate_bfull_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
