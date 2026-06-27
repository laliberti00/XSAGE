"""[venv-xsage] BATTERIA DUE-ASSI — X-SAGE su {B_blind(BPR), B_full(context-aware)}, accuratezza
+ fairness, κ selezionato su VAL per (backbone × metodo). Ingloba A.5 (UNI_glob=proxy CPFair,
Coverage) e la lente per-situazione. Multi-seed per le SD. Bootstrap+Holm. TOST sul non-degrado.

Headline: SIT-su-B_full vs B_full — il layer situazionale aggiunge fairness SENZA costare accuratezza?
Uso:  python scripts/ml1m/battery_bfull.py ml1m [n_seed]      (n_seed default 1; overnight: 5)
Output: outputs_results/battery_bfull_<city>.csv  —  NESSUN commit.
"""
import copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
try: from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x
from mind_prep import build_descriptor, membership_from_assign
from eval_kappa import select_K, select_eps
import os
from train_bfull import feats_from_df, score_test
SAFE = os.environ.get("BFULL_SAFE") == "1"   # memory-safe: matrice B_full in float16 (metà RAM)
def _score(model, feats, spec, icm, dev):
    return score_test(model, feats, spec, icm, dev, dtype=np.float16 if SAFE else np.float32)
def _rows(sc):
    return lambda idx: sc[idx]
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from pipeline.step02_models.xsage.backbone_full import ContextAwareFM, FeatureSpec, train_b_full
from xsage.recommendation import fit_situation_biases_z

K_TOP, BATCH, ALPHA, BOOT = 20, 1024, 50.0, 1500
KAPPA_GRID = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
BF_GRID = [(32, 5e-3), (64, 5e-3)]; BF_EPOCHS, BF_CKPT, VAL_SUB = 9, 3, 12000
N_CAND, ALPHA_STECK, LAM, TOST_MARGIN = 60, 0.01, 0.99, 0.005


def js(p, q, e=1e-9):
    p = p + e; p = p / p.sum(1, keepdims=True); q = q + e; q = q / q.sum(1, keepdims=True)
    m = .5 * (p + q); kl = lambda a, b: np.sum(a * np.log2(a / b), 1); return .5 * kl(p, m) + .5 * kl(q, m)


def gini(c):
    x = np.sort(c.astype(np.float64)); n = len(x); s = x.sum()
    return 0. if s <= 0 else float((2 * np.sum(np.arange(1, n + 1) * x) / (n * s)) - (n + 1) / n)


def full_eval(bb_rows, nudge, kappa, df, z, gam, icm, excl, G1, pur, K, nmac):
    """bb_rows(idx)->[B,n_items] backbone; nudge: ('macro',dmac)|('glob',None)|None."""
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); idc = np.zeros(n); jsv = np.zeros(n); lt = np.zeros(n)
    expo = np.zeros(len(icm)); sit_mac = np.zeros((K, nmac))
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); idx = np.arange(bs, be); u_b = u[idx]
        S = bb_rows(idx).astype(np.float32, copy=True)
        if nudge is not None:
            kind, dmac = nudge
            if kind == "macro": S = S + kappa * gam[idx][:, None].astype(np.float32) * dmac[idx][:, icm]
            else: S = S + kappa * G1[None, :].astype(np.float32)        # UNI_glob: flat long-tail su item
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(u_b[j])]:excl.indptr[int(u_b[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1); topk = np.take_along_axis(part, order, 1)
        mac = icm[topk]; match = mac == tm[idx, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
        cm[idx] = np.where(first > 0, 1. / np.maximum(first, 1), 0.); cn[idx] = np.where(first > 0, 1. / np.log2(np.maximum(first, 1) + 1.), 0.)
        s_tgt = S[np.arange(be - bs), i_t[idx]]; rank = (S > s_tgt[:, None]).sum(1) + 1
        ht[idx] = (rank <= K_TOP).astype(float); idc[idx] = np.where(rank <= K_TOP, 1. / np.log2(rank + 1.), 0.)
        lt[idx] = G1[topk].mean(1); np.add.at(expo, topk.ravel(), 1.)
        q = np.zeros((be - bs, nmac)); rows = np.repeat(np.arange(be - bs), K_TOP); np.add.at(q, (rows, mac.ravel()), 1.)
        jsv[idx] = js(pur[idx], q); zb = z[idx]
        for j in range(be - bs): np.add.at(sit_mac[zb[j]], mac[j], 1.)
    return dict(cm=cm, cn=cn, ht=ht, idc=idc, js=jsv, lt=lt, expo=expo, sit_mac=sit_mac, u=u, z=z)


def pu(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq)); np.add.at(s, inv, v); np.add.at(c, inv, 1); return float((s / c).mean())


def lens_spread(sit_mac):
    glob = sit_mac.sum(0); glob = glob / max(glob.sum(), 1); K = len(sit_mac); KL = np.zeros(K)
    for k in range(K):
        p = sit_mac[k] / max(sit_mac[k].sum(), 1); m = p > 0
        KL[k] = float(np.sum(p[m] * np.log2(p[m] / np.maximum(glob[m], 1e-12))))
    r = KL / max(KL.mean(), 1e-12); return float(r.max()), int((r >= 1.5).sum())


def boot(a, b, rng):
    d = a - b; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2. * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(np.std(bs)), float(lo), float(hi), float(min(p, 1.))


def greedy_steck(sb, df, pur, icm, excl, nmac, rows=None):
    # rows=None: sb è per-utente [n_users x n_items] (sb[uu]); rows!=None: matrice per-richiesta [n x n_items] (rows[r])
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); idc = np.zeros(n)
    for r in tqdm(range(n), desc="Steck-a", leave=False):
        uu = int(u[r]); s = (rows[r] if rows is not None else sb[uu]).astype(np.float64).copy(); cc = excl.indices[excl.indptr[uu]:excl.indptr[uu + 1]]
        if len(cc): s[cc] = -np.inf
        cand = np.argpartition(-s, N_CAND - 1)[:N_CAND]; cs = s[cand]; fin = np.isfinite(cs); cand = cand[fin]; cs = cs[fin]
        if len(cand) == 0: continue
        rel = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12); cmac = icm[cand]; p = pur[uu]
        ch = []; cnt = np.zeros(nmac); av = np.ones(len(cand), bool)
        for _ in range(min(K_TOP, len(cand))):
            L = len(ch); qd = (cnt[None, :] + np.eye(nmac)[cmac]) / (L + 1.); qs = (1 - ALPHA_STECK) * qd + ALPHA_STECK * p[None, :]
            kl = np.sum(p[None, :] * np.log2((p[None, :] + 1e-12) / (qs + 1e-12)), 1); ob = (1 - LAM) * rel - LAM * kl; ob[~av] = -np.inf
            pk = int(np.argmax(ob)); ch.append(pk); av[pk] = False; cnt[cmac[pk]] += 1
        items = cand[np.array(ch)]; mac = icm[items]; mt = mac == tm[r]
        if mt.any(): pos = int(mt.argmax()) + 1; cm[r] = 1. / pos; cn[r] = 1. / np.log2(pos + 1.)
        if (items == i_t[r]).any(): ht[r] = 1.; idc[r] = 1. / np.log2(np.where(items == i_t[r])[0][0] + 2.)
    return dict(cm=cm, cn=cn, ht=ht, idc=idc, js=np.zeros(n), lt=np.zeros(n), expo=np.zeros(len(icm)), sit_mac=np.zeros((1, nmac)), u=u, z=np.zeros(n, int))


def run_seed(city, seed, dev):
    rng = np.random.default_rng(seed)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]; G1 = D0["G1"].astype(np.float32)
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    def asg(v):
        _, k, comp, isb = _assign(v, fit.prototypes, eps); return membership_from_assign(k, comp, isb, K), (1. / np.maximum(comp.sum(1), 1).astype(np.float32)), k.astype(int)
    mem_va, gam_va, z_va = asg(vva); mem_te, gam_te, z_te = asg(vte)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); b_bar = b_z.mean(0)
    nU = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    bzu = fit_situation_biases_z(umac, cmt, nU, nmac, alpha=ALPHA)
    Pu = np.zeros((nU, nmac)); np.add.at(Pu, (umac, cmt), 1.); Pu[Pu.sum(1) == 0] = 1.; Pu /= Pu.sum(1, keepdims=True)
    dfv, dft = ds["df_val"], ds["df_test"]; uv = dfv["u_idx"].values.astype(np.int64); ut = dft["u_idx"].values.astype(np.int64)
    purv, purt = Pu[uv], Pu[ut]

    # ---- B_full: tuning su val, score val+test ----
    dfa = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icmF = (dfa.groupby("i_idx")["cat_macro"].first().map(ds["macro_to_idx"]).reindex(np.arange(ds["n_items"]), fill_value=0).values.astype(np.int64))
    ftr = feats_from_df(ds["df_train"], icmF, nmac); fva = feats_from_df(dfv, icmF, nmac); fte = feats_from_df(dft, icmF, nmac)
    mask = (ds["urm_train"] + ds["urm_val"]).tocsr(); mask.data[:] = 1.
    sub = rng.choice(len(dfv), min(VAL_SUB, len(dfv)), replace=False); dvs = dfv.iloc[sub].reset_index(drop=True); fvs = {k: v[sub] for k, v in fva.items()}
    best = {"val": -1, "st": None, "cfg": None}
    for (emb, lr) in tqdm(BF_GRID, desc=f"B_full s{seed}"):
        torch.manual_seed(seed); spec = FeatureSpec(n_users=nU, n_items=ds["n_items"], n_macros=nmac, n_fine=1, n_geo=0, n_intent_last=nmac); mdl = ContextAwareFM(spec, d=emb).to(dev)
        for st in range(0, BF_EPOCHS, BF_CKPT):
            train_b_full(mdl, ftr, mask, icmF, np.zeros(ds["n_items"], np.int64), dev, lr=lr, n_epochs=BF_CKPT, verbose=False)
            sc = _score(mdl, fvs, spec, icmF, dev); vm = full_eval(_rows(sc), None, 0, dvs, z_va[sub], gam_va[sub], icm, excl, G1, purv[sub], K, nmac)["cm"].mean()
            if vm > best["val"]: best = {"val": vm, "st": copy.deepcopy(mdl.state_dict()), "cfg": (emb, lr, st + BF_CKPT), "spec": spec}
    bm = ContextAwareFM(best["spec"], d=best["cfg"][0]).to(dev); bm.load_state_dict(best["st"])
    bfv = _score(bm, fva, best["spec"], icmF, dev); bft = _score(bm, fte, best["spec"], icmF, dev)

    backbones = {"B_blind": (lambda idx, u=uv: sb[u[idx]], lambda idx, u=ut: sb[u[idx]]),  # (val_fn, test_fn)
                 "B_full": (_rows(bfv), _rows(bft))}
    # ---- backbone CITABILI extra (env-gated): score val+test precalcolati a parte ----
    # XTRA_BACKBONES="EASE,SASRec,xDeepFM". Convenzione file in data/<city>/backbone/:
    #   <NAME>.scores_user.npy  [n_users x n_items]  → statico per-utente (come B_blind)
    #   <NAME>.scores_val.npy + <NAME>.scores_test.npy  [n_rows x n_items]  → per-richiesta
    # Ogni backbone così aggiunto eredita IDENTICO il trattamento (BASE/SIT/Steck-b/UNI, κ* su val, stat).
    bdir = CLEAN / "data" / city / "backbone"
    xtra_steck = {}   # name -> ('user', M_user) | ('rows', M_test)  per Steck-a coerente sui nuovi backbone
    for name in [x.strip() for x in os.environ.get("XTRA_BACKBONES", "").split(",") if x.strip()]:
        fu = bdir / f"{name}.scores_user.npy"
        if fu.exists():
            M = np.load(fu, mmap_mode="r")        # memory-map: resta su disco, batch on-demand (RAM-safe ml1m)
            backbones[name] = (lambda idx, M=M, u=uv: M[u[idx]], lambda idx, M=M, u=ut: M[u[idx]])
            xtra_steck[name] = ("user", M)
        else:
            sv = np.load(bdir / f"{name}.scores_val.npy", mmap_mode="r"); st = np.load(bdir / f"{name}.scores_test.npy", mmap_mode="r")
            backbones[name] = (_rows(sv), _rows(st))
            xtra_steck[name] = ("rows", st)
        print(f"  [+] backbone extra: {name}", flush=True)
    def dmac(method, split):
        mem = mem_va if split == "val" else mem_te; u = uv if split == "val" else ut
        if method == "SIT": return mem.astype(np.float32) @ b_z
        if method == "Steck-b": return bzu[u]
        if method == "UNI_mean": return np.broadcast_to(b_bar, (len(u), nmac))
        return None
    rows = []; store = {}
    for bbn, (vfn, tfn) in backbones.items():
        gv, gt, zv, zt = gam_va, gam_te, z_va, z_te
        # BASE (backbone nudo)
        e = full_eval(tfn, None, 0, dft, zt, gt, icm, excl, G1, purt, K, nmac); store[(bbn, "BASE")] = e
        # metodi additivi: κ* su val
        for method in ["SIT", "Steck-b", "UNI_mean"]:
            dv = dmac(method, "val"); best_k = max(KAPPA_GRID, key=lambda kap: full_eval(vfn, ("macro", dv), kap, dfv, zv, gv, icm, excl, G1, purv, K, nmac)["cm"].mean())
            dt = dmac(method, "test"); e = full_eval(tfn, ("macro", dt), best_k, dft, zt, gt, icm, excl, G1, purt, K, nmac); e["kstar"] = best_k; store[(bbn, method)] = e
        # UNI_glob (proxy CPFair): κ* su val
        best_k = max(KAPPA_GRID, key=lambda kap: full_eval(vfn, ("glob", None), kap, dfv, zv, gv, icm, excl, G1, purv, K, nmac)["cm"].mean())
        e = full_eval(tfn, ("glob", None), best_k, dft, zt, gt, icm, excl, G1, purt, K, nmac); e["kstar"] = best_k; store[(bbn, "UNI_glob")] = e
    store[("B_blind", "Steck-a")] = greedy_steck(sb, dft, Pu, icm, excl, nmac)
    # Steck-a (calibrazione greedy di Steck) anche sui backbone extra → set metodi COMPLETO come ml-1m
    if os.environ.get("XTRA_STECKA", "1") == "1":
        for name, (kind, Mx) in xtra_steck.items():
            if kind == "user": store[(name, "Steck-a")] = greedy_steck(Mx, dft, Pu, icm, excl, nmac)
            else: store[(name, "Steck-a")] = greedy_steck(None, dft, Pu, icm, excl, nmac, rows=Mx)

    for (bbn, mth), e in store.items():
        mx, nsink = lens_spread(e["sit_mac"]) if e["sit_mac"].shape[0] == K else (np.nan, 0)
        rows.append(dict(seed=seed, backbone=bbn, method=mth, kstar=e.get("kstar", ""),
                         CatMRR=float(e["cm"].mean()), CatNDCG=float(e["cn"].mean()), R20=pu(e["ht"], e["u"]),
                         NDCG20=pu(e["idc"], e["u"]), JS=float(e["js"].mean()) if e["js"].any() else np.nan,
                         LT20=float(e["lt"].mean()), Coverage=float((e["expo"] > 0).mean()), Gini=gini(e["expo"]),
                         lensKLmax=mx, n_sink=nsink))
    return rows, store, K, eps, ut


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; nseed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); rng = np.random.default_rng(0)
    print(f"=== BATTERIA B_full {city} (device={dev}, seeds={nseed}) ===", flush=True)
    allrows = []; last = None
    for s in range(nseed):
        print(f"\n--- SEED {42+s} ---", flush=True); r, store, K, eps, ut = run_seed(city, 42 + s, dev); allrows += r; last = (store, ut)
    df = pd.DataFrame(allrows)
    OUT = CLEAN / "outputs_results"; df.to_csv(OUT / f"battery_bfull_{city}.csv", index=False)
    try:  # riepilogo difensivo: il CSV è già salvato, un bug qui non rovina l'overnight
        num = ["CatMRR", "CatNDCG", "R20", "NDCG20", "JS", "LT20", "Coverage", "Gini", "lensKLmax"]
        agg = df.groupby(["backbone", "method"])[num].agg(['mean', 'std']).round(5)
        print(f"\n===== {city.upper()} — media ± SD ({nseed} seed) =====")
        print(agg[[("CatMRR", "mean"), ("CatMRR", "std"), ("R20", "mean"), ("R20", "std"),
                   ("LT20", "mean"), ("Coverage", "mean"), ("Gini", "mean"), ("lensKLmax", "mean")]].to_string())
    except Exception as ex:
        print(f"[riepilogo agg saltato: {ex}; CSV comunque salvato]")
    # headline: SIT-su-B_full vs B_full (ultimo seed, bootstrap+TOST)
    try:
        store, ut = last; rng = np.random.default_rng(123)
        sit = store[("B_full", "SIT")]; bf = store[("B_full", "BASE")]
        m, sd, lo, hi, p = boot(sit["cm"], bf["cm"], rng)
        dr, sdr, lor, hir, pr = boot(sit["ht"], bf["ht"], rng)
        print(f"\n===== HEADLINE: SIT-su-B_full vs B_full (ultimo seed) =====")
        print(f"  dCatMRR = {m:+.5f} +- {sd:.5f} [{lo:+.5f},{hi:+.5f}] p={p:.4f}")
        print(f"  dR@20   = {dr:+.5f} +- {sdr:.5f} [{lor:+.5f},{hir:+.5f}] p={pr:.4f}")
        tost = "EQUIVALENTE (non degrada)" if (lor > -TOST_MARGIN and hir < TOST_MARGIN) else ("NON degrada (>=)" if lor >= 0 else "DEGRADA")
        print(f"  TOST R@20 (margine +-{TOST_MARGIN}): {tost}")
        print(f"  dLT@20 = {sit['lt'].mean()-bf['lt'].mean():+.5f}  dGini = {gini(sit['expo'])-gini(bf['expo']):+.5f}  dCoverage = {(sit['expo']>0).mean()-(bf['expo']>0).mean():+.5f}")
    except Exception as ex:
        print(f"[headline saltato: {ex}; CSV salvato]")
    # headline per i backbone extra citabili (SIT-su-<NAME> vs <NAME>)
    try:
        store, ut = last; rng = np.random.default_rng(123)
        extra = sorted({bb for (bb, mth) in store if bb not in ("B_blind", "B_full")})
        for bb in extra:
            if (bb, "SIT") not in store or (bb, "BASE") not in store: continue
            s = store[(bb, "SIT")]; b = store[(bb, "BASE")]
            m, sd, lo, hi, p = boot(s["cm"], b["cm"], rng)
            dr, sdr, lor, hir, pr = boot(s["ht"], b["ht"], rng)
            tost = "EQUIVALENTE (non degrada)" if (lor > -TOST_MARGIN and hir < TOST_MARGIN) else ("NON degrada (>=)" if lor >= 0 else "DEGRADA")
            print(f"\n===== HEADLINE: SIT-su-{bb} vs {bb} (ultimo seed) =====")
            print(f"  dCatMRR = {m:+.5f} +- {sd:.5f} [{lo:+.5f},{hi:+.5f}] p={p:.4f}")
            print(f"  dR@20   = {dr:+.5f} +- {sdr:.5f} [{lor:+.5f},{hir:+.5f}] p={pr:.4f}  | TOST R@20: {tost}")
    except Exception as ex:
        print(f"[headline extra saltato: {ex}; CSV salvato]")
    print(f"\n-> outputs_results/battery_bfull_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
