"""[venv-xsage] S-1 Task A — Concatenazione: congiunta vs cascata, A CAPACITA' PARI.

PRE-REGISTRATO nel brief S-1. Sette varianti, regola di decisione fissata prima dei numeri.

NOTA FATTUALE (riportata, non nascosta): il brief dice che V5 (contesto->intento) non e' mai
stata provata. docs/ABLATION_STAGED_ml1m.txt riporta invece ENTRAMBI gli ordini —
staged-A ctx->intento 0.14687 e staged-B intento->ctx 0.14902, entrambi 5/5 sopra la congiunta.
Quel confronto esisteva gia' in forma preliminare; qui viene rifatto a capacita' pari e col
controllo di permutazione che mancava.

V1 JOINT auto (riferimento, = metodo sottomesso) · V2 JOINT@K=10 · V3 JOINT@K=12
V4 STAGED intento->contesto · V5 STAGED contesto->intento
V6 STAGED con livello-1 RIMESCOLATO a parita' esatta di taglie (controllo di falsificazione)
V7 blocco singolo: ctx@5, ctx@10, int@5, int@10

Backbone B_blind, 5 semi, kappa selezionato su VALIDATION per ogni variante (procedura corrente:
non esiste un kappa "congelato" per varianti che non sono in battery; per V1 la selezione su val
riproduce il kappa della battery, ed e' cio' che il gate verifica).
Bootstrap a due livelli: ricampiona UTENTI (cluster) e SEMI. Mai richieste.

Uso:  python -m scripts.exp.s1_concat [ml1m ...]
Out:  outputs_results/exp_s1/
"""
import csv as csv_mod
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import ALPHA, build_descriptor, membership_from_assign          # noqa: E402
from eval_kappa import KAPPA_GRID, SEED, cat_mrr, select_K, select_eps          # noqa: E402
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage.recommendation import fit_situation_biases_z                          # noqa: E402
from sklearn.metrics import adjusted_rand_score                                  # noqa: E402

OUT = CLEAN / "outputs_results" / "exp_s1"
SEEDS = [42, 43, 44, 45, 46]
BOOT, MIN_SPLIT, KTOP, BATCH = 800, 200, 20, 1024
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
VARIANTS = ["V1_joint_auto", "V2_joint_K10", "V3_joint_K12", "V4_staged_int2ctx",
            "V5_staged_ctx2int", "V6_staged_shuffled", "V7_ctx_K5", "V7_ctx_K10",
            "V7_int_K5", "V7_int_K10"]


def evaluate(sb, df, dmac, gam, icm, excl, kappa):
    """Cat-MRR@20 (convenzione del gate) + HR@20 + NDCG@20, per richiesta."""
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64)
    tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); hit = np.zeros(n); nd = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); ub = u[bs:be]
        S = sb[ub].astype(np.float32, copy=True)
        if dmac is not None and kappa > 0:
            S = S + kappa * gam[bs:be][:, None].astype(np.float32) * dmac[bs:be][:, icm]
        for j in range(be - bs):
            cc = excl.indices[excl.indptr[int(ub[j])]:excl.indptr[int(ub[j]) + 1]]
            if len(cc): S[j, cc] = -np.inf
        part = np.argpartition(-S, KTOP - 1, axis=1)[:, :KTOP]
        tk = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), axis=1), 1)
        mt = icm[tk] == tm[bs:be, None]; has = mt.any(1)
        first = np.where(has, mt.argmax(1) + 1, 0)
        cm[bs:be] = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
        pos = (tk == i_t[bs:be, None]); h = pos.any(1); rk = np.where(h, pos.argmax(1) + 1, 0)
        hit[bs:be] = h.astype(float)
        nd[bs:be] = np.where(h, 1.0 / np.log2(np.maximum(rk, 1) + 1.0), 0.0)
    return cm, hit, nd, tm


def macro(cm, tm, nmac, m=20):
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).sum() >= m]
    return float(np.mean(per)) if per else np.nan


def _assign_to(fit, eps, K, v):
    _, k, comp, isb = _assign(v, fit.prototypes, eps)
    return (membership_from_assign(k, comp, isb, K).astype(np.float32),
            (1.0 / np.maximum(comp.sum(1), 1)).astype(np.float32), k.astype(np.int64), isb)


def fit_joint(vtr, vva, vte, K, rng, seed, ceil=None):
    """Ritorna anche il fit e eps, cosi' la membership di VALIDATION si ricava dallo STESSO
    clustering invece di rifarlo (era anche la causa del bug: per le ablazioni a blocco singolo
    veniva passato il blocco di TEST come terzo argomento e la membership risultante, lunga
    len(test), veniva poi usata contro la validation, piu' lunga)."""
    if K is None: K = select_K(vtr, ceil, rng)
    eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    mem, gam, kte, isb = _assign_to(fit, eps, K, vte)
    return (fit.core_label.astype(np.int64), mem, gam, K, kte, isb, (fit, eps))


def fit_staged(vtr, vva, vte, A, order, ceil, rng, seed, shuffle_lvl1=False):
    """Cascata. order='ctx_int' -> livello1=contesto. shuffle_lvl1=True: partizione di livello-1
    permutata a parita' ESATTA di taglie e assegnazione di test casuale con le stesse proporzioni
    (controllo: il livello-1 non porta informazione)."""
    b1, b2 = (slice(0, A), slice(A, None)) if order == "ctx_int" else (slice(A, None), slice(0, A))
    vt1, vv1, ve1 = vtr[:, b1], vva[:, b1], vte[:, b1]
    vt2, vv2, ve2 = vtr[:, b2], vva[:, b2], vte[:, b2]
    K1 = select_K(vt1, ceil, rng); eps1 = select_eps(vt1, vv1, K1)
    f1 = fit_rough_kmeans(vt1, K=K1, eps=eps1, seed=seed, max_iter=80)
    z1_tr = f1.core_label.astype(np.int64)
    _, k1_va, _, _ = _assign(vv1, f1.prototypes, eps1)
    _, k1_te, _, _ = _assign(ve1, f1.prototypes, eps1)
    if shuffle_lvl1:
        r2 = np.random.default_rng(9000 + seed)
        z1_tr = r2.permutation(z1_tr)                       # taglie esatte, appartenenza casuale
        p = np.bincount(z1_tr, minlength=K1).astype(float); p /= p.sum()
        k1_va = r2.choice(K1, size=len(vv1), p=p)
        k1_te = r2.choice(K1, size=len(vte), p=p)
    leaf = np.empty(len(vtr), np.int64); reg = {}; base = 0
    for g in range(K1):
        m = (z1_tr == g); ng = int(m.sum())
        if ng < MIN_SPLIT:
            leaf[m] = base; reg[g] = (base, 1, None, None); base += 1; continue
        mv = (k1_va == g); vt2g = vt2[m]; vv2g = vv2[mv] if int(mv.sum()) > 0 else vt2g
        K2 = select_K(vt2g, ceil, rng); eps2 = select_eps(vt2g, vv2g, K2)
        f2 = fit_rough_kmeans(vt2g, K=K2, eps=eps2, seed=seed, max_iter=80)
        leaf[m] = base + f2.core_label.astype(np.int64); reg[g] = (base, K2, f2, eps2); base += K2
    L = base
    mem = np.zeros((len(vte), L), np.float32); comp_all = np.zeros((len(vte), L), np.float32)
    for g in range(K1):
        mt = (k1_te == g)
        if not mt.any(): continue
        b0, K2, f2, eps2 = reg[g]
        if f2 is None: mem[mt, b0] = 1.0; comp_all[mt, b0] = 1.0; continue
        _, k2, c2, isb2 = _assign(ve2[mt], f2.prototypes, eps2)
        rows = np.where(mt)[0]; cols = np.arange(b0, b0 + K2)
        mem[np.ix_(rows, cols)] = membership_from_assign(k2, c2, isb2, K2).astype(np.float32)
        comp_all[np.ix_(rows, cols)] = c2.astype(np.float32)
    gam = (1.0 / np.maximum(comp_all.sum(1), 1)).astype(np.float32)
    kte = mem.argmax(1).astype(np.int64); isb = comp_all.sum(1) > 1
    return leaf, mem, gam, L, kte, isb, comp_all


def _done_set(csv):
    """Combinazioni (variante, seme) gia' presenti nel CSV: la ripresa le salta."""
    if not csv.exists(): return set()
    try:
        d = pd.read_csv(csv)
        return {(r.variant, int(r.seed)) for r in d.itertuples()}
    except Exception:
        return set()


def _append(csv, row):
    """Append immediato con flush: il file e' leggibile anche mentre il run e' in corso, e
    un'interruzione costa al massimo UNA combinazione invece dell'intera notte."""
    new = not csv.exists()
    with open(csv, "a", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(row.keys()))
        if new: w.writeheader()
        w.writerow(row)
        f.flush(); os.fsync(f.fileno())


def run(city):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "peruser").mkdir(exist_ok=True); (OUT / "parts").mkdir(exist_ok=True)
    csv_path = OUT / f"s1_variants_{city}.csv"
    done = _done_set(csv_path)
    if done: print(f"[{city}] RIPRESA: {len(done)}/{len(SEEDS)*len(VARIANTS)} combinazioni gia' fatte", flush=True)
    bt = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")

    for seed in SEEDS:
        todo = [v for v in VARIANTS if (v, seed) not in done]
        if not todo:
            print(f"[{city}] seme {seed}: gia' completo, salto (nessun descrittore da ricostruire)", flush=True)
            continue
        rng = np.random.default_rng(seed)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]
        sb = D0["sb"]; cmt = D0["cmt"]
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        A = vtr.shape[1] - nmac; ceil = int(D0["attractors"].sum()) + 2
        dfv, dft = ds["df_val"], ds["df_test"]
        ute = dft["u_idx"].values.astype(np.int64)

        def blk(sl): return (vtr[:, sl], vva[:, sl], vte[:, sl])
        cfg = {
            "V1_joint_auto":      lambda: fit_joint(vtr, vva, vte, None, rng, seed, ceil),
            "V2_joint_K10":       lambda: fit_joint(vtr, vva, vte, 10, rng, seed),
            "V3_joint_K12":       lambda: fit_joint(vtr, vva, vte, 12, rng, seed),
            "V4_staged_int2ctx":  lambda: fit_staged(vtr, vva, vte, A, "int_ctx", ceil, rng, seed),
            "V5_staged_ctx2int":  lambda: fit_staged(vtr, vva, vte, A, "ctx_int", ceil, rng, seed),
            "V6_staged_shuffled": lambda: fit_staged(vtr, vva, vte, A, "int_ctx", ceil, rng, seed, True),
            "V7_ctx_K5":  lambda: fit_joint(*blk(slice(0, A)), 5, rng, seed),
            "V7_ctx_K10": lambda: fit_joint(*blk(slice(0, A)), 10, rng, seed),
            "V7_int_K5":  lambda: fit_joint(*blk(slice(A, None)), 5, rng, seed),
            "V7_int_K10": lambda: fit_joint(*blk(slice(A, None)), 10, rng, seed),
        }
        for v in todo:
            z_tr, mem, gam, Keff, kte, isb, extra = cfg[v]()
            b_z = fit_situation_biases_z(z_tr, cmt, Keff, nmac, alpha=ALPHA)
            if v.startswith(("V4", "V5", "V6")):
                kap = float(bt[(bt.seed == seed) & (bt.backbone == "B_blind") &
                               (bt.method == "SIT")]["kstar"].iloc[0])
            else:
                fitobj, epsobj = extra
                vv = vva if v.startswith(("V1", "V2", "V3")) else \
                    vva[:, slice(0, A) if "ctx" in v else slice(A, None)]
                memv, gamv, _, _ = _assign_to(fitobj, epsobj, Keff, vv)
                best = (-1.0, None)
                for kk in KAPPA_GRID:
                    m = float(cat_mrr(sb, dfv, memv @ b_z, gamv, icm, excl, kk).mean())
                    if m > best[0]: best = (m, kk)
                kap = best[1]
            cm, hit, nd, tm = evaluate(sb, dft, mem @ b_z, gam, icm, excl, kap)
            sizes = np.bincount(kte, minlength=Keff)
            row = dict(dataset=city, seed=seed, variant=v, kappa=kap, K_eff=int(Keff),
                       n_groups_nonempty=int((sizes > 0).sum()),
                       size_max_share=round(float(sizes.max() / sizes.sum()), 5),
                       size_min=int(sizes[sizes > 0].min()) if (sizes > 0).any() else 0,
                       boundary_share=round(float(isb.mean()), 5),
                       macroCatMRR=round(macro(cm, tm, nmac), 6), CatMRR=round(float(cm.mean()), 6),
                       HR20=round(float(hit.mean()), 6), NDCG20=round(float(nd.mean()), 6),
                       n_req=len(dft), n_users=int(dft.u_idx.nunique()))
            uq, inv = np.unique(ute, return_inverse=True)
            su = np.zeros(len(uq)); cu = np.zeros(len(uq))
            np.add.at(su, inv, cm); np.add.at(cu, inv, 1)
            np.save(OUT / "peruser" / f"{city}__{v}__{seed}.npy", su / cu)   # per i CI clusterizzati
            if seed == SEEDS[0]: np.save(OUT / "parts" / f"{city}__{v}.npy", kte)
            _append(csv_path, row)
            print(f"  [{city}] s{seed} {v:20} K={Keff:<3} kap={kap:<5} "
                  f"macro={row['macroCatMRR']:.5f}  [salvato]", flush=True)

    parts = {f.stem.split("__")[1]: np.load(f) for f in sorted((OUT / "parts").glob(f"{city}__*.npy"))}
    if len(parts) >= 2:
        ks = [v for v in VARIANTS if v in parts]
        ari = [dict(dataset=city, a=a, b=b,
                    ARI=round(float(adjusted_rand_score(parts[a], parts[b])), 4))
               for i, a in enumerate(ks) for b in ks[i + 1:]]
        pd.DataFrame(ari).to_csv(OUT / f"s1_ari_{city}.csv", index=False)
    n = len(_done_set(csv_path))
    print(f"[{city}] -> {csv_path.name}  ({n}/{len(SEEDS)*len(VARIANTS)} combinazioni)", flush=True)


def main():
    for c in (sys.argv[1:] or ["ml1m"]):
        print(f"\n===== S-1 · {c.upper()} =====", flush=True); run(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
