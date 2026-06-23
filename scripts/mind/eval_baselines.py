"""[venv-xsage] Confronto baseline su un 2°/3° dominio (MIND o ml1m), col MACCHINARIO della
calibrazione Foursquare: SIT vs BASE / UNI_mean / Steck-b (target utente) / Steck-a (greedy reale).
Selezione K/ε VERA (silhouette + tetto |A|, banda boundary). Bootstrap + Holm. κ=0.25 canonico.

Uso:  python scripts/mind/eval_baselines.py <city>     (city = mind | ml1m)
Output: outputs_results/baselines/<city>_calibration.csv
La domanda: su un dominio a utenti profondi (ml1m), SIT regge contro Steck-b/Steck-a o perde come Foursquare?
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, GAMMA, DEPTH, N
from pipeline.step02_models.xsage.l2_comprehension import _assign, adjusted_rand_score, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

KAPPA, K_TOP, BATCH, ALPHA, BOOT = 0.25, 20, 1024, 50.0, 1500
K_RANGE = [3, 4, 5, 6, 7, 8, 9]; EPS_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
BAND = (0.10, 0.30); SIL_N, SEED = 20000, 42
N_CAND, ALPHA_STECK, LAM = 60, 0.01, 0.99


def select_K(vtr, ceil, rng):
    n = len(vtr); sidx = rng.choice(n, SIL_N, replace=False) if n > SIL_N else np.arange(n)
    best_k, best_s = K_RANGE[0], -1
    for K in K_RANGE:
        r = fit_rough_kmeans(vtr, K=K, eps=0.0, seed=SEED, max_iter=60)
        lab = r.core_label
        if len(np.unique(lab)) < 2: continue
        s = silhouette_score(vtr[sidx], lab[sidx])
        if s > best_s: best_s, best_k = s, K
    return min(best_k, ceil)


def select_eps(vtr, vva, K):
    cells = []
    for eps in EPS_GRID:
        r = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
        _, _, _, isb = _assign(vva, r.prototypes, eps)
        cells.append((eps, float(isb.mean())))
    inb = [(e, b) for e, b in cells if BAND[0] <= b <= BAND[1]]
    if inb: return min(inb, key=lambda x: abs(x[1] - 0.20))[0]
    return min(cells, key=lambda x: abs(x[1] - 0.20))[0]


def js(p, q, eps=1e-9):
    p = p + eps; p /= p.sum(1, keepdims=True); q = q + eps; q /= q.sum(1, keepdims=True)
    m = 0.5 * (p + q); kl = lambda a, b: np.sum(a * np.log2(a / b), axis=1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def metrics(S, u_b, i_t, tm, excl, p_user_b, nmac, icm):
    B = S.shape[0]
    for j in range(B):
        uu = int(u_b[j]); cc = excl.indices[excl.indptr[uu]:excl.indptr[uu + 1]]
        if len(cc): S[j, cc] = -np.inf
    part = np.argpartition(-S, K_TOP - 1, axis=1)[:, :K_TOP]
    order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
    macros = icm[np.take_along_axis(part, order, 1)]
    match = macros == tm[:, None]; has = match.any(1); first = np.where(has, match.argmax(1) + 1, 0)
    cmrr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    cndcg = np.where(first > 0, 1.0 / np.log2(np.maximum(first, 1) + 1.0), 0.0)
    s_tgt = S[np.arange(B), i_t]; hit = ((S > s_tgt[:, None]).sum(1) + 1 <= K_TOP).astype(float)
    q = np.zeros((B, nmac)); rows = np.repeat(np.arange(B), K_TOP)
    np.add.at(q, (rows, macros.ravel()), 1.0)
    return cmrr, cndcg, hit, js(p_user_b, q)


def additive(sb_full, dmac_fn, df, gamma, icm, excl, p_user_rows, nmac):
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); jsv = np.zeros(n)
    for bs in range(0, n, BATCH):
        be = min(n, bs + BATCH); u_b = u[bs:be]
        S = sb_full[u_b].astype(np.float32, copy=True)
        dm = dmac_fn(bs, be)
        if dm is not None:
            S = S + KAPPA * gamma[bs:be][:, None].astype(np.float32) * dm[:, icm]
        cm[bs:be], cn[bs:be], ht[bs:be], jsv[bs:be] = metrics(S.copy(), u_b, i_t[bs:be], tm[bs:be], excl, p_user_rows[bs:be], nmac, icm)
    return cm, cn, ht, jsv, u


def greedy_steck(sb_full, df, p_user, icm, excl, nmac, lam):
    u = df["u_idx"].values.astype(np.int64); i_t = df["i_idx"].values.astype(np.int64); tm = icm[i_t]; n = len(df)
    cm = np.zeros(n); cn = np.zeros(n); ht = np.zeros(n); jsv = np.zeros(n)
    for r in range(n):
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
            pick = int(np.argmax(obj)); chosen.append(pick); avail[pick] = False; counts[cmac[pick]] += 1
        items = cand[np.array(chosen)]; mac = icm[items]; mt = mac == tm[r]
        if mt.any(): pos = int(mt.argmax()) + 1; cm[r] = 1.0 / pos; cn[r] = 1.0 / np.log2(pos + 1.0)
        ht[r] = float((items == i_t[r]).any())
        qd = np.zeros((1, nmac))
        for m in mac: qd[0, m] += 1
        jsv[r] = js(p[None, :], qd)[0]
    return cm, cn, ht, jsv, u


def pu(v, u):
    uq, inv = np.unique(u, return_inverse=True); s = np.zeros(len(uq)); c = np.zeros(len(uq))
    np.add.at(s, inv, v); np.add.at(c, inv, 1); return float((s / c).mean())


def boot_diff(a, b, rng):
    d = a - b; n = len(d); bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))


def holm(pv):
    idx = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pv[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "mind"
    rng = np.random.default_rng(SEED)
    print(f"[{city}] costruisco descrittore (train/val/test)...", flush=True)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    nA = int(D0["attractors"].sum()); ceil = nA + 2
    K = select_K(vtr, ceil, rng); eps = select_eps(vtr, vva, K)
    print(f"[{city}] K*={K} (|A|={nA}, tetto={ceil})  ε*={eps}", flush=True)

    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)
    _, k_te, comp_te, isb_te = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(k_te, comp_te, isb_te, K)
    gamma = (1.0 / np.maximum(comp_te.sum(1), 1).astype(np.float32))
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); b_bar = b_z.mean(0)
    df = ds["df_test"]; u_rows = df["u_idx"].values.astype(np.int64); n_users = int(ds["n_users"])
    umac = ds["df_train"]["u_idx"].values.astype(np.int64)
    Pu = np.zeros((n_users, nmac)); np.add.at(Pu, (umac, cmt), 1.0)
    Pu[Pu.sum(1) == 0] = 1.0; Pu /= Pu.sum(1, keepdims=True)
    b_z_user = fit_situation_biases_z(umac, cmt, n_users, nmac, alpha=ALPHA)
    p_user_rows = Pu[u_rows]; print(f"[{city}] situazioni K={K} boundary={isb_te.mean():.1%}", flush=True)

    ev = {}
    ev["BASE"] = additive(sb, lambda b, e: None, df, gamma, icm, excl, p_user_rows, nmac)
    ev["UNI_mean"] = additive(sb, lambda b, e: np.broadcast_to(b_bar, (e - b, nmac)), df, gamma, icm, excl, p_user_rows, nmac)
    ev["SIT"] = additive(sb, lambda b, e: mem[b:e].astype(np.float32) @ b_z, df, gamma, icm, excl, p_user_rows, nmac)
    ev["Steck-b"] = additive(sb, lambda b, e: b_z_user[u_rows[b:e]], df, gamma, icm, excl, p_user_rows, nmac)
    print(f"[{city}] additive fatti; Steck-a greedy...", flush=True)
    ev["Steck-a"] = greedy_steck(sb, df, Pu, icm, excl, nmac, LAM)

    rows = []
    for name, (cm, cn, ht, jsv, u) in ev.items():
        rows.append({"city": city, "K": K, "eps": eps, "baseline": name,
                     "CatMRR": round(float(cm.mean()), 5), "CatNDCG": round(float(cn.mean()), 5),
                     "JS_user": round(float(jsv.mean()), 5), "R20": round(pu(ht, u), 5)})
    diffs = {opp: boot_diff(ev["SIT"][0], ev[opp][0], rng) for opp in ["Steck-b", "Steck-a", "UNI_mean", "BASE"]}
    hp = holm(np.array([v[3] for v in diffs.values()]))
    for r in rows:
        if r["baseline"] == "SIT":
            for (k, (m, lo, hi, p)), ph in zip(diffs.items(), hp):
                r[f"d_SIT-{k}"] = round(m, 5); r[f"CI_{k}"] = f"[{lo:+.5f},{hi:+.5f}]"; r[f"pholm_{k}"] = round(float(ph), 4)
    OUT = CLEAN / "outputs_results" / "baselines"; OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"{city}_calibration.csv", index=False)

    print(f"\n===== {city.upper()} — SIT vs baseline (K={K}, ε={eps}, κ=0.25) =====")
    print(pd.DataFrame(rows)[["baseline", "CatMRR", "CatNDCG", "JS_user", "R20"]].to_string(index=False))
    print("\nConfronti vs SIT (Δ CatMRR, Holm):")
    for k, (m, lo, hi, p) in diffs.items():
        ph = hp[list(diffs).index(k)]
        verdict = "SIT vince" if (lo > 0) else "SIT perde" if (hi < 0) else "pari"
        print(f"  SIT − {k:<9}: {m:+.5f} [{lo:+.5f},{hi:+.5f}] p_holm={ph:.4f}  → {verdict}")
    print(f"\n→ outputs_results/baselines/{city}_calibration.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
