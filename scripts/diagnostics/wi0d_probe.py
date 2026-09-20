"""[venv-xsage] WI-0d Task 1 — L'IPOTESI BOUNDARY (ultimo probe diagnostico).

PRE-REGISTRATO nel brief WI-0d §1.4. Bucket, operazionalizzazioni e regola di decisione sono
fissati PRIMA dei numeri e non vengono rivisti dopo.

DICHIARAZIONE DI RISCHIO METODOLOGICO (brief §1.0, riportata anche nel summary):
questa e' la SECONDA ipotesi formulata dopo la falsificazione della prima (WI-0c). Se confermata
riscatterebbe la narrativa — ed e' esattamente il pattern che un council avversariale censura.
Per questo: regola scritta sopra ai numeri, AMBIGUO trattato come NON CONFERMATO, nessun terzo
tentativo previsto.

Ipotesi: l'entropia situazionale in una finestra di test di 3-9 richieste non misura un cambio
REALE di situazione ma l'INCERTEZZA DI ASSEGNAZIONE — utenti il cui descrittore sta vicino al
confine fra cluster, dove un'oscillazione minima fa saltare l'etichetta.

OPERAZIONALIZZAZIONI DICHIARATE (il brief da' criteri qualitativi; queste li rendono precisi):
 - boundary_share(u): frazione di richieste di test con assegnazione di confine. NESSUNA soglia
   arbitraria: il rough k-means produce `is_boundary` in modo strutturale (|T(v)|>1).
 - gate gamma: gamma_S = 1/|T|; |T| e' INTERO, quindi i livelli sono naturali (nessun binning).
 - "gradiente almeno altrettanto forte": |D(strato alto) - D(strato basso)| mediato sui quartili
   di lunghezza, confrontato fra stratificazione per boundary_share e per n_situazioni.
 - "BASE sostanzialmente piatto": BASE si muove meno della META' di quanto si muove Delta.

Uso:  python -m scripts.diagnostics.wi0d_probe [ml1m saopaulo nyc_tist]
Out:  outputs_results/probe_wi0d/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, str(CLEAN / "scripts" / "diagnostics"))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import ALPHA, build_descriptor, membership_from_assign          # noqa: E402
from eval_kappa import SEED, cat_mrr, select_K, select_eps                      # noqa: E402
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage.recommendation import fit_situation_biases_z                          # noqa: E402
from wi0c_probe import ols_cluster                                              # noqa: E402

OUT = CLEAN / "outputs_results" / "probe_wi0d"


def _jdef(o):
    """Encoder JSON robusto: numpy array -> lista, scalari numpy -> python. `default=float` da
    solo fallisce su array di lunghezza > 1 e fa perdere l'intero run alla scrittura."""
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    try: return float(o)
    except Exception: return str(o)


def _find_arrays(d, path=""):
    """Diagnostica: quali campi contengono array (la causa del crash precedente)."""
    bad = []
    if isinstance(d, dict):
        for k, v in d.items(): bad += _find_arrays(v, f"{path}.{k}" if path else str(k))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d[:3]): bad += _find_arrays(v, f"{path}[{i}]")
    elif isinstance(d, np.ndarray) and d.size > 1:
        bad.append(f"{path} (ndarray shape={d.shape})")
    return bad
SEEDS = [42, 43, 44, 45, 46]
BOOT = 600
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
BSHARE_BINS = [0.0, 1e-9, 0.25, 0.50, 1.01]          # [0] · (0,.25] · (.25,.5] · (.5,1]
BSHARE_LAB = ["bshare=0", "0<bs<=.25", ".25<bs<=.5", "bs>.5"]
DECISIVE, SUPPORT, UNDERPOWERED = "ml1m", "saopaulo", "nyc_tist"
RHO_MIN = 0.30                                        # soglia pre-registrata brief §1.4


def per_seed(city, seed, kap):
    rng = np.random.default_rng(seed)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
    cmt = D0["cmt"]; vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    _, k_te, comp, isb = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(k_te, comp, isb, K)
    ncomp = np.maximum(comp.sum(1), 1).astype(np.int64)      # |T(v)|, INTERO
    gam = (1.0 / ncomp).astype(np.float32)
    b_z = fit_situation_biases_z(fit.core_label.astype(np.int64), cmt, K, nmac, alpha=ALPHA)
    dft = ds["df_test"]
    q_sit = cat_mrr(sb, dft, mem.astype(np.float32) @ b_z, gam, icm, excl, kap)
    q_base = cat_mrr(sb, dft, None, gam, icm, excl, 0.0)
    return dict(K=K, eps=eps, k_te=k_te.astype(np.int64), isb=isb.astype(bool), ncomp=ncomp,
                q_sit=q_sit.astype(np.float64), q_base=q_base.astype(np.float64),
                u=dft["u_idx"].values.astype(np.int64), t=dft["time_local"].values)


def user_aggr(p, urank, nU):
    """quantita' per-utente: Delta, BASE, boundary_share, entropia, n_situazioni, lunghezza."""
    ur = urank[p["u"]]; K = p["K"]
    cnt = np.bincount(ur, minlength=nU)
    d = np.bincount(ur, weights=p["q_sit"] - p["q_base"], minlength=nU) / np.maximum(cnt, 1)
    b = np.bincount(ur, weights=p["q_base"], minlength=nU) / np.maximum(cnt, 1)
    bs = np.bincount(ur, weights=p["isb"].astype(float), minlength=nU) / np.maximum(cnt, 1)
    gm = np.bincount(ur, weights=p["ncomp"].astype(float), minlength=nU) / np.maximum(cnt, 1)
    ent = np.zeros(nU); nst = np.zeros(nU)
    df = pd.DataFrame({"u": ur, "k": p["k_te"]})
    for u_, g in df.groupby("u"):
        c = np.bincount(g["k"].values, minlength=K).astype(float); pr = c / c.sum(); nz = pr[pr > 0]
        ent[u_] = -(nz * np.log(nz)).sum() / np.log(K) if K > 1 else 0.0
        nst[u_] = int((c > 0).sum())
    return dict(delta=d, base=b, bshare=bs, meanT=gm, entropy=ent, nsit=nst, n=cnt)


def boot_cells(vals_per_seed, rng, B=BOOT):
    """bootstrap a due livelli: ricampiona UTENTI (cluster) e SEED. vals_per_seed = lista di array."""
    n = min(len(v) for v in vals_per_seed)
    if n < 10: return (np.nan, np.nan)
    out = np.empty(B)
    for r in range(B):
        si = rng.integers(0, len(vals_per_seed), len(vals_per_seed))
        out[r] = np.mean([vals_per_seed[j][rng.integers(0, len(vals_per_seed[j]),
                                                        len(vals_per_seed[j]))].mean() for j in si])
    return tuple(np.percentile(out, [2.5, 97.5]))


def run(city):
    bt = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")
    kmap = {int(r.seed): float(r.kstar) for _, r in
            bt[(bt.backbone == "B_blind") & (bt.method == "SIT")].iterrows()}
    P = []
    for s in SEEDS:
        print(f"[{city}] seed {s} ...", flush=True); P.append(per_seed(city, s, kmap[s]))
    # NB: nome distinto da `a`, che piu' sotto e' la variabile del ciclo `for a in A`.
    # Un `for` a livello di funzione SOVRASCRIVE la variabile (a differenza di una comprehension):
    # usare `a` per entrambi faceva finire in `anchor` il dizionario per-utente dell'ultimo seed.
    anchor_val = float(P[0]["q_sit"].mean()); ref = ANCHOR[city]
    ok = abs(anchor_val - ref) < 5e-6
    print(f"[{city}] GATE: {anchor_val:.5f} atteso {ref:.5f} -> {'OK' if ok else 'MISMATCH'}",
          flush=True)
    if not ok: return dict(dataset=city, gate_ok=False, anchor=anchor_val, anchor_ref=ref)

    uall = np.unique(np.concatenate([p["u"] for p in P]))
    urank = np.full(uall.max() + 1, -1, np.int64); urank[uall] = np.arange(len(uall))
    nU = len(uall); rng = np.random.default_rng(SEED)
    A = [user_aggr(p, urank, nU) for p in P]
    ok_u = A[0]["n"] > 0

    # --- 1.2.2 correlazioni (Spearman), mediate sui seed
    print(f"[{city}] correlazioni boundary_share ...", flush=True)
    rho_ent = np.mean([stats.spearmanr(a["bshare"][ok_u], a["entropy"][ok_u]).statistic for a in A])
    rho_nst = np.mean([stats.spearmanr(a["bshare"][ok_u], a["nsit"][ok_u]).statistic for a in A])
    rho_len = np.mean([stats.spearmanr(a["bshare"][ok_u], a["n"][ok_u]).statistic for a in A])
    C = [dict(dataset=city, coppia="boundary_share ~ entropia", spearman_rho=round(float(rho_ent), 4),
              n_utenti=int(ok_u.sum()), soglia_preregistrata=RHO_MIN,
              supera=int(abs(rho_ent) >= RHO_MIN)),
         dict(dataset=city, coppia="boundary_share ~ n_situazioni", spearman_rho=round(float(rho_nst), 4),
              n_utenti=int(ok_u.sum()), soglia_preregistrata=RHO_MIN, supera=int(abs(rho_nst) >= RHO_MIN)),
         dict(dataset=city, coppia="boundary_share ~ lunghezza_test", spearman_rho=round(float(rho_len), 4),
              n_utenti=int(ok_u.sum()), soglia_preregistrata=np.nan, supera=np.nan)]
    pd.DataFrame(C).to_csv(OUT / f"check_boundary_correlation_{city}.csv", index=False)

    # --- 1.2.3 E2 ri-stratificata per boundary_share, dentro quartili di lunghezza
    print(f"[{city}] E2 ri-stratificata ...", flush=True)
    ln = A[0]["n"].astype(float); qs = np.quantile(ln[ok_u], [.25, .5, .75])
    lbin = np.digitize(ln, qs)
    E = []
    for lb in range(4):
        for bi, lab in enumerate(BSHARE_LAB):
            vd, vb, nn = [], [], []
            for a in A:
                m = ok_u & (lbin == lb) & (np.digitize(a["bshare"], BSHARE_BINS) - 1 == bi)
                vd.append(a["delta"][m]); vb.append(a["base"][m]); nn.append(int(m.sum()))
            n0 = int(np.mean(nn))
            lo, hi = boot_cells(vd, rng) if n0 >= 10 else (np.nan, np.nan)
            E.append(dict(dataset=city, len_bin=f"Q{lb+1}", strato=lab, n_utenti=n0,
                          delta=round(float(np.mean([v.mean() for v in vd if v.size])), 5) if n0 else np.nan,
                          base=round(float(np.mean([v.mean() for v in vb if v.size])), 5) if n0 else np.nan,
                          ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5),
                          low_support=int(n0 < 20)))
    # stessa cosa per n_situazioni, per confrontare i due gradienti
    for lb in range(4):
        for st, lab in ((1, "nsit=1"), (2, "nsit=2"), (3, "nsit>=3")):
            vd, nn = [], []
            for a in A:
                s = np.where(a["nsit"] <= 1, 1, np.where(a["nsit"] == 2, 2, 3))
                m = ok_u & (lbin == lb) & (s == st)
                vd.append(a["delta"][m]); nn.append(int(m.sum()))
            n0 = int(np.mean(nn))
            E.append(dict(dataset=city, len_bin=f"Q{lb+1}", strato=lab, n_utenti=n0,
                          delta=round(float(np.mean([v.mean() for v in vd if v.size])), 5) if n0 else np.nan,
                          base=np.nan, ci_lo=np.nan, ci_hi=np.nan, low_support=int(n0 < 20)))
    dfE = pd.DataFrame(E); dfE.to_csv(OUT / f"check_e2_boundary_{city}.csv", index=False)

    def gradient(labels):
        g = []
        for lb in range(4):
            s = dfE[(dfE.len_bin == f"Q{lb+1}") & (dfE.strato.isin(labels)) & (dfE.n_utenti >= 20)]
            if len(s) >= 2: g.append(abs(s.delta.max() - s.delta.min()))
        return float(np.mean(g)) if g else np.nan
    grad_b = gradient(BSHARE_LAB); grad_n = gradient(["nsit=1", "nsit=2", "nsit>=3"])

    # --- 1.2.5 gradiente del gate gamma: |T| e' intero -> livelli naturali
    print(f"[{city}] gradiente gamma ...", flush=True)
    G = []
    lv = [1, 2, 3]
    for L in lv + ["4+"]:
        vd, vb, nn = [], [], []
        for p in P:
            m = (p["ncomp"] == L) if L != "4+" else (p["ncomp"] >= 4)
            vd.append((p["q_sit"] - p["q_base"])[m]); vb.append(p["q_base"][m]); nn.append(int(m.sum()))
        n0 = int(np.mean(nn))
        lo, hi = boot_cells(vd, rng) if n0 >= 10 else (np.nan, np.nan)
        G.append(dict(dataset=city, T_size=str(L), gamma=round(1.0 / (4 if L == "4+" else L), 4),
                      n_richieste=n0,
                      delta=round(float(np.mean([v.mean() for v in vd if v.size])), 5) if n0 else np.nan,
                      base=round(float(np.mean([v.mean() for v in vb if v.size])), 5) if n0 else np.nan,
                      ci_lo=round(float(lo), 5), ci_hi=round(float(hi), 5)))
    dfG = pd.DataFrame(G); dfG.to_csv(OUT / f"check_gamma_gradient_{city}.csv", index=False)

    # --- 1.2.4 regressione
    rows = []
    for si, a in enumerate(A):
        m = ok_u
        rows.append(pd.DataFrame({"u": np.arange(nU)[m], "delta": a["delta"][m],
                                  "bshare": a["bshare"][m], "entropy": a["entropy"][m],
                                  "loglen": np.log(np.maximum(a["n"][m], 1))}))
    D = pd.concat(rows, ignore_index=True)
    X = np.c_[np.ones(len(D)), D["bshare"], D["entropy"], D["loglen"]]
    beta, se, r2 = ols_cluster(X, D["delta"].values, D["u"].values)
    names = ["intercept", "beta_bshare", "beta_entropy", "beta_loglen"]
    R = [dict(dataset=city, term=n, coef=round(float(b), 6), se_cluster=round(float(s), 6),
              t=round(float(b / s), 3) if s > 0 else np.nan, sig=int(abs(b / s) > 1.96) if s > 0 else 0,
              n_obs=len(D), n_clusters=int(D.u.nunique()), r2=round(float(r2), 5))
         for n, b, s in zip(names, beta, se)]
    corr = D[["bshare", "entropy", "loglen"]].corr()
    for x in corr.columns:
        for y in corr.columns:
            if x < y: R.append(dict(dataset=city, term=f"corr({x},{y})",
                                    coef=round(float(corr.loc[x, y]), 4), se_cluster=np.nan,
                                    t=np.nan, sig=np.nan, n_obs=len(D),
                                    n_clusters=int(D.u.nunique()), r2=np.nan))
    pd.DataFrame(R).to_csv(OUT / f"check_regression_{city}.csv", index=False)

    # --- guardia di falsificazione + regola pre-registrata
    hi_lab = ".25<bs<=.5"; lo_lab = "bshare=0"
    sub = dfE[dfE.strato.isin([hi_lab, lo_lab, "bs>.5"]) & (dfE.n_utenti >= 20)]
    dr = float(sub.delta.max() - sub.delta.min()) if len(sub) >= 2 else np.nan
    br = float(sub.base.max() - sub.base.min()) if len(sub) >= 2 else np.nan
    guard = bool(np.isfinite(dr) and np.isfinite(br) and abs(br) < abs(dr) / 2)
    hi_unc = dfG[dfG.T_size.isin(["2", "3", "4+"])]
    neg_hi = bool(len(hi_unc) and (hi_unc.ci_hi < 0).any())
    c1 = abs(rho_ent) >= RHO_MIN
    c2 = bool(np.isfinite(grad_b) and np.isfinite(grad_n) and grad_b >= grad_n)
    v = "CONFERMATO" if (c1 and c2 and guard and neg_hi) else \
        ("AMBIGUO" if sum([c1, c2, guard, neg_hi]) >= 2 else "NON CONFERMATO")
    s = dict(dataset=city, gate_ok=True, anchor=anchor_val, anchor_ref=ref, n_users=nU,
             rho_ent=float(rho_ent), rho_nst=float(rho_nst), rho_len=float(rho_len),
             grad_bshare=grad_b, grad_nsit=grad_n, delta_range=dr, base_range=br,
             guard=guard, neg_high_unc=neg_hi, c1=bool(c1), c2=c2, verdetto=v,
             gamma=dfG.to_dict("records"), reg=R, e2=dfE.to_dict("records"))
    bad = _find_arrays(s)
    if bad: print(f"[{city}] NB campi array-valued: {bad}", flush=True)
    json.dump(s, open(OUT / f"_wi0d_{city}.json", "w"), indent=2, default=_jdef)
    print(f"[{city}] verdetto={v}", flush=True)
    return s


def summary():
    S = [json.load(open(p)) for p in sorted(OUT.glob("_wi0d_*.json"))]
    if not S: return
    L = ["# WI-0d — Audit di nomenclatura + ipotesi boundary", "",
         "## ⚠️ Dichiarazione di rischio metodologico (brief §1.0)", "",
         "L'ipotesi boundary e' la **seconda** formulata dopo la falsificazione della prima "
         "(WI-0c, esito NON CONFERMATO). Se confermata riscatterebbe la narrativa: e' esattamente "
         "il pattern che un council avversariale censura. Per questo la regola di decisione e' "
         "scritta **sopra** ai numeri, **AMBIGUO e' trattato come NON CONFERMATO**, e non e' "
         "previsto un terzo tentativo. Dopo WI-0d la fase diagnostica e' chiusa.", "",
         "## Task 0.a — Audit di nomenclatura dei backbone", "",
         "**Esito: 7/7 corrispondono. Il manoscritto e' corretto.** Il nome nel paper combacia "
         "con l'algoritmo reale per tutti i backbone (verifica per valore: ML-1M macro-Cat-MRR "
         "del manoscritto vs `results_record.csv`). L'unica anomalia e' **interna al repo**: il "
         "file `FM.scores.npy` contiene punteggi **BPR** (`cornac_backbone.py:26`), un nome "
         "ereditato; il paper lo chiama correttamente BPR e chiama FM il ContextAwareFM "
         "(`B_full`). Dettaglio in `task0a_backbone_naming.csv`. Nei capitoli di tesi non "
         "esistono tabelle di risultati con nomi di backbone: nessun rischio.", "",
         "## Task 0.b — Quale repo e' quello vero", "",
         "**Le due cartelle sono storie git NON collegate.**", "",
         "| | xsage-clean | X-SAGE |", "|---|---|---|",
         "| HEAD | `db608f8` | `5724fe6` |",
         "| remote | laliberti00/XSAGE | **knowmis/X-SAGE — l'URL citato nel paper** |",
         "| contiene `db608f8` | sì | **NO** |", "| file tracciati | 303 | 63 |", "",
         "Solo **11 file in comune**, di cui **9 differiscono nel contenuto** (inclusi "
         "`xsage/recommendation.py`, `l0_sensing.py`, `l1_perception.py`, `l3_projection.py`, "
         "`data.py`). `X-SAGE` e' un **repackaging deliberato**: ha backbone propri "
         "(`xsage/backbones/*.py`, assenti in xsage-clean), una pipeline propria "
         "(`scripts/run_pipeline.py`) e `data/processed/` committati. Il suo `recommendation.py` "
         "documenta il combiner **senza** lambda e senza `m_sel` — piu' aderente al paper V2 di "
         "quanto lo sia xsage-clean. **Nessuno ha pero' mai verificato che riproduca i numeri**: "
         "il gate 0.38479 non e' mai stato girato su quel codice. Dettaglio in "
         "`task0b_repo_diff.csv`.", "",
         "## Task 1 — Ipotesi boundary", "",
         "Backbone `B_blind` (stessa dichiarazione del Task 0.a), kappa* congelato, 5 seed, "
         "bootstrap a due livelli su utenti e seed. Priorita' dichiarata prima dei numeri: "
         f"decisivo **{DECISIVE}**, conferma **{SUPPORT}**, **{UNDERPOWERED}** non vota.", "",
         "### Gate di ancoraggio", "", "| dataset | Cat-MRR | atteso | esito |", "|---|---|---|---|"]
    for s in S:
        L.append(f"| {s['dataset']} | {s['anchor']:.5f} | {s['anchor_ref']:.5f} | "
                 f"{'OK' if s['gate_ok'] else 'MISMATCH'} |")
    G = [s for s in S if s["gate_ok"]]
    L += ["", "### Correlazioni (Spearman, mediate sui 5 seed)", "",
          f"Soglia pre-registrata: |rho| >= {RHO_MIN}", "",
          "| dataset | bshare~entropia | bshare~n_situazioni | bshare~lunghezza | supera soglia |",
          "|---|---|---|---|---|"]
    for s in G:
        L.append(f"| {s['dataset']} | {s['rho_ent']:+.4f} | {s['rho_nst']:+.4f} | "
                 f"{s['rho_len']:+.4f} | {'sì' if s['c1'] else 'NO'} |")
    L += ["", "### Gradiente del gate gamma (|T| intero: livelli naturali, nessun binning)", "",
          "Guardia di falsificazione: BASE riportato accanto a Delta.", ""]
    for s in G:
        L += [f"**{s['dataset']}**", "",
              "| \\|T\\| | gamma | n richieste | Δ | CI 95% | BASE |", "|---|---|---|---|---|---|"]
        for g in s["gamma"]:
            L.append(f"| {g['T_size']} | {g['gamma']:.4f} | {g['n_richieste']} | "
                     f"{g['delta']:+.5f} | [{g['ci_lo']:+.5f}, {g['ci_hi']:+.5f}] | "
                     f"{g['base']:.5f} |")
        L.append("")
    L += ["### Regressione Δ(u) ~ boundary_share + entropia + log(lunghezza)", "",
          "| dataset | β bshare | β entropia | β loglen | R² |", "|---|---|---|---|---|"]
    for s in G:
        r = {x["term"]: x for x in s["reg"]}
        f = lambda k: f"{r[k]['coef']:+.5f} ({r[k]['se_cluster']:.5f}){'*' if r[k]['sig'] else ''}"
        L.append(f"| {s['dataset']} | {f('beta_bshare')} | {f('beta_entropy')} | "
                 f"{f('beta_loglen')} | {r['beta_bshare']['r2']:.5f} |")
    L += ["", "Correlazioni fra regressori:", ""]
    for s in G:
        r = {x["term"]: x for x in s["reg"]}
        L.append(f"- **{s['dataset']}**: " + " · ".join(
            f"{k[5:-1]} {r[k]['coef']:+.3f}" for k in r if k.startswith("corr(")))
    L += ["", "### Esito (regola pre-registrata §1.4, applicata meccanicamente)", "",
          "Servono tutte e quattro: |rho| >= 0.30 · gradiente boundary >= gradiente n_situazioni · "
          "guardia superata (BASE si muove meno della meta' di Delta) · Delta significativamente "
          "negativo ad alta incertezza.", ""]
    for s in G:
        tag = ("decisivo" if s["dataset"] == DECISIVE else "conferma" if s["dataset"] == SUPPORT
               else "NON VOTA")
        L.append(f"- **{s['dataset']}** ({tag}) → **{s['verdetto']}** — rho: "
                 f"{'sì' if s['c1'] else 'no'} ({s['rho_ent']:+.3f}) · gradiente: "
                 f"{'sì' if s['c2'] else 'no'} ({s['grad_bshare']:.5f} vs {s['grad_nsit']:.5f}) · "
                 f"guardia: {'superata' if s['guard'] else 'VIOLATA'} "
                 f"(BASE varia {s['base_range']:.5f}, Δ varia {s['delta_range']:.5f}) · "
                 f"Δ<0 ad alta incertezza: {'sì' if s['neg_high_unc'] else 'no'}")
    dec = [s for s in G if s["dataset"] == DECISIVE]
    if dec:
        v = dec[0]["verdetto"]
        eff = "NON CONFERMATO" if v == "AMBIGUO" else v
        L += ["", f"### esito = {eff}", "",
              (f"(verdetto grezzo su {DECISIVE}: **{v}**; per la regola §1.0 AMBIGUO e' trattato "
               f"come NON CONFERMATO)" if v == "AMBIGUO" else
               f"Determinato su **{DECISIVE}**, con **{SUPPORT}** a conferma."), "",
              ("→ Il limite si scrive nella forma gia' stabilita da WI-0c: il beneficio si "
               "concentra sugli utenti a situazione stabile. Mezza pagina nei limiti di ogni "
               "capitolo. **Fase diagnostica chiusa: dopo WI-0d non si apre nient'altro.**"
               if eff != "CONFERMATO" else
               "→ Il limite si riscrive come contributo (brief §1.4). Il gate piu' severo resta "
               "future work, NON si implementa.")]
    L += ["", "## Domande emerse (riportate, non eseguite)", "",
          "1. **`knowmis/X-SAGE` non e' mai stato verificato numericamente.** E' il repo che "
          "diventera' pubblico ed e' l'URL citato nel manoscritto, ma non contiene `db608f8` e i "
          "suoi moduli core differiscono. Prima di renderlo pubblico andrebbe girato il gate "
          "(0.38479 / 0.34162 / 0.40045) su quel codice.",
          "2. **`FM.scores.npy` contiene BPR.** Il paper e' corretto, ma il nome del file e' una "
          "trappola per chiunque legga il repo (incluso un referee che scarichi il codice)."]
    (OUT / "wi0d_summary.md").write_text("\n".join(L) + "\n")
    print(f"\n-> {OUT / 'wi0d_summary.md'}", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for c in (sys.argv[1:] or [DECISIVE]):
        print(f"\n===== WI-0d · {c.upper()} =====", flush=True); run(c)
    summary(); return 0


if __name__ == "__main__":
    sys.exit(main())
