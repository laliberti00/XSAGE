"""[venv-xsage] A.4 — equita' PER SITUAZIONE, sulla griglia base e sulla griglia piena.

Oggi la sezione equita' poggia su UN backbone e TRE dataset. Qui va su
griglia base  = 3 dataset x 4 backbone x 5 semi   (risultato principale)
griglia piena = 5 dataset x 7 backbone x 5 semi   (appendice)
per i tre metodi BASE / SIT / Steck-b.

I ranghi per richiesta sono gia' su disco in outputs_results/cache/raw_<ds>.npz per tutti e 7 i
backbone, 5 dataset, 5 semi, 3 metodi: non si ricalcolano. Manca solo l'assegnazione situazionale
per richiesta, che costa ~30 s per (dataset, seme) e si riusa sui 7 backbone.

DUE SCELTE ANNOTATE (regola 1 del brief: dove il brief non decide, si sceglie il conservativo)

 1. LE ETICHETTE DI SITUAZIONE NON SONO CONFRONTABILI FRA SEMI. Il clustering e' ri-stimato a ogni
    seme (results_record.py:230, fit_rough_kmeans(seed=seed)), quindi la situazione 3 del seme 42
    non e' la situazione 3 del seme 43: gli indici sono arbitrari. Di conseguenza
      - la tabella per situazione e' riportata PER SEME, senza mai mediare un indice fra semi;
      - il CI a due livelli (utenti x semi) e' applicato solo alle sintesi SENZA ETICHETTA
        (minimo rawlsiano, quartile basso, divario, media), che sono statistiche d'ordine e quindi
        confrontabili fra semi.
    Mediare per indice di situazione fra semi darebbe un numero senza significato. Non si fa.

 2. IL GATE DI IDENTITA' E' AMBIGUO NEL BRIEF e qui e' verificato in DUE forme, entrambe esatte:
      micro : sum_s n_s * media_s(cm) / N            == CatMRR aggregato
      macro : ricomposizione per (situazione, categoria) == macro-Cat-MRR aggregato
    La media micro-pesata delle macro-per-situazione NON puo' riprodurre la macro aggregata (la media
    non pesata sulle categorie non si decompone cosi'): e' un test vacuo e non viene eseguito (regola 8).

CORREZIONE DELLA SOGLIA DEI CI: wi0d_probe.py:114 scarta una cella se il MINIMO di utenti fra i semi
e' < 10. Qui si usa la MEDIA. Le celle che cambiano stato sono elencate nell'output.

Uso:  python scripts/fairness/situational_qos.py [--datasets ...] [--boot 600]
Out:  outputs_results/fairness/situational_qos_<ds>.csv
      outputs_results/fairness/rawlsian_<ds>.csv
      outputs_results/fairness/summary.md
      outputs_results/fairness/soglia_cambi_stato.csv
"""
import sys, os, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "explain"))
from persist_records import situational, _prefix, _exp_trunc, macro_msupp, KTOP, MSUPP

ALL_DS = ["ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand"]
BASE_DS = ["ml1m", "nyc_tist", "saopaulo"]
ALL_BK = ["B_blind", "B_full", "EASE", "DeepFM", "AFM", "FPMC", "SASRec"]
BASE_BK = ["B_blind", "EASE", "AFM", "SASRec"]
METHODS = ["BASE", "SIT", "Steck-b"]
SEEDS = [42, 43, 44, 45, 46]
OUT = CLEAN / "outputs_results" / "fairness"
MIN_USERS, MIN_REQS = 8, 20                 # soglia low_support del brief
CI_MIN_USERS = 10                           # soglia dei CI (wi0d_probe.py:114)

# terna pre-registrata del brief (regola 6). Vale per i tre dataset della griglia base.
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
GATE_TOL = 5e-5


def published_catmrr(city):
    """Valore pubblicato di CatMRR per (city, B_blind, SIT, seme 42) da results_record.csv.
    Per i tre dataset della griglia base deve coincidere con ANCHOR: verificato qui sotto, cosi'
    il gate copre anche yelp e kuairand, che non hanno una terna pre-registrata."""
    rr = pd.read_csv(CLEAN / "outputs_results" / "results_record.csv")
    m = rr[(rr.dataset == city) & (rr.backbone == "B_blind") &
           (rr.metric == "CatMRR") & (rr.method == "SIT")]
    if m.empty: return None
    v = float(m.mean_s42.iloc[0])
    if city in ANCHOR and abs(v - ANCHOR[city]) > GATE_TOL:
        raise SystemExit(f"results_record.csv non concorda con la terna pre-registrata su {city}: "
                         f"{v} vs {ANCHOR[city]}. STOP.")
    return v


def log(msg, f=None):
    print(msg, flush=True)
    if f: f.write(msg + "\n"); f.flush()


def per_situation(cm, tm, s, u, nmac, K):
    """macro/micro Cat-MRR per situazione + numerosita'. Nessuna media fra semi."""
    rows = []
    for k in range(K):
        m = s == k
        n_req = int(m.sum())
        if n_req == 0:
            rows.append(dict(situation=k, n_requests=0, n_users=0, macroCatMRR=np.nan,
                             microCatMRR=np.nan, n_cats_supported=0, low_support=True)); continue
        cmk, tmk = cm[m], tm[m]
        per = [cmk[tmk == c].mean() for c in range(nmac) if (tmk == c).sum() >= max(MSUPP, 1)]
        n_us = int(len(np.unique(u[m])))
        rows.append(dict(situation=k, n_requests=n_req, n_users=n_us,
                         macroCatMRR=float(np.mean(per)) if per else np.nan,
                         microCatMRR=float(cmk.mean()), n_cats_supported=len(per),
                         low_support=bool(n_us < MIN_USERS or n_req < MIN_REQS)))
    return pd.DataFrame(rows)


def user_matrices(cm, tm, s, u, nmac, K):
    """A_sum/A_cnt (n_utenti x K*nmac): per il bootstrap a cluster di UTENTI senza rifare i ranghi."""
    uq, uinv = np.unique(u, return_inverse=True)
    col = s.astype(np.int64) * nmac + tm.astype(np.int64)
    idx = uinv.astype(np.int64) * (K * nmac) + col
    A_sum = np.bincount(idx, weights=cm, minlength=len(uq) * K * nmac).reshape(len(uq), K * nmac)
    A_cnt = np.bincount(idx, minlength=len(uq) * K * nmac).reshape(len(uq), K * nmac)
    return uq, A_sum.astype(np.float64), A_cnt.astype(np.float64)


def stats_from(S, C, nmac, K):
    """Da somme/conteggi per (situazione,categoria) -> (min, quartile basso, divario, media) sulle
    situazioni con supporto. Statistiche d'ordine: nessuna etichetta, confrontabili fra semi."""
    S = S.reshape(-1, K, nmac); C = C.reshape(-1, K, nmac)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = np.where(C >= MSUPP, S / np.maximum(C, 1), np.nan)
    macro_s = np.nanmean(mean_c, axis=2)                       # (B, K)
    out = np.empty((len(macro_s), 4))
    for j, v in enumerate(macro_s):
        v = v[np.isfinite(v)]
        if len(v) == 0: out[j] = np.nan; continue
        mu = v.mean()
        out[j] = (v.min(), np.percentile(v, 25), v.min() - mu, mu)
    return out                                                  # min, q25, divario, media


def boot_two_level(per_seed, rng, B, use_mean_rule=True):
    """CI a due livelli (utenti x semi), pattern di wi0d_probe.py:113-122.
    per_seed = lista di (A_sum, A_cnt, nmac, K), una per seme.

    SOGLIA CORRETTA: scarta sulla MEDIA degli utenti fra i semi, non sul minimo.

    Vettorizzato: il ricampionamento degli utenti di un seme e' una matrice di pesi W (B x n_utenti)
    moltiplicata per A (n_utenti x K*nmac), cioe' UNA chiamata BLAS invece di B matvec. Il loop
    ingenuo su B x semi x (backbone,metodo) costava ~1e11 flop per dataset ed era ineseguibile.
    """
    ns = [len(a[0]) for a in per_seed]
    n_ref = float(np.mean(ns)) if use_mean_rule else float(min(ns))
    if n_ref < CI_MIN_USERS: return (np.nan,) * 8
    V = np.empty((len(per_seed), B, 4))
    for j, (A_s, A_c, nmac, K) in enumerate(per_seed):
        n = len(A_s)
        # i pesi del bootstrap a cluster sono esattamente Multinomial(n, uniforme su n utenti):
        # molto piu' veloce di np.add.at su B*n indici sparsi.
        W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
        V[j] = stats_from(W @ A_s, W @ A_c, nmac, K)
    # livello 2: ricampiona i SEMI; per ogni seme estratto si prende un replicato indipendente
    si = rng.integers(0, len(per_seed), size=(B, len(per_seed)))
    ri = rng.integers(0, B, size=(B, len(per_seed)))
    acc = np.nanmean(V[si, ri], axis=1)
    lo = np.nanpercentile(acc, 2.5, axis=0); hi = np.nanpercentile(acc, 97.5, axis=0)
    return tuple(np.concatenate([lo, hi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ALL_DS))
    ap.add_argument("--boot", type=int, default=600)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = open(OUT / "run_log.txt", "a")
    log(f"=== A.4 situational_qos — avvio {time.strftime('%Y-%m-%d %H:%M:%S')} — B={a.boot} ===", f)

    gate_rows, thr_rows, rawls_all = [], [], []
    for city in a.datasets.split(","):
        z = np.load(CLEAN / "outputs_results" / "cache" / f"raw_{city}.npz", allow_pickle=True)
        nI = int(z["_shared|nI"]); nmac = int(z["_shared|nmac"]); tm = z["_shared|tm"]; uu = z["_shared|u"]
        H, _ = _prefix(nI)
        bks = [b for b in ALL_BK if f"{b}|SIT|42|catrk" in z]
        rows = []; cellcache = {}
        for seed in SEEDS:
            S = situational(city, seed); K = S["K"]
            if len(S["u"]) != len(tm):
                log(f"  !! STOP {city} seme {seed}: n_test {len(S['u'])} != cache {len(tm)}", f); return 2
            s_lab = S["kte"].astype(np.int64)
            log(f"[{city} seme {seed}] K={K} eps={S['eps']} bfrac={S['isb'].mean():.5f} ({S['secs']}s)", f)
            for bk in bks:
                for mth in METHODS:
                    cm = _exp_trunc(z[f"{bk}|{mth}|{seed}|catrk"], z[f"{bk}|{mth}|{seed}|gc"], KTOP, H)
                    t = per_situation(cm, tm, s_lab, uu, nmac, K)
                    t.insert(0, "dataset", city); t.insert(1, "backbone", bk)
                    t.insert(2, "seed", seed); t.insert(3, "method", mth)
                    rows.append(t)
                    uq, A_s, A_c = user_matrices(cm, tm, s_lab, uu, nmac, K)
                    cellcache.setdefault((bk, mth), []).append((A_s, A_c, nmac, K))
                    # --- gate di identita', due forme, solo sulla cella di ancoraggio ---
                    if bk == "B_blind" and mth == "SIT" and seed == 42:
                        n_s = t.n_requests.values; mic = t.microCatMRR.values
                        micro_rec = float(np.nansum(n_s * np.where(np.isfinite(mic), mic, 0)) / n_s.sum())
                        SS = A_s.sum(0); CC = A_c.sum(0)
                        Sc = SS.reshape(K, nmac).sum(0); Cc = CC.reshape(K, nmac).sum(0)
                        per = [Sc[c] / Cc[c] for c in range(nmac) if Cc[c] >= max(MSUPP, 1)]
                        exp = published_catmrr(city)
                        if exp is None:
                            log(f"  [{city}] nessun CatMRR pubblicato: gate non applicabile", f)
                        else:
                            gate_rows.append(dict(dataset=city, preregistrato=city in ANCHOR,
                                                  micro_atteso=exp, micro_ricomposto=micro_rec,
                                                  macro_atteso=macro_msupp(cm, tm, nmac),
                                                  macro_ricomposto=float(np.mean(per))))
        df = pd.concat(rows, ignore_index=True)
        df.to_csv(OUT / f"situational_qos_{city}.csv", index=False)
        log(f"  -> situational_qos_{city}.csv  {len(df)} righe  "
            f"({df.low_support.sum()} celle low_support, mai cancellate)", f)

        # --- soglia CI: quali celle cambiano stato passando da min a media ---
        for (bk, mth), lst in cellcache.items():
            for k in range(lst[0][3]):
                nus = []
                for A_s, A_c, nm, K in lst:
                    col = slice(k * nm, (k + 1) * nm)
                    nus.append(int((A_c[:, col].sum(1) > 0).sum()))
                old = min(nus) >= CI_MIN_USERS; new = float(np.mean(nus)) >= CI_MIN_USERS
                if old != new:
                    thr_rows.append(dict(dataset=city, backbone=bk, method=mth, situation=k,
                                         n_users_per_seed=nus, min=min(nus), media=round(float(np.mean(nus)), 1),
                                         vecchia_regola="scartata" if not old else "tenuta",
                                         nuova_regola="scartata" if not new else "tenuta"))
        # --- sintesi rawlsiane senza etichetta, con CI a due livelli ---
        rng = np.random.default_rng(2024)
        for (bk, mth), lst in cellcache.items():
            pt = np.nanmean([stats_from(A_s.sum(0)[None], A_c.sum(0)[None], nm, K)[0]
                             for A_s, A_c, nm, K in lst], axis=0)
            ci = boot_two_level(lst, rng, a.boot)
            rawls_all.append(dict(dataset=city, backbone=bk, method=mth,
                                  min_rawlsiano=pt[0], quartile_basso=pt[1], divario=pt[2], media=pt[3],
                                  min_lo=ci[0], q25_lo=ci[1], divario_lo=ci[2], media_lo=ci[3],
                                  min_hi=ci[4], q25_hi=ci[5], divario_hi=ci[6], media_hi=ci[7]))
        pd.DataFrame([r for r in rawls_all if r["dataset"] == city]).to_csv(
            OUT / f"rawlsian_{city}.csv", index=False)
        log(f"  -> rawlsian_{city}.csv", f)

    # ---------------- gate + summary ----------------
    log("", f); log("=== GATE DI IDENTITA' (B_blind, SIT, seme 42) ===", f)
    ok = True
    for r in gate_rows:
        d1 = abs(r["micro_ricomposto"] - r["micro_atteso"]); d2 = abs(r["macro_ricomposto"] - r["macro_atteso"])
        g1, g2 = d1 < GATE_TOL, d2 < GATE_TOL; ok &= (g1 and g2)
        tag = "pre-registrato" if r["preregistrato"] else "da results_record"
        log(f"  {r['dataset']:10s} [{tag:17s}] micro {r['micro_atteso']:.5f} vs {r['micro_ricomposto']:.6f} "
            f"(diff {d1:.1e}) {'OK' if g1 else 'FALLITO'} | macro {r['macro_atteso']:.6f} vs "
            f"{r['macro_ricomposto']:.6f} (diff {d2:.1e}) {'OK' if g2 else 'FALLITO'}", f)
    log(f"  GATE: {'PASSATO' if ok else 'FALLITO — FERMARSI'}", f)

    if thr_rows:
        pd.DataFrame(thr_rows).to_csv(OUT / "soglia_cambi_stato.csv", index=False)
    log(f"  celle che cambiano stato con la soglia media (invece del minimo): {len(thr_rows)}", f)
    pd.DataFrame(rawls_all).to_csv(OUT / "rawlsian_all.csv", index=False)
    f.close()
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
