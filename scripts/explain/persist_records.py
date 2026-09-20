"""[venv-xsage] A.3 — persiste il record PER RICHIESTA sulla griglia base.

Griglia base: 3 dataset (ml1m, nyc_tist, saopaulo) x 4 backbone (B_blind, EASE, AFM, SASRec)
x 5 semi (42-46) = 60 celle.

Sostituisce la selezione a cascata di scripts/yelp/reco_examples.py, che calcola queste
stesse quantita' per ogni richiesta e ne salva CINQUE, scelte con quattro filtri che tengono
solo le vittorie. Qui si salva tutto, senza filtri.

IL PUNTO TECNICO. Il nudge persistito e' `mem @ b_z` (boundary-aware), non `b_z[situazione_core]`.
Sulle richieste core i due coincidono (mem e' one-hot); sulle boundary no, e sono il 24,2% su ml1m.
Entrambi sono persistiti in colonne separate, cosi' la differenza e' misurabile.

DUE RIUSI che tolgono ore di calcolo:
 1. L'assegnazione situazionale NON dipende dal backbone: build_descriptor/select_K/select_eps/
    fit_rough_kmeans/_assign/b_z dipendono solo da (dataset, seme). Si calcolano una volta per
    (dataset, seme) e si riusano sui 4 backbone.
 2. I ranghi per richiesta sono GIA' su disco in outputs_results/cache/raw_<ds>.npz, prodotti da
    scripts/yelp/results_record.py: chiavi "<backbone>|<metodo>|<seme>|{rk,catrk,g,gc}".
    Non si ricalcolano: si leggono. Il gate di ancoraggio verifica che siano quelli giusti.

Scrittura incrementale: una cella gia' scritta viene saltata. Rilanciabile senza rifare nulla.

Uso:  python scripts/explain/persist_records.py [--datasets ml1m,nyc_tist,saopaulo] [--force]
Out:  outputs_results/explain/records/records_<ds>_<bk>_<seed>.parquet
      outputs_results/explain/records/persist_log.txt
"""
import sys, os, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

DATASETS = ["ml1m", "nyc_tist", "saopaulo"]
BACKBONES = ["B_blind", "EASE", "AFM", "SASRec"]
SEEDS = [42, 43, 44, 45, 46]
OUT = CLEAN / "outputs_results" / "explain" / "records"
LOG = OUT / "persist_log.txt"
KTOP, MSUPP = 20, 20                                  # da results_record.py:28-29

# ancoraggio: CatMRR, B_blind, SIT, seme 42 (results_record.csv, colonna mean_s42)
ANCHOR = {"ml1m": 0.38479, "nyc_tist": 0.34162, "saopaulo": 0.40045}
ANCHOR_TOL = 5e-5                                     # mezza unita' sul 4o decimale: i valori su disco
                                                      # sono gia' arrotondati a 5 decimali


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def _prefix(nI):
    """Somme prefisse per il rango ATTESO (McSherry-Najork). Copia di results_record.py:61-64."""
    p = np.arange(1, nI + 2)
    return np.concatenate([[0.], np.cumsum(1. / p)]), np.concatenate([[0.], np.cumsum(1. / np.log2(p + 1.))])


def _exp_trunc(rk, g, K, Hpre):
    """Copia di results_record.py:67-70."""
    b = rk - 1; L = len(Hpre) - 1
    return np.where(b < K, (Hpre[np.clip(np.minimum(b + g, K), 0, L)] - Hpre[np.clip(b, 0, L)]) / np.maximum(g, 1), 0.)


def macro_msupp(cm, tm, nmac, m=MSUPP):
    """Copia di results_record.py:145-148."""
    per = [cm[tm == c].mean() for c in range(nmac) if (tm == c).sum() >= max(m, 1)]
    return float(np.mean(per)) if per else 0.


def situational(city, seed):
    """Assegnazione situazionale per (dataset, seme). Replica ESATTAMENTE results_record.py:225-233:
    rng = default_rng(seed) per select_K, e seed=seed (non il SEED globale) per fit_rough_kmeans."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; cmt = D0["cmt"]
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
    mem = membership_from_assign(kte.astype(np.int64), compte, isbte, K)
    nudge = (mem.astype(np.float32) @ b_z)            # <<< boundary-aware: la forma di produzione
    nudge_core = b_z[kte]                             # <<< core-only: la forma di reco_examples.py:70
    dft = ds["df_test"]
    return dict(K=int(K), eps=float(eps), nmac=int(nmac), n_items=int(ds["n_items"]),
                u=dft["u_idx"].values.astype(np.int64), i=dft["i_idx"].values.astype(np.int64),
                kte=kte.astype(np.int16), isb=isbte.astype(bool), gam=gam.astype(np.float32),
                Tsize=np.asarray(compte.sum(1)).ravel().astype(np.int16),
                mem=mem.astype(np.float32), nudge=nudge.astype(np.float32),
                nudge_core=nudge_core.astype(np.float32),
                true_cat=icm[dft["i_idx"].values.astype(np.int64)].astype(np.int16),
                secs=round(time.perf_counter() - t0, 2))


def stratum(rb, rs):
    """win/neutral/harm sul rango di CATEGORIA: win = X-SAGE la fa risalire."""
    s = np.full(len(rb), "neutral", dtype=object)
    s[rs < rb] = "win"; s[rs > rb] = "harm"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== A.3 persist_records — griglia base — avvio ===")
    gate_rows = []; written = 0; skipped = 0

    for city in a.datasets.split(","):
        z = np.load(CLEAN / "outputs_results" / "cache" / f"raw_{city}.npz", allow_pickle=True)
        nI = int(z["_shared|nI"]); nmac_c = int(z["_shared|nmac"]); tm_c = z["_shared|tm"]
        H, _ = _prefix(nI)
        for seed in SEEDS:
            need = [b for b in BACKBONES
                    if a.force or not (OUT / f"records_{city}_{b}_{seed}.parquet").exists()]
            if not need:
                skipped += len(BACKBONES); log(f"[{city} seed {seed}] tutte le celle gia' presenti, salto"); continue
            S = situational(city, seed)
            log(f"[{city} seed {seed}] K={S['K']} eps={S['eps']} bfrac={S['isb'].mean():.5f} "
                f"n_test={len(S['u'])} — {S['secs']}s")
            if len(S["u"]) != len(tm_c):
                log(f"  !! STOP: n_test ricalcolato {len(S['u'])} != cache {len(tm_c)}"); return 2
            for bk in need:
                rb_c = z[f"{bk}|BASE|{seed}|catrk"]; rs_c = z[f"{bk}|SIT|{seed}|catrk"]
                df = pd.DataFrame(dict(
                    dataset=city, backbone=bk, seed=seed,
                    request_id=np.arange(len(S["u"]), dtype=np.int32),
                    user_id=S["u"].astype(np.int32),
                    situation_core=S["kte"], T_size=S["Tsize"], is_boundary=S["isb"], gamma=S["gam"],
                    membership=list(S["mem"]),
                    nudge_vector=list(S["nudge"]),               # mem @ b_z  (da usare)
                    nudge_core_only=list(S["nudge_core"]),       # b_z[k]     (per il confronto)
                    nudge_maxabs_diff=np.abs(S["nudge"] - S["nudge_core"]).max(1).astype(np.float32),
                    true_category=S["true_cat"],
                    rank_base=rb_c.astype(np.int32), rank_sit=rs_c.astype(np.int32),
                    rank_base_item=z[f"{bk}|BASE|{seed}|rk"].astype(np.int32),
                    rank_sit_item=z[f"{bk}|SIT|{seed}|rk"].astype(np.int32),
                    stratum=stratum(rb_c, rs_c),
                ))
                p = OUT / f"records_{city}_{bk}_{seed}.parquet"
                df.to_parquet(p, compression="zstd", index=False)
                written += 1
                # gate di ancoraggio: solo la cella che lo definisce
                if bk == "B_blind" and seed == 42:
                    cm = _exp_trunc(rs_c, z[f"{bk}|SIT|{seed}|gc"], KTOP, H)
                    gate_rows.append(dict(dataset=city, atteso=ANCHOR[city], ricalcolato=float(cm.mean()),
                                          macro=macro_msupp(cm, tm_c, nmac_c)))
                log(f"  scritto {p.name}  {len(df)} righe  {os.path.getsize(p)/1e6:.2f} MB  "
                    f"strati={dict(pd.Series(df.stratum).value_counts())}")

    log("")
    log("=== GATE DI ANCORAGGIO (CatMRR, B_blind, SIT, seme 42) ===")
    ok = True
    for r in gate_rows:
        d = abs(r["ricalcolato"] - r["atteso"]); good = d < ANCHOR_TOL; ok &= good
        log(f"  {r['dataset']:10s} atteso {r['atteso']:.5f}  ricalcolato {r['ricalcolato']:.6f}  "
            f"diff {d:.2e}  {'OK' if good else 'FALLITO'}   (macro-Cat-MRR {r['macro']:.5f})")
    log(f"  GATE: {'PASSATO' if ok else 'FALLITO — FERMARSI'}")
    log(f"=== celle scritte {written}, saltate {skipped} ===")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
