"""[venv-xsage] A.6 — corpus stratificato per il lavoro LLM.

Dai record per richiesta di A.3, campiona per OGNI cella della griglia base un corpus bilanciato
su cinque strati:  win · neutral · harm · boundary (|T|>1) · core (|T|=1).

~100 per strato, seme fisso, lista degli id salvata: il campione deve essere LO STESSO per tutti e
tre i bracci LLM (A = vettore vero, B = sola cronologia, C = vettore corrotto).

TOLTI TUTTI I FILTRI CHERRY-PICK di scripts/yelp/reco_examples.py, che oggi seleziona 5 casi su
60.000 tenendo solo le vittorie:
  :75  `if bool(isbte[r]): continue`     -> teneva solo le richieste CORE (scartava il 24,2%)
  :77  `if nudge[r][tc] <= 0: continue`  -> teneva solo dove la situazione favorisce la cat. vera
  :81  `if rb < 4: continue`             -> teneva solo dove il backbone gia' sbagliava
  :84  `if rs > 12 or rs >= rb: continue`-> teneva solo i miglioramenti che finiscono in alto
  :86  selezione del massimo guadagno    -> teneva, per situazione, solo il caso migliore
Qui non si filtra: si stratifica e si campiona a caso dentro ogni strato.

NOTA SULLA SOVRAPPOSIZIONE DEGLI STRATI. {win, neutral, harm} e {boundary, core} sono DUE partizioni
della stessa popolazione, non cinque insiemi disgiunti: una richiesta puo' essere insieme `win` e
`boundary`. Si campiona 100 per strato in modo indipendente e si tiene l'UNIONE, con una colonna di
flag per strato, cosi' ogni strato ha i suoi ~100 e le sovrapposizioni sono contate e riportate.

Uso:  python scripts/explain/build_corpus.py [--per-stratum 100] [--seed 20260921]
Out:  outputs_results/explain/corpus/corpus_<ds>_<bk>_<seed>.parquet
      outputs_results/explain/corpus/corpus_summary.md
"""
import sys, argparse, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
REC = CLEAN / "outputs_results" / "explain" / "records"
OUT = CLEAN / "outputs_results" / "explain" / "corpus"
DATASETS = ["ml1m", "nyc_tist", "saopaulo"]
BACKBONES = ["B_blind", "EASE", "AFM", "SASRec"]
SEEDS = [42, 43, 44, 45, 46]
STRATA = ["win", "neutral", "harm", "boundary", "core"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    M = a.per_stratum
    summary = []
    short = []

    for ds in DATASETS:
        for bk in BACKBONES:
            for sd in SEEDS:
                p = REC / f"records_{ds}_{bk}_{sd}.parquet"
                if not p.exists():
                    print(f"[MANCA] {p.name}", flush=True); continue
                df = pd.read_parquet(p)
                masks = {"win": df.stratum.values == "win",
                         "neutral": df.stratum.values == "neutral",
                         "harm": df.stratum.values == "harm",
                         "boundary": df.T_size.values > 1,
                         "core": df.T_size.values == 1}
                # seme deterministico per cella: stesso campione a ogni rilancio, e per i 3 bracci
                # hash deterministico: hash() su stringhe e' randomizzato per processo (PYTHONHASHSEED),
                # quindi darebbe un campione diverso a ogni rilancio e fra i tre bracci.
                tag = f"{a.seed}|{ds}|{bk}|{sd}".encode()
                rng = np.random.default_rng(int(hashlib.sha256(tag).hexdigest()[:16], 16) % (2**32))
                picked, counts = {}, {}
                for st in STRATA:
                    idx = np.flatnonzero(masks[st])
                    counts[st] = len(idx)
                    take = min(M, len(idx))
                    picked[st] = rng.choice(idx, take, replace=False) if take else np.array([], int)
                    if take < M:
                        short.append(dict(dataset=ds, backbone=bk, seed=sd, stratum=st,
                                          disponibili=len(idx), richiesti=M, presi=take))
                union = np.unique(np.concatenate([picked[s] for s in STRATA])) if any(
                    len(picked[s]) for s in STRATA) else np.array([], int)
                c = df.iloc[union].copy()
                for st in STRATA:
                    c[f"in_{st}"] = np.isin(union, picked[st])
                c["corpus_seed"] = a.seed
                c.to_parquet(OUT / f"corpus_{ds}_{bk}_{sd}.parquet", compression="zstd", index=False)
                row = dict(dataset=ds, backbone=bk, seed=sd, n_union=len(union))
                for st in STRATA:
                    row[f"n_{st}"] = int(len(picked[st])); row[f"pop_{st}"] = counts[st]
                row["sovrapposti"] = int(sum(len(picked[s]) for s in STRATA) - len(union))
                summary.append(row)
                print(f"  {ds}/{bk}/{sd}: unione {len(union)} richieste, "
                      f"{ {s: len(picked[s]) for s in STRATA} }", flush=True)

    S = pd.DataFrame(summary); S.to_csv(OUT / "corpus_summary.csv", index=False)
    with open(OUT / "corpus_summary.md", "w") as f:
        f.write("# A.6 — corpus stratificato: numerosita' REALE per strato\n\n")
        f.write(f"Seme del campionamento: **{a.seed}** (deterministico per cella; identico per i tre bracci LLM).\n")
        f.write(f"Obiettivo: **{M} per strato**. Strati: {', '.join(STRATA)}.\n\n")
        f.write("`{win, neutral, harm}` e `{boundary, core}` sono due partizioni della stessa "
                "popolazione: la colonna `sovrapposti` conta le richieste scelte da piu' di uno strato.\n\n")
        f.write(S.to_markdown(index=False))
        f.write("\n\n## Strati sotto quota\n\n")
        if short:
            f.write("Riportati, **non riempiti** da un altro strato.\n\n")
            f.write(pd.DataFrame(short).to_markdown(index=False))
        else:
            f.write("Nessuno: tutti gli strati raggiungono la quota in tutte le celle.\n")
    print(f"\ncelle scritte: {len(S)} | strati sotto quota: {len(short)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
