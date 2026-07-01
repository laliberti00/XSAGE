# X-SAGE — Freeze record (lock-down delle decisioni prima del re-run)

> **Cos'è e cosa NON è.** Questo è un **freeze record**: congela le decisioni per un re-run pulito e
> riproducibile, così i numeri finali sono coerenti e non si ritoccano in silenzio. **NON è la
> pre-registrazione di una legge.** Abbiamo deliberatamente **tagliato** la "legge a gate": niente
> LODO, niente dataset held-out predittivo, niente soglie a-priori predittive. Le caselle
> winner/ridondante/null sono **caratterizzazione descrittiva** (osservata e spiegata), non predizioni.
>
> **Natura onesta — freeze retrospettivo.** Le regole qui sotto sono **informate dai dati già visti**
> (ml1m winner, POI ridondanti, ecc.). Non è un freeze "prima di guardare". L'unico valore di
> *replica* sta nei dataset d'**appendice** (città TIST/TSMC non usate per le scelte). Va scritto così
> nel paper: una caratterizzazione empirica onesta, non una legge validata out-of-sample.

Da committare con questo `METRICS_FORMULAS.md`; annotare l'hash di freeze in §7.

---

## 1. Dataset (k-core 10 uniforme, split temporale, leave-one-out)
- **PRIMARI (7)** — headline + famiglie Holm: `ml1m` (winner-candidate), `nyc_tist`, `saopaulo`,
  `amazoncd` (ridondanti-candidate), `mind`, `yelp`, `kuairand` (null-candidate). Tutti ≥4K utenti;
  una sola coppia POI (nyc/sao, città diverse).
- **APPENDICE (5)** — robustezza/replica, fuori dai claim di testa: `istanbul`, `bangkok`,
  `tokyo_tist` (replica POI), `tsmc_nyc`, `tsmc_tky` (<4K, replica NYC/Tokyo).

> **Stato ESEGUITO (deviation):** eseguiti i **7 primari** — ml1m (winner, 7bk), nyc_tist + saopaulo
> (ridondanti POI, 7bk), amazoncd (ridondante non-POI, 2bk), mind + yelp + kuairand (null, 2bk). L'**appendice
> è rimandata** (bassa informatività: replica una casella POI-ridondante già doppia). Il set primario copre
> le 3 caselle su domini diversi (film / POI / retail / news / business / video). Dettaglio in `PROCESSO_SPERIMENTALE.md`.
- k-core = **10** su utenti e item, **uniforme** (verificato: tutti min_u/min_i ≥10; Amazon = 23 059
  utenti = versione k10, non il falso-winner k20 a 716). Conteggi post-filtro loggati nel run.

## 2. Backbone (7) e metodi (3)
- `B_blind`(BPR), **`B_full`(FM) = FOCALE, dichiarato A PRIORI** (coerente con tutta la tesi),
  `EASE`, `DeepFM`, `AFM`, `FPMC`, `SASRec`.
- Tuning iperparametri **solo su validation**; κ (X-SAGE) e K (cluster) **solo su validation**.
- Metodi: **BASE** (backbone), **SIT** (X-SAGE), **Steck-b** (calibrazione statica per-utente sul prior
  storico di categoria — verificato: cala la JS-utente sotto SIT su tutti i backbone).

## 3. Protocollo metriche
- **Full-ranking**, niente negative sampling (Krichene & Rendle 2020); item visti → −∞.
- **Tie-break = rango ATTESO** (mid-rank, McSherry & Najork 2008) come **primario** (stretto ri-derivabile
  dalla cache via g/gc). Misurato: invariante sulle @K, ~nullo anche su MRR pieno.
- **Metrica primaria del contributo:** **macro-Cat-MRR@20 per-richiesta**. Secondarie: Cat-MRR/Cat-NDCG@20,
  HR@20/NDCG@20/MRR (per-richiesta + `_u`), LT@20/Gini/Coverage, JS (costo dichiarato).
- Accuratezza **per-richiesta primaria**, `_u` user-balanced secondaria (lo scarto ~22-36% è dichiarato).
- Formule fedeli in `METRICS_FORMULAS.md` (classificazione standard/variante/nostra).

## 4. Tassonomia (descrittiva, applicata meccanicamente — NON una legge)
- **L1** — il segnale esiste? `SIT > BASE` su macro-Cat-MRR@20 per-richiesta, Δ significativo.
- **L2** — batte la personalizzazione statica? `SIT > Steck-b`, Δ significativo.
- **Casella:** L1-FAIL → *null*; L1-PASS ∧ L2-FAIL → *ridondante*; L1-PASS ∧ L2-PASS → *winner*.
- Le caselle si **descrivono e spiegano** (profondità/saturazione = spiegazione *post-hoc*, **non** gate
  predittivi); non si predicono out-of-sample.

## 5. macro-Cat-MRR — min-support (anti-rumore categorie rare)
- **MSUPP = 20** (a-priori): la macro media solo categorie con `|R_c| ≥ 20`.
- **Stability check obbligatorio:** ri-derivare con `MSUPP ∈ {20, 50}` dalla cache (secondi) e confermare
  che le caselle non si muovono. **`|R_c|` riportato comunque.**
- **GATE WINNER ml1m (esplicito):** poiché il min-support morde dove le categorie sono molte (ml1m=18),
  **"ml1m = winner" si scrive SOLO se ml1m L2 (SIT−Steck-b su macro-Cat-MRR) ha Δ > 0 e 5/5 seed concordi
  sotto MSUPP=20** (e stabile a 50). Se è 4/5, claim morbido ("direzionalmente consistente"), non winner.

## 6. Statistica
- **5 seed {42,43,44,45,46}.** Nessun claim a singolo seed.
- **Due fonti di rumore, due strumenti:** bootstrap per-richiesta (B=1500, ricampiona l'unità coerente —
  utenti per accuratezza-utente, richieste per categoriali/esposizione) = **potenza**; 5 seed = **consistenza**.
- **Significatività PRIMARIA = bootstrap-Δ** (`p_lX`) → Holm. **Robustezza = t cross-seed** (`p_lX_seed`, df=4),
  riportata ma NON gate. **Consistenza = `seeds_lX` (#seed con Δ>0)**.
- **Regola UNICA delle caselle** (winner/ridondante/nullo, su macro-Cat-MRR, per backbone):
  `Lk_pass = Δ_k>0 ∧ CI-bootstrap-esclude-0 ∧ seeds_k==5`. Il **5/5 è fissato a priori** (no "≥4").
- **Onestà:** il bootstrap-p è quasi-sempre ≈0 (near-vacuo) → **lead su effect size + consistenza-seed**,
  gli asterischi Holm sono il *pavimento*, non l'argomento.
- **Holm a due famiglie (solo PRIMARI):** PRIMARIA = `B_full × macro-Cat-MRR × {L1,L2} × 7` = **14 test**;
  ROBUSTEZZA-backbone = `(altri 6 backbone) × macro-Cat-MRR × {L1,L2} × 7` (famiglia separata).
  Appendice e altre metriche: descrittive (p_raw, niente Holm famiglia-wide).
- **TOST ±0.005** per i claim di equivalenza ("non costa accuracy") — affermati **solo dove TOST passa**,
  per quello specifico backbone/dataset; il costo item è **eterogeneo** (equivalente su FPMC/SASRec,
  grande su EASE, segno dataset-dipendente), riportato per-backbone × per-dataset, mai mediato.

## 7. Provenienza e freeze (operativo)
1. Commit di `results_record.py` + `aggregate_record.py` + `METRICS_FORMULAS.md` + questo doc.
   **freeze commit = f1743d7** (branch `exp/second-dataset-feasibility`).
2. Ambiente: `IntentAwareRS_thesis/.venv` (torch, numpy<2, pandas); seed globali = SEEDS.
3. Re-run UNA volta (caso cheap, score-cache già k10 → no re-train):
   `python scripts/yelp/results_record.py <7 primari [+ 5 appendice]>` → cache + CSV.
4. **Validare la cache**: round-trip `aggregate_record.py` == diretto (max|Δ|=0) prima di derivare numeri.
5. Stability MSUPP∈{20,50}; gate winner ml1m (§5); poi tabelle.

## Deviation log (dopo il freeze)
| Data | Cosa è cambiato | Perché | Impatto sui claim |
|---|---|---|---|
| | | | |
