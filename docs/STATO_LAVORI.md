# X-SAGE — Stato lavori & guida per riprendere altrove

Aggiornato: 2026-07-01. Branch `exp/second-dataset-feasibility`. Questo file è il punto d'ingresso:
dice **dove siamo**, **cosa manca**, e **come rifare tutto su un'altra macchina**.

## Documenti (leggi in quest'ordine)
- `SUNTO_TOTALE.md` — idea, framework, risultati, lente (per revisione).
- `PROCESSO_SPERIMENTALE.md` — checklist del processo eseguito.
- `PIPELINE_VALIDAZIONE.md` — pipeline passo-passo (Metodi).
- `METRICS_FORMULAS.md` — formule di tutte le metriche.
- `FREEZE_RECORD.md` — decisioni congelate.
- `EXPERIMENT_PLAN.md` — regole di decisione PRE-REGISTRATE (B6).
- `RISULTATI_COMPLETI.md` — dump numerico dei dataset finiti.

## Stato risultati (al 2026-07-01)
**Finiti (cache + CSV):** ml1m, nyc_tist, saopaulo (7 backbone), kuairand (2 backbone).
| dataset | casella (focale B_full) |
|---|---|
| ml1m | **winner** (su tutti e 7 i backbone) |
| nyc_tist, saopaulo | ridondante (POI) |
| kuairand | null (controprova) |

**Mancano:**
- `yelp` (2 backbone) — piccolo, gira in locale.
- `amazoncd`, `mind` — ⚠️ **NON girano su 16GB RAM** (matrici 1.8–2.0GB → picco oltre la RAM → swap/thrashing). Servono: ottimizzazione-memoria del pipeline (mmap sb, free B_full per-seed, cache per-seed) OPPURE una macchina ≥32GB.
- Ablation-contrasto B6 (`ablation_contrast.py`) su ml1m/nyc/sao — da lanciare.

## Cosa NON è nel repo (va rigenerato)
- `data/` (37GB, dataset) — gitignored. Vanno riottenuti dai preprocess (`scripts/*/preprocess_*.py`, k-core 10).
- `outputs_results/cache/*.npz` (cache grezza) — gitignored, rigenerabile con `results_record.py`.
- `results_record.csv` — output, rigenerabile con `aggregate_record.py` dalla cache.
- Gli **score dei 5 backbone extra** (EASE/DeepFM/AFM/FPMC/SASRec) — solo su ml1m/nyc/sao; rigenerabili con `run_cars_setA.sh <city>`.

## Come rifare su un'altra macchina
1. `git clone https://github.com/laliberti00/XSAGE.git` (branch `exp/second-dataset-feasibility`).
2. Ricrea i due venv (NON copiabili tra OS): `IntentAwareRS_thesis/.venv` (torch, numpy<2, pandas) e `.venv-cornac` (cornac). Il codice X-SAGE importato sta in `IntentAwareRS_thesis/` (serve anche quello).
3. ⚠️ **Path assoluti hardcoded** negli script (`CLEAN=/Users/.../xsage-clean`, `sys.path.insert('.../IntentAwareRS_thesis')`) — vanno adattati ai path della nuova macchina (o replica la stessa struttura).
4. Riottieni/preprocessa i dataset (k-core 10, split temporale).
5. Genera score backbone (`run_cars_setA.sh` per i 7-backbone; il resto usa BPR+FM).
6. Run: `results_record.py <city…>` → cache; `aggregate_record.py <city…>` → CSV; `stability_check.py` / `ablation_contrast.py` → gate.

## Prossimi passi (ordine)
1. **Stanotte (in locale, sicuro):** yelp + aggregate dei 5 + ablation B6.
2. **amazoncd + mind:** ottimizzazione-memoria o VM (amazoncd = cella ridondante-non-POI, prioritaria).
3. **Scrittura paper** (prosa allineata al codice/critica): soften "interpreted through"→joint, L3 diagnostico,
   fairness→z-scoring (non long-tail), intent_last esplicito, §setup (Steck-b/per-richiesta/min-support/rango
   atteso), 2 micro-fix framework (cap K +2, ε tiebreak). + recupera `sn-bibliography.bib`.

## Storico movimenti
= la **git history** (ogni fix/decisione è un commit descrittivo). `git log --oneline` è il diario.
