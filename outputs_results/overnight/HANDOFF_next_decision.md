# X-SAGE — Hand-off per decidere il criterio di selezione (ε + griglie)

Documento autosufficiente. Una nuova chat lo legge e può ragionare sulla
decisione aperta senza altro contesto. Data: 2026-06-19.

---

## 0. In una frase

La selezione congiunta plateau-aware dei parametri delle situazioni è girata
(notte, 3.11 h, pulita). **Ha funzionato come codice, ma ha rivelato due
problemi di criterio**: (1) selezionare ε sull'ARI è degenere (tutte le città
finiscono su ε minimo), (2) l'ottimo cade sui bordi della griglia. Decisione da
prendere: **come selezionare ε** e **come estendere le griglie**, poi ri-run.

---

## 1. Dove siamo (fasi del progetto)

Repo pulito: `~/Downloads/xsage-clean` (standalone per i backbone, importati con
checksum). Validazione per fasi in `docs/`:
- **00 Data & Backbone** — VALIDATO (integrità 5/5; backbone importati, 10/10 checksum).
- **01 Sensing (L0)** — VALIDATO (split causale 0/N su 5/5).
- **02 Perception (L1)** — VALIDATO CON RISERVA (logica 8/8; γ/depth/n flaggati
  spiky → da qui nasce la ri-selezione).
- **03+ Comprehension/Projection/Recommendation/Metrics/Fairness** — da validare.

La ri-selezione serve a risolvere il **selection bias**: nel dossier (K,ε) furono
scelti massimizzando l'ARI ai default di perception, e i default γ/depth/n
sedevano su picchi stretti.

Dati: Python `~/Downloads/IntentAwareRS_thesis/.venv/bin/python`. Le 5 città
(low→high transit): istanbul, bangkok, nyc_tist, saopaulo, tokyo_tist.

---

## 2. Cosa ha fatto il run notturno

Script: `scripts/overnight_selection.py` (lancia-e-dormi: checkpoint per cella,
log, resume, idempotente; già validato con uno smoke-test che riproduce NYC
X-SAGE R=0.0907/LT=0.0974 al bit). Criterio **maximin plateau-aware** su
VALIDATION (fit su train → assegna a val → ARI cross-seed), vicinato "a torre"
±1. Perception γ/depth/n SHARED; K/ε per-città. H=2, β=0.7 fissi. Regola sink
resa **per-request coerente**. Stadi: A (perception scan, S=3) → B (K/ε per
città + verifica plateau congiunta, S=5) → conferma finalisti S=10 → report
metriche su TEST.

Output in `outputs_results/overnight/`: `stage_a_ari.csv` (400 celle),
`stage_a_plateau_scores.csv`, `stage_b_ari.csv` (100+ celle), `selected_params.json`,
`report_metrics.csv` (15 righe), `sinks_new.csv` (30 righe), `SUMMARY.md`,
`logs/overnight_20260619_002207.log`.

### Parametri selezionati
- Perception (shared): **γ=0.7, depth=5, n=7** (plateau score P=0.766).
- Per città: istanbul K=8 ε=0.01 · bangkok K=4 ε=0.01 · nyc K=6 ε=0.01 ·
  saopaulo K=8 ε=0.01 · tokyo K=4 ε=0.01.
- ARI conferma S=10: ist 0.611, bkk 0.753, nyc 0.739, sao 0.864, tky 0.781.

---

## 3. 🚩 Problema 1 — selezionare ε sull'ARI è DEGENERE

**Tutte e 5 le città scelgono ε=0.01 = minimo della griglia.** Causa strutturale:
ε più piccolo → meno punti boundary → assegnazione più "hard"/deterministica →
ARI cross-seed più alto, **monotonicamente**. Evidenza dal log (ARI vs ε a K fisso):

| città (K) | ε=0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|
| nyc (6) | **0.879** | 0.837 | 0.826 | 0.683 |
| istanbul (8) | **0.657** | 0.582 | 0.447 | 0.290 |
| bangkok (4) | 0.673 | **0.823** | 0.632 | 0.589 |
| tokyo (4) | 0.798 | 0.761 | 0.751 | **0.809** |

Il trend è "più piccolo ε = più stabile" → il maximin spinge ε al minimo **per
costruzione**. Quindi **l'ARI da solo non è un criterio ben posto per ε**.

**Perché conta:** ε controlla la **boundary fraction**. Con ε=0.01 la frazione di
richieste boundary è quasi nulla → il meccanismo boundary di X-SAGE (γ_S=1/|T|,
la disambiguazione delle situazioni di confine) **quasi non si attiva**. Il
dossier evitava la degenerazione scegliendo ε con il **vincolo boundary fraction
∈ [10%, 30%]** (poi massimo ARI tra i candidati in banda). Il mio script ha
**tolto** quel vincolo (interpretando il brief "parametri scelti SOLO sull'ARI,
fairness fuori dal loop"). Ma la boundary fraction è una proprietà **strutturale**
del clustering, NON una metrica di fairness: va rimessa nel criterio di ε.

---

## 4. 🚩 Problema 2 — l'ottimo cade sui BORDI della griglia

| parametro | scelto | griglia | posizione |
|---|---|---|---|
| γ | 0.7 | {0.3,0.4,0.5,0.6,0.7} | **bordo max** |
| depth | 5 | {2,3,4,5} | **bordo max** |
| ε | 0.01 | {0.01,0.02,0.03,0.05} | **bordo min** (tutte) |
| K | 4 o 8 | {4,5,6,7,8} | bordo per 4 città su 5 |
| n | 7 | {3,5,7,10} | interno ✓ |

Un valore al bordo ha **un solo vicino** → il maximin non può certificare che sia
un plateau **interno**: la griglia non *racchiude* l'ottimo. La selezione è valida
data la griglia, ma "appoggiata al muro". Va estesa per bracketare l'ottimo.

---

## 5. Impatto a valle (perché non propagare ancora)

| | dossier | nuovo |
|---|---|---|
| NYC touch share | 6.8% | **23.0%** |
| NYC X-SAGE R@20 | 0.0907 | 0.0849 (−6%) |
| NYC X-SAGE APL/user | 0.0974 | 0.1706 (+75%) |
| sink istanbul/nyc/saopaulo | [0,4]/[5]/[5] | [6]/[3,4]/[4,7] |
| sink bangkok/tokyo | [2]/[] | [2]/[] (invariati) |

X-SAGE coi nuovi parametri spinge **molto più long-tail a costo di accuratezza**.
Sink/touch cambiano per 3 città su 5. Gli ARI nuovi sono più bassi del dossier per
4 città — in parte correzione di onestà (validation + S=10 + maximin vs train +
1-coppia + max), in parte costo della perception condivisa. **Se si propaga, tutto
il dossier (base table, sweep λ/κ, gating, sink, TOST) va rifatto sui nuovi
parametri** (backbone B_blind/B_full invariati: non dipendono dalle situazioni).

---

## 6. LA DECISIONE da prendere

**Come selezionare ε** (il nodo): tre opzioni nel merito —
- **(a) Vincolo boundary-band [10%,30%]** (come il dossier): tra le celle (K,ε)
  con boundary fraction in banda, scegli col plateau-maximin sull'ARI. Difendibile,
  riallinea al dossier, riattiva il meccanismo boundary. **Raccomandata.**
- (b) ε fissato a priori (es. mediana delle margini, `auto_epsilon` esistente in
  `xsage/l2_comprehension.py`) e fuori dalla selezione.
- (c) criterio composito ARI×(boundary in banda) come obiettivo unico.

**Come estendere le griglie** (per bracketare l'ottimo): γ→{...,0.8,0.9},
depth→{...,6,7}, ε→{0.005,0.002,...} (verso il basso), K più larga se serve.
Nota: se si mette il vincolo boundary-band su ε, estendere ε verso il basso
diventa meno urgente (la banda impedisce ε→0).

**Decisioni già fissate (non ridiscutere salvo motivo):** selezione su validation;
H=2/β=0.7 fissi; γ/depth/n shared + K/ε per-città; sink per-request coerente;
S=3/5/10. Convenzione LT@20 (per-user vs per-request) ANCORA aperta — il report
calcola entrambe; va decisa prima del ricalcolo finale del dossier per farlo una
volta sola.

---

## 7. Dove mettere le mani nel codice (per la Fase 2 esecuzione)

File unico: `scripts/overnight_selection.py`.
- **Griglie**: costanti in cima — `GAMMA_GRID, DEPTH_GRID, N_GRID, K_GRID, EPS_GRID`.
- **Vincolo boundary-band su ε**: la funzione `ari_on_validation(...)` oggi ritorna
  solo l'ARI. Va fatta ritornare ANCHE la boundary fraction (dal fit su train:
  `fit_rough_kmeans(...).is_boundary.mean()`, oppure sulla assegnazione val). Poi,
  nella selezione (K,ε) dentro `stage_b()` (il maximin per città su K×ε), filtrare
  le celle a `boundary_fraction ∈ [0.10, 0.30]` PRIMA del maximin (fallback:
  cella in banda più vicina al centro 20% se nessuna in banda, come fa il dossier).
- **Riferimento dossier** per il criterio originale (boundary-band): vecchio repo
  `pipeline/step02_models/xsage/orchestrator.py`, funzione `_tune_K_and_eps`
  (sweep K×ε con vincolo boundary in [0.10,0.30]) e `run_stage_a`.
- Lo script è **resumable** ma cambiare griglie/criterio = run nuovo (cancella o
  rinomina `outputs_results/overnight/*.csv` per ripartire pulito, ~3h).

## 8. Comandi

```bash
CLEAN=/Users/lucaaliberti/Downloads/xsage-clean
PY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
cd $CLEAN
# ispezionare i risultati attuali
cat outputs_results/overnight/SUMMARY.md
column -s, -t outputs_results/overnight/report_metrics.csv
# smoke-test del flusso (pochi minuti) dopo modifiche
$PY -m scripts.overnight_selection --smoke
# run vero (~3h, lancia-e-dormi)
./run_overnight.sh
```

## 9. Stato git / artefatti

Branch `main`. Backbone locali in `data/<city>/backbone/` (8.6 GB, gitignored).
Output selezione in `outputs_results/overnight/` (NON ancora propagati al dossier
— sono il risultato da validare/decidere, non i numeri finali del paper).

Fine documento.
