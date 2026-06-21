# Fase 00 — Data & Backbone

Stato: **VALIDATO** (2026-06-18)

Questa fase copre tutto ciò che X-SAGE riceve **in input** senza che sia una
nostra scelta di modello: i dati grezzi, la matrice utente-item (URM), e i
punteggi del backbone. Non è una fase del modello vero e proprio (quelle
iniziano da Sensing), ma è il fondamento su cui tutto il resto poggia, quindi va
validata per prima.

---

## 1. Dati grezzi — TIST2015

- **Cosa è:** TIST2015 (Foursquare global check-ins), file statici:
  `dataset_TIST2015_Checkins.txt` (2.1 GB), `dataset_TIST2015_POIs.txt` (221 MB),
  `dataset_TIST2015_Cities.txt` (28 KB).
- **Da dove viene:** dataset pubblico Foursquare. Finestra di compromesso usata:
  Apr 2012 – Feb 2013 (10 mesi), check-in globali Apr 2012 – Sep 2013.
- **Come è ottenuto:** scaricato una volta; risiede nel vecchio repo
  (`IntentAwareRS_thesis/data/raw/`). Per lo standalone completo va copiato nel
  clean repo (one-time, ~2.3 GB). Non è derivabile: è il punto di partenza.

## 2. URM e DataFrame preprocessati

- **Cosa è:** per ogni città, `URM_{train,val,test}.npz` (sparse) +
  `df_{train,val,test}.parquet` (per-interazione con feature contestuali) +
  `stats.json` + `mappings.json`.
- **Da dove viene:** generati dai grezzi dal carve per-città
  (`experiments/multicity/tist_carve.py` + `pipeline/step01_preprocessing/taxonomy.py`
  nel vecchio repo).
- **Come è ottenuto:** taglio per-città dai check-in globali → filtro k-core=10
  → cold-drop (rimozione item di val/test assenti in train) → split temporale
  per-utente 80/10/10. Processo **deterministico** (nessun RNG): split per
  timestamp + k-core iterativo.

### Esito validazione integrità (5 città)

Misure da `outputs_results/validation/A1_dataset_integrity.csv`:

| città | n_users | n_items | n_int_train | n_test | densità URM | u-min (full) | i-min (full) | vuoti u/i | K_mac | verdetto |
|---|---|---|---|---|---|---|---|---|---|---|
| Istanbul | 22631 | 9305 | 998030 | 136220 | 2.72e-3 | 9* | 10 | 0/0 | 10 | PASS |
| Bangkok | 6316 | 5185 | 323167 | 43556 | 5.48e-3 | 9* | 10 | 0/0 | 10 | PASS |
| NYC-TIST | 4113 | 3987 | 124270 | 17702 | 5.17e-3 | 10 | 10 | 0/0 | 10 | PASS |
| Sao Paulo | 4395 | 3207 | 172781 | 23821 | 6.27e-3 | 10 | 10 | 0/0 | 10 | PASS |
| Tokyo-TIST | 7160 | 5723 | 447563 | 59561 | 4.80e-3 | 9* | 10 | 0/0 | 9** | PASS |

**Criteri PASS** — (a) `n_test` esatti (5/5 al numero); (b) k-core=10 confermato;
(c) 0 utenti/item vuoti, densità in [1e-4, 1e-2], K_mac coerente. → **PASS 5/5**.

### Nota sul k-core (rigore di misura)

Il k-core=10 è imposto sul **set completo pre-split** (`train ∪ val ∪ test`), non
sui sottoinsiemi. Sul set completo l'**item-min = 10 esatto ovunque**.
\* L'**user-min = 9** in 3 città (Istanbul, Bangkok, Tokyo) è spiegato
**esattamente** dal cold-drop documentato in `stats.json` (11–27 interazioni
rimosse *dopo* il k-core perché item non presenti in train): è il protocollo del
floor, non una violazione.
\*\* Tokyo ha **K_mac=9** perché manca genuinamente la macro "Residence"
(0 venue): coerente, non un'anomalia.

> Lezione: misurare il k-core sul set giusto. Una verifica che lo misura su
> `train+val` (post-split) darebbe FAIL spurio — vedi
> [fase 01 — Sensing](../01_sensing/) per il dettaglio della correzione.

## 3. Backbone scores — B_blind e B_full

- **Cosa è:** i punteggi del modello di base che X-SAGE ri-rankizza.
  - `FM.scores.npy` = **B_blind**, backbone context-blind (FM / factorization
    machine, refit su `train ∪ val`), forma `(n_users, n_items)`.
  - `Bfull.scores.npy` = **B_full**, backbone context-aware per-richiesta,
    forma `(n_test, n_items)`.
- **Config esatta backbone:** FM con `n_components=128`, `n_epochs=20`,
  `learning_rate≈0.002`, regolarizzazione `user/item_alpha`,
  `negative_sampling_seed=2022`; HP da ricerca bayesiana congelati in
  `outputs/<city>/baselines/FM.best_hp.json`.
- **Come è ottenuto nel clean repo: IMPORT con checksum (non ri-addestramento).**
  I file sono stati copiati una volta dal vecchio repo nella posizione locale
  `data/<city>/backbone/`, con verifica SHA256 sorgente↔copia. Footprint totale
  **8.6 GB**. Il loader (`xsage/data.py:load_backbone_scores`) ora legge dalla
  posizione **locale** (fallback al vecchio solo se assente).

### Perché import-con-checksum e NON ri-addestramento

1. **Non-determinismo cross-hardware.** Il training FM non è bit-deterministico
   tra macchine: i driver fissano `torch.manual_seed` e il negative-sampling usa
   seed numpy fisso (2022), ma **non** c'è `torch.use_deterministic_algorithms`,
   quindi ordine delle operazioni float e threading BLAS introducono differenze
   minime. Ri-addestrando, i numeri cadrebbero entro la ±SD già caratterizzata
   (`base_table_union_SD.csv`) ma **non sarebbero identici al dossier al bit**.
2. **Il backbone non è una nostra scelta da validare.** È un punto di partenza
   **esterno** (un recommender di base). La tesi valida il *re-ranker* X-SAGE
   sopra un backbone fissato, non il backbone stesso. Importarlo verbatim isola
   il contributo da validare e garantisce numeri identici al dossier.
3. **Costo.** Ri-addestrare B_full sarebbe un job notturno (~8–12 h, Istanbul
   ~1.8 h di solo tuning). L'import è una copia verificata di pochi minuti.

### Esito validazione backbone (checksum)

Da `docs/00_data_backbone/checksums.csv` — **10/10 file verificati** (`verified=yes`),
SHA256 sorgente = copia per ogni file:

| città | file | dim (byte) | sha256 (prefisso) | verificato |
|---|---|---|---|---|
| istanbul | FM.scores.npy | 842325948 | `6166ebef2aae0bef…` | yes |
| istanbul | Bfull.scores.npy | 5070108528 | `5f7443a2eea631f0…` | yes |
| bangkok | FM.scores.npy | 130993968 | `c6b5838fc9a416b7…` | yes |
| bangkok | Bfull.scores.npy | 903351568 | `07e60ba86a8d5434…` | yes |
| nyc_tist | FM.scores.npy | 65594252 | `9ce5537d1f8f2402…` | yes |
| nyc_tist | Bfull.scores.npy | 282311624 | `cf45438ec4b1a89c…` | yes |
| saopaulo | FM.scores.npy | 56379188 | `f279a1dce4a5391f…` | yes |
| saopaulo | Bfull.scores.npy | 305575916 | `382bb4b93e29159f…` | yes |
| tokyo_tist | FM.scores.npy | 163906848 | `f3e7a051363ed14e…` | yes |
| tokyo_tist | Bfull.scores.npy | 1363470540 | `a114fa54e7fb8838…` | yes |

### Verifica end-to-end dopo lo switch al locale

Riproduzione leggendo dai **file locali** (non più dal vecchio repo):
- NYC-TIST X-SAGE: **R@20 = 0.0907, N@20 = 0.0381, LT@20 = 0.0974** ✓ (match dossier)
- Tokyo-TIST: **identità** X-SAGE = B_blind (0 sink) ✓
- `scripts/run_main_results` su 5 città: vedi
  `outputs_results/main_results.csv`.

---

**Verdetto fase 00: VALIDATO.** Integrità dati PASS 5/5; backbone importati con
10/10 checksum verificati; riproduzione locale combaciante.

I dati grezzi vanno copiati nel clean repo per lo standalone completo (non
ancora fatto in questa fase: il backbone è già locale, i grezzi restano l'unica
dipendenza dal vecchio repo per un eventuale ricalcolo futuro di URM/situazioni).
