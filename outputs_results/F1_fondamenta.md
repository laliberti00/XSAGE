# F-1 — Verifica delle fondamenta

**Data:** 2026-08-22 · **Natura:** verifica pura. Nessun esperimento, nessuna correzione, nessun commit.
**Codice scritto:** un solo script di misura, `f1_stage1_xsage.py`, **fuori dal repo** (scratchpad di
sessione), necessario perche' i due repo espongono entrambi un package chiamato `xsage` e vanno
importati in processi separati. Nessuna scrittura dentro `X-SAGE`: il redirect delle score matrix
e' un monkeypatch in memoria.

## Sintesi

| voce | esito | dove | ore | conf. | rischio per il piano |
|---|---|---|---|---|---|
| A · reversibilita' | **PULITO** — 0 file preesistenti modificati | `git status` | 0 | alta | nullo |
| B · gate su X-SAGE | **PASSA 3/3, esatto a 5 decimali** | vedi §B | 0 | alta | **rimosso** |
| C · maschera esclusione | **INTATTA** | `xsage/data.py:102` | 0 | alta | **rimosso** |
| D1 · `scripts/fairness/` | **NON ESISTE** | — | 6 | media | basso |
| D2 · variante min-max | non e' una variante: oggi non ottimizza nulla | `xsage/recommendation.py:35` | 16–24 | **bassa** | **alto** |
| D3 · kappa per-situazione | **nessuna modifica al combiner** | `eval_kappa.py:54` | 4 | medio-alta | nullo |
| D4 · plumbing LLM | **assente in tutto il repo** | — | 16 | media | medio |
| D5 · corpus 1000 stratificato | filtri confermati | `reco_examples.py:79-86` | 4 | alta | basso |
| D6 · demografia | **2 dataset**, non 1 | ml-1m + KuaiRand | 0 | alta | **ridotto** |
| D7 · baseline DP / microaggregazione | **assenti** | — | 7 | media | medio |
| E · tesi | compila, 127 pp., 0 ref irrisolti | D1 | 0 | alta | nullo |
| F · manoscritto | audit **14/14**, URL a 2 righe | `template.tex` | 0 | alta | basso |

---

## TASK A — Reversibilita'

Branch `exp/second-dataset-feasibility`, HEAD `db608f8`.
**Zero file tracciati modificati.** Solo aggiunte non tracciate:
`scripts/diagnostics/` (5 file), `scripts/exp/` (2), `outputs_results/{diagnostics,probe_wi0b,
probe_wi0c,probe_wi0d,exp_e1}/`, piu' i 3 file gia' presenti a inizio sessione
(`plot_costbenefit_grid.py` e le 2 figure).

`git clean -nd` elenca esattamente quelle **piu' una voce in piu': `.claude/`** — cartella vuota
che **preesisteva** alla sessione. Per una reversibilita' totale va esclusa:
`git clean -fd -e .claude`.

## TASK B — Il gate su X-SAGE

**Metodo.** Eseguita la catena di `X-SAGE` (descrittore -> `select_K`/`select_eps` -> rough
k-means -> `b_z`) e valutata con **il valutatore che definisce il gate** (`eval_kappa.cat_mrr`),
tenendo fisso il valutatore e scambiando solo la pipeline: cosi' un'eventuale divergenza sarebbe
attribuibile alla pipeline e non alla convenzione di metrica.

| dataset | K/eps clean | K/eps X-SAGE | clean | X-SAGE | atteso | esito |
|---|---|---|---|---|---|---|
| ml1m | K=5 e=0.03 | K=5 e=0.03 | 0.38479 | **0.38479** | 0.38479 | PASSA |
| nyc_tist | K=8 e=0.02 | K=8 e=0.02 | 0.34162 | **0.34162** | 0.34162 | PASSA |
| saopaulo | K=4 e=0.03 | K=4 e=0.03 | 0.40045 | **0.40045** | 0.40045 | PASSA |

Anelli a monte, tutti verificati **prima** del numero finale:
- **dati processati**: 15 file su 15 (3 dataset x 5 file) **identici per md5**;
- **iperparametri chiusi**: `config/params/*.json` identici a `outputs_results/params/*.json`;
- **costanti di selezione**: `K_RANGE`, `EPS_GRID`, `BAND`, `SIL_N`, `ALPHA`, `GAMMA/DEPTH/N/H/BETA` coincidono;
- **K ed eps selezionati**: identici su tutti e tre.

**Conseguenza.** Il refactoring e' numericamente identico, ed e' confermata di riflesso la scelta
sulle ~149 righe di `l1_perception` (tenere la versione di `IntentAwareRS`): e' quella giusta.
Si puo' procedere al packaging.

## TASK C — La maschera di esclusione

**INTATTA.** `load_excluded_mask(city, n_items, data_root)` (`xsage/data.py:102`) **non riceve
`n`** e ricarica sempre `URM_train + URM_val` da disco. In `mind_prep.build_descriptor:115` e'
invocata una sola volta, fuori da qualunque dipendenza dal troncamento.
Misura: utente 42 su ml1m -> **45 item esclusi, identici a n=1 e n=all** (nnz totale 901.340).

Il risultato di E-1 (troncare migliora il backbone) **non e'** l'artefatto temuto. Resta reale e
la sezione si puo' scrivere.

## TASK D — Inventario di fattibilita'

**D1 · `scripts/fairness/situational_qos.py`** — non esiste; la cartella `scripts/fairness/` non
esiste. Da scrivere da zero, ma gli ingredienti ci sono tutti (`macro_avg.py` per la macro,
`membership_from_assign` per la situazione, kappa* nelle battery). **6 h, confidenza media** —
il costo vero non e' il codice ma la regola di supporto delle celle (K situazioni x 18 categorie
con min-support 20 assottiglia parecchio su nyc_tist).

**D2 · `fit_situation_biases_z`** (`xsage/recommendation.py:35`).
Firma: `(z_train, cat_macro_train, K, n_macros, alpha=50.0) -> (K, n_macros) float32`.
**Cosa ottimizza oggi: NULLA.** E' uno stimatore in **forma chiusa** — conteggi per (situazione,
categoria), smoothing di Dirichlet verso il prior globale, log-odds, z-score dentro la situazione.
Nessuna funzione obiettivo, nessun gradiente, nessuna iterazione.

Percio' una variante **min-max non e' una variante di questa funzione**: e' un oggetto diverso.
Servirebbe introdurre un problema di ottimizzazione su K x M parametri con obiettivo
"massimizza la QoS della situazione peggio servita", scegliendo su quale split ottimizzare
(pena la circolarita'). **16–24 h, confidenza BASSA** — e' la stima meno affidabile del documento,
perche' il disegno e' aperto, non perche' il codice sia lungo.

**D3 · kappa per-situazione.** Oggi kappa e' **scalare**, entra in `eval_kappa.py:54` come
`S + kappa * gamma[:,None] * dmac[:,icm]` (identico in `battery_bfull.py:59`).
**Reperto utile:** un kappa per-situazione **non richiede alcuna modifica al combiner**. Poiche'
`dmac = mem @ b_z`, avere kappa_k per situazione equivale a **riscalare la riga k di `b_z`**:
`sum_k mem_k * kappa_k * b_z[k] = mem @ (diag(kappa) @ b_z)`. Basta passare un `b_z` riscalato.
Cambia solo la **selezione**, che da 1-dimensionale diventa K-dimensionale. **4 h** per una
sweep parametrizzata a una dimensione, **confidenza medio-alta**.

**D4 · Plumbing LLM.** **Nessun client API in tutto il repo** (nessun `anthropic`, `openai`,
`requests`, `httpx`, nessuna chiave). `scripts/explain/` non esiste. Da costruire: client con
prompt versionati e seed/temperature fissi, generatore a 3 condizioni, parser dei driver
asseriti, giudice di famiglia diversa dal generatore. **16 h di solo impianto**, escluse le
chiamate, **confidenza media**.

**D5 · `reco_examples.py`.** Confermato: i filtri che tengono **solo le vittorie** sono tre, non
due — riga **79** (`if nudge[r][tc] <= 0: continue`, la situazione deve favorire la categoria
vera), riga **81** (`if rb < 4: continue`), righe **84** (`if rs > 12 or rs >= rb: continue`).
Piu' la selezione del massimo per situazione (riga 86). Un corpus stratificato da 1000 casi
richiede di rimuovere i filtri, campionare per strato (vittorie/neutri/danni/boundary) e
aggiungere le colonne. **4 h, confidenza alta.**

**D6 · Attributi demografici.** **Due dataset, non uno:**
- `data/ml-1m/users.dat` — gender / age / occupation (baseline gia' note: 71,7% · 34,7% · 12,6%);
- `data/KuaiRand-Pure/data/user_features_pure.csv` — **27.285 utenti**, con
  `user_active_degree`, `is_live_streamer`, `is_video_author`, `follow/fans/friend_user_num_range`,
  `register_days_range`, `onehot_feat0..N` (demografia anonimizzata).

Yelp, MIND e i Foursquare **non** hanno attributi utente. La tabella della fuga (E8) **non e'
per forza mono-dataset**: KuaiRand da' il secondo, ed e' anche il dataset "nullo", il che rende
il contrasto piu' interessante.

**D7 · Baseline DP / microaggregazione.** **Assenti.** I tre hit del grep sono falsi positivi:
`epsilon` in `l2_comprehension.py:172` e `epsilon_final.py` e' il margine di boundary del rough
k-means, e `build_trust_html.py:67` cita la Differential Privacy proprio nell'elenco di cio' che
**non** e' coperto (`"C", "no", "Privacy formale (Differential Privacy)"`).
Stime: rumore DP sul profilo utente prima dello scoring **4 h**; microaggregazione (profilo
sostituito dal centroide del cluster) **3 h**. **Confidenza media** — il codice e' breve, la
taratura di epsilon e del livello di aggregazione no.

## TASK E — Stato della tesi

1. **D1 compila**: `Output written on main.pdf (127 pages)`, **0 riferimenti irrisolti**.
2. Conteggio parole per capitolo (D1):

| cap. | file | parole | stato |
|---|---|---|---|
| 1 | intro | 24 | stub |
| 2 | foundations | 10.238 | completo |
| 3 | review | 7.865 | completo |
| 4 | framework | 3.389 | completo |
| 5 | method | 33 | **vuoto** |
| 6 | explainability | 31 | **vuoto** |
| 7 | fairness | 30 | **vuoto** |
| 8 | privacy | 28 | **vuoto** |
| 9 | conclusions | 17 | stub |

Totale scritto: ~21.500 parole su 4 capitoli. **Cinque capitoli su nove sono da scrivere.**
(Nella madre esistono bozze in italiano: method 72 righe, explainability 342, fairness 226,
privacy 198 — narrativa senza numeri.)

3. **Capitolo `profiles` fuori dal build**: commentato in `main.tex:254`, e la cartella non esiste
   in D1 (solo nella madre). **0 `\ref` pendenti** verso di esso.
4. **Bibliografia: 179 voci, 0 chiavi duplicate, 0 warning bibtex.**
   (Nota: `bibliography.bib` e diversi `.tex` hanno terminatori di riga **CR**, quindi `grep -c "^@"`
   restituisce 0 — non e' un file vuoto. Serve `tr '\r' '\n'` prima di contare.)
5. **Topics API**: citata **una sola volta**, via `jha_topics_2023` in
   `chapters/foundations/foundations.tex:1166`, come studio di re-identificazione. Il contesto da
   rivedere e' pero' piu' ampio: `foundations.tex:1125` (*"from third-party cookies, have pushed
   towards keeping less"*) e `:1135` (*"cookieless designs"*), piu' `framework.tex:408`.
   **Non corretto**, come da vincolo.
6. **Numerazione coerente** fra madre e D1: entrambe 6=Explainability, 7=Fairness, 8=Privacy,
   9=Conclusions.

## TASK F — Manoscritto

1. **URL**, a **due** righe (non una):
   - `template.tex:927` — *"...is released at \url{https://github.com/knowmis/X-SAGE}."*
   - `template.tex:1429` — dentro la data availability statement.
2. **Data availability statement** (`template.tex:1426`), testo esatto:
   > The full implementation (preprocessing, backbones, the situational head, the selection
   > procedure, and the evaluation), together with the pre-registered decision rule and the
   > per-seed configuration, is released at \url{https://github.com/knowmis/X-SAGE}. The public
   > datasets used (MovieLens-1M, Foursquare TIST'15, Yelp, and KuaiRand-Pure) are available from
   > their original sources.

   Il DOI Zenodo va affiancato all'URL in **entrambe** le righe, 927 e 1429.
3. **Audit di nomenclatura esteso.** Il manoscritto ha 4 tabelle: `tab:notation`, `tab:trust`,
   `tab:acc`, `tab:exp`. Quelle con nomi di backbone sono **due**, non una — l'audit precedente
   copriva solo `tab:acc`. Verificata anche `tab:exp` (Gini su ml1m): **7/7 corrispondono**.
   **Totale audit: 14/14.**

---

## Le tre righe secche

- `gate X-SAGE = PASSA` (3/3, esatto a 5 decimali; dati, iperparametri, K ed eps identici a monte)
- `maschera di esclusione = INTATTA` (`xsage/data.py:102`, non funzione di n; 45 item su utente campione a ogni n)
- `curva di intervento fairness (E6) = FATTIBILE in <=2 giorni` — grazie a D3: kappa per-situazione
  equivale a riscalare le righe di `b_z`, quindi **zero modifiche al combiner**; il costo e' la
  sweep e i CI, non l'implementazione. Confidenza medio-alta.

## Questioni emerse (riportate, non eseguite)

1. **`.claude/` e' preesistente e vuota**, ma `git clean -fd` la rimuoverebbe. Usare
   `git clean -fd -e .claude` se si vuole reversibilita' stretta.
2. **Terminatori CR** in `bibliography.bib` e in alcuni `.tex` della tesi: fanno fallire in
   silenzio i conteggi con `grep "^..."`. Non e' un errore di contenuto, ma qualunque script di
   verifica sulla tesi deve normalizzarli prima.
3. **Il capitolo 5 (method) e' vuoto quanto i tre di trustworthiness.** Il piano dei 39 giorni
   considera 6/7/8; il 5 e' il capitolo che ospita il metodo gia' pubblicato ed e' anch'esso da
   scrivere da zero.
