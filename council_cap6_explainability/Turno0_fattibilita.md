# TURNO 0 — Accertamento tecnico
## Council Capitolo 6, asse Explainability · 28 agosto 2026

> **Mandato.** Rispondere alle cinque domande del brief §TURNO 0 **con misure, non con stime**.
> **Vincolo osservato.** Nessun commit, nessuna modifica a codice, dati o tesi. Tutti gli script di
> misura sono stati scritti **fuori dal repo** (scratchpad di sessione) e sono in sola lettura.
> L'unico file creato dentro il repo è questo documento.
> **Regola 7 del brief applicata:** dove un fatto contraddice l'istruttoria, vince il fatto (§7).

---

## 0 · Come sono state prese le misure

| voce | valore |
|---|---|
| repo | `/Users/lucaaliberti/Downloads/xsage-clean`, branch `exp/second-dataset-feasibility`, HEAD `db608f8` |
| interprete | `/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python` — numpy 1.26.4, pandas 2.1.4, scipy 1.13.1 |
| dataset | ml1m, backbone **B_blind**, seme **42**, κ* = **0,50** (da `outputs_results/battery_bfull_ml1m.csv`) |
| script di misura | 4 script in `…/scratchpad/t0_measure{,2,3,4}.py` — **fuori dal repo**, sola lettura |
| effetti collaterali sul repo | **nessuno** (`git status` invariato prima/dopo, a meno di questo file) |

**Ancoraggi riprodotti.** K = 5, ε = 0,03, macro = 18, dim(v) = 23, n_test = 97.199,
`bfrac` = 0,24229 — tutti coincidenti con `outputs_results/explain/cost_ml1m.txt` e con
`situation_space_ml1m.json`. La matrice `b_z` ricalcolata coincide con `fav_genres` di
`situation_profiles_ml1m.json` a tutte le cifre pubblicate (k=4: Documentary 1,528 · Drama 1,214 ·
Crime **1,081** — cioè esattamente il «nudge 1,081» del caso di riferimento dell'istruttoria §1).
**La catena di misura è ancorata.**

> ⚠️ **Nota di percorso, prima di tutto.** Il comando di lancio del brief (`cd ~/Desktop/xsage-clean`)
> **fallisce**: quella cartella non esiste. Il repo è in `~/Downloads/xsage-clean`. Anche l'istruttoria
> §11 indica `~/Desktop/xsage-clean/`. Da correggere prima di qualunque esecuzione non presidiata.

---

## 1 · Il record per richiesta — **SI PUÒ. Costo: ~15 secondi di macchina, ~1 ora di uomo.**

### Che cosa esiste già in memoria dentro `reco_examples.py`

Al momento del ciclo (riga 74) sono già calcolati e vivi:

| oggetto | riga | forma (ml1m) |
|---|---|---|
| `nudge = b_z[kte]` | `scripts/yelp/reco_examples.py:70` | (97.199, 18) — **ma core-only, vedi §3** |
| `kte`, `compte`, `isbte` | `:62` | (97.199,) ×3 |
| `gam_te` | `:63` | (97.199,) |
| `icm`, `ite`, `ute`, `hr`, `wk`, `il` | `:56`, `:68-69` | (n_items,) e (97.199,) |
| `sb` (B_blind), `excl` | `:56` via `build_descriptor` | memmap (6040, 3260) · CSR |
| `kap` | `:67` | scalare |

**Manca un solo oggetto: la matrice di appartenenza `mem`.** `membership_from_assign` è già
**importata** alla riga 17 e non viene **mai chiamata**. Una riga la aggiunge.

### Costo misurato di persistere tutto, per **tutte** le 97.199 richieste

Record = `{row, u, i, k, isb, gam, |T|, categoria_vera, mem (K), nudge (n_macros)}`:

| formato | dimensione | tempo di scrittura |
|---|---|---|
| `.npz` compresso | **0,63 MB** | 0,18 s |
| `.npz` non compresso | **10,89 MB** | 0,01 s |
| solo il vettore nudge float32 | 7,0 MB | — |

Scala sui 5 dataset della griglia (n_test: ml1m 97.199 · kuairand 52.768 · yelp 36.772 ·
saopaulo 23.821 · nyc_tist 17.702 = **228.262 richieste**): **sotto i 25 MB non compressi in totale.**
Il dimensionamento su disco **non è un vincolo** e non lo sarà mai.

### Righe da modificare

**Raccomandazione: non toccare `reco_examples.py`.** È uno script di *selezione di esempi*, con quattro
filtri in cascata (righe 75, 77, 81, 84) e `N = min(len(dft), 60000)` alla riga 73: il suo scopo è
opposto a quello del record. Se lo si volesse comunque adattare servono **4 modifiche**
(aggiungere `mem` dopo la riga 63; sostituire la riga 70 con `mem @ b_z`; togliere il tetto di 60.000
alla riga 73; inserire uno `np.savez` prima del ciclo).

**La via pulita esiste già ed è più corta.** `scripts/diagnostics/viability_probe.py:114`
`build_common(city)` costruisce e **restituisce** l'intero record: `mem_te` (`:132`), `gam_te`,
`k_te`, `isb_te` (`:157`), `nudge_te = mem_te @ b_z` (`:140`, `:159`), `ute`, `b_z`, `sb`, `excl`,
`icm`, più i vettori Cat-MRR per richiesta `q_sit`/`q_base` (`:153-154`). Uno script nuovo di
**~25 righe** che importa `build_common` e fa `np.savez` produce il record completo senza toccare
nulla di esistente.

### Tempi misurati, per fase (ml1m)

| fase | secondi |
|---|---|
| `build_descriptor` (L0+L1) | 8,71 |
| `select_K` | 62,33 |
| `select_eps` | 39,07 |
| `fit_rough_kmeans` | 4,89 |
| `fit_situation_biases_z` (b̃) | 0,03 |
| assegnazione test | 0,02 |
| `mem @ b_z` | 0,00 |
| **totale a freddo** | **115,1** |
| **totale con K/ε fissati** (sono già in `cost_ml1m.txt`) | **~14** |

> **Risposta all'incognita §8(b) dell'istruttoria: è «mezz'ora», non «mezza giornata»** — e la
> mezz'ora è quasi tutta scrittura di codice, non calcolo. 101 dei 115 secondi sono la *riselezione*
> di K e ε, che sono **già registrati** in `outputs_results/explain/cost_<ds>.txt` (`K=5 ε=0.03`) e
> possono essere fissati.

---

## 2 · Il riordino con colonne azzerate — **SI PUÒ, interamente da cache. 0,02 ms per richiesta.**

**Nessun riaddestramento di nulla.** Il backbone non viene mai toccato: `xsage/data.py:97` lo apre
in `mmap_mode="r"`. Il modulo situazionale viene **ri-derivato** in modo deterministico (seme 42) nei
~14 s di cui sopra, oppure persistito una volta e riletto.

### File necessari, e ci sono tutti (ml1m)

| file | stato |
|---|---|
| `data/ml1m/backbone/FM.scores.npy` (= B_blind, vedi §4) | ✅ (6040, 3260) float32, 78,8 MB |
| `data/processed/ml1m/df_{train,val,test}.parquet` | ✅ |
| `data/processed/ml1m/URM_{train,val}.npz` → maschera `excl` | ✅ 901.340 nnz |
| `outputs_results/params/ml1m.json` (parametri di percezione) | ✅ |
| `outputs_results/battery_bfull_ml1m.csv` (κ*) | ✅ κ* = 0,50 |
| `outputs_results/explain/cost_ml1m.txt` (K, ε) | ✅ |
| `outputs_results/explain/situation_profiles_ml1m.json` | ✅ **contiene già `b_z` completa**, arrotondata a 2 decimali, per **10 dataset** |

### Costo misurato

| operazione | misura |
|---|---|
| ranghi BASE **e** SIT dell'item held-out **su tutte le 97.199 richieste** | **4,5 s** |
| azzeramento colonne + riordino, 500 richieste | **0,01 s** → **0,02 ms/richiesta** |
| tripletta completa (pieno + ablato + solo-Ĉ), 2.000 richieste | **0,5 s** |

> **Costo per 500 richieste: 10 millisecondi**, più un setup una tantum di ~14 s.
> Il §6.5 dell'istruttoria («il resto è numpy su matrici in cache») è **corretto e persino
> conservativo**. Questa non è, e non sarà mai, la voce di costo del capitolo.

---

## 3 · Il caso boundary — **il codice lo calcola già, in due punti indipendenti. Ed è persistibile.**

### Dove

1. `scripts/mind/mind_prep.py:40` `membership_from_assign` → `mem` (B×K); usata in
   `mind_prep.py:151` dentro `score()`: `d = (mem @ b_z)[:, icm]`.
2. `scripts/diagnostics/viability_probe.py:132` `mem_te`, e `:140`
   `dmac["SIT"] = mem_te @ b_z`, restituito a `:159` come `nudge_te`.

Per una richiesta boundary con insieme di appartenenza T, `mem[r] = comp[r]/|T|` e il vettore vero è
`mem[r] @ b_z` — la **media pesata sulle appartenenze**. Lo smorzamento γ = 1/|T| resta separato e
viene applicato al momento del punteggio (`kap * gam[r] * nudge[r][icm]`).

### Quanto è persistibile

`(97.199, 18)` float32 = **7,0 MB**. Con l'intero record: 0,63 MB compresso (§1).

### Quanto è diverso dal ground truth usato oggi — **molto**

Verifica su tutte le 97.199 richieste (seme 42):

| | valore |
|---|---|
| righe **core** in cui `mem@b_z` è identico a `b_z[k]` | **73.649 / 73.649** (verifica `allclose`, esito vero) |
| righe **boundary** | 23.550 |
| righe boundary in cui i due vettori **coincidono** | **0 su 23.550** |
| mediana della massima differenza per categoria, sulle boundary | **1,626** |

Su una scala in cui l'intera matrice `b_z` di ml1m vive in [−2,24, +2,46], una differenza mediana di
**1,63** non è un dettaglio numerico: è un vettore diverso.

> **Conseguenza operativa.** `reco_examples.py:70` (`nudge = b_z[kte]`) è il ground truth **sbagliato**
> per il 24,2% delle richieste. Lo script non se ne accorge perché la riga 75 (`if bool(isbte[r]): continue`)
> scarta tutte le boundary. Qualunque script del capitolo 6 che riusi quella riga **eredita l'errore**
> nel momento in cui rimuove il filtro. Il record va costruito da `mem @ b_z`, non da `b_z[kte]`.

### Distribuzione di |T| — e una riconciliazione

Seme 42: |T|=1 → **73.649** · |T|=2 → **17.980** · |T|=3 → **5.182** · |T|=4 → **388**.

L'istruttoria §4.2 riporta 73.648 / 17.976 / 5.185 / 388. **Non è una discrepanza:**
`scripts/diagnostics/wi0d_probe.py:210` calcola `n0 = int(np.mean(nn))`, cioè la **media sui 5 semi**,
non il conteggio del seme 42. Chi cita quella tabella deve dire «media su 5 semi».

### Una precisazione che semplifica il disegno

κ e γ sono **scalari positivi per richiesta**. Quindi il vettore mostrato all'LLM (`mem@b_z`) e il
contributo effettivo (`κ·γ·mem@b_z`) hanno **lo stesso ordinamento di categorie e gli stessi segni**.
Ne segue che Precision, Recall, Kendall-τ e il «tasso di allucinazione» (nudge ≤ 0) sono **invarianti**
rispetto a quale dei due si sceglie. Cambia solo la magnitudine — e quindi solo le misure di
sottrazione (PN/PS). È una semplificazione reale del §6.3, e va scritta.

---

## 4 · Le matrici di punteggio — inventario e trappola

### Che cosa c'è su disco

| dataset | file backbone | griglia disponibile |
|---|---|---|
| **ml1m · nyc_tist · saopaulo · yelp · kuairand** | **11 ciascuno** | **7 backbone completi** (B_blind, B_full, EASE, DeepFM, AFM, FPMC, SASRec) |
| tokyo_tist | 7 | 5 backbone |
| mind · bangkok · istanbul · tsmc_nyc · tsmc_tky · amazoncd | 2 | solo B_blind + B_full |

`outputs_results/results_record.csv` ha **2.205 righe** = 5 dataset × 7 backbone × 3 metodi
(BASE / SIT / **Steck-b**) × 21 righe-metrica. **La griglia 5×7 esiste, è su disco, e Steck-b è già
calcolato su tutta la griglia.**

### La trappola nota, confermata

`FM.scores.npy` **contiene punteggi BPR**, non FM. Salvato con quel nome da
`scripts/mind/cornac_backbone.py:50`; documentato in
`outputs_results/probe_wi0d/task0a_backbone_naming.csv` (riga 2) e in `docs/CAP5_NUMERI.md:75`.
`xsage/data.py:97` lo carica come `blind`. **Il manoscritto è corretto** (lo chiama BPR); il nome del
file è un residuo ereditato. Il «FM» del paper è invece `Bfull.scores.npy` = ContextAwareFM = `B_full`.

### Due fatti di forma che il disegno deve conoscere

- `FM.scores.npy` è indicizzato **per utente**: (6040, 3260) → si accede con `sb[u]`.
- `Bfull.scores.npy` è indicizzato **per riga di test**: (97.199, 3260), 1,27 GB → **`sb[u]` non
  funziona**. Un braccio dell'esperimento su B_full richiede un percorso di indicizzazione diverso.
  Non è difficile, ma non è gratis, e nessuno l'ha notato.
- `data/` è in `.gitignore`: **nessuna matrice di punteggio è in git.** Rilevante per il Dossier F.

### Incognita §8(a): parzialmente risolta, ma il documento annunciato non esiste

`docs/VERIFICA_griglia_e_steckb.md` **non esiste sul disco.** La verifica annunciata dall'istruttoria
non è atterrata. La domanda «un backbone o sette?» è però già risolvibile da `results_record.csv`
(sopra): **sette, su cinque dataset.** La domanda «Steck-b per situazione è ricalcolo o run?» è
**ricalcolo**: `viability_probe.py:138` mostra che Steck-b è `fit_situation_biases_z` applicata
all'indice utente invece che all'indice situazione; la stessa chiamata costa **0,03 s** misurati.

---

## 5 · Zero codice LLM — **confermato, e in senso più forte di quanto l'istruttoria dichiari**

Ricerca su tutti i `.py .sh .toml .txt .cfg .json .yaml` del repo (escluso `.backup_freeze_20260630`)
per: `openrouter|openai|anthropic|api_key|OPENAI_API|litellm|langchain|transformers|huggingface|gpt-4|claude-3|chat/completions|prompt`.

**Zero corrispondenze.** Nessun client, nessun harness, nessun prompt versionato, nessun `.env`.

Di più: `requirements.txt` contiene **cinque righe** — `numpy`, `pandas`, `scipy`, `scikit-learn`,
`pyarrow`. **Non c'è `requests`.** Il repo non ha, oggi, nemmeno la capacità transitiva di fare una
chiamata HTTP. Tutto ciò che riguarda l'LLM — client, retry, snapshot, pinning, parsing di Ĉ — è
**da scrivere da zero**, e va contato per intero nel costo del capitolo.

---

## 6 · ⚠️ Quello che il Turno 0 ha trovato senza che gli fosse chiesto
## PN e PS, **come definite nel §6.3 dell'istruttoria, non misurano la fedeltà**

Il brief chiede: «Se una risposta è "non si può", dirlo qui: cambia il giudizio di tutti gli attori
successivi.» Questa è quella risposta. **L'operazione si può fare** (§2: costa 0,02 ms). **Il numero
che produce non è quello che il disegno crede.**

### L'esperimento di controllo

Campione casuale di **2.000 richieste** di test (seme 11), κ* = 0,50, B_blind, seme 42. Tre insiemi
rivendicati Ĉ, di cardinalità 3, applicati alla *stessa* richiesta:

- **Ĉ_vero** = le 3 categorie più spinte dalla situazione riconosciuta (l'insieme che una spiegazione
  perfettamente fedele nominerebbe — è letteralmente `fav_genres`);
- **Ĉ_falso** = le 3 categorie più **penalizzate** dalla stessa situazione;
- **Ĉ_casuale** = 3 categorie estratte a caso.

Una metrica di fedeltà che funziona deve separare nettamente il primo dagli altri due.

### Problema 1 — il supporto: PN è **indefinita** sul 94% delle richieste

PN chiede «azzerando Ĉ, l'item **esce** dal top-k?». La domanda ha senso solo se l'item **ci era**.

| denominatore | top-5 | top-10 | top-20 |
|---|---|---|---|
| item held-out nel top-k (SIT) | 43 / 2.000 (**2,2%**) | 70 (3,5%) | 128 (**6,4%**) |
| categoria vera nel top-k (SIT) | 1.026 (51,3%) | 1.260 (63,0%) | 1.473 (73,7%) |

Confermato su tutto il test: **6.303 richieste su 97.199 (6,5%)** hanno l'item held-out nel top-20 SIT.
Il rango mediano dell'item è **327** (BASE 354). Un campione stratificato da ~100 casi per strato
produrrebbe **~6 richieste utilizzabili per PN a livello di item.**

### Problema 2 — la discriminazione: a livello di categoria, **non c'è**

| misura | Ĉ_vero | Ĉ_falso | Ĉ_casuale | supporto |
|---|---|---|---|---|
| **PN** item, top-20 | 0,156 | 0,047 | 0,070 | 128 |
| **PN** item, top-5 | 0,279 | 0,070 | 0,070 | 43 |
| **PN** categoria, top-5 | 0,091 | 0,049 | 0,035 | 1.026 |
| **PN** categoria, top-20 | **0,019** | **0,023** | 0,014 | 1.473 |
| **PS** item, top-20 | 0,914 | 0,805 | 0,805 | 128 |
| **PS** categoria, top-5 | 0,937 | 0,877 | 0,864 | 1.026 |

Si legga la riga in grassetto. **Su PN a livello di categoria e top-20, l'insieme deliberatamente
sbagliato ottiene un punteggio più alto di quello giusto** (0,023 contro 0,019). Le altre righe
separano poco e nella direzione giusta, ma con valori assoluti minuscoli o al soffitto.

Verifica indipendente su un secondo campione (3.000 richieste con item nel top-20 SIT):
PN = **0,148**, PS = **0,905** con Ĉ = le 3 categorie vere. Coerente.

### Perché succede — e non è un bug

`κ·γ·b̃` è un **piccolo** termine additivo rispetto alla dispersione di `s_B`: sposta il rango mediano
dell'item da 354 a 327 (−7,6%). Azzerare 3 colonne su 18 quasi mai fa attraversare a un item la
soglia del top-k. **PN e PS, così definite, misurano l'intensità di κ, non la fedeltà della frase.**

Il risultato titolare del disegno sarebbe: *«anche quando l'LLM nomina esattamente le tre categorie
giuste, la necessità controfattuale è 0,15»* — che un revisore leggerebbe come «le vostre spiegazioni
sono false all'85%», quando in realtà sono **perfette** e il modulo è **debole**. È il peggiore degli
esiti: negativo, e per la ragione sbagliata.

### La buona notizia: l'asse è salvabile, e la cura resta dentro la bibliografia già citata

Stessa ablazione, ma misura **continua** invece che binaria — la quota di spostamento di rango
attribuibile a Ĉ:

```
quota(Ĉ) = (rango_ablato − rango_pieno) / (rango_base − rango_pieno)
```

| | Ĉ_vero (mediana [IQR]) | Ĉ_falso | Ĉ_casuale | supporto |
|---|---|---|---|---|
| a livello di **item** | **0,600** [0,167 – 0,778] | 0,110 [−0,129 – 0,400] | 0,005 [−0,014 – 0,200] | **1.966 / 2.000 (98,3%)** |
| a livello di **categoria** | **0,500** [0,000 – 0,852] | 0,000 [0,000 – 0,500] | 0,000 [0,000 – 0,200] | 1.467 / 2.000 (73,4%) |

Separazione di **due ordini di grandezza** sulla mediana (0,600 contro 0,005), e supporto sul **98,3%**
delle richieste invece che sul 6,4%. Costo identico: le stesse tre valutazioni già misurate a 0,25 ms.

**Questa non è una metrica inventata.** È la forma di *Performance Shift*, che l'istruttoria §5.2 e
§6.3 già citano dalla stessa fonte di PN e PS — Chen, Zhang & Wen 2022 §4.2. Il disegno ha preso due
delle tre misure di quel paragrafo e ha lasciato lì la terza, che è l'unica con dinamica utile su
questo modulo.

> ⚠️ **Limite dichiarato di questo rilievo.** Il Turno 0 **non ha aperto** il PDF di Chen, Zhang & Wen
> 2022. La corrispondenza fra la misura qui proposta e la loro *Performance Shift* è **da verificare a
> pagina** — è compito del Dossier A e del Dossier C, ed è **vincolante**: se la definizione non
> coincide, la misura va rinominata o rifondata, non spacciata.

---

## 7 · Materiale di dimensionamento per il Dossier E (§6.1 dell'istruttoria)

Ranghi su **tutte** le 97.199 richieste di test (4,5 s di calcolo).

**Percentili del rango** [25 · 50 · 75 · 90 · 99]:

| | BASE | SIT |
|---|---|---|
| item held-out | 122 · 354 · 802 · 1.373 · 2.354 | 112 · **327** · 758 · 1.336 · 2.330 |
| categoria vera | 2 · 5 · 19 · 69 · 369 | 1 · **5** · 21 · 68 · 372 |

Si noti che sulla **categoria** SIT migliora la testa (Q25 2→1) e **peggiora** la coda (Q75 19→21,
Q99 369→372): il guadagno medio non è uniforme.

**Strati (§6.1), con l'incrocio core/boundary che l'istruttoria tratta come un quinto strato ma è un
asse ortogonale:**

*A livello di item held-out:*

| | core | boundary | totale |
|---|---|---|---|
| vittorie | 38.782 | 12.086 | **50.868 (52,3%)** |
| neutri | 936 | 957 | **1.893 (1,9%)** |
| danni | 33.931 | 10.507 | **44.438 (45,7%)** |

*A livello di categoria vera (l'asse del Cat-MRR, cioè la metrica del capitolo 5):*

| | core | boundary | totale |
|---|---|---|---|
| vittorie | 27.884 | 7.491 | **35.375 (36,4%)** |
| neutri | 16.389 | 9.338 | **25.727 (26,5%)** |
| danni | 29.376 | 6.721 | **36.097 (37,1%)** |

**Conseguenze per il dimensionamento:**

1. **Nessuna cella è scarsa.** La più piccola (neutri × core, a livello di item) ha 936 casi. Un
   campione di 100 per cella è ampiamente sostenibile; il vincolo sarà il **budget di chiamate LLM**,
   non la disponibilità di richieste.
2. **«Neutri» è uno strato diverso a seconda del livello:** 1,9% a livello di item, 26,5% a livello di
   categoria. Il disegno deve dichiarare quale.
3. **I «danni» sono il 45,7% delle richieste** (37,1% sull'asse categoria). Non è uno strato di coda:
   è quasi metà del test. Chi progetta il campionamento deve sapere che *lo strato «il modulo ha
   peggiorato la raccomandazione» è grande quanto quello delle vittorie.* Questo è materiale
   direttamente rilevante per **C3**.
4. **Unità di ricampionamento:** le 97.199 richieste vengono da **6.040 utenti** (media 16,1
   richieste per utente). Un bootstrap sulle richieste sovrastima la precisione. È esattamente W5 del
   council precedente, e qui il rapporto è ancora più sfavorevole che nel capitolo 5.

---

## 8 · Fatti accertati che contraddicono o correggono l'istruttoria

| # | l'istruttoria dice | il disco dice | dove |
|---|---|---|---|
| 1 | §4.2: \|T\|=1 → 73.648, \|T\|=2 → 17.976, \|T\|=3 → 5.185 | seme 42: **73.649 · 17.980 · 5.182**. I numeri citati sono **medie su 5 semi** | `wi0d_probe.py:210` |
| 2 | §4.6: «65 dei 67 file di output non sono tracciati da git» | nelle sei directory delle sonde ci sono **125 file**, di cui **0 tracciati**. Anche i **6 script** che li producono sono non tracciati. Il conteggio «67» contava voci di directory, non file (`exp_s1` ne contiene 62) | `git ls-files`, `find` |
| 3 | §8(a): «output atteso in `docs/VERIFICA_griglia_e_steckb.md`» | **il file non esiste** | `ls docs/` |
| 4 | §11: repo in `~/Desktop/xsage-clean/` (idem il comando di lancio del brief) | il repo è in **`~/Downloads/xsage-clean/`**; `~/Desktop/xsage-clean` non esiste | `ls` |
| 5 | §3: «la riga che produce il ground truth — `reco_examples.py:70`» | quella riga produce il ground truth **solo per le richieste core**. Per il 24,2% boundary è **il vettore sbagliato** (§3 qui sopra) | `mind_prep.py:151`, `viability_probe.py:140` |
| 6 | §6.5: «il resto è numpy su matrici in cache» | **confermato, e sottostimato**: 0,02 ms/richiesta | misura §2 |
| 7 | §8(b): «nessuno sa se è mezz'ora o mezza giornata» | **mezz'ora**, quasi tutta di scrittura di codice | misura §1 |

Nessuno di questi punti 1–4 è grave in sé. Il punto 5 sì: è un errore che si propaga in silenzio se
qualcuno riusa `reco_examples.py:70` come modello.

---

## 9 · Tabella riassuntiva dei costi

| voce | ore-macchina | ore-uomo | dipendenze |
|---|---|---|---|
| record per richiesta, 1 dataset | **0,004** (14 s) | ~1 | nessuna |
| record per richiesta, 5 dataset | **0,02** (~70 s) | +0,5 | nessuna |
| ranghi BASE/SIT su tutto il test, 1 dataset | **0,001** (4,5 s) | — | record |
| campionamento stratificato | trascurabile | ~1 | record |
| ablazione + riordino, 500 richieste × 3 bracci | **trascurabile** (< 1 s) | ~2 | record |
| **client LLM, retry, snapshot, pinning, parsing di Ĉ** | — | **da scrivere da zero, nessuna riga esiste** | §5 |
| chiamate LLM (1.500 + ~150, stima §6.2/6.4) | dipende dal fornitore | — | client |

> **Il calcolo non è il collo di bottiglia di questo capitolo. Lo sono l'infrastruttura LLM
> (inesistente) e la definizione della metrica (§6).**

---

## 10 · Che cosa cambia per gli attori successivi

1. **Dossier A** non deve più chiedersi *se* PN/PS siano calcolabili. Deve giudicare un fatto misurato:
   **così come sono definite non separano una spiegazione fedele da una deliberatamente falsa**, e la
   critica C1 (tautologia) è meno urgente della scoperta che il braccio A1, anche in condizioni
   perfette, produrrebbe un numero basso per ragioni che non c'entrano con la fedeltà.
2. **Dossier B** ha già il Turno 0: il disegno **è** implementabile, il codice **regge**, e il punto in
   cui non regge è uno solo e preciso — `reco_examples.py:70` sulle boundary (§3, §8 punto 5).
3. **Dossier C** eredita un compito vincolante in più: verificare a pagina se la misura continua di §6
   coincide con la *Performance Shift* di Chen, Zhang & Wen 2022 §4.2.
4. **Dossier E** ha i numeri per dimensionare (§7): nessuna cella scarsa, 6.040 utenti per 97.199
   richieste, e lo strato «danni» al 45,7%.
5. **Dossier F** parte da un dato più duro del previsto: non esiste **nulla**, nemmeno `requests`, e
   `data/` è in `.gitignore` — le matrici su cui poggia tutto **non sono in git**.
6. **Persp_5 (executor)** può togliere dal preventivo l'intera voce «calcolo» e metterci sopra due voci
   che il preventivo non aveva: **scrivere l'infrastruttura LLM** e **ridefinire la metrica**.

---

## 11 · Verdetto del Turno 0, in tre righe

**Tutto ciò che il brief chiedeva di misurare come possibile, è possibile, ed è economico:** il record
per richiesta costa 14 secondi e 0,6 MB; l'ablazione e il riordino costano 0,02 ms per richiesta; il
caso boundary è già calcolato dal codice esistente in due punti e vale 7 MB su disco; le matrici ci
sono, per 7 backbone su 5 dataset.

**Ciò che non regge non è la fattibilità: è la metrica.** PN e PS come definite nel §6.3 sono
indefinite sul 94% delle richieste e, dove sono definite, assegnano all'insieme deliberatamente
sbagliato un punteggio indistinguibile da quello giusto.

**L'asse resta salvabile a costo zero di calcolo** passando alla misura continua di §6, che separa
0,600 da 0,005 con supporto sul 98,3% delle richieste — **a condizione** che il Dossier C confermi
sul PDF che quella misura ha già un nome in Chen, Zhang & Wen 2022 §4.2.
