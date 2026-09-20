# BRIEF A — risultati

**Eseguito:** notte fra il 20 e il 21 settembre 2026, non presidiato.
**Repo:** `/Users/lucaaliberti/Downloads/xsage-clean`, branch `exp/second-dataset-feasibility`.
**HEAD all'avvio:** `db608f8`.
**Regola 3 applicata in tutto il documento:** dove un numero su disco contraddice il brief, si riporta
il numero su disco e la discrepanza è elencata in fondo (§Discrepanze).

---

## A.0 — Igiene

### Cosa è entrato

Commit **`9a6dd58`**, `92 file, 13.468 inserzioni, 1.173.300 byte (1.145,8 KB)`.

| directory | file entrati | byte |
|---|---:|---:|
| `outputs_results/diagnostics/` | 11 | 29.709 |
| `outputs_results/probe_wi0b/` | 17 | 33.127 |
| `outputs_results/probe_wi0c/` | 13 | 37.009 |
| `outputs_results/probe_wi0d/` | 18 | 43.659 |
| `outputs_results/exp_e1/` | 4 | 32.602 |
| `outputs_results/exp_s1/` | 2 (i 60 `.npy` esclusi, vedi sotto) | 12.213 |
| `council_cap6_explainability/` | 1 | 24.580 |
| `scripts/diagnostics/` | 5 `.py` (i `.pyc` sono ignorati) | 97.213 |
| `scripts/exp/` | 3 `.py` | 24.148 |
| `scripts/validation/` | 4 `.py` | 45.300 |
| altri (CAP5/CAP6_NUMERI, conditional_prior×5, sweep_sensitivity×2, F1_fondamenta, 2 figure) | 14 | ~793.000 |

### La correzione al `.gitignore`

La regola `outputs_results/*.npy` **non è ricorsiva**: copre solo la radice di `outputs_results/`.
I 60 `.npy` in `exp_s1/parts/` (10) e `exp_s1/peruser/` (50) stavano in sottodirectory e sarebbero
entrati nel commit — **9,7 MB, il 97% del peso totale**.

Aggiunta `outputs_results/**/*.npy` (`.gitignore:5`). Verificato con `git check-ignore` su tutti e 60:
tutti ignorati. Verificato prima della modifica che **nessun `.npy` fosse già tracciato**
(`git ls-files '*.npy'` → 0 righe), quindi la regola non rimuove nulla dalla storia.

Sono rigenerabili da `scripts/exp/s1_concat.py`, che è ora sotto git.

### Scelta conservativa annotata (regola 1)

Il brief chiedeva di mettere sotto git le sei directory delle sonde più gli script. Erano untracked
anche i due CSV di settembre, `CAP5_NUMERI.md`, `CAP6_NUMERI.md`, `scripts/validation/*.py` e le due
figure — cioè il resto dei due mesi non committati. **Sono stati inclusi**: l'argomento del brief
(«se quella cartella si perde, il capitolo non è rigenerabile») vale identico per loro.

### `git clean`

Non eseguito, in nessuna forma.

### Resta fuori

`CLAUDE.md` — **0 byte**, file vuoto creato il 2 settembre. Non committato perché non contiene nulla.
Da scrivere o da cancellare: è una decisione, non un'omissione.

### Stato git di contorno, rilevato in ricognizione

- `main` è **70 commit indietro** rispetto al branch di lavoro e non ha commit unici: non è un riferimento valido.
- Il branch di lavoro era ed è allineato a `origin` (0/0). Il commit `9a6dd58` è **locale**: va pushato.
- **10 branch locali non hanno upstream** e non sono mai stati pushati: `abl/intent-constitutive`,
  `dpl/saopaulo-diagnosis`, `exp/b4-user-equity-tist`, `exp/projection-disambiguation-tist`,
  `exp/situational-risk-earlywarning`, `explore/cornac`, `fairness/positioning`,
  `feat/backbone-pluggability`, `feat/lastfm-onboarding`, `fix/boundary-structural`.
- Uno stash pendente (`stash@{0}`), contenuto: una riga di `.gitignore`. Irrilevante.

---

## A.1 — I due file di settembre

### A.1.1 `outputs_results/sweep_sensitivity.csv`

**2.920 righe di dati** (2.921 con header) × 16 colonne. Generato da `scripts/validation/sweep_sensitivity.py` (187 righe).

```
city,seed,backbone,phase,K,eps,kappa,gating,boundary_frac,macroCatMRR,CatMRR,HR20,NDCG20,Coverage,Gini,LT20
```

| colonna | distinti | valori |
|---|---:|---|
| `city` | 5 | kuairand, ml1m, nyc_tist, saopaulo, yelp |
| `seed` | 5 | 42–46 |
| `backbone` | 4 | AFM, B_blind, EASE, SASRec |
| `phase` | 2 | `kappa` (1.400 righe), `keps` (1.520 righe) |
| `K` | 5 | 3, 4, 5, 6, 8 |
| `eps` | 5 | 0.01, 0.02, 0.03, 0.05, 0.1 |
| `kappa` | 7 | 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5 |
| `gating` | 2 | double, single |
| `boundary_frac` | 266 | 0,0001 – 1,0 |
| `macroCatMRR` | 2.339 | 0,04196 – 0,34509 |
| `CatMRR` | 2.339 | 0,12441 – 0,76220 |
| `HR20` | 1.500 | 0,01874 – 0,32349 |
| `NDCG20` | 2.270 | 0,00903 – 0,13629 |
| `Coverage` | 1.195 | 0,04855 – 0,99850 |
| `Gini` | 2.354 | 0,46816 – 0,99297 |
| `LT20` | 2.227 | 0,00078 – 0,60990 |

**Griglia effettiva:** 5 dataset × 4 backbone × 5 semi = **100 celle**, tutte piene, in due fasi che
non si sovrappongono:

- **fase `kappa`** — varia `kappa` (7 valori, **incluso 0,0**) con `(K, eps)` **fissi al valore
  selezionato del dataset**: ml1m (5; 0,03) · nyc_tist (8; 0,02) · saopaulo (4; 0,03) ·
  kuairand (4; 0,01) · yelp (3; 0,05). Entrambi i gating.
- **fase `keps`** — varia `(K, eps)` su 15 combinazioni (16 per nyc_tist) con **`kappa` fisso a 0,25**
  e **solo gating `double`**.

**Intervalli di confidenza: NESSUNO.** `grep -n "ci_\|bootstrap\|quantile\|1\.96" scripts/validation/sweep_sensitivity.py`
→ zero match. Le colonne scritte sono le 9 chiavi di `sweep_sensitivity.py:142` più
`KEEP` di `sweep_sensitivity.py:46`; scrittura a `:145`. Sono **stime puntuali**: l'unica misura di
incertezza ricavabile è la dispersione fra i 5 semi.

### A.1.2 La domanda che conta — **SÌ, il gradiente si estende a 5 dataset × 4 backbone**

La colonna esiste e si chiama esattamente **`boundary_frac`**.

Il file **non contiene** una colonna di effetto: contiene `macroCatMRR` assoluto. L'effetto va costruito
come differenza contro il non-nudge, e le due fasi sembrano non incastrarsi — dove `boundary_frac`
varia (`keps`) `kappa` è fisso a 0,25 e manca il riferimento a `kappa=0`.

**Si incastrano, e il motivo è verificabile.** A `kappa=0` il nudge situazionale è azzerato, quindi il
punteggio è il backbone puro e **non dipende da K, ε o gating**. Verificato su tutte le **100 celle**
`(city, seed, backbone)`: il valore a `kappa=0` è identico sui due gating, **spread massimo 0,0 esatto**.
Quindi il `kappa=0` della fase `kappa` è baseline valida per **ogni** `(K, ε)` della fase `keps`.

Colonne necessarie, tutte presenti: `city`, `seed`, `backbone`, `phase`, `K`, `eps`, `kappa`,
`boundary_frac`, `macroCatMRR`. Join su `(city, seed, backbone)`; **1.520 righe `keps` su 1.520 trovano
la baseline, zero mancanti.**

**Gradiente dell'effetto per grado di ambiguità** — pendenza di `macroCatMRR(K,ε,κ=0,25) − macroCatMRR(κ=0)`
su `boundary_frac`, media delle pendenze per-seme (15–16 punti per seme, 5 semi):

| dataset | AFM | B_blind | EASE | SASRec |
|---|---:|---:|---:|---:|
| ml1m | −0,00611 | −0,00422 | **+0,01471** | +0,00398 |
| nyc_tist | −0,01907 | −0,02828 | **+0,02603** | −0,00115 |
| saopaulo | −0,02844 | −0,01531 | −0,01512 | −0,00400 |
| yelp | −0,00442 | −0,00534 | **+0,01458** | −0,00610 |
| kuairand | −0,00238 | −0,00103 | **+0,02190** | +0,00309 |

**14 celle su 20 hanno pendenza negativa** — l'effetto decade al crescere dell'ambiguità — e
**17 su 20 hanno tutti e 5 i semi concordi di segno**. L'eccezione è sistematica e non è rumore:
**EASE ha pendenza positiva in 4 dataset su 5**, cioè è l'unico backbone che *guadagna* dall'ambiguità.

**Deconfondimento.** `boundary_frac` è un esito di `(K, ε)`, non una manipolazione indipendente:
correla con `eps` a ρ = 0,784 e con `K` a ρ = 0,223. Rifacendo le pendenze **dentro ogni K** (varia solo ε,
19–20 fit per cella) il quadro non cambia: **stesso segno in 20 celle su 20**, magnitudini quasi identiche
(ml1m/AFM −0,00698 contro −0,00611; nyc_tist/B_blind −0,02806 contro −0,02828). Il gradiente non è un
artefatto della granularità del clustering.

**Tre limiti da dichiarare se si usa questo risultato:**
1. L'effetto è misurato a **κ = 0,25 fisso**, non al κ\* di ciascun dataset (ml1m ha κ\* = 0,50): è un
   gradiente al punto di lavoro sbagliato, non al punto di lavoro.
2. La fase `keps` ha **solo gating `double`**: nulla si può dire sul gating singolo.
3. **Nessun intervallo di confidenza nel file.** Qui si riporta la dispersione fra semi e la concordanza
   di segno, non un CI bootstrap. Non è la stessa cosa e non va scritto come se lo fosse.

Copertura: **5 dataset × 4 backbone**, contro i 3 dataset × 1 backbone di oggi.

### A.1.3 `outputs_results/conditional_prior.csv`

**200 righe di dati** × 13 colonne. Generato da `scripts/validation/conditional_prior.py` (349 righe).

```
city,seed,backbone,group,n_requests,n_cats_supported,macroCatMRR_sit,macroCatMRR_steck,delta,ci_lo,ci_hi,kappa_sit,kappa_steck
```

Griglia: 5 dataset × 4 backbone × 5 semi × 2 gruppi = **200 righe, piena**.
`n_cats_supported` ∈ {2, 6, 8, 9, 10, 16, 18, 39}; `delta` ∈ [−0,396889, +0,022451].

**Intervalli di confidenza: SÌ, bootstrap.** `ci_lo`/`ci_hi` in `COLS` a `conditional_prior.py:60`,
calcolati a `:238` (`boot_delta(cs, cb, t, nmac, np.random.default_rng(2024))`) e scritti a `:242`.
Protocollo dichiarato a `:15`: **1.500 resample a livello di richiesta, seme 42**. Percentili empirici,
non approssimazione normale (nessun `1.96` nel file).

### A.1.4 ON-PRIOR / OFF-PRIOR e la colonna `delta`

Definizione operativa a **`scripts/validation/conditional_prior.py:177-183`**:

```python
dom = Pu.argmax(1)
...
on  = (tm == dom[u]) & ~cold_req
off = (tm != dom[u]) & ~cold_req
```

documentata a `:8-9`:

- **ON-PRIOR** — `icm[i_test] == argmax(Pu[u_test])`: la categoria vera dell'item di test **è** la
  categoria dominante dell'utente. L'utente sta seguendo la propria abitudine.
- **OFF-PRIOR** — il complemento: l'utente si discosta dall'abitudine.

`Pu` è costruito **solo da `df_train`** (partizione leakage-free). `cold_req` esclude gli utenti senza
storia; in pratica non ne esiste nessuno nei cinque dataset, quindi ON ∪ OFF copre tutte le richieste.

**`delta` = `macroCatMRR_sit` − `macroCatMRR_steck`**, calcolato **dentro il gruppo** (`:242`):
quanto X-SAGE batte il profilo statico per-utente di Steck-b, separatamente sulle richieste abitudinarie
e su quelle fuori abitudine. Positivo = X-SAGE avanti.

Il file è già stato interpretato in `outputs_results/conditional_prior.md`: **H1 non confermata**,
PASS su 2 dataset su 5 contro i 3 richiesti. Il gate di ancoraggio del suo §5 era passato su tutte le 80 celle.

---

## A.2 — Ricerca degli script perduti — **TROVATI**

Gli script del Turno 0 (`t0_measure.py`, `t0_measure2.py`, `t0_measure3.py`, `t0_measure4.py`) **esistono
e sono integri.** `scripts/explain/` non esisteva: creata. **Non sono stati eseguiti.**

### Dove non erano

| luogo cercato | comando | esito |
|---|---|---|
| directory `scratchpad/` sul sistema | `find /Users/lucaaliberti -maxdepth 6 -type d -name scratchpad` | tutte **vuote** |
| per nome, ovunque nella home | `find /Users/lucaaliberti -name "t0_measure*"` | nessun risultato |
| scratchpad di sessione | `ls /private/tmp/claude-501/*/*/scratchpad` | 9 directory, **tutte a 0 file** |
| stash git | `git stash list` | 1 stash, contiene una riga di `.gitignore` |
| backup su Desktop | `ls ~/Desktop/backup/` | la directory non esiste |
| `~/Desktop/DEBUG_XSAGE/` | `ls` | 15 `.md` di audit, nessuno script |

`/private/tmp` viene azzerato al riavvio della macchina. La copia su disco era persa davvero.

### Dove erano

`Turno0_fattibilita.md:19` dichiara che gli script stavano nello scratchpad della sessione. Quella
sessione è **`5a2b49a6-047b-489b-ac20-87da8428826a`** (28 agosto 2026, 15:37–15:57). Gli script erano
stati creati con heredoc via Bash, quindi **il sorgente integrale è registrato nel transcript**:

```
~/.claude/projects/-Users-lucaaliberti-Downloads-xsage-clean/5a2b49a6-047b-489b-ac20-87da8428826a.jsonl
```

`grep -c t0_measure` → 15 occorrenze; quattro blocchi `tool_use` di tipo `Bash` con `command` di
7.812 / 5.955 / 4.090 / 3.782 caratteri, ciascuno un `cat > …/t0_measureN.py <<'PYEOF'`.
Estratti **verbatim** dai campi `input.command`, senza riscrivere una riga.

| file | righe | byte | compila | misura |
|---|---:|---:|:--:|---|
| `t0_measure.py` | 145 | 7.522 | ✓ | ancoraggi, costo di persistenza, core-only vs boundary-aware |
| `t0_measure2.py` | 93 | 5.587 | ✓ | supporto di PN/PS, strati su tutto il test, costo |
| `t0_measure3.py` | 61 | 3.729 | ✓ | PN/PS discrimina Ĉ vero da Ĉ falso? |
| `t0_measure4.py` | 53 | 3.420 | ✓ | quota di spostamento attribuibile a Ĉ |

Integrità verificata con `python -m py_compile` (compila, non esegue): 4 su 4.
Le docstring corrispondono una a una alle misure descritte nel brief.

### Conseguenza: **A.5 decade**

Il brief prevedeva A.5 («Ricostruzione della validazione delle metriche») *solo se* A.2 falliva.
A.2 non è fallito. La validazione delle metriche non va rifatta: va **rieseguita** quando serve,
a partire da codice che ora è sotto git.

### Il caveat che conta più del recupero

Gli script usano la forma **corretta** per il nudge — `mem @ b_z` a `t0_measure3.py:22`,
`t0_measure4.py:21`, `t0_measure.py:58` — e `t0_measure.py:59` tiene apposta anche
`nudge_core_only = b_z[kte]` per misurare la differenza fra le due. Su quello il Turno 0 è pulito.

**Ma l'insieme Ĉ è costruito core-only.** `t0_measure3.py:33`:

```python
top3 = np.argsort(-b_z, axis=1)[:, :3]; bot3 = np.argsort(b_z, axis=1)[:, :3]
...  gets = lambda r: set(top3[int(kte[r])].tolist())
```

cioè le top-3 della **riga `b_z[k]` della situazione principale**, non le top-*m* del vettore
`mem @ b_z` **della richiesta**, che è ciò che specifica A.5. Sulle richieste core coincidono;
sulle boundary (**24,2% su ml1m**) no.

Le due definizioni non sono intercambiabili. I numeri del Turno 0 sono validi **per la definizione
core-only con cui sono stati prodotti** e vanno citati così. Se B.2 usa «la quota di attribuzione di
A.5», deve dichiarare quale delle due sta usando. Registrato in `scripts/explain/README.md`.

---

## A.3 — Il record per richiesta

`scripts/explain/persist_records.py`. Sostituisce la selezione a cascata di
`scripts/yelp/reco_examples.py`, che calcola queste stesse quantità per 60.000 richieste e **ne salva
cinque**. Qui si salva ogni richiesta di test della griglia base, senza filtri.

### Due riusi che tolgono ore di calcolo

**1. L'assegnazione situazionale non dipende dal backbone.** `build_descriptor`, `select_K`,
`select_eps`, `fit_rough_kmeans`, `_assign`, `b_z` e quindi `mem` e `nudge` dipendono **solo** da
(dataset, seme): in `results_record.py:225-233` sono infatti calcolati **fuori** dal ciclo sui
backbone, che cambia soltanto la funzione di punteggio e κ. Si calcola una volta per (dataset, seme)
e si riusa sui 4 backbone: **20 calcoli invece di 60**.

**2. I ranghi per richiesta erano già su disco.** `outputs_results/cache/raw_<ds>.npz` contiene
`<backbone>|<metodo>|<seme>|{rk,catrk,g,gc,tk50}` per **7 backbone × 3 metodi × 5 semi**, su tutti e
5 i dataset, più le chiavi `_shared|{u,tm,icm,nmac,nI,G1,Pu}`. Verificato: **zero celle mancanti**
sulla griglia base. I ranghi non si ricalcolano — si leggono. Era la parte cara.

### Il costo vero — il brief sbaglia di 7× su ml1m

Il brief stima «circa 22 secondi per dataset», sommando da `cost_ml1m.txt` descrittore 8,25 +
k-means 13,89 + bias 0,09 = 22,23. **Quella somma salta i due stadi dominanti**, che sono nello
stesso file: selezione K **72,47 s** e selezione ε **73,64 s**. Il file dichiara `FIT totale 168,34 s`.

Vince il disco. Tempo **misurato** per (dataset, seme):

| dataset | atteso dal brief | `FIT totale` su disco | misurato (media 5 semi) |
|---|---:|---:|---:|
| ml1m | 22 s | 168,34 s | **~134 s** |
| nyc_tist | 22 s | 30,85 s | **~35 s** |
| saopaulo | 22 s | 31,15 s | ~32 s |

I due dataset piccoli tornano; ml1m è più veloce del file di costo ma **sei volte** la stima del
brief. Costo totale reale di A.3: **~18 minuti**, non «22 s per dataset».

### Il punto tecnico — quantificato

Il brief avverte che `reco_examples.py:70` usa `nudge = b_z[kte]` mentre la produzione usa `mem @ b_z`,
e che sulle boundary i due divergono. Ora è misurato. Entrambe le forme sono persistite in colonne
separate (`nudge_vector` e `nudge_core_only`) più la differenza massima per riga.

| cella | `bfrac` | core: max\|Δ\| | boundary: mediana di max\|Δ\| | boundary: max | **boundary con top-1 diverso** |
|---|---:|---:|---:|---:|---:|
| ml1m seme 42 | 0,24229 | **0,00e+00** | 1,6258 | 2,7095 | **91,2%** |
| ml1m seme 43 | 0,24236 | 0,00e+00 | 1,6272 | 2,6981 | 91,1% |
| nyc_tist seme 42 | 0,25681 | 0,00e+00 | 1,2566 | 2,2842 | 67,9% |
| nyc_tist seme 46 | 0,35403 | 0,00e+00 | 1,4271 | 2,7418 | 55,6% |

Due letture:

- **Sulle richieste core le due forme sono identiche bit a bit** (differenza massima esattamente zero,
  non «entro tolleranza»): `mem` è one-hot, come atteso. Chi ha usato `b_z[k]` sulle core non ha
  sbagliato nulla.
- **Sulle boundary il 91,2% delle richieste ha una categoria dominante DIVERSA** fra le due forme, su
  ml1m. Non è uno scarto numerico: è un'altra risposta alla domanda «quale categoria sta spingendo
  questa raccomandazione». Combinato con `bfrac`: usare `b_z[k]` nominerebbe il driver sbagliato su
  **0,24229 × 0,912 = 22,1% di tutte le richieste di test di ml1m**.

Una spiegazione costruita su `b_z[k]` sarebbe quindi sbagliata su circa una richiesta su cinque, e
sbagliata in modo invisibile — la frase resta plausibile, cita solo la categoria che non c'entra.

### Composizione (ml1m, seme 42, B_blind)

`|T|` — dimensione dell'insieme di situazioni: 1 → 73.649 · 2 → 17.980 · 3 → 5.182 · 4 → 388.

| strato | boundary | core |
|---|---:|---:|
| `harm` | 6.721 | 29.376 |
| `neutral` | 9.338 | 16.389 |
| `win` | 7.491 | 27.884 |

Le boundary **non** sono concentrate nelle vittorie: sono sovra-rappresentate nei `neutral`
(9.338 su 25.727 = 36,3%, contro un `bfrac` del 24,2%). Il campione di `reco_examples.py`, che le
scartava tutte alla riga 75, non era solo più piccolo: era **sbilanciato**.

### Verifica finale di A.3

**60 celle su 60 scritte**, 36 MB in parquet zstd (1,25–1,30 MB per cella su ml1m, 0,28 su nyc_tist,
0,33 su saopaulo). Le colonne sono quelle del brief più `nudge_core_only`, `nudge_maxabs_diff`,
`rank_base_item` e `rank_sit_item`.

**Gate di ancoraggio — PASSATO.** Ricalcolato dai record persistiti, `CatMRR` di `B_blind|SIT|42`:

| dataset | atteso | ricalcolato | differenza |
|---|---:|---:|---:|
| ml1m | 0,38479 | 0,384789 | 1,04e-06 |
| nyc_tist | 0,34162 | 0,341624 | 4,35e-06 |
| saopaulo | 0,40045 | 0,400448 | 2,35e-06 |

Le differenze sono **sotto 5e-06**, cioè il solo errore di memorizzazione: `results_record.csv`
conserva i valori già arrotondati a 5 decimali. È la stessa osservazione fatta dal run di
`conditional_prior` dell'11 settembre (massimo osservato lì: 4,9e-06). Per questo il gate confronta
la **differenza** contro una tolleranza di 5e-5 e non i valori arrotondati: `round(v,4)==round(rif,4)`
dà un falso mismatch quando il valore cade sul bordo di arrotondamento.

`bfrac` ricalcolato su ml1m è **0,24229 / 0,24236 / 0,24226 / 0,24229 / 0,24226** sui cinque semi,
identico a `outputs_results/explain/cost_ml1m.txt` e a `sweep_sensitivity.csv` (0,2423). La catena
di misura è ancorata a monte, non solo a valle.

---

## A.6 — Corpus stratificato

`scripts/explain/build_corpus.py`. **60 celle scritte, 1,7 MB, 29.853 richieste in totale.**

| voce | valore |
|---|---|
| seme di campionamento | **20260921**, deterministico per cella |
| obiettivo per strato | 100 |
| **strati sotto quota** | **nessuno**, in nessuna delle 60 celle |
| unione media per cella | 497,6 richieste |
| sovrapposizioni medie | 2,4 |

### Il seme doveva essere fisso, e per poco non lo era

Il brief chiede che il campione sia **lo stesso per tutti e tre i bracci LLM**. La prima stesura
derivava il seme di cella da `hash((seed, ds, bk, sd))`: `hash()` su stringhe in Python è
**randomizzato per processo** (`PYTHONHASHSEED`), quindi avrebbe prodotto un campione diverso a ogni
rilancio — e diverso fra il braccio A, il B e il C, che è esattamente il confronto che il disegno
deve reggere. Sostituito con `sha256` dei campi: verificato che due chiamate indipendenti diano lo
stesso insieme di id.

### Filtri rimossi

Tutti e cinque quelli di `scripts/yelp/reco_examples.py`, elencati riga per riga nella docstring:
`:75` (solo core), `:77` (`nudge <= 0`), `:81` (`rb < 4`), `:84` (`rs > 12 or rs >= rb`),
`:86` (massimo guadagno per situazione).

### La sovrapposizione degli strati, dichiarata

`{win, neutral, harm}` e `{boundary, core}` sono **due partizioni della stessa popolazione**, non
cinque insiemi disgiunti: una richiesta può essere insieme `win` e `boundary`. Si campionano 100 per
strato in modo indipendente e si tiene l'unione, con una colonna di flag per strato — così ogni
strato ha i suoi 100 pieni e le sovrapposizioni sono contate (`sovrapposti`, in media 2,4 per cella)
invece di essere nascoste da una deduplicazione silenziosa.

### Popolazioni disponibili per strato

| dataset | `win` | `neutral` | `harm` | `boundary` | `core` |
|---|---|---|---|---|---|
| ml1m | 10.231–35.376 | 25.716–68.644 | 18.324–36.110 | 23.547–23.557 | 73.642–73.652 |
| nyc_tist | 3.950–7.240 | 1.617–7.277 | 6.451–10.984 | 3.415–6.267 | 11.435–14.287 |
| saopaulo | 5.507–11.019 | 5.610–11.579 | 6.295–8.917 | 4.111–6.192 | 17.629–19.710 |

Il minimo assoluto su tutta la griglia è **1.617** (`neutral`, nyc_tist): sedici volte la quota. Il
margine è ampio ovunque, quindi la quota di 100 può essere alzata senza rifare nulla se B.2 lo chiede.

---

## A.4 — Equità per situazione

`scripts/fairness/situational_qos.py` (la directory `scripts/fairness/` non esisteva: creata).
Copertura: **griglia base 3 × 4 × 5** come risultato principale, **griglia piena 5 × 7 × 5** in
appendice, per i tre metodi `BASE` / `SIT` / `Steck-b`. Righe prodotte: 525 (ml1m, K=5) ·
840 (nyc_tist, K=8) · 420 (saopaulo, K=4). **Zero celle `low_support`** su tutta la griglia base:
la soglia (< 8 utenti o < 20 richieste) non scarta nulla, nessuna cella è stata cancellata.

### Gate di identità — PASSATO, e in quale forma

Il brief chiede che «la media micro pesata per richieste sulle situazioni riproduca esattamente la
macro-Cat-MRR aggregata». Preso alla lettera è **un test vacuo**: la macro è una media *non pesata*
sulle categorie e non si decompone per situazione, quindi nessuna ricombinazione pesata delle
macro-per-situazione può riprodurla. Applicata la regola 8 — controllare la vacuità prima di
eseguire — il gate è verificato nelle due forme che sono esatte:

| forma | identità | esito |
|---|---|---|
| **micro** | Σ_s n_s · media_s(cm) / N = `CatMRR` aggregato | ✅ |
| **macro ricomposta** | ricombinazione per (situazione, categoria) → `macro-Cat-MRR` aggregato | ✅ |

Forma micro contro i valori pubblicati, `B_blind|SIT|42`:

| dataset | pubblicato | ricomposto | differenza |
|---|---:|---:|---:|
| ml1m | 0,38479 | 0,384789 | 1,04e-06 |
| nyc_tist | 0,34162 | 0,341624 | 4,35e-06 |
| saopaulo | 0,40045 | 0,400448 | 2,35e-06 |

Un controllo indipendente: la ricomposizione pesata del **contrasto** `SIT − BASE` su ml1m/B_blind dà
**+0,02517**, cioè esattamente il `delta_l1` pubblicato in `results_record.csv` per `CatMRR`. La
partizione per situazione è completa e correttamente pesata.

### Due scelte annotate

**1. Le etichette di situazione non sono confrontabili fra semi.** Il clustering è ri-stimato a ogni
seme (`results_record.py:230`, `fit_rough_kmeans(seed=seed)`): la situazione 3 del seme 42 non è la
situazione 3 del seme 43, gli indici sono arbitrari. Quindi la tabella per situazione è riportata
**per seme**, senza mai mediare un indice fra semi, e il CI a due livelli è applicato solo alle
sintesi **senza etichetta** — minimo rawlsiano, quartile basso, divario, media — che sono statistiche
d'ordine e quindi confrontabili.

Ha una conseguenza pratica: il divario va calcolato **per seme e poi mediato**, non mediando prima le
macro-per-situazione. L'ordine sbagliato dà su ml1m/B_blind −0,00445 → −0,00981 invece del corretto
−0,00918 → −0,01421: stessa direzione, magnitudini diverse.

**2. Il costo del bootstrap.** Il pattern di `wi0d_probe.py:113-122` è un doppio ciclo Python su
B × semi; sulla griglia piena sarebbe stato ~1e11 flop per dataset. Riscritto come una matrice di
pesi (B × n_utenti) moltiplicata per l'aggregato per (utente, situazione, categoria): una chiamata
BLAS invece di B matvec, con i pesi estratti da una multinomiale (che *è* il bootstrap a cluster).
Misurato: 0,27 s per seme. Risultato identico, tempo da ore a secondi.

### Correzione della soglia dei CI

`wi0d_probe.py:114` scarta una cella se il **minimo** di utenti fra i semi è < 10; qui si usa la
**media**. Su tutta la griglia base il CI a due livelli è calcolato su **36 celle su 36, zero NaN**.

### Il risultato — SIT peggiora il divario in 9 celle su 12

Divario rawlsiano (minimo − media sulle situazioni), media dei divari per-seme:

| dataset | backbone | BASE | SIT | SIT − BASE | |
|---|---|---:|---:|---:|---|
| ml1m | B_blind | −0,00918 | −0,01421 | −0,00502 | peggiora |
| ml1m | EASE | −0,01181 | −0,01702 | −0,00520 | peggiora |
| ml1m | AFM | −0,09610 | −0,09520 | +0,00090 | migliora |
| ml1m | SASRec | −0,05993 | −0,05718 | +0,00275 | migliora |
| nyc_tist | B_blind | −0,03516 | −0,04540 | −0,01023 | peggiora |
| nyc_tist | EASE | −0,03874 | −0,05061 | −0,01188 | peggiora |
| nyc_tist | AFM | −0,02592 | −0,05254 | −0,02662 | peggiora |
| nyc_tist | SASRec | −0,04105 | −0,06548 | −0,02443 | peggiora |
| saopaulo | B_blind | −0,02076 | −0,03034 | −0,00958 | peggiora |
| saopaulo | EASE | −0,01881 | −0,04663 | −0,02782 | peggiora |
| saopaulo | AFM | −0,03258 | −0,06967 | −0,03709 | peggiora |
| saopaulo | SASRec | −0,04742 | −0,04600 | +0,00143 | migliora |

**Il pattern è per dataset, non per famiglia di backbone.** Su ml1m il segno si divide fra ciechi al
contesto (peggiora) e consapevoli (migliora); su nyc_tist peggiora su tutti e quattro; su saopaulo su
tre su quattro. La lettura «X-SAGE distribuisce meglio dove guadagna meno» regge **solo su ml1m** e
non va generalizzata.

Dove i CI a due livelli **non si sovrappongono** fra BASE e SIT — il test più conservativo, perché
confronta i due intervalli separati e non la differenza appaiata — il peggioramento è netto su
**saopaulo/AFM** (BASE [−0,04303 · −0,02282] contro SIT [−0,08119 · −0,05713]) e
**saopaulo/EASE** (BASE [−0,02874 · −0,01475] contro SIT [−0,05851 · −0,03443]). Altrove gli
intervalli si toccano, il che con questo test non significa assenza di effetto.

### La media non pesata delle macro-per-situazione

Terza quantità, negativa in **11 celle su 12** (da −0,00124 a −0,06557; unica eccezione
saopaulo/AFM, +0,00675). Non contraddice il guadagno aggregato: smette di pesare per dimensione
**sia le situazioni sia le categorie**, quindi misura una cosa diversa. Dice però una cosa precisa:
**il guadagno aggregato di X-SAGE è concentrato nelle situazioni e nelle categorie grandi**, non
distribuito su tutte.

### Steck-b dentro le situazioni

Su **nyc_tist e saopaulo il profilo statico per-utente serve le situazioni molto più uniformemente di
entrambi gli altri metodi** — macro-per-situazione 0,306 contro 0,208 di BASE su nyc_tist/AFM,
0,319 contro 0,211 su saopaulo/AFM. Su ml1m sono alla pari (0,131 contro 0,130 su B_blind).

Non è una sorpresa isolata: è coerente con `conditional_prior.md` dell'11 settembre, che ha trovato
Steck-b avanti sull'aggregato negli stessi due dataset. Qui si aggiunge che lo è anche **dentro** le
situazioni, cioè sul terreno dove X-SAGE dovrebbe vincere per costruzione.

**Nessun p-value sul confronto lens-KL ↔ ΔQoS**, come da brief: è descrittivo.

