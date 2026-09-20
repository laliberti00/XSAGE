# Capitolo 6 (trustworthiness) — Estratto numerico verificato

**Generato:** 2026-08-27 · **Natura:** sola estrazione. Nessun esperimento, nessun ricalcolo,
nessuna modifica al metodo.
**Regola:** ogni numero accanto alla sua fonte (file + riga/colonna). Cio' che non si trova e'
`[NON TROVATO]` con la ricerca eseguita. **Il disco ha sempre ragione**: dove un valore su disco
contraddice il brief, si riporta quello del disco e si segnala.

---

## 0. Inventario delle fonti

**Tutte le sonde hanno prodotto output, e tutti gli output sono su disco.** Nessuna e' assente.

> ⚠️ **Su 67 file di output, 65 sono NON TRACCIATI da git.** Gli unici due tracciati sono
> `outputs_results/lt_convention_probe.csv` e `scripts/lt_convention_probe.py`.
> Anche i quattro script delle sonde (`scripts/diagnostics/*.py`, 102 KB complessivi) e i due
> di esperimento (`scripts/exp/*.py`) sono non tracciati. Un `git clean -fd` li cancellerebbe.

### Output per sonda

| directory | voci | contenuto |
|---|---|---|
| `outputs_results/diagnostics/` | 11 | WI-0: CHECK A (varianza), CHECK B (identificabilita'), CHECK C (stabilita') + `WI0_SUMMARY.md` |
| `outputs_results/probe_wi0b/` | 17 | CHECK A2 (ristretto), CHECK C2 (orizzonti + finestra), CHECK D (minimizzazione) + summary |
| `outputs_results/probe_wi0c/` | 13 | E1 per distanza, E2 stratificato, E3 regressione + summary |
| `outputs_results/probe_wi0d/` | 18 | boundary, gate gamma, regressione, audit nomenclatura e repo + summary |
| `outputs_results/exp_e1/` | 4 | curva di troncamento, guardia di permutazione, kappa riselezionato, ancoraggi |
| `outputs_results/exp_s1/` | 4 (+2 dir) | varianti di concatenazione, ARI, per-utente, partizioni |

### Log delle esecuzioni

`logs/`: `wi0_ml1m.log` · `wi0_all.log` · `wi0b_ml1m.log` · `wi0b_poi.log` · `wi0c_ml1m.log` ·
`wi0c_poi.log` · `wi0d.log` · `e1_ml1m.log` · `s1_ml1m.log` — tutti non tracciati.

### Script che li producono

`scripts/diagnostics/{viability_probe,wi0b_probe,wi0c_probe,wi0d_probe}.py` ·
`scripts/exp/{e1_truncation,s1_concat}.py` — **tutti non tracciati**.

### Copertura per dataset

| sonda | ml1m | nyc_tist | saopaulo | altri |
|---|---|---|---|---|
| WI-0 (CHECK A, B) | ✓ | ✓ | ✓ | — |
| WI-0 (CHECK C) | ✓ | — | — | ml1m-only per disegno |
| WI-0b (A2, C2) | ✓ | ✓ | ✓ | — |
| WI-0b (CHECK D) | ✓ | — | — | ml1m-only (solo ml1m ha demografia) |
| WI-0c (E1/E2/E3) | ✓ | ✓ | ✓ | — |
| WI-0d | ✓ | ✓ | ✓ | — |
| **E-1 (troncamento)** | **✓** | **—** | **—** | vedi §1.5 |
| S-1 (concatenazione) | ✓ | — | — | — |

*(La ricerca iniziale per pattern aveva mancato i file di WI-0, perche' si chiamano `checkA_…`
senza underscore e `WI0_…` senza trattino: l'elenco sopra viene dal listato diretto delle
directory.)*

---

## 1. I cinque punti aperti

### 1.1 — L'occupazione in CHECK D: **PRODOTTA**

Il risultato esiste ed e' completo. Fonte: `outputs_results/probe_wi0b/check_d_minimization_ml1m.csv`,
blocco `occupation`, 12 righe. **Nessun rilancio necessario.**

| attributo | classi | baseline di maggioranza | (a) etichetta situazione | (b) istogramma situazioni | (c) storia item grezza |
|---|---|---|---|---|---|
| gender | 2 | **71.705%** | 71.705% · bal 0.5000 | 72.599% · bal 0.5496 | 78.013% · bal 0.7190 |
| age | 7 | **34.702%** | 34.702% · bal 0.1429 | 34.685% · bal 0.1433 | 47.301% · bal 0.3825 |
| **occupation** | **21** | **12.566%** | **13.262% · bal 0.0521** | **13.907% · bal 0.0583** | **15.033% · bal 0.0939** |

Δ rispetto alla baseline: gender **+0.0000 / +0.0089 / +0.0631** · age **+0.0000 / −0.0002 /
+0.1260** · occupation **+0.0070 / +0.0134 / +0.0247**.

Validazione incrociata a **5 fold** stratificati su tutti e tre. Con tre attributi protetti la
tabella e' difendibile, non solo indicativa.

*(Nota: su occupation l'etichetta di situazione da' +0.70 punti sopra la maggioranza, non
esattamente zero come su gender e age — va riportato cosi', non arrotondato a zero.)*

### 1.2 — La colonna JS: **TROVATA, 105 righe, completa**

Fonte: `outputs_results/results_record.csv`, righe con `metric == "JS"`.
**105 = 5 dataset × 7 backbone × 3 metodi** — la griglia e' piena, nessuna cella mancante.

Indicizza `dataset × backbone × metodo`, non "dataset × backbone × qualcos'altro".
La divergenza di Jensen-Shannon e' fra il **prior storico di categoria dell'utente** e
l'istogramma di categoria della top-20 (`results_record.py:55-58`, funzione `js`).

Media sui 7 backbone, per dataset:

| dataset | BASE | SIT | Steck-b |
|---|---|---|---|
| ml1m | 0.26983 | **0.30642** | 0.25910 |
| nyc_tist | 0.33783 | **0.37479** | 0.25737 |
| saopaulo | 0.29411 | **0.30702** | 0.20535 |
| yelp | 0.10629 | **0.11925** | 0.12012 |
| kuairand | 0.30953 | **0.32190** | 0.25865 |

**SIT ha JS piu' alta di BASE su tutti e cinque i dataset**, e piu' alta di Steck-b su quattro
su cinque (eccezione: yelp, dove Steck-b e' 0.12012 contro 0.11925). E' il comportamento atteso
e dichiarato: il metodo si scosta dal gusto abituale dell'utente per servire lo scopo del
momento, mentre Steck-b — che calibra proprio sull'utente — ha la JS piu' bassa. **E' il costo,
non un difetto**, ed e' misurato su tutta la griglia.

### 1.3 — La figura KL per situazione: **i dati ci sono, su 10 dataset**

Granularita' **per singola situazione**, in `outputs_results/explain/situation_profiles_<ds>.json`,
campo `lens_KL` di ogni voce di `situations`:

| dataset | K | lens_KL per situazione |
|---|---|---|
| **ml1m** | 5 | 0.215 · 0.072 · **0.653** · **0.016** · 0.206 |
| nyc_tist | 8 | 0.209 · 0.200 · 0.518 · 0.063 · 0.180 · 0.137 · 0.176 · 0.040 |
| saopaulo | 4 | 0.270 · 0.057 · 0.172 · 0.193 |
| yelp | 3 | 0.005 · 0.036 · 0.000 |
| kuairand | 4 | 0.018 · 0.060 · 0.029 · 0.230 |
| bangkok | 4 | 0.037 · 0.048 · 0.200 · 0.910 |
| istanbul | 5 | 0.058 · 0.012 · 0.078 · 0.192 · 0.097 |
| tokyo_tist | 4 | 0.075 · 0.229 · 0.196 · 0.018 |
| tsmc_nyc | 7 | 0.134 · 0.294 · 0.174 · 0.153 · 0.131 · 0.273 · 0.352 |
| tsmc_tky | 4 | 0.151 · 0.066 · 0.225 · 0.159 |

Esiste anche un aggregato per (seme, backbone): colonna **`lensKLmax`**, tredicesima di
`battery_bfull_<ds>.csv`.

**Due script generano gia' la figura**: `scripts/yelp/plot_explain.py` →
`explain/fig2_inequity_lens_ml1m.{pdf,png}` e `scripts/yelp/plot_paper_figs.py` →
`paper/figs/fig_inequity_lens_ml1m.{pdf,png}` (versione Elsevier). **La figura non e' stata
rigenerata**, come da vincolo: qui si dichiara solo che i dati e lo script esistono.

Osservazione utile: **yelp ha lens_KL fra 0.000 e 0.036**, cioe' le sue tre situazioni sono
quasi indistinguibili dalla distribuzione globale — il che e' coerente con la sua saturazione al
76% su una sola categoria. **bangkok ha un valore di 0.910**, il massimo su tutti i dataset.

### 1.4 — Stato delle sonde: **tutte PRESENTI**

| sonda | stato | percorso |
|---|---|---|
| CHECK A (varianza) | **PRESENTE** | `outputs_results/diagnostics/checkA_variance_{ml1m,nyc_tist,saopaulo}.csv` |
| CHECK B (identificabilita') | **PRESENTE** | `outputs_results/diagnostics/checkB_identifiability_*.csv` (3 dataset) |
| CHECK C (stabilita') | **PRESENTE** | `outputs_results/diagnostics/checkC_stability_ml1m.csv` (ml1m per disegno) |
| CHECK A2 | **PRESENTE** | `probe_wi0b/check_a2_{restricted,per_situation}_*.csv` (3 dataset) |
| CHECK C2 | **PRESENTE** | `probe_wi0b/check_c2_{entropy_horizons,window}_*.csv` (3 dataset) |
| CHECK D | **PRESENTE** | `probe_wi0b/check_d_minimization_ml1m.csv` (ml1m per disegno) |
| WI-0b | **PRESENTE** | `probe_wi0b/wi0b_summary.md` + 3 JSON |
| WI-0c | **PRESENTE** | `probe_wi0c/` — 13 file, 3 dataset |
| WI-0d | **PRESENTE** | `probe_wi0d/` — 18 file, 3 dataset |
| **E-1** | **PARZIALE** | solo **ml1m** — vedi 1.5 |
| S-1 | **PRESENTE** | `exp_s1/` — solo ml1m (previsto dal disegno) |

**Nessuna sonda va rilanciata.** Il piano del capitolo resta «una settimana di scrittura», senza
rilanci — con l'unica riserva sulla copertura di E-1.

### 1.5 — E-1: **solo ml1m**

Tutti e quattro i file di `outputs_results/exp_e1/` contengono **`dataset = ['ml1m']`** e nulla
piu': `e1_curve` (160 righe = 8 valori di n × 5 semi × 4 bracci), `e1_kappa_reselected` (80),
`e1_permutation_guard` (40), `e1_anchors` (4). **Zero file per altri dataset.**

> **La frase nella tesi deve essere «su MovieLens-1M», non «sistematicamente».**

Ragione tecnica, non dimenticanza: E-1 tronca la storia dell'utente sfruttando il fatto che il
punteggio di EASE e' **lineare** nel vettore di interazioni (`s_u = r_u · B`), l'unico backbone
per cui il troncamento non richieda riaddestramento. L'estensione ad altri dataset e' possibile
— richiede di ricostruire `B` per ciascuno — ma non e' stata eseguita.

---

## 2. L'estratto

### 2.1 — Precondizione: la situazione e' distinguibile dall'utente?

**Fonte:** `outputs_results/diagnostics/checkB_identifiability_<ds>.csv` (18-19 righe per
dataset) e `_wi0_<ds>.json`.

**Configurazione comune a tutte le colonne** — la stessa per tutti e tre i dataset:
backbone **B_blind** (`viability_probe.py:119`, `D0["sb"]`, cioe' BPR), split **test**,
metrica via `eval_kappa.cat_mrr`, seme 42.

| | ml1m | nyc_tist | saopaulo |
|---|---|---|---|
| K | 5 | 8 | 4 |
| ε | 0.03 | 0.02 | 0.03 |
| κ\* | 0.50 | 0.50 | 0.25 |
| richieste (n) | 97.199 | 17.702 | 23.821 |
| categorie | 18 | 10 | 10 |
| colonne di contesto | 5 | 6 | 6 |

#### R² della regressione situazione ~ utente

| | mediano | Q1 | Q3 | aggregato (pooled) |
|---|---|---|---|---|
| ml1m | **0.056545** | 0.011724 | 0.158982 | 0.003863 |
| nyc_tist | **0.099866** | 0.021158 | 0.276594 | 0.041629 |
| saopaulo | **0.098918** | 0.024648 | 0.256324 | 0.034833 |

Il vettore situazionale spiega fra il **5,7% e il 9,9%** della varianza di quello per-utente.
L'aggregato e' molto piu' basso del mediano (0,4–4,2%): l'allineamento c'e' per singole
richieste ma **non sopravvive all'impilamento**, cioe' non e' una relazione sistematica.

#### Coseno fra i due vettori

| | mediano | Q1 | Q3 |
|---|---|---|---|
| ml1m | **+0.051475** | −0.163727 | +0.302370 |
| nyc_tist | **+0.212320** | −0.062146 | +0.506326 |
| saopaulo | **+0.206510** | −0.084234 | +0.448612 |

**Su tutti e tre il primo quartile e' negativo**: per un quarto delle richieste i due vettori
puntano in direzioni opposte.

#### Flip-rate

| | argmax situazione ≠ argmax utente | top-1 di categoria cambia (SIT vs BASE) |
|---|---|---|
| ml1m | **91.858%** | **32.501%** |
| nyc_tist | **73.528%** | **51.796%** |
| saopaulo | **77.092%** | **29.130%** |

Denominatori: 97.199 · 17.702 · 23.821 richieste di test.
La prima colonna dice che la categoria piu' favorita dalla situazione e' **diversa** da quella
piu' favorita dal profilo utente in tre casi su quattro o piu'. La seconda dice che
l'effetto arriva davvero in cima alla lista in un caso su tre (uno su due su nyc_tist).

#### R² del nudge ~ contesto (identificabilita')

| | R²(nudge su categoria vera ~ c̃) | R²(norma del nudge ~ c̃) | colonne c̃ |
|---|---|---|---|
| ml1m | **0.031849** | 0.003143 | 5 |
| nyc_tist | **0.009354** | 0.020199 | 6 |
| saopaulo | **0.075283** | 0.009987 | 6 |

Il contesto osservabile spiega **fra lo 0,9% e il 7,5%** del termine additivo: il nudge non e'
una ricodifica delle feature di contesto, e i pesi additivi sono identificabili.

Correlazioni di Pearson per singola colonna di contesto contro la norma del nudge — la massima
in valore assoluto e' **−0.1268** (nyc_tist, colonna c0); tutte le altre stanno sotto 0.091.

---

### 2.2 — La spiegazione: che cosa e' esattamente, e quanto ne abbiamo

#### La riga che produce i driver

`scripts/yelp/reco_examples.py`, riga **70** (nel blocco 60-95):

```python
nudge = b_z[kte]   # core: r=1 → nudge della situazione assegnata
```

e la riga **82** che lo applica:

```python
sit = base + kap * gam_te[r] * nudge[r][icm]
```

Questa e' **tutta** la spiegazione. `nudge[r]` e' un vettore di lunghezza `n_macros` che dice,
per ogni categoria, di quanto la situazione riconosciuta sposta il punteggio. Non e' una stima
post-hoc del contributo: e' **il termine stesso** che entra nella somma. Fedelta' per
costruzione, nel senso stretto — non c'e' un modello surrogato da cui possa divergere.

Attenzione al condizionamento: `nudge = b_z[kte]` prende il **core label**. Per le richieste di
boundary (24,2% su ml1m) il nudge effettivo e' la media pesata sulle appartenenze, ed e'
smorzato da γ = 1/|T|. Lo script degli esempi salta le richieste di boundary
(`if bool(isbte[r]): continue`, riga 75).

#### Le due matrici che costituiscono il modulo

Da `outputs_results/explain/cost_ml1m.txt` (riga «parametri del modulo»):

| oggetto | forma | numeri |
|---|---|---|
| `b_z` — bias per situazione × categoria | **5 × 18** | 90 |
| prototipi rough-k-means — situazione × dimensioni di v | **5 × 23** | 115 |
| **totale** | | **205** |

Le 23 dimensioni di `v = [c̃ ‖ e]` sono **5 di contesto + 18 di intento** (una per macro-genere).
Le 5 colonne di contesto sono le funzioni di contributo stimate su
`c_hour, c_dow, c_isweekend, c_month, intent_last_cat_idx`
(`mind_prep.MIND_ATTRS`, riga 23 — ml1m **non** usa geohash).
Il file di costo conferma tutto: `K=5 ε=0.03 macro=18 dim(v)=23`.

#### Le 18 categorie

Da `situation_profiles_ml1m.json`, campo `genres`, nell'ordine con cui indicizzano `b_z`:

`Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy,
Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western`.

#### I cinque nomi scritti a mano

`situation_names_ml1m.json` — 131 byte, **cinque stringhe in italiano**:

| k | nome assegnato | etichetta descrittiva generata dal profilo | quota | lens-KL |
|---|---|---|---|---|
| 0 | Commedia serale | sera, settimana, dopo Comedy | 30,9% | 0,215 |
| 1 | Horror pomeridiano | pomeriggio, settimana, dopo Horror | 16,4% | 0,072 |
| 2 | Azione notturna | notte, weekend, dopo Action | 17,7% | **0,653** |
| 3 | Fantasy pomeridiano | pomeriggio, settimana, dopo Action | 9,7% | **0,016** |
| 4 | Dramma del weekend | sera, weekend, dopo Drama | 25,3% | 0,206 |

Da distinguere con cura: la colonna «etichetta descrittiva» **e'** prodotta dal codice
(ora modale + quota weekend + macro recente); la colonna «nome assegnato» **no**, e' stata
scritta da un umano leggendo la precedente. Il manoscritto lo dichiara. Le tre categorie
favorite per situazione (`fav_genres`) sono anch'esse lette da `b_z`, quindi verificabili:
k=0 → Animation 1,83 · Comedy 1,78 · Children's 1,64; k=2 → Action 2,34 · Fantasy 1,61 ·
Sci-Fi 1,31; k=4 → Documentary 1,53 · Drama 1,21 · Crime 1,08.

#### Quanti esempi esistono davvero, e come sono stati scelti

`reco_examples_ml1m.json` — 2851 byte, `city=ml1m`, `kappa=0.5`, **`examples` contiene 5
elementi**: uno per situazione, non uno in piu'.

| k | ora | weekend | intento recente | categoria vera | rank BASE → SIT | nudge sulla categoria vera |
|---|---|---|---|---|---|---|
| 0 | 21 | si' | Adventure | Comedy | **23 → 1** | 1,785 |
| 1 | 7 | no | Horror | Horror | **46 → 1** | 2,463 |
| 2 | 18 | no | Action | Action | **28 → 1** | 2,342 |
| 3 | 0 | no | Action | Sci-Fi | **5 → 1** | 1,383 |
| 4 | 0 | si' | Drama | Crime | **17 → 1** | 1,081 |

**Il criterio di selezione e' esplicitamente il migliore-per-situazione**, con quattro filtri in
cascata (righe 74-85):

1. `if bool(isbte[r]): continue` — solo richieste **core**, niente boundary;
2. `if nudge[r][tc] <= 0: continue` — la situazione deve **favorire la categoria vera**;
3. `if rb < 4: continue` — il backbone doveva tenerla **sotto il terzo posto**;
4. `if rs > 12 or rs >= rb: continue` — X-SAGE deve portarla **entro il dodicesimo** e migliorare;
5. fra i sopravvissuti, `gain = 1/rs − 1/rb` massimo.

Ricerca sulle prime `N = min(len(dft), 60000)` richieste di test (riga 73), quindi **60.000
delle 97.199**.

Questo va scritto nel capitolo senza giri di parole: i cinque casi sono **i migliori casi**,
per costruzione — e infatti tutti e cinque atterrano al rango 1. Non sono un campione, non
supportano nessuna statistica, e la loro funzione e' esclusivamente illustrativa. Serve un
corpus stratificato (vittorie, neutri, danni, boundary) per qualunque affermazione quantitativa
sulla spiegazione.

#### Lo spazio delle situazioni (figura)

`situation_space_ml1m.json`: `points` **1800 × 4** (x PCA, y PCA, situazione, flag di boundary —
sottocampione, non le 97.199 richieste), 5 `centroids` con `{sit, x, y, top}`,
varianza spiegata dalle due componenti **0,356 e 0,165** (**52,1%** in totale),
`bfrac = 0,242`.

---

### 2.3 — Equita': la situazione conta al netto dell'utente?

**Fonti:** `checkA_variance_<ds>.csv` (WI-0) per la decomposizione completa,
`probe_wi0b/check_a2_restricted_<ds>.csv` per quella ristretta e i test,
`probe_wi0b/check_a2_per_situation_<ds>.csv` per la tabella per situazione.

#### La quota di varianza, nei due ordini

Il fatto centrale del capitolo, e va detto per primo perche' e' scomodo:

| | situazione **da sola** | situazione **dato l'utente** | utente da solo | utente **data la situazione** | residuo |
|---|---|---|---|---|---|
| ml1m | 3,043% | **0,119%** | 30,796% | 27,872% | 69,08% |
| nyc_tist | 1,850% | **0,027%** | 39,990% | 38,167% | 59,98% |
| saopaulo | 3,907% | **0,136%** | 36,477% | 32,707% | 63,39% |

L'utente assorbe **da 25 a 69 volte** la varianza che resta alla situazione. Al netto
dell'utente la situazione spiega **un ottavo, un settantesimo e un trentesimo di punto
percentuale** rispettivamente. ICC di situazione: 0,0395 · 0,0214 · 0,0559 (contro 0,262 ·
0,218 · 0,221 per l'utente).

C'e' un artefatto che va disinnescato subito: gli utenti **mono-situazione** contribuiscono per
costruzione **zero** alla varianza situazionale netta dell'utente (nel CSV compare come
`1,05e-16`, `−1,75e-16`, `8,64e-17` — zero macchina). Sono il 55,9% delle richieste su ml1m,
il **90,1%** su nyc_tist, il 58,9% su saopaulo. La decomposizione completa e' quindi *diluita*
da utenti che non possono, in linea di principio, mostrare l'effetto.

#### La decomposizione ristretta ai soli utenti multi-situazione

| | utenti multi | richieste multi | sit \| utente | ICC sit |
|---|---|---|---|---|
| ml1m | 2.661/6.040 (44,1%) | 49.534/97.199 (50,96%) | **0,269%** | 0,0131 |
| nyc_tist | 311/4.113 (7,6%) | 1.759/17.702 (9,94%) | **0,330%** | 0,0401 |
| saopaulo | 1.437/4.395 (32,7%) | 9.794/23.821 (41,11%) | **0,371%** | 0,0075 |

Restringendo alla popolazione in cui l'effetto **puo'** manifestarsi, la quota **raddoppia o
triplica** — e resta comunque sotto lo 0,4%.

#### I test: significativo ma minuscolo

| | Kruskal-Wallis H (deviazione per situazione, entro utente) | p | n | permutazione: spread osservato | media nulla | p95 nulla | p a una coda | B |
|---|---|---|---|---|---|---|---|---|
| ml1m | 29,248 | **6,96e-06** | 5.883 | **0,21053** | 0,18298 | 0,18939 | **0,001996** | 500 |
| nyc_tist | 2,805 | 0,7300 | 622 | 0,17222 | 0,17185 | 0,19446 | 0,3533 | 500 |
| saopaulo | 50,806 | **5,38e-11** | 2.920 | **0,24630** | 0,21952 | 0,22968 | **0,001996** | 500 |

Su ml1m e saopaulo l'effetto **c'e'** ed e' fuori dal nullo di permutazione entro-utente
(p = 0,001996 e' il pavimento con B=500, cioe' 1/501: nessuna permutazione ha superato
l'osservato). Su **nyc_tist non c'e'**: Kruskal non rifiuta e lo spread osservato coincide con
la media nulla alla terza cifra. Su nyc_tist il test poggia su 311 utenti e 622 osservazioni,
quindi la lettura corretta e' *nessuna evidenza*, non *evidenza di assenza*.

La distanza fra osservato e nulla, su ml1m, e' **0,21053 − 0,18298 = +0,0275** di Cat-MRR:
significativo, e piccolo. Va scritta cosi'.

#### Chi e' servito peggio, per situazione

`check_a2_per_situation_<ds>.csv` — mediana entro-utente della deviazione dalla propria media,
con Wilcoxon e CI:

| dataset | situazione | n utenti | deviazione mediana | Wilcoxon p | CI 95% |
|---|---|---|---|---|---|
| ml1m | **1** (Horror pomeridiano) | 1.697 | **−0,00801** | **5,47e-06** | [−0,01471, 0,0] |
| ml1m | 2 (Azione notturna) | 488 | +0,01211 | 0,0286 | [0,0, +0,03285] |
| ml1m | 0, 3, 4 | 856 / 1.849 / 993 | 0,0 | 0,545 / 0,070 / 0,112 | contiene 0 |
| saopaulo | **0** | 242 | **−0,04199** | **5,82e-05** | [−0,06551, −0,00694] |
| saopaulo | 3 | 1.154 | −0,01257 | 0,00674 | [−0,02381, −0,00272] |
| saopaulo | 2 | 1.012 | +0,02440 | 3,65e-07 | [+0,01190, +0,03552] |
| nyc_tist | tutte | 0–160 | tutte ≈ 0 | 0,197–0,889 | contengono 0 |

Due situazioni su nyc_tist (**k=2 con 7 utenti, k=4 con 0**) sono marcate `n<8` e non hanno
test. Su ml1m la situazione peggio servita e' la **1**, su saopaulo la **0**.

#### Guadagno di Simpson per situazione (Δ SIT − BASE)

| | per situazione | positive | aggregato |
|---|---|---|---|
| ml1m | s0 +0,0368 · s1 **−0,0210** · s2 +0,0890 · s3 +0,0069 · s4 +0,0175 | **4/5** | +0,02517 |
| nyc_tist | s0 +0,0930 · s1 +0,0926 · s2 +0,0437 · s3 +0,1139 · s4 +0,0116 · s5 +0,0042 · s6 **−0,0203** · s7 +0,0247 | **7/8** | +0,06131 |
| saopaulo | s0 +0,0473 · s1 +0,0112 · s2 +0,0370 · s3 +0,0418 | **4/4** | +0,03562 |

Numerosita' delle celle: ml1m 8.845–31.819; nyc_tist **254**–3.716; saopaulo 2.431–10.625.
Nota che **la situazione peggio servita su ml1m (s1) e' anche l'unica che perde in Simpson**, e
che su ml1m la situazione con il massimo guadagno (s2, +0,0890) e' quella con il **lens-KL piu'
alto** (0,653, «Azione notturna»). E' una coincidenza su 5 punti, non un pattern: va riportata
come osservazione, mai come relazione.

#### Peggio servita per segmento

Segmentazioni disponibili: quintili di attivita' e split UGF (attivi = top 5%).

| dataset | segmento | peggiore | migliore | spread |
|---|---|---|---|---|
| ml1m | Q1 → Q5 | **sit_1 in tutti e cinque** (0,3498 → 0,2789) | sit_2 in tutti e cinque | 0,257 → **0,190** |
| ml1m | UGF inattivi | sit_1 (0,3002) | sit_2 (0,5668) | 0,267 |
| ml1m | UGF attivi | sit_1 (0,2854) | sit_0 (0,3952) | **0,110** |
| nyc_tist | Q5 | sit_7 (0,2635) | sit_2 (0,5634) | 0,300 |
| nyc_tist | UGF attivi | sit_4 (0,2917, **n=24**) | sit_2 (0,6084) | 0,317 |
| saopaulo | Q5 | sit_3 (0,3194) | sit_0 (0,5917) | 0,272 |
| saopaulo | UGF attivi | sit_3 (0,3656) | sit_0 (0,6459) | 0,280 |

Su ml1m il divario fra situazioni **si restringe al crescere dell'attivita'** (0,257 su Q1 →
0,190 su Q5; 0,110 sugli utenti attivi UGF): piu' storico ha l'utente, meno la situazione
discrimina la qualita' del servizio. Su nyc_tist e saopaulo va nella direzione **opposta**
(nyc 0,232 → 0,300; sao 0,142 → 0,272). Il pattern **non e' condiviso fra i domini**, e va
riportato come tale. Una sola cella e' `low_support` (nyc_tist Q1 sit_4, n=18) ed e' esclusa
da minimi e massimi qui sopra.

#### Che cosa NON esiste

**Non esiste una tabella di QoS per situazione × backbone.** Tutti i numeri di questa sezione
stanno su **un solo backbone (B_blind)** e su **tre dataset**. `scripts/fairness/` non esiste;
`outputs_results/validation/situational_relevance.csv` (51 righe) e' un'altra cosa: e' la curva
κ per SIT/UNI_mean sulle **cinque citta' Foursquare** del vecchio percorso, senza scomposizione
per situazione e senza ml1m.

---

### 2.4 — Esposizione e non-fuga: quanto rivela la traiettoria di situazioni?

**Fonti:** `probe_wi0b/check_d_minimization_ml1m.csv` (attacco di inferenza d'attributo) e
`probe_wi0b/check_c2_entropy_horizons_<ds>.csv` (persistenza dell'etichetta).

#### Le tre rappresentazioni messe a confronto

Costruite in `wi0b_probe.py:279-286`, tutte a livello di **utente**, su train+val+test uniti:

| sigla | che cos'e' | forma |
|---|---|---|
| **(a)** `a_situation_label` | one-hot della situazione **modale** dell'utente | 6.040 × 5 |
| **(b)** `b_situation_histogram` | istogramma normalizzato delle situazioni visitate | 6.040 × 5 |
| **(c)** `c_item_history` | bag-of-items binario (il **confronto** — cio' che un sistema convenzionale espone) | 6.040 × 3.706 |

Classificatore: `LogisticRegression(max_iter=2000, random_state=42)`, con `StandardScaler` per
(a) e (b) e senza per (c) sparsa; `cross_val_predict` con `StratifiedKFold(n_splits=min(5,
classe piu' rara), shuffle=True, random_state=42)`. Cinque fold per tutti e tre gli attributi.
Solo **ml1m** (`if city != "ml1m": return None`, riga 266): e' l'unico dataset con demografia.

#### I risultati, per attributo

| attributo | classi | baseline maggioranza | | (a) etichetta | (b) istogramma | (c) storico item |
|---|---|---|---|---|---|---|
| **gender** | 2 | 0,71705 | accuratezza | 0,71705 | 0,72599 | **0,78013** |
| | | (bilanciata 0,5000) | bil. | **0,50000** | 0,54963 | **0,71898** |
| | | | Δ vs maggioranza | **+0,00000** | +0,00894 | **+0,06308** |
| **age** | 7 | 0,34702 | accuratezza | 0,34702 | 0,34685 | **0,47301** |
| | | (bilanciata 0,14286) | bil. | **0,14286** | 0,14329 | **0,38251** |
| | | | Δ vs maggioranza | **+0,00000** | **−0,00017** | **+0,12599** |
| **occupation** | 21 | 0,12566 | accuratezza | 0,13262 | 0,13907 | **0,15033** |
| | | (bilanciata 0,04762) | bil. | 0,05214 | 0,05828 | **0,09388** |
| | | | Δ vs maggioranza | +0,00695 | +0,01341 | **+0,02467** |

n = 6.040 utenti in tutte le celle.

Il risultato piu' netto del probe: con la sola **etichetta di situazione**, gender e age danno
accuratezza bilanciata **esattamente pari al caso** (0,50000 su 2 classi, 0,14286 = 1/7 su 7) —
cioe' il classificatore **collassa sulla classe maggioritaria** e non estrae nulla. Lo storico
degli item, sulla stessa popolazione e con lo stesso classificatore, guadagna **+6,3 punti** su
gender e **+12,6** su age.

Su **occupation** l'etichetta di situazione **non** e' a zero: +0,70 punti di accuratezza e
+0,45 di bilanciata sopra il caso. E' piccolo, ma e' **diverso da zero** e va detto: la
rappresentazione situazionale non e' informativamente vuota, e' informativamente **povera**.

Il rapporto e' il numero da riportare: sui tre attributi, la traiettoria di situazioni concede
all'attaccante **fra lo 0% e il 28%** di quanto concede lo storico degli item
(rappresentazione (a): 0,00/6,31 · 0,00/12,60 · 0,70/2,47 punti di accuratezza).
Con l'istogramma (b) il rapporto sale, restando parziale: 14% · 0% · **54%**
(0,89/6,31 · −0,02/12,60 · 1,34/2,47).

#### I due caveat da dichiarare nel capitolo

1. **L'attacco e' non-adattivo.** Un solo classificatore lineare, nessuna ricerca di
   iperparametri, nessun avversario che progetti feature contro questa rappresentazione. E' un
   limite superiore *debole*: dice che l'informazione non e' **facilmente** estraibile, non che
   non ci sia. Va scritto cosi', altrimenti un revisore lo smonta in una riga.
2. **Non esistono baseline di privacy.** Verificato: nessuna occorrenza di *differential
   privacy*, *Laplace*, *microaggregazione* o iniezione di rumore in `scripts/diagnostics/` o
   `scripts/exp/`. Non c'e' un punto di riferimento contro cui dire «X-SAGE espone quanto un
   meccanismo a ε = …». Il confronto disponibile e' interno (a/b contro c) e va presentato solo
   come tale.

#### Persistenza dell'etichetta: l'artefatto dell'orizzonte

`check_c2_entropy_horizons_<ds>.csv` misura l'entropia normalizzata della sequenza di situazioni
per utente su **quattro orizzonti**: train, val, test, e la **linea temporale intera** (`all`).
E' la correzione decisiva, perche' misurare sul solo test da' un risultato falso.

| | | entropia mediana | quota dominante mediana | situazioni distinte (mediana) | % utenti con dominante > 80% | n |
|---|---|---|---|---|---|---|
| **ml1m** | train | 0,8196 | 0,4350 | 5 | 4,06% | 6.040 |
| | test | **0,0000** | **1,0000** | **1** | **68,37%** | 4.293 |
| | **all** | **0,8241** | 0,4293 | **5** | **3,91%** | 6.040 |
| **nyc_tist** | train | 0,7026 | 0,4000 | 6 | 1,85% | 4.113 |
| | test | **0,0000** | **1,0000** | **1** | **91,97%** | 1.271 |
| | **all** | **0,7054** | 0,4000 | **6** | **1,70%** | 4.113 |
| **saopaulo** | train | 0,7142 | 0,5952 | 4 | 16,86% | 4.395 |
| | test | **0,0000** | **1,0000** | **1** | **69,76%** | 1.888 |
| | **all** | **0,7186** | 0,5870 | **4** | **17,04%** | 4.395 |

Letto sul solo blocco di test, l'utente mediano sta in **una sola situazione** e il 68–92% degli
utenti ha una situazione dominante oltre l'80%: sembrerebbe un'etichetta persistente, quasi un
identificatore. Letto sulla linea temporale intera, l'utente mediano ne visita **4–6** e la
quota dominante scende al 43–59%: la percentuale con dominante > 80% crolla a **3,9% · 1,7% ·
17,0%**.

La spiegazione e' geometrica, non sostanziale, ed e' nel CSV: il test mediano e' lungo
**9 · 3 · 4** interazioni contro linee temporali di **96 · 28 · 36** — l'**9,6% · 11,4% · 11,1%**
del totale. Con 3 osservazioni l'entropia e' quasi forzata a zero. **La persistenza apparente e'
un artefatto della finestra**, e questa e' la forma corretta in cui il capitolo deve riportarla.

Corollario per la privacy: l'etichetta di situazione **non** e' uno pseudo-identificatore
stabile. E' esattamente coerente con il collasso del classificatore in CHECK D.

#### Geometria dello split (controllo di non-fuga)

| | timeline mediana | test mediano | quota test | violazioni **strette** | pareggi esatti di timestamp |
|---|---|---|---|---|---|
| ml1m | 96 | 9 | 9,64% | **0** | **120** |
| nyc_tist | 28 | 3 | 11,43% | **0** | 0 |
| saopaulo | 36 | 4 | 11,11% | **0** | 0 |

Zero violazioni strette su tutti e tre: lo split e' causale. I 120 casi su ml1m sono **pareggi
esatti** al confine — timestamp MovieLens a risoluzione di secondo — non fuga di futuro. La
distinzione va tenuta: contarli come violazioni sarebbe un falso positivo.

---

### 2.5 — Il costo della trustworthiness

**Fonte:** `outputs_results/explain/cost_<ds>.txt`, generati da `scripts/yelp/profile_cost.py`
(57 righe). Esistono per **10 dataset**, non solo ml1m.

#### Parametri del modulo

`profile_cost.py:44`: `params = K * nmac + K * dim` — i bias di situazione piu' i prototipi.
Nient'altro: il modulo non ha embedding, non ha pesi per utente, non ha pesi per item.

| dataset | K | ε | macro | dim(v) | **parametri** |
|---|---|---|---|---|---|
| tokyo_tist | 4 | 0,07 | 9 | 15 | **96** |
| tsmc_tky | 4 | 0,05 | 9 | 15 | **96** |
| bangkok | 4 | 0,05 | 10 | 16 | **104** |
| saopaulo | 4 | 0,03 | 10 | 16 | **104** |
| istanbul | 5 | 0,01 | 10 | 16 | **130** |
| yelp | 3 | 0,05 | 21 | 26 | **141** |
| tsmc_nyc | 7 | 0,02 | 9 | 15 | **168** |
| **ml1m** | 5 | 0,03 | 18 | 23 | **205** |
| nyc_tist | 8 | 0,02 | 10 | 16 | **208** |
| kuairand | 4 | 0,01 | 43 | 48 | **364** |

Intervallo: **96–364 parametri**. Il numero cresce con K e col numero di categorie, mai con
utenti o item — ed e' questo il punto sostanziale, non la cifra assoluta.

#### Tempo di fit, per stadio

| dataset | train | descrittore | selezione K | selezione ε | k-means | bias b̃ | **FIT totale** |
|---|---|---|---|---|---|---|---|
| nyc_tist | 124.270 | 1,60 | 23,09 | 5,73 | 0,43 | 0,00 | **30,85 s** |
| saopaulo | 172.781 | 2,19 | 24,39 | 3,91 | 0,64 | 0,01 | **31,15 s** |
| tsmc_nyc | 118.736 | 1,47 | 26,13 | 7,15 | 0,57 | 0,00 | **35,32 s** |
| yelp | 339.736 | 4,10 | 26,07 | 7,37 | 0,24 | 0,01 | **37,79 s** |
| bangkok | 323.167 | 3,58 | 28,16 | 11,68 | 4,46 | 0,01 | **47,88 s** |
| tsmc_tky | 358.977 | 3,57 | 38,10 | 21,32 | 1,23 | 0,01 | **64,25 s** |
| tokyo_tist | 447.563 | 4,78 | 29,19 | 29,45 | 1,92 | 0,02 | **65,36 s** |
| kuairand | 490.859 | 5,59 | 41,15 | 21,41 | 3,00 | 0,02 | **71,17 s** |
| istanbul | 998.030 | 10,94 | 110,18 | 29,62 | 6,61 | 0,04 | **157,39 s** |
| **ml1m** | 801.218 | 8,25 | 72,47 | 73,64 | 13,89 | 0,09 | **168,34 s** |

Fra 31 s e 168 s, su tutti e dieci i dataset. Il dettaglio che conta per il capitolo: la
**selezione degli iperparametri domina il fit**, sempre. Su ml1m, selezione K + selezione ε =
146,11 s su 168,34, cioe' l'**86,8%**; il clustering vero (13,89 s) e' l'8,2% e la stima dei
bias e' **0,09 s**, lo 0,05%. Se il fit sembra lento non e' per il metodo: e' per la ricerca su
griglia che lo precede. A iperparametri gia' fissati restano, su ml1m, **22,23 s** (descrittore 8,25 +
k-means 13,89 + bias 0,09), di cui **13,98 s** per il modulo situazionale vero e proprio.

#### Latenza di inferenza — la provenienza esatta, e il suo limite

| dataset | tempo blocco inferenza | richieste di test | **ms/richiesta** |
|---|---|---|---|
| saopaulo | 1,03 s | 23.821 | **0,043** |
| nyc_tist | 1,19 s | 17.702 | **0,068** |
| kuairand | 4,73 s | 52.768 | **0,090** |
| bangkok | 4,23 s | 43.556 | **0,097** |
| **ml1m** | 9,89 s | 97.199 | **0,102** |
| tsmc_nyc | 1,70 s | 14.298 | **0,119** |
| tokyo_tist | 7,23 s | 59.561 | **0,121** |
| yelp | 4,97 s | 36.772 | **0,135** |
| tsmc_tky | 6,77 s | 43.719 | **0,155** |
| istanbul | 24,05 s | 136.220 | **0,177** |

Va dichiarato con precisione **come** e' misurato, perche' la lettura ingenua e' sbagliata.
`profile_cost.py:36-40` cronometra un blocco che contiene **quattro cose**: `_assign` del test,
`membership_from_assign`, il prodotto `mem @ b_z`, e **l'intera chiamata `cat_mrr(...)`**. E
`cat_mrr` fa il re-ranking completo: copia della matrice di punteggi, mascheramento degli item
esclusi utente per utente, `argpartition` + `argsort` sul top-20. Quindi:

- **0,102 ms/richiesta e' un costo ammortizzato su batch da 1024**, non una latenza per singola
  richiesta in servizio;
- **include il re-ranking del backbone**, non solo il modulo situazionale;
- **esclude** il calcolo dei punteggi del backbone (`sb` e' letto da disco, gia' pronto).

La cifra e' difendibile come *sovraccarico di re-ranking a richiesta in condizioni di batch*.
Non e' difendibile come *latenza online*, e non c'e' una misura separata del solo modulo.

#### Il confronto col backbone

L'ultima riga di ogni file di costo e':
«Confronto (osservato, stessa macchina): BPR fit ~secondi; B_full (FM torch+MPS) ~1-2
min/training.»

E' una **stima a occhio, non una misurazione**: non c'e' un cronometro nel codice che produca
quei numeri, non hanno cifre decimali, e la formula di confronto sui parametri e' scritta come
approssimazione (`vs B_full ≈ (n_users+n_items+ctx)·d`). Per ml1m con 6.040 utenti, 3.706 item
e d = 32 questo darebbe circa **312 mila** parametri contro i 205 del modulo — ordine
**1:1500** — ma il conteggio esatto di B_full **non e' stato fatto**. Il capitolo puo' usare il
rapporto come ordine di grandezza dichiarato tale; non puo' usarlo come misura.

---

### 2.6 — I limiti misurati

Questa sezione raccoglie i numeri che il capitolo deve riportare **contro** il proprio metodo.
Sono i piu' importanti: se non ci sono, il capitolo non e' credibile.

#### 2.6.1 — WI-0c: il nudge non aiuta al cambio di regime (ml1m)

**Fonte:** `probe_wi0c/check_e1_by_distance_<ds>.csv`. Le richieste di test sono raggruppate per
**distanza dal cambio di situazione**: `d=0` e' la richiesta in cui la situazione cambia, `d>5`
e' a piu' di cinque passi dal cambio, `stable` sono gli utenti che non cambiano mai.
`pre_first_change` non era pre-registrato ed e' marcato come tale nel CSV (`preregistered=0`).

| bucket | richieste/seme | SIT | BASE | **Δ** | CI 95% |
|---|---|---|---|---|---|
| **ml1m d=0** | 14.314 | 0,33533 | 0,34704 | **−0,01171** | [−0,01594, −0,00721] |
| ml1m d1-2 | 11.067 | 0,33332 | 0,34221 | −0,00889 | [−0,01428, −0,00325] |
| ml1m d3-5 | 6.581 | 0,33812 | 0,33962 | −0,00149 | [−0,00953, +0,00587] |
| ml1m d>5 | 9.496 | 0,35595 | 0,34321 | +0,01274 | [−0,00086, +0,02627] |
| **ml1m stable** | 47.665 | 0,43589 | 0,38167 | **+0,05422** | [+0,04604, +0,06294] |

Il fatto che il capitolo deve riportare per intero: **su ml1m, nel momento esatto in cui la
situazione cambia, X-SAGE peggiora le raccomandazioni** (Δ = −0,0117, CI esclude zero), e va nel
verso opposto — *positivo* — solo lontano dal cambio. Il contrasto pre-registrato
`d=0 − stable` vale **−0,06593** [−0,07546, −0,05659]. Il beneficio si concentra dove non
succede nulla.

Gli altri due dataset **non** confermano il pattern, e va detto:

| bucket | nyc_tist Δ | CI | saopaulo Δ | CI |
|---|---|---|---|---|
| d=0 | **+0,05450** | [+0,01889, +0,09086] | **+0,02314** | [+0,01020, +0,03948] |
| stable | +0,05450 | [+0,04339, +0,06568] | +0,03246 | [+0,02297, +0,04296] |
| contrasto d=0 − stable | **+0,00001** | [−0,03918, +0,04138] | −0,00932 | [−0,02778, +0,01061] |

Su nyc_tist il contrasto e' **0,00001**, cioe' nullo alla quinta cifra; su saopaulo il CI
contiene zero. **Il degrado al cambio di regime e' un fenomeno di ml1m, non una proprieta' del
metodo.** Una lettura possibile: su ml1m il descrittore e' dominato dal contesto, e il contesto
al cambio e' esattamente cio' che si sta rompendo.

Nota sui bucket `WITHIN::` (confronto entro-utente): su ml1m `WITHIN::d=0` = **+0,00856**
[+0,00618, +0,01105], **positivo**, contro il −0,01171 tra-utente. I due estimatori non
concordano: al netto dell'utente, la richiesta di cambio e' servita **meglio** della media
dell'utente. La divergenza va riportata, non risolta a favore di uno dei due.

#### 2.6.2 — WI-0d: il gradiente di γ, serie completa

**Fonte:** `probe_wi0d/check_gamma_gradient_<ds>.csv`. Le richieste sono raggruppate per
`|T(v)|`, cioe' quante situazioni sono nell'insieme di appartenenza; γ = 1/|T|.

| dataset | \|T\| | γ | richieste | **Δ (SIT−BASE)** | BASE | CI 95% |
|---|---|---|---|---|---|---|
| **ml1m** | 1 | 1,0000 | 73.648 | **+0,03465** | 0,36246 | [+0,03363, +0,03554] |
| | 2 | 0,5000 | 17.976 | **−0,00300** | 0,34529 | [−0,00395, −0,00213] |
| | 3 | 0,3333 | 5.185 | **−0,00938** | 0,36587 | [−0,01059, −0,00824] |
| | 4+ | 0,2500 | 388 | −0,00722 | 0,40111 | [−0,00998, −0,00467] |
| **nyc_tist** | 1 | 1,0000 | 13.312 | **+0,06423** | 0,28838 | [+0,05796, +0,07042] |
| | 2 | 0,5000 | 3.282 | +0,02618 | 0,25371 | [+0,02103, +0,03185] |
| | 3 | 0,3333 | 988 | +0,01048 | 0,26832 | [+0,00466, +0,01574] |
| | 4+ | 0,2500 | 119 | +0,00072 | 0,28119 | — (n<CI) |
| **saopaulo** | 1 | 1,0000 | 18.739 | **+0,03677** | 0,36939 | [+0,03099, +0,04153] |
| | 2 | 0,5000 | 4.589 | +0,00414 | 0,34901 | [+0,00029, +0,00970] |
| | 3 | 0,3333 | 491 | −0,00287 | 0,36583 | [−0,01925, +0,01316] |
| | 4+ | 0,2500 | **1** | 0,00000 | 1,00000 | — (n=1) |

Il gradiente e' **monotono e netto su tutti e tre**: il beneficio e' quasi interamente sulle
richieste **core** (|T|=1), e si annulla — o si inverte — appena l'appartenenza si sfoca.
Su ml1m il segno **cambia**: +0,0347 con |T|=1, **−0,0030 e −0,0094** con |T|=2 e 3, con CI che
escludono zero. Su nyc_tist resta positivo ma cade di **89 volte** (+0,0642 → +0,0007).

Lettura onesta: γ = 1/|T| e' presentato come uno smorzamento prudente sotto ambiguita', e i dati
dicono che sulle richieste ambigue **il modulo non solo non aiuta: su ml1m fa danno**. Le celle
`4+` di nyc_tist (n=119) e saopaulo (**n=1**) non hanno CI e non vanno usate.

#### 2.6.3 — E-1: le quattro braccia e la sorpresa della troncatura

**Fonte:** `outputs_results/exp_e1/` — 4 file, **solo ml1m** (nessun altro dataset).
Backbone **EASE** ricostruito, pool troncato agli `n` item piu' recenti per utente,
`n ∈ {1, 2, 3, 5, 8, 13, 21, all}`, 5 semi (42–46), κ = 0,1 fisso.

Le quattro braccia: **BASE** (EASE troncato, nessun nudge) · **PRIOR** (nudge dal prior globale
di categoria, senza situazione) · **SIT** (nudge situazionale) · **PERM** (etichette di
situazione permutate — la guardia).

macro-Cat-MRR, media sui 5 semi:

| n | BASE | SIT | PRIOR | PERM | SIT−BASE | SIT−PRIOR | **SIT−PERM** |
|---|---|---|---|---|---|---|---|
| 1 | 0,17763 | 0,13046 | 0,12815 | 0,07397 | −0,04717 | +0,00231 | **+0,05649** |
| 2 | 0,18266 | 0,12898 | 0,18648 | 0,08995 | −0,05368 | −0,05750 | **+0,03904** |
| 3 | 0,18644 | 0,14363 | 0,19380 | 0,10295 | −0,04281 | −0,05017 | **+0,04068** |
| **5** | **0,18906** | 0,15830 | 0,18755 | 0,10025 | −0,03076 | −0,02925 | **+0,05805** |
| 8 | 0,18492 | 0,15669 | 0,19658 | 0,10087 | −0,02822 | −0,03989 | **+0,05582** |
| 13 | 0,18140 | 0,15940 | 0,19851 | 0,10913 | −0,02200 | −0,03911 | **+0,05028** |
| 21 | 0,17343 | 0,15931 | 0,19108 | 0,11626 | −0,01412 | −0,03177 | **+0,04305** |
| **all** | **0,14044** | 0,15350 | 0,14488 | 0,13379 | **+0,01306** | +0,00862 | **+0,01971** |

Tre letture, in ordine di scomodita' crescente.

**(a) La guardia di permutazione tiene.** SIT > PERM su **tutti e otto** i punti, con margini fra
+0,020 e +0,058. Le etichette di situazione portano informazione reale, non e' un effetto di
partizionamento qualunque. E' il controllo che valida il resto.

**(b) SIT perde contro BASE ovunque tranne che a pool intero.** Con il pool troncato il nudge
**peggiora** il backbone, da −0,014 a −0,054. Solo a `n=all` diventa +0,013. E perde anche
contro **PRIOR** — il nudge senza situazione — su 6 punti su 8: a `n=2` di ben **−0,0575**.
Sotto troncatura, il prior globale di categoria e' un re-ranker migliore di quello situazionale.

**(c) La sorpresa vera e' sul BASE, non su SIT.** BASE passa da **0,14044** a pool intero a
**0,18906** con pool troncato a 5 item: **+0,04862**. Troncare il profilo utente di EASE a
cinque interazioni recenti migliora la macro-Cat-MRR di quasi **cinque punti** — un effetto
**3,7 volte** piu' grande del +0,01306 che SIT ottiene a pool intero, e ottenuto **buttando via
dati** invece che aggiungendo un modulo. La curva ha un massimo a n=5 e ridiscende (0,17343 a
n=21).

Va scritto senza attenuazioni: su questo backbone e questa metrica, la troncatura del profilo e'
un intervento piu' potente della re-classifica situazionale. E' contemporaneamente il risultato
piu' interessante del probe e il piu' scomodo per la tesi del capitolo.

Due cautele obbligatorie: **(i)** K ed ε sono **riselezionati a ogni n** (n=1 → K=6, ε=0,05;
n=2,3 → K=3; n≥5 → K=5), quindi le braccia non condividono la stessa granularita' e il confronto
lungo la curva non e' a parita' di capacita'; **(ii)** non ho persistito gli array per richiesta,
quindi **nessuna di queste differenze ha un CI bootstrap con clustering per utente** — sono medie
su 5 semi, nient'altro. Le tre braccia non-BASE riselezionano anche κ (file
`e1_kappa_reselected_ml1m.csv`, κ = 0,05 uniformemente).

**Ancoraggi** (`e1_anchors_ml1m.csv`) — la ricostruzione di EASE e' verificata:

| sorgente | Cat-MRR BASE | Cat-MRR SIT | macro BASE | macro SIT |
|---|---|---|---|---|
| EASE salvata (pubblicata) | 0,375581 | 0,401146 | 0,12947 | 0,14229 |
| EASE ricostruita, pool TRAIN | 0,375666 | 0,401276 | 0,12943 | 0,14232 |
| EASE ricostruita, pool TRAIN+VAL | 0,395428 | 0,413205 | 0,14044 | 0,15349 |

La ricostruzione con pool TRAIN combacia con la matrice pubblicata alla quarta cifra
(scarto 8,5e-05 su Cat-MRR). La riga TRAIN+VAL e' il braccio `n=all` della curva. La differenza
fra le due righe ricostruite (+0,011 macro) **e' solo l'aggiunta di validation al pool**, non un
effetto di metodo.

---

## 3. Questioni emerse

Cose trovate durante l'estrazione che non erano nel brief e che vanno decise prima di scrivere
il capitolo. Nessuna richiede un nuovo esperimento; tutte richiedono una scelta editoriale.

**3.1 — Un solo backbone regge l'intero capitolo.** Ogni numero delle sezioni 2.1, 2.3, 2.4 e
2.6.1-2.6.2 sta su **B_blind**, seme 42, su **tre** dataset. Il capitolo 6 non ha nulla di
paragonabile alla griglia 5 × 7 del paper. Va dichiarato in apertura di capitolo, non in nota.

**3.2 — I cinque esempi di spiegazione non sono un campione.** Sono i migliori casi per
costruzione (quattro filtri in cascata + argmax del guadagno), e tutti e cinque atterrano al
rango 1. Vanno presentati come **illustrazioni**, mai come evidenza. Qualunque frase del tipo
«X-SAGE porta la categoria corretta in cima» va riscritta.

**3.3 — Il nome delle situazioni resta un atto d'autore, in italiano.** Cinque stringhe scritte
a mano. Il paper lo dichiara; il capitolo deve fare lo stesso, e servono comunque in inglese.
Le **etichette descrittive** (`sera, weekend, dopo Drama`) sono invece generate dal codice: la
distinzione va tenuta esplicita perche' e' la differenza fra output del metodo e glossa umana.

**3.4 — Il degrado al cambio di regime e' di ml1m, non del metodo.** Δ = −0,0117 su ml1m con CI
che esclude zero, ma contrasto **+0,00001** su nyc_tist e CI che contiene zero su saopaulo. Se il
capitolo lo presenta come limite strutturale sbaglia in una direzione; se lo omette sbaglia
nell'altra. Va riportato come **specifico del dominio**.
In piu': su ml1m gli stimatori tra-utente ed entro-utente **discordano sul segno** (−0,01171
contro +0,00856). Serve una frase che lo dica.

**3.5 — Su nyc_tist l'effetto situazionale entro-utente non e' rilevabile.** Kruskal p = 0,730,
permutazione p = 0,353, spread osservato uguale alla media nulla alla terza cifra. Il capitolo
non puo' scrivere «l'effetto e' presente sui tre dataset». Con 311 utenti multi-situazione la
formula corretta e' **nessuna evidenza**, non evidenza di assenza.

**3.6 — Il gradiente di γ e' un limite, e non e' ancora scritto da nessuna parte.** Su ml1m il
nudge fa **danno statisticamente significativo** sulle richieste di boundary (−0,0030 e −0,0094,
CI escludono zero) — cioe' sul 24,2% delle richieste. γ = 1/\|T\| smorza, ma non abbastanza. E'
il candidato piu' forte per una sezione «quando non usarlo».
Controllo di coerenza superato: la quota di richieste con \|T\|=1 nel file del gradiente da'
boundary **24,2% / 24,8% / 21,3%**, e il 24,2% di ml1m combacia esattamente con `bfrac = 0,242`
in `situation_space_ml1m.json` e con il «24%» del manoscritto.

**3.7 — La sorpresa della troncatura e' piu' grande del contributo del metodo.** BASE guadagna
+0,04862 troncando il pool di EASE a 5 item; SIT guadagna +0,01306 a pool intero. Rapporto
**3,7:1**, a favore del buttare via dati. E' materiale ottimo per il capitolo 8 (minimizzazione)
e scomodo per il 6. Va riportato in entrambi, con lo stesso numero.

**3.8 — E-1 non ha CI e non ha un secondo dataset.** Non ho persistito gli array per richiesta:
le differenze della curva sono medie su 5 semi **senza** bootstrap con clustering per utente. E
tutti e quattro i file sono `dataset = ['ml1m']`. Ogni frase deve dire «su MovieLens-1M», mai
«sistematicamente». Rifarlo con gli array persistiti costa un rerun, non un ripensamento.

**3.9 — L'attacco di privacy e' non-adattivo e senza baseline.** Una regressione logistica, zero
tuning, nessun meccanismo di riferimento (verificato: nessuna occorrenza di *differential
privacy* / Laplace / microaggregazione nel codice). Regge come *l'informazione non e'
facilmente estraibile*; non regge come *non c'e' informazione*. E su **occupation** l'etichetta
di situazione **non** e' a zero (+0,70 punti sopra la maggioranza): la frase «rivela nulla» e'
falsa e va sostituita con il rapporto 0-28% rispetto allo storico item.

**3.10 — La latenza di 0,102 ms non e' una latenza online.** Il blocco cronometrato include
`cat_mrr` per intero — copia della matrice, mascheramento, `argpartition` sul top-20 — su batch
da 1024, ed esclude il calcolo dei punteggi del backbone. E' un **sovraccarico di re-ranking
ammortizzato**, e va chiamato cosi'. Non esiste una misura del solo modulo situazionale.

**3.11 — Il confronto col backbone e' una stima a occhio.** «BPR ~secondi, B_full ~1-2 min» non
proviene da nessun cronometro nel repo, e il rapporto sui parametri (~205 contro ~312k, ordine
1:1500) e' calcolato da una formula approssimata, mai dal conteggio reale di B_full. Usabile
come ordine di grandezza dichiarato; non come misura.

**3.12 — Nulla di tutto questo e' sotto git.** 65 dei 67 file di output usati in questo
documento sono **untracked**. Se la cartella si perde, il capitolo 6 non e' rigenerabile senza
rifare tutti i probe. E' la cosa piu' economica da sistemare di questo elenco: sono CSV e JSON
piccoli, non le matrici da 45 GB.

**3.13 — Il confondente della granularita' resta aperto anche qui.** In E-1 K ed ε sono
**riselezionati a ogni n** (K = 6, 3, 3, 5, 5, 5, 5, 5): la curva non e' a parita' di capacita'.
E' lo stesso confondente gia' annotato per l'ablazione a stadi, e la stessa risposta —
`joint @ K forzato` — lo scioglierebbe in entrambi i casi.

---

## 4. Che cosa NON esiste (elenco chiuso)

Perche' non venga cercato di nuovo:

- tabella di QoS per situazione × backbone (`scripts/fairness/` non esiste);
- CI bootstrap sulle differenze di E-1;
- E-1 su qualunque dataset diverso da ml1m;
- misura isolata della latenza del modulo, separata dal re-ranking;
- conteggio reale dei parametri di B_full;
- cronometraggio del fit dei backbone;
- baseline di privacy (DP, microaggregazione, rumore);
- attacco di inferenza d'attributo su dataset diversi da ml1m (sono gli unici con demografia);
- corpus di spiegazioni stratificato: ne esistono **5**, tutti best-case;
- nomi delle situazioni in inglese, e per dataset diversi da ml1m.
