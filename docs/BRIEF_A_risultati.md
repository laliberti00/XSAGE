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

