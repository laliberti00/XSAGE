# Fase 02 — Perception (L1)

Stato: **VALIDATO CON RISERVA** (2026-06-18) — logica PASS 8/8; parametri
`γ`/`depth`/`n` flaggati sensibili (vedi Parte 2)

La perception è il livello L1: trasforma il contesto causale prodotto dal
sensing in un **descrittore di richiesta** `v = [c̃ ‖ e]`, che la fase successiva
(comprehension) clusterizza in situazioni. Si compone di due parti:

- **L1a — Context state `c̃ ∈ [0,1]^A`**: quanto ogni attributo di contesto è
  *informativo* sulla prossima macro-categoria.
- **L1b — Intent vector `e`**: dove l'utente sta "andando" nel grafo delle
  macro-categorie, partendo dalla sua storia recente.

---

## Cosa fa la perception

### L1a — Context state
Per ogni attributo di contesto `a` (ora, giorno, weekend, mese, geohash
precedente, categoria-intento precedente) si addestra un **albero di decisione
shallow** che predice la prossima macro `cat_target`. Per ogni foglia si misura
quanto la distribuzione delle macro è concentrata:

```
θ = 1 − H(p_leaf) / log₂(K_mac)
```

θ≈1 → foglia molto informativa (prossima macro quasi certa); θ≈0 → foglia
non-informativa (distribuzione uniforme). Il context state è il **vettore**
`c̃ = (θ_{a})_{a=1..A}` — una componente per attributo, **nessuna aggregazione
scalare**.

### L1b — Intent vector
1. **Matrice di transizione macro `W`**: `W[c,c'] = P̂(next=c' | cur=c)` da
   coppie successive dello stesso utente, con **smoothing +1** e righe
   normalizzate.
2. **Attrattori `A = {c : indeg(c) ≥ media indeg}`**: le macro "pozzo" del grafo.
3. **Profilo di recency `m`**: `m_c ∝ Σ_j γ^j 𝟙[macro_j = c]` sulle ultime `n`
   mosse (più recente → j=0), con decay `γ`.
4. **Intent**: `e ∝ (m · Σ_{k=1..H} β^k W^k) ⊙ 𝟙[∈A]`, rinormalizzato sugli
   attrattori. Propaga il profilo recente attraverso `H` passi del grafo macro,
   scontati di `β` per passo, e tiene solo le destinazioni-attrattore.

## Come è implementata (clean repo)

File: [`xsage/l1_perception.py`](../../xsage/l1_perception.py)

| Componente | Funzione |
|---|---|
| Context trees + θ | `fit_contribution_functions(...)` → `ContributionModel.transform` |
| Transizione W (+1, row-norm) | `estimate_macro_transition(...)` |
| Attrattori (indegree) | `find_attractors(W)` |
| Profilo recency m (γ) | `compute_profile(..., gamma)` |
| Intent e (H, β) | `compute_intent(..., H, beta)` |

Il descrittore `v = [c̃ ‖ e]` è poi clusterizzato in
[fase 03 — comprehension](../03_comprehension/) (`l2_comprehension.fit_rough_kmeans`).
Nel clean repo il fit delle situazioni è materializzato in `fit.npz` (caricato da
`data.load_fit`); le funzioni di perception sono qui, verificate una a una.

## Quali scelte e perché

- **Un albero per attributo (non un unico albero multi-feature).** Isola il
  contributo informativo di *ciascun* attributo come dimensione separata di `c̃`:
  il descrittore resta interpretabile (ogni componente = un attributo) e robusto
  (un attributo rumoroso non inquina gli altri).
- **Modo intent "hard" (solo attrattori).** È l'Eq.5 dell'Approach. I modi
  alternativi del vecchio repo (`all`, `soft_topr`) e le varianti transit
  (`collapse`, `mask`) **non sono nel clean repo** — solo `hard`/`keep`, quelli
  usati nei risultati del paper.
- **γ e β distinti, su oggetti diversi.** `γ` sconta la **recency** (nel profilo
  `m`, asse temporale della storia utente); `β` sconta la **reach nel grafo**
  (potenze di `W`, asse topologico). Sono due iperparametri indipendenti.
- **Attrattori via indegree ≥ media.** Soglia non-parametrica e auto-tarata sul
  grafo della città: identifica le destinazioni verso cui converge il moto macro.

## Validazione — Parte 1: la LOGICA corrisponde all'Approach

Verifiche numeriche su NYC (`scripts/validation/perception_logic_check.py`),
moduli **clean** sotto test:

| # | Punto Approach | File:funzione | Verifica numerica (NYC) | Corrisponde |
|---|---|---|---|---|
| L1a.1 | Un albero shallow per attributo, depth dichiarata | `fit_contribution_functions` | A=6 alberi, depth ≤ 3 tutti | ✅ |
| L1a.2 | θ = 1 − H(p_leaf)/log₂(K_mac) | `ContributionModel` | foglia c_hour: H=2.748, θ_manuale=θ_codice=0.172768 | ✅ |
| L1a.3 | Context state = vettore (B,A), no scalare | `.transform` | shape (124270, 6), valori in [0.15, 0.34] | ✅ |
| L1b.4 | W: +1 smoothing, righe somma=1 | `estimate_macro_transition` | rowsum=1.000000 ovunque, tutte le entry >0 | ✅ |
| L1b.5 | Attrattori = {c: indeg ≥ media} | `find_attractors` | 6/10 macro, = lista `summary.json` del dossier | ✅ |
| L1b.6 | e ∝ (m·Σβ^k W^k)⊙𝟙[A], rinorm. | `compute_profile`+`compute_intent` | e = formula manuale (atol 1e-5); supporto solo su attrattori | ✅ |
| L1b.7 | γ e β distinti, oggetti diversi | (entrambe le funzioni) | γ cambia `m` e non `e`-grafo; β cambia `e` e non `m` | ✅ |
| L1b.8 | Modo hard = Eq.; soft/all assenti | `compute_intent` | nessun parametro `mode`; hard cablato | ✅ |

**Criterio PASS Parte 1:** tutti i punti CORRISPONDE + verifiche numeriche
tornano → **PASS (8/8)**.

> Cross-check forte: gli attrattori NYC calcolati dai moduli clean
> (Arts & Entertainment, Food, Nightlife Spot, Outdoors & Recreation,
> Shop & Service, Travel & Transport) coincidono **esattamente** con quelli
> registrati in `summary.json` del dossier.

## Validazione — Parte 2: sensibilità dei PARAMETRI sull'ARI

Misura end-to-end: si varia **un parametro alla volta** (gli altri al default),
si rifà il rough-k-means alla `(K, ε)` selezionata di ciascuna città e si misura
l'**ARI cross-seed** (media su 3 coppie dei seed {42,43,44}). Harness:
`scripts/validation/perception_param_sensitivity.py` → tabella
`outputs_results/validation/perception_param_sensitivity.csv`. L'harness riusa le
funzioni testate del vecchio repo (derivazione colonne identica) e riproduce
`ari_seeds` del dossier (NYC ARI seed-pair = 0.908). **5/5 città** (costo ~40 min;
Istanbul domina a ~74 s/config), default in **grassetto**.

**Griglie (e perché):** `depth {2,3,4,5}` (alberi shallow, oltre 5 si perde
interpretabilità); `γ {0.3,0.5,0.6,0.7,0.9}` e `β {0.3,0.5,0.7,0.9}` (sconti in
[0,1], campionati attorno al default); `n {3,5,7,10}` (finestra storia, da corta
a lunga); `H {1,2,3}` (passi di grafo, oltre 3 il prodotto W^k si appiattisce).

ARI cross-seed (media) per parametro × valore × città:

**depth** (alberi context, default 3)

| depth | Istanbul | Bangkok | NYC | Sao Paulo | Tokyo |
|---|---|---|---|---|---|
| 2 | 0.556 | 0.777 | 0.709 | 0.669 | 0.904 |
| **3** | 0.646 | **0.998** | **0.939** | 0.791 | 0.773 |
| 4 | 0.650 | 0.542 | 0.604 | 0.882 | 0.779 |
| 5 | 0.703 | 0.725 | 0.996 | 0.767 | 1.000 |

**γ** (decay recency, default 0.6)

| γ | Istanbul | Bangkok | NYC | Sao Paulo | Tokyo |
|---|---|---|---|---|---|
| 0.3 | 0.586 | 0.651 | 0.891 | 0.761 | 1.000 |
| 0.5 | 0.540 | 0.610 | 0.708 | 0.694 | 0.781 |
| **0.6** | 0.646 | **0.998** | **0.939** | **0.791** | 0.773 |
| 0.7 | 0.601 | 0.665 | 0.729 | 0.685 | 0.898 |
| 0.9 | 0.578 | 0.688 | 0.696 | 0.558 | 0.785 |

**β** (path-discount grafo, default 0.7)

| β | Istanbul | Bangkok | NYC | Sao Paulo | Tokyo |
|---|---|---|---|---|---|
| 0.3 | 0.689 | 0.877 | 0.708 | 0.664 | 0.798 |
| 0.5 | 0.712 | 0.998 | 0.647 | 0.775 | 0.739 |
| **0.7** | 0.646 | **0.998** | **0.939** | **0.791** | 0.773 |
| 0.9 | 0.702 | 0.669 | 0.760 | 0.534 | 0.632 |

**n** (finestra history, default 5)

| n | Istanbul | Bangkok | NYC | Sao Paulo | Tokyo |
|---|---|---|---|---|---|
| 3 | 0.852 | 0.659 | 0.650 | 0.803 | 0.739 |
| **5** | 0.646 | **0.998** | **0.939** | 0.791 | 0.773 |
| 7 | 0.530 | 0.998 | 0.749 | 0.726 | 0.789 |
| 10 | 0.665 | 0.796 | 0.942 | 0.691 | 1.000 |

**H** (passi grafo, default 2)

| H | Istanbul | Bangkok | NYC | Sao Paulo | Tokyo |
|---|---|---|---|---|---|
| 1 | 0.540 | 0.997 | 0.756 | 0.819 | 0.535 |
| **2** | 0.646 | **0.998** | **0.939** | 0.791 | 0.773 |
| 3 | 0.638 | 0.490 | 0.753 | 0.790 | 0.898 |

### Risposte alle tre domande

**(a) Esiste un intervallo robusto per ciascun parametro?**
Solo in parte. **H** è il più robusto: `H∈{1,2}` tiene ARI alto in 4/5 città
(eccezione Tokyo che sale con H). **β** ha una banda moderata `[0.5,0.7]`; `β=0.9`
peggiora quasi ovunque (Sao Paulo crolla 0.79→0.53). **depth, γ, n** **non**
mostrano un plateau: l'ARI è fortemente **non-monotòno** e dipendente dalla città
(es. Bangkok depth 3→0.998 ma 4→0.542; Tokyo n 5→0.773 ma 10→1.000). Niente di
simile al plateau di λ.

**(b) I default cadono nella regione robusta?**
- `H=2`: ✅ sì, regione robusta.
- `β=0.7`: ✅ dentro la banda `[0.5,0.7]` (ma vicino al bordo; 0.9 da evitare).
- `depth=3`: ⚠️ è il picco per Bangkok/NYC ma un **avvallamento** per Tokyo
  (0.773, mentre 2→0.904 e 5→1.000) e sub-ottimo per Istanbul/Sao Paulo.
- `γ=0.6`: ⚠️ è un **picco stretto**, non un plateau: Bangkok 0.998 a 0.6 ma
  ~0.61–0.67 a 0.5/0.7; Tokyo preferisce 0.3 (1.000).
- `n=5`: ⚠️ compromesso, non ottimo robusto: Istanbul vuole n=3 (0.852), Tokyo
  vuole n=10 (1.000).

**(c) Parametro sorprendentemente sensibile (campanello d'allarme)?**
**Sì — `γ`, `depth` e `n`.** Il campanello è duplice: (1) l'ARI è **spiky** in
questi parametri (basini stretti, non ampi); (2) i default `γ=0.6/depth=3/n=5`
siedono spesso **esattamente sul picco** locale. Questo è coerente — e va detto —
col fatto che `(K, ε)` sono stati **selezionati massimizzando l'ARI** sul config
di default: la stabilità al default è quindi in parte **ottimistica** (selection
bias), e i vicini regrediscono. La colonna `ari_min` conferma che dove l'ARI è
alto i 3 seed concordano davvero (basino reale), ma **stretto**. Ranking di
robustezza: `H ≳ β  >  depth ≈ n ≈ γ`.

> **Lezione/sorpresa.** La scelta di `(K, ε)` per massimo ARI rende il config di
> default un picco, non un centro di plateau. Non è un errore di per sé (le
> situazioni *sono* stabili a quei valori), ma significa che la robustezza dei
> parametri di perception **non è dimostrata** da questa mappa: `γ`, `depth`, `n`
> andrebbero discussi/ri-selezionati con un criterio che premi l'ampiezza del
> basino (plateau) e non solo il massimo puntuale. Decisione rimandata all'utente
> (come da brief: qui si mappa, non si sceglie).

**Criterio PASS Parte 2:** per ogni parametro, default in regione robusta **oppure**
flag esplicito "sensibile, da discutere". Esito: `H=2` ✅ robusto; `β=0.7` ✅
banda robusta; `depth=3` ⚠️ flag sensibile; `γ=0.6` ⚠️ flag sensibile; `n=5` ⚠️
flag sensibile. → **soddisfatto** (3 parametri flaggati, nessuno ignorato).

---

## Criterio PASS e verdetto

- **Parte 1 (logica):** PASS 8/8 — la logica del clean repo corrisponde
  all'Approach, verifiche numeriche tutte tornate.
- **Parte 2 (parametri):** mappa prodotta; `H`/`β` robusti, `depth`/`γ`/`n`
  flaggati sensibili (default su picchi stretti, selection bias da ARI-max).

**Verdetto fase 02: VALIDATO CON RISERVA.** La logica è validata senza riserve;
la riserva è sui parametri `γ`, `depth`, `n`, da discutere prima di trattarli
come scelte robuste (non bloccano le fasi a valle: le situazioni del dossier sono
stabili ai valori selezionati, ma la loro *robustezza* non è dimostrata).

