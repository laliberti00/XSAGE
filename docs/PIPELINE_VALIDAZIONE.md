# X-SAGE — Pipeline di validazione (fedele al codice, passo per passo)

Ossatura della sezione Metodi. Fedele a `scripts/yelp/{results_record,aggregate_record,stability_check}.py`
+ `mind_prep.build_descriptor` + `pipeline/step02_models/xsage/`. Formule = quelle eseguite.

Notazione: richiesta `r` = (utente `u_r`, item vero held-out `i_r`, contesto); categoria vera `t_r=c(i_r)`;
score del modello `s_{r,j}` per ogni item `j`; catalogo `N` item, `M` macro-categorie.

---

## FASE 0 — Input congelato
- **Dataset** k-core 10 uniforme, **split temporale**, **leave-one-out** (1 item vero per richiesta).
- 7 backbone × 3 metodi (BASE, SIT, Steck-b) × 5 seed {42–46}. Primari (7) per Holm; appendice (5) replica.

---

## FASE 1 — Descrittore della situazione  `v = [c̃ ‖ e]`  (★ punto anti-circolarità)

Per ogni richiesta si costruisce un vettore `v` da **soli segnali disponibili prima della richiesta**.

**1a. Finestra recente (strettamente causale)** — `build_recent_window` (`l0_sensing.py`):
per la richiesta di `u_r` al tempo `t`, prende le **ultime n interazioni dello stesso utente con
`time_local < t`** dalla storia `hist[split]`. → L'item held-out (a tempo `t`) **non entra mai**.
- Confine `hist`: `train→train`, `val→train`, **`test→train+val`** (mai test). `mind_prep.py:83-84`.

**1b. Profilo di recency** `m ∈ ℝ^M` — `compute_profile(recent_macro, n_prior, M, γ)`:
conteggio decaduto (decadimento `γ`) delle macro nella finestra, normalizzato → distribuzione di
"da dove viene" l'utente sulle categorie.

**1c. Intento propagato** `e ∈ ℝ^M` — `compute_intent(m, W, attractors, H, β, mode=hard)`:
propaga `m` sul **grafo di transizione tra macro** `W` per `H` salti (con attrattori), ottenendo lo
"scopo-categoria" verso cui l'utente sta andando. `W = estimate_macro_transition(df_train)`,
`attractors = find_attractors(W)` — **stimati solo su train**.

**1d. Contesto informativo** `c̃` — `fit_contribution_functions(df_train, …, max_depth=depth, min_leaf=200)`
poi `.transform(target)`: informatività degli attributi di contesto (ora/giorno/weekend/mese, ecc.)
rispetto alla **prossima** macro. Alberi **fittati su train**.

**1e.** `v = [c̃ ‖ e]` (concatenazione).

> **Anti-circolarità (verificato nel codice, non assunto):** la finestra è `time_local < t` (target
> escluso); `hist[test]=train+val`; `contrib`, `W`, `attractors` (e più sotto `b_z`, `Pu`) sono stimati
> **solo su train**. L'item/categoria held-out **non tocca** `v`. È il primo punto che un revisore controlla.

---

## FASE 2 — Situazioni e bias (per seed, su train)

**2a. Clustering** rough-k-means su `v_train`: `K` selezionato su **validation** (silhouette + banda di
boundary-fraction), `ε` su validation. Etichette core `z_tr`. *(selezione iperparametri = standard "su val".)*

**2b. Bias di situazione** `b_z ∈ ℝ^{K×M}` — `fit_situation_biases_z(z_tr, cmt, K, M, λ)` (`recommendation.py:127`):
log-odds shrinkati e **z-scorati dentro la situazione** (media 0, std 1 per situazione):
```
counts_k[k,m] = #{train in cluster k, macro m};   p̃[k,·] = (counts_k[k,·] + λ·p_glob) / Σ(·)
b[k,m] = log p̃[k,m] − log p_glob[m];   b_z[k,·] = (b[k,·] − μ_k) / σ_k        (λ = 50)
```
→ κ interpretabile come "nudge additivo di ≈κ deviazioni standard".

**2c. Prior utente** `Pu ∈ ℝ^{U×M}`: distribuzione storica di categoria dell'utente (da train), normalizzata
(usata per JS e per Steck-b).

**2d. Steck-b** = **stessa formula di 2b ma con etichetta = utente** (`fit_situation_biases_z(umac, cmt, U, M, λ)`)
→ `b_z_user[u,·]` = calibratore **statico per-utente** sul prior di categoria (NON situazionale).

---

## FASE 3 — Combiner additivo (i 3 metodi)
Membership `mem[r,k]` (soft), `γ_r = 1/|componenti|`. Nudge per categoria:
- **SIT**: `nudge[r,·] = (mem @ b_z)[r,·]`   · **Steck-b**: `nudge[r,·] = b_z_user[u_r,·]`   · **BASE**: nessuno.

Score combinato (bias broadcast sugli item via categoria `c(j)`):
```
ŝ_{r,j} = s_{r,j} + κ · γ_r · nudge[r, c(j)]
```
- **κ\*** scelto **su validation** (per metodo/backbone) massimizzando Cat-MRR. BASE: κ=0.
- Item già visti dall'utente → `ŝ = −∞` (full-ranking, niente negative sampling; Krichene & Rendle 2020).

---

## FASE 4 — Ranking grezzo + conteggi di tie  (`per_request_eval` → cache)
```
rk_r   = |{j : ŝ_{r,j} > ŝ_{r,i_r}}| + 1            g_r  = |{j : ŝ_{r,j} = ŝ_{r,i_r}}|
best_r = max_{j:c(j)=t_r} ŝ_{r,j}
catrk_r= |{j : ŝ_{r,j} > best_r}| + 1               gc_r = |{j : ŝ_{r,j} = best_r}|
tk50_r = top-50 item per ŝ
```
**Cache** `raw_<city>.npz`: per `(backbone, metodo, seed)` → `rk, catrk, g, gc, tk50`;
shared `u, tm, G1, icm, Pu, nI, nmac`.

---

## FASE 5 — Metriche col RANGO ATTESO (McSherry-Najork 2008; `derive`, `per_req_quantity`)
Blocco di tie ai posti `[rk, rk+g−1]`, `b=rk−1`. Prefissi `H[k]=Σ_{p≤k}1/p`, `Hlog[k]=Σ_{p≤k}1/log₂(p+1)`.

| metrica | formula (attesa; guardia `b<K`) |
|---|---|
| MRR-item | `(H[b+g]−H[b]) / g` |
| HR@K | `clip((K−b)/g, 0, 1)` |
| NDCG@K | `(Hlog[min(b+g,K)]−Hlog[b]) / g` se `b<K`, else 0 |
| Cat-MRR@20 `cm` | `(H[min(bc+gc,20)]−H[bc]) / gc` se `bc<20`, else 0  (`bc=catrk−1`) |
| Cat-NDCG@20 `cn` | `(Hlog[min(bc+gc,20)]−Hlog[bc]) / gc` se `bc<20`, else 0 |

Aggregazione (`metrics_from`, 21 metriche):
- **macro-Cat-MRR** (min-support m): `(1/|C⁺_m|) Σ_{c:|R_c|≥m} mean_{r:t_r=c} cm_r`  (m=20 a-priori).
- Cat-MRR/Cat-NDCG = `mean_r cm_r / cn_r`; HR/NDCG/MRR primarie = media **per-richiesta**, `_u` per-utente.
- LT@20 = `mean_r mean_{j∈top20} 1(j∈G1)` (G1 = fuori top-20% popolarità); Coverage = `|∪top20|/N`;
  Gini = `(2Σ i·xᵢ)/(N·Σx) − (N+1)/N`; **JS** = `mean_r JS(Pu[u_r], istogramma-cat(top20_r))` (costo).

---

## FASE 6 — Incertezza, contrasti e SIGNIFICATIVITÀ  (★ design a 3 condizioni, congelato)

**Due fonti di rumore, due strumenti:**
- **Bootstrap per-richiesta** (B=1500): cattura il rumore di **campionamento** (n=migliaia → potenza).
- **5 seed**: catturano il rumore di **inizializzazione/clustering** (n=5 → consistenza, non potenza).

Per ogni `(backbone, metrica)`, riga per metodo (mean/sd_seed/se_boot/CI) e sulla riga **SIT** i due
contrasti **L1 = SIT−BASE**, **L2 = SIT−Steck-b**:
- `delta_lX` = differenza medie 5-seed; `ci_lX_lo/hi` = **CI bootstrap del Δ**.
- `p_lX` = **bootstrap-Δ** (PRIMARIO, potenza): `p = 2·min(P(Δ*≤0), P(Δ*≥0))` → **Holm×14** in finalize.
- `p_lX_seed` = **t appaiato cross-seed** (df=4) — robustezza conservativa, **non** gate.
- `seeds_lX` = #seed con Δ>0 (0–5) — **consistenza**.

**Casella (regola UNICA per tutte: winner/ridondante/nullo)** su macro-Cat-MRR, per backbone:
```
Lk_pass  =  (Δ_k > 0)  ∧  (CI bootstrap esclude 0: ci_lk_lo > 0)  ∧  (seeds_k == 5)
nullo      se ¬L1_pass        ridondante se L1_pass ∧ ¬L2_pass        winner se L1_pass ∧ L2_pass
```
- **5/5 fissato a priori** (sign-test p≈0.06): un winner 4/5 NON è winner pulito → "direzionalmente
  consistente, un seed dissente" (claim morbido). `sd_seed` è descrittivo, non un secondo criterio.
- **Onestà di reporting:** il `p_lX` bootstrap è quasi-sempre `≈0` (potenza enorme) → **near-vacuo**. Il
  lead è **effect size (Δ, CI) + consistenza-seed**; gli asterischi Holm sono il *pavimento*, non l'argomento.
  *(Es. nyc/DeepFM: `sig_l1=***` ma `box=nullo` perché Δ_L1<0 e 4/5 — gli asterischi non bastano.)*
- **TOST ±0.005** (HR20/NDCG20) per claim di equivalenza "non costa accuracy", per-backbone/dataset.

---

## FASE 7 — Holm a due famiglie (`finalize`, cross-dataset, solo PRIMARI)
- **PRIMARIA** = `B_full × macro-Cat-MRR × {L1,L2} × 7 primari` = **14 test**.
- **ROBUSTEZZA-backbone** = `(altri 6) × macro-Cat-MRR × {L1,L2} × 7` (famiglia separata).
- Holm: `p_holm[rank] = min(1, max(prev, (m−rank)·p))`; `*`<.05, `**`<.01, `***`<.001.
- Appendice + altre metriche → descrittive (p_raw, niente Holm). FOCALE = **B_full dichiarato a priori**.

---

## FASE 8 — Gate di validazione (post-freeze)
1. **Round-trip** `aggregate_record`==`results_record`, `max|Δ|=0` → cache fedele.
2. **Stability** (`stability_check.py`, ~2.5s): caselle invarianti per `MSUPP∈{20,50}` (regola Δ>0 ∧ 5/5).
3. **GATE winner ml1m esplicito**: "ml1m=winner" SOLO se `Δ_L2(SIT−Steck-b)@m=20 > 0` e **5/5 concordi**.
4. **3 asserzioni** (verde su nyc): JS Steck-b<SIT (calibratore-utente); nyc `Δ_L2≤0` (ridondante);
   segni L1/L2 invarianti stretto-vs-atteso.

---

*Natura del lavoro: caratterizzazione empirica onesta (vedi `FREEZE_RECORD.md`), non una legge predittiva.
Le caselle si osservano e si spiegano; non si predicono out-of-sample.*
