# X-SAGE — Sunto totale (per revisione scientifica)

Documento autocontenuto: idea, framework, codice, protocollo, risultati, lente diagnostica, limiti.
Tutto è ancorato al codice eseguito (`scripts/yelp/{results_record,aggregate_record,stability_check}.py`,
`mind_prep.build_descriptor`, `pipeline/step02_models/xsage/`).

---

## 1. Idea e tesi

**Problema.** I recommender context-aware (FM, DeepFM, AFM, sequenziali) fondono il contesto **in modo
latente e opaco**: migliorano le metriche ma non dicono *perché* né *quando* il contesto conta. Non è
diagnosticabile né interpretabile.

**Idea centrale.** Rendere esplicita la nozione di **situazione**:
$$\textbf{situazione} = \textbf{contesto} \;+\; \textbf{intento}$$
- **contesto** = segnali osservabili al momento della richiesta (ora, giorno, weekend, mese; geohash grezzo
  del check-in precedente per i POI);
- **intento** = dove sta andando l'utente sullo spazio delle categorie, propagato causalmente dalla sua
  storia recente.

**X-SAGE** (situational additive re-ranker) è una **testa additiva** montata **sopra un backbone congelato**:
riconosce la situazione della richiesta e ri-ordina spingendo verso le categorie appropriate a quella
situazione. È **esplicita** (le situazioni sono oggetti ispezionabili), **interpretabile** (la spiegazione
è la formula stessa) e **agnostica al backbone** (si monta su qualsiasi modello).

**Claim scientifico** (onesto, non massimalista): non "X-SAGE vince sempre", ma una **caratterizzazione**:
X-SAGE aggiunge valore **dove il gusto è situazionale** (film), è **ridondante dove il gusto è stabile
per-utente** (luoghi POI: basta "chi sei"), è **nullo dove non c'è segnale** (feed randomizzato). Più una
**lente diagnostica** che localizza *dove* e *per chi* c'è iniquità di categoria.

---

## 2. Il framework, strato per strato (ancorato al codice)

Pipeline percettiva a strati (`pipeline/step02_models/xsage/`), poi clustering e combiner.

**L0 — Sensing** (`l0_sensing.build_recent_window`).
Per una richiesta di utente $u$ al tempo $t$: **finestra strettamente causale** delle ultime $n$
interazioni dello stesso utente con `time_local < t`. → l'item held-out (a $t$) non entra mai.

**L1 — Perception** → descrittore $v = [\tilde c \,\|\, e]$:
- **contesto informativo** $\tilde c$ (`fit_contribution_functions` su train): informatività degli attributi
  di contesto rispetto alla prossima macro-categoria (alberi, max_depth=depth, min_leaf=200).
- **profilo di recency** $m \in \Delta^M$ (`compute_profile`): distribuzione decaduta ($\gamma$) delle
  macro-categorie nella finestra.
- **intento propagato** $e$ (`compute_intent`): $m$ propagato sul **grafo di transizione tra categorie**
  $W$ (stimato su train) per $H$ salti, con attrattori → "scopo-categoria" verso cui l'utente va.

**L2 — Comprehension** (`l2_comprehension.fit_rough_kmeans`).
Clustering **rough k-means** di $v$ in $K$ **situazioni**; $K$ ed $\varepsilon$ (margine di boundary)
selezionati **su validation** (silhouette + banda di boundary-fraction). Etichette core $z$.

**Bias di situazione** (`recommendation.fit_situation_biases_z`).
Per ogni situazione $k$ e categoria $m$, log-odds shrinkati e **z-scorati dentro la situazione**:
$$\tilde p[k,\cdot]=\frac{\text{count}[k,\cdot]+\lambda\,p_{\text{glob}}}{\sum(\cdot)},\quad
b[k,m]=\log\tilde p[k,m]-\log p_{\text{glob}}[m],\quad b_z=\frac{b-\mu_k}{\sigma_k}\;(\lambda{=}50)$$
→ $\kappa$ interpretabile come "nudge additivo di $\approx\kappa$ deviazioni standard".

**Combiner** (il cuore, `battery_bfull.py:59`).
$$\boxed{\;\hat s_{r,j} = s^{\text{backbone}}_{r,j} \;+\; \kappa\cdot\gamma_r\cdot \tilde b\big[r,\,c(j)\big]\;}$$
dove $\tilde b[r,\cdot]=\text{mem}_r@b_z$ (soft-assignment della richiesta alle situazioni), $\gamma_r$ =
inverso del n° di componenti, $c(j)$ = categoria dell'item $j$. **Additivo e per-categoria** → la spiegazione
*è* la formula: "in questa situazione la categoria X riceve $+\kappa\gamma b$". $\kappa$ selezionato su val.
Item già visti → $-\infty$.

**L3 — descrittivo** (boundary disambiguation): testato ma **inerte** nel combiner (dichiarato).

---

## 3. Design sperimentale — la tassonomia a due livelli

L'esperimento non è "SIT batte BASE" (troppo debole), ma **due contrasti** che separano tre esiti:

- **L1 = SIT − BASE** → *il segnale situazionale esiste?* (X-SAGE aiuta oltre il backbone)
- **L2 = SIT − Steck-b** → *batte la personalizzazione STATICA?* dove **Steck-b** = calibrazione per-utente
  sul prior storico di categoria dell'utente (à la Steck 2018), calcolata con la **stessa** formula di
  $b_z$ ma con etichetta = utente invece che situazione. È il baseline di non-ridondanza.

**Caselle** (regola unica, su macro-Cat-MRR, per backbone):
$$L_k\text{-pass}=(\Delta_k>0)\wedge(\text{CI bootstrap esclude }0)\wedge(\text{5/5 seed concordi})$$
- ¬L1 → **nullo** (nessun segnale) · L1∧¬L2 → **ridondante** (aiuta, ma lo statico fa meglio) · L1∧L2 → **winner**.

**Setup.** 7 backbone (BPR, FM, EASE, DeepFM, AFM, FPMC, SASRec; **FM=focale**), 3 metodi (BASE/SIT/Steck-b),
**5 seed** {42–46}. k-core 10 uniforme, split temporale, leave-one-out. Train impara / val sceglie iper-
parametri e $\kappa$ / test misura una volta. **Anti-circolarità verificata nel codice**: finestra causale
`t'<t`, `hist[test]=train+val`, tutti gli stimatori ($\tilde c, W, b_z, P_u$) solo su train → l'item/categoria
held-out non tocca mai $v$.

**Metriche (21).**
- *Categoriali (faro):* Cat-MRR@20, Cat-NDCG@20, **macro-Cat-MRR@20** (media non-pesata sulle categorie con
  min-support $|R_c|\ge 20$) — è la metrica di **equità categoriale**, decide le caselle.
- *Accuratezza item:* HR/NDCG@{5,10,20}, MRR — **per-richiesta** primarie + `_u` user-balanced secondarie.
- *Esposizione:* Coverage, Gini, LT@20.
- *Costo:* JS (Jensen-Shannon tra prior-utente e istogramma-categoria della top-20; peggiora by-design).

**Statistica.**
- Rango **atteso sotto tie** (McSherry-Najork) come primario; conteggi tie in cache → stretto ri-derivabile
  (misurato: impatto ~0 sulle @K).
- `mean ± se_boot` (bootstrap B=1500, ricampiona l'unità coerente). `sd_seed` cross-seed riportata.
- **Significatività primaria = bootstrap-Δ per-richiesta**; **t cross-seed = robustezza**; **consistenza =
  5/5 seed**. Il bootstrap-p è quasi-vacuo (n=migliaia) → **il lead è effect size + consistenza-seed**, gli
  asterischi Holm sono il pavimento.
- **Holm a 2 famiglie** (solo primari): primaria `FM×macroCatMRR×{L1,L2}×dataset`; robustezza-backbone
  separata. **TOST ±0.005** (HR@20/NDCG@20) per i claim di equivalenza.

**Cache-once / derive-many.** Layer costoso (preprocess→train→score→ranking) una volta → cache grezza
per-richiesta (rk, catrk, tie-counts, top-50, prior-utente); ogni metrica/@K/min-support si ri-deriva in
minuti (`aggregate_record.py`), con **round-trip bit-identico** verificato.

---

## 4. Risultati (mappa a domini)

Sul **focale FM**, metrica-faro macro-Cat-MRR, 5 seed:

| dominio | dataset | ΔL1 (SIT−BASE) | ΔL2 (SIT−Steck-b) | casella |
|---|---|---|---|---|
| film | **ml1m** | +0.0040 (5/5) | **+0.0053 (5/5)** | **WINNER** |
| POI | nyc_tist | +0.0075 (5/5) | −0.0200 (0/5) | ridondante |
| POI | saopaulo | +0.0075 (5/5) | −0.0256 (0/5) | ridondante |
| video (feed) | kuairand | +0.0001 (4/5) | −0.0038 (0/5) | **nullo** |
| retail / news / business | amazoncd / mind / yelp | *(in esecuzione)* | | atteso ridondante-non-POI / null |

**Su ml1m il winner è robusto su TUTTI e 7 i backbone** (ΔL2>0 ovunque, 5/5) — non è un artefatto del focale.

**Letture chiave:**
1. **ml1m winner**: il gusto-film è situazionale (contesto temporale) → SIT batte sia il backbone sia lo
   statico Steck-b, su ogni backbone.
2. **POI ridondanti**: SIT aiuta il backbone (L1>0) **ma perde contro Steck-b** (L2<0) → il gusto-luogo è
   stabile per-utente, la situazione non aggiunge. *Onesto: non è fallimento, è il confine del metodo.*
3. **kuairand null (controprova)**: esposizione randomizzata → ΔL1 ≈ **+0.0001** (~40× più piccolo di ml1m).
   Il metodo **non fabbrica un winner** dove non c'è segnale → questo *rende credibile* il winner di ml1m.
4. **Accuratezza item**: Pareto su ml1m (categoria↑ *e* item↑ leggero); su POI piccolo costo item,
   **eterogeneo per backbone** (max su EASE, ~0 su FPMC/SASRec, dichiarato via TOST). Mai mediato.
5. **Esposizione**: Coverage↑ / Gini↓ / LT@20↑ **ovunque** (X-SAGE distribuisce meglio il catalogo) — è la
   parte fairness "senza costo, simultanea".

---

## 5. La lente diagnostica (interpretabilità — case-study su ml1m)

Il valore forte di X-SAGE non è la magnitudine (piccola), è **essere diagnostico e fedele**:

- **Spiegazione fedele-per-costruzione**: il nudge è il termine additivo $\kappa\gamma\tilde b[c(j)]$ →
  la spiegazione ("categoria X spinta di $+\Delta$ in questa situazione") **non è post-hoc**, è il modello.
- **Situazioni nominate + spazio delle situazioni** (PCA): le $K$ situazioni sono ispezionabili (es.
  "commedia serale", "azione notturna"), con regioni e boundary visualizzabili.
- **lente KL (lensKL)**: deviazione KL per-situazione dalla distribuzione globale → localizza **sink/attrattori**
  (situazioni che concentrano o disperdono categorie), cioè *dove* c'è iniquità.
- **Fairness diagnostica > correttiva**: la lente **localizza** l'iniquità di categoria per-situazione (e
  per-gruppo, es. per-genere su ml1m); non pretende di correggerla a forza. È il claim difendibile.
- **Costo dichiarato (JS)**: X-SAGE si scosta dal gusto-utente per servire lo scopo situazionale → riportato
  come trade-off, non nascosto (Steck-b, che calibra sull'utente, ha JS più bassa — coerente).

---

## 6. Cosa NON è (limiti onesti — per il revisore)

- **Non è una legge predittiva**: niente LODO/held-out con soglie a-priori. È una **caratterizzazione
  empirica** descrittiva: le caselle si osservano e si spiegano col tipo di dato, non si predicono.
- **Magnitudini piccole**: ml1m ΔL1≈+0.004, ΔL2≈+0.005 — significative e consistenti (5/5) ma **modeste**.
- **Bootstrap near-vacuo**: con n=migliaia quasi ogni Δ è "significativo" → non vendiamo gli asterischi;
  il criterio vero è effect-size + consistenza-seed + il gate a 3 condizioni.
- **Varianza-seed minima sui backbone cachati** (deterministici + clustering stabile su ml1m: sd~4e-6):
  lì il "5/5" è quasi-vacuo → la casella poggia sul bootstrap; ma il **focale FM ha sd>0 ovunque** → il
  winner di testa è seed-protetto.
- **Eterogeneità backbone**: 7 backbone solo su ml1m/nyc/sao; gli altri (kuairand, amazoncd, mind, yelp) a
  2 backbone (BPR, FM). La casella si decide sul focale FM, presente ovunque → comparabile.
- **Interpretabilità dimostrata come case-study su ml1m**, non su tutti i dataset (applicabile in linea di
  principio, non dimostrata con figure ovunque).

---

## 7. In una frase

X-SAGE è un **re-ranker situazionale additivo, esplicito e fedele-per-costruzione**, che tratta la situazione
come **contesto + intento**; su una batteria a 7 backbone / 5 seed con statistica blindata (rango atteso,
bootstrap+consistenza-seed, Holm a 2 famiglie, controprova su dati senza segnale) produce una
**caratterizzazione onesta**: *vince dove il gusto è situazionale (film), è ridondante dove è stabile
per-utente (luoghi), è nullo dove non c'è segnale (feed)* — con una **lente che localizza l'iniquità di
categoria** invece di limitarsi a ottimizzarla.

---

### Domande su cui vorremmo il parere di un revisore
1. La caratterizzazione "winner/ridondante/null per tipo di dato" è un contributo sufficiente, o serve
   spingere verso una legge predittiva (che abbiamo deliberatamente evitato)?
2. Con magnitudini così piccole ma consistenti, il framing "diagnostico + esplicito" regge come contributo
   principale, o l'accuratezza modesta è un tallone?
3. La coppia L1/L2 (vs BASE e vs Steck-b) è il modo giusto di isolare "situazionale vs statico"? Steck-b è il
   baseline di non-ridondanza corretto?
4. Il tie-break atteso + bootstrap-primario + consistenza-seed 5/5 è un protocollo statistico difendibile, o
   sovra/sotto-ingegnerizzato?
