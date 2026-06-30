# X-SAGE — Definizioni precise di TUTTE le metriche (fedeli al codice eseguito)

Sorgente: `scripts/yelp/results_record.py` (funzioni `per_request_eval`, `per_req_quantity`,
`metrics_from`, `macro_of`, `gini`, `pu_mean`, `boot_paired`) + `long_tail_groups`
(`pipeline/step02_models/xsage/metrics.py`). Questo file serve per il check di coerenza vs SOTA.

---

## 0. Notazione e protocollo (comune a tutte le metriche)

- **Richieste di test** $r \in \mathcal{R}$. Protocollo **leave-one-out / next-item**: ogni richiesta ha
  **esattamente 1 item rilevante** $i_r$ (item held-out) per l'utente $u_r$ in un dato contesto/situazione.
- Il modello produce uno **score** $s_{r,j}$ per ogni item del catalogo $j \in \{1,\dots,N\}$.
- **Mascheramento**: gli item già visti da $u_r$ in training sono posti a $-\infty$ (esclusi dal ranking).
  Ranking **full** sul catalogo non visto (no sampling di negativi). *(Scelta rigorosa, non un fianco:
  Krichene & Rendle, KDD 2020, "On Sampled Metrics for Item Recommendation", mostra che le metriche
  campionate sono inconsistenti → il full-ranking è la scelta corretta.)*
- **Rango dell'item vero** — conteggio: $b_r=|\{j:s_{r,j}>s_{r,i_r}\}|$, $g_r=|\{j:s_{r,j}=s_{r,i_r}\}|$ (tie).
  Il **rango RIPORTATO è il rango ATTESO sotto tie** (mid-rank, **McSherry & Najork 2008**): l'item-vero è
  uniforme sui posti $[b_r+1,\,b_r+g_r]$. Le metriche di rango si calcolano come **valore atteso sul
  blocco di tie** (forme chiuse esatte in `PIPELINE_VALIDAZIONE.md` FASE 5). Senza tie ($g_r=1$) coincide
  col rango stretto $b_r+1$. *(Lo stretto `>` resta in cache come conteggio; il riportato è l'atteso.)*
- Categoria dell'item $j$: $c(j)$ (mappa `icm`). Categoria vera: $t_r = c(i_r)$. **Rango categoria** =
  conteggio analogo sul punteggio del miglior item di $t_r$: $bc_r=|\{j:s_{r,j}>\max_{c(j')=t_r}s_{r,j'}\}|$,
  $gc_r$ = tie a quel punteggio; rango categoria = **atteso** su $[bc_r+1,\,bc_r+gc_r]$.
- **Lista top-K**: $L_r^{K}$ = i $K$ item con score più alto (dopo il mascheramento).
- **Aggregazione**: accuratezza item in **DUE versioni** — **per-richiesta primaria** (standard,
  Voorhees/NCF) + **per-utente `_u` secondaria** (user-balanced); categoriali ed esposizione per-richiesta.
- $K_{\text{report}} = 20$ per le metriche categoriali ed esposizione; HR/NDCG riportate a $K\in\{5,10,20\}$.

---

## 1. Metriche categoriali (lente di equità X-SAGE)

> **STATUS: NOSTRE (varianti novel).** `catrk` = rango del miglior item della categoria vera =
> "reciproco del rango della prima categoria corretta". **Non** è una metrica standard con riferimento
> diretto: è un **adattamento dell'MRR/NDCG alla rilevanza a livello di categoria**. Da dichiarare
> esplicitamente come tale (un revisore di metrica incalza proprio qui). Parenti concettuali citabili:
> metriche subtopic/aspect — **α-NDCG (Clarke et al., SIGIR 2008)**, **IA-metrics (Agrawal et al., WSDM 2009)**.

### 1.1 Cat-MRR@K  (K=20)  — *nostra (MRR adattato a categoria)*
$$ \text{Cat-MRR@}K \;=\; \frac{1}{|\mathcal{R}|}\sum_{r}\; \frac{1}{\mathrm{catrk}_r}\,\mathbf{1}[\mathrm{catrk}_r \le K] $$
Reciprocal rank applicato al **target di categoria** (troncato a $K$). Base: MRR — Voorhees (1999).
Dichiarazione: *"adattiamo l'MRR alla rilevanza a livello di categoria"* (non metrica nota).

### 1.2 Cat-NDCG@K  (K=20)  — *nostra (NDCG adattato a categoria)*
$$ \text{Cat-NDCG@}K \;=\; \frac{1}{|\mathcal{R}|}\sum_{r}\; \frac{1}{\log_2(\mathrm{catrk}_r+1)}\,\mathbf{1}[\mathrm{catrk}_r \le K] $$
NDCG a singolo rilevante, gain binario, $\mathrm{IDCG}=1$ (categoria al rango 1 → $1/\log_2 2 = 1$). Base: Järvelin & Kekäläinen (2002).

### 1.3 macro-Cat-MRR@K  — *nostra (macro-avg della Cat-MRR), metrica-faro del contributo*
$$ \text{macro-Cat-MRR@}K \;=\; \frac{1}{|\mathcal{C}^{+}_m|}\sum_{c:\,|\mathcal{R}_c|\ge m}\; \frac{1}{|\mathcal{R}_c|}\sum_{r:\,t_r=c} \overline{cm}_r \qquad \overline{cm}_r=\text{Cat-MRR@}K\text{ (rango atteso) della richiesta }r $$
$\mathcal{R}_c=\{r:t_r=c\}$; somma sulle categorie con **min-support** $|\mathcal{R}_c|\ge m$ (**$m=20$ a-priori**,
stabilità verificata su $m\in\{20,50\}$). **Macro-averaging** (media non pesata sulle categorie): ogni
categoria pesa uguale → mitiga il dominio delle categorie di testa, e il min-support toglie il rumore
delle categorie rare. Citabile come macro-averaging: **Sebastiani (2002)**.

> ⚠️ **Caveat interpretativo (da blindare).** La macro-Cat-MRR dà **peso uguale a categorie con
> pochissime richieste** ($|\mathcal{R}_c|$ piccolo): quelle hanno varianza altissima e dominano la media
> macro (è lo stesso meccanismo che ha reso instabile Amazon a campione piccolo). Quando la usi come
> metrica-faro di equità, **riporta anche $|\mathcal{R}_c|$ per categoria** (e/o un macro-avg pesato come
> confronto), altrimenti un revisore obietta che la metrica-faro è rumore sulle categorie rare.

> Nota: `cm` nel codice = $\frac{1}{\mathrm{catrk}}\mathbf{1}[\mathrm{catrk}\le 20]$ per richiesta; `cn` = versione NDCG.

---

## 2. Metriche di accuratezza standard (item vero) — **doppia media**

> **STATUS: STANDARD (primarie) + variante dichiarata (secondarie).** Riportiamo ogni metrica in
> **due aggregazioni**, per togliere l'obiezione di replicabilità:
> - **Primaria — per-RICHIESTA** (colonne `MRR/HR@K/NDCG@K`): è la formula standard (Voorhees;
>   leave-one-out di NCF/SASRec/BERT4Rec), media su tutte le richieste/istanze held-out. **Usare questa
>   per il confronto col filone recommender.**
> - **Secondaria — per-UTENTE / user-balanced** (colonne con suffisso `_u`): media per-utente poi sugli
>   utenti (gli utenti pesano uguale). È una **variante** (non la formula standard) e va **dichiarata**.
>
> Lo scarto è materiale (nyc ≈ 4.3 richieste/utente): es. HR@20 di B_blind = **0.066 per-richiesta** vs
> **0.089 user-balanced** (+36% rel.). Chi replica con la formula standard ottiene la per-richiesta.

Sia $q_r$ la quantità per-richiesta: $q_r = 1/\mathrm{rk}_r$ (MRR), $\mathbf{1}[\mathrm{rk}_r\le K]$ (HR),
$\frac{1}{\log_2(\mathrm{rk}_r+1)}\mathbf{1}[\mathrm{rk}_r\le K]$ (NDCG).

**Primaria (per-richiesta, standard):**
$$ M \;=\; \frac{1}{|\mathcal{R}|}\sum_{r\in\mathcal{R}} q_r $$
**Secondaria (per-utente, user-balanced, `_u`):**
$$ M_u \;=\; \frac{1}{|\mathcal{U}|}\sum_{u} \frac{1}{|\mathcal{R}_u|}\sum_{r\in\mathcal{R}_u} q_r $$

### 2.1 MRR (item):  $q_r = 1/\mathrm{rk}_r$
### 2.2 HR@K (K=5,10,20):  $q_r = \mathbf{1}[\mathrm{rk}_r\le K]$ — con 1 rilevante **HR@K = Recall@K = Hit@K**
### 2.3 NDCG@K (K=5,10,20):  $q_r = \frac{1}{\log_2(\mathrm{rk}_r+1)}\mathbf{1}[\mathrm{rk}_r\le K]$, IDCG=1

**SOTA**: protocollo leave-one-out HR/NDCG — NCF (He et al. 2017), SASRec (Kang & McAuley 2018),
BERT4Rec (Sun et al. 2019); NDCG — Järvelin & Kekäläinen (2002); MRR — Voorhees (1999). Il bootstrap
ricampiona l'unità coerente con la media: **richieste** per le primarie, **utenti** per le `_u`.

---

## 3. Metriche di esposizione / beyond-accuracy

Esposizione aggregata: $e_j = \bigl|\{\, r : j \in L_r^{20} \,\}\bigr|$ (quante volte l'item $j$ compare in una top-20).

### 3.1 Coverage (catalog coverage / aggregate diversity)
$$ \text{Coverage} \;=\; \frac{1}{N}\Bigl|\{\, j : e_j > 0 \,\}\Bigr| \;=\; \frac{\bigl|\bigcup_r L_r^{20}\bigr|}{N} $$
Frazione del catalogo che appare in **almeno una** top-20.
**SOTA**: catalog coverage / aggregate diversity — Herlocker et al. (2004); Adomavicius & Kwon (2012).

### 3.2 Gini (concentrazione dell'esposizione)
Con $x_1 \le x_2 \le \dots \le x_N$ = gli $e_j$ ordinati crescenti, $S=\sum_j x_j$:
$$ \text{Gini} \;=\; \frac{2\sum_{i=1}^{N} i\,x_i}{N\,S} \;-\; \frac{N+1}{N} \qquad (0=\text{esposizione uniforme},\; \to 1=\text{concentrata}) $$
**SOTA**: formula = **coefficiente di Gini classico** (formula discreta standard). Uso come
concentrazione di esposizione/popolarità nei RecSys: **Abdollahpouri et al. (2019)**;
**Mansoury et al. (2020, FairMatch)**. *(NON citare Vargas & Castells 2011 qui: è novelty/diversity —
semmai vale per ILD/novelty, non per il Gini.)*

### 3.3 LT@20 (long-tail share)
Sia $G_1$ = item **fuori dal top-20% per popolarità** (popolarità = conteggio interazioni train+val;
`short_head_share = 0.20`, $n_{\text{head}}=\lceil 0.20\,N\rceil$; $G_1$ = "coda lunga", l'80% meno popolare).
$$ \text{LT@20} \;=\; \frac{1}{|\mathcal{R}|}\sum_{r}\; \frac{1}{20}\sum_{j \in L_r^{20}} \mathbf{1}[\,j \in G_1\,] $$
Frazione media di item di coda-lunga nelle top-20.
**SOTA**: split short-head/long-tail (cut-off 20%) — Celma (2009); Abdollahpouri et al. (2017,
"Controlling Popularity Bias in Learning-to-Rank"); Yin et al. (2012).

### 3.4 JS — divergenza di calibrazione (costo dichiarato)
Sia $P_u\in\Delta^M$ il **prior di categoria dell'utente** (frequenze storiche su train, normalizzate) e
$Q_r\in\Delta^M$ l'**istogramma di categoria della top-20** della richiesta $r$. Costo medio:
$$ \text{JS} \;=\; \frac{1}{|\mathcal{R}|}\sum_r \text{JSD}(P_{u_r}\,\|\,Q_r),\quad \text{JSD}(P\|Q)=\tfrac12 D_{KL}(P\|\!M)+\tfrac12 D_{KL}(Q\|\!M),\ M=\tfrac12(P{+}Q)\ \text{(bit)} $$
**Interpretazione:** misura quanto la lista si scosta dal gusto-categoria storico dell'utente. X-SAGE
**peggiora la JS by-design** (sposta verso lo scopo situazionale, non verso la calibrazione-utente):
si riporta come **trade-off dichiarato**, non si nasconde. *(Verificato: Steck-b — calibratore-utente —
ha JS più bassa di SIT su tutti i backbone, conferma di correttezza.)* **SOTA**: calibrazione delle
raccomandazioni — Steck (2018), "Calibrated Recommendations".

---

## 4. Incertezza e significatività (come riportate nel CSV)

- **Seed**: 5 ripetizioni $\{42,43,44,45,46\}$. `mean` = media sui 5 seed; `sd_seed` = std campionaria
  (ddof=1) sui 5 seed. (= 0 per i backbone deterministici/cachati; il B_full retrainato per-seed dà sd>0.)
- **Bootstrap** (`B=1500`, su seed 42): si **ricampionano le unità** — utenti per le metriche di
  accuratezza, richieste per categoriali/esposizione (per Coverage/Gini si ricalcola $e_j$ sul resample) —
  e si ricalcola la metrica. → `ci_lo,ci_hi` = percentili [2.5, 97.5]; `se_boot` = **deviazione standard
  della distribuzione bootstrap** (sempre > 0: incertezza onesta anche quando `sd_seed`=0).
  Tabella presentabile come $\text{mean} \pm \text{se\_boot}$.
- **Contrasti** (riga SIT): **L1 = SIT−BASE**, **L2 = SIT−Steck-b**. Due fonti di rumore, due strumenti:
  - **`p_lX` = bootstrap-Δ a due code** $p = 2\min(P(\Delta^*\le0),P(\Delta^*\ge0))$ — **PRIMARIO** (potenza, n=migliaia) → Holm.
  - **`p_lX_seed` = t appaiato cross-seed** (df=4) — robustezza conservativa, **non** gate.
  - **`seeds_lX` = #seed con Δ>0** (0–5) — consistenza.
- **Casella** (regola UNICA, su macro-Cat-MRR, per backbone): $L_k\text{-pass} = (\Delta_k>0)\wedge(\text{CI bootstrap esclude }0)\wedge(\text{seeds}_k=5)$;
  `nullo` se ¬L1, `ridondante` se L1∧¬L2, `winner` se L1∧L2. Il **5/5 è a priori**.
- **Holm–Bonferroni a due famiglie (solo primari)**: PRIMARIA `B_full×macroCatMRR×{L1,L2}×7`=**14**; ROBUSTEZZA-backbone separata.
  `sig` = `***`<.001, `**`<.01, `*`<.05.
- **Onestà:** il bootstrap-p è quasi-sempre ≈0 (near-vacuo) → **lead su effect size + consistenza-seed**, gli asterischi sono il *pavimento*.
- **TOST** (equivalenza, ±0.005) su HR@20 e NDCG@20: equivalente se il CI bootstrap di $\Delta \subset (-0.005,+0.005)$.

**SOTA**: bootstrap percentile — Efron & Tibshirani (1993); Holm (1979); TOST — Schuirmann (1987),
Lakens (2017); tie-break atteso — McSherry & Najork (2008).

---

## 5. Dichiarazione di citabilità (per il revisore di metrica)

### 5.1 Classificazione di ogni metrica
| Metrica | Status | Riferimento da citare |
|---|---|---|
| Cat-MRR@20 | **NOSTRA** (MRR adattato a categoria) | base MRR Voorhees 1999; affini α-NDCG Clarke 2008, IA Agrawal 2009 |
| Cat-NDCG@20 | **NOSTRA** (NDCG adattato a categoria) | base NDCG Järvelin & Kekäläinen 2002 |
| macro-Cat-MRR@20 | **NOSTRA** (macro-avg della Cat-MRR) | macro-averaging Sebastiani 2002 |
| MRR / HR@K / NDCG@K (per-richiesta) | **STANDARD** | Voorhees 1999; NCF He 2017; SASRec 2018; BERT4Rec 2019 |
| MRR_u / HR@K_u / NDCG@K_u (per-utente) | **VARIANTE dichiarata** (user-balanced) | — (scelta motivata) |
| Coverage | STANDARD | Herlocker 2004; Adomavicius & Kwon 2012 (TKDE) |
| Gini (esposizione) | STANDARD (formula classica) | Gini classico; uso RecSys Abdollahpouri 2019 / Mansoury 2020 |
| LT@20 | STANDARD (cutoff 20% = convenzione) | Celma 2009; Abdollahpouri 2017 |
| Bootstrap CI/SE, Holm, TOST | STANDARD | Efron & Tibshirani 1993; Holm 1979; Schuirmann 1987 / Lakens 2017 |
| Full-ranking, no neg-sampling | STANDARD (scelta rigorosa) | Krichene & Rendle 2020 |

### 5.2 I tre scostamenti DA DICHIARARE (non nascondere)
1. **catrk / Cat-MRR / Cat-NDCG / macro-Cat-MRR sono nostre** (adattamento alla rilevanza di categoria).
   Scrivere *"adattiamo MRR/NDCG a livello di categoria"*, citare gli affini (Clarke 2008, Agrawal 2009).
2. **Tie-break ottimistico** (`>` stretto in $\mathrm{rk}$ e $\mathrm{catrk}$): l'item/categoria vero è
   posto **sopra** i pari-score (caso *best-case*). Lo SOTA per i pari-score raccomanda il rango
   *atteso/medio*: **McSherry & Najork (ECIR 2008)**, "Computing IR Performance Measures Efficiently in the
   Presence of Tied Scores".
   **Verificato empiricamente su catalogo grande (nyc, n_items≈4k) E denso (ml1m, n_items≈3.3k),
   seed 42, B_blind + EASE**: i tie all'item-vero sono *frequenti* (38–73% delle richieste, inflazione
   media ~13–18) MA **profondi** (rank ≫ 20): l'item-vero cade in un blocco a basso score con già >20 item
   sopra → fuori top-20 a prescindere dal tie-break.
   - **Metriche troncate @20 (HR@20, NDCG@20, Cat-MRR@20)**: **invarianti** stretto-vs-atteso (rango medio
     McSherry-Najork) entro ~$10^{-3}$. Es. ml1m Cat-MRR Δ(SIT−BASE): B_blind +0.02517 → +0.02517 (identico),
     EASE +0.02485 → +0.02554; HR@20 Δ B_blind +0.00397 → +0.00397. nyc idem.
   - **MRR-item NON troncato** (l'unica metrica che il troncamento non protegge): **misurata**. Scarto
     BASE stretto-vs-atteso = +0.00000 (B_blind) / +0.00005 (EASE); Δ(SIT−BASE) invariante a ~$3\cdot10^{-5}$.
     Motivo: le richieste ad alto MRR (item in cima) non hanno tie; quelle con tie sono profonde dove $1/p$
     è minuscolo → l'aggregato non si muove.
   → **Citare McSherry & Najork e dichiarare l'impatto misurato ~nullo su TUTTE le metriche riportate**
   (troncate e MRR pieno), con le cifre sopra. Non liquidare a parole.
3. **Doppia media accuratezza** (per-richiesta primaria + user-balanced `_u`): la `_u` è una variante,
   non la formula standard; scarto materiale (+~30% rel., vedi §2). Dichiarata e riportata in parallelo.

### 5.3 Note di coerenza già allineate
- **HR@K = Recall@K = Hit@K** sotto 1 rilevante per richiesta (stessa quantità).
- **NDCG / Cat-NDCG a singolo rilevante** → $\mathrm{IDCG}=1$ (DCG troncato), coerente con leave-one-out.
- **Coverage/Gini/LT** calcolate sulle **top-20** (esposizione), non sull'intero ranking.
- **Aggregazione**: accuratezza in **due** versioni (per-richiesta + per-utente); categoriali/esposizione per-richiesta.
