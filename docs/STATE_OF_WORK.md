# X-SAGE — Stato del lavoro (documento vivo)

> Documento canonico dello stato del progetto. **Va aggiornato e committato a ogni
> modifica sostanziale**, così la storia git riflette cosa stiamo facendo.
> Ultimo aggiornamento: 2026-06-22.

## Changelog (sintetico — il dettaglio è nei commit)
- Selezione congiunta parametri situazioni (overnight) → poi corretta (ε degenere → boundary-band).
- Selezione finale K (criteri interni + tetto |A|+2), ε (banda), depth (trade-off), n (sensibilità).
- Fronte fairness/sink **bocciato** (7 diagnostiche) → ri-framing del contributo.
- Contributo CORE: **enhancer categoriale situazionale** — significativo (5/5), specifico, robusto al cutoff K.
- `γ_S = 1/|T|` applicato (sostituisce 0.5 hardcoded).
- L3 projection: descrittivo/leggibile ma non predittivo (≈persistenza); inerte sul re-ranking.
- Ablazione intento (FULL vs CTX): **intento costitutivo 4/5, dannoso a Sao Paulo**.

---

## 0. Cos'è X-SAGE, oggi
Re-ranker **additivo situation-aware**: corregge i punteggi di un backbone in base alla
**situazione** della richiesta. **Framing attuale**: *enhancer di rango/calibrazione
CATEGORIALE situazionale, interpretabile* — NON un fair re-ranker (fronte fairness chiuso
in negativo, vedi §7).

## 1. I dati
- TIST2015 (Foursquare check-in). 5 città (low→high transit): Istanbul, Bangkok, NYC, Sao Paulo, Tokyo.
- k-core=10, split **temporale per-utente 80/10/10** (causale). Backbone FM importato con checksum (locale, gitignored).
- **Richiesta** = (utente `u`, istante `t`) con next-item vero `i_target` nascosto. Campi: `cat_macro` (~10 macro), contesto (`c_hour, c_dow, c_isweekend, c_month, geohash5, intent_last_cat`).

## 2. Flusso dati passo-passo (formule + file + parametri finali)

| # | stadio | file | formula | parametri |
|---|---|---|---|---|
| 0 | Backbone | `xsage/data.py` | `s_B(u,i)` = score FM (context-blind) | FM |
| 1 | L0 Sensing | `xsage/l0_sensing.py` | ultime `n` macro con `time<t` (stretto) | n=3 |
| 2a | L1a Context c̃ | `xsage/l1_perception.py` | `θ_a = 1 − H(p_foglia)/log₂(K_mac)`; c̃=(θ_a)_{a=1..6} | depth=3, min_leaf=200 |
| 2b | L1b Intent e | `xsage/l1_perception.py` | W=+1 row-norm; attrattori `indeg≥media`; `m_c∝Σ_j γ^j 1[macro_j=c]`; `e∝(m·Σ_{k=1..H}β^k W^k)⊙1[∈A]` rinorm | γ=0.4, β=0.7, H=2 |
| 3 | L2 Situazioni | `xsage/l2_comprehension.py` | rough k-means su `v=[c̃‖e]`: core `r_{k*}=1`, boundary `r_k=1/\|T\|` | K,ε per città |
| 4 | Bias b̃^(k) | `xsage/recommendation.py:fit_situation_biases_z` | log-odds smussato vs globale, **z-scored** per situazione | α=50 |
| 5 | L3 Projection | `xsage/l3_projection.py` | T transizioni; `r̃_k∝r_k·T_{z_prev,k}` (eq.18) | **non nel path** (§6) |
| 6 | Combiner | `xsage/recommendation.py:unified_combine_scores` | `ŝ=s_B+κ·m_sel·γ_S·Σ_k r_k(b̃^(k)_{c(i)}+λ·b^LT_{c(i)})` | **γ_S=1/\|T\|** |
| 7 | Ranking | `xsage/pipeline.py` | maschera item visti, top-20 | K_TOP=20 |
| 8 | Metriche | `xsage/metrics.py` + script exp | §4 | short_head=0.20 |

**K finali/città:** ist 5, bkk 6, nyc 6, sao 3, tky 4. **ε finali/città:** .01, .01, .02, .05, .07.
**Config CORE (SIT):** `m_sel≡1`, `λ=0`, `κ=0.25` → **`ŝ = s_B + 0.25·γ_S·Σ_k r_k·b̃^(k)_{c(i)}`**.

## 3. Cosa fa ciascuna parte (una riga)
- **c̃**: quanto ogni attributo di contesto è informativo sulla prossima categoria.
- **e**: verso quale categoria-attrattore tende l'utente (storia recente nel grafo macro).
- **Situazione**: cluster su [contesto‖intento]; membership `r` per richiesta.
- **b̃^(k)**: la direzione di categoria preferita dalla situazione k (appresa dai next-item di training).
- **Combiner**: somma a `s_B` una spinta verso le categorie della situazione (scalata κ, γ_S).
- **γ_S=1/|T|**: smorza la spinta sulle richieste boundary (ambigue tra |T| situazioni).

## 4. Metriche
| metrica | cosa | ruolo |
|---|---|---|
| R@20, NDCG@20 | item vero nel top-20 (per-utente) | controllo accuracy |
| **Cat-MRR / Cat-NDCG** | posizione della *categoria vera* del next-item | **guida del CORE** |
| JS-Calibration | `JS(p(g\|s)‖q(g\|top-K))`, target p da TRAIN | Steck ri-targettata |
| LT@20, Coverage, Gini | esposizione coda lunga | **DELIM** (fairness) |

## 5. Cosa è VALIDATO (CORE) e con quale processo (anti-circolare)
| elemento | processo | artefatto |
|---|---|---|
| integrità dati | conteggi, k-core su set completo | `A1_*.csv` |
| sensing causale | 0 violazioni 5/5 | (test) |
| logica perception | 8/8 verifiche numeriche | `perception_param_sensitivity.csv` |
| **K/città** | criteri interni silhouette/CH/DB + tetto **\|A\|+2** | `K_final_rule.csv`, `K_ceiling_candidates.csv` |
| **ε/città** | vincolo **boundary-band [10%,30%]** + plateau-maximin | `epsilon_final.csv` |
| depth=3 | trade-off interpretabilità/geometria/stabilità | `depth_tradeoff.csv` |
| n=3 | sensibilità (bordo innocuo) | `n_sensitivity.csv` |
| **γ_S=1/\|T\|** | formula; effetto <0.0002 (\|T\|=2 nell'87-94%) | `boundary_structural.csv` |
| **Risultato categoriale** | bootstrap 2000: Cat-MRR/NDCG **5/5**, calibration **4/5** | `bootstrap_categorical.csv` |
| **Specificità SIT>UNI_mean** | bootstrap **5/5** a ogni K | `bootstrap_categorical.csv`, `k_robustness_categorical.csv` |
| **Robustezza cutoff K** | plateau K∈{5..100}, **25/25** | `k_robustness_categorical.csv` |
| **Intento costitutivo** | ablazione FULL vs CTX: **4/5 sì** (1 no) | `intent_ablation.csv` |

**Risultato CORE in sintesi:** il meccanismo situazionale migliora il **rango di categoria** in
modo **significativo, specifico, robusto al cutoff**. Effetto **piccolo** (+2–10%), **gratis**
(no costo item) su 3/5 città, con costo-item-in-cima su nyc/saopaulo che svanisce a K=100.
**L'intento è costitutivo in 4/5** (FULL>CTX signif.), **dannoso a Sao Paulo** (CTX meglio):
la definizione "situazione=contesto+intento" regge nella maggioranza, con un'eccezione onesta.

## 6. Aperto / non validato
| aperto | stato |
|---|---|
| **α=50, κ=0.25** | κ fissato a priori; **manca** curva di robustezza α/κ sul Cat-MRR |
| **vs CPFair sull'asse CALIBRAZIONE** | **non fatto** — head-to-head che prova "situazionale > uniforme sull'asse giusto" |
| **L3 Projection** | descrittivo/leggibile ma non predittivo (≈persistenza), inerte sul re-ranking. Decisione: descrittivo / nota / rimuovi — **pendente** |
| **Sao Paulo** | unica città dove l'intento danneggia → capire perché (over-steering?) |
| **SARE** | competitor latente, assente dal repo, non confrontato |
| **Documentazione** | docs 03–07 **vuote**; README e 02 **stale** (vecchio framing) |

## 7. DELIMITAZIONE (fairness morta, da scrivere come tale)
7 diagnostiche concordi: regola sink non difendibile (LT inerte, KL fragile); backbone
situazione-cieco; gate selettivo morto; timone debole; SIT **perde** contro l'uniforme
(`UNI_glob`) sul minimo intervento per la fairness. → "la situazione **non** è una leva di
fairness" è un **risultato negativo validato**. Artefatti: `sink_*.csv`, `minimal_intervention.csv`,
`corrective_decisive.csv`, `backbone_intent_alignment.csv`, `signal_*.csv`.

## 8. Confronti (related work)
- **CPFair/CPFairRank**: stessa famiglia additiva; `UNI_glob` ne è il gemello interno (batte SIT sulla fairness).
- **BankFair, Tax-rank**: dual/OT su esposizione globale; innesto situazionale frammenta.
- **CAPRI-FAIR**: unico innesto allineato (pesi di fusione per-situazione).
- **SARE**: competitor di paradigma (latente vs il nostro esplicito/interpretabile) — da confrontare.
- **UNI_mean** (direzione media): ablazione interna → SIT la batte signif. 5/5.

## 9. Prossimi passi (priorità)
1. **Decidere L3** (descrittivo / nota / rimuovi).
2. **Robustezza α/κ** sul Cat-MRR (chiude "ogni numero giustificato").
3. **Head-to-head vs CPFair sull'asse calibrazione** (non fairness).
4. **Capire Sao Paulo** (intento dannoso): proprietà o artefatto?
5. **Documentare il CORE** (docs 03/05/06) + riscrivere README e 02 + scrivere 07 (delimitazione).
6. (Futuro) confronto vs **SARE**.

## 10. Repo
**github.com/laliberti00/XSAGE** (privato). `main` = sorgente di verità (tutti i branch di
esperimento consolidati). Branch storici: `exp/situational-category-relevance`,
`fix/boundary-structural`, `abl/intent-constitutive`.
