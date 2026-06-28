# MAPPING — Framework SARS (Endsley + Trustworthy) ↔ codice X-SAGE + copertura trustworthiness

> **Natura:** mapping read-only (codice esistente + risultati già prodotti). Nessun esperimento.
> **Regola d'oro:** verdetto onesto a 3 livelli — ✅ IMPLEMENTATO (file+funzione reale) · 🟡 PARZIALE/PROXY · ❌ NON COPERTO (→ limiti/future work).
> **Fonti codice:** `xsage/` (pacchetto modello: l0_sensing, l1_perception, l2_comprehension, l3_projection, recommendation, pipeline), `scripts/` (preprocess, battery, explain). Modello B_full da `IntentAwareRS_thesis/pipeline/step02_models/xsage/backbone_full.py`.
> **Survey trustworthiness:** Ge et al. 2024, *"A survey on trustworthy recommender systems"*, ACM TORS 3(2) (`ge2024survey` in `sn-bibliography.bib`). **Nota:** nel repo NON esiste una checklist enumerata delle proprietà; la lista in Parte C è il set canonico di quella survey.

---

## PARTE A — ASSE 1: Situation Awareness (Endsley) ↔ codice

| stadio framework | formalizzazione | file : funzione (riga) | verdetto | note onestà |
|---|---|---|---|---|
| **L0 — SENSING** (raw → C₀) | finestra causale: ultime *n* macro prima di *t*, no leakage | `xsage/l0_sensing.py : build_recent_window` (58) — `cut = np.searchsorted(rec["t"], t, side="left")` (91) | ✅ | taglio causale stretto via searchsorted; ritorna `recent_macro`, `recent_dt_min`, `n_prior`. Nessun uso del futuro. |
| **L1a — LOW-LEVEL PERCEPTION φ₀** (raw → contesto osservabile) | informatività `θ_a = 1 − H/log₂K` per attributo | `xsage/l1_perception.py : fit_contribution_functions` (65) — `theta = 1.0 − H/log2K` (99); `.transform` (47) → c̃ ∈ [0,1]^A | ✅ | un albero shallow (depth=3) per attributo; attributi = `c_hour,c_dow,c_isweekend,c_month,prev_geohash5,intent_last_cat_idx`. θ z-scorato. |
| **L1b — HIGH-LEVEL PERCEPTION φ₁** ((c₀,u) → C₁) | descrittore `v=[c̃‖e]`: profilo recency `m` + intento `e` propagato su grafo macro `Σ β^k W^k` | profilo: `compute_profile` (143, decadimento γ^j); intento: `compute_intent` (169, `m @ Σβ^k W^k` su attrattori); grafo: `estimate_macro_transition` (113); attrattori: `find_attractors` (133); assemblaggio `v`: `scripts/mind/mind_prep.py:105` `np.concatenate([c,e])` | ✅ | include l'utente *u* via lo storico per-utente (recency profile). Propagazione a *H* hop esatti (`Wk = Wk@W`). |
| **L2 — COMPREHENSION f** ((C₁,u,Σ,G) → {(S_k,p_k)}) | rough k-means → situazioni S con membership p (core/boundary); bias `b̃^(k)` | `xsage/l2_comprehension.py : fit_rough_kmeans` (105), `_assign` (68, `competing = d−d_star ≤ eps`), membership (147–154: core r=1; boundary r=1/\|T\|); bias: `xsage/recommendation.py : fit_situation_biases_z` (35, log-odds shrinkati α, z-score) | ✅ | **S_k = `prototypes`** (centri situazione), **p_k = `membership`** (r_k). core/boundary espliciti (`is_boundary`, `competing_sets`). |
| **L3 — PROJECTION π** ((C₁,S,p,u) → (L,(S⁺,p⁺))) | matrice transizione situazioni T (Markov); lista L dal combiner | T: `xsage/l3_projection.py : estimate_transition` (17, `T = counts/row_sum`); disambiguazione opzionale: `boundary_disambiguate` (70); lista L: combiner (vedi sotto) | 🟡 | **ONESTÀ:** L3 è **descrittivo, NON ottimizza il re-ranking**. Testato in `scripts/.../boundary_structural.py` come A/B (eq.18 `r̃_k ∝ r_k·T_{z_prev,k}`) → ≈ persistenza, inerte/non migliora. NON cablato in `pipeline.py` di default. |
| **COMBINER** (produce L) | `ŝ = s_B + κ·m_sel·γ_S·Σ_k r_k·b̃^(k)` | `xsage/recommendation.py : unified_combine_scores` (74): `learned = membership @ b_z` (106), `nudge = (κ·m_sel·γ_S)·pref` (114-116), `return s_B + nudge` (117); γ_S: `pipeline.py:62-67` `1/max(\|T\|,1)` | ✅ | **κ=0 ⇒ ritorna s_B esatto** (matched-off, 102-103). γ_S = 1 su core, 1/\|T\| su boundary (gate di incertezza). m_sel = sink/core mask. |

**Sintesi A:** i 5 livelli Endsley sono **funzioni first-class** nel pacchetto `xsage/`. Unico verdetto non-pieno: **L3 (proiezione) è descrittivo/diagnostico**, non un ottimizzatore del ranking — dichiarato apertamente (→ future work: L3 come segnale di confidenza per un gate selettivo).

---

## PARTE B — ASSE 2: Trustworthy Principles (pipeline) ↔ codice

| blocco | sotto-voce | verdetto | evidenza-codice / "non coperto" |
|---|---|---|---|
| **1. DATA PREPARATION** (Trustworthy Data Processing) | Data cleaning | ✅ | k-core=10 uniforme (`scripts/*/preprocess_*.py`, es. `preprocess_ml1m.py:42-46`); dedup esatta `(u,i,utc)`; tz-localization (`MATH_WALKTHROUGH.md §1.1`). |
| | Data correction | ✅ | cold-filter val/test, NaN→0 haversine, UTC→local (`MATH_WALKTHROUGH.md §1.1.4/1.1.6`). |
| | Data debias | 🟡 | **PROXY:** "debias = anti-leakage" — split temporale per-utente 80/10/10 causale + intent strict-prior (`preprocess_*.py:56-60`; loro tabella `MATH_WALKTHROUGH §1.2`). **NON** c'è debiasing algoritmico/demografico né riponderazione di esposizione sui dati. |
| | Decentralization / Federated | ❌ | dichiarato "n/a, federated deferred" (`MATH_WALKTHROUGH §1.2`). → limiti. |
| **2. DATA REPRESENTATION** (Robust Data Representation) | Robust Representation Learning | 🟡 | descrittore `v=[c̃‖e]` + rough clustering (`l1_perception.py`, `l2_comprehension.py`): rappresentazione **interpretabile e a bassa dimensione**, ma "robust" non in senso adversarial. |
| | Federated Learning | ❌ | non coperto. |
| | **Representation Visualization** | ✅ **(forte)** | 3 generatori + 30 JSON: `scripts/yelp/situation_profiles.py` (situazioni nominate, b̃, lens), `situation_space.py` (PCA 2D, boundary fraction), `situation_transitions.py` (matrice T, persistenza). Output in `outputs_results/explain/`. |
| **3. RECOMMENDATION** (Fair & Transparent) | Robustness | 🟡 | pluggabilità su **7 backbone** (B_blind/B_full/EASE/SASRec/FPMC/DeepFM/AFM), anti-circolarità, multi-seed + bootstrap 1500 (`battery_bfull.py`). **No** adversarial/poisoning/distribution-shift test. |
| | Fairness | 🟡 | **vedi Parte D**: esposizione migliora vs backbone in modo consistente; equità-categoria migliora sui non-saturi; **amplifica sul saturo estremo (yelp)**. Diagnostica (lens), non correttiva. |
| | **Explainability** | ✅ **(forte)** | nudge **fedele-per-costruzione** (`unified_combine_scores`: il contributo è esattamente `κ·γ_S·Σ r_k·b̃`, ispezionabile); situazioni nominate (`situation_profiles`); lente per-situazione (`lens_spread`, `battery_bfull.py:81`). |
| | Adaptivity | 🟡 | la situazione adatta il ranking al contesto via combiner per-richiesta; gate di incertezza γ_S=1/\|T\|. **No** online/continual learning. |
| **4. EVALUATION** | Technical (accuracy + trustworthy eval) | ✅ | Cat-MRR micro+MACRO (`battery_bfull.py:full_eval`, `scripts/yelp/macro_avg.py`), R@20/NDCG@20, JS, LT@20, Coverage, Gini, bootstrap+TOST(±0.005)+t cross-seed; anti-circolarità (κ/K/ε su val). |
| | Ethical (Responsibility / Social Impact) | ❌ | nessuno studio di impatto sociale né responsibility audit. → limiti onesti. |

---

## PARTE C — Proprietà di trustworthiness (Ge et al. 2024, ACM TORS) — copertura

> Fonte: `ge2024survey` (sn-bibliography.bib). Set canonico della survey. Verifica nel repo per ogni voce.

| proprietà | copriamo? | COME / dove nel codice | forza evidenza |
|---|---|---|---|
| **Explainability / Transparency** | ✅ | spiegazione fedele-per-costruzione (`recommendation.py:unified_combine_scores`), situazioni esplicite e nominate (`situation_profiles.py` → JSON), lente di iniquità per-situazione (`battery_bfull.py:lens_spread`) | **alta** — punto di forza centrale, artefatti versionati |
| **Non-discrimination / Fairness** | 🟡 | esposizione vs backbone: Gini↓ **21/21**, Coverage↑ **21/21**, LT↑ **20/21** (Parte D); macro-Cat-MRR↑ sui non-saturi. MA amplifica su saturazione estrema (yelp); diagnostica via lens, non correttiva | media — vedi Parte D, con limite dichiarato |
| **Efficiency / Environmental well-being** | ✅ | `scripts/yelp/profile_cost.py` → **205 parametri** (K·macro+K·dim), **0.10 ms/richiesta**, FIT ~168 s (ml1m); modulo pluggable ordini di grandezza più piccolo di un backbone neurale | **alta** — argomento "green", numeri concreti in `cost_*.txt` |
| **Safety / Robustness** | 🟡 | pluggabilità 7 backbone, anti-circolarità, multi-seed+bootstrap+TOST. No adversarial/poisoning/shift | media |
| **Privacy** | 🟡 | la situazione richiede solo **continuità intra-sessione** (finestra recente causale, `l0_sensing.build_recent_window`), **non** identità persistente cross-sessione → vantaggio cookieless/argomentativo. MA nessuna privacy formale (DP) | bassa/argomentativa — onesto come claim di design, non garanzia formale |
| **Accountability / Controllability** | 🟡 | κ è un parametro **ispezionabile e regolabile** (controllabilità; `unified_combine_scores` argomento `kappa`); lente per-situazione = auditabilità. No audit formale | media |
| **Privacy formale / DP / Federated** | ❌ | non coperto (dichiarato deferred) | — (→ limiti) |

**Nota di metodo (onestà):** nel repo non esiste una checklist enumerata di queste proprietà; il framework *operazionalizzato* nei doc è **Endsley (L0–L3)** + il layer **Data→Rappresentazione→Raccomandazione→Valutazione**. La mappatura sopra allinea quel layer alle dimensioni della survey Ge et al. 2024.

---

## PARTE D — LA DOMANDA CENTRALE: X-SAGE migliora la FAIRNESS *vs il backbone* (BASE)?

> Solo confronto **SIT − BASE** (non vs uniforme). Dati: `outputs_results/battery_bfull_{nyc_tist,saopaulo,ml1m}.csv`, 7 backbone × 3 dataset = **21 combinazioni**, media 5 seed. (Macro-Cat-MRR: da `macro_avg_*.csv`, calcolata sul backbone B_blind.)

### D.1 — Direzione (esposizione vs backbone)
| effetto | conteggio | esito |
|---|---|---|
| **LT@20 sale** (coda lunga ↑) | **20/21** | unica eccezione: saopaulo/SASRec (−0.0026) |
| **Coverage sale** (catalogo servito ↑) | **21/21** | sempre |
| **Gini scende** (concentrazione ↓ = più equa) | **21/21** | sempre |
| **Cat-MRR micro sale** | **20/21** | unica eccezione: ml1m/SASRec (−0.0007, saturazione del sequenziale SOTA) |

→ **Rispetto al backbone, X-SAGE ridistribuisce l'esposizione in modo consistente** (Gini↓ e Coverage↑ su *tutte* le 21 combinazioni).

### D.2 — Costo: esposizione↑ con o senza costo di accuracy
- **Pareto / gratuito** (esposizione↑ **senza** degradare R@20, ΔR@20 ≥ −0.0005): **10/21** — concentrate sui **backbone forti / ml1m**: *ml1m* B_blind, B_full, EASE, DeepFM, AFM (tutti e 5); *saopaulo* B_blind, DeepFM, FPMC; *nyc* FPMC, SASRec.
- **Con trade-off** (esposizione↑ **ma** R@20 degrada): **11/21** — concentrate sui **backbone deboli / POI** dove κ* spinge forte: *nyc* B_blind/B_full/EASE/DeepFM/AFM (es. EASE: Coverage **+0.34** ma R@20 **−0.042**; DeepFM: Cov **+0.14**, R@20 **−0.017**); *saopaulo* B_full/EASE/AFM/SASRec; *ml1m* FPMC/SASRec.

→ **Sui backbone forti il guadagno di esposizione è Pareto (gratuito); sui deboli/POI ha un costo di accuracy** perché lì il combiner deve spingere di più per ridistribuire.

### D.3 — Il MECCANISMO (perché migliora l'esposizione)
Il combiner `ŝ = s_B + κ·γ_S·Σ_k r_k·b̃^(k)_c(i)` (`recommendation.py:unified_combine_scores:106-117`) **somma un bias per-categoria-della-situazione** agli score del backbone. Poiché `b̃^(k)` premia le **categorie tipiche della situazione** (non i soli popolari globali del backbone), il re-ranking **redistribuisce l'esposizione su più categorie/item** invece di concentrarla sulla testa popolare di `s_B`. Effetto diretto: Coverage↑, Gini↓, LT↑. Il gate `γ_S=1/\|T\|` (boundary) attenua la spinta nell'incertezza → la ridistribuzione è **mirata, non cieca**.

### D.4 — Il LIMITE onesto (fairness di esposizione, NON correttore di pozzi)
Questa è **fairness di esposizione** (LT/Coverage/Gini), **non** correzione strutturale dell'iniquità. La prova è la **macro-Cat-MRR** (equità di qualità *per-categoria*, demaschera l'amplificazione della maggioranza):
| dataset | saturazione | Δmacro-Cat-MRR (SIT−BASE) | lettura |
|---|---|---|---|
| nyc_tist | 0.26 | **+0.043** | migliora equità-categoria |
| saopaulo | 0.25 | **+0.023** | migliora equità-categoria |
| ml1m | 0.28 | **+0.014** | migliora equità-categoria |
| tsmc_tky | 0.57 | +0.008 | migliora (saturo moderato) |
| tokyo_tist | 0.63 | +0.015 | migliora (saturo moderato) |
| **yelp** | **0.76** | **−0.003** | **AMPLIFICA la maggioranza** (peggiora equità) |

→ **Onestà richiesta:** solo sul **saturo estremo (yelp, 76%)** la macro-Cat-MRR scende: lì il meccanismo spinge la categoria modale e **approfondisce il pozzo**. Sui saturi *moderati* (tokyo/tsmc, ~57-63%) migliora ancora. Quindi: **non è un correttore di fairness**, ed è diagnostico (la lente lo *misura*), ma l'effetto sulla qualità-categoria è positivo ovunque tranne l'estremo saturo.

### D.5 — Equità di qualità categoriale (macro-Cat-MRR) vs BASE, per TUTTI i 7 backbone
Macro-Cat-MRR = media delle Cat-MRR per-categoria (le minoritarie pesano uguale → demaschera l'amplificazione della maggioranza). Calcolata sulle matrici-score TEST + κ* della battery (script `scripts/yelp/macro_all_backbones.py`, `outputs_results/macro_allbk_*.csv`). **Auto-validata:** il BASE micro ricalcolato combacia col CSV della battery per **tutti i 7 backbone** (B_full riallenato seed 42 con selezione su val).

**Δmacro-Cat-MRR (SIT − BASE):**
| backbone | nyc_tist | saopaulo | ml1m |
|---|---|---|---|
| B_blind | +0.0433 | +0.0226 | +0.0138 |
| B_full | +0.0079 | +0.0031 | +0.0038 |
| EASE | +0.0305 | +0.0332 | +0.0128 |
| SASRec | +0.0076 | +0.0072 | +0.0013 |
| FPMC | +0.0125 | +0.0118 | +0.0063 |
| DeepFM | **−0.0204** | +0.0086 | +0.0048 |
| AFM | +0.0640 | +0.0524 | +0.0097 |

→ **SIT migliora la macro-Cat-MRR rispetto al backbone su 20/21 combinazioni** (tutte le famiglie: statico, CF, context-aware, sequenziale). **Unica eccezione: nyc/DeepFM (−0.020) a κ*=1.50** — quando il κ selezionato è aggressivo, su un backbone con BASE-equità già alta il nudge spinge troppo la categoria modale e peggiora l'equità (stesso fenomeno dell'amplificazione, qui indotto da κ alto, non da saturazione del dataset). Sui saopaulo/ml1m DeepFM (κ* moderato 0.25/0.10) Δmacro è positivo. → l'equità-qualità migliora quasi-ovunque vs backbone, ma il **κ aggressivo è un rischio** (collega F2/F3 future work: gate selettivo).

### D.6 — Conclusione (una frase)
> **Rispetto al backbone, X-SAGE migliora l'esposizione in modo consistente (Gini↓ 21/21, Coverage↑ 21/21, LT↑ 20/21) e l'equità-di-qualità-categoriale (macro-Cat-MRR↑ su 20/21 backbone×dataset, tutte le famiglie), in modo Pareto sui backbone forti (esposizione↑ senza costo accuracy, 10/21, tutti i 5 su ml1m) e con trade-off sui deboli/POI; NON corregge i pozzi e sul saturo estremo a livello dataset (yelp) o con κ aggressivo (nyc/DeepFM) li amplifica — è fairness di esposizione e diagnostica (la lente la misura), non correttiva.**

---

## RIEPILOGO VERDETTI (per la sezione Framework + Limiti)

**Forti (✅):** Endsley L0–L2 implementati first-class · combiner fedele/matched-off · Explainability (situazioni nominate + nudge fedele + lente) · Representation Visualization (30 JSON) · Efficiency (205 param, 0.10 ms) · Evaluation tecnica completa (anti-circolare, bootstrap, TOST, multi-seed).

**Parziali/proxy (🟡):** L3 descrittivo non-ottimizzatore · Robustness (no adversarial) · Fairness (esposizione sì/consistente, non correttiva, amplifica su yelp) · Privacy (cookieless-by-design, no DP) · Accountability/Controllability (κ ispezionabile, no audit formale) · Data-debias (anti-leakage, non algoritmico).

**Non coperti (❌ → future work/limiti):** Federated/Decentralization · Privacy formale/DP · Ethical/Social-Impact evaluation · Robustness adversarial.
