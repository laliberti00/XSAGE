# X-SAGE — Cronologia dei ragionamenti (timeline)

> Registro cronologico delle domande poste, dei test fatti e dei verdetti. Serve a ricostruire
> *come* siamo arrivati allo stato attuale. Da aggiornare a ogni passo. Ultimo: 2026-06-23.
> Stato sintetico vivo: [STATE_OF_WORK.md](STATE_OF_WORK.md).

## Pre-storia (repo OLD `IntentAwareRS_thesis`, mag–giu 2026)
Baseline tesi → detour Gowalla (abbandonato) → Foursquare TSMC2014 NYC/TKY → FM situation-aware
round-1 (bilinear gate, **RED "forced but useless"**, archiviato) → **X-SAGE round-2** (combiner
additivo, lente fairness GREEN) → round-3 credibility hardening (lente robusta 8 backbone, eq.18,
explainability) → round-4 multi-città TIST2015 (legge riformulata su ricchezza-attrattori). Combiner
**harmonic → additivo**. Dossier 5 città.

## Campagna repo clean `xsage-clean` (giu 2026) — il filo dei ragionamenti

| # | domanda / ipotesi | test | verdetto |
|---|---|---|---|
| 1 | parametri situazioni giustificati? | selezione K/ε/depth/n (criteri interni, anti-circolare) | ✅ con caveat (bkk K=tetto, ε fuori-banda; ist K=fallback-ARI) |
| 2 | esiste un segnale categoriale? | Cat-MRR SIT vs BASE/UNI_mean, bootstrap, plateau-K | ✅ SIT>BASE/>UNI_mean 5/5; plateau 25/25 — ma effetto piccolo |
| 3 | γ_S=0.5 hardcoded giustificato? | γ_S=1/\|T\| | ✅ applicato (\|T\|=2 nell'87-94%, effetto <0.0002) |
| 4 | L3 projection aiuta il re-ranking? | boundary_structural | ❌ inerte/leggermente negativo |
| 5 | L3 ha valore descrittivo/predittivo? | situation_evolution (struttura/predittività/interpret.) | ⚠️ descrittiva sì, predittiva ≈ persistenza |
| 6 | l'intento è costitutivo della situazione? | ablazione FULL=[c‖e] vs CTX=[c] | ⚠️ costitutivo **4/5**, **dannoso São Paulo** |
| 7 | São Paulo: artefatto o proprietà? | ri-selezione K/ε indipendente + R²(e\|c̃) | ✅ **PROPRIETÀ** — legge **beneficio∝(1−R²)** (sao R²=0.91) |
| 8 | siamo pubblicabili? cosa manca? | audit gap | gap: backbone singolo, no baseline esterne, no robustezza |
| 9 | Cornac ci dà calibrazione? | ispezione | ❌ niente calibrazione; ✅ libreria di backbone |
| 10 | il pipeline è backbone-agnostico? (B1) | import BPR, SIT−BASE | ✅ +0.031 su BPR (pluggabilità OK) |
| 11 | SIT batte le baseline reali? | Steck-b/Steck-a (Holm) | ❌ **SIT perde** vs Steck-b 4/5, vs Steck-a 5/5 |
| 12 | esplicito batte contesto-grezzo? (B3) | SIT vs B_full (stesse feature, Holm) | ❌ **SIT perde 5/5** (B_full 2–3×) |
| 13 | la trustworthiness è misurata o illustrativa? | ricostruzione repo + lettura CSV reali | ✅ **la LENTE è misurata e positiva** (KL ~75× tra situazioni, robusta 8 backbone) |
| 14 | come siamo messi sulla fairness? | lente vs intervento; SIT vs UNI_glob | lente ✅ (diagnosi); intervento ❌ (Pareto-dominato dall'uniforme) |
| 15 | la projection disambigua nel nuovo setting? | eq.18 su 5 città TIST, vs geom e vs persistenza | ✅ vs geom 5/5; ❌ vs **persistenza** 1/5 → **L3 chiuso** |
| 16 | una eq.18 persistence-aware recupera? | blend π_train·persist + (1−π)·T | ❌ NON PASS (3/5, soglia ≥4/5) |
| 17 | B4 equità lato-utente regge sulle 5 città? | ΔGini utenti, situato vs uniforme | ⚠️ funziona 5/5 **ma dominato dall'uniforme 3/5** |
| 18 | la projection è early-warning di pozzo? | PR-AUC/recall-MOVE vs persistenza | ❌ NON PASS (non batte la persistenza) |
| 19 | selezionare le richieste a rischio aiuta? | risk-coverage, situato vs casuale vs naive | ❌ NON PASS (dominato da casuale+naive 0/5) |
| 20 | le soglie del comprehension sono giustificate? | audit K_final_rule/epsilon_final | ⚠️ 3/5 pulite; bkk e ist sono ripieghi (dichiarati) |
| 21 | **perché SARE vince e noi no?** | lettura paper+codice SARE | **baratto strutturale**: SARE appreso/personalizzato/latente; X-SAGE non-sup/fisso/grezzo. Interpretabilità ⟂ performance |
| 22 | 2° dataset fattibile? | ricognizione MIND-large + Last.fm-1K | Last.fm=port pulito (nodo: genere esterno); MIND=categoria nativa + protocollo SARE (utente non-persistente) |
| 23 | Last.fm: scaricare il genere? | fetch MusicBrainz (top-N per play, rate-limit, checkpoint) | ✅ fattibile (top-5000=80% ascolti); fermato a 85 artisti (ripartibile) |
| 24 | MIND: utenti abbastanza profondi? | diagnostico click/utente | ❌ mediana 3 click/utente → rompe split per-utente; **scelta: k-core=10** (105K utenti profondi) |
| 25 | MIND porting step-01 | preprocess_mind.py → schema X-SAGE | ✅ **MIND è una "città"**: 101K utenti, 5278 item, 15 macro, split 80/10/10, no geohash |
| 26 | MIND step-02/03: eseguibile? | backbone BPR (cornac) + mind_prep (no geohash) + smoke | ✅ **X-SAGE gira end-to-end su MIND**. ΔCat-MRR(SIT−BASE)=+0.002 (K/ε placeholder, no test) |
| 27 | MIND eval (accuracy+fairness, placeholder) | mind_eval (bootstrap + lente) | SIT−BASE +0.002 (sig, minuscolo); fairness trascurabile; **lente rivela disparità** (KL 0.03–2.03×, sit3 sink) |
| 28 | ml-1m: stesso porting? | preprocess_ml1m + backbone + eval (city-param) | ✅ **port pulito** (6K utenti PROFONDI ~165 rating, 18 generi). SIT−BASE **+0.020** (sig) + R@20 +0.005 + fairness↑; lente sit3 sink (16K). **Il segnale scala con la profondità utente** |
| 29 | ml-1m: SIT regge vs baseline reali? | eval_baselines (Steck-b/a, K/ε selezionati, Holm) | ⚡ **SIT BATTE Steck-b (+0.024)** + UNI/BASE; perde solo vs Steck-a (metric-gaming, ma SIT>R@20). **Su dominio profondo la situazione batte la personalizzazione statica** (ribalta Foursquare) |
| 30b | ml-1m: κ impeccabile (su val, per-metodo) | eval_kappa.py | ⚡⚡ **SIT batte Steck-b a 8/8 κ** e al κ*=0.5 (val): +0.019 signif. κ ereditato 0.25 era subottimale. **Vittoria ROBUSTA, non artefatto di κ** |
| 30 | B_full su MIND/ml1m | train_bfull.py (scaffold, no geo/fine, torch+MPS) | scaffold pronto, scoring DA VERIFICARE — gate finale per l'angolo "competitivo" |
| 31 | GATE: SIT-su-BPR batte B_full? | gate_bfull.py (B_full tunato su val, TOST) | ❌ **FAIL**: B_full batte SIT-su-BPR di +0.053 → angolo "sostituto" cade |
| 32 | chiusura parametri ml-1m | close_params.py (val Cat-MRR + plateau; β/H/α sensibilità) | ✅ γ/n/H **cambiati** (ereditati subottimali); β/α piatti. ⚠️ bordo-griglia |
| 33 | **batteria due-assi su B_full** | battery_bfull.py (5 seed, params chiusi, bootstrap+Holm+TOST) | ⚡ **SIT-MONTATO-su-B_full MIGLIORA B_full**: Cat-MRR +0.008, R@20 +0.002 (TOST non degrada), LT +0.005, Gini −0.004; JS-user peggiora (by-design). **Enhancer interpretabile a valore aggiunto** |
| 34 | JS-calibration | lettura tabellone | SIT peggiora JS-**user** (de-calibra dalla media-utente → vince sul task). Manca JS-**situazione** (O1) |
| 35 | 4° dataset: Yelp fattibile? | sonda business+review (5GB stream) | ✅ port più ricco: 100% geo, k-core20 Philadelphia = **4842 utenti, 52 review/ut** (profondo), 17 macro |
| 36 | Yelp porting + GEO riabilitato | preprocess_yelp.py + `city_attrs` (prev_geohash5) | ✅ pipeline gira con geo; backbone BPR; Fase A chiusa (γ0.4/d2/n5/β0.7/H2/α10, quasi tutti interni) |
| 37 | Yelp: SIT regge? (batteria 5 seed) | battery_bfull.py yelp | ❌ **NULL**: SIT-su-B_full −0.0011 (p=0, minuscolo); fairness Δ≈0. Profondità c'era → perché? |
| 38 | **macro-averaged: perché Yelp è null?** | macro_avg.py (micro vs MACRO per-categoria) | 🔑🔑 Yelp **saturo 83.7%** → SIT è **amplificatore di maggioranza**: macro-Δ **−0.010, 1/17 cat**. Il +micro era artefatto. ml-1m macro-Δ **+0.014, 16/18** (vero). MIND neutro (shallow) |
| 39 | macro-averaged su MIND+saturazione 4sq | macro_avg.py mind + df_test TIST | ✅ **LEGGE A DUE GATE**: profondità ∧ non-saturazione. MIND fallisce profondità, Yelp saturazione, ml-1m nessuno. tokyo(62.5%) come Yelp. macro-averaged = metrica diagnostica |

## Convergenza (lo stato del pensiero, 2026-06-23) — AGGIORNATA
Il quadro è cambiato col 2°/3° dataset. Su Foursquare valeva: *le situazioni diagnosticano ma non
ottimizzano* (SIT perde vs Steck/B_full). **Ma il pattern è CONDIZIONATO alla profondità del dominio**:
- **MIND** (shallow ~5): SIT **dannoso** (perde anche vs BASE, −0.020).
- **Foursquare** (medio): SIT > BASE ma < Steck-b/B_full.
- **ml-1m** (profondo ~165): SIT **batte Steck-b** (+0.019, robusto a 8/8 κ, κ selezionato su val).
→ **Legge di caratterizzazione**: il valore della situazione esplicita scala monotòno con la
profondità comportamentale, da dannoso a vincente-vs-personalizzazione.

**Esito del gate B_full (2026-06-24), che chiude l'angolo nel modo giusto:**
- SIT-su-backbone-debole (BPR) **< B_full** (gate FAIL, −0.053) → X-SAGE NON è un *sostituto* del context-aware.
- **SIT-MONTATO su B_full > B_full** (batteria, params chiusi, 5 seed): Cat-MRR +0.008, R@20 +0.002 (TOST
  non degrada), fairness-esposizione ↑ (LT +0.005, Gini −0.004) → X-SAGE È un **enhancer interpretabile
  a valore aggiunto**, al costo dichiarato della calibrazione-utente (JS-user ↑, by-design).

**Il contributo, consolidato (2026-06-24):** *X-SAGE = enhancer situazionale interpretabile che, sui
domini behavioral-profondi, aggiunge valore (accuratezza + fairness-esposizione) anche sopra un backbone
context-aware forte; con una **legge** che caratterizza quando aiuta e una **lente**
di audit per-situazione che i latenti non producono. Non un recommender-che-vince-da-solo.*

**LEGGE A DUE GATE (raffinata dal 4° dataset, 2026-06-24):** SIT aggiunge valore **sse passa due gate
indipendenti** — *(1) profondità* comportamentale (storia utente stimabile) **e** *(2) non-saturazione*
del target-categoria (nessuna macro domina). Evidenza sui 4 dataset, ognuno fallisce un gate diverso:
- **MIND** (shallow ~5, NON saturo 24.9%): fallisce **profondità** → SIT **neutro** (κ*→0.05, si auto-spegne; Steck-b vince).
- **Yelp** (profondo ~52, SATURO 83.7%): fallisce **saturazione** → SIT **amplificatore di maggioranza** (macro-Δ −0.010, 1/17 cat).
- **ml-1m** (profondo ~165, NON saturo 27.7%): passa **entrambi** → **win vero e distribuito** (macro-Δ +0.014, 16/18 cat).
- **Foursquare**: medio; tokyo (62.5%) saturo come Yelp, le altre no — SIT-macro pieno = O3.
La **macro-averaged Cat-MRR** è la metrica che smaschera l'amplificatore (la micro mente sui dati saturi):
da ora SIT si giudica sulla macro-averaged + wins-per-categoria, non sulla micro.

**CONSULTO asse-portante (2026-06-24).** Tensione "asse=fairness (intuizione)" vs "asse=numeri-che-migliorano
(accuracy)". Esito: la **fairness-con-guadagno esiste già** — la **macro-averaged È equità per-categoria**
(qualità equa fra categorie), gain-backed su ml-1m (+0.014, 16/18). L'intervento anti-modale di esposizione NON
è la via (uniforme domina per costruzione su metriche globali; premessa "pozzo=iniquità" non-validata) → future
work. **Asse deciso: stato situazionale NEUTRO (L2, una volta) → usi multipli (diagnosi=lente, azione=re-ranking)
+ legge a due gate.** Il decoupling è una virtù da dichiarare. Precisione: lo stato è neutro nel clustering ma le
feature `[c̃‖e]` sono category-aware a monte → **ablazione di neutralità** (raw vs full) per stabilire se il
valore è strutturale o cucito nelle feature.

## Allineamento finale (2026-06-25)
- **Scoperto disallineamento k-core** (ml-1m/mind=10; yelp/kuairand/amazon=20) → confronto non valido.
- **Riallineato a k-core=10 uniforme** (`run_kcore10_all.sh`): preprocess→backbone→close_params(FRESH)→
  battery5→macro_avg→neutrality→explainability→costi. yelp **no-geo** (come ml-1m).
- **Tabellone validato (5 seed, bootstrap+Holm+TOST)**: ml-1m UNICO winner (SIT-su-B_full **+0.0082**,
  8× la SD; batte anche Steck-b); mind/yelp null significativi-negativi; kuairand null trascurabile/instabile;
  amazon ridondante. **I verdetti non cambiano dal k-core misto** → robusti.
- **Amazon-5seed**: morta al freeze del Mac (catalogo 20.7K → RAM) → params calibrati ma battery a 1-seed,
  DA RI-LANCIARE.
- **Foursquare TSMC2014** (nyc/tokyo): diagnostico → ridondanti (POI abitudinari, Steck-b ≫ SIT).
- **Riproducibilità ml-1m**: deterministico (seed=42) ovunque tranne B_full su MPS (±0.001 = la SD).
  Winner stabile a ogni ri-run; numeri congelati nei CSV committati.
- **Tassonomia a 2 livelli**: gate1-3 → SIT>BASE; non-ridondanza(1−R²) → SIT>Steck-b. Solo ml-1m passa entrambi.
