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
| 26 | MIND step-02/03: eseguibile? | backbone BPR (cornac) + mind_prep (no geohash) + smoke | ✅ **X-SAGE gira end-to-end su MIND**. ΔCat-MRR(SIT−BASE)=+0.002 (K/ε placeholder, no test). Prossimo: selezione K/ε + significatività |

## Convergenza (lo stato del pensiero, 2026-06-23)
Pattern inequivocabile e ripetuto: **le situazioni DIAGNOSTICANO ma non sono una leva operativa** —
ogni tentativo di *agire* (re-ranking, fairness item/utente, early-warning, selezione) è eguagliato
o battuto da un baseline triviale (uniforme/persistenza/casuale/naive); e come *ottimizzatore* SIT
perde vs personalizzazione (Steck) e context-aware (B_full). Il confronto con **SARE** spiega il
perché in modo strutturale (interpretabilità ⟂ performance). 

**Due strade aperte:**
1. **Paper trasparenza/audit** (lente + legge 1−R² + operazionalizzazione Endsley) — difendibile oggi.
2. **Modello nuovo** SARE-con-situazione-esplicita su un 2° dominio (MIND o Last.fm) — costruzione, scommessa.
