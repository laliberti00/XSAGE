# X-SAGE — Processo sperimentale ESEGUITO (checklist definitiva)

Record operativo di **cosa è stato effettivamente eseguito** (non il piano). Dettagli formali in
`FREEZE_RECORD.md` (decisioni), `METRICS_FORMULAS.md` (formule), `PIPELINE_VALIDAZIONE.md` (pipeline).
Legenda: ✅ fatto · ⏳ in corso · 📝 polish minore pendente.

---

## A. Scope effettivo (dataset × backbone)
- [✅] **ml1m** — 7 backbone — casella: **WINNER**
- [✅] **nyc_tist** — 7 backbone — casella: **ridondante**
- [✅] **saopaulo** — 7 backbone — casella: **ridondante**
- [✅] **kuairand** — 2 backbone (B_blind, B_full) — casella: **nullo** (controprova feed-mediato)
- [⏳] **tokyo_tist** — 7 backbone (5 extra in generazione via `run_cars_setA.sh`) — attesa: ridondante
- [❌] NON eseguiti: amazoncd, mind, yelp, istanbul, bangkok, tsmc_nyc, tsmc_tky (fermati per scelta)

**7 backbone** = B_blind(BPR), **B_full(FM, FOCALE)**, EASE, DeepFM, AFM, FPMC, SASRec.
I 5 extra esistono solo dove generati (ml1m/nyc/sao/[tokyo]); kuairand ha solo B_blind+B_full.

---

## B. Dati e split (uguale per tutti)
- [✅] **k-core = 10** su utenti e item, uniforme (verificato: tutti min_u/min_i ≥10; Amazon 23k = k10).
- [✅] **Split temporale** per-utente 80/10/10; **leave-one-out** (1 item vero per richiesta).
- [✅] Train impara · **Validation** sceglie iperparametri · **Test** misura una volta.

---

## C. Modello delle situazioni (per seed, solo su train) — anti-circolarità
- [✅] Descrittore `v = [c̃ ‖ e]`: `c̃` = contesto informativo (alberi su train), `e` = intento propagato
      sul grafo di transizione macro (`W` su train).
- [✅] Finestra recente **strettamente causale** (`time_local < t`, stesso utente) → target mai in `v`.
- [✅] `hist[test] = train+val` (test mai visto). `contrib`, `W`, `b_z`, `Pu` stimati **solo su train**.
- [✅] Clustering rough-k-means: **K, ε selezionati su validation**; etichette `z_tr`.
- [✅] Bias situazione `b_z` = log-odds shrinkati z-scorati per situazione (λ=50).

---

## D. Combiner e metodi (3)
- [✅] Score: `ŝ = s_backbone + κ·γ·b̃[categoria]` (additivo).
- [✅] **BASE** (κ=0) · **SIT** (nudge situazionale `mem@b_z`) · **Steck-b** (nudge statico per-utente `b_z_user`).
- [✅] **κ\* selezionato su validation** (per metodo/backbone/seed) massimizzando Cat-MRR.
- [✅] Full-ranking su tutto il catalogo, **niente negative sampling**; item visti → −∞.

---

## E. Ranking e metriche (21)
- [✅] **Rango ATTESO** (McSherry-Najork) come primario; conteggi tie `g/gc` in cache → stretto ri-derivabile.
- [✅] 5 seed {42,43,44,45,46}.
- [✅] **Categoriali (faro):** Cat-MRR@20, Cat-NDCG@20, **macro-Cat-MRR@20** (metrica-faro).
- [✅] **Accuratezza item:** HR/NDCG @5/10/20 + MRR — **per-richiesta primarie** + `_u` user-balanced secondarie.
- [✅] **Esposizione:** Coverage, Gini, LT@20.
- [✅] **Costo:** JS (calibrazione utente vs top-20; peggiora by-design, dichiarato).
- [✅] **macro-Cat-MRR con min-support** `|R_c| ≥ 20` (a-priori); stabilità verificata su m∈{20,50}.

---

## F. Incertezza e significatività
- [✅] `mean` = media 5 seed. `sd_seed` = std cross-seed (piccola sui backbone cachati di ml1m ~1e-6,
      grande su B_full ~1e-3; **non zero**, solo arrotondata a 5 dec — vedi §I).
- [✅] **Bootstrap B=1500** → `se_boot` (SE, **sempre >0**), CI 95%. Ricampiona l'unità coerente
      (utenti per accuratezza-utente, richieste per categoriali/esposizione).
- [✅] Contrasti riga SIT: **L1 = SIT−BASE**, **L2 = SIT−Steck-b**.
- [✅] **Significatività primaria = bootstrap-Δ** (`p_lX`); robustezza = **t cross-seed** (`p_lX_seed`);
      consistenza = **`seeds_lX` (#seed con Δ>0)**.
- [✅] **Casella (regola UNICA)**: `Lk-pass = Δ>0 ∧ CI-bootstrap-esclude-0 ∧ seeds==5/5`.
      `nullo` / `ridondante` / `winner`. Il 5/5 è a priori.
- [✅] **Onestà**: bootstrap-p near-vacuo (≈0 ovunque) → lead su **effect size + consistenza-seed**,
      asterischi = pavimento, non l'argomento.
- [✅] **Holm 2 famiglie** (solo dataset primari): PRIMARIA `B_full×macroCatMRR×{L1,L2}×primari`;
      ROBUSTEZZA-backbone separata. Appendice/altre metriche = descrittive.
- [✅] **TOST ±0.005** su HR@20/NDCG@20 per i claim di equivalenza.

---

## G. Gate di validazione (eseguiti)
- [✅] **Round-trip** `aggregate_record`==`results_record`: `max|Δ|=0` (cache fedele al bit).
- [✅] **Stability** MSUPP∈{20,50}: caselle invarianti (nyc/sao/kuairand/ml1m).
- [✅] **Gate winner ml1m**: L2(SIT−Steck-b)@20 = +0.0053, **5/5 → PASS**.
- [✅] **3 asserzioni**: JS Steck-b<SIT (calibratore-utente); nyc Δ_L2≤0 (ridondante); segni L1/L2 invarianti stretto-vs-atteso.
- [✅] **Controprova kuairand**: Δ_L1≈+0.0001 (~0) → **nullo** = il metodo non fabbrica un winner senza segnale.

---

## H. Fix e decisioni di questa sessione (deviation log)
- [✅] `f1743d7` freeze pipeline (rango atteso, g/gc, Steck-b, JS, significatività a 3 condizioni, Holm 2-famiglie, min-support).
- [✅] `55d4b29` salta i backbone senza score-cache (i 5 extra solo sui dataset ricchi).
- [✅] `d0b3e64` broad-set con battery a solo seed 42 → riuso κ\* seed 42 (riguarda solo i lean NON tenuti).
- [✅] `9859d00` vettorizzato catrk/gc (loop su categorie) — identico al loop, cruciale per cataloghi grandi.
- [✅] Scelta di scope: fermati a ml1m + nyc + sao (+kuairand controprova, +tokyo 3ª città POI).

---

## I. Pendenti / limiti dichiarati (onesti)
- [📝] `sd_seed` arrotondato a 5 decimali → su ml1m i backbone cachati mostrano `0.00000` (valore reale ~4e-6,
      NON zero). Da alzare a 7 decimali per visualizzarlo. *(Il bootstrap `se_boot` è comunque sempre >0.)*
- [📝] `FREEZE_RECORD.md §1` elenca 7 primari + 5 appendice; lo scope **eseguito** è 3 città + kuairand + tokyo →
      aggiornare al set reale.
- [❗] Dove `sd_seed`≈0 (ml1m cachati) la condizione "5/5 seed" è quasi-vacua → lì la casella poggia sul
      bootstrap; ma il **focale B_full ha sd_seed>0 ovunque** → il winner di testa è seed-protetto.
- [❗] 7-backbone solo su ml1m/nyc/sao/[tokyo]; kuairand a 2 backbone (i 5 extra non generati). La casella
      si decide sul focale B_full, presente ovunque → comparabile.
- [❗] Non è una **legge predittiva** (no LODO/held-out): è **caratterizzazione empirica** — caselle osservate e spiegate.

---

## J. Riproducibilità
- [✅] Cache grezza per dataset (`outputs_results/cache/raw_<city>.npz`): rk, catrk, g, gc, tk50 + shared (u, tm, G1, icm, Pu).
- [✅] Da cache si ri-deriva qualsiasi metrica/@K/min-support in minuti (`aggregate_record.py`), niente re-eval.
- [✅] Codice committato (branch `exp/second-dataset-feasibility`); ambiente `IntentAwareRS_thesis/.venv` (+ `.venv-cornac` per EASE/SASRec/FPMC).
