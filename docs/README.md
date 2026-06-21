# X-SAGE — documentazione di validazione

X-SAGE è un re-ranker **situational + intent-aware additivo**: prende il punteggio
di un backbone di raccomandazione context-blind e gli somma un termine selettivo
che agisce solo dove serve (situazioni "sink" su utenti core), spingendo
esposizione verso macro-categorie coerenti con la situazione e verso item
long-tail. La valutazione è multi-città su **TIST2015** (Foursquare global
check-ins) per 5 città a diversa quota di transito: Istanbul, Bangkok, NYC-TIST,
Sao Paulo, Tokyo-TIST.

Formula del termine unito (X-SAGE):

```
ŝ(u, i) = s_B(u, i) + κ · m_sel(req) · γ_S(v) · Σ_k r_k · (b̃^(k)_c + λ · b^LT_c)
```

Identità strutturale: con κ=0 oppure 0 sink → ŝ = s_B esatto.

Questa documentazione è organizzata **per fase del modello** (non con codici
A1/A2). Validiamo il progetto una fase alla volta: ogni cartella contiene cosa
fa la fase, come è implementata nel clean repo, quali scelte sono state fatte e
perché, e l'esito della validazione con i numeri.

## Stato di validazione per fase

| Fase | Cosa copre | Stato | Data |
|---|---|---|---|
| [00 — Data & Backbone](00_data_backbone/) | Dati grezzi TIST2015, URM, backbone B_blind/B_full (import con checksum) | **VALIDATO** | 2026-06-18 |
| [01 — Sensing (L0)](01_sensing/) | Split cronologico per-utente, k-core=10, intent proxy causale | **VALIDATO** | 2026-06-18 |
| [02 — Perception (L1)](02_perception/) | Context state (alberi/θ), intent vector (W/attrattori/γ/β) | **VALIDATO CON RISERVA** | 2026-06-18 |
| [03 — Comprehension (L2)](03_comprehension/) | Bias per (situazione, macro) shrunk log-odds z-scored | da validare | — |
| [04 — Projection (L3)](04_projection/) | Boundary gate, proiezione membership | da validare | — |
| [05 — Recommendation](05_recommendation/) | Combiner unito, κ/λ, maschera selettiva, γ | da validare | — |
| [06 — Metrics](06_metrics/) *(trasversale)* | R@20, NDCG@20, LT@20, KL, Gini | da validare | — |
| [07 — Fairness & Sinks](07_fairness_sinks/) *(trasversale)* | Regola sink (kl_mult, lt_gap), Stage B | da validare | — |

Legenda stato: **da validare** → **in corso** → **VALIDATO**.

## Riproducibilità

Il clean repo è standalone per i backbone (import con checksum, vedi fase 00).
Riproduzione punto-stima dei numeri del paper:

```bash
PY=/Users/lucaaliberti/Downloads/IntentAwareRS_thesis/.venv/bin/python
cd xsage-clean && $PY -m scripts.run_main_results     # 5 città
$PY -m tests.test_reproduction                         # NYC + Tokyo target
```
