# Overnight selection — SUMMARY

Run timestamp: 20260619_161521

## Parametri selezionati (plateau-aware, su validation)

- Perception SHARED: **γ=0.4, depth=6, n=3** (H=2, β=0.7 fissi)
- Plateau score P(perception) = 0.762 (neigh mean 0.813)

## (K*, ε*) per città + boundary fraction + ARI vecchio vs nuovo

Banda boundary primaria: [10%, 30%]

| città | K vecchio→nuovo | ε vecchio→nuovo | boundary frac (in banda?) | ARI dossier | ARI nuovo (S=10) |
|---|---|---|---|---|---|
| istanbul | 8→4 | 0.01→0.01 | 19.9% ✅ | 0.722 | 0.670 |
| bangkok | 4→4 | 0.03→0.03 | 15.7% ✅ | 0.998 | 0.876 |
| nyc_tist | 6→4 | 0.02→0.02 | 16.3% ✅ | 0.908 | 0.796 |
| saopaulo | 8→4 | 0.02→0.03 | 15.2% ✅ | 0.760 | 0.856 |
| tokyo_tist | 4→4 | 0.05→0.07 | 20.5% ✅ | 0.951 | 0.814 |

## Robustezza alla banda (band → (K*,ε*) per città)

Se i (K*,ε*) NON cambiano tra le bande, la scelta è robusta (come il plateau di λ).

| città | 5%-35% | 10%-30% | 15%-25% |
|---|---|---|---|
| istanbul | K=4,ε=0.01 | K=4,ε=0.01 | K=4,ε=0.01 |
| bangkok | K=4,ε=0.03 | K=4,ε=0.03 | K=4,ε=0.03 |
| nyc_tist | K=4,ε=0.01 | K=4,ε=0.02 | K=4,ε=0.03 |
| saopaulo | K=4,ε=0.01 | K=4,ε=0.03 | K=4,ε=0.03 |
| tokyo_tist | K=4,ε=0.07 | K=4,ε=0.07 | K=4,ε=0.07 |

(* = fallback fuori banda)

## ⚠️ Controlli automatici

Da verificare (ottimo al bordo / fuori banda / depth alta):
- depth=6 (bordo max)
- depth=6 ≥6 (tensione interpretabilità — DISCUTERE)
- istanbul: K=4 (bordo griglia)
- bangkok: K=4 (bordo griglia)
- nyc_tist: K=4 (bordo griglia)
- saopaulo: K=4 (bordo griglia)
- tokyo_tist: K=4 (bordo griglia)

## Sink + touch share (nuovi parametri, regola per-request)

| città | sink vecchi | sink nuovi | touch share nuovo |
|---|---|---|---|
| istanbul | [0, 4] | [0] | 32.8% |
| bangkok | [2] | [2] | 4.6% |
| nyc_tist | [5] | [1] | 15.8% |
| saopaulo | [5] | [3] | 24.8% |
| tokyo_tist | [] | [] | 0.0% |

## Confronto col dossier (NYC X-SAGE, dove disponibile)

- NYC X-SAGE R@20: dossier 0.0907 → nuovo 0.0891
- NYC X-SAGE APL(per-user): dossier 0.0974 → nuovo 0.1462 (per-request 0.1630)

## File prodotti

- stage_a_ari.csv, stage_a_plateau_scores.csv
- stage_b_ari.csv (con colonna bfrac)
- band_robustness.csv
- finalist_confirm_ari.csv
- selected_params.json
- report_metrics.csv (TEST, 3×5)
- sinks_new.csv
- logs/overnight_20260619_161521.log
