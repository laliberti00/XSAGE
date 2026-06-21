# Overnight selection — SUMMARY

Run timestamp: 20260619_002207

## Parametri selezionati (plateau-aware, su validation)

- Perception SHARED: **γ=0.7, depth=5, n=7** (H=2, β=0.7 fissi)
- Plateau score P(perception) = 0.766 (neigh mean 0.795)

## (K*, ε*) per città + ARI vecchio vs nuovo

| città | K vecchio→nuovo | ε vecchio→nuovo | ARI dossier | ARI nuovo (S=10) |
|---|---|---|---|---|
| istanbul | 8→8 | 0.01→0.01 | 0.722 | 0.611 |
| bangkok | 4→4 | 0.03→0.01 | 0.998 | 0.753 |
| nyc_tist | 6→6 | 0.02→0.01 | 0.908 | 0.739 |
| saopaulo | 8→8 | 0.02→0.01 | 0.760 | 0.864 |
| tokyo_tist | 4→4 | 0.05→0.01 | 0.951 | 0.781 |

## Sink + touch share (nuovi parametri, regola per-request)

| città | sink vecchi | sink nuovi | touch share nuovo |
|---|---|---|---|
| istanbul | [0, 4] | [6] | 11.5% |
| bangkok | [2] | [2] | 5.1% |
| nyc_tist | [5] | [3, 4] | 23.0% |
| saopaulo | [5] | [4, 7] | 8.8% |
| tokyo_tist | [] | [] | 0.0% |

## Confronto col dossier (NYC X-SAGE, dove disponibile)

- NYC X-SAGE R@20: dossier 0.0907 → nuovo 0.0849
- NYC X-SAGE APL(per-user): dossier 0.0974 → nuovo 0.1706 (per-request 0.1943)

## File prodotti

- stage_a_ari.csv, stage_a_plateau_scores.csv
- stage_b_ari.csv
- finalist_confirm_ari.csv
- selected_params.json
- report_metrics.csv (TEST, 3×5)
- sinks_new.csv
- logs/overnight_20260619_002207.log
