# WI-0b — Probe di disambiguazione

Follow-up di WI-0. Diagnostica pura su artefatti in cache, seed 42, nessun retraining.
**Nessun giudizio di merito: i numeri, e la regola di lettura pre-registrata.**

## Gate di ancoraggio

| dataset | Cat-MRR SIT | atteso | esito |
|---|---|---|---|
| ml1m | 0.38479 | 0.38479 | OK |
| nyc_tist | 0.34162 | 0.34162 | OK |
| saopaulo | 0.40045 | 0.40045 | OK |

## CHECK A2 — ristretto agli utenti multi-situazione

| dataset | utenti multi | richieste multi | ss_sit_given_user | ss_user_alone | spread oss. | spread null | p perm. | p Kruskal |
|---|---|---|---|---|---|---|---|---|
| ml1m | 44.1% (2661) | 51.0% | 0.27% | 20.2% | 0.21053 | 0.18298 | 0.0020 | 6.96e-06 |
| nyc_tist | 7.6% (311) | 9.9% | 0.33% | 32.1% | 0.17222 | 0.17185 | 0.3533 | 7.30e-01 |
| saopaulo | 32.7% (1437) | 41.1% | 0.37% | 27.6% | 0.24630 | 0.21952 | 0.0020 | 5.38e-11 |

Controllo di Simpson — situazioni con Δ(SIT−BASE) > 0 e Δ aggregato:

| dataset | situazioni positive | Δ aggregato |
|---|---|---|
| ml1m | 4/5 | +0.02517 |
| nyc_tist | 7/8 | +0.06131 |
| saopaulo | 4/4 | +0.03562 |

## CHECK C2 — entropia per orizzonte

| dataset | train | val | test | timeline intera | finestra 20 | finestra 50 | len test mediana | len timeline mediana |
|---|---|---|---|---|---|---|---|---|
| ml1m | 0.8196 | 0.0000 | 0.0000 | 0.8241 | 0.5920 | 0.6948 | 9 | 96 |
| nyc_tist | 0.7026 | 0.0000 | 0.0000 | 0.7054 | 0.6712 | 0.7018 | 3 | 28 |
| saopaulo | 0.7142 | 0.0000 | 0.0000 | 0.7186 | 0.6662 | 0.7145 | 4 | 36 |

Quota di utenti con situazione dominante >80%, per orizzonte:

| dataset | train | val | test | timeline intera | violazioni STRETTE | pareggi timestamp |
|---|---|---|---|---|---|---|
| ml1m | 4.1% | 69.9% | 68.4% | 3.9% | 0 | 120 |
| nyc_tist | 1.8% | 92.9% | 92.0% | 1.7% | 0 | 0 |
| saopaulo | 16.9% | 72.0% | 69.8% | 17.0% | 0 | 0 |

## CHECK D — curva di minimizzazione (ml1m)

Ogni accuratezza va letta **contro la sua baseline di maggioranza**.

| target | baseline | (a) etichetta situazione | (b) istogramma situazioni | (c) storico item |
|---|---|---|---|---|
| gender | 71.7% | 71.7% | 72.6% | 78.0% |
| age | 34.7% | 34.7% | 34.7% | 47.3% |
| occupation | 12.6% | 13.3% | 13.9% | 15.0% |

## Esito vs pre-registrazione

- **ml1m · A2**: ambiguo — significativo ma sotto l'1% di varianza (ss_sit_given_user=0.27%, p_perm=0.0020, p_kruskal=6.96e-06)
- **ml1m · C2**: caso migliore — artefatto dello split (entropia molto piu' alta sull'orizzonte lungo) (entropia test=0.0000 vs timeline=0.8241, divario=+0.8241)
- **nyc_tist · A2**: caso peggiore — null vero anche dove misurabile (ss_sit_given_user=0.33%, p_perm=0.3533, p_kruskal=7.30e-01)
- **nyc_tist · C2**: caso migliore — artefatto dello split (entropia molto piu' alta sull'orizzonte lungo) (entropia test=0.0000 vs timeline=0.7054, divario=+0.7054)
- **saopaulo · A2**: ambiguo — significativo ma sotto l'1% di varianza (ss_sit_given_user=0.37%, p_perm=0.0020, p_kruskal=5.38e-11)
- **saopaulo · C2**: caso migliore — artefatto dello split (entropia molto piu' alta sull'orizzonte lungo) (entropia test=0.0000 vs timeline=0.7186, divario=+0.7186)

## Domande emerse (RIPORTATE, non eseguite — come da brief)

1. **Il val ha la stessa compressione del test** (entropia mediana 0.0000 su tutti e tre gli orizzonti di validazione). Poiche' kappa* e' selezionato SU VAL, la selezione avviene su una fetta in cui la situazione e' quasi costante per utente. Non e' un errore di protocollo — val e test sono simmetrici, quindi la selezione resta anti-circolare — ma e' una domanda aperta su quanto sia informativa.
2. **ml1m ha 120 utenti con pareggio esatto di timestamp** al confine train/test (violazioni strette = 0). Non e' leakage: i timestamp MovieLens hanno risoluzione al secondo. Nota collaterale: `tests/test_causal_split.py` copre solo `D.DEFAULT_CITIES` (le 5 citta' Foursquare), quindi lo split causale di ml1m non e' mai stato nella suite.
3. **nyc_tist e' sotto-potenziato per A2 per costruzione**: solo 311 utenti multi-situazione (7,6%). Il suo esito non-significativo non e' un voto contrario; servirebbe un disegno diverso per misurarlo su quel dominio.
