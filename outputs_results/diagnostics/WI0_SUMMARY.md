# WI-0 — Viability probe: risultati

Numeri prodotti da `scripts/diagnostics/viability_probe.py` su artefatti in cache (seed 42, percorso battery, nessun riaddestramento). **Nessun giudizio di merito.**

## Ancoraggio (gate di correttezza)

| dataset | K | eps | kappa* | Cat-MRR micro SIT | riferimento macro_avg | esito |
|---|---|---|---|---|---|---|
| ml1m | 5 | 0.03 | 0.5 | 0.38479 | 0.38479 | OK |
| nyc_tist | 8 | 0.02 | 0.5 | 0.34162 | 0.34162 | OK |
| saopaulo | 4 | 0.03 | 0.25 | 0.40045 | 0.40045 | OK |

## A — quota di varianza (SS sequenziali, entrambi gli ordini)

| dataset | sit per prima | sit dato utente | utente per primo | utente data sit | ICC utente | ICC sit | n richieste | n utenti |
|---|---|---|---|---|---|---|---|---|
| ml1m | 3.0% | 0.1% | 30.8% | 27.9% | 0.2622 | 0.0395 | 97199 | 6040 |
| nyc_tist | 1.9% | 0.0% | 40.0% | 38.2% | 0.2184 | 0.0214 | 17702 | 4113 |
| saopaulo | 3.9% | 0.1% | 36.5% | 32.7% | 0.2212 | 0.0559 | 23821 | 4395 |

| dataset | utenti con >=2 situazioni | spread mediano intra-utente |
|---|---|---|
| ml1m | 44.1% | 0.21053 |
| nyc_tist | 7.6% | 0.17222 |
| saopaulo | 32.7% | 0.24630 |

## B — separabilita' dalla preferenza statica

| dataset | R2 mediano sit~utente | R2 aggregato | coseno mediano | flip argmax | flip top-1 | R2 nudge~contesto |
|---|---|---|---|---|---|---|
| ml1m | 0.0565 | 0.0039 | +0.0515 | 91.9% | 32.5% | 0.0318 |
| nyc_tist | 0.0999 | 0.0416 | +0.2123 | 73.5% | 51.8% | 0.0094 |
| saopaulo | 0.0989 | 0.0348 | +0.2065 | 77.1% | 29.1% | 0.0753 |

## C — persistenza dell'etichetta

| dataset | entropia mediana | dominante mediana | % utenti dominante >80% | P(z_t+1=z_t) test | n utenti >=5 req |
|---|---|---|---|---|---|
| ml1m | 0.0000 | 1.0000 | 68.4% | 0.8451 | 4293 |

Baseline di maggioranza (ml1m): gender 71.7% · age 34.7% · occupation 12.6%

## Le tre risposte

**ml1m** — A) quota di varianza da situazione = 3.0% da sola, 0.1% al netto dell'utente (utente = 30.8%). B) R2 mediano situazione~utente = 0.057; flip-rate argmax = 91.9%, top-1 = 32.5%. C) entropia mediana = 0.000; utenti con dominante >80% = 68.4%; baseline gender = 71.7%.

**nyc_tist** — A) quota di varianza da situazione = 1.9% da sola, 0.0% al netto dell'utente (utente = 40.0%). B) R2 mediano situazione~utente = 0.100; flip-rate argmax = 73.5%, top-1 = 51.8%. C) non calcolato (ml1m-only).

**saopaulo** — A) quota di varianza da situazione = 3.9% da sola, 0.1% al netto dell'utente (utente = 36.5%). B) R2 mediano situazione~utente = 0.099; flip-rate argmax = 77.1%, top-1 = 29.1%. C) non calcolato (ml1m-only).

