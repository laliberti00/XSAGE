# WI-0c — Il guadagno si concentra dove la situazione cambia?

Pre-registrato nel brief WI-0c. Bucket, contrasti e regola di decisione fissati prima dei numeri. 5 seed {42–46}, kappa* congelato dalla battery, bootstrap a due livelli (ricampiona UTENTI e SEED, mai richieste). Nessun retraining, nessun ri-split.

> **Backbone.** Usato `B_blind`: e' cio' che produce i valori del gate ed e' cio' su cui girano WI-0/WI-0b. Nel repo B_blind sta in `FM.scores.npy` ma contiene punteggi **BPR** (`cornac_backbone.py:1`); l'FM context-aware e' `B_full`, che `results_record.py:200` **riallena per seed** — incompatibile col vincolo \"zero retraining\" e col gate. Ambiguita' del brief dichiarata, non risolta in silenzio.

> **Priorita' dichiarata prima dei numeri:** decisivo = **ml1m**; conferma = **saopaulo**; **nyc_tist e' strutturalmente sotto-potenziato** (test mediano 3, 92% mono-situazione) — riportato, non vota.

## Gate di ancoraggio

| dataset | Cat-MRR SIT (seed 42) | atteso | esito |
|---|---|---|---|
| ml1m | 0.38479 | 0.38479 | OK |
| nyc_tist | 0.34162 | 0.34162 | OK |
| saopaulo | 0.40045 | 0.40045 | OK |

## E1 — Delta per distanza dal cambio di situazione

SIT e BASE riportati separatamente: e' la guardia di falsificazione.

**ml1m**

| bucket | n richieste | n utenti | SIT | BASE | Δ | CI 95% Δ |
|---|---|---|---|---|---|---|
| d=0 | 14314 | 2661 | 0.33533 | 0.34704 | -0.01171 | [-0.01594, -0.00721] |
| d1-2 | 11067 | 2230 | 0.33332 | 0.34221 | -0.00889 | [-0.01428, -0.00325] |
| d3-5 | 6581 | 1523 | 0.33812 | 0.33962 | -0.00149 | [-0.00953, +0.00587] |
| d>5 | 9496 | 868 | 0.35595 | 0.34321 | +0.01274 | [-0.00086, +0.02627] |
| stable | 47665 | 3379 | 0.43589 | 0.38167 | +0.05422 | [+0.04604, +0.06294] |
| pre_first_change *(fuori contrasto)* | 8073 | 2661 | 0.31326 | 0.31118 | +0.00208 | [-0.00616, +0.01074] |

Contrasti pre-registrati (vicino − lontano):

| contrasto | Δ(vicino)−Δ(lontano) | CI 95% | CI esclude 0 | ΔSIT | ΔBASE |
|---|---|---|---|---|---|
| d=0 - d>5 | -0.02445 | [-0.03757, -0.01092] | sì | -0.02062 | +0.00383 |
| d=0 - stable | -0.06593 | [-0.07546, -0.05659] | sì | -0.10056 | -0.03463 |
| d1-2 - d>5 | -0.02163 | [-0.03477, -0.00796] | sì | -0.02264 | -0.00101 |
| d1-2 - stable | -0.06311 | [-0.07329, -0.05247] | sì | -0.10258 | -0.03946 |

Versione within-user centrata (effetto-utente rimosso per costruzione):

| bucket | celle utente×seed | Δ centrato | CI 95% |
|---|---|---|---|
| d=0 | 13305 | +0.00856 | [+0.00618, +0.01105] |
| d1-2 | 11150 | -0.00119 | [-0.00361, +0.00114] |
| d3-5 | 7615 | -0.00365 | [-0.00647, -0.00080] |
| d>5 | 4340 | -0.00993 | [-0.01318, -0.00670] |
| stable | 16895 | +0.00000 | [+0.00000, +0.00000] |
| pre_first_change | 13305 | -0.02331 | [-0.02626, -0.02047] |

**nyc_tist**

| bucket | n richieste | n utenti | SIT | BASE | Δ | CI 95% Δ |
|---|---|---|---|---|---|---|
| d=0 | 906 | 1025 | 0.35878 | 0.30428 | +0.05450 | [+0.01889, +0.09086] |
| d1-2 | 503 | 578 | 0.35050 | 0.31533 | +0.03518 | [+0.00220, +0.06722] |
| d3-5 | 175 | 190 | 0.32158 | 0.28544 | +0.03614 | [-0.01460, +0.08580] |
| d>5 | 96 | 59 | 0.26269 | 0.27085 | -0.00816 | [-0.06636, +0.08781] |
| stable | 15045 | 4036 | 0.33338 | 0.27888 | +0.05450 | [+0.04339, +0.06568] |
| pre_first_change *(fuori contrasto)* | 976 | 1025 | 0.31148 | 0.30129 | +0.01018 | [-0.02551, +0.04992] |

Contrasti pre-registrati (vicino − lontano):

| contrasto | Δ(vicino)−Δ(lontano) | CI 95% | CI esclude 0 | ΔSIT | ΔBASE |
|---|---|---|---|---|---|
| d=0 - d>5 | +0.06267 | [-0.03907, +0.12936] | no | +0.09610 | +0.03343 |
| d=0 - stable | +0.00001 | [-0.03918, +0.04138] | no | +0.02541 | +0.02540 |
| d1-2 - d>5 | +0.04334 | [-0.05626, +0.10336] | no | +0.08782 | +0.04448 |
| d1-2 - stable | -0.01932 | [-0.05806, +0.01628] | no | +0.01712 | +0.03645 |

Versione within-user centrata (effetto-utente rimosso per costruzione):

| bucket | celle utente×seed | Δ centrato | CI 95% |
|---|---|---|---|
| d=0 | 2364 | +0.02712 | [+0.01868, +0.03449] |
| d1-2 | 1295 | -0.00528 | [-0.01591, +0.00554] |
| d3-5 | 417 | -0.01692 | [-0.03723, +0.00396] |
| d>5 | 125 | -0.01774 | [-0.04675, +0.01592] |
| stable | 18201 | +0.00000 | [+0.00000, +0.00000] |
| pre_first_change | 2364 | -0.02191 | [-0.03115, -0.01246] |

**saopaulo**

| bucket | n richieste | n utenti | SIT | BASE | Δ | CI 95% Δ |
|---|---|---|---|---|---|---|
| d=0 | 2271 | 2246 | 0.40131 | 0.37817 | +0.02314 | [+0.01020, +0.03948] |
| d1-2 | 1428 | 1488 | 0.41467 | 0.39401 | +0.02066 | [+0.00584, +0.04337] |
| d3-5 | 537 | 592 | 0.42503 | 0.39373 | +0.03130 | [+0.01181, +0.05171] |
| d>5 | 306 | 193 | 0.49168 | 0.44566 | +0.04602 | [+0.00426, +0.09344] |
| stable | 17105 | 4169 | 0.39314 | 0.36068 | +0.03246 | [+0.02297, +0.04296] |
| pre_first_change *(fuori contrasto)* | 2172 | 2246 | 0.39264 | 0.37376 | +0.01888 | [+0.00369, +0.03438] |

Contrasti pre-registrati (vicino − lontano):

| contrasto | Δ(vicino)−Δ(lontano) | CI 95% | CI esclude 0 | ΔSIT | ΔBASE |
|---|---|---|---|---|---|
| d=0 - d>5 | -0.02288 | [-0.06621, +0.01583] | no | -0.09037 | -0.06748 |
| d=0 - stable | -0.00932 | [-0.02778, +0.01061] | no | +0.00818 | +0.01750 |
| d1-2 - d>5 | -0.02536 | [-0.06939, +0.01350] | no | -0.07701 | -0.05165 |
| d1-2 - stable | -0.01180 | [-0.03158, +0.01322] | no | +0.02153 | +0.03333 |

Versione within-user centrata (effetto-utente rimosso per costruzione):

| bucket | celle utente×seed | Δ centrato | CI 95% |
|---|---|---|---|
| d=0 | 4694 | +0.00440 | [+0.00168, +0.00732] |
| d1-2 | 2972 | -0.00474 | [-0.00946, -0.00043] |
| d3-5 | 1084 | -0.00340 | [-0.01098, +0.00310] |
| d>5 | 347 | -0.00626 | [-0.01843, +0.00557] |
| stable | 17281 | +0.00000 | [+0.00000, +0.00000] |
| pre_first_change | 4694 | -0.00520 | [-0.00834, -0.00208] |

## E2 — stratificazione a lunghezza di test controllata

Vedi `check_e2_stratified_<ds>.csv` (celle n_situazioni × quartile di lunghezza).

## E3 — variare vs avere piu' dati

| dataset | β₁ entropia | β₂ log(len) | β₃ n_transizioni | R² | n oss. | n cluster |
|---|---|---|---|---|---|---|
| ml1m | -0.15198 (0.01304)* | +0.01057 (0.00291)* | -0.00081 (0.00042) | 0.03215 | 30200 | 6040 |
| nyc_tist | +0.01371 (0.02927) | +0.04986 (0.00641)* | -0.00738 (0.00279)* | 0.01414 | 20565 | 4113 |
| saopaulo | -0.05829 (0.01044)* | +0.00968 (0.00405)* | +0.00240 (0.00145) | 0.00480 | 21975 | 4395 |

Correlazione fra regressori (se quasi collineari, i coefficienti separati non sono interpretabili):

- **ml1m**: entropia~loglen +0.086 · entropia~ntrans +0.543 · loglen~ntrans +0.456
- **nyc_tist**: entropia~loglen +0.106 · entropia~ntrans +0.722 · loglen~ntrans +0.254
- **saopaulo**: entropia~loglen +0.162 · entropia~ntrans +0.707 · loglen~ntrans +0.379

## Esito (regola pre-registrata, applicata meccanicamente)

- **ml1m** (decisivo) → **NON CONFERMATO** — contrasti disgiunti nel verso dell'ipotesi: no (4/4 disgiunti nel verso OPPOSTO) · guardia di falsificazione: VIOLATA · β₁/β₃ significativo nel verso dell'ipotesi: no (significativo ma ROVESCIATO)
- **nyc_tist** (NON VOTA (sotto-potenziato)) → **NON CONFERMATO** — contrasti disgiunti nel verso dell'ipotesi: no (0/4 disgiunti nel verso OPPOSTO) · guardia di falsificazione: superata · β₁/β₃ significativo nel verso dell'ipotesi: no (significativo ma ROVESCIATO)
- **saopaulo** (conferma) → **NON CONFERMATO** — contrasti disgiunti nel verso dell'ipotesi: no (0/4 disgiunti nel verso OPPOSTO) · guardia di falsificazione: VIOLATA · β₁/β₃ significativo nel verso dell'ipotesi: no (significativo ma ROVESCIATO)

### esito = NON CONFERMATO

Determinato su **ml1m** (dataset decisivo), con **saopaulo** come conferma. **nyc_tist** non concorre alla decisione.

## Domande emerse (RIPORTATE, non eseguite — come da brief §8)

1. **Rovesciamento fra-utenti vs entro-utente (tipo Simpson).** Il contrasto FRA bucket e' rovesciato su ml1m, ma la versione ENTRO utente e' positiva a `d=0` su **3/3 dataset** (+0.0086 ml1m, +0.0271 nyc_tist, +0.0044 saopaulo) e decresce in modo monotono con la distanza. Le due analisi NON sono sulla stessa popolazione: il bucket `stable` vale 0.00000 **per costruzione** nella versione entro-utente (chi non cambia occupa un solo bucket, quindi la sua deviazione dalla propria media e' identicamente nulla) ed e' quindi escluso dal confronto appaiato.
2. **I bucket differiscono per difficolta' intrinseca, non solo per trattamento.** BASE da solo varia fra bucket: 0.311-0.382 su ml1m, 0.271-0.446 su saopaulo. Il confronto fra bucket confonde composizione della popolazione ed effetto.
3. **beta1 e beta3 non sono separabili fra loro** (corr entropia~n_transizioni 0.54-0.72): misurano quasi la stessa cosa. Sono invece separabili da beta2 (corr entropia~loglen 0.09-0.16), quindi la distinzione *variare* vs *avere piu' dati* — quella che il brief chiedeva — resta interpretabile.
4. **E2 e' la misura meno confusa** (lunghezza del test neutralizzata) e su ml1m e' monotona e negativa in tutti e quattro i quartili: 1 situazione +0.033/+0.058, 2 situazioni -0.065/+0.001, 3+ situazioni ~-0.014/-0.000.

Nessuna di queste e' stata investigata: come da brief, si riportano e ci si ferma.
