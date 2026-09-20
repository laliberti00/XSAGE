# A.4 — equità per situazione: sintesi

Divario rawlsiano = (minimo sulle situazioni − media sulle situazioni) della macro-Cat-MRR.
Più negativo = la situazione peggio servita è più lontana dalla media.
Calcolato **per seme e poi mediato**: le etichette di situazione non sono confrontabili fra semi.
Celle `low_support` (< 8 utenti o < 20 richieste) escluse da minimi e massimi, mai cancellate.

## Griglia base — 3 dataset × 4 backbone (risultato principale)

| dataset   | backbone   |     BASE |      SIT |   Steck-b |   SIT_meno_BASE |
|:----------|:-----------|---------:|---------:|----------:|----------------:|
| ml1m      | AFM        | -0.0961  | -0.0952  |  -0.08884 |         0.0009  |
| ml1m      | B_blind    | -0.00918 | -0.01421 |  -0.00865 |        -0.00502 |
| ml1m      | EASE       | -0.01181 | -0.01702 |  -0.01231 |        -0.0052  |
| ml1m      | SASRec     | -0.05993 | -0.05718 |  -0.0575  |         0.00275 |
| nyc_tist  | AFM        | -0.02592 | -0.05254 |  -0.06534 |        -0.02662 |
| nyc_tist  | B_blind    | -0.03516 | -0.0454  |  -0.03317 |        -0.01023 |
| nyc_tist  | EASE       | -0.03874 | -0.05061 |  -0.03895 |        -0.01188 |
| nyc_tist  | SASRec     | -0.04105 | -0.06548 |  -0.05458 |        -0.02443 |
| saopaulo  | AFM        | -0.03258 | -0.06967 |  -0.08673 |        -0.03709 |
| saopaulo  | B_blind    | -0.02076 | -0.03034 |  -0.03446 |        -0.00958 |
| saopaulo  | EASE       | -0.01881 | -0.04663 |  -0.04093 |        -0.02782 |
| saopaulo  | SASRec     | -0.04742 | -0.046   |  -0.03966 |         0.00143 |

**SIT peggiora il divario in 9 celle su 12.**

## Griglia piena — 5 dataset × 7 backbone (appendice)

SIT − BASE del divario, per cella:

| dataset   |     AFM |   B_blind |   B_full |   DeepFM |    EASE |    FPMC |   SASRec |
|:----------|--------:|----------:|---------:|---------:|--------:|--------:|---------:|
| kuairand  | -0.0042 |   -0.0028 |  -0.0022 |  -0.0077 | -0.0119 | -0.0017 |  -0.0021 |
| ml1m      |  0.0009 |   -0.005  |   0.0035 |  -0.0004 | -0.0052 |  0.0075 |   0.0028 |
| nyc_tist  | -0.0266 |   -0.0102 |  -0.0317 |  -0.0099 | -0.0119 | -0.0081 |  -0.0244 |
| saopaulo  | -0.0371 |   -0.0096 |  -0.0231 |  -0.0084 | -0.0278 |  0.0002 |   0.0014 |
| yelp      | -0.0039 |   -0.0083 |  -0.0036 |  -0.0055 | -0.0057 | -0.0083 |  -0.0052 |

**SIT peggiora il divario in 29 celle su 35.** L'unico dataset con esito misto è ml1m.

## Celle a supporto basso

| dataset   |   sum |   count |   pct |
|:----------|------:|--------:|------:|
| kuairand  |     0 |     420 |   0   |
| ml1m      |     0 |     525 |   0   |
| nyc_tist  |     0 |     840 |   0   |
| saopaulo  |     0 |     420 |   0   |
| yelp      |    84 |     315 |  26.7 |

Solo yelp ne produce: **84 su 315, il 26,7%**. Sugli altri quattro dataset è zero.

## Stabilità del clustering fra semi (`bfrac`)

| dataset | per seme | spread |
|---|---|---:|
| ml1m | 0.242, 0.242, 0.242, 0.242, 0.242 | 0.000 |
| kuairand | 0.243, 0.244, 0.243, 0.236, 0.237 | 0.008 |
| saopaulo | 0.173, 0.249, 0.260, 0.190, 0.195 | 0.087 |
| nyc_tist | 0.257, 0.234, 0.203, 0.193, 0.354 | 0.161 |
| yelp | 0.231, 0.769, 0.207, 0.769, 0.769 | 0.562 |

ml1m e kuairand sono stabili; nyc_tist e saopaulo oscillano; **yelp collassa** — su tre semi
su cinque il 77% delle richieste è boundary, cioè l'assegnazione situazionale non discrimina.

## Correzione della soglia dei CI — l'evidenza dice che la vecchia regola era giusta

Il brief chiede di passare dal **minimo** alla **media** degli utenti fra i semi. Applicato.
Cambiano stato **21 celle**, tutte da *scartata* a *tenuta*, e sono tutte **la stessa cella**:
yelp, situazione 2, ripetuta sui 7 backbone × 3 metodi.

Utenti per seme in quella cella: **[0, 0, 3222, 0, 0]** — la situazione **non esiste in quattro semi su cinque**.

- vecchia regola (minimo = 0 < 10) → scartata. **Corretto.**
- nuova regola (media = 644,4 ≥ 10) → tenuta. **Sbagliato**: calcolerebbe un CI a due livelli su cinque semi di cui quattro non contribuiscono nulla.

La media nasconde l'assenza; il minimo la rileva. **Raccomandazione: tenere il minimo.**
Sulla griglia base la correzione è comunque inerte — zero celle cambiano stato.
