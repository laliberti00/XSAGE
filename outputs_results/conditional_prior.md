# Analisi condizionale (R4#9): X-SAGE vs profilo statico, dentro e fuori dal prior utente

Contrasto **macro-Cat-MRR(X-SAGE) − macro-Cat-MRR(Steck-b)** calcolato **dentro** due gruppi di
richieste di test, non sull'aggregato. Partizione leakage-free: `Pu` costruito **solo da `df_train`**
(stesse righe di `results_record.py` ~238).

- **ON-PRIOR** — `icm[i_test] == argmax(Pu[u_test])`: la categoria vera e' la dominante dell'utente
- **OFF-PRIOR** — il complemento: l'utente si discosta dalla propria abitudine

Backbone: B_blind, EASE, AFM, SASRec (B_full escluso: richiede training). Seed: [42, 43, 44, 45, 46]. Bootstrap: 1500 resample a livello di richiesta, protocollo `boot_paired`.

## 1 · Ipotesi e criterio (fissati prima di guardare i numeri)

> **H1.** Sulle richieste OFF-PRIOR, X-SAGE supera il profilo statico su macro-Cat-MRR.
>
> **PASS(dataset, backbone)** = media-seed(Δ) > 0 ∧ CI95 bootstrap (seed 42) esclude lo zero ∧ 5/5 seed con Δ>0
> (e' la regola `passpos` del paper). **PASS(dataset)** = maggioranza dei backbone (≥3 su 4).
> **H1 confermata** = PASS su ≥3 dei 5 dataset.

## 2 · Gate di ancoraggio (§5) — **passato**

Ricombinando ON-PRIOR ∪ OFF-PRIOR si riottengono i valori pubblicati in `results_record.csv`
(`metric=macroCatMRR`, `method` ∈ {SIT, Steck-b}) su **tutte** le celle (dataset × backbone × {seed 42 vs `mean_s42`, media-5-seed vs `mean`}).

Nota di implementazione: `results_record.csv` memorizza i valori gia' arrotondati a 5 decimali, quindi
il gate confronta le **differenze** (tolleranza 5e-5 = mezza unita' sul 4o decimale) e non i valori
arrotondati: `round(v,4)==round(ref,4)` da' un falso mismatch quando il valore vero cade sul bordo di
arrotondamento. Differenza massima osservata: **4.9e-06**, cioe' il solo errore di memorizzazione.

## 3 · Diagnostica della partizione

| dataset | richieste ON | richieste OFF | % OFF | utenti senza storia in train | utenti in pareggio su `argmax(Pu)` |
|---|---:|---:|---:|---:|---:|
| ml1m | 28507 | 68692 | 70.7% | 0 | 244/6040 (4.0%) |
| nyc_tist | 6399 | 11303 | 63.9% | 0 | **438/4113 (10.6%)** |
| saopaulo | 9297 | 14524 | 61.0% | 0 | **371/4395 (8.4%)** |
| yelp | 27919 | 8853 | 24.1% | 0 | 133/14322 (0.9%) |
| kuairand | 13002 | 39766 | 75.4% | 0 | **2298/18572 (12.4%)** |

**§6.5** — nessun dataset ha utenti privi di storia in training fra quelli di test: la riga uniforme di
fallback di `Pu` non viene mai usata, quindi ON ∪ OFF copre **tutte** le richieste di test e il gate di §5
vale esattamente come scritto (nessuna richiesta esclusa).

**§6.4** — i pareggi su `argmax(Pu)` superano il 5% su **nyc_tist (10.6%)**, **saopaulo (8.4%)**, **kuairand (12.4%)**: per questi utenti la categoria
"dominante" e' scelta arbitrariamente da `argmax`,
quindi una quota delle loro richieste e' assegnata al gruppo sbagliato. Si riporta, non si corregge.

## 4 · Risultato principale — H1 **NON confermata**

Δ = macro-Cat-MRR(X-SAGE) − macro-Cat-MRR(statico), media sui 5 seed; CI95 bootstrap dal seed 42
(stessa convenzione di `results_record.py`, dove Δ e' la media-5-seed e il CI viene dal seed 42).

### OFF-PRIOR

| dataset | backbone | n richieste | #cat con supporto | X-SAGE | statico | Δ | CI95 | seed con Δ>0 | PASS |
|---|---|---:|---:|---:|---:|---:|---|---:|:-:|
| ml1m | B_blind | 68692 | 18 | 0.1142 | 0.0962 | **+0.0180** | [+0.0167, +0.0194] | 5/5 | ✅ |
| ml1m | EASE | 68692 | 18 | 0.1238 | 0.1065 | **+0.0173** | [+0.0161, +0.0186] | 5/5 | ✅ |
| ml1m | AFM | 68692 | 18 | 0.2110 | 0.1932 | **+0.0179** | [+0.0164, +0.0193] | 5/5 | ✅ |
| ml1m | SASRec | 68692 | 18 | 0.3364 | 0.3309 | **+0.0055** | [+0.0040, +0.0070] | 5/5 | ✅ |
| **ml1m** | **→ dataset** | | | | | | | **4/4 backbone** | **PASS** |
| nyc_tist | B_blind | 11303 | 9 | 0.2211 | 0.2050 | **+0.0161** | [+0.0046, +0.0240] | 5/5 | ✅ |
| nyc_tist | EASE | 11303 | 9 | 0.1825 | 0.1954 | **-0.0129** | [-0.0341, -0.0108] | 1/5 | — |
| nyc_tist | AFM | 11303 | 9 | 0.1854 | 0.2086 | **-0.0231** | [-0.0389, -0.0152] | 0/5 | — |
| nyc_tist | SASRec | 11303 | 9 | 0.2726 | 0.2546 | **+0.0180** | [+0.0071, +0.0204] | 5/5 | ✅ |
| **nyc_tist** | **→ dataset** | | | | | | | **2/4 backbone** | **no** |
| saopaulo | B_blind | 14524 | 10 | 0.1705 | 0.1560 | **+0.0144** | [+0.0143, +0.0252] | 5/5 | ✅ |
| saopaulo | EASE | 14524 | 10 | 0.1645 | 0.1602 | **+0.0043** | [+0.0033, +0.0217] | 5/5 | ✅ |
| saopaulo | AFM | 14524 | 10 | 0.1963 | 0.1928 | **+0.0036** | [-0.0029, +0.0169] | 5/5 | — |
| saopaulo | SASRec | 14524 | 10 | 0.2365 | 0.2206 | **+0.0159** | [+0.0093, +0.0196] | 5/5 | ✅ |
| **saopaulo** | **→ dataset** | | | | | | | **3/4 backbone** | **PASS** |
| yelp | B_blind | 8853 | 16 | 0.0904 | 0.0929 | **-0.0025** | [-0.0102, -0.0053] | 0/5 | — |
| yelp | EASE | 8853 | 16 | 0.0676 | 0.0776 | **-0.0100** | [-0.0185, -0.0038] | 0/5 | — |
| yelp | AFM | 8853 | 16 | 0.0790 | 0.0787 | **+0.0003** | [-0.0026, +0.0005] | 3/5 | — |
| yelp | SASRec | 8853 | 16 | 0.1167 | 0.1162 | **+0.0005** | [-0.0049, +0.0019] | 4/5 | — |
| **yelp** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |
| kuairand | B_blind | 39766 | 39 | 0.0746 | 0.0745 | **+0.0001** | [-0.0010, +0.0013] | 5/5 | — |
| kuairand | EASE | 39766 | 39 | 0.0698 | 0.0716 | **-0.0018** | [-0.0041, +0.0005] | 0/5 | — |
| kuairand | AFM | 39766 | 39 | 0.0647 | 0.0674 | **-0.0027** | [-0.0034, -0.0021] | 0/5 | — |
| kuairand | SASRec | 39766 | 39 | 0.1001 | 0.1000 | **+0.0001** | [-0.0005, +0.0006] | 5/5 | — |
| **kuairand** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |

### ON-PRIOR

| dataset | backbone | n richieste | #cat con supporto | X-SAGE | statico | Δ | CI95 | seed con Δ>0 | PASS |
|---|---|---:|---:|---:|---:|---:|---|---:|:-:|
| ml1m | B_blind | 28507 | 6 | 0.6976 | 0.7266 | **-0.0290** | [-0.0460, -0.0132] | 0/5 | — |
| ml1m | EASE | 28507 | 6 | 0.6431 | 0.7177 | **-0.0746** | [-0.0945, -0.0567] | 0/5 | — |
| ml1m | AFM | 28507 | 6 | 0.5002 | 0.5749 | **-0.0748** | [-0.0897, -0.0611] | 0/5 | — |
| ml1m | SASRec | 28507 | 6 | 0.6889 | 0.7130 | **-0.0242** | [-0.0343, -0.0162] | 0/5 | — |
| **ml1m** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |
| nyc_tist | B_blind | 6399 | 8 | 0.4128 | 0.7001 | **-0.2873** | [-0.3146, -0.2747] | 0/5 | — |
| nyc_tist | EASE | 6399 | 8 | 0.4270 | 0.7214 | **-0.2944** | [-0.3097, -0.2705] | 0/5 | — |
| nyc_tist | AFM | 6399 | 8 | 0.4334 | 0.7366 | **-0.3033** | [-0.3175, -0.2753] | 0/5 | — |
| nyc_tist | SASRec | 6399 | 8 | 0.4735 | 0.6434 | **-0.1699** | [-0.1942, -0.1607] | 0/5 | — |
| **nyc_tist** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |
| saopaulo | B_blind | 9297 | 8 | 0.2895 | 0.6560 | **-0.3664** | [-0.3831, -0.3290] | 0/5 | — |
| saopaulo | EASE | 9297 | 8 | 0.3378 | 0.6208 | **-0.2830** | [-0.2931, -0.2681] | 0/5 | — |
| saopaulo | AFM | 9297 | 8 | 0.3717 | 0.7631 | **-0.3915** | [-0.4044, -0.3744] | 0/5 | — |
| saopaulo | SASRec | 9297 | 8 | 0.4535 | 0.7088 | **-0.2553** | [-0.2331, -0.2116] | 0/5 | — |
| **saopaulo** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |
| yelp | B_blind | 27919 | 2 | 0.4993 | 0.4956 | **+0.0036** | [-0.0039, +0.0144] | 5/5 | — |
| yelp | EASE | 27919 | 2 | 0.4734 | 0.6311 | **-0.1577** | [-0.1503, -0.0037] | 0/5 | — |
| yelp | AFM | 27919 | 2 | 0.5021 | 0.5092 | **-0.0071** | [-0.0284, +0.0040] | 0/5 | — |
| yelp | SASRec | 27919 | 2 | 0.7273 | 0.7361 | **-0.0088** | [-0.0461, +0.0129] | 0/5 | — |
| **yelp** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |
| kuairand | B_blind | 13002 | 8 | 0.4417 | 0.5877 | **-0.1460** | [-0.1655, -0.1349] | 0/5 | — |
| kuairand | EASE | 13002 | 8 | 0.3626 | 0.5118 | **-0.1492** | [-0.2024, -0.1266] | 0/5 | — |
| kuairand | AFM | 13002 | 8 | 0.2318 | 0.3496 | **-0.1178** | [-0.1386, -0.1100] | 0/5 | — |
| kuairand | SASRec | 13002 | 8 | 0.5118 | 0.5429 | **-0.0311** | [-0.0367, -0.0259] | 0/5 | — |
| **kuairand** | **→ dataset** | | | | | | | **0/4 backbone** | **no** |

**Esito.** H1 richiedeva PASS su ≥3 dataset su 5. Osservati: **2/5** (ml1m, saopaulo). → **H1 NON CONFERMATA**.

**Attesa di controllo (non e' il test).** Su ON-PRIOR il profilo statico e' avanti su **19/20** celle (dataset × backbone); X-SAGE non passa in nessun dataset (0/5). La partizione si comporta come
previsto e i risultati sono interpretabili. Va detto pero' che questo controllo e' **quasi tautologico**:
su ON-PRIOR la categoria vera *coincide per costruzione* con l'argmax del prior, quindi un profilo statico
centrato su quell'argmax ha ragione per definizione. Il controllo esclude un errore di partizione, non e'
una validazione indipendente del metodo.

**Nota di lettura.** Δ e' la media sui 5 seed mentre il CI e' il bootstrap del **solo seed 42**: e' la
convenzione di `results_record.py` (`contrast`), riusata qui per coerenza col paper. Di conseguenza in
3 celle su 40 il CI non contiene il Δ medio — segnale di forte varianza
cross-seed, non di un errore: saopaulo/SASRec/ON-PRIOR (Δ=-0.2553, CI [-0.2331, -0.2116]); yelp/EASE/ON-PRIOR (Δ=-0.1577, CI [-0.1503, -0.0037]); yelp/B_blind/OFF-PRIOR (Δ=-0.0025, CI [-0.0102, -0.0053]).
In tutte queste celle il segno di Δ e' comunque concorde su tutti i seed, quindi la classificazione
PASS/no non cambia.

## 5 · Cosa i numeri dicono davvero: dove sta il deficit aggregato

| dataset | Δ aggregato (pubblicato) | Δ dentro ON-PRIOR | Δ dentro OFF-PRIOR | % richieste OFF |
|---|---:|---:|---:|---:|
| ml1m | +0.0103 | -0.0507 | +0.0147 | 70.7% |
| nyc_tist | -0.0683 | -0.2637 | -0.0005 | 63.9% |
| saopaulo | -0.0760 | -0.3241 | +0.0095 | 61.0% |
| yelp | -0.0105 | -0.0425 | -0.0029 | 24.1% |
| kuairand | -0.0053 | -0.1110 | -0.0011 | 75.4% |

Il deficit aggregato contro il profilo statico (−0.068 su nyc_tist, −0.076 su saopaulo) e' **localizzato
quasi interamente nelle richieste ON-PRIOR**. Dentro OFF-PRIOR il divario si chiude: da −0.26/−0.32 a
circa zero. L'intuizione che motivava il brief — l'aggregato non isola cio' che il paper rivendica — e'
quindi **corretta**. Ma la conseguenza che il brief ipotizzava non segue: chiudere il divario non e'
superarlo. Su 3 dataset su 5 X-SAGE **non** batte il profilo statico nemmeno dove il profilo statico e'
strutturalmente sbagliato.

## 6 · Insidie note (§6)

**§6.1 Supporto per categoria — grave su ON-PRIOR.** Restringendo il gruppo il supporto cala e le
categorie ammesse a `MSUPP=20` si riducono:

| dataset | n. categorie | #cat su tutte le richieste | #cat ON-PRIOR | #cat OFF-PRIOR |
|---|---:|---:|---:|---:|
| ml1m | 18 | 18 | 6 | 18 |
| nyc_tist | 10 | 9 | 8 | 9 |
| saopaulo | 10 | 10 | 8 | 10 |
| yelp | 21 | 16 | 2 | 16 |
| kuairand | 43 | 39 | 8 | 39 |

OFF-PRIOR conserva praticamente tutte le categorie. **ON-PRIOR no**: su yelp restano **2 categorie su 21**
(una sola copre il 99.7% del gruppo), su kuairand 8 su 43, su ml1m 6 su 18. La "macro"-media ON-PRIOR e'
quindi una media su pochissime categorie, non commensurabile con quella sull'insieme completo: i valori
ON-PRIOR vanno letti come **direzione**, non come magnitudine.

**§6.2** I due gruppi non sono confrontabili fra loro e non e' stata calcolata nessuna differenza fra gruppi:
OFF-PRIOR e' piu' difficile per entrambi i metodi (X-SAGE passa da ~0.70 a ~0.11 su ml1m). Confrontabile e'
solo il contrasto X-SAGE vs statico **dentro** un gruppo.

**§6.3 Composizione per categoria distorta** (concentrazione delle richieste, quota della categoria piu'
frequente e HHI):

| dataset | top-share ALL | HHI ALL | top-share ON | HHI ON | top-share OFF | HHI OFF |
|---|---:|---:|---:|---:|---:|---:|
| ml1m | 0.277 | 0.183 | 0.422 | 0.346 | 0.220 | 0.142 |
| nyc_tist | 0.264 | 0.166 | 0.376 | 0.223 | 0.201 | 0.149 |
| saopaulo | 0.251 | 0.164 | 0.449 | 0.266 | 0.228 | 0.146 |
| yelp | 0.762 | 0.589 | 0.997 | 0.995 | 0.270 | 0.136 |
| kuairand | 0.242 | 0.103 | 0.703 | 0.528 | 0.132 | 0.067 |

Come atteso, OFF-PRIOR sovra-rappresenta le categorie non dominanti (HHI sempre piu' basso dell'aggregato)
e ON-PRIOR le concentra. Si riporta, non si corregge.

## 7 · Conseguenza per §5.3 del paper

La frase «*the same user heads toward different genres at different moments, and this within-user,
time-varying component is precisely what a static profile cannot represent*» **non e' sostenuta** da questa
misura. Sulle richieste in cui l'utente si discosta dalla propria abitudine — esattamente il regime in cui
un profilo costante e' strutturalmente sbagliato — X-SAGE e' consistentemente avanti solo su ml1m e
saopaulo; su nyc_tist e' diviso (2 backbone su 4), su yelp e kuairand il contrasto e' nullo o negativo.

La formulazione va ammorbidita in qualcosa di misurato, del tipo: il deficit di X-SAGE rispetto alla
calibrazione statica per-utente si concentra sulle richieste che cadono sulla categoria abituale
dell'utente; fuori da quel regime i due metodi sono sostanzialmente alla pari, con un vantaggio piccolo e
consistente per X-SAGE su 2 dei 5 dataset. Nessuna partizione alternativa e' stata cercata dopo aver visto
i risultati.

## 8 · Riproducibilita'

```bash
python scripts/validation/conditional_prior.py --smoke     # gate su nyc_tist/42/B_blind
python scripts/validation/conditional_prior.py             # run completa (ricalcolo dalla pipeline)
python scripts/validation/conditional_prior.py --from-cache  # idem, dagli array grezzi cachati
```

Le due sorgenti — ricalcolo dalla pipeline (~70 min su questa macchina) e array per-richiesta gia'
cachati dalla run pubblicata (`outputs_results/cache/raw_<city>.npz`, ~2 min) — danno righe
**numericamente identiche su tutte le 200 righe** (max |diff| = 0.0 su ogni colonna), e i due
percorsi passano lo stesso gate con la stessa differenza massima per dataset.
`results_record.py`, `mind_prep.py` e `xsage/` non sono stati modificati.
