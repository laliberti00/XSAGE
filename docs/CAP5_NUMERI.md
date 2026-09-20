# Capitolo 5 — Estratto numerico verificato

**Generato:** 2026-08-22 · **Regola:** ogni cifra ha la sua fonte. Cio' che non si trova e'
marcato `[NON TROVATO]` con la ricerca fatta. Niente stime, niente deduzioni, niente ricalcoli
da altre quantita'.

**Fonte primaria:** `outputs_results/results_record.csv` — 2205 righe = 5 dataset x 7 backbone
x 21 metriche x 3 metodi. Protocollo: 5 semi {42-46}, rango atteso sotto pareggi
(McSherry-Najork), bootstrap B=1500, Holm su due famiglie, TOST +/-0.005.

---

## 0. Domanda aperta chiusa: quali cinque dataset

**Verificato, non dedotto** (`results_record.csv`, colonna `dataset`):

`kuairand` · `ml1m` · `nyc_tist` · `saopaulo` · `yelp`

**Non presenti nel record finale:** `amazoncd`, `mind`, `tokyo_tist`, `bangkok`, `istanbul`,
`tsmc_nyc`, `tsmc_tky`, `yelp_bal`.

Attenzione: `amazoncd` e `mind` compaiono nei documenti di luglio
(`docs/FREEZE_RECORD.md` §1 li elenca fra i 7 primari) ma **non sono nel record**.
`docs/STATO_LAVORI.md` (2026-07-08) ne da' la ragione: *"NON girano su 16GB RAM (matrici
1.8-2.0GB -> picco oltre la RAM)"*. Da dichiarare nei limiti del capitolo.

---

## 1. Tabella della caratterizzazione — backbone focale `B_full`

Metrica: **macro-Cat-MRR@20**, min-support 20. `L1 = SIT - BASE`, `L2 = SIT - Steck-b`.
Fonte: `results_record.csv`, righe `backbone=B_full`, `metric=macroCatMRR`, `method=SIT`,
colonne `delta_l1`, `seeds_l1`, `delta_l2`, `seeds_l2`, `box`.

| dataset | ΔL1 | semi | ΔL2 | semi | casella |
|---|---|---|---|---|---|
| **ml1m** | **+0.00395** | 5/5 | **+0.00525** | **5/5** | **winner** |
| nyc_tist | +0.00747 | 5/5 | −0.02003 | 0/5 | ridondante |
| saopaulo | +0.00754 | 5/5 | −0.02555 | 0/5 | ridondante |
| yelp | +0.00203 | 5/5 | +0.00034 | 3/5 | ridondante |
| kuairand | +0.00011 | 4/5 | −0.00377 | 0/5 | **nullo** |

Regola della casella (`docs/FREEZE_RECORD.md` §6): un contrasto passa se
`Δ>0 ∧ CI bootstrap esclude 0 ∧ 5/5 semi concordi`. ¬L1 -> nullo · L1∧¬L2 -> ridondante ·
L1∧L2 -> winner.

**Nota su yelp:** ΔL2 e' positivo (+0.00034) ma con 3/5 semi, quindi non passa il gate e la
casella e' *ridondante*. I documenti di luglio lo descrivevano come "amplificatore della
maggioranza" (`macro_avg_summary.csv`, verdetto `majority-amplifier (micro fake)`): sono due
letture della stessa cosa, ma il record finale dice ridondante.

**Nota su kuairand:** e' la sonda nulla, e ΔL1 = +0.00011 e' ~36 volte piu' piccolo di ml1m.
E' l'argomento di credibilita': il metodo non fabbrica un vincitore dove non c'e' segnale.

---

## 2. Robustezza sui sette backbone — ml1m

Il dato che sostiene *"non e' un artefatto del focale"*.
Fonte: `results_record.csv`, `dataset=ml1m`, `metric=macroCatMRR`, `method=SIT`.

| backbone | ΔL1 | semi | ΔL2 | semi |
|---|---|---|---|---|
| B_blind (= **BPR** nel paper) | +0.01385 | 5/5 | **+0.01354** | 5/5 |
| B_full (= **FM**, focale) | +0.00395 | 5/5 | **+0.00525** | 5/5 |
| EASE | +0.01285 | 5/5 | **+0.01094** | 5/5 |
| DeepFM | +0.00480 | 5/5 | **+0.00640** | 5/5 |
| AFM | +0.00974 | 5/5 | **+0.01312** | 5/5 |
| FPMC | +0.00628 | 5/5 | **+0.00815** | 5/5 |
| SASRec | +0.00130 | 5/5 | **+0.00380** | 5/5 |

**ΔL2 > 0 con 5/5 semi su tutti e sette**, da quello context-blind al sequenziale forte.
Sette su sette, nessuna eccezione.

⚠️ **Trappola di nomenclatura**: nel repo `B_blind` sta nel file `FM.scores.npy` ma contiene
punteggi **BPR** (`scripts/mind/cornac_backbone.py:26`). Il manoscritto e' corretto — chiama BPR
cio' che e' BPR e FM il ContextAwareFM. Verificato 14/14 su entrambe le tabelle di risultati.

---

## 3. Conteggi di esposizione — denominatore 28

**Il denominatore e' 7 backbone x 4 COLONNE del manoscritto**, dove la colonna Foursquare e' la
**media di nyc_tist e saopaulo**. Le 4 colonne sono: **ML-1M · Foursquare (=nyc+sao) · Yelp ·
KuaiRand**.

> ⚠️ **I quattro dataset di `tab:exp` NON coincidono con i quattro della caratterizzazione.**
> La caratterizzazione (§1) elenca 5 dataset separati; la tabella di esposizione ne mostra 4
> perche' fonde le due citta' Foursquare in una colonna. Sono 5 dataset in entrambi i casi, ma
> presentati con granularita' diversa. Scriverlo esplicitamente nel capitolo, o il lettore
> conta male.

| effetto | conteggio | fonte |
|---|---|---|
| **Gini scende** (meno concentrazione) | **26 / 28** | manoscritto `tab:exp`; ricalcolo indipendente su `results_record.csv`: **26/28** ✓ |
| **Coverage sale** | **26 / 28** | manoscritto; ricalcolo: **26/28** ✓ |
| **Long-tail@20 sale** | **20 / 28** | manoscritto; vedi §3.1 per il criterio |

Caso piu' netto citato dal manoscritto: **EASE su Yelp, LT@20 da 0.0067 a 0.1041**.

### 3.1 Il criterio del 20/28 — DETERMINATO

Un ricalcolo a piena precisione su `results_record.csv` da' **21/28**, non 20. La differenza e'
**una sola cella**: `AFM × KuaiRand`, che migliora di **+0.00004**.

**Il conteggio del manoscritto e' fatto sui valori come STAMPATI in tabella, a 4 decimali.**
A quella precisione AFM/KuaiRand mostra `0.0011 -> 0.0011`, cioe' nessun aumento. Contando
cosi' si ottengono **esattamente 20/28**, verificato riparsando `tab:exp` dal `template.tex`.

Conferma indipendente: le 8 eccezioni a precisione stampata sono BPR/Yelp, FM/KuaiRand,
AFM/KuaiRand, FPMC/Yelp, FPMC/KuaiRand, SASRec/Foursquare, SASRec/Yelp, SASRec/KuaiRand —
cioe' **7 su sonde nulle + 1 su SASRec-Foursquare**, che e' *alla lettera* la descrizione del
manoscritto:

> *"the long-tail share increases in 20 of 28; the exceptions lie on the null probes or on the
> SASRec--Foursquare cell, and are small in absolute value."*

Frase sul criterio generale di marcatura (stessa sezione):

> *"Under the same rule, 62 of the 84 exposure comparisons resolve to a reliable improvement and
> 16 to a confirmed equivalence, so 78 carry an explicit mark."*

**La tesi puo' adottare 20/28 con le parole del manoscritto**, e il criterio e' riportabile:
conteggio sui valori tabulati.

---

## 4. K ed ε selezionati per dataset

Fonte: `outputs_results/neutrality_ablation_<ds>.csv`, riga `mode=full`, colonne `K`, `eps`,
`bfrac`. Selezione anti-circolare: K per silhouette sull'assegnazione core, ε perche' la quota
di confine cada nella banda [0.10, 0.30].

| dataset | K | ε | quota di confine |
|---|---|---|---|
| ml1m | 5 | 0.03 | 24.2 % |
| nyc_tist | 8 | 0.02 | 25.7 % |
| saopaulo | 4 | 0.03 | 17.3 % |
| yelp | 3 | 0.05 | 23.1 % |
| kuairand | 4 | 0.01 | 24.3 % |

Tutte e cinque le quote di confine cadono nella banda dichiarata. Il manoscritto cita per ml1m
il **24 %**, coerente.

---

## 5. Costi del modulo

Fonte: `outputs_results/explain/cost_ml1m.txt` (misurato, stessa macchina).

| voce | valore |
|---|---|
| **parametri del modulo** | **205** (K·macro + K·dim = 5·18 + 5·23) |
| **inferenza per richiesta** | **0.102 ms** |
| **fit totale** | **168.34 s** |
| — descrittore (L0+L1) | 8.25 s |
| — selezione K | 72.47 s |
| — selezione ε | 73.64 s |
| — rough k-means (L2) | 13.89 s |
| — bias di situazione b̃ | 0.09 s |
| — inferenza (assegna + re-rank test) | 9.89 s |
| dati | train 801 218 · test 97 199 · macro 18 · dim(v) 23 |

Confronto riportato nella stessa fonte: BPR fit ~secondi; B_full (FM torch+MPS) ~1-2 min per
addestramento. Il modulo situazionale e' **ordini di grandezza piu' piccolo di un backbone
neurale**: 205 parametri contro `(n_users + n_items + ctx) · d`.

---

## 6. Gate di ancoraggio

Valore di riferimento: **Cat-MRR@20 micro di SIT@κ\*** su backbone `B_blind`, seme 42,
percorso battery.

| dataset | valore | κ\* |
|---|---|---|
| ml1m | **0.38479** | 0.5 |
| nyc_tist | **0.34162** | 0.5 |
| saopaulo | **0.40045** | 0.25 |

Replicato **esatto a 5 decimali 15 volte su 15** nei probe WI-0, WI-0b, WI-0c, WI-0d
(fonte: `outputs_results/{diagnostics,probe_wi0b,probe_wi0c,probe_wi0d}/*_summary.md`).

**Replica su ri-organizzazione indipendente** (`outputs_results/probe_wi0d/wi0d_summary.md`,
Task 0.b): la pipeline di `~/Downloads/X-SAGE` — codebase rifattorizzato, storia git separata,
moduli core diversi — riproduce **tutti e tre i valori esatti a 5 decimali**, con dati
processati identici per md5 (15 file su 15), iperparametri identici e stessi K/ε selezionati.

---

## Questioni emerse (riportate, non risolte)

1. **`amazoncd` e `mind` sono nei documenti di freeze ma non nel record.** Il freeze li elenca
   fra i 7 dataset primari; il record ne ha 5. La ragione documentata e' il limite di RAM.
   Va dichiarato nei limiti, altrimenti il freeze e il record si contraddicono.
2. **Il brief P-1 riporta 22 come ricalcolo diretto del long-tail; io ottengo 21.** La
   differenza dipende probabilmente da come viene trattata la colonna Foursquare (fusa o
   separata): con le 4 colonne del manoscritto sono 21 a piena precisione e 20 a precisione
   stampata. Con 5 dataset separati il denominatore sarebbe 35, non 28.
3. **La casella di `yelp` e' *ridondante* nel record ma *majority-amplifier* nei documenti di
   luglio.** Non e' una contraddizione numerica (ΔL2 = +0.00034 con 3/5 semi non passa il gate)
   ma sono due descrizioni diverse dello stesso dato: scegliere quale usare nel capitolo.
