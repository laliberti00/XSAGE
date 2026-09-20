# WI-0d — Audit di nomenclatura + ipotesi boundary

## ⚠️ Dichiarazione di rischio metodologico (brief §1.0)

L'ipotesi boundary e' la **seconda** formulata dopo la falsificazione della prima (WI-0c, esito NON CONFERMATO). Se confermata riscatterebbe la narrativa: e' esattamente il pattern che un council avversariale censura. Per questo la regola di decisione e' scritta **sopra** ai numeri, **AMBIGUO e' trattato come NON CONFERMATO**, e non e' previsto un terzo tentativo. Dopo WI-0d la fase diagnostica e' chiusa.

## Task 0.a — Audit di nomenclatura dei backbone

**Esito: 7/7 corrispondono. Il manoscritto e' corretto.** Il nome nel paper combacia con l'algoritmo reale per tutti i backbone (verifica per valore: ML-1M macro-Cat-MRR del manoscritto vs `results_record.csv`). L'unica anomalia e' **interna al repo**: il file `FM.scores.npy` contiene punteggi **BPR** (`cornac_backbone.py:26`), un nome ereditato; il paper lo chiama correttamente BPR e chiama FM il ContextAwareFM (`B_full`). Dettaglio in `task0a_backbone_naming.csv`. Nei capitoli di tesi non esistono tabelle di risultati con nomi di backbone: nessun rischio.

## Task 0.b — Quale repo e' quello vero

**Le due cartelle sono storie git NON collegate.**

| | xsage-clean | X-SAGE |
|---|---|---|
| HEAD | `db608f8` | `5724fe6` |
| remote | laliberti00/XSAGE | **knowmis/X-SAGE — l'URL citato nel paper** |
| contiene `db608f8` | sì | **NO** |
| file tracciati | 303 | 63 |

Solo **11 file in comune**, di cui **9 differiscono nel contenuto** (inclusi `xsage/recommendation.py`, `l0_sensing.py`, `l1_perception.py`, `l3_projection.py`, `data.py`). `X-SAGE` e' un **repackaging deliberato**: ha backbone propri (`xsage/backbones/*.py`, assenti in xsage-clean), una pipeline propria (`scripts/run_pipeline.py`) e `data/processed/` committati. Il suo `recommendation.py` documenta il combiner **senza** lambda e senza `m_sel` — piu' aderente al paper V2 di quanto lo sia xsage-clean. **Nessuno ha pero' mai verificato che riproduca i numeri**: il gate 0.38479 non e' mai stato girato su quel codice. Dettaglio in `task0b_repo_diff.csv`.

## Task 1 — Ipotesi boundary

Backbone `B_blind` (stessa dichiarazione del Task 0.a), kappa* congelato, 5 seed, bootstrap a due livelli su utenti e seed. Priorita' dichiarata prima dei numeri: decisivo **ml1m**, conferma **saopaulo**, **nyc_tist** non vota.

### Gate di ancoraggio

| dataset | Cat-MRR | atteso | esito |
|---|---|---|---|
| ml1m | 0.38479 | 0.38479 | OK |
| nyc_tist | 0.34162 | 0.34162 | OK |
| saopaulo | 0.40045 | 0.40045 | OK |

### Correlazioni (Spearman, mediate sui 5 seed)

Soglia pre-registrata: |rho| >= 0.3

| dataset | bshare~entropia | bshare~n_situazioni | bshare~lunghezza | supera soglia |
|---|---|---|---|---|
| ml1m | +0.5473 | +0.5549 | +0.0778 | sì |
| nyc_tist | +0.2729 | +0.2727 | +0.0427 | NO |
| saopaulo | +0.4497 | +0.4535 | +0.1520 | sì |

### Gradiente del gate gamma (|T| intero: livelli naturali, nessun binning)

Guardia di falsificazione: BASE riportato accanto a Delta.

**ml1m**

| \|T\| | gamma | n richieste | Δ | CI 95% | BASE |
|---|---|---|---|---|---|
| 1 | 1.0000 | 73648 | +0.03465 | [+0.03363, +0.03554] | 0.36246 |
| 2 | 0.5000 | 17976 | -0.00300 | [-0.00395, -0.00213] | 0.34529 |
| 3 | 0.3333 | 5185 | -0.00938 | [-0.01059, -0.00824] | 0.36587 |
| 4+ | 0.2500 | 388 | -0.00722 | [-0.00998, -0.00467] | 0.40111 |

**nyc_tist**

| \|T\| | gamma | n richieste | Δ | CI 95% | BASE |
|---|---|---|---|---|---|
| 1 | 1.0000 | 13312 | +0.06423 | [+0.05796, +0.07042] | 0.28838 |
| 2 | 0.5000 | 3282 | +0.02618 | [+0.02103, +0.03185] | 0.25371 |
| 3 | 0.3333 | 988 | +0.01048 | [+0.00466, +0.01574] | 0.26832 |
| 4+ | 0.2500 | 119 | +0.00072 | [+nan, +nan] | 0.28119 |

**saopaulo**

| \|T\| | gamma | n richieste | Δ | CI 95% | BASE |
|---|---|---|---|---|---|
| 1 | 1.0000 | 18739 | +0.03677 | [+0.03099, +0.04153] | 0.36939 |
| 2 | 0.5000 | 4589 | +0.00414 | [+0.00029, +0.00970] | 0.34901 |
| 3 | 0.3333 | 491 | -0.00287 | [-0.01925, +0.01316] | 0.36583 |
| 4+ | 0.2500 | 1 | +0.00000 | [+nan, +nan] | 1.00000 |

### Regressione Δ(u) ~ boundary_share + entropia + log(lunghezza)

| dataset | β bshare | β entropia | β loglen | R² |
|---|---|---|---|---|
| ml1m | -0.05247 (0.00659)* | -0.12595 (0.01086)* | +0.00855 (0.00236)* | 0.03764 |
| nyc_tist | -0.04350 (0.00492)* | +0.00196 (0.02085) | +0.04735 (0.00613)* | 0.01899 |
| saopaulo | -0.06272 (0.00436)* | -0.00612 (0.00658) | +0.01126 (0.00370)* | 0.02009 |

Correlazioni fra regressori:

- **ml1m**: bshare,entropy +0.459 · bshare,loglen +0.014 · entropy,loglen +0.086
- **nyc_tist**: bshare,entropy +0.182 · bshare,loglen -0.002 · entropy,loglen +0.106
- **saopaulo**: bshare,entropy +0.366 · bshare,loglen +0.063 · entropy,loglen +0.162

### Esito (regola pre-registrata §1.4, applicata meccanicamente)

Servono tutte e quattro: |rho| >= 0.30 · gradiente boundary >= gradiente n_situazioni · guardia superata (BASE si muove meno della meta' di Delta) · Delta significativamente negativo ad alta incertezza.

- **ml1m** (decisivo) → **AMBIGUO** — rho: sì (+0.547) · gradiente: sì (0.07477 vs 0.07239) · guardia: VIOLATA (BASE varia 0.15342, Δ varia 0.10245) · Δ<0 ad alta incertezza: sì
- **nyc_tist** (NON VOTA) → **NON CONFERMATO** — rho: no (+0.273) · gradiente: sì (0.05282 vs 0.03779) · guardia: VIOLATA (BASE varia 0.10755, Δ varia 0.16720) · Δ<0 ad alta incertezza: no
- **saopaulo** (conferma) → **AMBIGUO** — rho: sì (+0.450) · gradiente: sì (0.05759 vs 0.03004) · guardia: VIOLATA (BASE varia 0.06940, Δ varia 0.08582) · Δ<0 ad alta incertezza: no

### esito = NON CONFERMATO

(verdetto grezzo su ml1m: **AMBIGUO**; per la regola §1.0 AMBIGUO e' trattato come NON CONFERMATO)

→ Il limite si scrive nella forma gia' stabilita da WI-0c: il beneficio si concentra sugli utenti a situazione stabile. Mezza pagina nei limiti di ogni capitolo. **Fase diagnostica chiusa: dopo WI-0d non si apre nient'altro.**

## Domande emerse (riportate, non eseguite)

1. **`knowmis/X-SAGE` non e' mai stato verificato numericamente.** E' il repo che diventera' pubblico ed e' l'URL citato nel manoscritto, ma non contiene `db608f8` e i suoi moduli core differiscono. Prima di renderlo pubblico andrebbe girato il gate (0.38479 / 0.34162 / 0.40045) su quel codice.
2. **`FM.scores.npy` contiene BPR.** Il paper e' corretto, ma il nome del file e' una trappola per chiunque legga il repo (incluso un referee che scarichi il codice).
