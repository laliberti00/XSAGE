# Fase 01 — Sensing (L0)

Stato: **VALIDATO** (2026-06-18)

Il sensing è il livello L0: trasforma lo stream di check-in grezzi (già
preprocessati dalla fase 00) in input **strettamente causali** per-richiesta.
È volutamente sottile — la maggior parte del lavoro pesante (split, k-core,
cold-drop) è già stata fatta e congelata nella fase 00; qui si estrae, per ogni
richiesta `q = (u, t)`, il contesto causale dell'utente.

---

## Cosa fa il sensing

1. **Split cronologico per-utente.** Ogni utente ha le sue interazioni ordinate
   nel tempo e divise 80/10/10 (train/val/test). Lo split è **per-utente**, non
   globale: ogni utente contribuisce a tutti e tre gli split secondo il proprio
   asse temporale.
2. **Filtro di supporto k-core=10.** Si tengono solo utenti e item con ≥10
   interazioni nel set completo (imposto in fase 00, pre-split).
3. **Intent proxy causale.** Per ogni richiesta `(u, t)` si estraggono le ultime
   `n` macro-categorie visitate dall'utente **strettamente prima** di `t`
   (`time_local < t`), con i relativi delta-tempo. Questo è il proxy d'intento
   che alimenta la perception (L1).

## Come è implementato (clean repo)

- **File:** [`xsage/l0_sensing.py`](../../xsage/l0_sensing.py)
- **Funzione cardine:** `build_recent_window(target_df, history_df, macro_to_idx, n=5)`
  costruisce, per ogni riga di `target_df`, la finestra causale delle ultime `n`
  macro dello **stesso utente** a `time_local < t`.
  - La causalità è garantita da `np.searchsorted(rec["t"], t, side="left")`: il
    cut prende solo gli eventi **strettamente precedenti** `t` (`side="left"` →
    esclude eventi al tempo `t` stesso).
  - Regola di storia: train→train (self, prima di t), val→train (full),
    test→train ∪ val (refit).
- Lo split, il k-core e il cold-drop a monte sono materializzati nei parquet/URM
  letti da [`xsage/data.py:load_city`](../../xsage/data.py).

## Quali scelte e perché

- **Split per-utente, non globale.** Uno split globale per timestamp metterebbe
  in train gli utenti "vecchi" e in test quelli "nuovi", confondendo cold-start
  con drift. Lo split per-utente valuta la previsione del *prossimo* comportamento
  di **ogni** utente, che è il claim della tesi (re-ranking situazionale per
  l'utente che sta agendo ora).
- **Intent strettamente `< t` (causalità).** Il contesto non deve mai includere
  l'evento target né eventi simultanei: includerli sarebbe leakage. `side="left"`
  realizza la disuguaglianza stretta.
- **k-core=10.** Soglia di supporto standard per avere FM/CF stabili ed evitare
  utenti/item con segnale insufficiente.

## Come è stato validato

### Split causale — 0 violazioni su 5/5

Test: [`tests/test_causal_split.py`](../../tests/test_causal_split.py). Invariante:
per ogni utente, `max(train.time) < min(test.time)`. Conteggio utenti che
violano (deve essere 0):

| città | utenti con train ≥ test | totale utenti | esito |
|---|---|---|---|
| Istanbul | 0 | 22631 | PASS |
| Bangkok | 0 | 6316 | PASS |
| NYC-TIST | 0 | 4113 | PASS |
| Sao Paulo | 0 | 4395 | PASS |
| Tokyo-TIST | 0 | 7160 | PASS |

**Criterio PASS (deterministico, R6):** 0 violazioni su tutti gli utenti di tutte
le città. → **PASS 5/5**, 0 violazioni totali.

### Integrità del supporto (k-core) e nota metodologica

L'integrità del set (k-core=10, n_test esatti, densità, 0 vuoti) è in
[fase 00](../00_data_backbone/), con verdetto **PASS 5/5**.

> **Lezione di rigore (documentata di proposito).** Lo script di verifica
> dell'integrità inizialmente dava **FAIL 5/5**, perché misurava il k-core su
> `train+val` invece che sul set completo `train ∪ val ∪ test`. Per costruzione,
> dopo lo split 80/10/10 il minimo di interazioni per utente in `train+val`
> scende sotto 10 (un utente con esattamente 10 interazioni ne ha ~8 in train),
> quindi quel FAIL era un **artefatto della misura**, non un problema dei dati.
> Misurando sul set su cui il k-core è realmente imposto, l'item-min risulta = 10
> esatto ovunque e i `n_test` coincidono al numero → **PASS 5/5**.
> **Abbiamo corretto la verifica, non i dati.** È un esempio del principio: un
> criterio sbagliato può far fallire dati corretti; il verdetto va sempre
> ancorato al set/convenzione giusti.

---

**Verdetto fase 01: VALIDATO.** Split causale 0/N su 5/5 città; intent proxy
strettamente causale per costruzione (`side="left"`); supporto k-core validato
in fase 00 con la correzione metodologica documentata.
