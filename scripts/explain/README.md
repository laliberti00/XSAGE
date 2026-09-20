# `scripts/explain/` — provenienza degli script del Turno 0

I quattro `t0_measure*.py` **non sono stati scritti qui**. Sono gli script che hanno prodotto le misure
del Turno 0 (council cap. 6, explainability, 28 agosto 2026), documentate in
`council_cap6_explainability/Turno0_fattibilita.md`.

Erano stati scritti fuori dal repo, nello scratchpad di sessione
(`/private/tmp/claude-501/.../5a2b49a6-.../scratchpad/`), per rispettare il vincolo «nessuna scrittura
nel repo» del brief di quel turno. `/private/tmp` viene azzerato al riavvio: allo sbianchettamento
dello scratchpad gli script risultavano perduti.

**Come sono stati recuperati (21 settembre 2026).** Erano stati creati con heredoc via Bash, quindi il
sorgente integrale era registrato nel transcript della sessione:
`~/.claude/projects/-Users-lucaaliberti-Downloads-xsage-clean/5a2b49a6-047b-489b-ac20-87da8428826a.jsonl`.
Estratti verbatim dai campi `input.command` dei blocchi `tool_use`, senza riscrittura di una sola riga.

Verifica di integrità: tutti e quattro compilano (`python -m py_compile`). **Non sono stati eseguiti**,
come da brief.

| file | righe | byte | misura |
|---|---:|---:|---|
| `t0_measure.py` | 145 | 7.522 | ancoraggi, costo di persistenza, core-only vs boundary-aware |
| `t0_measure2.py` | 93 | 5.587 | supporto di PN/PS, strati su tutto il test, costo |
| `t0_measure3.py` | 61 | 3.729 | PN/PS discrimina Ĉ vero da Ĉ falso? item-level vs category-level |
| `t0_measure4.py` | 53 | 3.420 | quota di spostamento attribuibile a Ĉ (misura continua) |

## Due cose da sapere prima di riusarli

**1. I percorsi sono cablati.** Ogni script punta a `SCRATCH = /private/tmp/claude-501/.../5a2b49a6-.../scratchpad`,
che non esiste più, e aggiunge al `sys.path` l'interprete della tesi
(`/Users/lucaaliberti/Downloads/IntentAwareRS_thesis`). Per rieseguirli va cambiato `SCRATCH`.
Sono conservati **byte-identici** a quelli che hanno prodotto i numeri pubblicati: non si correggono
in place, si copia.

**2. Ĉ è definito sulla riga della situazione, non sul vettore della richiesta.**
`t0_measure3.py:33` e `t0_measure4.py` costruiscono gli insiemi Ĉ come
`np.argsort(-b_z, axis=1)[:, :3]` indicizzato con `kte`, cioè le top-3 della riga **`b_z[k]`** della
situazione principale — mentre il nudge applicato è correttamente `mem @ b_z` (`:22`).
Sulle richieste **core** le due cose coincidono; sulle **boundary** (24,2% su ml1m) no.

Il brief A.5 specifica invece Ĉ come le top-*m* di **`mem @ b_z` della richiesta**. Le due definizioni
non sono la stessa: i numeri del Turno 0 sono validi per la definizione core-only con cui sono stati
prodotti, e vanno citati così. Se B.2 usa «la quota di attribuzione di A.5», deve dichiarare quale
delle due definizioni sta usando.
