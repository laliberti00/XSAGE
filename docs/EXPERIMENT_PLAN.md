# X-SAGE — Pre-registrazione delle regole di decisione

> Questo file è committato **PRIMA** di far girare gli esperimenti sotto. La regola di decisione è fissata
> qui, non scelta dopo aver visto i numeri. L'hash del commit e il timestamp sono la prova.

**Timestamp pre-registrazione:** 2026-07-01 17:36:26 CEST
**Autore:** L. Aliberti · branch `exp/second-dataset-feasibility`

---

## B6 — Ablation-contrasto: "joint > best-single-component"

**Contesto.** L'ablation clusterizza le situazioni su tre descrittori: `ctx` (solo c̃), `intent` (solo e),
`joint` ([c̃‖e]). Metrica-faro: macro-Cat-MRR@20 (min-support |R_c|≥20), backbone B_blind, 5 seed {42–46}.

**Quantità testata.** Contrasto `Δ = macro(joint) − macro(best)`, dove **best = argmax(ctx, intent)** per
quel dataset (best su saopaulo = intent; su ml1m = ctx), **non** un fisso joint−ctx.

**Regola di decisione (fissata a priori) — IDENTICA ai contrasti principali (regola-casella a 3 condizioni):**
> **`joint > best-single-component`** per un dataset **⟺** tutte e tre:
> 1. **`Δ > 0`** (media 5-seed);
> 2. **CI-bootstrap esclude lo zero**: il bootstrap per-richiesta della differenza (ricampiona le richieste, paired) ha `CI_lo > 0`;
> 3. **5/5 seed concordi**: `Δ_seed > 0` su tutti e cinque i seed.
>
> **`joint ≈ best`** (equivalenza) **⟺ TOST ±0.005**: il CI-bootstrap del Δ è **dentro** `(−0.005, +0.005)`.
> Se non vale né la superiorità né l'equivalenza → **inconcludente** (Δ piccolo ma seed-fragile).

> **Deviation log (2026-07-01):** la prima versione (commit 74b0564) usava `Δ > +0.005` come 1ª condizione.
> Corretta **prima del run finale (5 dataset)** a `Δ > 0`, per **coerenza con la regola-casella dei contrasti
> principali** (winner/ridondante/null); la banda ±0.005 resta come test di **equivalenza (TOST)** separato,
> non come soglia di superiorità. Nessun numero visto tra le due versioni (il run finale non era ancora girato).

**Requisiti tecnici del test (obbligatori):**
- il bootstrap è sulla **differenza per-richiesta** di macro-Cat-MRR (ricampiona le richieste), non sul
  confronto di due medie aggregate;
- gli array per-richiesta delle tre varianti sono salvati (`ablation_contrast.py`, cm42);
- CI bootstrap su seed 42; consistenza sui 5 seed.

**Esito atteso (stima pre-run, NON un claim):** ml1m Δ(joint−ctx)≈−0.0012 → equivalenza/ctx≥joint;
nyc Δ(joint−intent)≈+0.0070 → possibile `joint>best`; **saopaulo Δ(joint−intent)≈+0.0015 → dentro banda →
equivalenza.** L'esito di saopaulo è deciso da questa regola, scritta prima di rilanciarlo.

**Script:** `scripts/yelp/ablation_contrast.py`. **Output:** `logs/tonight.log` (sezione B6).

---

## Nota di scope (non una regola, un promemoria)
- I 7 backbone esistono solo su ml1m/nyc_tist/saopaulo; kuairand/amazoncd/mind/yelp sono a 2 (BPR, FM).
  Le caselle si decidono sul focale B_full, presente ovunque.
- κ è selezionato su validation e il test è misurato una volta; l'identità κ=0 è un fallback sicuro
  (recupera il backbone), **non** una garanzia di non-regressione (SIT può regredire su una metrica —
  es. nyc/EASE/HR@20_u = −0.042).
