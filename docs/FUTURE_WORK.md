# X-SAGE — Future Work (documento vivo)

> Registro delle direzioni future: idee proposte, perché, come, fattibilità, open points.
> Compagno di [STATE_OF_WORK.md](STATE_OF_WORK.md) (stato attuale) e [STORIA](STORIA_PROGETTO_XSAGE.md) (cronologia).
> Da aggiornare quando un'idea matura, viene eseguita (→ passa in STATE_OF_WORK) o scartata.
> Ultimo aggiornamento: 2026-06-26.

---

## Contesto (da dove nascono queste idee)
Dopo il protocollo uniforme k-core10, il quadro è a **due livelli**:
- **Enhancer** (SIT-su-B_full > B_full): winner multi-dominio → **ml-1m (film) + nyc, saopaulo (POI)**, cross-seed t=3-9.
- **Non-ridondanza** (SIT > Steck-b): **solo ml-1m**. Ovunque altrove la preferenza-utente statica (Steck-b) vince, sui POI nettamente (i check-in sono abitudinari).

Le idee qui sotto nascono per **premiare/operazionalizzare il meccanismo situazionale**: non spingere sempre, ma **agire quando serve** (non-ridondante e confidente), e **misurarlo**.

---

## F1 — Modello IBRIDO statico + situazionale (Steck-b + SIT non-ridondante)
**Idea.** Usare Steck-b (preferenza statica, vince quasi ovunque) come **base**, e far agire SIT **solo per la parte non-ridondante** (ciò che la situazione aggiunge oltre l'abitudine).
- **(A) Additivo**: `ŝ = s_B + κ_steck·b_utente[u] + κ_sit·b_situazione[z]` (κ tarati su val).
- **(B) Residualizzato**: `b_sit_nuovo = b_situazione − proj(b_situazione su b_utente)` → SIT contribuisce solo l'ortogonale (= 1−R² dentro il modello).

**Perché.** L'ibrido è **provabilmente ≥ Steck-b ovunque** (non perde mai contro la personalizzazione statica) e **> su ml-1m**. Trasforma "ml-1m unico winner" in *"un modello mai peggiore della baseline forte, migliore dove la situazione è non-ridondante"*. Resta interpretabile (si vede *quando* il residuo è grande).

**Fattibilità.** ALTA. Riusa `b_z_user` + `b_z` (già calcolati). Misura su **B_blind = quasi gratis** (dati salvati, niente re-train); validazione su B_full = solo ri-run stadio battery (re-train B_full), non l'intera pipeline.

**Rischio/framing.** Ri-posiziona Steck-b da *baseline* a *componente del modello* → la situazione diventa correzione residua. Onesto e più robusto, ma cambia la narrativa (situazione co-protagonista, non star). Da decidere: enhancer puro (claim principale) + ibrido come "si può fare ancora meglio".

**Open points.** κ 2D vs residualizzazione fissa; tenere anti-circolare (b_utente da train); quale framing.

---

## F2 — Intervento situazionale SELETTIVO e MISURABILE (gate + audit del "quando")
**Idea.** Trasformare la spinta da *nudge cieco su ogni richiesta* a **intervento deliberato e gated**: la situazione agisce solo quando **confidente** e **non-ridondante**, e si **misura** tasso ed esito.

**Cosa già abbiamo (forma soft).** Il rough-clustering già attenua nell'incertezza:
- **core** (assegnazione confidente) → `r_{k*}=1`, `γ_S=1` → spinta piena.
- **boundary** (incerto, tra `|T|` situazioni) → `r_k=comp/|T|`, **`γ_S=1/|T|`** → spinta **ridotta e mediata** (bias opposti si cancellano parzialmente).
→ Già oggi *in stato di incertezza la situazione agisce meno* (in `membership_from_assign` + `gamma=1/comp.sum`). MA è implicito, non misurato, e ignora non-ridondanza e confidenza temporale.

**La proposta = gate esplicito a 3 segnali di confidenza** (tutti già calcolabili):
1. **Confidenza clustering** (l'abbiamo): core vs boundary / margine membership.
2. **Non-ridondanza vs utente** (nuovo, "1−R²" operativo): `‖b̃^(z) − b_utente‖`. Coincidono → astieniti; divergono → agisci.
3. **Confidenza temporale = proiezione L3** (vedi F3): persistenza alta → agisci; stato-transito → attenua.

`κ_eff(richiesta) = κ · g(confidenza_cluster, non-ridondanza, persistenza_L3)` — soft (scala κ) o hard (act/abstain, soglia su val).

**Le metriche NUOVE (il valore).**
- **Tasso di intervento** (coverage): % richieste dove la situazione muove davvero il ranking.
- **Qualità intervento** (precision): sulle intervenute, % che migliora Cat-MRR vs % che danneggia.
- **Effetto netto selettivo**: Cat-MRR intervenute vs astenute.
- → claim: *"X-SAGE interviene sul Y% delle richieste; quando lo fa migliora nel W% (danno nel V%), e si astiene dove la situazione è ridondante con l'abitudine."*

**Perché premia il meccanismo.** Situazione = attore *deliberato*, non spinta indiscriminata. **Spiega i casi ridondanti** (POI): il gate si **astiene** lì e lo **dimostriamo** → da "perdiamo vs Steck-b" a "*il modello capisce che lì la situazione non serve e si fa da parte*". Interpretabile/auditabile (si sposa con la lente).

**Fattibilità.** ALTA. Tasso/qualità d'intervento = dal scoring esistente (nudge, ranking, Cat-MRR per-richiesta) → **misura su B_blind gratis, dati salvati**. Il gate = piccolo cambio al combiner. B_full = solo ri-run battery.

**Open points.** Definizione di "intervento" (cambio top-1 / magnitudo nudge / shift rango); soft vs hard; tarare soglia su val (anti-circolare); rischio narrativo se il tasso è basso ovunque tranne ml-1m ("interviene poco ma bene").

---

## F3 — Resuscitare L3 (proiezione) come SEGNALE DI CONFIDENZA
**Idea.** L3 (matrice di transizione situazioni, persistenza) era **chiusa come ottimizzatore** (≈ persistenza, non batte il re-ranking). Ma come **confidenza temporale per il gate (F2)** è naturale:
- persistenza alta (es. *Notte d'azione* 82%) = situazione stabile → **agisci con fiducia**.
- stato-transito (es. *Mondo fantasy* 25%) = stai per cambiare → **incerto → attenua**.

**Perché.** Chiude il loop di **Endsley**: la *proiezione* informa *quanto fidarsi* della *comprensione*. È il riutilizzo onesto di L3 che mancava (da "descrittiva inerte" a "segnale operativo per il gate").

**Fattibilità.** ALTA. La persistenza è già calcolata (`situation_transitions.py`); va solo agganciata al `κ_eff` di F2.

---

## F4 — Misurare la NON-RIDONDANZA (1−R²) direttamente
**Idea.** Quantificare in tabella il "1−R²": regressione del segnale-SIT (`b̃^(z)`) sul segnale-Steck-b (`b_utente`) → R²; **1−R² deve predire dove SIT batte Steck-b** (alto per ml-1m, basso per i POI).
**Perché.** Trasforma la "legge 1−R²" da frase a **numero misurato** → un revisore lo accetta. Spiega *perché* ml-1m vince e i POI no.
**Fattibilità.** ALTA, gratis (dati salvati, solo calcolo). Da fare a dataset tutti pronti.

---

## F5 — JS-verso-SITUAZIONE (la metrica-calibrazione mancante)
**Idea.** Abbiamo la JS-verso-utente (dove SIT "deve" perdere by-design). Manca la gemella **JS-verso-situazione** (`JS(p(g|situazione) ‖ q_lista)`), dove SIT **dovrebbe vincere** (è ciò che ottimizza).
**Perché.** Senza, un revisore dice "metrica di calibrazione scelta dove SIT perde". Dà l'**asse di vittoria** trasversale.
**Fattibilità.** ALTA, ~30 min. (= ex-O1.)

---

## F6 — Altri / carry-over
- **Fase A griglie larghe** (γ→0.8, n→10, H→5): i parametri ml-1m erano al bordo → il +0.008 è conservativo (ex-O2).
- **Anti-modale situazionale** (fairness-esposizione mirata ai pozzi): future work, ma servirebbe prima **validare la premessa** (servito-vs-domanda nei pozzi: è sovra-concentrazione o domanda?). Su metriche globali l'uniforme domina per costruzione (ex-O10).
- **CPFair reale** vs proxy UNI_glob (ex-O7, bassa).
- **Esposizione Singh–Joachims + permutazione lente** (ex-O5, se il referee lo chiede).

---

## Priorità (per quando si riapre il cantiere)
| id | idea | valore | costo | gratis su B_blind? |
|---|---|---|---|---|
| **F4** | misurare 1−R² | alto (chiude la legge) | basso | ✅ |
| **F1** | ibrido Steck-b+SIT | alto (≥ baseline ovunque) | medio (B_full=ri-run battery) | ✅ misura |
| **F2+F3** | gate selettivo + L3 confidenza + tasso/qualità intervento | alto (premia la situazione, audit) | medio | ✅ misura |
| **F5** | JS-situazione | medio (asse di vittoria) | basso | ✅ |
| F6 | griglie larghe, anti-modale, CPFair | vario | vario | — |

**Nota trasversale:** F1/F2/F4 si **testano quasi gratis su B_blind** (riuso dati salvati, niente re-train); la validazione su B_full richiede solo il ri-run dello **stadio battery** (re-train B_full), non l'intera pipeline.
