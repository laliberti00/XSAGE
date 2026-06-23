# X-SAGE — Stato del lavoro (documento vivo)

> Documento canonico. Aggiornato e committato a ogni modifica sostanziale.
> Ultimo aggiornamento: 2026-06-23 (consolidamento: optimizer morto, lente/audit viva).

## Changelog
- Selezione parametri situazioni (K/ε/depth/n) anti-circolare; γ_S=1/|T|.
- Filo categoriale: SIT>BASE e >UNI_mean signif. 5/5, plateau-K 25/25.
- Ablazione intento: costitutivo 4/5, dannoso São Paulo = **proprietà** (legge beneficio∝1−R²).
- L3 projection **CHIUSA**: eq.18 batte il geometrico 5/5 ma NON la persistenza (4/5) → nota descrittiva.
- **Baseline Steck**: SIT perde su Cat-MRR vs Steck-b (4/5) e Steck-a (5/5), Holm.
- **B3**: SIT perde vs B_full (context-aware, stesse feature) 5/5, Holm.
- **B1**: pluggabilità OK (SIT porta su BPR, +0.031).
- **Fairness**: la LENTE (diagnosi per-situazione) regge; l'INTERVENTO no — B4 lato-utente e
  selezione situata **dominati dall'uniforme/casuale/naive**; early-warning proiezione NON batte persistenza.
- **Verdetto framing**: X-SAGE non è un ottimizzatore competitivo; è uno **strumento situation-aware
  interpretabile/auditabile**. Le situazioni DIAGNOSTICANO, non sono una leva operativa.
- **Confronto SARE**: spiega il gap (appreso/personalizzato/latente vs non-sup/fisso/grezzo) → §8.
- **2° dataset** (ricognizione): Last.fm-1K (port pulito, nodo=genere esterno) vs MIND-large
  (categoria nativa + protocollo SARE). Vedi [STORIA](STORIA_PROGETTO_XSAGE.md).

---

## 0. Cos'è X-SAGE, oggi (framing onesto)
Pipeline **situation-aware NON supervisionata** (Endsley L0→L3) su POI, che (a) costruisce situazioni
esplicite contesto+intento, (b) le usa come **lente** di iniquità per-situazione, (c) applica un
re-ranker additivo leggero. **NON è un recommender competitivo**: il suo valore validato è la
**leggibilità/auditabilità** (situazioni nominate + mappa di iniquità per-situazione), non l'accuratezza.

## 1. Dati
TIST2015 (Foursquare), 5 città, k-core=10, split temporale per-utente 80/10/10 (causale).
Backbone FM importato con checksum (gitignored). Richiesta = (utente, istante), next-item nascosto.

## 2. Flusso dati — fase per fase, formula per formula
| fase | entra | operazione | esce |
|---|---|---|---|
| **L0 Sensing** | log, (u,t) | ultime `n=3` macro con `time<t` (causale) | sequenza macro recente |
| **L1a Contesto c̃** | 6 attributi `c_hour,c_dow,c_isweekend,c_month,prev_geohash5,intent_last` | albero/attributo → `θ_a = 1−H(p_foglia)/log₂(K_mac)` | `c̃=(θ_a)_{1..6}` |
| **L1b Intento e** | sequenza macro recente | `m_c∝Σ_j γ^j 1[macro_j=c]`; `W=P̂(c'\|c)` (+1,row-norm); `A={indeg≥media}`; `e∝(m·Σ_{k=1..H}β^k W^k)⊙1[∈A]` | `e` su attrattori |
| **L2 Comprehension** | `v=[c̃‖e]` | rough k-means (train→test): core `r_{k*}=1`, boundary `r_k=1/\|T\|` | membership `r`, situazione `z` |
| **Bias b̃^(k)** | `z_train`+macro veri train | log-odds smussato vs globale, z-scored (α=50) | `b̃^(k)∈ℝ^{n_macro}` |
| **L3 Projection** | sequenze `z` | `T=P̂(s_{t+1}\|s_t)`; eq.18 `r̃_k∝r_k·T_{z_prev,k}` | **fuori dal path** (chiuso) |
| **Combiner** | `s_B`,`r`,`b̃`,`γ_S` | `ŝ=s_B+κ·m_sel·γ_S·Σ_k r_k(b̃^(k)_{c(i)}+λ·b^LT_{c(i)})` | `ŝ` |
| **Ranking** | `ŝ` | maschera item visti, top-20 | liste |

**Config CORE (SIT)**: `m_sel≡1, λ=0, κ=0.25, γ_S=1/|T|` → `ŝ = s_B + 0.25·γ_S·Σ_k r_k·b̃^(k)_{c(i)}`.

## 3. Parametri finali e selezione (anti-circolare)
- Perception: γ=0.4, β=0.7, H=2, n=3, depth=3.
- **K**: ist5,bkk6,nyc6,sao3,tky4 — criteri interni (silhouette/CH/DB) + tetto `|A|+2`.
  ⚠️ *Caveat: bkk=tetto (interno voleva 8); ist=fallback-ARI (silhouette piatta).*
- **ε**: .01/.01/.02/.05/.07 — banda boundary [10%,30%]. ⚠️ *bkk ε=0.01 fuori-banda (8.7%).*
- **κ=0.25, α=50**: fissati a priori — manca curva di robustezza (debito aperto).

## 4. Cosa è VALIDATO, con quale metrica ed esito
Metriche: Cat-MRR/Cat-NDCG (rango categoria vera), R@20/NDCG@20 (item), JS-calibration, LT@20/Gini.

| claim | metrica | esito | significatività |
|---|---|---|---|
| SIT > BASE (backbone cieco) | Cat-MRR | ✅ 5/5 (+0.016…+0.036) | bootstrap 2000 |
| SIT > UNI_mean (specificità) | Cat-MRR | ✅ 5/5 | bootstrap 2000 |
| robustezza cutoff K | Cat-MRR | ✅ 25/25 (K∈{5..100}) | bootstrap |
| intento costitutivo | Cat-MRR FULL vs CTX | ⚠️ 4/5 (sao dannoso) | bootstrap 2000 |
| São Paulo = proprietà | Δ @K-ottimale | ✅ | bootstrap |
| legge beneficio∝(1−R²) | R²(e\|c̃) | ✅ monotòna | — |
| pluggabilità BPR | Cat-MRR SIT−BASE | ✅ +0.031 | bootstrap 1500 |
| **SIT vs Steck-b** (target utente) | Cat-MRR | ❌ perde 4/5 | bootstrap 1500 + **Holm** |
| **SIT vs Steck-a** (greedy reale) | Cat-MRR | ❌ perde 5/5 | bootstrap + Holm |
| **SIT vs B_full** (context-aware) | Cat-MRR | ❌ perde 5/5 (B_full 2–3×) | bootstrap + Holm |
| **lente per-situazione** | KL/LT/Gini/situazione | ✅ rivela pozzi (KL ~75×), robusta 8 backbone | — |
| intervento fairness item | LT/Gini servito | ❌ dominato dall'uniforme | bootstrap |
| B4 equità lato-utente | ΔGini utenti | ⚠️ funziona 5/5 **ma dominato dall'uniforme 3/5** | bootstrap 1500 |
| L3 disambigua boundary | F1 vs geom/persist | ✅ vs geom 5/5; ❌ vs persistenza 1/5 | bootstrap 1500 |
| proiezione early-warning | recall-MOVE/AP | ❌ non batte persistenza | bootstrap 1500 |
| policy selettiva situata | risk-coverage | ❌ dominata da casuale+naive (0/5) | bootstrap |

## 5. Test di significatività statistica fatti
- **Bootstrap percentile CI** (1500–2000 ricampioni) su ogni Δ, per-richiesta o **per-utente** (Gini/equità).
- **Correzione Holm** per confronti multipli (calibrazione, B3).
- **Soglie pre-registrate** (≥4/5, decise prima dei numeri) per intento, persistence-aware, B4, risk.
- Anti-circolarità: criteri/target da train/val, test mai in selezione; clustering/T/sink dal train.
- *(Repo OLD: anche Wilcoxon/Pratt, TOST, McNemar, permutazione — non rieseguiti nel clean.)*

## 6. Inventario onesto
**✅ Regge (misurato):**
- La **lente situazionale** (diagnosi iniquità per-situazione), robusta a 8 backbone. *Il più forte.*
- Situazioni come costrutto: SIT>BASE/>UNI_mean 5/5; archetipi nominati; intento costitutivo 4/5 + legge 1−R².
- Pluggabilità della lente (model-agnostic).

**❌ Morto (sotto controllo):**
- SIT come ottimizzatore: perde vs Steck-b/Steck-a/B_full.
- Ogni azione situata (re-ranking, fairness item/utente, early-warning, selezione) eguagliata/battuta
  da baseline triviale (uniforme/persistenza/casuale/naive).
- L3 projection (chiusa, descrittiva).

**❓ Aperto:**
- Robustezza α/κ; secondo dataset; "so what" dell'audit (l'intervento sui pozzi non batte il random);
  caveat K/ε di bangkok/istanbul; documentazione (docs 03–07, README/02 stale).

## 7. Sintesi per convergere
Il filo che tiene tutto: **le situazioni DIAGNOSTICANO ma non sono una leva operativa.** L'unico
vantaggio che sopravvive a ogni controllo è la **leggibilità**: situazioni esplicite, nominate, +
mappa di iniquità per-situazione che un context-aware (B_full) o un latente (SARE) non producono,
pur raccomandando meglio. *Quel baratto accuratezza→trasparenza è il contributo.*

**Tre cose pubblicabili (nessuna è "un recommender migliore"):**
1. **La lente** — audit di fairness per-situazione, model-agnostic. Il pezzo più forte. *Diagnostica, non raccomanda.*
2. **La legge 1−R²** — quando la situazione esplicita porta segnale vs è ridondante. Sezione-chiave.
3. **L'operazionalizzazione** — pipeline Endsley L0→L3 non supervisionata, selezione anti-circolare (stile SA-WCS).

**NON pubblicabile**: "un situation-aware recommender competitivo" — i dati non lo reggono.

**Bivio**: paper di **trasparenza/audit** (coerente, difendibile oggi) vs continuare a cercare un
vantaggio da ottimizzatore (i dati attuali dicono di no → servirebbe un dominio nuovo, scommessa).

## 8. Confronto con SARE (Li et al. 2025) — perché loro vincono e noi no
SARE e X-SAGE condividono il nome ma sono macchine opposte: SARE = rete di **conditioning
appresa end-to-end, personalizzata** (UCPE per-utente sul full-item, PSF percezione per-utente,
combiner con confidence, loss rec+situ); X-SAGE = **clustering non supervisionato + nudge additivo
fisso su ~10 macro, globale**. SARE vince perché ha le 3 cose che X-SAGE ha rinunciato *per scelta*
— e che i nostri stessi test misurano come gap: **personalizzazione** (= Steck-b batte SIT 4/5),
**capacità appresa full-item** (= B_full batte SIT 5/5), **supervisione+confidence**. SARE valuta
inoltre su **impression-ranking** (più facile del nostro top-N). È un **baratto, non un paradosso**:
la performance di SARE viene dal macchinario che la rende NON ispezionabile; l'ispezionabilità di
X-SAGE viene dal macchinario che costa la performance. → Per "performance + situazione spiegabile"
serve un **modello nuovo**: conditioning appreso/personalizzato à-la-SARE ma **condizionato sulla
situazione ESPLICITA** (collo di bottiglia interpretabile), non un nudge fuori da un backbone congelato.

## 9. Prossimi passi
- **2° dataset** (ricognizione fatta): **Last.fm-1K** = port pulito del pipeline attuale (1 variabile:
  genere da MusicBrainz); **MIND-large** = categoria nativa (18 macro) + protocollo impression-ranking
  nativo per il confronto vs SARE (ma utente non-persistente). Vedi [STORIA timeline](STORIA_PROGETTO_XSAGE.md).
- **Modello nuovo**: SARE-con-situazione-esplicita (da progettare).

### Porting MIND-large (in corso — scelta utente: k-core=10, setup attuale)
- ✅ **step-01 dati** (`scripts/mind/preprocess_mind.py`): click→k-core10→split per-utente 80/10/10.
  MIND = "città": **101.131 utenti, 5.278 item, 1.76M interazioni, 15 macro native**, contesto
  temporale (**no geohash**), intent_last_cat. `load_city('mind', data_root='.')` OK.
- ⏳ **prossimo blocco = BACKBONE**: il pipeline ha bisogno di `FM.scores.npy`/`Bfull.scores.npy`;
  per MIND vanno generati (il clean repo importa scores, non li allena). Via pragmatica: **BPR via
  venv-cornac** (riuso `scripts/cornac/`) → matrice [n_users×n_items] come B_blind.
- ⏳ poi: situazioni (build_v con attributi MIND **senza geohash**), poi SIT/lente/metriche.
- ⚠️ caveat dichiarati: utenti poco profondi (k-core scarta 86%), span ~1 settimana, niente geo.
