# X-SAGE — Stato del lavoro (documento vivo)

> Documento canonico. Aggiornato e committato a ogni modifica sostanziale.
> Ultimo aggiornamento: 2026-06-24 (CONSULTO asse-portante: stato neutro/usi-molti + macro-averaged = equità-categoria; ablazione-neutralità in prep — vedi §13, O9).

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
- ✅ **step-02 backbone** (`scripts/mind/cornac_backbone.py`): BPR via venv-cornac →
  `data/mind/backbone/FM.scores.npy` [101131×5278] (B_blind context-blind, sostituisce l'FM importato;
  Bfull=stub non context-aware). 0 utenti cold.
- ✅ **step-03 prep+smoke** (`scripts/mind/mind_prep.py`): `build_mind_prep` replica il prep con dati
  clean + attributi MIND (no geohash) + transit off; fix schema (`cat_target`, `user_id` alias).
  **X-SAGE gira end-to-end su MIND**: BASE Cat-MRR=0.352, SIT=0.354, Δ=+0.002 (K/ε placeholder, no test).
- ✅ **eval MIND** (placeholder K/ε): SIT−BASE Cat-MRR +0.002 (sig, minuscolo); fairness trascurabile;
  **lente rivela disparità** (KL 0.03–2.03× tra situazioni).
- ⚠️ caveat: utenti poco profondi (mediana 3), span 1 settimana, no geo, backbone=BPR.

### Porting MovieLens-1M (city-param, riuso macchina MIND)
- ✅ **dati** (`scripts/ml1m/preprocess_ml1m.py`): rating=interazione, genere primario=macro, k-core10.
  **6040 utenti PROFONDI (~165 rating/utente), 3260 film, 18 generi**, contesto temporale (no geo).
- ✅ **backbone+eval** (riuso `cornac_backbone.py`/`mind_eval.py` city-parametrizzati):
  **SIT−BASE Cat-MRR +0.020 (sig) + R@20 +0.005 + fairness↑** (LT 0.238→0.245, Gini 0.745→0.740);
  lente sit3 = sink (16K req). **In range Foursquare.**
- 🔑 **Lettura cross-dataset**: il segnale situazionale **scala con la profondità comportamentale**
  (ml-1m profondo +0.020 ≫ MIND shallow +0.002). ml-1m = dominio più promettente per l'ipotesi situazionale.
- ⚡ **ml-1m baseline reali** (`eval_baselines.py`, K=3/ε=.07 selezionati, Holm): **SIT BATTE Steck-b
  (+0.024)**, UNI_mean (+0.016), BASE (+0.015); perde solo vs Steck-a (greedy metric-gaming, ma SIT
  miglior R@20). **Ribalta Foursquare** (lì Steck-b batteva SIT 4/5) → *il valore situazionale scala
  con la profondità comportamentale: su domini profondi la situazione batte la personalizzazione statica.*
- 🔑 **DUE angoli vendibili ora**: (1) lente/audit + legge profondità(1−R²); (2) **dominio (ml-1m) dove
  SIT è ottimizzatore competitivo** (batte Steck-b) + interpretabile. Gate finale: **B_full su ml-1m**.
- ⚡⚡ **ml-1m κ IMPECCABILE** (`eval_kappa.py`, κ selezionato su VAL per-metodo, anti-circolare):
  SIT batte Steck-b **a 8/8 κ** + al κ\*=0.5; SIT−Steck-b=+0.019 [+0.017,+0.021]. κ ereditato 0.25
  era subottimale (κ\*ml1m=0.5, dataset-dipendente). Steck-b ha curva monotòna in discesa (la sua
  spinta danneggia). → **la vittoria ml-1m è ROBUSTA, non un artefatto del κ ereditato**.

### ml-1m — STATO CHIUSO E VALIDATO (2026-06-24)
**Fase A** (`scripts/ml1m/close_params.py`, val Cat-MRR + plateau): parametri chiusi
`γ=0.6, depth=2, n=5, β=0.7, H=3, α=10`. γ/n/H **cambiati** dagli ereditati (subottimali);
β/α confermati piatti. ⚠️ *finding: γ/n/H al bordo-griglia → griglie da allargare.*

**Fase B** (`scripts/ml1m/battery_bfull.py`, 5 seed, parametri chiusi, K/ε/κ su val):
tabellone {B_blind(BPR), B_full(context-aware)} × {BASE,SIT,Steck-b,Steck-a,UNI_mean,UNI_glob}
× accuratezza+fairness, bootstrap+Holm+TOST. SD≤0.002.
- **B_blind**: SIT Cat-MRR **0.385** > BASE 0.360, Steck-b 0.358, UNI_glob 0.359; SIT R@20 0.113.
- **B_full**: B_full 0.430 → **SIT-su-B_full 0.438**.
- **HEADLINE (SIT-su-B_full vs B_full)**: ΔCat-MRR **+0.008** p=0 · ΔR@20 **+0.002** (TOST: non degrada)
  · ΔLT **+0.005** · ΔGini **−0.004** · ΔCoverage **+0.005** · **JS-user +0.036 (peggiora, by-design:
  SIT calibra sulla situazione, non sull'utente — ed è ciò che lo fa vincere)**.
- **VERDETTO ml-1m**: *X-SAGE = **enhancer interpretabile** che migliora un backbone context-aware forte
  su **accuratezza + fairness-esposizione**, al costo dichiarato della calibrazione-utente. Batte Steck-b
  e UNI_glob; perde solo vs Steck-a (greedy che però azzera la coda lunga).* Distinzione col gate:
  SIT-su-backbone-debole < B_full (gate FAIL); SIT-**montato su** B_full *lo migliora* → X-SAGE è un
  **enhancer**, non un sostituto.
- Artefatti: `params/ml1m.json`, `param_closure_ml1m.csv`, `battery_bfull_ml1m.csv`, `gate_bfull_ml1m.csv`.

### Yelp + la SCOPERTA macro-averaged → legge a DUE GATE (2026-06-24)
**Port Yelp** (`scripts/yelp/preprocess_yelp.py`): metro=Philadelphia, k-core20, macro=categoria-radice
(17), **GEO riabilitato** (`prev_geohash5` = geohash6 della review precedente, anti-leakage come
Foursquare; `city_attrs("yelp")` in mind_prep). **4842 utenti PROFONDI (~52 review/utente)**.
- **Batteria** (`battery_bfull.py yelp 5`): **null** — SIT-su-B_full ΔCat-MRR **−0.0011** (p=0, minuscolo);
  su B_blind +0.0035 (dentro SD). Fairness Δ trascurabili. Profondità c'era, eppure niente.
- **PERCHÉ** (`scripts/yelp/macro_avg.py`, Cat-MRR **micro vs MACRO-averaged** su B_blind):
  Yelp è **saturo all'83.7%** (Restaurants). Il "+micro" di SIT è un **artefatto**: SIT guadagna SOLO
  sulla dominante (+0.016) e **peggiora le 16 minoritarie** → **macro-Δ = −0.010, vince 1/17 categorie**.
  *Su dati saturi SIT degenera in AMPLIFICATORE DELLA CLASSE MAGGIORITARIA*; la micro lo nasconde, la
  macro-averaging lo smaschera. ml-1m (dominante 27.7%): macro-Δ **+0.014, 16/18 cat** → vittoria *vera e
  distribuita*. MIND (24.9%, ma shallow): micro −0.001 / macro +0.003, κ* val=0.05 (SIT si auto-spegne) →
  **neutro** (Steck-b vince).
- 🔑🔑 **LEGGE A DUE GATE**: SIT aggiunge valore sse passa *(1) profondità* **E** *(2) non-saturazione*
  del target. Ogni dataset ne fallisce uno diverso: **MIND→profondità**, **Yelp→saturazione**,
  **ml-1m→nessuno** (unico win). Foursquare-tokyo (62.5%) predice lo stesso di Yelp.
  → La **macro-averaged è la metrica diagnostica** che separa "situazionale vero" da "amplificatore".
  Artefatti: `battery_bfull_yelp.csv`, `params/yelp.json`, `macro_avg_summary.csv`.
- ⏳ macro-averaged COMPLETA di SIT su Foursquare = non-gratis (città TIST su repo OLD = O3); la
  **saturazione** TIST è in `macro_avg_summary.csv` (istanbul 23%…tokyo 62%).

## 12. OPEN POINTS (cosa resta)
| # | open point | priorità | nota |
|---|---|---|---|
| O1 | **JS-verso-SITUAZIONE** (gemella di JS-user) | alta | senza, un revisore dice "metrica di calibrazione scelta dove SIT perde". SIT *dovrebbe* vincerla. ~30min |
| O2 | **Fase A griglie larghe** (γ→0.8, n→10, H→5) | alta | γ/n/H al bordo → l'ottimo vero è oltre; il +0.008 è conservativo |
| O3 | **Foursquare con la batteria + macro-averaged** | media | adattare `build_v` OLD-coupled (+geohash); chiude il SIT-macro su TIST (saturazione già nota) |
| O3b | **Yelp macro-fini sul cibo** (riduci saturazione) | media | test falsificabile: se macro-Δ torna >0 era saturazione; giudicare su MACRO non micro |
| O4 | **β/H NON piatti su ml-1m** (H è una vera selezione, non fix) | media | dichiarare H come selezionato, non fix-by-design |
| O5 | esposizione **Singh–Joachims** + **permutazione lente** | bassa | solo OLD; se il paper li vuole |
| O6 | **MIND batteria** (shallow, per il 3° punto della legge) | bassa | controprova: SIT-su-B_full su shallow |
| O7 | **CPFair reale** vs proxy `UNI_glob` | bassa | deciso: proxy dichiarato, CPFair vero assente |
| O8 | docs 03–07, README/02 stale (vecchio framing) | media | da riscrivere col framing enhancer/legge-profondità |
| O9 | **Ablazione di neutralità** (in preparazione) | **alta** | `neutrality_ablation.py`: raw_ctx/raw_int vs full su ml-1m → il valore è strutturale o cucito nelle feature? Blinda la fondazione |
| O10 | **Anti-modale situazionale** + validazione premessa | future work | servito-vs-domanda nei pozzi (misura cheap) PRIMA del modello; headwind strutturale su metriche globali (P2 consulto) |

## 13. CONSULTO asse-portante: fairness vs accuracy (2026-06-24) — DECISIONE
Domanda: l'asse del paper dev'essere fairness (intuizione utente) o accuracy? Esito del consulto:
- **La fairness-con-guadagno ESISTE GIÀ, ma non è l'esposizione-item**: la **Cat-MRR macro-averaged è una
  metrica di equità per-categoria** (qualità di servizio equa fra le categorie, come macro-F1). ml-1m:
  +0.014 su 16/18 cat = *equità di qualità, con numero che sale*. Yelp: −0.010 su 1/17 = *iniquità reale*
  (amplifica la maggioranza) diagnosticata dalla stessa metrica.
- **L'intervento anti-modale (fairness-esposizione) NON è la via**: (1) l'uniforme domina **per costruzione**
  su metriche globali (Gini/Cov), strutturale; (2) la premessa "pozzo = iniquità" è **non-validata** (la lente
  misura concentrazione del *servito*, non *servito−domandato*); → future work (O10).
- 🔑 **ASSE PORTANTE DECISO**: *uno **stato situazionale neutro** (L2, costruito una volta) → **usi multipli**:
  **diagnosi** (lente, vince sempre) + **azione** (re-ranking, vince nel regime a **due gate**); più la **legge**
  che dice quale uso paga e quando.* Il decoupling "stato neutro / usi molti" è una **virtù architetturale da
  dichiarare**, non un problema da risolvere.
- **Precisione tecnica**: lo stato è neutro nel *clustering* (k-means non-sup), ma le *feature* `[c̃‖e]` sono già
  category-aware a monte (c̃ = informatività vs prossima-macro; e = proiettata sul grafo-macro). Da qui O9:
  l'**ablazione di neutralità** verifica se il valore è strutturale (stato) o cucito nelle feature.
