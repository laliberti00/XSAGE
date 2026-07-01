"""[venv-xsage] Genera un HTML singolo e navigabile sulle dimensioni di TRUSTWORTHINESS di X-SAGE:
proprietà per proprietà (allineate alla survey Ge et al. 2024), per ognuna cosa significa, cosa
facciamo, COME nel codice (file·funzione reale) e l'evidenza/numeri. Per i tutor.
Uso:  python scripts/yelp/build_trust_html.py
Out:  outputs_results/explain/demo_trust_ml1m.html
"""
import json, re
from pathlib import Path
import pandas as pd
CL = Path("/Users/lucaaliberti/Downloads/xsage-clean"); P = CL / "outputs_results" / "explain"
BK = ["B_blind", "B_full", "EASE", "DeepFM", "AFM", "FPMC", "SASRec"]; CITIES = ["nyc_tist", "saopaulo", "ml1m"]
c = {"LT": 0, "Cov": 0, "Gini": 0, "Cat": 0}
for ci in CITIES:
    b = pd.read_csv(CL / f"outputs_results/battery_bfull_{ci}.csv")
    for bk in BK:
        s = b[b.backbone == bk]; g = lambda m: s[s.method == "SIT"][m].mean() - s[s.method == "BASE"][m].mean()
        c["LT"] += g("LT20") > 0; c["Cov"] += g("Coverage") > 0; c["Gini"] += g("Gini") < 0; c["Cat"] += g("CatMRR") > 0
macro = sum((pd.read_csv(CL / f"outputs_results/macro_allbk_{ci}.csv").dMacro > 0).sum() for ci in CITIES)
cost = open(P / "cost_ml1m.txt").read()
NP = re.search(r"parametri del modulo: (\d+)", cost).group(1)
INF = re.search(r"per richiesta: ([\d.]+) ms", cost).group(1)
FIT = re.search(r"FIT totale \(pluggable\)\s+([\d.]+)", cost).group(1)

CARDS = [
 # group, verdict(ok/part/no), title, what, do, code, ev
 ("A", "ok", "Trasparenza & spiegabilità",
  "Si può capire <i>perché</i> il sistema raccomanda una certa cosa, senza fidarsi ciecamente.",
  "Le situazioni non sono fattori nascosti ma stati <b>nominati e leggibili</b> (es. «Commedia serale»). E il consiglio è spiegabile <b>per costruzione</b>: il punteggio finale è «punteggio del modello base + una spinta situazionale» — quella spinta <i>è</i> la spiegazione, esatta, non un'approssimazione costruita a posteriori. Una <b>lente</b> rende anche ispezionabili le situazioni che concentrano troppo l'esposizione.",
  ["xsage/recommendation.py · combiner additivo", "scripts/yelp/situation_profiles.py · situazioni nominate", "battery_bfull.py · lens_spread (lente)"],
  "5 situazioni nominate su ml-1m; spiegazione fedele al 100% (additiva); vedi il demo di explainability."),
 ("A", "ok", "Efficienza & sostenibilità (green)",
  "Un sistema affidabile non dovrebbe costare enormi risorse di calcolo/energia.",
  "Il modulo situazionale è un <b>re-ranker leggero</b> calcolato in forma chiusa, <b>senza addestramento a gradiente</b> e senza ri-allenare il modello sotto. Si aggancia a qualsiasi backbone.",
  ["scripts/yelp/profile_cost.py · misura di costo"],
  f"<b>{NP} parametri</b> totali · inferenza <b>{INF} ms / richiesta</b> · fit completo <b>~{float(FIT):.0f} s</b> — ordini di grandezza più piccolo di un backbone neurale."),
 ("A", "ok", "Fondamenta causali (niente leakage)",
  "Valutare in modo onesto: il sistema non deve «sbirciare nel futuro» dell'utente.",
  "Lo split è <b>temporale per-utente 80/10/10</b> (prima nel tempo = train, dopo = test) e il proxy d'intento legge solo l'interazione <b>precedente</b> (shift causale). Tutti i parametri sono scelti su validation, il test misurato una volta sola (anti-circolarità).",
  ["scripts/*/preprocess_*.py · split temporale per-utente", "intent_last = categoria interazione precedente (shift)"],
  "Garanzia strutturale: nessuna richiesta è descritta con informazioni dal proprio futuro."),

 ("B", "part", "Equità & non-discriminazione",
  "Non concentrare l'esposizione sempre sugli stessi item popolari; dare visibilità anche al resto.",
  "La spinta situazionale premia le categorie tipiche della situazione invece dei soli popolari globali del backbone → <b>ridistribuisce l'esposizione</b>. Inoltre la <b>lente</b> rende l'iniquità <i>misurabile</i> per situazione.",
  ["xsage/recommendation.py · combiner (bias per-situazione)", "scripts/yelp/macro_avg.py · equità macro"],
  f"Rispetto al backbone: Gini↓ <b>{c['Gini']}/21</b>, Coverage↑ <b>{c['Cov']}/21</b>, coda-lunga↑ <b>{c['LT']}/21</b>, equità-categoria↑ <b>{macro}/21</b>. <span style='color:#b45309'>Limite onesto:</span> è equità di <b>esposizione + diagnostica</b>, non un correttore; su saturazione estrema (un dataset, yelp) amplifica la maggioranza."),
 ("B", "part", "Robustezza & sicurezza",
  "Il metodo deve reggere su modelli e condizioni diverse, non solo su un caso fortunato.",
  "È un <b>re-ranker pluggable</b>: lo abbiamo montato su <b>7 backbone</b> di famiglie diverse (statici, context-aware, sequenziali) con lo stesso protocollo. Validazione con 5 seed, bootstrap e test di equivalenza; l'identità κ=0 <b>recupera esattamente il backbone come fallback sicuro</b> (con il κ scelto su validation una singola metrica può comunque regredire su test).",
  ["scripts/ml1m/battery_bfull.py · 7 backbone, 5 seed, bootstrap, TOST"],
  "Coerente su 7 backbone × 3 dataset. <span style='color:#b45309'>Limite:</span> nessun test adversarial/poisoning/distribution-shift."),
 ("B", "part", "Privacy",
  "Raccomandare bene chiedendo meno dati personali persistenti possibile.",
  "La situazione si costruisce solo dalla <b>continuità intra-sessione</b> (la finestra causale recente), <b>non</b> da un'identità persistente tracciata fra sessioni → un vantaggio «cookieless» di design: il sistema reagisce al <i>momento</i>, non a un profilo permanente.",
  ["xsage/l0_sensing.py · build_recent_window (finestra causale, locale)"],
  "<span style='color:#b45309'>Limite onesto:</span> è un argomento di design, non una garanzia formale — non implementiamo privacy differenziale (DP)."),
 ("B", "part", "Controllabilità & accountability",
  "Chi gestisce il sistema deve poter ispezionare e regolare il suo comportamento.",
  "L'intensità dell'intervento è un <b>singolo parametro leggibile</b> (κ): separa la decisione di <i>quanto</i> agire, ed è impostabile/auditabile. La lente per-situazione fornisce una traccia ispezionabile di dove e come la situazione agisce.",
  ["xsage/recommendation.py · κ (coefficiente esplicito)", "battery_bfull.py · lens_spread (audit)"],
  "<span style='color:#b45309'>Limite:</span> nessun audit formale/processo di accountability codificato."),

 ("C", "no", "Federated / decentralizzazione",
  "Addestrare/servire senza centralizzare i dati degli utenti.",
  "Non implementato in questo lavoro — direzione futura dichiarata.", [],
  "→ future work."),
 ("C", "no", "Privacy formale (Differential Privacy)",
  "Garanzie matematiche di privacy (es. rumore DP) sui dati/parametri.",
  "Non implementata: la privacy qui è un vantaggio di design (cookieless), non una garanzia formale.", [],
  "→ limite dichiarato."),
 ("C", "no", "Valutazione etica / impatto sociale",
  "Studio dell'impatto sociale, responsibility audit, effetti a lungo termine.",
  "Non condotto: la valutazione è tecnica (accuratezza + equità di esposizione + diagnostica), non uno studio di impatto sociale.", [],
  "→ limite dichiarato."),
 ("C", "no", "Robustezza adversarial",
  "Tenuta contro attacchi mirati (poisoning, manipolazione del ranking).",
  "Non testata in questo lavoro.", [],
  "→ future work."),
]

VB = {"ok": ("implementato", "#dcfce7", "#15803d"), "part": ("parziale", "#fef9c3", "#a16207"), "no": ("non coperto", "#fee2e2", "#b91c1c")}
GR = {"A": ("Punti di forza", "Proprietà pienamente implementate, con evidenza nel codice e nei numeri."),
      "B": ("Coperture parziali", "Proprietà affrontate con un meccanismo reale, ma con un limite dichiarato onestamente."),
      "C": ("Limiti dichiarati", "Proprietà che la letteratura elenca ma che NON copriamo — diventano future work. Dichiararle è parte della trasparenza.")}
nok = sum(1 for x in CARDS if x[1] == "ok"); npart = sum(1 for x in CARDS if x[1] == "part"); nno = sum(1 for x in CARDS if x[1] == "no")


def card(x):
    _, v, t, what, do, code, ev = x; lab, bg, fg = VB[v]
    cc = "".join(f"<code>{x}</code>" for x in code)
    return f"""<div class="tcard"><div class="th"><span class="vb" style="background:{bg};color:{fg}">{lab}</span><b>{t}</b></div>
<div class="row"><span class="k">Cosa significa</span><div>{what}</div></div>
<div class="row"><span class="k">Cosa facciamo</span><div>{do}</div></div>
{('<div class="row"><span class="k">Nel codice</span><div class="codes">'+cc+'</div></div>') if code else ''}
<div class="row"><span class="k">Evidenza</span><div>{ev}</div></div></div>"""


secs = ""
for g in ["A", "B", "C"]:
    gt, gd = GR[g]
    secs += f'<h2>{gt}</h2><p class="lead">{gd}</p>' + "".join(card(x) for x in CARDS if x[0] == g)

HTML = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X-SAGE — Trustworthiness</title><style>
:root{{--bg:#f7f8fa;--card:#fff;--ink:#1a1d24;--mut:#6b7280;--line:#e5e7eb;--accent:#2563eb}}
*{{box-sizing:border-box}}body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}}
.wrap{{max-width:980px;margin:0 auto;padding:28px 22px 60px}}
h1{{font-size:26px;font-weight:600;margin:0 0 4px}}.sub{{color:var(--mut);font-size:15px;margin:0 0 14px}}
.intro{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;font-size:15px;margin-bottom:16px}}.intro b{{color:var(--accent)}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 18px}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:130px}}.stat b{{font-size:20px;font-weight:600;display:block}}.stat span{{color:var(--mut);font-size:12px}}
h2{{font-size:20px;font-weight:600;margin:26px 0 4px}}.lead{{color:var(--mut);font-size:14px;margin:0 0 14px;max-width:780px}}
.tcard{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}}
.th{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}.th b{{font-size:16px}}
.vb{{font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em}}
.row{{display:grid;grid-template-columns:130px 1fr;gap:12px;padding:7px 0;border-top:1px solid #f1f3f5;font-size:14px}}
.row .k{{color:var(--mut);font-size:12.5px;font-weight:600;padding-top:2px}}
.codes{{display:flex;flex-wrap:wrap;gap:6px}}code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;padding:2px 8px;color:#334155}}
@media(max-width:560px){{.row{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>X-SAGE — Affidabilità (trustworthiness), proprietà per proprietà</h1>
<div class="sub">MovieLens-1M e oltre · cosa significa, cosa facciamo, come nel codice</div>
<div class="intro"><b>L'idea.</b> L'affidabilità non è un'aggiunta finale: in X-SAGE è <b>distribuita lungo tutta la pipeline</b> (sensing → percezione → comprensione → proiezione → raccomandazione). Qui leggiamo il sistema <b>per proprietà</b> — secondo le dimensioni di affidabilità della letteratura (Ge et al. 2024) — con un verdetto onesto a tre livelli: <span class="vb" style="background:#dcfce7;color:#15803d">implementato</span> <span class="vb" style="background:#fef9c3;color:#a16207">parziale</span> <span class="vb" style="background:#fee2e2;color:#b91c1c">non coperto</span>. I limiti sono dichiarati apertamente: anche questo è trasparenza.</div>
<div class="stats">
<div class="stat"><b style="color:#15803d">{nok}</b><span>proprietà implementate</span></div>
<div class="stat"><b style="color:#a16207">{npart}</b><span>coperture parziali</span></div>
<div class="stat"><b style="color:#b91c1c">{nno}</b><span>limiti dichiarati</span></div>
</div>
{secs}
</div></body></html>"""
out = P / "demo_trust_ml1m.html"; out.write_text(HTML, encoding="utf-8")
print(f"-> {out}  ({len(HTML)} char)")
