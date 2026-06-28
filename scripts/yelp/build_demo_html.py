"""[venv-xsage] Genera un HTML singolo, autonomo e navigabile per mostrare la parte di
explainability di X-SAGE (ml1m) ai tutor: spazio situazioni (PCA), profili + lente di iniquità,
proiezione L3 (transizioni), esempi di raccomandazione reali. Tutti i dati embeddati inline dai
JSON in outputs_results/explain/ → il file si apre senza server né dipendenze.
Uso:  python scripts/yelp/build_demo_html.py
Out:  outputs_results/explain/demo_xsage_ml1m.html
"""
import json
from pathlib import Path
CL = Path("/Users/lucaaliberti/Downloads/xsage-clean"); P = CL / "outputs_results" / "explain"

prof = json.load(open(P / "situation_profiles_ml1m.json"))
space = json.load(open(P / "situation_space_ml1m.json"))
trans = json.load(open(P / "situation_transitions_ml1m.json"))
reco = json.load(open(P / "reco_examples_ml1m.json"))
names = json.load(open(P / "situation_names_ml1m.json"))

DATA = {"names": names, "K": prof["K"],
        "profiles": prof["situations"], "genres": prof["genres"],
        "space": {"var": space["var"], "bfrac": space["bfrac"], "points": space["points"], "centroids": space["centroids"]},
        "trans": {"T": trans["T"], "sits": trans["situations"]},
        "reco": reco}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X-SAGE — Explainability (MovieLens-1M)</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d24;--mut:#6b7280;--line:#e5e7eb;--accent:#2563eb}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}
.wrap{max-width:1060px;margin:0 auto;padding:28px 22px 60px}
header h1{font-size:26px;font-weight:600;margin:0 0 4px}header p{color:var(--mut);margin:0 0 18px;font-size:15px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0 22px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.stat b{font-size:20px;font-weight:600;display:block}.stat span{color:var(--mut);font-size:12px}
nav{display:flex;gap:6px;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:5;border-bottom:1px solid var(--line)}
nav button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;border-radius:8px;cursor:pointer}
nav button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
section{display:none;animation:f .25s}section.on{display:block}@keyframes f{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
h2{font-size:20px;font-weight:600;margin:24px 0 6px}.lead{color:var(--mut);font-size:14px;margin:0 0 16px;max-width:760px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.sit-h{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.dot{width:12px;height:12px;border-radius:50%;flex:none}.sit-h b{font-size:15px}.sit-h .sz{margin-left:auto;color:var(--mut);font-size:12px}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin:3px 0 6px}.bar i{display:block;height:100%}
.gen{display:flex;justify-content:space-between;font-size:13px;color:#374151}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.lens{margin-top:8px;font-size:12px;color:var(--mut)}
svg{max-width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0}.legend span{font-size:13px;display:flex;align-items:center;gap:6px}
.reco{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center}
.reco .pick{padding:10px 12px;border-radius:10px;border:1px solid var(--line)}
.reco .base{background:#f9fafb}.reco .sit{background:#eff6ff;border-color:#bfdbfe}
.reco .g{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.reco .t{font-size:15px;font-weight:500;margin-top:2px}
.arrow{text-align:center;color:var(--accent);font-size:13px;font-weight:600}
.ctx{font-size:13px;color:var(--mut);margin:2px 0 10px}
.note{font-size:13px;color:var(--mut);background:#f1f5f9;border-left:3px solid var(--accent);padding:10px 14px;border-radius:0 8px 8px 0;margin-top:12px}
@media(max-width:620px){.reco{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}
</style></head><body><div class="wrap">
<header>
<h1>X-SAGE — Riconoscimento delle situazioni e spiegabilità</h1>
<p>MovieLens-1M · 5 situazioni apprese · ogni raccomandazione spiegabile per costruzione</p>
</header>
<div class="stats" id="stats"></div>
<nav>
<button data-s="0" class="on">Spazio delle situazioni</button>
<button data-s="1">Profili & lente di iniquità</button>
<button data-s="2">Proiezione (L3)</button>
<button data-s="3">Esempi di raccomandazione</button>
</nav>
<section class="on" id="s0">
<h2>Spazio delle situazioni</h2>
<p class="lead">Ogni punto è una richiesta, proiettata in 2D (PCA) dal suo descrittore contesto+intento. Le richieste si raggruppano in situazioni nominate; quelle al confine fra due situazioni (cerchi vuoti) sono le <b>regioni di incertezza</b>, dove il sistema agisce con cautela.</p>
<div id="scatter"></div><div class="legend" id="leg"></div>
<div class="note">Il <b>24%</b> delle richieste cade in zona-confine (incertezza): lì la spinta situazionale viene attenuata automaticamente (gate di confidenza γ<sub>S</sub>).</div>
</section>
<section id="s1">
<h2>Profili delle situazioni & lente di iniquità</h2>
<p class="lead">Ogni situazione ha un profilo leggibile: quando accade, quali generi favorisce e quanto concentra l'esposizione. La <b>lente</b> (KL) misura quanto una situazione si concentra su poche categorie: valori alti = potenziale <b>pozzo di iniquità</b>, reso così ispezionabile.</p>
<div class="grid" id="profiles"></div>
</section>
<section id="s2">
<h2>Proiezione — dove si sta muovendo la situazione</h2>
<p class="lead">La matrice di transizione stima dove evolve la situazione dell'utente. La diagonale è la <b>persistenza</b>: situazioni stabili (es. «Azione notturna», 82%) vs transitorie (es. «Fantasy pomeridiano», 25%).</p>
<div id="heat"></div>
<div class="note">Letture: «Azione notturna» e «Commedia serale» sono stati stabili; «Fantasy pomeridiano» è transitorio e tende a evolvere verso «Horror pomeridiano» — la proiezione rende leggibile la dinamica temporale.</div>
</section>
<section id="s3">
<h2>Esempi di raccomandazione spiegati</h2>
<p class="lead">Per ogni richiesta: la situazione riconosciuta, cosa proponeva il backbone (BASE) e cosa propone X-SAGE, con il <b>nudge</b> che spiega il cambio. La spiegazione è esatta per costruzione: il contributo situazionale <i>è</i> la spiegazione.</p>
<div id="reco"></div>
</section>
</div>
<script>const D=/*DATA*/;
const COL=["#2563eb","#0d9488","#d97706","#dc2626","#7c3aed"];
const nm=k=>D.names[String(k)]||("S"+k);
// stats
document.getElementById('stats').innerHTML=
 `<div class="stat"><b>${D.K}</b><span>situazioni</span></div>`+
 `<div class="stat"><b>${Math.round(D.space.bfrac*100)}%</b><span>richieste in zona-incertezza</span></div>`+
 `<div class="stat"><b>${D.genres.length}</b><span>categorie (generi)</span></div>`+
 `<div class="stat"><b>${D.reco.examples.length}</b><span>esempi spiegati</span></div>`;
// nav
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 document.querySelectorAll('section').forEach((s,i)=>s.classList.toggle('on',i==+b.dataset.s));});
// scatter
(function(){const W=1000,H=560,pad=46;const pts=D.space.points,cen=D.space.centroids;
 const sx=v=>pad+v*(W-2*pad),sy=v=>H-pad-v*(H-2*pad);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 s+=`<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="#cbd5e1"/>`;
 s+=`<text x="${W/2}" y="${H-12}" font-size="13" fill="#6b7280" text-anchor="middle">PC1 (${Math.round(D.space.var[0]*100)}% varianza)</text>`;
 s+=`<text x="16" y="${H/2}" font-size="13" fill="#6b7280" text-anchor="middle" transform="rotate(-90 16 ${H/2})">PC2 (${Math.round(D.space.var[1]*100)}%)</text>`;
 for(const p of pts){const[x,y,k,b]=p;s+=b?`<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="3" fill="none" stroke="${COL[k]}" stroke-width="1.1" opacity=".7"/>`:`<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="3" fill="${COL[k]}" opacity=".5"/>`;}
 for(const c of cen){const X=sx(c.x),Y=sy(c.y);s+=`<circle cx="${X}" cy="${Y}" r="9" fill="#fff" stroke="${COL[c.sit]}" stroke-width="3"/><text x="${X}" y="${Y-13}" font-size="13" font-weight="600" fill="#1a1d24" text-anchor="middle">${nm(c.sit)}</text>`;}
 s+=`</svg>`;document.getElementById('scatter').innerHTML=s;
 document.getElementById('leg').innerHTML=cen.map(c=>`<span><span class="dot" style="background:${COL[c.sit]}"></span>${nm(c.sit)}</span>`).join('')+`<span><span class="dot" style="background:none;border:1.5px solid #94a3b8"></span>confine / incertezza</span>`;
})();
// profiles
(function(){const maxKL=Math.max(...D.profiles.map(s=>s.lens_KL));
 document.getElementById('profiles').innerHTML=D.profiles.map(s=>{
  const sink=s.lens_KL>=0.4;const fav=s.fav_genres.slice(0,3);const mx=Math.max(...fav.map(g=>g[1]));
  const bars=fav.map(g=>`<div class="gen"><span>${g[0]}</span><span>${g[1].toFixed(2)}</span></div><div class="bar"><i style="width:${Math.round(g[1]/mx*100)}%;background:${COL[s.k]}"></i></div>`).join('');
  return `<div class="card"><div class="sit-h"><span class="dot" style="background:${COL[s.k]}"></span><b>${nm(s.k)}</b><span class="sz">${s.size_pct}%</span></div>
   ${bars}
   <div class="lens">lente (concentrazione) <span class="badge" style="background:${sink?'#fee2e2':'#dcfce7'};color:${sink?'#b91c1c':'#15803d'}">KL ${s.lens_KL.toFixed(2)}${sink?' · pozzo':''}</span></div></div>`;
 }).join('');
})();
// heatmap
(function(){const K=D.K,T=D.trans.T,sits=D.trans.sits,W=720,cell=84,L=160,Tp=70;
 let s=`<svg viewBox="0 0 ${W} ${Tp+K*cell+30}" role="img">`;
 for(let j=0;j<K;j++)s+=`<text x="${L+j*cell+cell/2}" y="${Tp-10}" font-size="12" fill="#6b7280" text-anchor="middle">S${j}</text>`;
 for(let i=0;i<K;i++){s+=`<text x="${L-8}" y="${Tp+i*cell+cell/2+4}" font-size="12.5" fill="#1a1d24" text-anchor="end">${nm(i)}</text>`;
  for(let j=0;j<K;j++){const v=T[i][j];const a=(0.08+0.92*v).toFixed(2);const dia=i==j;
   s+=`<rect x="${L+j*cell}" y="${Tp+i*cell}" width="${cell-3}" height="${cell-3}" rx="6" fill="#2563eb" opacity="${a}"/>`;
   s+=`<text x="${L+j*cell+cell/2-1}" y="${Tp+i*cell+cell/2+4}" font-size="13" font-weight="${dia?'700':'400'}" fill="${v>0.45?'#fff':'#374151'}" text-anchor="middle">${v.toFixed(2)}</text>`;}}
 s+=`<text x="${L+K*cell/2}" y="${Tp+K*cell+22}" font-size="12" fill="#6b7280" text-anchor="middle">situazione successiva →</text></svg>`;
 document.getElementById('heat').innerHTML=s;
})();
// reco
document.getElementById('reco').innerHTML=D.reco.examples.map(e=>{
 const h=e.contesto_ora,wd=e.weekend?'weekend':'giorno feriale';
 return `<div class="card"><div class="sit-h"><span class="dot" style="background:${COL[e.k]}"></span><b>${e.situazione}</b><span class="sz">${e.core?'riconoscimento sicuro (core)':'confine'}</span></div>
  <div class="ctx">contesto: ore ${h}, ${wd}, ultimo genere visto: ${e.intento_recente}</div>
  <div class="reco"><div class="pick base"><div class="g">Backbone propone</div><div class="t">${e.BASE_top.genere} · ${e.BASE_top.titolo}</div></div>
  <div class="arrow">→<br>nudge +${e.nudge_genere_promosso}<br>su ${e.SIT_top.genere}</div>
  <div class="pick sit"><div class="g">X-SAGE propone</div><div class="t">${e.SIT_top.genere} · ${e.SIT_top.titolo}</div></div></div></div>`;
}).join('');
</script></body></html>"""

html = TEMPLATE.replace("/*DATA*/", json.dumps(DATA, ensure_ascii=False))
out = P / "demo_xsage_ml1m.html"
out.write_text(html, encoding="utf-8")
print(f"-> {out}  ({len(html)} char)")
