"""[venv-xsage] Genera un HTML singolo, autonomo e navigabile per mostrare la parte di
explainability di X-SAGE (ml1m) a un pubblico NON tecnico: spazio situazioni con regioni e confini,
profili + lente di iniquità, proiezione L3, esempi di raccomandazione REALI dal test-set. Ogni
metrica è spiegata in parole semplici; chiavi di lettura in ogni sezione. Dati embeddati inline.
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
DATA = {"names": names, "K": prof["K"], "profiles": prof["situations"], "genres": prof["genres"],
        "space": {"var": space["var"], "bfrac": space["bfrac"], "points": space["points"], "centroids": space["centroids"]},
        "trans": {"T": trans["T"], "sits": trans["situations"]}, "reco": reco}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>X-SAGE — Spiegabilità (MovieLens-1M)</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--ink:#1a1d24;--mut:#6b7280;--line:#e5e7eb;--accent:#2563eb}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55}
.wrap{max-width:1060px;margin:0 auto;padding:28px 22px 60px}
header h1{font-size:26px;font-weight:600;margin:0 0 4px}header .sub{color:var(--mut);margin:0 0 14px;font-size:15px}
.intro{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 20px;font-size:15px;margin-bottom:16px}
.intro b{color:var(--accent)}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:120px}
.stat b{font-size:20px;font-weight:600;display:block}.stat span{color:var(--mut);font-size:12px}
nav{display:flex;gap:6px;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:5;border-bottom:1px solid var(--line)}
nav button{font:inherit;font-size:14px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 14px;border-radius:8px;cursor:pointer}
nav button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
section{display:none;animation:f .25s}section.on{display:block}@keyframes f{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
h2{font-size:20px;font-weight:600;margin:24px 0 6px}.lead{color:var(--mut);font-size:14px;margin:0 0 14px;max-width:780px}
.key{background:#eef5ff;border:1px solid #d6e4ff;border-radius:10px;padding:12px 16px;font-size:13.5px;margin:14px 0;max-width:820px}
.key b{color:#1e40af}.key .q{display:inline-block;width:18px;height:18px;line-height:18px;text-align:center;border-radius:50%;background:#2563eb;color:#fff;font-size:11px;font-weight:700;margin-right:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.sit-h{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.dot{width:12px;height:12px;border-radius:50%;flex:none}.sit-h b{font-size:15px}.sit-h .sz{margin-left:auto;color:var(--mut);font-size:12px}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin:3px 0 6px}.bar i{display:block;height:100%}
.gen{display:flex;justify-content:space-between;font-size:13px;color:#374151}
.badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.lens{margin-top:8px;font-size:12px;color:var(--mut)}
svg{max-width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;align-items:center}.legend span{font-size:13px;display:flex;align-items:center;gap:6px}
.toggle{display:inline-flex;align-items:center;gap:8px;font-size:13.5px;cursor:pointer;background:var(--card);border:1px solid var(--line);padding:7px 12px;border-radius:8px;margin:4px 0 8px}
.reco{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center}
.reco .pick{padding:10px 12px;border-radius:10px;border:1px solid var(--line)}
.reco .base{background:#f9fafb}.reco .sit{background:#eff6ff;border-color:#bfdbfe}
.reco .g{font-size:12px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.reco .t{font-size:15px;font-weight:500;margin-top:2px}
.arrow{text-align:center;color:var(--accent);font-size:13px;font-weight:600}
.ctx{font-size:13px;color:var(--mut);margin:2px 0 10px}
@media(max-width:620px){.reco{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}
</style></head><body><div class="wrap">
<header>
<h1>X-SAGE — Riconoscere la situazione per spiegare la raccomandazione</h1>
<div class="sub">MovieLens-1M · un caso di studio sulla spiegabilità</div>
</header>
<div class="intro">
<b>L'idea in una frase.</b> Un sistema di raccomandazione classico ti propone film guardando <i>chi sei</i>, ma ignora <i>in che momento</i> stai guardando. X-SAGE riconosce la <b>situazione</b> (es. «una commedia, di sera, in famiglia») e adatta i suggerimenti di conseguenza — <b>senza ri-allenare</b> il modello sottostante, e in modo <b>spiegabile</b>: il contributo che aggiunge <i>è</i> la spiegazione.
</div>
<div class="stats" id="stats"></div>
<nav>
<button data-s="0" class="on">1 · Spazio delle situazioni</button>
<button data-s="1">2 · Profili & lente di iniquità</button>
<button data-s="2">3 · Proiezione (dove va la situazione)</button>
<button data-s="3">4 · Esempi reali di raccomandazione</button>
</nav>

<section class="on" id="s0">
<h2>Lo spazio delle situazioni</h2>
<p class="lead">X-SAGE impara automaticamente, dai dati, un piccolo numero di <b>situazioni ricorrenti</b>. Qui le vediamo: ogni puntino è una richiesta di raccomandazione reale, e richieste simili (per orario, giorno e cosa si stava guardando) finiscono vicine e nello stesso colore.</p>
<label class="toggle"><input type="checkbox" id="tgB"> Evidenzia le <b>zone di confine</b> (incertezza)</label>
<div id="scatter"></div><div class="legend" id="leg"></div>
<div class="key"><span class="q">i</span><b>Come leggerlo.</b> Le <b>aree colorate</b> sono le regioni di ciascuna situazione; dove si toccano e si sovrappongono nascono i <b>confini</b>. I <b>cerchi pieni</b> sono richieste assegnate con sicurezza a una situazione; i <b>cerchi vuoti</b> sono "a cavallo" fra due situazioni — il sistema non è sicuro. Attiva l'interruttore qui sopra per accenderli.</div>
<div class="key"><span class="q">?</span><b>Perché è in 2D?</b> Ogni richiesta vive in molte dimensioni; per poterla <i>disegnare</i> l'abbiamo schiacciata su un piano (tecnica chiamata PCA). I due assi catturano insieme circa il <b id="pcav"></b> della variabilità totale: la vicinanza fra punti resta indicativa della loro somiglianza.</div>
<div class="key"><span class="q">!</span><b>A cosa serve il confine?</b> Sul <b id="bf2"></b> delle richieste il sistema è incerto fra due situazioni. Lì X-SAGE <b>spinge di meno</b> (un "freno" automatico di prudenza): agisce con forza solo quando ha riconosciuto la situazione con sicurezza.</div>
</section>

<section id="s1">
<h2>Profili delle situazioni & lente di iniquità</h2>
<p class="lead">Ogni situazione non è un'etichetta opaca: ha un <b>profilo leggibile</b> — quanto è frequente e quali generi predilige. La <b>lente</b> ci dice quanto una situazione è "stretta" su pochi generi.</p>
<div class="key"><span class="q">i</span><b>Cos'è la "lente" (KL)?</b> È un numero che misura quanto i generi di una situazione si discostano dalla media generale. <b>Basso</b> = la situazione propone un po' di tutto (varietà). <b>Alto</b> = si concentra su pochissimi generi: comodo, ma rischia di diventare un <b>"pozzo"</b> che mostra sempre le stesse cose. Rendere questo numero visibile è già un atto di trasparenza: i pozzi non restano nascosti.</div>
<div class="grid" id="profiles"></div>
</section>

<section id="s2">
<h2>Proiezione — dove sta andando la situazione</h2>
<p class="lead">Le situazioni non sono statiche: dopo una commedia tendi a guardarne un'altra, dopo un'azione magari cambi. X-SAGE stima queste <b>transizioni</b>.</p>
<div class="key"><span class="q">i</span><b>Come si legge la griglia.</b> Ogni riga è la situazione di <i>adesso</i>, ogni colonna quella del <i>passo successivo</i>; il numero è la probabilità di passaggio (più scuro = più probabile). La <b>diagonale</b> (in grassetto) è la <b>persistenza</b>: la probabilità di <i>restare</i> nella stessa situazione. Alta = situazione "appiccicosa" e stabile; bassa = situazione di passaggio.</div>
<div id="heat"></div>
<div class="key"><span class="q">i</span><b>Esempio.</b> «Azione notturna» persiste all'<b>82%</b> (chi è in quella situazione tende a restarci), mentre «Fantasy pomeridiano» solo al <b>25%</b> ed evolve spesso verso «Horror pomeridiano»: la proiezione rende leggibile la dinamica nel tempo, non solo l'istante.</div>
</section>

<section id="s3">
<h2>Esempi reali — e hanno migliorato il consiglio?</h2>
<p class="lead"><b>Casi reali</b> dal test di MovieLens-1M, con un controllo: sappiamo cosa l'utente ha <b>effettivamente guardato</b> in quella richiesta. Mostriamo dove il modello base teneva quel genere <b>in fondo</b> e come X-SAGE lo fa <b>risalire</b>.</p>
<div class="key"><span class="q">i</span><b>Come si misura il beneficio.</b> Per ogni richiesta conosciamo il film che l'utente ha poi scelto (e il suo genere). Guardiamo in che <b>posizione</b> compariva il primo film di quel genere nella lista: più è in alto, meglio è. Qui il modello base lo nascondeva (es. posizione 23, fuori dai consigli visibili), mentre X-SAGE — riconoscendo la situazione — lo porta in cima. Il "<b>nudge</b>" è la spinta data a quel genere, in deviazioni standard: il numero <i>è</i> la spiegazione, niente di nascosto.</div>
<div id="reco"></div>
<div class="key"><span class="q">!</span><b>Onestà.</b> Non ogni richiesta è una vittoria così netta: in media il guadagno è piccolo ma <b>consistente e statisticamente significativo</b>. Questi sono casi rappresentativi in cui il backbone sbagliava e la situazione ha aiutato.</div>
</section>
</div>
<script>const D=/*DATA*/;
const COL=["#2563eb","#0d9488","#d97706","#dc2626","#7c3aed"];
const nm=k=>D.names[String(k)]||("S"+k);
document.getElementById('pcav').textContent=Math.round((D.space.var[0]+D.space.var[1])*100)+'%';
document.getElementById('bf2').textContent=Math.round(D.space.bfrac*100)+'%';
document.getElementById('stats').innerHTML=
 `<div class="stat"><b>${D.K}</b><span>situazioni apprese</span></div>`+
 `<div class="stat"><b>${Math.round(D.space.bfrac*100)}%</b><span>richieste incerte (di confine)</span></div>`+
 `<div class="stat"><b>${D.genres.length}</b><span>generi di film</span></div>`+
 `<div class="stat"><b>${D.reco.examples.length}</b><span>esempi reali spiegati</span></div>`;
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 document.querySelectorAll('section').forEach((s,i)=>s.classList.toggle('on',i==+b.dataset.s));window.scrollTo(0,0);});
// convex hull (monotone chain)
function hull(pts){if(pts.length<3)return pts;pts=pts.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const lo=[];
 for(const p of pts){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p);}
 const up=[];for(let i=pts.length-1;i>=0;i--){const p=pts[i];while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p);}
 return lo.slice(0,-1).concat(up.slice(0,-1));}
function drawScatter(showB){const W=1000,H=560,pad=46;const pts=D.space.points,cen=D.space.centroids;
 const sx=v=>pad+v*(W-2*pad),sy=v=>H-pad-v*(H-2*pad);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;
 for(let k=0;k<D.K;k++){const core=pts.filter(p=>p[2]===k&&p[3]===0).map(p=>[sx(p[0]),sy(p[1])]);
  const h=hull(core);if(h.length>2)s+=`<polygon points="${h.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ')}" fill="${COL[k]}" fill-opacity="${showB?.05:.10}" stroke="${COL[k]}" stroke-opacity=".35" stroke-width="1.5"/>`;}
 s+=`<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#cbd5e1"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="#cbd5e1"/>`;
 s+=`<text x="${W/2}" y="${H-12}" font-size="13" fill="#6b7280" text-anchor="middle">somiglianza →  (asse PC1, ${Math.round(D.space.var[0]*100)}%)</text>`;
 for(const p of pts){const[x,y,k,b]=p;const X=sx(x).toFixed(1),Y=sy(y).toFixed(1);
  if(b)s+=`<circle cx="${X}" cy="${Y}" r="${showB?4:3}" fill="${showB?COL[k]:'none'}" stroke="${COL[k]}" stroke-width="${showB?1.6:1.1}" opacity="${showB?1:.7}"/>`;
  else s+=`<circle cx="${X}" cy="${Y}" r="3" fill="${COL[k]}" opacity="${showB?.12:.5}"/>`;}
 for(const c of cen){const X=sx(c.x),Y=sy(c.y);s+=`<circle cx="${X}" cy="${Y}" r="9" fill="#fff" stroke="${COL[c.sit]}" stroke-width="3"/><text x="${X}" y="${Y-13}" font-size="13" font-weight="600" fill="#1a1d24" text-anchor="middle" paint-order="stroke" stroke="#fff" stroke-width="3">${nm(c.sit)}</text>`;}
 s+=`</svg>`;document.getElementById('scatter').innerHTML=s;}
drawScatter(false);
document.getElementById('tgB').onchange=e=>drawScatter(e.target.checked);
document.getElementById('leg').innerHTML=D.space.centroids.map(c=>`<span><span class="dot" style="background:${COL[c.sit]}"></span>${nm(c.sit)}</span>`).join('')+`<span><span class="dot" style="background:none;border:1.5px solid #94a3b8"></span>richiesta di confine (incertezza)</span>`;
// profiles
document.getElementById('profiles').innerHTML=D.profiles.map(s=>{
 const sink=s.lens_KL>=0.4;const fav=s.fav_genres.slice(0,3);const mx=Math.max(...fav.map(g=>g[1]));
 const bars=fav.map(g=>`<div class="gen"><span>${g[0]}</span><span>+${g[1].toFixed(2)}</span></div><div class="bar"><i style="width:${Math.round(g[1]/mx*100)}%;background:${COL[s.k]}"></i></div>`).join('');
 return `<div class="card"><div class="sit-h"><span class="dot" style="background:${COL[s.k]}"></span><b>${nm(s.k)}</b><span class="sz">${s.size_pct}% delle richieste</span></div>
  <div style="font-size:12px;color:#6b7280;margin-bottom:6px">generi che questa situazione favorisce</div>${bars}
  <div class="lens">varietà → concentrazione <span class="badge" style="background:${sink?'#fee2e2':'#dcfce7'};color:${sink?'#b91c1c':'#15803d'}">lente ${s.lens_KL.toFixed(2)}${sink?' · pozzo':''}</span></div></div>`;
}).join('');
// heatmap
(function(){const K=D.K,T=D.trans.T,W=720,cell=84,L=170,Tp=72;
 let s=`<svg viewBox="0 0 ${W} ${Tp+K*cell+34}" role="img">`;
 s+=`<text x="${L+K*cell/2}" y="22" font-size="13" fill="#1a1d24" text-anchor="middle" font-weight="600">situazione successiva</text>`;
 for(let j=0;j<K;j++)s+=`<text x="${L+j*cell+cell/2}" y="${Tp-10}" font-size="12" fill="#6b7280" text-anchor="middle">S${j}</text>`;
 for(let i=0;i<K;i++){s+=`<text x="${L-8}" y="${Tp+i*cell+cell/2+4}" font-size="12.5" fill="#1a1d24" text-anchor="end">${nm(i)}</text>`;
  for(let j=0;j<K;j++){const v=T[i][j];const a=(0.08+0.92*v).toFixed(2);const dia=i==j;
   s+=`<rect x="${L+j*cell}" y="${Tp+i*cell}" width="${cell-3}" height="${cell-3}" rx="6" fill="#2563eb" opacity="${a}"/>`;
   s+=`<text x="${L+j*cell+cell/2-1}" y="${Tp+i*cell+cell/2+4}" font-size="13" font-weight="${dia?'700':'400'}" fill="${v>0.45?'#fff':'#374151'}" text-anchor="middle">${v.toFixed(2)}</text>`;}}
 s+=`</svg>`;document.getElementById('heat').innerHTML=s;})();
// reco
document.getElementById('reco').innerHTML=D.reco.examples.map(e=>{
 const wd=e.weekend?'weekend':'giorno feriale';const bad=e.rank_base>20;
 return `<div class="card"><div class="sit-h"><span class="dot" style="background:${COL[e.k]}"></span><b>${e.situazione}</b><span class="sz">situazione riconosciuta con sicurezza</span></div>
  <div class="ctx">contesto: ore ${e.contesto_ora}, ${wd}, ultimo genere visto: ${e.intento_recente}</div>
  <div style="background:#fefce8;border:1px solid #fde68a;border-radius:8px;padding:8px 12px;font-size:14px;margin-bottom:10px">Verità — l'utente ha poi guardato: <b>${e.guardato.genere} · ${e.guardato.titolo}</b></div>
  <div style="font-size:12.5px;color:#6b7280;margin-bottom:6px">posizione del primo film «${e.guardato.genere}» nella lista raccomandata:</div>
  <div class="reco">
   <div class="pick base" style="text-align:center"><div class="g">modello base</div><div class="t" style="color:${bad?'#b91c1c':'#374151'}">posizione ${e.rank_base}${bad?'<div style="font-size:11px;font-weight:400">fuori dai top-20</div>':''}</div></div>
   <div class="arrow">→<br>spinta +${e.nudge_cat_vera}<br>su «${e.guardato.genere}»</div>
   <div class="pick sit" style="text-align:center"><div class="g">X-SAGE</div><div class="t" style="color:#15803d">posizione ${e.rank_sit}${e.rank_sit==1?'<div style="font-size:11px;font-weight:400">in cima ai consigli</div>':''}</div></div>
  </div>
  <div style="font-size:12.5px;color:#6b7280;margin-top:10px">primo consiglio in lista: <b>${e.BASE_top.genere}</b> «${e.BASE_top.titolo}» &nbsp;→&nbsp; <b>${e.SIT_top.genere}</b> «${e.SIT_top.titolo}»</div>
 </div>`;
}).join('');
</script></body></html>"""
html = TEMPLATE.replace("/*DATA*/", json.dumps(DATA, ensure_ascii=False))
out = P / "demo_xsage_ml1m.html"; out.write_text(html, encoding="utf-8")
print(f"-> {out}  ({len(html)} char)")
