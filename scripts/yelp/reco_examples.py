"""[venv-xsage] Esempi di raccomandazione REALI con BENEFICIO verificabile (ml1m).
Per richieste test dove SIT migliora il consiglio rispetto a BASE (backbone B_blind), legati alla
VERITÀ nota (il film effettivamente interagito nella richiesta di test e la sua categoria):
mostra come X-SAGE faccia RISALIRE la categoria vera (ciò che misura il Cat-MRR) dove il backbone
la teneva in basso. Selezione: 1 caso per situazione, core, situazione che favorisce la categoria
vera, e rango-categoria SIT < BASE (miglioramento reale). Ricostruisce i_idx→titolo dal preprocess.
Uso:  python scripts/yelp/reco_examples.py
Out:  outputs_results/explain/reco_examples_ml1m.json + stampa leggibile
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
ML = CLEAN / "data" / "ml-1m"


def reconstruct_titles():
    gen = {}
    for ln in open(ML / "movies.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 3:
            g = p[2].split("|")[0].strip()
            if g and g != "(no genres listed)": gen[p[0]] = (p[1], g)
    rows = []
    for ln in open(ML / "ratings.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 4 and p[1] in gen: rows.append((p[0], p[1]))
    df = pd.DataFrame(rows, columns=["user", "item"])
    while True:
        uc = df.user.value_counts(); ic = df.item.value_counts()
        m = df.user.isin(uc[uc >= 10].index) & df.item.isin(ic[ic >= 10].index)
        if m.all(): break
        df = df[m]
    return {i: gen[it][0] for i, it in enumerate(sorted(df.item.unique()))}


def cat_rank(s, cc, true_cat, icm):
    """Rango (1-based) del PRIMO item di categoria true_cat nella lista ordinata per s (mask cc)."""
    s = s.copy(); s[cc] = -np.inf
    mt = icm == true_cat
    if not np.isfinite(s[mt]).any(): return 9999
    best = s[mt].max()
    return int((s > best).sum()) + 1


def main():
    titles = reconstruct_titles()
    D0 = build_descriptor("ml1m", splits=("train", "val", "test")); rng = np.random.default_rng(SEED)
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    i2m = {v: k for k, v in ds["macro_to_idx"].items()}
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    gam_te = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
    EXP = CLEAN / "outputs_results" / "explain"
    name = {int(k): v for k, v in json.load(open(EXP / "situation_names_ml1m.json")).items()}
    csv = pd.read_csv(CLEAN / "outputs_results" / "battery_bfull_ml1m.csv")
    kap = float(csv[(csv.seed == 42) & (csv.backbone == "B_blind") & (csv.method == "SIT")]["kstar"].iloc[0])
    dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
    hr = dft["c_hour"].values; wk = dft["c_isweekend"].values; il = dft["intent_last_cat_idx"].values
    nudge = b_z[kte]  # core: r=1 → nudge della situazione assegnata

    best = {}                                    # un caso (il miglior miglioramento) per situazione
    N = min(len(dft), 60000)
    for r in range(N):
        if bool(isbte[r]): continue
        z = int(kte[r]); tc = int(icm[ite[r]])
        if nudge[r][tc] <= 0: continue           # la situazione deve favorire la categoria VERA
        u = int(ute[r]); cc = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        base = sb[u].astype(np.float64)
        rb = cat_rank(base, cc, tc, icm)
        if rb < 4: continue                       # il backbone teneva la categoria vera sotto i primi 3
        sit = base + kap * gam_te[r] * nudge[r][icm]
        rs = cat_rank(sit, cc, tc, icm)
        if rs > 12 or rs >= rb: continue          # X-SAGE la porta in posizione VISIBILE (≤12) e migliora
        gain = 1.0 / rs - 1.0 / rb                # guadagno di Cat-MRR (premia l'arrivo in alto)
        if z not in best or gain > best[z]["gain"]:
            bt = base.copy(); bt[cc] = -np.inf; st = sit.copy(); st[cc] = -np.inf
            btop, stop = int(np.argmax(bt)), int(np.argmax(st))
            best[z] = dict(gain=gain, situazione=name.get(z, f"S{z}"), k=z,
                           contesto_ora=int(hr[r]), weekend=bool(wk[r]),
                           intento_recente=i2m.get(int(il[r]), "—") if il[r] < nmac else "—",
                           guardato={"genere": i2m[tc], "titolo": titles.get(int(ite[r]), f"item{ite[r]}")},
                           rank_base=rb, rank_sit=rs,
                           BASE_top={"genere": i2m[icm[btop]], "titolo": titles.get(btop, f"item{btop}")},
                           SIT_top={"genere": i2m[icm[stop]], "titolo": titles.get(stop, f"item{stop}")},
                           nudge_cat_vera=round(float(nudge[r][tc]), 3))
    examples = [best[k] for k in sorted(best)]
    json.dump({"city": "ml1m", "kappa": kap, "examples": examples}, open(EXP / "reco_examples_ml1m.json", "w"), ensure_ascii=False, indent=2)
    print(f"=== ESEMPI CON BENEFICIO (ml1m, κ={kap}) ===")
    for i, e in enumerate(examples, 1):
        print(f"\n[{i}] «{e['situazione']}» — ore {e['contesto_ora']}, {'weekend' if e['weekend'] else 'feriale'}, ultimo: {e['intento_recente']}")
        print(f"    L'utente ha guardato: {e['guardato']['genere']} · «{e['guardato']['titolo']}»")
        print(f"    categoria vera ({e['guardato']['genere']}): backbone la metteva in posizione {e['rank_base']} → X-SAGE in posizione {e['rank_sit']}")
        print(f"    top-1 backbone: {e['BASE_top']['genere']} «{e['BASE_top']['titolo']}»  |  top-1 X-SAGE: {e['SIT_top']['genere']} «{e['SIT_top']['titolo']}»")
    return 0


if __name__ == "__main__":
    sys.exit(main())
