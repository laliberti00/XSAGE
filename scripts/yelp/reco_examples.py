"""[venv-xsage] Esempi di raccomandazione REALI con spiegazione situazionale (ml1m).
Per richieste test dove SIT cambia il top-1 vs BASE (backbone B_blind), stampa:
contesto recente → situazione inferita (nominata, core/boundary) → top-1 BASE vs top-1 SIT
(genere + TITOLO film) → il nudge situazionale che spiega il cambio.
Ricostruisce i_idx→titolo replicando l'enumerazione del preprocess (deterministica).
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
    """i_idx → titolo, replicando preprocess_ml1m (genre filter + k-core10 + sorted unique)."""
    gen = {}
    for ln in open(ML / "movies.dat", encoding="latin-1"):
        p = ln.rstrip("\n").split("::")
        if len(p) >= 3:
            g = p[2].split("|")[0].strip()
            if g and g != "(no genres listed)": gen[p[0]] = (p[1], g)   # movieId -> (titolo, genere)
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
    items = sorted(df.item.unique())
    return {i: gen[it][0] for i, it in enumerate(items)}   # i_idx -> titolo


def main():
    city = "ml1m"; rng = np.random.default_rng(SEED)
    titles = reconstruct_titles()
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    i2m = {v: k for k, v in ds["macro_to_idx"].items()}
    vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, vva, K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z_tr = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    mem_te = membership_from_assign(kte, compte, isbte, K); gam_te = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
    labels = json.load(open(CLEAN / "outputs_results" / "explain" / "situation_profiles_ml1m.json"))["situations"]
    name = {s["k"]: s["label"] for s in labels}
    csv = pd.read_csv(CLEAN / "outputs_results" / "battery_bfull_ml1m.csv")
    kap = float(csv[(csv.seed == 42) & (csv.backbone == "B_blind") & (csv.method == "SIT")]["kstar"].iloc[0])

    dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
    hr = dft["c_hour"].values; wk = dft["c_isweekend"].values; il = dft["intent_last_cat_idx"].values
    nudge = mem_te.astype(np.float32) @ b_z                              # [n_test x nmac]
    by_sit = {}                                                          # un esempio DIVERSO per situazione
    for r in range(len(dft)):
        z = int(kte[r])
        if z in by_sit or bool(isbte[r]): continue                      # situazione confidente (core), una per situazione
        u = int(ute[r]); s = sb[u].astype(np.float64).copy(); cc = excl.indices[excl.indptr[u]:excl.indptr[u + 1]]
        s[cc] = -np.inf; base_top = int(np.argmax(s))
        sit = s + kap * gam_te[r] * nudge[r][icm]; sit[cc] = -np.inf; sit_top = int(np.argmax(sit))
        if base_top == sit_top or nudge[r][icm[sit_top]] <= 0: continue  # SIT cambia il top-1 verso un genere favorito
        by_sit[z] = {
            "situazione": name.get(z, f"S{z}"), "k": z, "core": True,
            "contesto_ora": int(hr[r]), "weekend": bool(wk[r]),
            "intento_recente": i2m.get(int(il[r]), "—") if il[r] < nmac else "—",
            "BASE_top": {"genere": i2m[icm[base_top]], "titolo": titles.get(base_top, f"item{base_top}")},
            "SIT_top": {"genere": i2m[icm[sit_top]], "titolo": titles.get(sit_top, f"item{sit_top}")},
            "nudge_genere_promosso": round(float(nudge[r][icm[sit_top]]), 3),
        }
        if len(by_sit) >= K: break
    examples = [by_sit[k] for k in sorted(by_sit)]

    out = CLEAN / "outputs_results" / "explain" / "reco_examples_ml1m.json"
    json.dump({"city": city, "kappa": kap, "examples": examples}, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"=== ESEMPI DI RACCOMANDAZIONE (ml1m, κ={kap}) ===")
    for i, e in enumerate(examples, 1):
        print(f"\n[{i}] situazione: «{e['situazione']}» ({'core' if e['core'] else 'boundary'})")
        print(f"    contesto: ore {e['contesto_ora']}, {'weekend' if e['weekend'] else 'settimana'}, ultimo genere visto = {e['intento_recente']}")
        print(f"    BASE proponeva:  {e['BASE_top']['genere']:12} → «{e['BASE_top']['titolo']}»")
        print(f"    X-SAGE propone:  {e['SIT_top']['genere']:12} → «{e['SIT_top']['titolo']}»   (nudge +{e['nudge_genere_promosso']} su {e['SIT_top']['genere']} nella situazione)")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
