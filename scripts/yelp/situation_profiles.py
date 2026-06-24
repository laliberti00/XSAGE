"""[venv-xsage] Estrae i PROFILI delle situazioni (per la visualizzazione/explainability) su ml-1m:
per ogni situazione k → etichetta-contesto (attributi sovra-rappresentati), generi favoriti (b̃^k),
lente KL, dimensione; + la matrice b_z (K×generi) per la heatmap; + un esempio di spiegazione
situazionale (una raccomandazione dove SIT cambia il top-1). Stampa JSON su stdout.
Uso:  python scripts/yelp/situation_profiles.py ml1m
"""
import json, sys
from pathlib import Path
import numpy as np
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA, K_TOP
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

HOUR = {**{h: "notte" for h in range(0, 6)}, **{h: "mattina" for h in range(6, 12)},
        **{h: "pomeriggio" for h in range(12, 18)}, **{h: "sera" for h in range(18, 24)}}


def over(series, glob_p, labels=None):
    """valore più sovra-rappresentato (lift) con supporto decente."""
    p = series.value_counts(normalize=True)
    best, blift = None, 0
    for v, pv in p.items():
        if pv < 0.15: continue
        lift = pv / max(glob_p.get(v, 1e-9), 1e-9)
        if lift > blift: best, blift = v, lift
    return (labels.get(best, best) if (labels and best is not None) else best), float(p.get(best, 0))


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"; rng = np.random.default_rng(SEED)
    D0 = build_descriptor(city, splits=("train", "val", "test"))
    ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]; cmt = D0["cmt"]
    i2m = {v: k for k, v in ds["macro_to_idx"].items()}; genres = [i2m[i] for i in range(nmac)]
    vtr, vte = D0["vs"]["train"], D0["vs"]["test"]
    K = select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = select_eps(vtr, D0["vs"]["val"], K)
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); z = fit.core_label.astype(np.int64)
    b_z = fit_situation_biases_z(z, cmt, K, nmac, alpha=ALPHA)             # K × generi

    dtr = ds["df_train"].reset_index(drop=True)
    gp_hour = dtr["c_hour"].map(HOUR).value_counts(normalize=True).to_dict()
    gp_we = dtr["c_isweekend"].value_counts(normalize=True).to_dict()
    gp_int = dtr["intent_last_cat_idx"].value_counts(normalize=True).to_dict()

    # lente: KL(p(macro|z) ‖ globale)
    cnt = np.zeros((K, nmac))
    for zi, ci in zip(z, cmt): cnt[zi, ci] += 1
    glob = cnt.sum(0); glob = glob / glob.sum()
    sits = []
    for k in range(K):
        m = z == k; sub = dtr[m]
        hb, _ = over(sub["c_hour"].map(HOUR), gp_hour)
        we, _ = over(sub["c_isweekend"], gp_we, {0: "settimana", 1: "weekend"})
        it, _ = over(sub["intent_last_cat_idx"], gp_int, {**i2m, nmac: "nessuno"})
        fav = sorted(range(nmac), key=lambda g: -b_z[k, g])[:3]
        p = cnt[k] / max(cnt[k].sum(), 1); mask = p > 0
        kl = float(np.sum(p[mask] * np.log2(p[mask] / np.maximum(glob[mask], 1e-12))))
        sits.append({"k": int(k), "size_pct": round(100 * m.mean(), 1),
                     "label": f"{hb or '—'}, {we or '—'}, dopo {it or '—'}",
                     "recent": it, "fav_genres": [[genres[g], round(float(b_z[k, g]), 2)] for g in fav],
                     "lens_KL": round(kl, 3)})

    # esempio di spiegazione: una richiesta test dove SIT alza il top-1 verso un genere favorito
    _, kte, comp, isb = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(kte, comp, isb, K); gam = 1.0 / np.maximum(comp.sum(1), 1)
    dft = ds["df_test"].reset_index(drop=True); u = dft["u_idx"].values.astype(np.int64)
    kap = 0.5; example = None
    dmac = mem.astype(np.float32) @ b_z
    for j in rng.permutation(len(dft))[:4000]:
        uj = int(u[j])
        cc = excl.indices[excl.indptr[uj]:excl.indptr[uj + 1]]
        base = sb[uj].astype(np.float32).copy(); base[cc] = -np.inf
        sit = base + kap * gam[j] * dmac[j][icm]
        b1, s1 = int(base.argmax()), int(sit.argmax())
        if b1 != s1 and icm[s1] != icm[b1]:
            example = {"situation": sits[int(kte[j])]["label"], "k": int(kte[j]),
                       "favored": genres[icm[s1]], "base_top_genre": genres[icm[b1]],
                       "sit_top_genre": genres[icm[s1]], "nudge": round(float(kap*gam[j]*dmac[j][icm[s1]]), 3)}
            break

    print(json.dumps({"city": city, "K": int(K), "genres": genres,
                      "b_z": [[round(float(x), 2) for x in row] for row in b_z],
                      "situations": sits, "example": example}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
