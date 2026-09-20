"""[venv-xsage] Misura del costo computazionale della testa situazionale — per la revisione BDCC.

Risponde a R1#5: il paper afferma "negligible computational cost" senza numeri.

Misura, per ogni dataset:
  (A) FITTING, una tantum offline, fase per fase
      L1a  fit_contribution_functions      predittori shallow per attributo
      L1b  estimate_macro_transition        matrice di transizione macro + attrattori
      L2   select_K / select_eps / fit_rough_kmeans   clustering rough
      L3   fit_situation_biases_z           tabella dei bias per situazione
  (B) INFERENZA, per richiesta, sul solo split di test
      featurization  descrittore v = [c~ || e]
      assignment     K distanze + membership + gate di certezza
      re-ranking     somma additiva sul catalogo
  (C) PARAMETRI della testa: prototipi K x d, bias K x C, nodi degli alberi.

Le fasi sono le stesse chiamate di mind_prep.build_descriptor: nessuna
reimplementazione, quindi i tempi si riferiscono al codice che produce i
risultati del paper.

IMPORTANTE: lanciare a macchina scarica. Se gira in parallelo ad altro
(per esempio sweep_sensitivity.py) i tempi non hanno senso.

Uso:
  python scripts/validation/measure_cost.py                 # tutti i dataset
  python scripts/validation/measure_cost.py --cities ml1m   # uno solo
  python scripts/validation/measure_cost.py --repeats 5     # ripetizioni per la mediana

Out: outputs_results/cost_measurements.csv  (una riga per city/phase)
     + un riepilogo a video, gia' nella forma della tabella dell'appendice.
"""
import sys, time, json, platform, argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "mind"))
sys.path.insert(0, str(CLEAN / "scripts" / "yelp"))
sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")

from mind_prep import city_attrs, closed_params, membership_from_assign, ALPHA  # noqa: E402
from pipeline.step02_models.xsage.l0_sensing import build_recent_window  # noqa: E402
from pipeline.step02_models.xsage.l1_perception import (                 # noqa: E402
    compute_intent, compute_profile, estimate_macro_transition,
    fit_contribution_functions, find_attractors)
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans  # noqa: E402
from xsage import data as D                                             # noqa: E402
from xsage.recommendation import fit_situation_biases_z                  # noqa: E402
from eval_kappa import select_K, select_eps                              # noqa: E402

SEED = 42


class T:
    """Cronometro a mediana su piu' ripetizioni; tiene anche il risultato dell'ultima."""

    def __init__(self, repeats):
        self.repeats = repeats
        self.out = {}

    def __call__(self, name, fn, repeats=None):
        r = self.repeats if repeats is None else repeats
        ts, val = [], None
        for _ in range(r):
            t0 = time.perf_counter()
            val = fn()
            ts.append(time.perf_counter() - t0)
        self.out[name] = float(np.median(ts))
        return val


def count_tree_nodes(obj, seen=None, depth=0):
    """Somma i nodi di ogni albero sklearn raggiungibile dall'oggetto dei contributi.
    Difensivo: se la struttura non e' quella attesa restituisce None."""
    if depth > 4:
        return 0
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    tot = 0
    tree = getattr(obj, "tree_", None)
    if tree is not None and hasattr(tree, "node_count"):
        return int(tree.node_count)
    if isinstance(obj, dict):
        for v in obj.values():
            tot += count_tree_nodes(v, seen, depth + 1)
        return tot
    if isinstance(obj, (list, tuple)):
        for v in obj:
            tot += count_tree_nodes(v, seen, depth + 1)
        return tot
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for v in d.values():
            tot += count_tree_nodes(v, seen, depth + 1)
    return tot


def run_city(city, repeats):
    t = T(repeats)
    P = closed_params(city)
    gamma, depth, n, beta, hh = P["gamma"], P["depth"], P["n"], P["beta"], P["H"]
    attrs = city_attrs(city)

    # --- caricamento dati (non e' costo della testa: escluso dai totali) -------
    t0 = time.perf_counter()
    ds = D.load_city(city, data_root=str(CLEAN))
    m2i = ds["macro_to_idx"]; nmac = ds["n_macros"]; nI = ds["n_items"]
    for k in ("df_train", "df_val", "df_test"):
        ds[k] = ds[k].copy()
        ds[k]["cat_target"] = ds[k]["cat_macro"].map(m2i).astype(np.int64)
        ds[k]["user_id"] = ds[k]["u_idx"]
    t_load = time.perf_counter() - t0

    hist = {"train": ds["df_train"], "val": ds["df_train"],
            "test": pd.concat([ds["df_train"], ds["df_val"]], ignore_index=True)}

    # --- (A) FITTING (una tantum: una sola esecuzione, la giratura non e' critica) ---
    contrib = t("fit_L1a_contribution", lambda: fit_contribution_functions(
        ds["df_train"], m2i, attributes=attrs, max_depth=depth, min_leaf=200), repeats=1)
    W = t("fit_L1b_transition", lambda: estimate_macro_transition(
        ds["df_train"], m2i, transit_macros=[], transit_mode="keep"), repeats=1)
    attractors = t("fit_L1b_attractors", lambda: find_attractors(W, exclude_indices=None), repeats=1)

    def featurize(split):
        tgt = ds[f"df_{split}"]
        l0 = build_recent_window(tgt, hist[split], m2i, n=n)
        c = contrib.transform(tgt)
        m = compute_profile(l0.recent_macro, l0.n_prior, nmac, gamma=gamma)
        e = compute_intent(m, W, attractors, H=hh, beta=beta, mode="hard")
        return np.concatenate([c, e], axis=1).astype(np.float32)

    vtr = t("fit_featurize_train", lambda: featurize("train"), repeats=1)
    vva = featurize("val")
    vte = t("infer_featurize_test", lambda: featurize("test"))     # ripetuto: e' costo online
    d_desc = int(vte.shape[1])
    nq = int(vte.shape[0])

    rng = np.random.default_rng(SEED)
    K = t("fit_L2_selectK", lambda: select_K(vtr, int(attractors.sum()) + 2,
                                             np.random.default_rng(SEED)), repeats=1)
    eps = t("fit_L2_selectEps", lambda: select_eps(vtr, vva, K), repeats=1)
    fit = t("fit_L2_kmeans", lambda: fit_rough_kmeans(vtr, K=K, eps=eps,
                                                      seed=SEED, max_iter=80), repeats=1)
    cmt = ds["df_train"]["cat_macro"].map(m2i).values.astype(np.int64)
    z_tr = fit.core_label.astype(np.int64)
    b_z = t("fit_L3_biases", lambda: fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA),
            repeats=1)

    # --- (B) INFERENZA per richiesta ------------------------------------------
    def assign_step():
        _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
        mem = membership_from_assign(kte, compte, isbte, K)
        gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
        return mem.astype(np.float32) @ b_z, gam.astype(np.float32)

    nudge, gam = t("infer_assign", assign_step)

    df_all = pd.concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (df_all.groupby("i_idx")["cat_macro"].first().map(m2i)
           .reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    sB = rng.standard_normal((min(nq, 2048), nI)).astype(np.float32)   # score fittizi del backbone
    nb = sB.shape[0]
    kap = 0.25

    def rerank_step():
        return sB + (kap * gam[:nb, None]) * nudge[:nb][:, icm]

    t("infer_rerank", rerank_step)

    # --- (C) PARAMETRI --------------------------------------------------------
    p_proto = int(np.prod(fit.prototypes.shape))
    p_bias = int(np.prod(np.asarray(b_z).shape))
    p_trees = count_tree_nodes(contrib)
    p_W = int(np.asarray(W).size)

    o = t.out
    fit_total = (o["fit_L1a_contribution"] + o["fit_L1b_transition"] + o["fit_L1b_attractors"]
                 + o["fit_featurize_train"] + o["fit_L2_selectK"] + o["fit_L2_selectEps"]
                 + o["fit_L2_kmeans"] + o["fit_L3_biases"])
    per_req_us = {k: o[k] / nq * 1e6 for k in ("infer_featurize_test", "infer_assign")}
    per_req_us["infer_rerank"] = o["infer_rerank"] / nb * 1e6
    infer_total_us = sum(per_req_us.values())

    row = dict(city=city, n_requests_test=nq, n_items=nI, n_macros=nmac,
               K=int(K), eps=float(eps), d_descriptor=d_desc,
               t_load_data_s=round(t_load, 3),
               **{k: round(v, 4) for k, v in o.items()},
               fit_total_s=round(fit_total, 3),
               us_featurize=round(per_req_us["infer_featurize_test"], 2),
               us_assign=round(per_req_us["infer_assign"], 2),
               us_rerank=round(per_req_us["infer_rerank"], 2),
               us_total=round(infer_total_us, 2),
               params_prototypes=p_proto, params_bias=p_bias,
               params_trees_nodes=p_trees, params_transition=p_W,
               params_total=p_proto + p_bias + p_trees + p_W)
    return row


def main():
    p = argparse.ArgumentParser(description="Misura del costo computazionale della testa situazionale")
    p.add_argument("--cities", nargs="+",
                   default=["ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand"])
    p.add_argument("--repeats", type=int, default=5,
                   help="ripetizioni per la mediana delle misure ONLINE (il fitting gira una volta sola)")
    p.add_argument("--out", default=str(CLEAN / "outputs_results" / "cost_measurements.csv"))
    a = p.parse_args()

    print("ATTENZIONE: lanciare a macchina scarica, altrimenti i tempi non sono validi.\n")
    print(f"Hardware: {platform.processor() or platform.machine()} | "
          f"{platform.system()} {platform.release()} | Python {platform.python_version()} | "
          f"NumPy {np.__version__}\n", flush=True)

    rows = []
    for c in a.cities:
        print(f"=== {c} ===", flush=True)
        r = run_city(c, a.repeats)
        rows.append(r)
        print(f"  fitting totale {r['fit_total_s']:.2f}s | per richiesta {r['us_total']:.1f} us "
              f"(feat {r['us_featurize']:.1f} + assign {r['us_assign']:.1f} + rerank {r['us_rerank']:.1f}) "
              f"| parametri {r['params_total']:,}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n--- riepilogo per l'appendice ---")
    cols = ["city", "fit_total_s", "us_featurize", "us_assign", "us_rerank", "us_total",
            "K", "d_descriptor", "params_total"]
    print(df[cols].to_string(index=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
