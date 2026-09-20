"""[venv-xsage] Sensitivity sweep su kappa, K, epsilon e gating — per la revisione BDCC.

Risponde a: R1#3 (sensibilita' a K, epsilon, peso della componente situazionale),
R4#8 (frazione di boundary alternativa; gating singolo vs doppio/quadratico).

Riusa ESATTAMENTE la pipeline di valutazione del paper importando le funzioni da
results_record.py, cosi' i numeri sono confrontabili con results_record.csv.
Nessun ri-addestramento: gli score dei backbone sono gia' in cache su disco
(data/<city>/backbone/*.npy). B_full richiede training ed e' opt-in (--with-bfull).

Costo: per ogni (city, seed, config) si ricalcola solo il clustering (secondi) e
una passata di scoring per backbone/kappa. La matrice di score si carica in mmap.

Uso:
  # smoke test veloce (consigliato PRIMA della run notturna)
  python scripts/validation/sweep_sensitivity.py --smoke

  # run completa (notturna)
  python scripts/validation/sweep_sensitivity.py \
      --cities ml1m nyc_tist saopaulo yelp kuairand \
      --backbones B_blind EASE AFM SASRec \
      --out outputs_results/sweep_sensitivity.csv

Out: CSV tidy con una riga per (city, seed, backbone, phase, K, eps, kappa, gating).
     Scrittura INCREMENTALE: se la run si interrompe, i risultati gia' ottenuti restano.
"""
import sys, argparse, itertools, time
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "yelp"))

# Import dalla pipeline di valutazione del paper (results_record ha la guardia __main__,
# quindi l'import non esegue nulla).
from results_record import (                      # noqa: E402
    build_descriptor, membership_from_assign, ALPHA,
    select_K, select_eps, _assign, fit_rough_kmeans,
    fit_situation_biases_z, per_request_eval, metrics_from,
    SEEDS, BK,
)

# Metriche registrate (sottoinsieme di metrics_from, sufficiente per la sensitivity)
KEEP = ["macroCatMRR", "CatMRR", "HR20", "NDCG20", "Coverage", "Gini", "LT20"]


def score_fn_for(bk, city, ute, sb, ds, icm, excl, nmac, dev=None, seed=42, with_bfull=False):
    """Costruisce la funzione di scoring del backbone, come in results_record.run_city.
    Ritorna None se gli score non sono disponibili (backbone saltato)."""
    if bk == "B_blind":
        return (lambda idx, u=ute: sb[u[idx]])
    if bk == "B_full":
        if not with_bfull:
            return None                                  # richiede training: opt-in
        from results_record import bfull_scores          # import lazy (torch)
        bft = bfull_scores(ds, icm, excl, nmac, dev, seed)
        return (lambda idx, S=bft: S[idx])
    bdir = CLEAN / "data" / city / "backbone"
    fu, ft = bdir / f"{bk}.scores_user.npy", bdir / f"{bk}.scores_test.npy"
    if fu.exists():
        M = np.load(fu, mmap_mode="r")
        return (lambda idx, M=M, u=ute: M[u[idx]])
    if ft.exists():
        Mt = np.load(ft, mmap_mode="r")
        return (lambda idx, Mt=Mt: Mt[idx])
    return None


def situation_state(vtr, vte, cmt, K, eps, seed, nmac):
    """Rifitta il livello di comprensione per una data (K, eps) e restituisce
    nudge, gate di certezza e frazione di boundary."""
    fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80)
    z_tr = fit.core_label.astype(np.int64)
    _, kte, compte, isbte = _assign(vte, fit.prototypes, eps)
    mem = membership_from_assign(kte, compte, isbte, K)
    gam = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
    b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA)
    nudge = mem.astype(np.float32) @ b_z
    return nudge, gam.astype(np.float32), float(isbte.mean())


def run(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = None
    if args.with_bfull:
        import torch
        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    wrote_header = out.exists() and out.stat().st_size > 0
    t0 = time.time()

    for city in args.cities:
        print(f"=== {city} ===", flush=True)
        D0 = build_descriptor(city, splits=("train", "val", "test"))
        ds, nmac, icm, excl = D0["ds"], D0["n_macros"], D0["icm"], D0["excl"]
        sb, cmt = D0["sb"], D0["cmt"]
        G1 = D0["G1"].astype(np.float32)
        nI = int(ds["n_items"])
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        dft = ds["df_test"]
        ute = dft["u_idx"].values.astype(np.int64)
        ite = dft["i_idx"].values.astype(np.int64)

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            K_sel = select_K(vtr, int(D0["attractors"].sum()) + 2, rng)
            eps_sel = select_eps(vtr, vva, K_sel)
            print(f"  [{city}] seed {seed}: K*={K_sel} eps*={eps_sel}", flush=True)

            # configurazioni: (fase, K, eps)
            cfgs = [("kappa", K_sel, eps_sel)]
            if not args.no_keps:
                for K in args.k_grid:
                    for e in args.eps_grid:
                        if (K, e) != (K_sel, eps_sel):
                            cfgs.append(("keps", K, e))

            for phase, K, eps in cfgs:
                nudge, gam, bfrac = situation_state(vtr, vte, cmt, K, eps, seed, nmac)
                # gating: doppio = membership(1/|T|) * gamma(1/|T|)  [default, come nel paper]
                #         singolo = solo membership (gamma=1)
                gatings = ["double"] if phase == "keps" and not args.gating_both else args.gatings
                kappas = args.kappa_grid if phase == "kappa" else [args.kappa_ref]

                for bk in args.backbones:
                    sfn = score_fn_for(bk, city, ute, sb, ds, icm, excl, nmac,
                                       dev=dev, seed=seed, with_bfull=args.with_bfull)
                    if sfn is None:
                        continue
                    for gating in gatings:
                        g = gam if gating == "double" else np.ones_like(gam)
                        rows = []
                        for kappa in kappas:
                            ev = per_request_eval(sfn, nudge, float(kappa), g,
                                                  ute, ite, icm, excl, G1, nmac, None)
                            mm = metrics_from(ev, nmac, nI)
                            rows.append(dict(city=city, seed=seed, backbone=bk, phase=phase,
                                             K=K, eps=eps, kappa=kappa, gating=gating,
                                             boundary_frac=round(bfrac, 4),
                                             **{k: mm[k] for k in KEEP}))
                        df = pd.DataFrame(rows)
                        df.to_csv(out, mode="a", header=not wrote_header, index=False)
                        wrote_header = True
                        print(f"    {bk:8} phase={phase:5} K={K} eps={eps} gating={gating} "
                              f"-> {len(rows)} righe  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n-> {out}   (tempo totale {time.time()-t0:.0f}s)", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser(description="Sensitivity sweep (kappa, K, eps, gating)")
    p.add_argument("--cities", nargs="+", default=["ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand"])
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--backbones", nargs="+", default=["B_blind", "EASE", "AFM", "SASRec"],
                   help=f"sottoinsieme di {BK}; B_full richiede --with-bfull")
    p.add_argument("--kappa-grid", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5])
    p.add_argument("--kappa-ref", type=float, default=0.25, help="kappa fisso nella fase K/eps")
    p.add_argument("--k-grid", nargs="+", type=int, default=[3, 4, 5, 6])
    p.add_argument("--eps-grid", nargs="+", type=float, default=[0.01, 0.03, 0.05, 0.1])
    p.add_argument("--gatings", nargs="+", default=["double", "single"],
                   help="double = come nel paper (attenuazione quadratica); single = solo membership")
    p.add_argument("--gating-both", action="store_true", help="testa entrambi i gating anche nella fase K/eps")
    p.add_argument("--no-keps", action="store_true", help="salta la griglia K/eps (solo kappa)")
    p.add_argument("--with-bfull", action="store_true", help="include B_full (richiede training per seed)")
    p.add_argument("--out", default=str(CLEAN / "outputs_results" / "sweep_sensitivity.csv"))
    p.add_argument("--smoke", action="store_true", help="test rapido: 1 city, 1 seed, 2 kappa, no K/eps")
    a = p.parse_args()

    if a.smoke:
        a.cities = ["nyc_tist"]          # il piu' piccolo
        a.seeds = [42]
        a.backbones = ["B_blind"]
        a.kappa_grid = [0.0, 0.25]
        a.gatings = ["double"]
        a.no_keps = True
        a.out = str(CLEAN / "outputs_results" / "sweep_sensitivity_SMOKE.csv")
        print("[SMOKE] 1 city, 1 seed, 1 backbone, 2 kappa\n", flush=True)

    return run(a)


if __name__ == "__main__":
    sys.exit(main())
