"""[venv-xsage] Analisi condizionale R4#9: X-SAGE vs profilo statico (Steck-b), DENTRO e FUORI dal prior utente.

Testa la frase di §5.3 ("the same user heads toward different genres at different moments, and this
within-user, time-varying component is precisely what a static profile cannot represent"), che oggi e'
affermata ma non misurata: il confronto pubblicato e' aggregato su TUTTE le richieste di test.

PARTIZIONE (leakage-free: Pu e' costruito SOLO da df_train, come in results_record.py ~riga 238)
  ON-PRIOR   : icm[i_test] == argmax(Pu[u_test])   (la categoria vera e' la dominante dell'utente)
  OFF-PRIOR  : il complemento

=============================== PRE-REGISTRAZIONE (fissata PRIMA di guardare i numeri) ==============
H1. Sulle richieste OFF-PRIOR, X-SAGE (SIT) supera il profilo statico (Steck-b) su macro-Cat-MRR.

PASS(city, backbone, group) :=  mean_seed(delta) > 0
                            AND CI95 bootstrap del delta (seed 42, 1500 resample a livello di RICHIESTA,
                                protocollo di boot_paired) esclude lo zero
                            AND #seed con delta>0 == 5                      [= regola `passpos` del paper]
PASS(city, group)           :=  PASS vale sulla MAGGIORANZA dei backbone valutati (>=3 su 4)
H1 CONFERMATA               :=  PASS(city, OFF-PRIOR) su ALMENO 3 dei 5 dataset

Attesa di controllo (NON e' il test): su ON-PRIOR il profilo statico dovrebbe essere avanti. Se X-SAGE
fosse avanti anche li', la partizione non misura cio' che crediamo e va ricontrollata prima di
interpretare qualunque cosa.
Esito negativo -> si riporta come tale e la frase di §5.3 sul within-user va ammorbidita. NON si cercano
partizioni alternative dopo aver visto i numeri.
=====================================================================================================

GATE DI ANCORAGGIO (§5, obbligatorio, PRIMA di interpretare): ricombinando ON+OFF la macro-Cat-MRR deve
coincidere a 4 decimali con outputs_results/results_record.csv (metric=macroCatMRR, method in {SIT,Steck-b}):
  per-seed 42 -> colonna mean_s42 ; media 5 seed -> colonna mean. Se il gate non passa: STOP.

Uso:
  python scripts/validation/conditional_prior.py --smoke              # nyc_tist, seed 42, B_blind
  python scripts/validation/conditional_prior.py                      # run completa (4 backbone x 5 seed x 5 dataset)
  python scripts/validation/conditional_prior.py --from-cache         # idem, dagli array grezzi gia' cachati
  python scripts/validation/conditional_prior.py --with-bfull         # include B_full (richiede training)
Out: outputs_results/conditional_prior.csv (incrementale) + conditional_prior_composition.csv
"""
import sys, os, argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN / "scripts" / "yelp"))
import results_record as RR          # riuso: build_descriptor, select_K/eps, fit_*, per_request_eval, macro_msupp

MSUPP, BOOT, KTOP = RR.MSUPP, RR.BOOT, RR.KTOP
SEEDS = RR.SEEDS
CITIES = ["ml1m", "nyc_tist", "saopaulo", "yelp", "kuairand"]
BACKBONES = ["B_blind", "EASE", "AFM", "SASRec"]          # score in cache; B_full solo con --with-bfull
GROUPS = ["ON-PRIOR", "OFF-PRIOR"]
OUT = CLEAN / "outputs_results" / "conditional_prior.csv"
OUT_COMP = CLEAN / "outputs_results" / "conditional_prior_composition.csv"
# tolleranza del gate: "coincide a 4 decimali" = mezza unita' sul 4o decimale. NB: results_record.csv
# memorizza i valori gia' arrotondati a 5 decimali, quindi confrontare round(v,4)==round(ref,4) sbaglia
# quando il valore vero cade sul bordo di arrotondamento (es. 0.14225): si confrontano le DIFFERENZE.
GATE_TOL = 5e-5
COLS = ["city", "seed", "backbone", "group", "n_requests", "n_cats_supported",
        "macroCatMRR_sit", "macroCatMRR_steck", "delta", "ci_lo", "ci_hi", "kappa_sit", "kappa_steck"]


# ---------------------------------------------------------------- macro-Cat-MRR (punto + bootstrap)
def macro_pt(cm, tm, nmac):
    """Punto: usa ESATTAMENTE macro_msupp di results_record.py (nessuna reimplementazione)."""
    return RR.macro_msupp(cm, tm, nmac, MSUPP)


def _macro_fast(cm, tm, nmac, m):
    """Equivalente vettoriale di macro_msupp, solo per il loop bootstrap (verificato contro macro_pt)."""
    s = np.bincount(tm, weights=cm, minlength=nmac); c = np.bincount(tm, minlength=nmac)
    ok = c >= max(m, 1)
    return float((s[ok] / c[ok]).mean()) if ok.any() else 0.


def n_supported(tm, nmac):
    return int((np.bincount(tm, minlength=nmac) >= max(MSUPP, 1)).sum())


def boot_delta(cm_sit, cm_stb, tm, nmac, rng):
    """CI95 del delta SIT-Steck-b: ricampionamento appaiato a livello di RICHIESTA (protocollo boot_paired,
    ramo macroCatMRR: il supporto per categoria viene RICALCOLATO dentro ogni resample)."""
    n = len(cm_sit); bd = np.empty(BOOT)
    for b in range(BOOT):
        ix = rng.integers(0, n, n); t = tm[ix]
        bd[b] = _macro_fast(cm_sit[ix], t, nmac, MSUPP) - _macro_fast(cm_stb[ix], t, nmac, MSUPP)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- sorgenti dei cm per-richiesta
def kappas(city, seed, bk):
    csvb = pd.read_csv(CLEAN / f"outputs_results/battery_bfull_{city}.csv")

    def kget(mth):
        r = csvb[(csvb.seed == seed) & (csvb.backbone == bk) & (csvb.method == mth)]["kstar"]
        if not len(r):
            r = csvb[(csvb.seed == 42) & (csvb.backbone == bk) & (csvb.method == mth)]["kstar"]
        return float(r.iloc[0])
    return kget("SIT"), kget("Steck-b")


def cells_from_cache(city, seeds, backbones):
    """Array grezzi per-richiesta gia' prodotti da results_record.py (cache/raw_<city>.npz)."""
    z = np.load(CLEAN / f"outputs_results/cache/raw_{city}.npz")
    u = z["_shared|u"].astype(np.int64); tm = z["_shared|tm"].astype(np.int64)
    Pu = z["_shared|Pu"].astype(np.float64); nmac = int(z["_shared|nmac"]); nI = int(z["_shared|nI"])
    H, _ = RR._prefix(nI)
    cnt = None                                        # conteggi grezzi non cachati: fallback uniforme -> riga costante
    out = {}
    for seed in seeds:
        for bk in backbones:
            try:
                cm = {}
                for mth in ("SIT", "Steck-b"):
                    cr = z[f"{bk}|{mth}|{seed}|catrk"].astype(np.int64); gc = z[f"{bk}|{mth}|{seed}|gc"].astype(np.int64)
                    cm[mth] = RR._exp_trunc(cr, gc, KTOP, H)
                out[(seed, bk)] = cm
            except KeyError:
                continue
    return dict(u=u, tm=tm, Pu=Pu, cnt=cnt, nmac=nmac, nI=nI, cells=out)


def cells_recompute(city, seeds, backbones, with_bfull, dev):
    """Ricalcolo dalla pipeline (build_descriptor -> rough-k-means -> bias -> per_request_eval),
    identico al loop principale di results_record.run_city (righe ~225-260), senza B_full salvo richiesta."""
    bdir = CLEAN / "data" / city / "backbone"
    out = {}; shared = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        D0 = RR.build_descriptor(city, splits=("train", "val", "test"))
        ds = D0["ds"]; nmac = D0["n_macros"]; icm = D0["icm"]; excl = D0["excl"]; sb = D0["sb"]
        cmt = D0["cmt"]; G1 = D0["G1"].astype(np.float32); nI = int(ds["n_items"])
        vtr, vva, vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
        K = RR.select_K(vtr, int(D0["attractors"].sum()) + 2, rng); eps = RR.select_eps(vtr, vva, K)
        fit = RR.fit_rough_kmeans(vtr, K=K, eps=eps, seed=seed, max_iter=80); z_tr = fit.core_label.astype(np.int64)
        _, kte, compte, isbte = RR._assign(vte, fit.prototypes, eps)
        mem_te = RR.membership_from_assign(kte, compte, isbte, K)
        gam_te = 1.0 / np.maximum(compte.sum(1), 1).astype(np.float32)
        b_z = RR.fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=RR.ALPHA)
        nudge = mem_te.astype(np.float32) @ b_z
        dft = ds["df_test"]; ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
        nU = int(ds["n_users"]); umac = ds["df_train"]["u_idx"].values.astype(np.int64)
        b_z_user = RR.fit_situation_biases_z(umac, cmt, nU, nmac, alpha=RR.ALPHA); nudge_stb = b_z_user[ute]
        cnt = np.zeros((nU, nmac)); np.add.at(cnt, (umac, cmt), 1.)          # conteggi grezzi: cold-start = riga nulla
        Pu = cnt.copy(); Pu[Pu.sum(1) == 0] = 1.; Pu /= Pu.sum(1, keepdims=True)
        bft = RR.bfull_scores(ds, icm, excl, nmac, dev, seed) if (with_bfull and "B_full" in backbones) else None
        print(f"  [{city}] seed {seed}: K={K} eps={eps}", flush=True)
        for bk in backbones:
            if bk == "B_blind": sfn = (lambda idx, u=ute: sb[u[idx]])
            elif bk == "B_full":
                if bft is None: continue
                sfn = (lambda idx: bft[idx])
            else:
                fu = bdir / f"{bk}.scores_user.npy"; ft = bdir / f"{bk}.scores_test.npy"
                if fu.exists(): M = np.load(fu, mmap_mode="r"); sfn = (lambda idx, M=M, u=ute: M[u[idx]])
                elif ft.exists(): Mt = np.load(ft, mmap_mode="r"); sfn = (lambda idx, Mt=Mt: Mt[idx])
                else: continue
            k_sit, k_stb = kappas(city, seed, bk)
            ev_s = RR.per_request_eval(sfn, nudge, k_sit, gam_te, ute, ite, icm, excl, G1, nmac)
            ev_b = RR.per_request_eval(sfn, nudge_stb, k_stb, gam_te, ute, ite, icm, excl, G1, nmac)
            out[(seed, bk)] = {"SIT": ev_s["cm"], "Steck-b": ev_b["cm"]}
        if not shared:
            shared = dict(u=ute, tm=icm[ite].astype(np.int64), Pu=Pu, cnt=cnt, nmac=nmac, nI=nI)
    shared["cells"] = out
    return shared


# ---------------------------------------------------------------- run per citta'
def run_city(city, seeds, backbones, src, with_bfull, dev, rec):
    D = cells_from_cache(city, seeds, backbones) if src == "cache" else \
        cells_recompute(city, seeds, backbones, with_bfull, dev)
    u, tm, Pu, nmac = D["u"], D["tm"], D["Pu"], D["nmac"]
    cnt = D.get("cnt")

    # --- partizione + diagnostica (§6.4 pareggi, §6.5 utenti senza storia)
    dom = Pu.argmax(1)
    if cnt is not None: cold_user = cnt.sum(1) == 0
    else: cold_user = np.isclose(Pu, 1.0 / nmac).all(1)          # fallback: riga uniforme
    cold_req = cold_user[u]
    tied = (np.isclose(Pu, Pu.max(1)[:, None])).sum(1) > 1
    tu = np.unique(u); tied_pct = float(tied[tu].mean() * 100)
    on = (tm == dom[u]) & ~cold_req
    off = (tm != dom[u]) & ~cold_req
    print(f"  [{city}] partizione: ON={int(on.sum())} OFF={int(off.sum())} "
          f"| utenti-senza-storia-in-train: {int(cold_user.sum())} (richieste escluse: {int(cold_req.sum())}) "
          f"| utenti in pareggio su argmax(Pu): {int(tied[tu].sum())}/{len(tu)} ({tied_pct:.1f}%)"
          f"{'  <-- >5%, da riportare' if tied_pct > 5 else ''}", flush=True)

    # --- composizione per categoria (§6.3: si riporta, non si corregge)
    comp = []
    for gname, gmask in (("ON-PRIOR", on), ("OFF-PRIOR", off), ("ALL", np.ones_like(on))):
        c = np.bincount(tm[gmask], minlength=nmac).astype(float)
        for k in range(nmac):
            comp.append(dict(city=city, group=gname, cat=k, n=int(c[k]), share=round(float(c[k] / max(c.sum(), 1)), 5)))

    # --- GATE DI ANCORAGGIO (§5): ON u OFF u esclusi deve ricomporre i totali pubblicati
    gate_rows = []; gate_ok = True
    for bk in backbones:
        full = {m: [] for m in ("SIT", "Steck-b")}
        for seed in seeds:
            if (seed, bk) not in D["cells"]: continue
            for m in ("SIT", "Steck-b"):
                full[m].append((seed, macro_pt(D["cells"][(seed, bk)][m], tm, nmac)))
        for m in ("SIT", "Steck-b"):
            if not full[m]: continue
            pub = rec[(rec.dataset == city) & (rec.backbone == bk) & (rec.metric == "macroCatMRR") & (rec.method == m)]
            if not len(pub): continue
            for seed, v in full[m]:
                if seed == 42:
                    ref = float(pub["mean_s42"].iloc[0]); ok = abs(v - ref) <= GATE_TOL
                    gate_ok &= ok; gate_rows.append((bk, m, "s42", v, ref, ok))
            if len(full[m]) == len(SEEDS):
                v = float(np.mean([x[1] for x in full[m]])); ref = float(pub["mean"].iloc[0])
                ok = round(v, 4) == round(ref, 4); gate_ok &= ok; gate_rows.append((bk, m, "mean5", v, ref, ok))
    worst = max((abs(v - ref) for _, _, _, v, ref, _ in gate_rows), default=0.)
    for bk, m, kind, v, ref, ok in gate_rows:
        if not ok:
            print(f"    [gate] {city:9s} {bk:8s} {m:8s} {kind:6s} ricomposto={v:.6f} pubblicato={ref:.6f} "
                  f"|diff|={abs(v - ref):.2e} *** MISMATCH ***", flush=True)
    print(f"    [gate] {city}: {sum(1 for r in gate_rows if r[5])}/{len(gate_rows)} celle OK "
          f"(ON+OFF ricompone i totali pubblicati) — max |diff| = {worst:.2e} (tol {GATE_TOL:.0e})", flush=True)
    if not gate_ok:
        raise SystemExit(f"GATE DI ANCORAGGIO FALLITO su {city}: l'analisi condizionale non ricompone i "
                         f"totali pubblicati -> non interpretabile. STOP (§5).")

    # --- righe per (seed, backbone, group)
    rows = []
    for seed in seeds:
        for bk in backbones:
            if (seed, bk) not in D["cells"]: continue
            cm_s = D["cells"][(seed, bk)]["SIT"]; cm_b = D["cells"][(seed, bk)]["Steck-b"]
            k_sit, k_stb = kappas(city, seed, bk)
            for gname, gmask in (("ON-PRIOR", on), ("OFF-PRIOR", off)):
                cs, cb, t = cm_s[gmask], cm_b[gmask], tm[gmask]
                ms, mb = macro_pt(cs, t, nmac), macro_pt(cb, t, nmac)
                assert abs(ms - _macro_fast(cs, t, nmac, MSUPP)) < 1e-12, "macro_fast != macro_msupp"
                lo, hi = boot_delta(cs, cb, t, nmac, np.random.default_rng(2024))
                rows.append(dict(city=city, seed=seed, backbone=bk, group=gname,
                                 n_requests=int(gmask.sum()), n_cats_supported=n_supported(t, nmac),
                                 macroCatMRR_sit=round(ms, 6), macroCatMRR_steck=round(mb, 6),
                                 delta=round(ms - mb, 6), ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                                 kappa_sit=k_sit, kappa_steck=k_stb))
        # scrittura INCREMENTALE dopo ogni seed
        if rows:
            df = pd.DataFrame(rows)[COLS]
            df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False); rows = []
    pd.DataFrame(comp).to_csv(OUT_COMP, mode="a", header=not OUT_COMP.exists(), index=False)
    return dict(city=city, tied_pct=tied_pct, n_cold=int(cold_user.sum()), n_cold_req=int(cold_req.sum()),
                n_on=int(on.sum()), n_off=int(off.sum()))


# ---------------------------------------------------------------- decisione pre-registrata
def summarize(df, backbones):
    print("\n" + "=" * 100)
    print("RIEPILOGO — delta = macro-Cat-MRR(X-SAGE) − macro-Cat-MRR(profilo statico), DENTRO ciascun gruppo")
    print("(i due gruppi NON sono confrontabili fra loro: §6.2)")
    print("=" * 100)
    decision = {}
    for g in GROUPS:
        print(f"\n### {g}")
        print(f"{'dataset':10s} {'backbone':9s} {'n_req':>7s} {'#cat':>5s} {'X-SAGE':>8s} {'statico':>8s} "
              f"{'delta':>9s} {'CI95(s42)':>20s} {'seed+':>6s} {'PASS':>5s}")
        for city in sorted(df.city.unique(), key=lambda c: CITIES.index(c) if c in CITIES else 99):
            npass = 0; nbk = 0
            for bk in backbones:
                s = df[(df.city == city) & (df.backbone == bk) & (df.group == g)]
                if not len(s): continue
                nbk += 1
                dmean = float(s.delta.mean()); spos = int((s.delta > 0).sum()); nseed = len(s)
                r42 = s[s.seed == 42]
                lo = float(r42.ci_lo.iloc[0]) if len(r42) else np.nan
                hi = float(r42.ci_hi.iloc[0]) if len(r42) else np.nan
                ok = (dmean > 0) and (lo > 0) and (spos == nseed == len(SEEDS))
                npass += int(ok)
                print(f"{city:10s} {bk:9s} {int(s.n_requests.mean()):7d} {s.n_cats_supported.mean():5.1f} "
                      f"{s.macroCatMRR_sit.mean():8.4f} {s.macroCatMRR_steck.mean():8.4f} {dmean:+9.4f} "
                      f"[{lo:+.4f},{hi:+.4f}] {spos:d}/{nseed:d}   {'PASS' if ok else '—':>5s}")
            decision[(city, g)] = (npass, nbk)
            if nbk:
                need = nbk // 2 + 1            # maggioranza dei backbone valutati (4 -> 3)
                print(f"{city:10s} {'→ dataset':9s} PASS su {npass}/{nbk} backbone (serve {need}) "
                      f"→ {'DATASET-PASS' if npass >= need else 'no'}")
    # roll-up compatto per dataset (§7): delta medio e #seed con delta>0, nei due gruppi
    print("\n### ROLL-UP per dataset (media sui backbone valutati)")
    print(f"{'dataset':10s} | {'ON-PRIOR: delta':>16s} {'seed+':>7s} {'#cat':>5s} | {'OFF-PRIOR: delta':>17s} {'seed+':>7s} {'#cat':>5s}")
    for city in sorted(df.city.unique(), key=lambda c: CITIES.index(c) if c in CITIES else 99):
        cells = []
        for g in GROUPS:
            s = df[(df.city == city) & (df.group == g)]
            npos = int((s.delta > 0).sum()); ntot = len(s)
            cells.append((s.delta.mean(), npos, ntot, s.n_cats_supported.iloc[0]))
        print(f"{city:10s} | {cells[0][0]:+16.4f} {cells[0][1]:3d}/{cells[0][2]:<3d} {cells[0][3]:5d} | "
              f"{cells[1][0]:+17.4f} {cells[1][1]:3d}/{cells[1][2]:<3d} {cells[1][3]:5d}")

    print("\n" + "=" * 100)
    print("DECISIONE sulla regola pre-registrata (fissata prima di guardare i numeri):")
    dpass = {}
    for g in GROUPS:
        dpass[g] = sorted(c for (c, gg), (np_, nb) in decision.items()
                          if gg == g and nb and np_ >= nb // 2 + 1)
        print(f"  {g:10s}: DATASET-PASS su {len(dpass[g])}/"
              f"{len([1 for (c, gg) in decision if gg == g])} dataset  {dpass[g]}")
    ok = len(dpass["OFF-PRIOR"]) >= 3
    print(f"\n  H1 (X-SAGE > profilo statico su OFF-PRIOR; maggioranza dei backbone; >=3 dataset su 5): "
          f"{'CONFERMATA' if ok else 'NON CONFERMATA'}  ({len(dpass['OFF-PRIOR'])}/5 dataset)")
    n_on = len(dpass["ON-PRIOR"])
    print(f"  Attesa di controllo su ON-PRIOR (il profilo statico dovrebbe essere avanti): "
          f"X-SAGE passa su {n_on}/5 dataset"
          f"{'  <-- ATTENZIONE: la partizione va ricontrollata (§4)' if n_on >= 3 else '  (coerente con l attesa)'}")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--with-bfull", action="store_true")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffisso sui file di output (per run di cross-check)")
    a = ap.parse_args()
    if a.tag:
        global OUT, OUT_COMP
        OUT = OUT.with_name(f"conditional_prior{a.tag}.csv")
        OUT_COMP = OUT_COMP.with_name(f"conditional_prior_composition{a.tag}.csv")
    cities = a.cities or (["nyc_tist"] if a.smoke else CITIES)
    seeds = [42] if a.smoke else SEEDS
    bks = (["B_blind"] if a.smoke else BACKBONES) + (["B_full"] if a.with_bfull else [])
    src = "cache" if a.from_cache else "recompute"
    dev = None
    if src == "recompute":
        import torch; dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for p in (OUT, OUT_COMP):                      # scrittura incrementale: parti sempre da file pulito
        if p.exists(): p.unlink()
    rec = pd.read_csv(CLEAN / "outputs_results" / "results_record.csv")
    print(f"MSUPP={MSUPP} BOOT={BOOT} KTOP={KTOP} src={src} cities={cities} seeds={seeds} backbones={bks}\n", flush=True)
    diag = []
    for city in cities:
        print(f"=== {city} ===", flush=True)
        diag.append(run_city(city, seeds, bks, src, a.with_bfull, dev, rec))
    print("\nDIAGNOSTICA PARTIZIONE")
    print(pd.DataFrame(diag).to_string(index=False))
    summarize(pd.read_csv(OUT), bks)
    print(f"\n-> {OUT}\n-> {OUT_COMP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
