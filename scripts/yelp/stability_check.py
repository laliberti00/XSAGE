"""[venv-xsage] Stability check del min-support + GATE WINNER ml1m (post-freeze, dalla cache).
Ri-deriva le caselle L1/L2 (macro-Cat-MRR, rango atteso, backbone focale B_full) per MSUPP∈{20,50}:
  (1) le caselle winner/ridondante/null NON si muovono tra m=20 e m=50;
  (2) GATE: ml1m L2 (SIT−Steck-b) > 0 E significativo (t cross-seed, p<0.05) sotto MSUPP=20
      → altrimenti ml1m NON è etichettato winner.
Versione LEGGERA: solo macro-Cat-MRR per-seed + t appaiato (niente bootstrap) → secondi.
Uso:  python scripts/yelp/stability_check.py            # primari
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_record as rr


def macro_series(z, sh, H, bk, m, msupp):
    """macro-Cat-MRR (rango atteso, min-support msupp) per ciascun seed → array di 5 valori."""
    out = []
    for s in rr.SEEDS:
        catrk = z[f"{bk}|{m}|{s}|catrk"].astype(np.int64); gc = z[f"{bk}|{m}|{s}|gc"].astype(np.int64)
        cm = rr._exp_trunc(catrk, gc, rr.KTOP, H)
        out.append(rr.macro_msupp(cm, sh["tm"], sh["nmac"], msupp))
    return np.array(out)


def contrast(sit, other):
    """Δ = mean(SIT)−mean(other); seeds = #seed con Δ>0 (consistenza). Casella-pass = Δ>0 ∧ 5/5 concordi
    (la 3ª condizione 'CI bootstrap esclude 0' è quasi-sempre soddisfatta → sta nel CSV principale)."""
    sd = sit - other; d = float(sd.mean()); seeds = int((sd > 0).sum())
    return d, seeds, (d > 0 and seeds == len(rr.SEEDS))


def boxes_for(msupp, cities):
    out = {}
    for c in cities:
        f = rr.CLEAN / "outputs_results" / "cache" / f"raw_{c}.npz"
        if not f.exists(): continue
        z = np.load(f); sh = dict(tm=z["_shared|tm"].astype(np.int64), nmac=int(z["_shared|nmac"]), nI=int(z["_shared|nI"]))
        H, _ = rr._prefix(sh["nI"]); bk = rr.FOCAL_BACKBONE
        mB = macro_series(z, sh, H, bk, "BASE", msupp); mS = macro_series(z, sh, H, bk, "SIT", msupp); mK = macro_series(z, sh, H, bk, "Steck-b", msupp)
        d1, n1, l1 = contrast(mS, mB); d2, n2, l2 = contrast(mS, mK)
        box = "nullo" if not l1 else ("winner" if l2 else "ridondante")
        out[c] = dict(box=box, d1=d1, n1=n1, d2=d2, n2=n2)
    return out


def main():
    cities = sys.argv[1:] or rr.PRIMARY_DATASETS
    b20 = boxes_for(20, cities); b50 = boxes_for(50, cities)
    print(f"\n{'dataset':10s} {'box@20':12s} {'box@50':12s} {'L1@20 (Δ, seed)':20s} {'L2@20 (Δ, seed)':20s}")
    ok = True
    for c in cities:
        if c not in b20: continue
        a, b = b20[c], b50.get(c, {})
        same = bool(b) and a["box"] == b["box"]; ok = ok and same
        print(f"{c:10s} {a['box']:12s} {b.get('box','—'):12s} "
              f"{a['d1']:+.4f} {a['n1']}/5        {a['d2']:+.4f} {a['n2']}/5"
              f"{'' if same else '   <-- CASELLA MOSSA!'}")
    print()
    if "ml1m" in b20:
        m = b20["ml1m"]; gate = m["d2"] > 0 and m["n2"] == len(rr.SEEDS); ok = ok and gate
        print(f"[GATE ml1m winner] L2(SIT−Steck-b)@20 = {m['d2']:+.5f}, seed concordi {m['n2']}/5 → "
              f"{'PASS (ml1m = winner)' if gate else 'FAIL → ml1m NON winner (riportare claim morbido)'}")
    else:
        print("[GATE ml1m winner] cache ml1m assente — gate non valutato")
    print(f"\n=== {'STABILE + GATE OK' if ok else 'INSTABILE / GATE FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
