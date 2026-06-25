"""Colla i risultati ALLINEATI di tutti i dataset in UNA matrice (stesso protocollo ovunque).
Legge, per ogni città: battery_bfull_<city>.csv + macro_avg_<city>.csv + neutrality_ablation_<city>.csv
(salta quelle assenti). Scrive outputs_results/aligned_matrix.csv + stampa la tabella.
Uso:  python scripts/foursquare/aligned_matrix.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean"); OUT = CLEAN / "outputs_results"
CITIES = ["mind", "ml1m", "yelp", "kuairand", "nyc_tist", "tokyo_tist", "saopaulo", "bangkok", "istanbul"]


def gm(df, bk, me, col):
    s = df[(df.backbone == bk) & (df.method == me)][col]
    return float(s.mean()) if len(s) else np.nan


def main():
    rows = []
    for c in CITIES:
        r = {"city": c}
        fb = OUT / f"battery_bfull_{c}.csv"
        if fb.exists():
            b = pd.read_csv(fb)
            r["Bblind_dCatMRR"] = round(gm(b, "B_blind", "SIT", "CatMRR") - gm(b, "B_blind", "BASE", "CatMRR"), 5)
            r["Bfull_BASE"] = round(gm(b, "B_full", "BASE", "CatMRR"), 5)
            r["Bfull_SIT"] = round(gm(b, "B_full", "SIT", "CatMRR"), 5)
            r["Bfull_dCatMRR"] = round(gm(b, "B_full", "SIT", "CatMRR") - gm(b, "B_full", "BASE", "CatMRR"), 5)
            r["Bfull_dR20"] = round(gm(b, "B_full", "SIT", "R20") - gm(b, "B_full", "BASE", "R20"), 5)
            r["Bfull_dGini"] = round(gm(b, "B_full", "SIT", "Gini") - gm(b, "B_full", "BASE", "Gini"), 5)
        fm = OUT / f"macro_avg_{c}.csv"
        if fm.exists():
            m = pd.read_csv(fm); d = m[m.method == "SIT_minus_BASE"]
            if len(d):
                r["dominant"] = float(d.dominant_share.iloc[0])
                r["Bblind_dMICRO"] = float(d.micro.iloc[0]); r["Bblind_dMACRO"] = float(d.macro.iloc[0])
        fn = OUT / f"neutrality_ablation_{c}.csv"
        if fn.exists():
            n = pd.read_csv(fn)
            full = n[n["mode"] == "full"]; rb = n[n["mode"] == "raw_both"]
            if len(full) and len(rb):
                r["neut_full_dMacro"] = float(full.dMacro.iloc[0]); r["neut_rawboth_dMacro"] = float(rb.dMacro.iloc[0])
                surv = (rb.dMacro.iloc[0] > 0) and (rb.lens_KLratio.iloc[0] >= 0.6 * full.lens_KLratio.iloc[0])
                r["neut_verdict"] = "STRUTT" if surv else "FEATURE"
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "aligned_matrix.csv", index=False)
    print("\n===== MATRICE ALLINEATA (stesso protocollo su tutti i dataset) =====")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print(df.to_string(index=False))
    print("\nLegenda: dominant=saturazione; Bblind_dMACRO=equità-categoria di SIT; "
          "Bfull_dCatMRR=SIT enhancer su B_full; neut_verdict=valore strutturale vs feature-coupled.")
    print("→ outputs_results/aligned_matrix.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
