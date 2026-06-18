"""Main-results pass on the clean repo: produce the base-table accuracy and
LT@20 cells for 3 configs (B_blind, B_full, X-SAGE unified λ=1 κ=1) × 5 cities.

Output: outputs_results/main_results.csv (no SD — pure point estimate).
For the full ± SD bootstrap table use scripts/sweep_lambda_kappa.py and the
existing dossier scripts in the original repo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xsage import pipeline


def main():
    rows = []
    for city in pipeline.D.DEFAULT_CITIES:
        prep = pipeline.prepare(city)
        sinks = pipeline.resolve_sinks(city, prep)
        out = pipeline.score_request_metrics(prep, sinks, kappa=1.0, lam=1.0)
        agg = pipeline.aggregate(out)
        meta = agg["__meta__"]
        for cfg in ("blind", "full", "xsage"):
            rows.append({
                "city": city, "config": cfg,
                "R20": agg[cfg]["R20"], "N20": agg[cfg]["N20"],
                "LT20": agg[cfg]["LT20"],
                "n_test": meta["n_test"],
                "n_sinks": len(meta["sink_ids"]),
                "touch_share": meta["touch_share"],
            })
        print(f"[{city}] sinks={sinks} touch={meta['touch_share']:.1%}  "
                f"xsage R={agg['xsage']['R20']:.4f} LT={agg['xsage']['LT20']:.4f}")
    df = pd.DataFrame(rows)
    OUT = ROOT / "outputs_results"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "main_results.csv", index=False)
    print(f"\n→ {OUT/'main_results.csv'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
