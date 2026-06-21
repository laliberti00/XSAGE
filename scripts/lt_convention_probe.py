"""Probe: LT@20 per-USER vs per-REQUEST on 5 cities × 3 configs.

Quick A/B on the same per-request 'lt' array: per-request micro-mean vs
per-user macro-mean. Numbers feed the convention-decision discussion.
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
        u = out["__meta__"]["u_test"]
        uniq, inv = np.unique(u, return_inverse=True)
        n_users = len(uniq)
        n_req = out["__meta__"]["n_test"]
        # how many requests per user (variation)
        cnt = np.bincount(inv, minlength=n_users)
        for cfg in ("blind", "full", "xsage"):
            lt = out[cfg]["lt"]
            lt_req = float(lt.mean())                # per-request micro
            # per-user: aggregate then mean of means
            s = np.zeros(n_users); c = np.zeros(n_users)
            np.add.at(s, inv, lt.astype(np.float64))
            np.add.at(c, inv, 1)
            lt_user = float((s / c).mean())
            rows.append({"city": city, "config": cfg,
                              "LT_per_request": lt_req,
                              "LT_per_user": lt_user,
                              "diff_user_minus_request": lt_user - lt_req,
                              "n_req": n_req, "n_users": n_users,
                              "req_per_user_p50": int(np.median(cnt)),
                              "req_per_user_p95": int(np.percentile(cnt, 95))})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"
                              if isinstance(x, float) else str(x)))
    OUT = ROOT / "outputs_results"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lt_convention_probe.csv", index=False)
    print(f"\n→ {OUT/'lt_convention_probe.csv'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
