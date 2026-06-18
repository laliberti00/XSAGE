"""Reproduction: the unified combiner in the clean repo must reproduce the
numbers of the dossier (round4-multicity @ 5119f2f).

Targets (X-SAGE = unified term λ=1, κ=1, sink defaults 1.5/0.05):
    NYC-TIST   R@20=0.0907  N@20=0.0381  LT@20=0.0974
    Tokyo-TIST identity = B_blind (0 sinks)

Tolerance: 1e-4 absolute on accuracy, 1e-3 on LT@20 (sensitive to top-K
tiebreaks under float32).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xsage import pipeline


TARGETS = {
    "nyc_tist": {"R20": 0.0907, "N20": 0.0381, "LT20": 0.0974},
    "tokyo_tist": {"R20": 0.0747, "N20": 0.0340, "LT20": 0.0233},
}
TOL_ACC = 1e-4
TOL_LT = 1e-3


def reproduce_one(city: str):
    prep = pipeline.prepare(city)
    sinks = pipeline.resolve_sinks(city, prep)
    out = pipeline.score_request_metrics(prep, sinks, kappa=1.0, lam=1.0)
    agg = pipeline.aggregate(out)
    return agg, sinks


def test_reproduce_nyc_tist():
    agg, sinks = reproduce_one("nyc_tist")
    print(f"  NYC-TIST sinks={sinks}  touch={agg['__meta__']['touch_share']:.1%}")
    print(f"    X-SAGE R={agg['xsage']['R20']:.4f}  N={agg['xsage']['N20']:.4f}  "
            f"LT={agg['xsage']['LT20']:.4f}")
    print(f"    blind  R={agg['blind']['R20']:.4f}  N={agg['blind']['N20']:.4f}  "
            f"LT={agg['blind']['LT20']:.4f}")
    t = TARGETS["nyc_tist"]
    assert abs(agg["xsage"]["R20"] - t["R20"]) < TOL_ACC, \
        f"NYC R@20 {agg['xsage']['R20']:.4f} vs {t['R20']:.4f}"
    assert abs(agg["xsage"]["N20"] - t["N20"]) < TOL_ACC, \
        f"NYC N@20 {agg['xsage']['N20']:.4f} vs {t['N20']:.4f}"
    assert abs(agg["xsage"]["LT20"] - t["LT20"]) < TOL_LT, \
        f"NYC LT@20 {agg['xsage']['LT20']:.4f} vs {t['LT20']:.4f}"


def test_reproduce_tokyo_identity():
    agg, sinks = reproduce_one("tokyo_tist")
    print(f"  Tokyo sinks={sinks}  touch={agg['__meta__']['touch_share']:.1%}")
    # Tokyo: no sinks ⇒ X-SAGE = B_blind exactly
    assert sinks == [], f"Tokyo should have no sinks, got {sinks}"
    assert agg["xsage"]["R20"] == agg["blind"]["R20"]
    assert agg["xsage"]["N20"] == agg["blind"]["N20"]
    assert agg["xsage"]["LT20"] == agg["blind"]["LT20"]


if __name__ == "__main__":
    test_reproduce_nyc_tist()
    print("NYC OK")
    test_reproduce_tokyo_identity()
    print("Tokyo OK")
