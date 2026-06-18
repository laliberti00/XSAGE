"""κ=0 identity (structural): unified_combine_scores(κ=0) == scores_B exactly.

This is a unit test on the formula AND a 5-city sanity check that the
combiner short-circuits before any computation. 0 violations expected.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xsage.recommendation import unified_combine_scores
from xsage import pipeline


def test_unit_kappa0_identity_synthetic():
    rng = np.random.default_rng(0)
    B, I, K_sit, M = 32, 100, 4, 6
    scores_B = rng.normal(size=(B, I)).astype(np.float32)
    membership = rng.uniform(size=(B, K_sit)).astype(np.float32)
    b_z = rng.normal(size=(K_sit, M)).astype(np.float32)
    item_cat_macro = rng.integers(0, M, I)
    bLT = rng.integers(0, 2, I).astype(np.float32)
    sink_core = rng.integers(0, 2, B).astype(np.float32)
    gamma = np.where(sink_core > 0, 1.0, 0.5).astype(np.float32)

    s = unified_combine_scores(scores_B, membership, b_z, bLT, sink_core,
                                     item_cat_macro, kappa=0.0, lam=1.0,
                                     gamma_per_request=gamma)
    assert np.array_equal(s, scores_B.astype(np.float32))


def test_5_cities_kappa0_identity():
    """For every city, evaluating X-SAGE with κ=0 must reproduce B_blind R@20
    and N@20 exactly (zero violations)."""
    violations = []
    for city in pipeline.D.DEFAULT_CITIES:
        prep = pipeline.prepare(city)
        sinks = pipeline.resolve_sinks(city, prep)
        out = pipeline.score_request_metrics(prep, sinks, kappa=0.0, lam=1.0)
        agg = pipeline.aggregate(out)
        dR = agg["xsage"]["R20"] - agg["blind"]["R20"]
        dN = agg["xsage"]["N20"] - agg["blind"]["N20"]
        if abs(dR) > 1e-12 or abs(dN) > 1e-12:
            violations.append({"city": city, "dR": dR, "dN": dN})
        print(f"  {city:<14} dR={dR:+.2e}  dN={dN:+.2e}")
    assert not violations, f"κ=0 identity violated: {violations}"


if __name__ == "__main__":
    test_unit_kappa0_identity_synthetic()
    print("unit OK")
    test_5_cities_kappa0_identity()
    print("5-city OK")
