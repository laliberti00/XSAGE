# X-SAGE (clean) — Approach-aligned, minimal pipeline

This is the cleaned, paper-aligned subset of `round4-multicity @ 5119f2f`.
Everything that was either legacy (harmonic combiner, alternative intent /
transit modes) or scattered across experiment scripts (the *unified additive
combiner* and the *selective sink gate*) has been brought into the pipeline
module so that one function call reproduces the X-SAGE numbers of the paper.

The formula implemented as `xsage.recommendation.unified_combine_scores`:

    ŝ(u, i) = s_B + κ · m_sel(req) · γ_S(v) · Σ_k r_k · (b̃^(k)_c + λ · b^LT_c)

with
- `b̃` = z-scored shrunk-log-odds bias per (situation, macro), smoothing α
- `b^LT` = {0,1} long-tail indicator per item (cutoff at top-20% popularity)
- `m_sel` = {0,1} request-level selective gate (1 on core∩sink, 0 else)
- `γ_S(v)` = 1 on core, 1/|T(v)| on boundary
- κ, λ = scalar dosages
- κ = 0 ⇒ ŝ = s_B exactly (matched-OFF, structural)

Sink situations are identified by the explicit two-threshold rule in
`xsage.sinks.identify_sinks(stats, kl_mult=1.5, lt_gap=0.05)`.

## Layout
```
xsage/
  l0_sensing.py        L0 — strictly-causal per-request inputs (eq.split, eq.intent_proxy)
  l1_perception.py     L1 — context state c̃ (eq.1) and intent vector e (eq.5)
  l2_comprehension.py  L2 — rough k-means (Lingras–West): situations + T(v)
  l3_projection.py     L3 — situation transition T, boundary disambiguation
  recommendation.py    Unified additive combiner — the formula of the Approach
  sinks.py             Two-threshold sink rule (kl_mult, lt_gap)
  metrics.py           LT@K, KL, top-K, Gini-related primitives
  pipeline.py          One-call scoring for any (λ, κ, sink set)
  data.py              Parquet + artefact loader (reads from the source repo)
scripts/
  run_main_results.py  Point estimate for blind / full / X-SAGE on 5 cities
tests/                 4 invariants (see below)
outputs_results/       CSV/JSON results (gitignored)
```

## What is NOT in this repo (intentionally)

These pieces are in the original `round4-multicity` tree and are NOT used
by the paper-reported results:

- `harmonic_combine`, `fit_situation_biases` (the non-z-scored version) — the
  legacy harmonic combiner from rounds 1-2.
- `intent_mode ∈ {soft_topr, all}`, `intent_transit ∈ {collapse, mask}` —
  alternative L1 variants studied as ablations but not used in the main results.
- `auto_epsilon`, the (K, ε) tuning grid — not run here; the chosen (K, ε)
  per city are read from the cached Stage-A `fit.npz`.
- All round1/round2/round3 experiment scripts.

## Running

Set `XSAGE_DATA_ROOT` to the source repo (default points at
`~/Downloads/IntentAwareRS_thesis`) so the loader can read parquet + cached
backbone scores.

```bash
# install
pip install -r requirements.txt

# verify
python -m tests.test_kappa0_identity        # κ=0 ⇒ ŝ = s_B, 5/5 cities
python -m tests.test_additive_fidelity      # synthetic algebraic check
python -m tests.test_causal_split           # train_max < test_min per user
python -m tests.test_reproduction           # NYC + Tokyo match dossier

# run
python -m scripts.run_main_results
```

## Naming clarification

In the original code the Dirichlet smoothing constant of the bias was
called `lam` (default 50.0). To avoid clashing with λ (long-tail dosage),
the clean repo renames it to `alpha`. There is no behaviour change.

## Boundary projection

`xsage.l3_projection.boundary_disambiguate` now ASSERTS the boundary gate
internally (`r̃` is applied only where `is_boundary == True`). In the
original code this gating was the caller's responsibility.
