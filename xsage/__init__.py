"""X-SAGE (clean) — aligned to the Approach.

  s = s_B + κ · m_sel · γ_S(v) · Σ_k r_k · (b̃^(k)_c + λ · b^LT_c)

Layer modules:
  l0_sensing         — strictly-causal per-request inputs
  l1_perception      — context-state c̃ and intent vector e
  l2_comprehension   — rough k-means → situations + boundary set T(v)
  l3_projection      — situation-transition T, boundary disambiguation
  recommendation     — unified additive combiner (the formula above)
  sinks              — explicit (kl_mult, lt_gap)-thresholded sink rule
  pipeline           — load + score one city under any (λ, κ)
  data               — light parquet + artefact loader

Only the named, paper-used pieces from round4-multicity @ 5119f2f are kept;
legacy variants (harmonic combiner, alternative intent/transit modes) live
in the original repo.
"""

__all__ = [
    "l0_sensing", "l1_perception", "l2_comprehension", "l3_projection",
    "recommendation", "sinks", "pipeline", "data", "metrics",
]
