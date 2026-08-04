"""
Confidence-threshold routing between a structural score and a semantic
score, for pipelines that produce continuous risk scores (e.g. a trained
classifier) rather than hard schema violations.

This is the online (compute-semantic-only-when-needed) form of
fusion.confidence_cascade — see fusion.py's module docstring for the
full rationale, the other available strategies, and IMPORTANTLY when
confidence-threshold routing is and isn't appropriate for your
structural signal (it is not a safe default for a rule-based veto's 0/1
output reinterpreted as a score).

`veto.py` in this package is the deterministic sibling of this — a hard
schema-rule violation (unauthorized edge, unknown tool) blocks outright,
no threshold involved, because there's no ambiguity to route on. This
module is for the case where "structural risk" is itself a continuous
score rather than a yes/no rule.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from .fusion import confidence_cascade as _confidence_cascade_fn


@dataclass
class RoutedScore:
    effective_score: float
    source: str              # "structural" | "semantic"
    structural_score: float
    semantic_score: Optional[float]
    semantic_skipped: bool
    tau: float


def _is_decisive(structural_score: float, tau: float) -> bool:
    return structural_score >= tau or structural_score <= (1 - tau)


def confidence_route(
    structural_score: float,
    semantic_score_fn: Optional[Callable[[], float]] = None,
    tau: float = 0.75,
) -> RoutedScore:
    """
    Route a single example's effective risk score. Online form: calls
    semantic_score_fn on demand, only when needed — see
    fusion.confidence_cascade for the underlying rule and the offline
    (both-scores-already-known) form used by evaluate.py.

    structural_score: continuous risk score in [0, 1] from a structural
        model or heuristic.
    semantic_score_fn: zero-arg callable that computes and returns the
        semantic/linguistic score, called ONLY when structural_score
        falls in the ambiguous band (1-tau, tau) — this is where the
        cost saving happens, since this is typically the expensive
        call (an LLM, a larger classifier).
    tau: confidence threshold. See fusion.py's module docstring for
        when confidence-threshold routing is and isn't the right choice
        for your structural signal.
    """
    decisive = _is_decisive(structural_score, tau)

    if decisive or semantic_score_fn is None:
        return RoutedScore(
            effective_score=structural_score,
            source="structural",
            structural_score=structural_score,
            semantic_score=None,
            semantic_skipped=True,
            tau=tau,
        )

    semantic_score = semantic_score_fn()
    return RoutedScore(
        effective_score=semantic_score,
        source="semantic",
        structural_score=structural_score,
        semantic_score=semantic_score,
        semantic_skipped=False,
        tau=tau,
    )


def batch_cascade_scores(struct_scores, semantic_scores, tau: float = 0.75):
    """
    Vectorized offline form: given parallel lists of precomputed
    structural and semantic scores, return the effective score per
    example under the same rule as confidence_route. This is a thin
    wrapper over fusion.confidence_cascade(tau) — kept as a named
    function here since it predates fusion.py and cascade-validation
    imports it by this name; fusion.py's BUILTIN_STRATEGIES is the
    place to add new strategies, not this one.
    """
    fn = _confidence_cascade_fn(tau)
    return [fn(s, g) for s, g in zip(struct_scores, semantic_scores)]
