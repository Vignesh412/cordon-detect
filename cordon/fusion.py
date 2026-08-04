"""
Pluggable fusion strategies for combining a structural score and a
semantic score into one fused risk score.

Why this exists: the trace-experiment (research/trace-experiment) showed
that plain averaging's advantage was a threshold accident, not a real
property. When the structural signal is silent (0) and semantic is
confident (e.g. 0.85), a 50/50 average lands at 0.425 — under most
thresholds, that's an attack that goes uncaught, for no principled
reason. Different mathematical combinations fail differently, and which
one is right depends on your data — that's why this is a menu you
evaluate against your own labeled examples (see evaluate.py /
cordon-evaluate), not a single hardcoded formula.

IMPORTANT — read before using confidence_cascade with a rule-based veto:

confidence_cascade treats a LOW structural score as "confidently benign"
and skips the semantic check entirely. That's only correct if your
structural score comes from something trained on both classes, where a
genuinely low score reflects real evidence of benign-ness (this is what
the IRAI paper's struct classifier does). It is NOT correct if your
structural "score" is really a rule-based veto's 0/1 output reinterpreted
as a float — there, 0 means "no known rule was violated", not "confirmed
safe", and a lot of real attacks (anything purely semantic, like social
engineering) will structurally look like 0 by construction. Skipping
semantic in that case means never catching them. For rule-based veto,
use cascade.run() (veto.py + cascade.py) instead, which always falls
through to the semantic check unless a hard rule actually fired — see
that module's docstring. confidence_cascade here is for genuine
continuous structural scorers only.
"""

from typing import Callable

FusionFn = Callable[[float, float], float]


def weighted_average(struct_weight: float = 0.5) -> FusionFn:
    """The gated-fusion baseline. Dilutes whichever signal is more
    confident whenever the other is silent — this is the behavior the
    other strategies exist to avoid."""
    def fn(structural: float, semantic: float) -> float:
        return struct_weight * structural + (1 - struct_weight) * semantic
    return fn


def max_fusion(structural: float, semantic: float) -> float:
    """Trust whichever signal is more alarmed. Never dilutes a confident
    signal, at the cost of always needing both scores computed (no
    early-exit cost saving)."""
    return max(structural, semantic)


def noisy_or_fusion(structural: float, semantic: float) -> float:
    """Treats both scores as independent probabilities of attack and
    combines them as P(at least one is right): 1 - (1-s)(1-g). Similar
    non-dilution property to max_fusion, smoother near the boundary."""
    return 1 - (1 - structural) * (1 - semantic)


def confidence_cascade(tau: float = 0.75) -> FusionFn:
    """Trust the structural score outright when decisive in either
    direction; otherwise defer entirely to semantic. See the module
    docstring above for when this is and isn't the right choice — this
    is the only strategy here with a real early-exit cost saving,
    because it's the only one where the semantic score is allowed to
    not matter at all for a given example."""
    def fn(structural: float, semantic: float) -> float:
        if structural >= tau or structural <= (1 - tau):
            return structural
        return semantic
    return fn


# Registry used by evaluate.py / cordon-evaluate to compare strategies
# side by side against real labeled data, rather than assuming one wins.
BUILTIN_STRATEGIES: dict[str, FusionFn] = {
    "naive_average": weighted_average(0.5),
    "max": max_fusion,
    "noisy_or": noisy_or_fusion,
    "cascade_tau_0.60": confidence_cascade(0.60),
    "cascade_tau_0.75": confidence_cascade(0.75),
    "cascade_tau_0.90": confidence_cascade(0.90),
}
