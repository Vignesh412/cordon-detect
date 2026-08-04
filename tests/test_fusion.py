import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon.fusion import weighted_average, max_fusion, noisy_or_fusion, confidence_cascade, BUILTIN_STRATEGIES


def test_naive_average_dilutes_a_confident_signal_against_silence():
    # This is the exact failure mode from the trace-experiment: a
    # confident semantic score (0.85) paired with a silent structural
    # score (0.0) should NOT survive a 50/50 average above a 0.5
    # decision threshold — that's the bug these other strategies fix.
    avg = weighted_average(0.5)
    fused = avg(0.0, 0.85)
    assert fused < 0.5


def test_max_fusion_never_dilutes():
    assert max_fusion(0.0, 0.85) == 0.85
    assert max_fusion(0.9, 0.1) == 0.9
    assert max_fusion(0.0, 0.0) == 0.0


def test_noisy_or_never_dilutes_below_either_input():
    # noisy-or should never rate the fused risk below either individual
    # input — combining two pieces of evidence shouldn't look safer
    # than either piece of evidence alone.
    fused = noisy_or_fusion(0.0, 0.85)
    assert fused >= 0.85
    fused2 = noisy_or_fusion(0.6, 0.6)
    assert fused2 >= 0.6


def test_confidence_cascade_matches_routing_module_behavior():
    fn = confidence_cascade(0.75)
    assert fn(0.9, 0.1) == 0.9      # decisive high -> trust structural
    assert fn(0.05, 0.6) == 0.05    # decisive low -> trust structural
    assert fn(0.55, 0.85) == 0.85   # ambiguous -> defer to semantic


def test_builtin_strategies_registry_has_expected_entries():
    assert "naive_average" in BUILTIN_STRATEGIES
    assert "max" in BUILTIN_STRATEGIES
    assert "noisy_or" in BUILTIN_STRATEGIES
    assert any(k.startswith("cascade_tau_") for k in BUILTIN_STRATEGIES)
    # Every strategy should be callable as (structural, semantic) -> float
    for name, fn in BUILTIN_STRATEGIES.items():
        result = fn(0.5, 0.5)
        assert isinstance(result, float)
