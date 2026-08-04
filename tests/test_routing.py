import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import confidence_route, batch_cascade_scores


def test_decisive_high_structural_score_skips_semantic():
    calls = []

    def semantic():
        calls.append(1)
        return 0.5

    result = confidence_route(structural_score=0.92, semantic_score_fn=semantic, tau=0.75)
    assert result.source == "structural"
    assert result.semantic_skipped is True
    assert result.effective_score == 0.92
    assert calls == []  # never invoked


def test_decisive_low_structural_score_skips_semantic():
    calls = []

    def semantic():
        calls.append(1)
        return 0.5

    result = confidence_route(structural_score=0.05, semantic_score_fn=semantic, tau=0.75)
    assert result.source == "structural"
    assert result.semantic_skipped is True
    assert calls == []


def test_ambiguous_structural_score_falls_back_to_semantic():
    def semantic():
        return 0.88

    result = confidence_route(structural_score=0.55, semantic_score_fn=semantic, tau=0.75)
    assert result.source == "semantic"
    assert result.semantic_skipped is False
    assert result.effective_score == 0.88


def test_no_semantic_fn_always_uses_structural():
    result = confidence_route(structural_score=0.55, semantic_score_fn=None, tau=0.75)
    assert result.source == "structural"
    assert result.effective_score == 0.55


def test_batch_cascade_matches_single_example_routing():
    # The batch (offline evaluation) form and the online form must agree,
    # since cascade-validation's AUC analysis uses the batch form to
    # stand in for what confidence_route would decide online.
    struct = [0.95, 0.02, 0.55, 0.55]
    semantic = [0.10, 0.90, 0.80, 0.20]
    tau = 0.75

    batch_out = batch_cascade_scores(struct, semantic, tau)
    online_out = [
        confidence_route(s, (lambda v=g: v), tau).effective_score
        for s, g in zip(struct, semantic)
    ]
    assert batch_out == online_out
