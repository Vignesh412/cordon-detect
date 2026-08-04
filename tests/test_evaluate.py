import sys
import os
import csv
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import compute_auc, evaluate_routing, evaluate_groups, load_labeled_csv
from cordon.evaluate import main as cli_main


def test_compute_auc_perfect_separation():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert compute_auc(scores, labels) == 1.0


def test_compute_auc_perfect_inversion():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    assert compute_auc(scores, labels) == 0.0


def test_compute_auc_chance():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert compute_auc(scores, labels) == 0.5


def test_evaluate_routing_cascade_recovers_structural_when_structural_dominates():
    # structural score is a near-perfect signal, semantic score is noise —
    # the best strategy should land close to structural-alone, clearly
    # beating naive_average, which is exactly the failure mode this
    # feature exists to catch.
    n = 60
    labels = [1] * n + [0] * n
    structural = [0.95] * n + [0.05] * n
    semantic = [0.5, 0.51] * n  # ~chance

    result = evaluate_routing(structural, semantic, labels, group="structural_dominant")
    assert result.structural_auc > 0.95
    assert result.best_strategy_auc > 0.9
    assert result.strategy_auc["naive_average"] <= result.best_strategy_auc


def test_evaluate_routing_reports_all_builtin_strategies():
    n = 20
    labels = [1] * n + [0] * n
    structural = [0.8] * n + [0.2] * n
    semantic = [0.7] * n + [0.3] * n

    result = evaluate_routing(structural, semantic, labels)
    assert "naive_average" in result.strategy_auc
    assert "max" in result.strategy_auc
    assert "noisy_or" in result.strategy_auc
    assert result.best_strategy in result.strategy_auc


def test_constant_structural_score_breaks_confidence_cascade_not_max_or_avg():
    # This is the exact failure mode found against the real trace
    # experiment: when structural is constant (0.0 for every example,
    # both classes), confidence_cascade always treats it as "decisive
    # low" and never consults semantic — AUC collapses to chance (0.5).
    # avg/max/noisy_or all correctly reduce to a monotonic function of
    # semantic alone and preserve its AUC exactly.
    n = 40
    labels = [1] * n + [0] * n
    structural = [0.0] * (2 * n)  # constant — no structural signal at all
    semantic = [0.9] * n + [0.1] * n  # perfectly separates on its own

    result = evaluate_routing(structural, semantic, labels)
    assert result.strategy_auc["cascade_tau_0.75"] == 0.5
    assert result.strategy_auc["max"] == result.semantic_auc
    assert result.strategy_auc["noisy_or"] == result.semantic_auc
    assert result.strategy_auc["naive_average"] == result.semantic_auc


def test_threshold_metrics_present_and_bounded():
    n = 20
    labels = [1] * n + [0] * n
    structural = [0.8] * n + [0.2] * n
    semantic = [0.7] * n + [0.3] * n

    result = evaluate_routing(structural, semantic, labels)
    for name, metrics in result.strategy_at_threshold.items():
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["fpr"] <= 1.0


def test_evaluate_groups_from_csv_roundtrip():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "true_label", "structural_score", "semantic_score"])
        for i in range(30):
            writer.writerow(["groupA", 1, 0.9, 0.5])
            writer.writerow(["groupA", 0, 0.1, 0.5])
        path = f.name

    try:
        rows = load_labeled_csv(path)
        assert "groupA" in rows
        assert len(rows["groupA"]) == 60
        results = evaluate_groups(rows)
        assert results["groupA"].structural_auc == 1.0
    finally:
        os.unlink(path)


def test_cli_runs_end_to_end(capsys):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "true_label", "structural_score", "semantic_score"])
        for i in range(20):
            writer.writerow(["demo", 1, 0.9, 0.6])
            writer.writerow(["demo", 0, 0.1, 0.4])
        path = f.name

    try:
        exit_code = cli_main([path])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "demo" in captured.out
        assert "Struct" in captured.out
    finally:
        os.unlink(path)


def test_cli_custom_column_names():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack_family", "label", "struct", "sem"])
        for i in range(20):
            writer.writerow(["tool_hijacking", 1, 0.9, 0.5])
            writer.writerow(["tool_hijacking", 0, 0.1, 0.5])
        path = f.name

    try:
        exit_code = cli_main([
            path,
            "--group-col", "attack_family",
            "--label-col", "label",
            "--structural-col", "struct",
            "--semantic-col", "sem",
        ])
        assert exit_code == 0
    finally:
        os.unlink(path)
