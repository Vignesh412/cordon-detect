"""
Evaluate and tune fusion strategy against your own labeled data. This is
the "actually check it" half of what a fusion-formula claim needs — not
just a mechanism (fusion.py) but a way to find out, on your own examples,
which formula wins and by how much.

Given labeled examples with a structural score and a semantic score each,
reports structural-alone AUC, semantic-alone AUC, and AUC under every
registered fusion strategy — per group (attack type, category, whatever
grouping is meaningful for your data) — so you pick a formula based on
evidence, not a default we picked for you.
"""

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .fusion import BUILTIN_STRATEGIES, confidence_cascade, FusionFn


def compute_auc(scores, labels) -> Optional[float]:
    """Rank-based (Mann-Whitney) ROC-AUC. No sklearn dependency."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


@dataclass
class GroupEvaluation:
    group: str
    n: int
    structural_auc: Optional[float]
    semantic_auc: Optional[float]
    strategy_auc: dict = field(default_factory=dict)   # {strategy_name: auc}
    strategy_at_threshold: dict = field(default_factory=dict)  # {strategy_name: {"recall":, "fpr":}}
    best_strategy: Optional[str] = None
    best_strategy_auc: Optional[float] = None


def _recall_and_fpr_at_threshold(scores, labels, threshold: float) -> dict:
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
    tn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 0)
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {"recall": recall, "fpr": fpr}


def evaluate_routing(
    structural_scores: list[float],
    semantic_scores: list[float],
    labels: list[int],
    strategies: Optional[dict] = None,
    group: str = "all",
    decision_threshold: float = 0.5,
) -> GroupEvaluation:
    """
    Core single-group evaluation. Computes structural-alone AUC,
    semantic-alone AUC, and AUC under every fusion strategy in
    `strategies` (defaults to fusion.BUILTIN_STRATEGIES) — plus, since
    AUC alone can hide a strategy that ranks correctly but squashes
    every score toward the decision boundary (this is exactly what
    naive averaging does — see fusion.py), recall and false-positive
    rate at a fixed decision_threshold for each strategy too. A strategy
    can have strong AUC and still fail in practice if it never actually
    crosses whatever threshold you block on.
    """
    if strategies is None:
        strategies = BUILTIN_STRATEGIES

    structural_auc = compute_auc(structural_scores, labels)
    semantic_auc = compute_auc(semantic_scores, labels)

    strategy_auc = {}
    strategy_at_threshold = {}
    for name, fn in strategies.items():
        fused = [fn(s, g) for s, g in zip(structural_scores, semantic_scores)]
        strategy_auc[name] = compute_auc(fused, labels)
        strategy_at_threshold[name] = _recall_and_fpr_at_threshold(fused, labels, decision_threshold)

    valid = {k: v for k, v in strategy_auc.items() if v is not None}
    best_strategy = max(valid, key=lambda k: valid[k]) if valid else None
    best_auc = strategy_auc[best_strategy] if best_strategy else None

    return GroupEvaluation(
        group=group,
        n=len(labels),
        structural_auc=structural_auc,
        semantic_auc=semantic_auc,
        strategy_auc=strategy_auc,
        strategy_at_threshold=strategy_at_threshold,
        best_strategy=best_strategy,
        best_strategy_auc=best_auc,
    )


def evaluate_groups(rows_by_group: dict, strategies: Optional[dict] = None,
                     decision_threshold: float = 0.5) -> dict:
    """
    rows_by_group: {group_name: [{"label": 0|1, "structural": float, "semantic": float}, ...]}
    Returns {group_name: GroupEvaluation}.
    """
    results = {}
    for group, examples in rows_by_group.items():
        labels = [e["label"] for e in examples]
        structural = [e["structural"] for e in examples]
        semantic = [e["semantic"] for e in examples]
        results[group] = evaluate_routing(structural, semantic, labels, strategies,
                                           group=group, decision_threshold=decision_threshold)
    return results


def load_labeled_csv(path: str, group_col: str = "group",
                      structural_col: str = "structural_score",
                      semantic_col: str = "semantic_score",
                      label_col: str = "true_label") -> dict:
    """
    Load a CSV of labeled examples, grouped by `group_col`. Column names
    are configurable so this reads whatever your existing evaluation
    pipeline already exports, not a fixed schema Cordon dictates.
    true_label: 1 = positive/attack, 0 = negative/benign.
    """
    rows = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = row[group_col]
            rows[group].append({
                "label": int(row[label_col]),
                "structural": float(row[structural_col]),
                "semantic": float(row[semantic_col]),
            })
    return dict(rows)


def format_report(results: dict) -> str:
    # Collect strategy names in a stable order from the first result.
    any_result = next(iter(results.values()), None)
    strategy_names = list(any_result.strategy_auc.keys()) if any_result else []

    short = {
        "naive_average": "avg",
        "max": "max",
        "noisy_or": "n-or",
    }

    def col_label(name):
        return short.get(name, name.replace("cascade_tau_", "c@"))

    header = f"{'Group':<20}{'N':>5}{'Struct':>8}{'Sem':>7}"
    for name in strategy_names:
        header += f"{col_label(name):>8}"
    header += f"  {'best':<18}"

    lines = [header, "-" * len(header)]
    for group, r in results.items():
        line = f"{group:<20}{r.n:>5}{r.structural_auc:>8.3f}{r.semantic_auc:>7.3f}"
        for name in strategy_names:
            v = r.strategy_auc.get(name)
            line += f"{v:>8.3f}" if v is not None else f"{'--':>8}"
        line += f"  {r.best_strategy or '--':<18}"
        lines.append(line)

    lines.append("")
    lines.append("avg = naive weighted average (the gated-fusion baseline that dilutes)")
    lines.append("max/n-or = non-diluting combinations; c@X = confidence cascade at tau=X")
    lines.append("")
    lines.append("'best' picks the strategy with highest AUC on THIS data — that's an")
    lines.append("in-sample choice and will overstate the winner's true performance.")
    lines.append("For a number you'd report externally, select the strategy and any")
    lines.append("tau on a held-out split, then evaluate on a separate test split.")

    # Second table: recall/FPR at the actual decision threshold. AUC is
    # threshold-free — it can rate a strategy highly even if that
    # strategy squashes every score so far toward the boundary that it
    # never actually crosses a real block threshold. This table is what
    # catches that; a strategy that looks great above and terrible here
    # is not safe to deploy at the threshold you tested.
    any_result = next(iter(results.values()), None)
    if any_result and any_result.strategy_at_threshold:
        lines.append("")
        lines.append("Recall at decision threshold (does the fused score actually cross")
        lines.append("the block threshold on real attacks — not just rank them correctly):")
        lines.append("")
        header2 = f"{'Group':<20}"
        for name in strategy_names:
            header2 += f"{col_label(name):>8}"
        lines.append(header2)
        lines.append("-" * len(header2))
        for group, r in results.items():
            line = f"{group:<20}"
            for name in strategy_names:
                rec = r.strategy_at_threshold.get(name, {}).get("recall")
                line += f"{rec:>8.3f}" if rec is not None else f"{'--':>8}"
            lines.append(line)

    return "\n".join(lines)


def print_report(results: dict) -> None:
    print(format_report(results))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="cordon-evaluate",
        description="Compare fusion strategies against your own labeled examples.",
    )
    parser.add_argument("csv_path", help="CSV of labeled examples")
    parser.add_argument("--group-col", default="group")
    parser.add_argument("--structural-col", default="structural_score")
    parser.add_argument("--semantic-col", default="semantic_score")
    parser.add_argument("--label-col", default="true_label")
    parser.add_argument("--tau-sweep", action="store_true",
                         help="Replace the 3 preset cascade taus with a finer 0.50-0.95 sweep")
    parser.add_argument("--decision-threshold", type=float, default=0.5,
                         help="Threshold for the recall/FPR-at-threshold table (default 0.5)")
    args = parser.parse_args(argv)

    rows = load_labeled_csv(
        args.csv_path,
        group_col=args.group_col,
        structural_col=args.structural_col,
        semantic_col=args.semantic_col,
        label_col=args.label_col,
    )

    strategies = dict(BUILTIN_STRATEGIES)
    if args.tau_sweep:
        strategies = {k: v for k, v in strategies.items() if not k.startswith("cascade_tau_")}
        tau = 0.5
        while tau <= 0.95 + 1e-9:
            strategies[f"cascade_tau_{tau:.2f}"] = confidence_cascade(tau)
            tau += 0.05

    results = evaluate_groups(rows, strategies=strategies, decision_threshold=args.decision_threshold)
    try:
        print_report(results)
    except BrokenPipeError:
        sys.stderr.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
