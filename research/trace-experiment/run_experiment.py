"""
Runs cordon-detect's REAL, installed veto/cascade logic against the
generated traces, and reports whatever comes out. Nothing here is preset.

Requires cordon-detect installed: pip install -e . from the repo root.

Compares four ways of deciding attack-or-not per trace:
  structural_only — cordon.veto.check alone
  semantic_only   — the heuristic scorer alone, thresholded at 0.5
  naive_blend     — averages a 0/1 structural signal with the semantic
                    score and thresholds at 0.5 (stands in for what
                    gated-fusion-style averaging does to a strong
                    structural signal)
  cascade         — cordon.cascade.run(), the real shipped mechanism:
                    structural veto decides alone when it fires; the
                    semantic scorer is only invoked when structure
                    doesn't already flag something

Reports recall per attack category (did we catch it) and false positive
rate on benign traces (did we wrongly flag it) for each method, plus how
many times each method actually invoked the semantic scorer — the
concrete cost-saving number, measured, not asserted.
"""

import random
import sys
from collections import defaultdict

from cordon import WorkflowSchema, tokenize, structural_check as cordon_veto_check
from cordon import run as cordon_cascade_run, SemanticResult

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from generate_traces import generate_dataset, AGENTS, TOOLS
import semantic_scorer


SCHEMA = WorkflowSchema(
    allowed_edges={
        ("intake", "policy_check"),
        ("policy_check", "risk_assess"),
        ("risk_assess", "approval"),
    },
    known_tools=set(TOOLS.values()),
    required_agents=set(AGENTS),
)


def evaluate_dataset(dataset, seed=7):
    rng = random.Random(seed)  # drives the semantic scorer's noise term

    semantic_calls = {"semantic_only": 0, "naive_blend": 0, "cascade": 0}

    # predictions[method][category] -> list of (predicted_attack: bool, true_label: int)
    predictions = defaultdict(lambda: defaultdict(list))

    for example in dataset:
        trace = example["trace"]
        label = example["label"]
        category = example["category"]
        tokens = tokenize(trace)

        struct_result = cordon_veto_check(tokens, SCHEMA)
        struct_flag = struct_result.flagged

        # --- structural_only ---
        predictions["structural_only"][category].append((struct_flag, label))

        # --- semantic_only (always calls the scorer) ---
        sem_score = semantic_scorer.score_trace(trace, rng)
        semantic_calls["semantic_only"] += 1
        predictions["semantic_only"][category].append((sem_score >= 0.5, label))

        # --- naive_blend (always calls the scorer, then averages) ---
        semantic_calls["naive_blend"] += 1
        blended = 0.5 * (1.0 if struct_flag else 0.0) + 0.5 * sem_score
        predictions["naive_blend"][category].append((blended >= 0.5, label))

        # --- cascade: cordon's actual shipped run() function ---
        def sem_check(trace, tokens, _rng=rng):
            semantic_calls["cascade"] += 1
            s = semantic_scorer.score_trace(trace, _rng)
            return SemanticResult(flagged=(s >= 0.5), confidence=s)

        cascade_result = cordon_cascade_run(trace, SCHEMA, semantic_check=sem_check)
        predictions["cascade"][category].append((cascade_result.blocked, label))

    return predictions, semantic_calls


def compute_metrics(preds_for_category):
    tp = sum(1 for p, l in preds_for_category if p and l == 1)
    fn = sum(1 for p, l in preds_for_category if not p and l == 1)
    fp = sum(1 for p, l in preds_for_category if p and l == 0)
    tn = sum(1 for p, l in preds_for_category if not p and l == 0)

    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return {"recall": recall, "fpr": fpr, "precision": precision, "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def print_report(predictions, semantic_calls, n_total):
    methods = ["structural_only", "semantic_only", "naive_blend", "cascade"]
    categories = ["tool_hijacking", "unknown", "social_engineering"]

    print(f"{'Category':<20}", end="")
    for m in methods:
        print(f"{m:>17}", end="")
    print()
    print("-" * (20 + 17 * len(methods)))

    for cat in categories:
        print(f"{cat:<20}", end="")
        for m in methods:
            metrics = compute_metrics(predictions[m][cat])
            print(f"{metrics['recall']:>16.3f} ", end="")
        print("  <- recall (attack caught rate)")

    print()
    benign_all = []
    for m in methods:
        benign_all = predictions[m]["benign"]
        metrics = compute_metrics(benign_all)
        print(f"{'benign FPR':<20}{m:>17}: {metrics['fpr']:.3f}")

    print()
    print(f"Semantic scorer calls out of {n_total} traces:")
    for m in ["semantic_only", "naive_blend", "cascade"]:
        pct = 100 * semantic_calls[m] / n_total
        print(f"  {m:<16} {semantic_calls[m]:>5} calls ({pct:.0f}%)")


if __name__ == "__main__":
    dataset = generate_dataset(n_per_category=150, seed=42)
    predictions, semantic_calls = evaluate_dataset(dataset)
    print_report(predictions, semantic_calls, len(dataset))
