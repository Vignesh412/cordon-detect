"""
Runs cordon's real veto + the independent semantic scorer over the
generated traces (same as run_experiment.py) and exports per-example
scores to CSV, so cordon-evaluate's multi-strategy comparison can run
against genuinely emergent data — not the target-AUC synthetic scores
in ../cascade-validation, which turned out NOT to reproduce the
dilution failure mode at all (see README for why).
"""

import csv
import random
import sys

from cordon import WorkflowSchema, tokenize, structural_check as cordon_veto_check

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


def main(out_path="real_trace_scores.csv", n_per_category=150, seed=42):
    dataset = generate_dataset(n_per_category=n_per_category, seed=seed)
    rng = random.Random(7)  # same seed used by run_experiment.py's semantic noise

    scored = []
    for example in dataset:
        trace = example["trace"]
        tokens = tokenize(trace)
        struct_flag = cordon_veto_check(tokens, SCHEMA).flagged
        sem_score = semantic_scorer.score_trace(trace, rng)
        scored.append({
            "category": example["category"],
            "label": example["label"],
            "structural_score": 1.0 if struct_flag else 0.0,
            "semantic_score": round(sem_score, 4),
        })

    benign = [s for s in scored if s["category"] == "benign"]
    attack_categories = sorted({s["category"] for s in scored if s["category"] != "benign"})

    # Match the paper's Table I structure: each attack family evaluated
    # against the shared benign pool, not against itself. A group made
    # of only positives (or only negatives) has no AUC to compute.
    rows = []
    row_id = 0
    for category in attack_categories:
        positives = [s for s in scored if s["category"] == category]
        for s in positives + benign:
            rows.append([row_id, category, s["label"], s["structural_score"], s["semantic_score"]])
            row_id += 1

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "true_label", "structural_score", "semantic_score"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path} ({len(attack_categories)} families x "
          f"[family positives + shared benign negatives])")


if __name__ == "__main__":
    main()
