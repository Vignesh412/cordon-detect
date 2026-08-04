"""
Generates a SYNTHETIC per-example score CSV that approximately reproduces
the aggregate AUCs from your own Table I — NOT your real experiment data.

This exists to give cordon's evaluate feature (`cordon-evaluate` /
`cordon.evaluate_routing`) something to run against right now, so you can
see the analysis working before plugging in real per-example scores.
Do not treat this file's numbers as evidence for anything. Delete it
once you have the real export.
"""

import csv
import math
import random
import statistics

# Table I, columns this script needs: struct AUC, gated AUC.
TARGET_AUC = {
    "social_engineering": {"struct": 0.67, "gated": 0.89},
    "prompt_injection":   {"struct": 0.81, "gated": 0.83},
    "data_exfiltration":  {"struct": 0.85, "gated": 0.62},
    "tool_hijacking":     {"struct": 0.85, "gated": 0.60},
    "unknown":            {"struct": 0.97, "gated": 0.92},
    "iid":                {"struct": 0.93, "gated": 0.89},
}

N_PER_CLASS = 120  # positives and negatives each, per family


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def gen_family_scores(auc_target, n, seed):
    """
    Draws pos/neg scores from separated normals such that the expected
    AUC (via the probit relationship AUC = Phi(delta / sqrt(2))) matches
    auc_target, then squashes through a sigmoid (order-preserving, so
    empirical AUC is unaffected) to land in (0, 1) like a real risk score.
    """
    rng = random.Random(seed)
    nd = statistics.NormalDist(0, 1)
    auc_target = min(max(auc_target, 0.02), 0.98)
    delta = math.sqrt(2) * nd.inv_cdf(auc_target)
    pos = [sigmoid(rng.gauss(delta / 2, 1)) for _ in range(n)]
    neg = [sigmoid(rng.gauss(-delta / 2, 1)) for _ in range(n)]
    return pos, neg


FAMILY_SEEDS = {
    "social_engineering": 101,
    "prompt_injection": 202,
    "data_exfiltration": 303,
    "tool_hijacking": 404,
    "unknown": 505,
    "iid": 606,
}


def main(out_path="example_scores_SYNTHETIC.csv"):
    rows = []
    ex_id = 0
    for family, targets in TARGET_AUC.items():
        base_seed = FAMILY_SEEDS[family]
        struct_pos, struct_neg = gen_family_scores(targets["struct"], N_PER_CLASS, seed=base_seed)
        gated_pos, gated_neg = gen_family_scores(targets["gated"], N_PER_CLASS, seed=base_seed + 1)

        for i in range(N_PER_CLASS):
            rows.append([ex_id, family, 1, round(struct_pos[i], 4), round(gated_pos[i], 4)])
            ex_id += 1
        for i in range(N_PER_CLASS):
            rows.append([ex_id, family, 0, round(struct_neg[i], 4), round(gated_neg[i], 4)])
            ex_id += 1

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "attack_family", "true_label", "struct_score", "gated_score"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {out_path}")


if __name__ == "__main__":
    main()
