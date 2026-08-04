"""
Validates the cascade-routing fix against real per-example scores.

This answers a specific question your Discussion section already raised
and left unresolved: does routing (struct decides alone when confident,
falls back to gated only when ambiguous) recover the AUC that gated
fusion's averaging gives away on structural attacks (0.60 vs struct-alone
0.85), without losing gated's advantage on social engineering (0.89)?

Needs real per-example scores to answer for real. See README.md for the
CSV format and how to export them from your original experiment code.
"""

import csv
from collections import defaultdict

# This is the point of this refactor: cascade_scores below is not a
# reimplementation that happens to agree with cordon-detect — it's the
# exact same function the shipped library uses. This validation is
# therefore testing production code, not a parallel copy of it.
#
# Requires cordon-detect installed: `pip install -e ../cordon-detect`
# (or wherever you've placed it) from this script's environment.
try:
    from cordon import batch_cascade_scores as cascade_scores
except ImportError as e:
    raise ImportError(
        "cascade_validation.py depends on the cordon-detect package for its "
        "core routing function, so the two stay provably in sync. Install "
        "it first: pip install -e /path/to/cordon-detect"
    ) from e


def compute_auc(scores, labels):
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


def load_scores_csv(path):
    """
    Expected columns: id, attack_family, true_label, struct_score, gated_score
    true_label: 1 = attack, 0 = benign
    struct_score / gated_score: continuous risk scores in [0, 1]
    """
    rows = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fam = row["attack_family"]
            rows[fam].append({
                "label": int(row["true_label"]),
                "struct": float(row["struct_score"]),
                "gated": float(row["gated_score"]),
            })
    return rows


def evaluate(rows_by_family, tau_grid=None):
    """
    For each attack family: struct-alone AUC, gated-alone AUC, and cascade
    AUC across a tau sweep. Returns a dict ready to print or export.
    """
    if tau_grid is None:
        tau_grid = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95

    results = {}
    for family, examples in rows_by_family.items():
        labels = [e["label"] for e in examples]
        struct = [e["struct"] for e in examples]
        gated = [e["gated"] for e in examples]

        struct_auc = compute_auc(struct, labels)
        gated_auc = compute_auc(gated, labels)

        cascade_by_tau = {}
        for tau in tau_grid:
            cs = cascade_scores(struct, gated, tau)
            cascade_by_tau[tau] = compute_auc(cs, labels)

        best_tau = max(cascade_by_tau, key=lambda t: cascade_by_tau[t] or -1)

        results[family] = {
            "n": len(examples),
            "struct_auc": struct_auc,
            "gated_auc": gated_auc,
            "cascade_by_tau": cascade_by_tau,
            "best_tau": best_tau,
            "best_cascade_auc": cascade_by_tau[best_tau],
        }
    return results


def print_report(results):
    print(f"{'Family':<20} {'N':>5} {'Struct':>8} {'Gated':>8} {'Cascade':>9} {'@tau':>6}")
    print("-" * 60)
    for family, r in results.items():
        print(f"{family:<20} {r['n']:>5} "
              f"{r['struct_auc']:>8.3f} {r['gated_auc']:>8.3f} "
              f"{r['best_cascade_auc']:>9.3f} {r['best_tau']:>6.2f}")
    print()
    print("Cascade column should beat gated on structural attack families")
    print("(tool hijacking, data exfiltration, unknown) while staying close")
    print("to gated on linguistic ones (social engineering, prompt injection).")
    print()
    print("NOTE: best_tau here is chosen by looking at the same data it's")
    print("evaluated on, which overstates cascade AUC. For a result you'd")
    print("report in the paper, select tau on a held-out split and evaluate")
    print("on a separate test split — same discipline CARM's own threshold")
    print("selection used (Table 5).")
