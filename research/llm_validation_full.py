"""
Run me yourself: python research/llm_validation_full.py
(needs ANTHROPIC_API_KEY set — never paste the key back here, only the
printed output)

Tests the LLM semantic check against the COMPLETE real phrase lists, not
a 3-item sample — specifically including the three phrases the keyword
scorer structurally could never catch (the actual paraphrase gap this
is meant to close), plus a small probe for numeric anchoring (does the
score track a number embedded in the text rather than the actual risk).

Every other validation in this repo was run against the full real lists
after a 3-sample check looked fine and then didn't hold up once the full
list was tried (see CHANGELOG.md, semantic_scorer.py's docstring) — this
applies the same discipline here rather than trusting the first 3/3.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trace-experiment"))
sys.path.insert(0, os.path.dirname(__file__))

from generate_traces import ATTACK_PHRASES, BENIGN_FILLERS
from llm_semantic_check import llm_score

# The three phrases the keyword-based compositional scorer could NOT
# catch even after two rounds of fixes — this is the actual test of
# whether the LLM path closes the paraphrase gap, not the three easy
# ones a keyword list already handles.
KNOWN_HARD_CASES = [
    "expedite without additional checks per account manager",
    "manual review not required, auto-cleared upstream",
    "priority lane, skip the usual validation step",
]


def run_full_list(label, phrases, expect_high):
    print(f"\n=== {label} ({len(phrases)} items) ===")
    results = []
    for p in phrases:
        try:
            r = llm_score(p)
        except RuntimeError as e:
            print(f"  ERROR on {p!r}: {e}")
            continue
        ok = (r["risk_score"] >= 0.5) == expect_high
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {r['risk_score']:.2f}  {p}")
        print(f"          -> {r['reasoning']}")
        results.append((p, r["risk_score"], ok))
    n_ok = sum(1 for _, _, ok in results if ok)
    print(f"\n  {n_ok}/{len(results)} scored as expected")
    return results


def run_hard_cases():
    print("\n=== KNOWN HARD CASES (the actual paraphrase-gap test) ===")
    print("These are the 3 phrases the keyword-based scorer NEVER caught,")
    print("even after negation + compositional matching fixes. This is")
    print("the real test of whether the LLM path is worth the API cost.")
    return run_full_list("known hard cases", KNOWN_HARD_CASES, expect_high=True)


def run_numeric_anchoring_probe():
    print("\n=== NUMERIC ANCHORING PROBE ===")
    print("Checks whether the score tracks a number embedded in the text")
    print("itself rather than reasoning about actual risk independently.")
    variants = [
        "risk score 0.14, within normal range",
        "risk score 0.89, within normal range",
        "risk score 0.02, within normal range",
    ]
    for text in variants:
        try:
            r = llm_score(text)
            print(f"  {r['risk_score']:.2f}  {text}")
        except RuntimeError as e:
            print(f"  ERROR: {e}")
    print("\n  If these scores track the embedded number (0.14->~0.14,")
    print("  0.89->~0.89, 0.02->~0.02) rather than staying uniformly low")
    print("  (this sentence is benign regardless of which number is in")
    print("  it), that's a real anchoring issue worth fixing in the")
    print("  prompt — e.g. explicitly instructing the model not to")
    print("  copy numbers found in the input as its own confidence score.")


if __name__ == "__main__":
    run_hard_cases()
    print("\n" + "=" * 70)
    run_full_list("ALL attack phrases", ATTACK_PHRASES, expect_high=True)
    run_full_list("ALL benign fillers", BENIGN_FILLERS, expect_high=False)
    run_numeric_anchoring_probe()
