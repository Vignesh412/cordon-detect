# trace-experiment

The honest version of the fusion-strategy question: real generated
traces with genuine variability, cordon's actual veto/scoring code run
against them, and results that emerged rather than being set in advance.

## Files

- `generate_traces.py` — ~150 traces per category, randomized content.
- `semantic_scorer.py` — v3: exact-phrase matching + negation-scope
  detection + compositional (action+control) matching. See its
  docstring for the full debugging history — five real bugs across two
  fixes, each caught by a test, not by inspection.
- `test_semantic_scorer.py` — run after any change to the scorer.
- `run_experiment.py` — discrete comparison: structural-only,
  semantic-only, naive-blend, cordon's real `cascade.run()`.
- `export_scores.py` — exports continuous per-example scores for
  `cordon-evaluate`.
- `../llm_semantic_check.py` — reference implementation wiring a real
  LLM into cordon's semantic_check slot. Verified request-format-correct
  against the live API (a fake key gets a real 401 auth error, not a
  malformed-request error); the actual semantic judgment quality is
  untested — this environment has no API key.

## Run it

```
pytest test_semantic_scorer.py -v
python export_scores.py
cordon-evaluate real_trace_scores.csv --group-col category \
    --structural-col structural_score --semantic-col semantic_score \
    --label-col true_label
```

## Current results (scorer v3: negation + compositional matching)

```
Group                   N  Struct    Sem     avg     max    n-or  c@0.60  c@0.75  c@0.90
social_engineering    300   0.500  0.989   0.989   0.989   0.989   0.500   0.500   0.500
tool_hijacking         300   1.000  0.395   1.000   1.000   1.000   1.000   1.000   1.000
unknown                300   1.000  0.541   1.000   1.000   1.000   1.000   1.000   1.000

Recall at decision threshold:
social_engineering     0.013   0.987   0.987   0.000   0.000   0.000
tool_hijacking          1.000   1.000   1.000   1.000   1.000   1.000
unknown                 1.000   1.000   1.000   1.000   1.000   1.000
```

Progression across every fix in this repo, on the same real dataset:

| | benign FPR | social_engineering recall (semantic-alone) |
|---|---|---|
| v1 (exact phrase, no negation) | 0.313 | 0.767 (partly inflated by the same bug causing the FPR) |
| v2 (+ negation, 3 sub-fixes) | 0.000 | 0.667 |
| v3 (+ compositional matching, 1 sub-fix) | 0.000 | 0.987 |

**And across every version, the fusion-strategy finding hasn't moved at
all:** `avg` and `confidence_cascade` are still at essentially zero
recall on social engineering, regardless of how good the underlying
semantic model gets. That's not a coincidence — it's the point. A better
model doesn't fix bad fusion math (averaging squashes a confident lone
signal no matter how confident); bad fusion math throws away a good
model's output no matter how good the model is. Only `max_fusion` and
`noisy_or_fusion` pass both the AUC and real-recall tables, with every
version of the scorer tried here.

## What's still open

The compositional matcher has a real ceiling: a paraphrase sharing no
vocabulary with either word cluster will still be missed — this isn't
fixable by extending word lists further, it's the structural limit of
any keyword-based approach. `../llm_semantic_check.py` is the actual
next step for closing that gap, using real semantic understanding
instead of pattern matching — verified correct against the API contract,
not yet verified for judgment quality, since that needs a real key this
environment doesn't have.
