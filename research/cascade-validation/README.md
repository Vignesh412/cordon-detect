# cascade-validation

A worked example applying cordon's `evaluate_routing` / `cordon-evaluate`
feature to the question your Discussion section raised and left as
future work: does routing (structural score decides alone when
confident, falls back to a second score only when ambiguous) recover the
AUC gated fusion's averaging gives away on structural attacks (0.60 vs
struct-alone 0.85 on tool hijacking, 0.62 vs 0.85 on data exfiltration)?

There's no separate analysis script here anymore — that logic now lives
in `cordon/evaluate.py` as a real feature of the library itself, exercised
through the `cordon-evaluate` CLI (installed automatically with
`pip install -e .`) or the `evaluate_routing()` / `evaluate_groups()`
Python API. This folder just generates a labeled CSV and points the
feature at it.

## Run it now, on synthetic data

```
python generate_synthetic_scores.py
cordon-evaluate example_scores_SYNTHETIC.csv \
    --group-col attack_family \
    --structural-col struct_score \
    --semantic-col gated_score \
    --label-col true_label
```

**The numbers this prints are not evidence for anything.** They come
from a distribution built to approximately reproduce your Table I's
aggregate AUCs, not your real experiments. Delete
`example_scores_SYNTHETIC.csv` once you're running real data.

## Run it on your real data

Export one CSV with a row per test example from your original IRAI
experiment code — wherever you currently call something like
`roc_auc_score(labels, scores)` to produce Table I, add a line just
before that to dump the raw scores (both the struct-only run and the
gated-fusion run) alongside the label and attack family:

```
attack_family,true_label,struct_score,gated_score
tool_hijacking,1,0.91,0.58
tool_hijacking,0,0.12,0.34
...
```

Then either:

```
cordon-evaluate your_real_scores.csv \
    --group-col attack_family --structural-col struct_score \
    --semantic-col gated_score --label-col true_label
```

or from Python, if you want the results as objects rather than printed:

```python
from cordon import load_labeled_csv, evaluate_groups, print_report

rows = load_labeled_csv(
    "your_real_scores.csv",
    group_col="attack_family",
    structural_col="struct_score",
    semantic_col="gated_score",
)
results = evaluate_groups(rows)
print_report(results)
```

## Reading the output

| column | meaning |
|---|---|
| Structural | struct-alone AUC — should match your published Table I struct column exactly, since it's the same scores |
| Semantic | gated-fusion AUC — should match your published gated column exactly |
| Cascade | AUC under cascade routing at the best tau found on this data |
| @tau | the confidence threshold that produced that cascade AUC |

If Structural/Semantic don't reproduce your published numbers, the CSV
export has a bug — fix that before trusting the Cascade column.

## Before this goes in the paper

`cordon-evaluate`'s default tau selection scans the same data it scores,
which optimistically biases the cascade numbers. For a result you'd
report, split your test set: pick tau on a held-out slice (the `--tau-*`
flags let you narrow the sweep), then evaluate cascade AUC on a separate
slice using that fixed tau. Your own CARM paper's threshold selection
(Table 5, sweeping on a held-out 10% and evaluating on the rest) is the
exact template to follow here.
