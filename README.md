# cordon-detect

A small library that watches what an AI agent pipeline *does* — not what it
*says* — and flags execution that doesn't match the shape you declared.

See [`SUMMARY.md`](SUMMARY.md) for the problem/mechanism/validation
narrative — this README is usage docs; that's the short version of why
this exists and what the evidence for it actually is.

```python
from cordon import WorkflowSchema, run

schema = WorkflowSchema(
    allowed_edges={("intake", "policy_check"), ("policy_check", "risk_assess")},
    known_tools={"extract", "validate", "score"},
)

result = run(trace, schema)
if result.blocked:
    print(result.reason)
```

## Project layout

```
cordon/                    the installable library
  veto.py                  hard schema rules — deterministic, no threshold
  routing.py, fusion.py    continuous-score routing/fusion strategies
  cascade.py                structural-veto-first orchestration
  evaluate.py               cordon-evaluate: compare strategies on your own data
tests/                      product tests (pytest tests/)
examples/quickstart.py      runnable end-to-end example
research/
  trace-experiment/         real generated traces, emergent (not preset) results —
                             the evidence behind the fusion-strategy guidance below
  cascade-validation/       worked example of cordon-evaluate against synthetic
                             scores shaped like the IRAI paper's own Table I
  llm_semantic_check.py     reference implementation wiring a real LLM into the
                             semantic_check slot — request-format-verified,
                             judgment-quality untested (needs your own API key)
CHANGELOG.md                 what changed and why, including the scorer's
                             debugging history
```

## What it does

Most agent traces are safe, and checking whether they're safe should be
cheap. `cordon-detect` runs a fast, deterministic structural check first —
does the call graph match what you declared, is every tool one you've
recognized before — and only escalates to a slower semantic check (an LLM
call, a classifier, whatever you already have) when structure alone can't
rule something out.

Two consequences of checking structure first:

- **It catches attack patterns it's never seen a signature for.** A
  structural check doesn't ask "does this match a known attack" — it asks
  "have I ever declared this edge or this tool." A genuinely novel attack
  still produces an unfamiliar shape.
- **Most traffic never touches the expensive check.** In the included
  example, a structurally-invalid trace resolves in ~0.02ms with the
  semantic step skipped entirely; only ambiguous traces — clean structure,
  suspicious content — pay for the slower check.

## Two ways to route, depending on what "structural" means for you

**Hard schema rules** (`WorkflowSchema` + `run()`, in `veto.py`) — you
declare legal agent-to-agent edges and known tools; any trace outside
that is a deterministic violation, blocked outright, no threshold
involved. Use this when "structural" means "did the execution graph
match what I declared."

**Continuous structural risk scores** (`confidence_route()` /
`fusion.py`'s strategies) — for when structural risk itself comes from a
model or heuristic that outputs a score, not a yes/no rule. See "Which
fusion strategy" below before picking one — the obvious default
(averaging) is the one that actually fails.

## What it doesn't do

On purpose, to keep this small and dependency-free:

- No bundled semantic/LLM model. You provide that callback; Cordon only
  decides when to call it. `research/llm_semantic_check.py` is a
  reference implementation if you want a real one.
- No policy resolution, no severity scoring, no cross-agent blast-radius
  reasoning. This is a detection signal, not a governance decision —
  wire the result into whatever policy layer, logging, or gateway you
  already run.
- No framework integration shipped yet. The trace format is a plain list
  of dicts, so it drops into a LangGraph callback, an MCP tool-call hook,
  or a custom logger without needing Cordon to know about any of them.

## Install

```
pip install -e .
```

(not yet published to PyPI — clone and install locally)

## Quickstart

```
python examples/quickstart.py
```

Runs three traces through the cascade: a clean one, one with a skipped
handoff (caught structurally, semantic check never runs), and one with
clean structure but manipulative text in a tool observation (caught by
the semantic hook).

## Declaring a schema

```python
from cordon import WorkflowSchema

schema = WorkflowSchema(
    allowed_edges={
        ("intake", "policy_check"),
        ("policy_check", "risk_assess"),
        ("risk_assess", "approval"),
    },
    known_tools={"extract", "validate", "score", "decide"},
    required_agents={"intake", "policy_check", "risk_assess", "approval"},
)
```

- `allowed_edges` — legitimate agent-to-agent handoffs. A call sequence
  producing an edge outside this set is flagged.
- `known_tools` — tools Cordon has been told about. Leave empty to skip
  this check entirely.
- `required_agents` — agents that must appear somewhere in a complete
  trace. Catches silent omission, not just wrong order.

## Trace format

```python
trace = [
    {"type": "tool", "agent": "intake", "tool": "extract", "args": {...}},
    {"type": "observation", "content": "..."},
    ...
]
```

Supported `type` values: `sys`, `user`, `assistant`, `tool`, `observation`,
`output`, `error`.

## Plugging in a semantic check

```python
from cordon import SemanticResult

def my_semantic_check(trace, tokens):
    # tokens is the structural token list — tok.content gives you
    # observation/output text if you want to inspect it.
    if "auto-approve" in " ".join(t.content or "" for t in tokens):
        return SemanticResult(flagged=True, reason="override language", confidence=0.8)
    return SemanticResult(flagged=False)

result = run(trace, schema, semantic_check=my_semantic_check)
```

If you don't pass `semantic_check`, Cordon runs structural-only — still
useful on its own, since most attack patterns show up in execution shape
before they show up in language. For a real (not keyword-based) semantic
check, see `research/llm_semantic_check.py`.

## Continuous score routing

If you have a structural risk model that outputs a score rather than a
hard rule violation:

```python
from cordon import confidence_route

def my_expensive_semantic_check():
    # only called when structural score is ambiguous
    return call_llm_judge(trace)

result = confidence_route(
    structural_score=my_structural_model(trace),  # 0.0-1.0
    semantic_score_fn=my_expensive_semantic_check,
    tau=0.75,  # tune against your own held-out data
)
print(result.source, result.effective_score, result.semantic_skipped)
```

For offline evaluation over a dataset where you already have both scores
precomputed:

```python
from cordon import batch_cascade_scores

effective = batch_cascade_scores(struct_scores, semantic_scores, tau=0.75)
```

## Which fusion strategy should you actually use?

Don't assume — `research/trace-experiment/` ran `naive_average`, `max`,
`noisy_or`, and `confidence_cascade` (at several tau values) against
real generated traces, not preset scores, and found a genuine split:

- **Naive averaging can look fine on AUC while having zero real recall**
  at the actual decision threshold — it ranks attacks correctly but
  squashes every score toward the boundary, so it never actually crosses
  a block threshold. AUC alone hides this; `cordon-evaluate` reports
  both AUC and recall-at-threshold specifically because of what this
  experiment found.
- **Confidence-threshold cascading can silently never call the semantic
  check at all** if your structural score is a rule-based veto's 0/1
  output rather than a trained probability — a constant "no violation
  found" (0) gets treated as "confirmed safe," which is a different,
  wrong claim.
- **`max_fusion` and `noisy_or_fusion`** were the only strategies that
  held up on both AUC and real threshold recall, across every category
  tested, with three different versions of the semantic scorer as it
  improved — see `research/trace-experiment/README.md` for the full
  numbers and the debugging story behind them.

Run the same comparison against your own data before trusting any of
this for your own structural/semantic signals — that's what
`cordon-evaluate` is for:

```
cordon-evaluate your_labeled_scores.csv
```

Expects a CSV with columns `group, true_label, structural_score,
semantic_score` (all four names configurable via flags — see
`cordon-evaluate --help`).

```python
from cordon import load_labeled_csv, evaluate_groups, print_report

rows = load_labeled_csv("your_labeled_scores.csv")
results = evaluate_groups(rows)
print_report(results)
```

## Tests

```
pip install pytest
pytest tests/ -v
```

## Research

- `research/trace-experiment/` — real generated traces, cordon's actual
  code, emergent results. The evidence behind the fusion guidance above.
- `research/cascade-validation/` — the same `cordon-evaluate` feature
  applied to synthetic scores shaped like the IRAI paper's Table I; see
  its README for how to point it at real experiment data instead.
- `research/llm_semantic_check.py` — reference implementation for a real
  LLM-backed semantic check, verified request-format-correct against the
  live API, not yet verified for judgment quality.

See `CHANGELOG.md` for what changed across versions and why, including
the semantic scorer's full debugging history.

## License

Apache-2.0 — see `LICENSE`.