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

## Architecture

Structural veto runs on every trace, deterministically, in about
0.02ms. Semantic check — the expensive part, usually an LLM call in a
real deployment — only ever runs on the traces structure couldn't
already rule out. The animation below loops through three **real**
traces, pulled directly from `tests/test_cascade.py`, so you can see
exactly what triggers each path rather than take that on faith:

1. **Required step skipped** — `intake` hands off straight to
   `risk_assess`, skipping `policy_check` entirely. Structural veto
   catches the undeclared handoff on its own; semantic check never runs.
2. **Persuasive override language** — structurally clean (every step
   present, every handoff declared) but an observation contains
   "auto-approve" language. Structure has nothing to flag here —
   semantic check is the one that catches it.
3. **Clean run** — every step present, every handoff declared, nothing
   suspicious said. Passes both checks, allowed.

![cordon-detect: structural veto first, semantic check only when it can't decide](docs/architecture-animated.svg)

The two dashed purple boxes are offline, authoring-time paths, not
something that runs per trace: one is an alternate way to *produce*
the trace (from your existing OpenTelemetry instrumentation instead of
hand-written dicts), the other an alternate way to *draft* the
`WorkflowSchema` that structural veto checks the trace against (from
real traffic, reviewed by a human before use — see "Drafting a schema
from traces" below). Both feed into the pipeline; neither is part of
what runs on the hot path.

For a version you drive yourself — pick a trace, watch a live log,
rather than a fixed loop — open [`docs/architecture.html`](docs/architecture.html)
directly in a browser (GitHub renders it as source, not live; clone
the repo or download the file to run it).

## Project layout

```
cordon/                    the installable library
  veto.py                  hard schema rules — deterministic, no threshold
  tokenizer.py             trace -> tokens, and the parent/child call graph
  routing.py, fusion.py    continuous-score routing/fusion strategies
  cascade.py                structural-veto-first orchestration
  evaluate.py               cordon-evaluate: compare strategies on your own data
  adapters/otel.py          OpenTelemetry GenAI span -> trace format adapter
  infer.py                  draft_from_traces(): reviewable schema drafting
tests/                      product tests (pytest tests/)
examples/quickstart.py      runnable end-to-end example
docs/architecture-animated.svg  looping diagram, embedded in this README
docs/architecture.html      the same walkthrough, interactive (open in a browser)
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
pip install cordon-detect
```

To work on the library itself, clone the repository and use
`pip install -e .` instead.

## Quickstart

```
python examples/quickstart.py
```

Runs three traces through the cascade: a clean one, one with a skipped
handoff (caught structurally, semantic check never runs), and one with
clean structure but manipulative text in a tool observation (caught by
the semantic hook).

## LangGraph demonstration

The [`examples/langgraph_demo`](examples/langgraph_demo/) example runs a
purchase-approval graph and places Cordon before the simulated payment step.
It demonstrates a clean run, a structural bypass, and a semantic override
attack without requiring a model API key:

```
python -m pip install cordon-detect -r examples/langgraph_demo/requirements.txt
python examples/langgraph_demo/demo.py
```

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

Writing this by hand doesn't scale past a handful of agents. For a
starting point drafted from real traces instead, see "Drafting a
schema from traces" below.

## Drafting a schema from traces

`draft_from_traces()` builds a starting `WorkflowSchema` from a batch
of traces, instead of writing `allowed_edges` by hand for a large
graph. It is deliberately **not** "learn the schema from production
traffic and trust it" — that would silently launder whatever's in the
batch, including a trace that was itself already compromised, into
your security policy as legitimate. It returns a reviewable draft, not
something to deploy unattended:

```python
from cordon import draft_from_traces

schema, report = draft_from_traces(clean_traces, verified_clean=True)
print(report.summary())    # counts per edge/agent/tool, rare ones flagged
schema.to_json()           # check the reviewed result into your repo
```

- `verified_clean=True` is required and does nothing on its own — it's
  you asserting you've checked this batch isn't itself carrying an
  attack, not something this function can verify for you. An inferred
  schema is only as trustworthy as what it was inferred from.
- `required_agents` is inferred **strictly**: only an agent present in
  *every* trace in the batch. An agent present in most-but-not-all
  traces is deliberately not inferred as required — that's exactly the
  case a looser rule gets wrong (a rare-but-important gate silently
  downgraded to optional). It still shows up in `report` so you can add
  it by hand if it should be required.
- `allowed_edges`/`known_tools` include everything observed at least
  once; `report.rare_edges()`/`report.rare_tools()` flag anything seen
  fewer than `rare_below` times (default 2) so a single stray edge in
  the batch doesn't get baked in with the same authority as one seen
  thousands of times, unnoticed.

`WorkflowSchema.to_json()`/`.from_json()` (and `.to_dict()`/`.from_dict()`)
work on hand-written schemas too — useful for checking a schema into a
repo as reviewable data independent of drafting. See `tests/test_infer.py`
and `tests/test_schema_serialization.py`.

## Handling legitimate order variation

If your agents don't always run in the same order — e.g. two
independent checks that can happen in either sequence depending on
which upstream service responds first — a schema that only declares one
order will false-flag the other, legitimate one. This is a real gap if
you only declare the order you happened to test.

The fix doesn't need new code, just both directions declared, combined
with `required_agents`:

```python
schema = WorkflowSchema(
    allowed_edges={
        ("intake", "risk_assess"), ("intake", "compliance_check"),
        ("risk_assess", "compliance_check"), ("compliance_check", "risk_assess"),
        ("risk_assess", "approval"), ("compliance_check", "approval"),
    },
    known_tools={...},
    required_agents={"intake", "risk_assess", "compliance_check", "approval"},
)
```

Both orders now pass. Importantly, this doesn't quietly open a bypass —
broadening the edges to permit reordering does NOT also permit skipping
one of the two checks entirely, because `required_agents` independently
checks that every required agent appeared *somewhere*, regardless of
which edges were used to get there. Skipping `compliance_check` and
going straight `risk_assess -> approval` is still caught, even though
that exact edge is legal (it has to be, to support the other valid
order) — just caught by a different rule than the edge check. See
`tests/test_cascade.py`'s reordering tests for this verified directly,
including the case where a check runs *after* approval already
happened — present in the trace, but too late to have gated anything —
which is still caught, via the edge check this time.

**What this doesn't solve, and structural checking fundamentally can't:**
if the *correct decision* actually depends on execution order — e.g.
`compliance_check` should apply a different policy when it runs first
and doesn't yet have `risk_assess`'s output — no amount of schema
configuration can express that, because it's a data-dependency
question, not a control-flow question. That's a genuine boundary of
what this kind of structural veto can check, not a gap to be closed
with more edges.

## Concurrent / parallel agents

The bidirectional-edges trick above handles *reordering* — the same
agents, one order or the other. Genuine concurrency (an orchestrator
spawning two agents that run in parallel, with no defined order between
them at all) is a distinct case, and it's handled structurally rather
than by declaring every possible interleaving: `allowed_edges` isn't
checked against a flattened sequence, it's checked against the real
parent → child call graph. Give each step an `"id"` and the id of the
step that spawned it as `"parent_id"`, and two children sharing one
`parent_id` produce two edges (parent → each child) and — importantly —
no edge between the children themselves, because they never actually
handed off to each other:

```python
trace = [
    {"type": "tool", "agent": "orchestrator", "tool": "dispatch", "args": {},
     "id": "s1", "parent_id": None},
    {"type": "tool", "agent": "risk_assess", "tool": "score", "args": {},
     "id": "s2", "parent_id": "s1"},
    {"type": "tool", "agent": "compliance_check", "tool": "check", "args": {},
     "id": "s3", "parent_id": "s1"},
]

schema = WorkflowSchema(
    allowed_edges={("orchestrator", "risk_assess"), ("orchestrator", "compliance_check")},
    known_tools={"dispatch", "score", "check"},
)
```

Both fan-out orderings pass with no wildcard edges declared, and a
child that wasn't declared for that parent is still caught, same as
any other unauthorized edge. If a trace has no `"id"`/`"parent_id"` on
any step, nothing changes — the graph falls back to treating each step
as the child of the one before it, which is exactly the old flat-sequence
behavior. See `cordon.agent_call_graph()` and `tests/test_graph.py`.

## Integrations

`cordon.adapters.otel.from_otel_spans()` converts OpenTelemetry GenAI
spans into cordon's trace format, carrying real span parent/child
linkage through as `id`/`parent_id` (see above) so concurrent agent
fan-out is represented correctly rather than flattened into a fake
sequence. It's dependency-free — spans are read duck-typed, so plain
dicts (an OTLP/JSON export) and SDK-style span objects both work
without installing `opentelemetry` itself:

```python
from cordon import WorkflowSchema, run, tokenize
from cordon.adapters.otel import from_otel_spans

trace = from_otel_spans(spans)  # spans from your tracer/exporter
result = run(trace, schema)
```

GenAI semantic-convention attribute names (`gen_ai.agent.name`,
`gen_ai.tool.name`, ...) are still evolving upstream; if your
instrumentation uses different keys, pass overrides
(`agent_attr=`, `tool_attr=`, `tool_args_attr=`, `operation_attr=`)
rather than forking the adapter. See `cordon/adapters/otel.py`'s
docstring for the exact span → step mapping, including the one real
caveat: a span that's reasoning/chat rather than a tool call (no
`gen_ai.operation.name == "execute_tool"`) becomes a passthrough
`"observation"` step and — matching cordon's existing tool-call-centric
model — won't itself register as a graph node or satisfy
`required_agents`. If you need an orchestrator's own dispatch step to
anchor edges to its children, give it a real tool-call span, not just
an `invoke_agent` one. See `tests/test_otel_adapter.py`.

## Trace format

```python
trace = [
    {"type": "tool", "agent": "intake", "tool": "extract", "args": {...}},
    {"type": "observation", "content": "..."},
    ...
]
```

Supported `type` values: `sys`, `user`, `assistant`, `tool`, `observation`,
`output`, `error`. Every step accepts optional `agent` (who acted) and,
for `"tool"` steps, `tool`/`args`. Two more optional fields — `id` and
`parent_id` — enable the real call-graph behavior described above;
omit both and cordon behaves exactly as if they didn't exist.

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
