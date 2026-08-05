# Changelog

## 0.6.0
- **Added `cordon.draft_from_traces()`** — builds a starting
  `WorkflowSchema` from a batch of traces instead of hand-writing
  `allowed_edges` for a large graph. Deliberately not "learn the
  schema from traffic and trust it": returns `(schema, report)`, where
  `report` carries per-edge/per-agent/per-tool observation counts so
  thin evidence (a rare edge seen once) is visible before being kept,
  rather than silently baked in with the same authority as something
  seen thousands of times. Requires an explicit `verified_clean=True`
  — a deliberate speed bump, not a formality, since an inferred schema
  is only as trustworthy as the traces it came from and this function
  has no way to check that itself. `required_agents` is inferred
  strictly (present in every trace, not just most) so a
  rare-but-important gate can't get silently downgraded to optional by
  a looser threshold. See README "Drafting a schema from traces" and
  `tests/test_infer.py`.
- **Added `WorkflowSchema.to_dict()`/`.to_json()`/`.from_dict()`/`.from_json()`**
  — schemas (hand-written or drafted) can now be checked into a repo as
  reviewable data instead of only existing as a Python literal. See
  `tests/test_schema_serialization.py`.

## 0.5.0
- **Structural veto now checks a real parent/child call graph, not a
  flattened sequence.** `tokenizer.agent_call_graph()` derives
  `WorkflowSchema.allowed_edges` checks from explicit `id`/`parent_id`
  linkage on trace steps when present, falling back to "previous step"
  chaining when they're absent — so every existing trace (none of which
  set these fields) behaves identically to before. This was a real gap,
  not a hypothetical one: flattening concurrent agent fan-out into a
  list invents an edge between siblings that never actually handed off
  to each other, and either false-flags legitimate parallel execution
  or forces declaring edges that don't correspond to any real handoff.
  Two children of one parent now correctly produce two parent→child
  edges and zero sibling edges. `veto.py`'s edge check was switched
  from `zip(sequence, sequence[1:])` to this graph. `required_agents`
  and `known_tools` checks are unaffected (already set-membership, not
  order-dependent). See README "Concurrent / parallel agents" and
  `tests/test_graph.py`.
- **Added `cordon.adapters.otel.from_otel_spans()`** — converts
  OpenTelemetry GenAI spans into cordon's trace format, carrying real
  span parent/child linkage through so the graph work above applies to
  actual production traces, not just hand-written ones. Dependency-free
  (duck-typed span reading — works with plain dicts or SDK-style
  objects without installing `opentelemetry`). One documented
  limitation, not fixed because it isn't a bug: a span that's
  reasoning/chat rather than a tool call doesn't register as a graph
  node, matching cordon's existing tool-call-centric model for
  `required_agents`. See README "Integrations" and
  `tests/test_otel_adapter.py`.
- Checked a review claim that `WorkflowSchema` is "brittle" for dynamic
  agent loops with legitimate order variation. Confirmed true for a
  naively-configured schema (declaring only one order really does
  false-flag the other legitimate order) — but resolved by existing
  features, not new code: declaring both edge directions plus
  `required_agents` together handles reordering without opening a
  bypass (verified: skipping a check entirely, or running it too late
  to matter, are both still caught, via `required_agents` and the edge
  check respectively). Documented in README; the genuine remaining
  limitation — order-dependent *decisions*, not just order-dependent
  *validity* — is a data-dependency question no structural schema can
  express, named explicitly rather than implied to be fixed. (Note:
  the graph work above solves the adjacent *concurrency* case; this
  reordering fix is still what you want for genuinely sequential but
  variable-order agents.)

## 0.4.0
- Added `cordon/fusion.py` — pluggable fusion strategies (naive average,
  max, noisy-or, confidence cascade) instead of one hardcoded formula.
- `cordon-evaluate` now compares all strategies side by side, and reports
  both AUC and recall-at-threshold — the two can disagree (a strategy
  can rank correctly while never crossing a real decision threshold),
  and only the threshold table catches that.
- `research/trace-experiment/` added: real generated traces, not
  target-AUC synthetic data, run through cordon's actual detection code.
  This is what surfaced that naive averaging can look fine on AUC while
  having zero real recall, and that confidence-threshold cascading fails
  silently on a rule-based veto's 0/1 output.
- `routing.py` and `fusion.py` deduplicated — `confidence_cascade` in
  `fusion.py` is now the single implementation both use.

## 0.3.0
- Added `cordon/evaluate.py` and the `cordon-evaluate` CLI — evaluating
  routing/fusion choices against labeled data became a real feature of
  the library, not a separate research script.
- Folded the standalone `cascade-validation` project into this repo as
  `research/cascade-validation/`, since it's evidence for one design
  choice in this library, not a separate product.

## 0.2.0
- Added `cordon/routing.py` — confidence-threshold routing
  (`confidence_route`, `batch_cascade_scores`) for continuous structural
  scores, as opposed to `veto.py`'s hard schema rules.

## 0.1.0
- Initial release: `WorkflowSchema`, `veto.py` (deterministic structural
  checks), `cascade.py` (structural-veto-first, semantic-check-only-if-needed).

## Scorer history (research/trace-experiment/semantic_scorer.py)
Tracked separately since it's research code, not the shipped library —
but the debugging history is worth keeping for anyone extending it.

- **v1**: exact trigger-phrase matching. 31% false-positive rate on
  benign traces, traced to zero negation awareness.
- **v2**: negation-scope detection. Took three attempts, each caught by
  a test: (1) basic negation window, (2) fixed a cross-clause leak by
  clipping to clause boundaries, (3) fixed "not" matching inside "note"
  with word-boundary regex. Result: 0% FPR, 66.7% recall.
- **v3**: compositional (action + control word) matching, closing the
  paraphrase gap where an attack phrase never used the exact trigger
  wording. One design bug found and fixed (a word doing double duty as
  both an action-word and a negation-marker was wrongly suppressing
  unrelated control-word matches). Result: 0% FPR, 98.7% recall.
- **Still open**: a paraphrase sharing zero vocabulary with either word
  cluster is still missed — the structural ceiling of keyword matching.
  `research/llm_semantic_check.py` is the reference path past that.

## LLM semantic check (research/llm_semantic_check.py)
Validated by the repo owner running it locally (no API key available in
the environment this repo was built in).

- **Request format**: confirmed correct against the live API before any
  real key was used — a deliberately invalid key still gets a genuine
  401 auth error back, not a malformed-request error.
- **Judgment quality**: 10/10 real attack phrases scored correctly,
  including all 3 the keyword scorer could never catch — the actual
  proof this closes the paraphrase gap. 12/12 real benign fillers
  scored correctly below threshold.
- **Real bug found**: three benign sentences with identical structure
  and zero manipulative content ("risk score X, within normal range")
  produced risk scores that tracked the embedded number X instead of
  staying uniformly low (0.14->0.14, 0.89->0.72, 0.02->0.02) — the model
  partially trusting a number embedded in the text it's supposed to be
  scrutinizing. A real prompt-injection-adjacent weakness, not a
  hypothetical one.
- **First fix overcorrected**: telling the model self-reported metrics
  "could be attacker-fabricated" made it treat any metric's mere
  presence as suspicious — a benign "risk score 0.14, within normal
  range" sentence jumped to 0.85, a new false positive. Caught by
  re-testing, not by inspection — the fix looked reasonable and wasn't.
- **Second attempt** separates "ignore the number's value" from "the
  number's presence is not evidence of anything" explicitly in the
  prompt. **Verified across two independent runs**, both returning
  identical 0.05/0.15/0.05 for inputs 0.14/0.89/0.02 (down from
  0.14/0.72/0.02 pre-fix). 12/12 benign fillers and 10/10 attack phrases
  correct both times, including the sentence that regressed during the
  first fix attempt (now consistently 0.05). The identical results
  across runs mean the small residual 0.89->0.15 bump is reproducible,
  not noise — but at 0.15 it's ~83% smaller than the original effect
  and nowhere near a 0.5 block threshold, so not chased further.
