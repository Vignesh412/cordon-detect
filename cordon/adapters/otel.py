"""
OpenTelemetry GenAI semantic-convention adapter.

Converts a list of OTel spans into cordon's plain trace format (list of
step dicts, per tokenizer.tokenize()) so structural_check()/run() can
run against traces produced by any OTel-instrumented agent stack —
directly, or via a system that exports/consumes OTel GenAI spans
(several agent observability tools do). No opentelemetry package is
imported here and none is required: spans are read duck-typed, so both
plain dicts (e.g. from an OTLP/JSON export) and SDK-style span objects
(anything exposing .name / .attributes / .context.span_id) work.

Span -> step mapping:
  - A span is a "tool" step if its `gen_ai.operation.name` attribute is
    "execute_tool", or it carries a `gen_ai.tool.name` attribute at
    all. Everything else (chat/invoke_agent/etc.) becomes a passthrough
    "observation" step: kept in the trace, but — matching cordon's
    existing tool-call-centric model (see tokenizer.agent_call_graph
    and required_agents) — it does not itself register as a graph node
    or satisfy required_agents. An agent that only reasons and never
    calls a recognized tool won't be "present" as far as structural
    checking is concerned; that's the same rule flat traces have always
    followed, not something this adapter changes.
  - Each span's id and parent id are carried through as the step's
    "id"/"parent_id", so agent_call_graph() builds the real call graph
    from actual span parentage — including concurrent fan-out (two
    children of one parent span produce two edges and no edge between
    the children) — instead of assuming spans form one flat sequence.

GenAI semantic convention attribute names are still evolving upstream;
if your instrumentation uses different keys, override them via the
keyword arguments below rather than forking this function.
"""

import json
from typing import Any, Optional


def from_otel_spans(
    spans: list,
    *,
    agent_attr: str = "gen_ai.agent.name",
    tool_attr: str = "gen_ai.tool.name",
    tool_args_attr: str = "gen_ai.tool.call.arguments",
    operation_attr: str = "gen_ai.operation.name",
) -> list[dict]:
    """
    Convert a list of OTel spans (any order) into a cordon trace.

    agent_attr/tool_attr/tool_args_attr/operation_attr let you point
    this at non-standard attribute keys without needing a fork —
    e.g. from_otel_spans(spans, agent_attr="my.custom.agent_name").

    Spans with no agent_attr value are still included (agent=None) so
    tool-name/unknown_tool checking still applies to them, but they
    won't participate in the handoff graph or required_agents (both
    require a known agent) — set agent_attr correctly, or enrich your
    spans with it before calling this, if that matters for your schema.
    """
    steps: list[dict] = []
    for span in spans:
        attrs = _field(span, "attributes", default={}) or {}
        agent = attrs.get(agent_attr)
        operation = attrs.get(operation_attr, "")
        span_id = _span_id(span)
        parent_id = _parent_id(span)

        is_tool_call = operation == "execute_tool" or tool_attr in attrs
        if is_tool_call:
            tool_name = attrs.get(tool_attr) or _field(span, "name", default="")
            steps.append({
                "type": "tool",
                "agent": agent,
                "tool": tool_name,
                "args": _tool_args(attrs.get(tool_args_attr, {})),
                "id": span_id,
                "parent_id": parent_id,
            })
        else:
            steps.append({
                "type": "observation",
                "agent": agent,
                "content": _field(span, "name", default=""),
                "id": span_id,
                "parent_id": parent_id,
            })

    return steps


def _tool_args(raw: Any) -> dict:
    """gen_ai.tool.call.arguments is commonly a JSON string on the
    wire (OTel attribute values are scalars/strings) but may already
    be a dict if the span object exposes it pre-parsed."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def _field(span: Any, *names: str, default: Any = None) -> Any:
    """Duck-typed lookup: try each name as a dict key, then as an
    attribute, so both plain dicts and SDK-style span objects work."""
    for name in names:
        if isinstance(span, dict):
            if name in span:
                return span[name]
        elif hasattr(span, name):
            return getattr(span, name)
    return default


def _stringify_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    return str(raw)


def _span_id(span: Any) -> Optional[str]:
    ctx = _field(span, "context")
    if ctx is not None:
        sid = _field(ctx, "span_id")
        if sid is not None:
            return _stringify_id(sid)
    return _stringify_id(_field(span, "span_id", "id"))


def _parent_id(span: Any) -> Optional[str]:
    parent = _field(span, "parent")
    if parent is not None:
        pid = _field(parent, "span_id")
        if pid is not None:
            return _stringify_id(pid)
    return _stringify_id(_field(span, "parent_span_id", "parent_id"))
