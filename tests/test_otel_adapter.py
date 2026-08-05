import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import WorkflowSchema, tokenize, structural_check
from cordon.adapters.otel import from_otel_spans


def dict_span(span_id, parent_id, name, attributes):
    return {"span_id": span_id, "parent_span_id": parent_id, "name": name, "attributes": attributes}


class SdkStyleContext:
    def __init__(self, span_id):
        self.span_id = span_id


class SdkStyleParent:
    def __init__(self, span_id):
        self.span_id = span_id


class SdkStyleSpan:
    """Duck-typed stand-in for an opentelemetry-sdk ReadableSpan, to
    prove the adapter doesn't require the real SDK as a dependency."""
    def __init__(self, span_id, parent_id, name, attributes):
        self.context = SdkStyleContext(span_id)
        self.parent = SdkStyleParent(parent_id) if parent_id is not None else None
        self.name = name
        self.attributes = attributes


def test_dict_spans_convert_to_tool_steps():
    spans = [
        dict_span("1", None, "dispatch", {
            "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "orchestrator",
        }),
        dict_span("2", "1", "extract", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.agent.name": "intake",
            "gen_ai.tool.name": "extract",
            "gen_ai.tool.call.arguments": '{"path": "/in"}',
        }),
    ]
    steps = from_otel_spans(spans)
    assert steps[0]["type"] == "observation"
    assert steps[0]["agent"] == "orchestrator"
    assert steps[1] == {
        "type": "tool", "agent": "intake", "tool": "extract",
        "args": {"path": "/in"}, "id": "2", "parent_id": "1",
    }


def test_sdk_style_objects_are_duck_typed_the_same_way():
    spans = [
        SdkStyleSpan("1", None, "dispatch", {"gen_ai.agent.name": "orchestrator"}),
        SdkStyleSpan("2", "1", "score", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.agent.name": "risk_assess",
            "gen_ai.tool.name": "score",
        }),
        SdkStyleSpan("3", "1", "check", {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.agent.name": "compliance_check",
            "gen_ai.tool.name": "check",
        }),
    ]
    trace = from_otel_spans(spans)
    schema = WorkflowSchema(
        allowed_edges={("orchestrator", "risk_assess"), ("orchestrator", "compliance_check")},
        known_tools={"score", "check"},
        required_agents={"risk_assess", "compliance_check"},
    )
    # But orchestrator's own span is an "invoke_agent", not a tool call,
    # so it never becomes a graph node itself -- risk_assess/compliance_check
    # both declare orchestrator as parent_id "1", and since no tool step
    # has id "1", they fall back to having no resolvable parent action,
    # meaning no edge is asserted for them at all. This is the documented
    # limitation: register the dispatching step as a tool step too if you
    # need the edge enforced.
    result = structural_check(tokenize(trace), schema)
    assert result.flagged is False


def test_undeclared_tool_call_child_still_caught():
    spans = [
        dict_span("1", None, "dispatch", {
            "gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": "orchestrator",
            "gen_ai.tool.name": "dispatch",
        }),
        dict_span("2", "1", "score", {
            "gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": "risk_assess",
            "gen_ai.tool.name": "score",
        }),
        dict_span("3", "1", "wire_transfer", {
            "gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": "payments",
            "gen_ai.tool.name": "wire_transfer",
        }),
    ]
    schema = WorkflowSchema(
        allowed_edges={("orchestrator", "risk_assess")},
        known_tools={"dispatch", "score", "wire_transfer"},
    )
    trace = from_otel_spans(spans)
    result = structural_check(tokenize(trace), schema)
    assert result.flagged is True
    assert result.kind == "unauthorized_edge"
    assert result.detail == {"from": "orchestrator", "to": "payments"}


def test_custom_attribute_keys():
    spans = [dict_span("1", None, "run", {
        "my.agent": "intake", "my.op": "execute_tool", "my.tool": "extract",
    })]
    steps = from_otel_spans(
        spans, agent_attr="my.agent", operation_attr="my.op", tool_attr="my.tool",
    )
    assert steps[0]["agent"] == "intake"
    assert steps[0]["tool"] == "extract"
    assert steps[0]["type"] == "tool"


def test_malformed_json_arguments_do_not_crash():
    spans = [dict_span("1", None, "run", {
        "gen_ai.operation.name": "execute_tool", "gen_ai.agent.name": "intake",
        "gen_ai.tool.name": "extract", "gen_ai.tool.call.arguments": "{not valid json",
    })]
    steps = from_otel_spans(spans)
    assert steps[0]["args"] == {"_raw": "{not valid json"}
