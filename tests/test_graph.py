import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import WorkflowSchema, tokenize, structural_check
from cordon.tokenizer import agent_call_graph


def test_flat_trace_graph_matches_old_sequential_behavior():
    # No id/parent_id anywhere -> falls back to "previous action"
    # chaining, exactly reproducing pre-DAG behavior.
    trace = [
        {"type": "tool", "agent": "intake", "tool": "extract", "args": {}},
        {"type": "tool", "agent": "policy_check", "tool": "validate", "args": {}},
        {"type": "tool", "agent": "risk_assess", "tool": "score", "args": {}},
    ]
    assert agent_call_graph(tokenize(trace)) == [
        ("intake", "policy_check"),
        ("policy_check", "risk_assess"),
    ]


def test_concurrent_fan_out_produces_no_sibling_edge():
    # orchestrator spawns risk_assess and compliance_check concurrently
    # (both children of the same parent span). A flat-list flattening
    # would fabricate a spurious risk_assess -> compliance_check (or
    # reverse) edge; the graph model must not.
    trace = [
        {"type": "tool", "agent": "orchestrator", "tool": "dispatch", "args": {},
         "id": "s1", "parent_id": None},
        {"type": "tool", "agent": "risk_assess", "tool": "score", "args": {},
         "id": "s2", "parent_id": "s1"},
        {"type": "tool", "agent": "compliance_check", "tool": "check", "args": {},
         "id": "s3", "parent_id": "s1"},
    ]
    edges = agent_call_graph(tokenize(trace))
    assert set(edges) == {
        ("orchestrator", "risk_assess"),
        ("orchestrator", "compliance_check"),
    }
    # The point of this test: no edge was invented between the siblings.
    assert ("risk_assess", "compliance_check") not in edges
    assert ("compliance_check", "risk_assess") not in edges


def test_concurrent_fan_out_passes_structural_check_when_both_edges_declared():
    schema = WorkflowSchema(
        allowed_edges={
            ("orchestrator", "risk_assess"),
            ("orchestrator", "compliance_check"),
        },
        known_tools={"dispatch", "score", "check"},
        required_agents={"orchestrator", "risk_assess", "compliance_check"},
    )
    trace = [
        {"type": "tool", "agent": "orchestrator", "tool": "dispatch", "args": {},
         "id": "s1", "parent_id": None},
        {"type": "tool", "agent": "compliance_check", "tool": "check", "args": {},
         "id": "s2", "parent_id": "s1"},
        {"type": "tool", "agent": "risk_assess", "tool": "score", "args": {},
         "id": "s3", "parent_id": "s1"},
    ]
    # Both orderings of the concurrent children pass — order between
    # siblings was never meaningful, unlike the old flat-sequence model
    # where it would have mattered.
    result = structural_check(tokenize(trace), schema)
    assert result.flagged is False


def test_concurrent_fan_out_still_catches_a_real_unauthorized_child():
    schema = WorkflowSchema(
        allowed_edges={("orchestrator", "risk_assess")},
        known_tools={"dispatch", "score", "check"},
    )
    trace = [
        {"type": "tool", "agent": "orchestrator", "tool": "dispatch", "args": {},
         "id": "s1", "parent_id": None},
        {"type": "tool", "agent": "risk_assess", "tool": "score", "args": {},
         "id": "s2", "parent_id": "s1"},
        # compliance_check was never declared as a legitimate child of
        # orchestrator — still flagged even though it ran "concurrently".
        {"type": "tool", "agent": "compliance_check", "tool": "check", "args": {},
         "id": "s3", "parent_id": "s1"},
    ]
    result = structural_check(tokenize(trace), schema)
    assert result.flagged is True
    assert result.kind == "unauthorized_edge"
    assert result.detail == {"from": "orchestrator", "to": "compliance_check"}


def test_explicit_parent_id_overrides_positional_fallback():
    # s3 comes third in the list but explicitly declares s1 (not s2) as
    # its parent — the graph must follow the explicit link, not order.
    trace = [
        {"type": "tool", "agent": "a", "tool": "t", "args": {}, "id": "s1", "parent_id": None},
        {"type": "tool", "agent": "b", "tool": "t", "args": {}, "id": "s2", "parent_id": "s1"},
        {"type": "tool", "agent": "c", "tool": "t", "args": {}, "id": "s3", "parent_id": "s1"},
    ]
    edges = agent_call_graph(tokenize(trace))
    assert set(edges) == {("a", "b"), ("a", "c")}
    assert ("b", "c") not in edges
