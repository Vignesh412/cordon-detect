import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import WorkflowSchema


def test_round_trip_via_json():
    schema = WorkflowSchema(
        allowed_edges={("intake", "risk_assess"), ("risk_assess", "approval")},
        known_tools={"extract", "score"},
        required_agents={"intake", "approval"},
    )
    restored = WorkflowSchema.from_json(schema.to_json())
    assert restored.allowed_edges == schema.allowed_edges
    assert restored.known_tools == schema.known_tools
    assert restored.required_agents == schema.required_agents


def test_to_dict_is_json_safe_sorted_lists():
    schema = WorkflowSchema(
        allowed_edges={("b", "c"), ("a", "b")},
        known_tools={"z", "a"},
        required_agents={"b", "a"},
    )
    d = schema.to_dict()
    assert d["allowed_edges"] == [["a", "b"], ["b", "c"]]
    assert d["known_tools"] == ["a", "z"]
    assert d["required_agents"] == ["a", "b"]


def test_from_dict_missing_keys_default_empty():
    schema = WorkflowSchema.from_dict({})
    assert schema.allowed_edges == set()
    assert schema.known_tools == set()
    assert schema.required_agents == set()
