import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cordon import draft_from_traces


def trace_for(agents):
    return [{"type": "tool", "agent": a, "tool": f"{a}_run", "args": {}} for a in agents]


BASELINE = [
    trace_for(["intake", "risk_assess", "approval"]),
    trace_for(["intake", "compliance_check", "approval"]),
    trace_for(["intake", "risk_assess", "approval"]),
]


def test_requires_verified_clean_flag():
    with pytest.raises(ValueError, match="verified_clean"):
        draft_from_traces(BASELINE)


def test_requires_nonempty_traces():
    with pytest.raises(ValueError):
        draft_from_traces([], verified_clean=True)


def test_edges_and_tools_are_union_of_all_traces():
    schema, report = draft_from_traces(BASELINE, verified_clean=True)
    assert schema.allowed_edges == {
        ("intake", "risk_assess"), ("risk_assess", "approval"),
        ("intake", "compliance_check"), ("compliance_check", "approval"),
    }
    assert schema.known_tools == {
        "intake_run", "risk_assess_run", "approval_run", "compliance_check_run",
    }


def test_required_agents_is_strict_intersection_not_majority():
    # risk_assess appears in 2/3 traces, compliance_check in 1/3 -- neither
    # is "required" under strict inference, even though risk_assess is a
    # majority. Only agents present in every single trace qualify.
    schema, report = draft_from_traces(BASELINE, verified_clean=True)
    assert schema.required_agents == {"intake", "approval"}
    assert "risk_assess" not in schema.required_agents
    assert "compliance_check" not in schema.required_agents
    # But it's not hidden -- the report shows the counts either way.
    assert report.agent_trace_counts["risk_assess"] == 2
    assert report.agent_trace_counts["compliance_check"] == 1
    assert report.agent_trace_counts["intake"] == 3


def test_rare_edges_flagged_by_default_threshold():
    schema, report = draft_from_traces(BASELINE, verified_clean=True)
    # (intake, risk_assess) and (risk_assess, approval) were seen twice;
    # the compliance_check edges were seen once each -- rare by default
    # (rare_below=2).
    assert report.rare_edges() == {
        ("intake", "compliance_check"), ("compliance_check", "approval"),
    }


def test_rare_threshold_is_configurable():
    schema, report = draft_from_traces(BASELINE, verified_clean=True, rare_below=3)
    # Now everything seen fewer than 3 times counts as rare, including
    # the 2x edges.
    assert report.rare_edges() == set(schema.allowed_edges)


def test_summary_is_a_nonempty_readable_string():
    schema, report = draft_from_traces(BASELINE, verified_clean=True)
    text = report.summary()
    assert "intake" in text
    assert "3" in text  # total_traces shows up somewhere


def test_report_to_dict_is_json_safe():
    import json
    schema, report = draft_from_traces(BASELINE, verified_clean=True)
    encoded = json.dumps(report.to_dict())  # must not raise
    assert "intake" in encoded
