"""
Structural veto: Step 1 of the cascade.

Checks execution shape against a declared schema. No model call, no
inference latency worth mentioning — this is a graph/set lookup. Runs on
every action; only escalates to a semantic check when it can't rule
something out on structure alone.
"""

from dataclasses import dataclass
from typing import Optional

from .schema import WorkflowSchema
from .tokenizer import Token, agent_call_sequence, agent_call_graph, tools_used


@dataclass
class VetoResult:
    flagged: bool
    reason: Optional[str] = None
    kind: Optional[str] = None          # "unauthorized_edge" | "unknown_tool" | "missing_required"
    detail: Optional[dict] = None
    elapsed_ms: float = 0.0


def check(tokens: list[Token], schema: WorkflowSchema) -> VetoResult:
    import time
    t0 = time.perf_counter()

    seq = agent_call_sequence(tokens)
    used_tools = tools_used(tokens)

    # 1. Unrecognized tool — catches novel/unseen attack patterns even
    #    with no signature for them, because the check is "have I ever
    #    heard of this action" not "does this match a known attack".
    for tool in used_tools:
        if not schema.knows_tool(tool):
            return _result(True, "unknown_tool",
                            f'Tool "{tool}" is not in the declared tool set — unrecognized action.',
                            {"tool": tool}, t0)

    # 2. Unauthorized / skipped handoff — a parent->child edge in the
    #    actual call graph that isn't in the declared graph. Built from
    #    real span parent/child linkage when the trace provides it
    #    (concurrent fan-out included), falling back to sequential
    #    adjacency otherwise — see agent_call_graph()'s docstring.
    if schema.allowed_edges:
        for a, b in agent_call_graph(tokens):
            if not schema.allows_edge(a, b):
                return _result(True, "unauthorized_edge",
                                f'Handoff "{a}" -> "{b}" is not a declared edge.',
                                {"from": a, "to": b}, t0)

    # 3. Required agent missing entirely from the trace.
    if schema.required_agents:
        missing = schema.required_agents - set(seq)
        if missing:
            return _result(True, "missing_required",
                            f"Required agent(s) never appeared: {sorted(missing)}.",
                            {"missing": sorted(missing)}, t0)

    return _result(False, None, None, None, t0)


def _result(flagged, kind, reason, detail, t0) -> VetoResult:
    import time
    elapsed = (time.perf_counter() - t0) * 1000
    return VetoResult(flagged=flagged, reason=reason, kind=kind, detail=detail, elapsed_ms=elapsed)
