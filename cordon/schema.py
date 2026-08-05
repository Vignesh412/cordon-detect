"""
WorkflowSchema: the one thing a developer has to declare.

Cordon doesn't infer what your pipeline is supposed to do — you tell it,
once, in a few lines. Everything downstream (the structural veto) checks
the live trace against this declaration.
"""

import json
from dataclasses import dataclass, field


@dataclass
class WorkflowSchema:
    # Legitimate agent-to-agent handoffs/spawns, e.g.
    # {("intake", "validate"), ("validate", "score")}. A parent agent
    # fanning out to several children concurrently just needs one edge
    # per child declared (e.g. ("intake", "validate"), ("intake", "score")
    # for two agents intake spawns in parallel) — no edge between the
    # children themselves is needed or checked, since they never handed
    # off to each other. See tokenizer.agent_call_graph() for how the
    # actual edges are derived from a trace.
    allowed_edges: set[tuple[str, str]] = field(default_factory=set)

    # Tools Cordon has ever been told about. A tool call outside this set is
    # treated as an unknown/unrecognized action — this is what catches novel,
    # never-seen-before attack patterns without needing a signature for them.
    known_tools: set[str] = field(default_factory=set)

    # Agents that MUST appear somewhere in a complete trace. Missing one is
    # flagged even if every edge that *does* appear is individually legal
    # (e.g. an agent silently omitted from a fan-out).
    required_agents: set[str] = field(default_factory=set)

    def allows_edge(self, from_agent: str, to_agent: str) -> bool:
        return (from_agent, to_agent) in self.allowed_edges

    def knows_tool(self, tool: str) -> bool:
        # Empty known_tools means "not tracking this" — don't false-flag.
        if not self.known_tools:
            return True
        return tool in self.known_tools

    def to_dict(self) -> dict:
        """JSON-safe representation — sets/tuples become sorted lists,
        so a schema can be checked into a repo as data and diffed like
        any other config, whether written by hand or via
        cordon.infer.draft_from_traces()."""
        return {
            "allowed_edges": sorted([list(edge) for edge in self.allowed_edges]),
            "known_tools": sorted(self.known_tools),
            "required_agents": sorted(self.required_agents),
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowSchema":
        return cls(
            allowed_edges={tuple(edge) for edge in data.get("allowed_edges", [])},
            known_tools=set(data.get("known_tools", [])),
            required_agents=set(data.get("required_agents", [])),
        )

    @classmethod
    def from_json(cls, raw: str) -> "WorkflowSchema":
        return cls.from_dict(json.loads(raw))
