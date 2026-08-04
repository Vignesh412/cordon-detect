"""
WorkflowSchema: the one thing a developer has to declare.

Cordon doesn't infer what your pipeline is supposed to do — you tell it,
once, in a few lines. Everything downstream (the structural veto) checks
the live trace against this declaration.
"""

from dataclasses import dataclass, field


@dataclass
class WorkflowSchema:
    # Legitimate agent-to-agent handoffs, e.g. {("intake", "validate"), ("validate", "score")}.
    # A call sequence that produces an edge NOT in this set is a structural anomaly.
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
