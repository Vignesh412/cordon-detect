"""
Schema drafting: build a starting WorkflowSchema from a batch of traces
you've already verified are clean.

This is deliberately NOT "learn the schema from production traffic and
trust it" — doing that silently launders whatever's in the baseline,
including any trace that was itself already compromised, into your
security policy as legitimate, and under-samples any legitimate-but-rare
path as forbidden. draft_from_traces() instead returns a *reviewable
draft*: a WorkflowSchema paired with a SchemaDraftReport carrying
per-edge/per-agent/per-tool observation counts, so a human can see how
much evidence backs each rule before keeping it — the same relationship
a generated diff has to an auto-merge.
"""

from dataclasses import dataclass, field

from .schema import WorkflowSchema
from .tokenizer import tokenize, agent_call_graph, agent_call_sequence, tools_used


@dataclass
class SchemaDraftReport:
    total_traces: int
    edge_counts: dict = field(default_factory=dict)          # (from, to) -> trace count
    agent_trace_counts: dict = field(default_factory=dict)   # agent -> trace count present in
    tool_counts: dict = field(default_factory=dict)          # tool -> trace count
    rare_below: int = 2

    def rare_edges(self) -> set:
        """Edges seen fewer than rare_below times — thin evidence,
        review before trusting them as much as the common ones."""
        return {edge for edge, n in self.edge_counts.items() if n < self.rare_below}

    def rare_tools(self) -> set:
        return {tool for tool, n in self.tool_counts.items() if n < self.rare_below}

    def to_dict(self) -> dict:
        return {
            "total_traces": self.total_traces,
            "edge_counts": {f"{a}->{b}": n for (a, b), n in self.edge_counts.items()},
            "agent_trace_counts": dict(self.agent_trace_counts),
            "tool_counts": dict(self.tool_counts),
            "rare_below": self.rare_below,
        }

    def summary(self) -> str:
        """Human-readable review sheet — print this before deciding
        whether to keep the draft as-is."""
        lines = [f"Drafted from {self.total_traces} trace(s)."]

        lines.append("\nEdges (parent -> child): traces seen in")
        for edge, n in sorted(self.edge_counts.items(), key=lambda kv: -kv[1]):
            flag = "  <- rare, review before keeping" if n < self.rare_below else ""
            lines.append(f"  {edge[0]} -> {edge[1]}: {n}{flag}")

        lines.append("\nAgents: traces present in / total")
        for agent, n in sorted(self.agent_trace_counts.items()):
            tag = "  [required: present in every trace]" if n == self.total_traces else ""
            lines.append(f"  {agent}: {n}/{self.total_traces}{tag}")

        lines.append("\nTools: traces seen in")
        for tool, n in sorted(self.tool_counts.items(), key=lambda kv: -kv[1]):
            flag = "  <- rare, review before keeping" if n < self.rare_below else ""
            lines.append(f"  {tool}: {n}{flag}")

        return "\n".join(lines)


def draft_from_traces(
    traces: list,
    *,
    verified_clean: bool = False,
    rare_below: int = 2,
) -> tuple:
    """
    Build a DRAFT WorkflowSchema from a batch of traces.

    Returns (schema, report) — a starting point for you to review and
    edit, not something to deploy unattended.

    Raises ValueError unless verified_clean=True. That's a deliberate
    speed bump, not a formality: an inferred schema is only as
    trustworthy as the traces it was inferred from, and this function
    has no way to tell a genuinely clean baseline from one that's
    already carrying an attack. Passing verified_clean=True is you
    asserting you've checked that, not this function checking it.

    required_agents is inferred strictly: only an agent present in
    EVERY trace in the batch. An agent present in most-but-not-all
    traces is deliberately NOT inferred as required — that's exactly
    the case a looser threshold gets wrong (a rare-but-important gate
    silently downgraded to optional) — it still shows up in the report
    so you can add it by hand if it should be required.

    allowed_edges/known_tools include everything observed at least
    once; report.rare_edges()/rare_tools() flag anything seen fewer
    than `rare_below` times so thin evidence doesn't get baked in with
    the same authority as something seen thousands of times without
    you noticing.
    """
    if not verified_clean:
        raise ValueError(
            "draft_from_traces() requires verified_clean=True — an explicit "
            "acknowledgment that this batch of traces has been checked and "
            "isn't itself carrying an attack. An inferred schema is only as "
            "trustworthy as what it was inferred from; this isn't a formality."
        )
    if not traces:
        raise ValueError("draft_from_traces() needs at least one trace.")

    edge_counts: dict = {}
    agent_trace_counts: dict = {}
    tool_counts: dict = {}

    for trace in traces:
        tokens = tokenize(trace)
        for edge in agent_call_graph(tokens):
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        for agent in set(agent_call_sequence(tokens)):
            agent_trace_counts[agent] = agent_trace_counts.get(agent, 0) + 1
        for tool in tools_used(tokens):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    total = len(traces)
    required_agents = {agent for agent, n in agent_trace_counts.items() if n == total}

    schema = WorkflowSchema(
        allowed_edges=set(edge_counts.keys()),
        known_tools=set(tool_counts.keys()),
        required_agents=required_agents,
    )
    report = SchemaDraftReport(
        total_traces=total,
        edge_counts=edge_counts,
        agent_trace_counts=agent_trace_counts,
        tool_counts=tool_counts,
        rare_below=rare_below,
    )
    return schema, report
