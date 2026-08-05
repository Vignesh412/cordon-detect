"""
Structural tokenization for AI agent execution traces.

Encodes *what an agent did* (tool calls, argument patterns, observation
sequence) rather than *what was said*. Attackers can paraphrase content
freely, but they can't hide the shape of execution.
"""

from dataclasses import dataclass
from typing import Any, Optional


# The compact structural vocabulary. Deliberately small — abstraction
# level, not vocabulary size, is what drives generalization to attack
# patterns never seen before.
SYS = "SYS"
USER = "USER"
ASSISTANT = "ASSISTANT"
TOOL = "TOOL"
ARGS = "ARGS"
OBS = "OBS"
OUTPUT = "OUTPUT"
ERROR = "ERROR"
OTHER = "OTHER"


@dataclass
class Token:
    kind: str
    step_index: int
    agent: Optional[str] = None
    tool: Optional[str] = None
    content: Optional[str] = None
    raw: Optional[dict] = None
    id: Optional[str] = None           # this step's own id, if the trace provides one
    parent_id: Optional[str] = None    # explicit parent step id, if the trace provides one


def tokenize(trace: list[dict[str, Any]]) -> list[Token]:
    """
    Convert a raw execution trace into structural tokens.

    Each step is a dict with at least a "type" key:
        {"type": "sys" | "user" | "assistant" | "tool" | "observation" | "output" | "error",
         "agent": str,            # which agent performed this step
         "tool": str,             # tool name, for type="tool"
         "args": dict,            # tool arguments, for type="tool"
         "content": str,          # free text, for observation/output/user/assistant
         "id": str,               # optional: this step's own id (e.g. a span id)
         "parent_id": str}        # optional: id of the step that spawned this one

    "id"/"parent_id" are optional and only meaningful for "tool" steps —
    they let agent_call_graph() build the real handoff/spawn graph
    (including concurrent fan-out, where two children share one parent
    and there's no edge between them) instead of assuming every step
    follows in a single line. Omit them and everything behaves exactly
    as before: each tool step's implicit parent is simply the previous
    tool step. See adapters/otel.py for a source that provides them.

    Returns a flat list of Token objects, one or more per step.
    """
    tokens: list[Token] = []

    for i, step in enumerate(trace):
        kind = step.get("type", "").lower()
        agent = step.get("agent")
        step_id = step.get("id")
        parent_id = step.get("parent_id")

        if kind == "sys":
            tokens.append(Token(SYS, i, agent=agent, raw=step, id=step_id, parent_id=parent_id))
        elif kind == "user":
            tokens.append(Token(USER, i, agent=agent, raw=step, id=step_id, parent_id=parent_id))
        elif kind == "assistant":
            tokens.append(Token(ASSISTANT, i, agent=agent, raw=step, id=step_id, parent_id=parent_id))
        elif kind == "tool":
            tool_name = step.get("tool", "")
            tokens.append(Token(TOOL, i, agent=agent, tool=tool_name, raw=step,
                                 id=step_id, parent_id=parent_id))
            tokens.append(Token(ARGS, i, agent=agent, tool=tool_name,
                                 content=str(step.get("args", {})), raw=step,
                                 id=step_id, parent_id=parent_id))
        elif kind == "observation":
            tokens.append(Token(OBS, i, agent=agent,
                                 content=step.get("content", ""), raw=step,
                                 id=step_id, parent_id=parent_id))
        elif kind == "output":
            tokens.append(Token(OUTPUT, i, agent=agent,
                                 content=step.get("content", ""), raw=step,
                                 id=step_id, parent_id=parent_id))
        elif kind == "error":
            tokens.append(Token(ERROR, i, agent=agent,
                                 content=step.get("content", ""), raw=step,
                                 id=step_id, parent_id=parent_id))
        else:
            tokens.append(Token(OTHER, i, agent=agent, raw=step, id=step_id, parent_id=parent_id))

    return tokens


def agent_call_sequence(tokens: list[Token]) -> list[str]:
    """
    Extract the ordered sequence of distinct agents that acted, collapsing
    consecutive repeats. Order-only view — used for required_agents
    membership and for display/debugging. The structural veto's edge
    check uses agent_call_graph() instead, not this, since a flat
    sequence can't represent concurrent fan-out correctly.
    """
    seq: list[str] = []
    for tok in tokens:
        if tok.kind == TOOL and tok.agent is not None:
            if not seq or seq[-1] != tok.agent:
                seq.append(tok.agent)
    return seq


def _action_key(tok: Token) -> str:
    """Stable identity for an agent-action token: its own id if the
    trace provided one, else a synthetic key derived from position."""
    return tok.id if tok.id is not None else f"_step{tok.step_index}"


def agent_call_graph(tokens: list[Token]) -> list[tuple[str, str]]:
    """
    Build the (parent_agent, child_agent) handoff/spawn edges actually
    observed in the trace, in first-observed order with duplicates
    removed. This is what the structural veto checks against
    WorkflowSchema.allowed_edges.

    Each TOOL token is a graph node (an "agent action"). Its parent is:
      - the action whose id matches this step's "parent_id", if the
        trace provided one for this step (see tokenize()) — this is
        how real span data (tokenize()'d via adapters.otel or any
        other tracer, which sets "id"/"parent_id" per step) expresses
        an explicit call graph, including concurrent fan-out: two
        actions sharing one parent_id produce two edges and — crucially
        — no edge between the two of them, because they never actually
        handed off to each other.
      - otherwise, the immediately preceding action in the trace — this
        reproduces the classic flat/linear behavior exactly, so a trace
        with no "id"/"parent_id" anywhere behaves identically to before.

    Edges where the parent and child are the same agent are omitted
    (that's not a handoff to check, just the same agent acting twice
    in a row).
    """
    actions = [t for t in tokens if t.kind == TOOL and t.agent is not None]
    by_key = {_action_key(tok): tok for tok in actions}

    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    prev: Optional[Token] = None

    for tok in actions:
        step = tok.raw or {}
        if "parent_id" in step:
            parent = by_key.get(step["parent_id"])
        else:
            parent = prev

        if parent is not None and parent.agent != tok.agent:
            edge = (parent.agent, tok.agent)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

        prev = tok

    return edges


def tools_used(tokens: list[Token]) -> set[str]:
    """All distinct tool names invoked in the trace."""
    return {tok.tool for tok in tokens if tok.kind == TOOL and tok.tool}
