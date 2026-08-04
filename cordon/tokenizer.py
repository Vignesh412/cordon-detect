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


def tokenize(trace: list[dict[str, Any]]) -> list[Token]:
    """
    Convert a raw execution trace into structural tokens.

    Each step is a dict with at least a "type" key:
        {"type": "sys" | "user" | "assistant" | "tool" | "observation" | "output" | "error",
         "agent": str,            # which agent performed this step
         "tool": str,             # tool name, for type="tool"
         "args": dict,            # tool arguments, for type="tool"
         "content": str}          # free text, for observation/output/user/assistant

    Returns a flat list of Token objects, one or more per step.
    """
    tokens: list[Token] = []

    for i, step in enumerate(trace):
        kind = step.get("type", "").lower()
        agent = step.get("agent")

        if kind == "sys":
            tokens.append(Token(SYS, i, agent=agent, raw=step))
        elif kind == "user":
            tokens.append(Token(USER, i, agent=agent, raw=step))
        elif kind == "assistant":
            tokens.append(Token(ASSISTANT, i, agent=agent, raw=step))
        elif kind == "tool":
            tool_name = step.get("tool", "")
            tokens.append(Token(TOOL, i, agent=agent, tool=tool_name, raw=step))
            tokens.append(Token(ARGS, i, agent=agent, tool=tool_name,
                                 content=str(step.get("args", {})), raw=step))
        elif kind == "observation":
            tokens.append(Token(OBS, i, agent=agent,
                                 content=step.get("content", ""), raw=step))
        elif kind == "output":
            tokens.append(Token(OUTPUT, i, agent=agent,
                                 content=step.get("content", ""), raw=step))
        elif kind == "error":
            tokens.append(Token(ERROR, i, agent=agent,
                                 content=step.get("content", ""), raw=step))
        else:
            tokens.append(Token(OTHER, i, agent=agent, raw=step))

    return tokens


def agent_call_sequence(tokens: list[Token]) -> list[str]:
    """
    Extract the ordered sequence of distinct agents that acted, collapsing
    consecutive repeats. This is the sequence structural veto checks
    against the expected call graph.
    """
    seq: list[str] = []
    for tok in tokens:
        if tok.kind == TOOL and tok.agent is not None:
            if not seq or seq[-1] != tok.agent:
                seq.append(tok.agent)
    return seq


def tools_used(tokens: list[Token]) -> set[str]:
    """All distinct tool names invoked in the trace."""
    return {tok.tool for tok in tokens if tok.kind == TOOL and tok.tool}
