"""
Cascade: wires structural veto and an optional semantic check together.

Structural veto runs on every trace and decides on its own when it's
confident. Semantic/linguistic analysis — usually the expensive part,
an LLM call in most real deployments — only runs on the traces
structure couldn't already rule out. Most traffic never needs it.
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .schema import WorkflowSchema
from .tokenizer import Token, tokenize
from .veto import VetoResult, check as structural_check


@dataclass
class SemanticResult:
    flagged: bool
    reason: Optional[str] = None
    confidence: Optional[float] = None


# A semantic check is any callable you provide: (trace, tokens) -> SemanticResult.
# Cordon has no opinion on how you implement it — a keyword rule, a small
# classifier, an LLM call. It only decides *when* to call it.
SemanticCheck = Callable[[list[dict], list[Token]], SemanticResult]


@dataclass
class CascadeResult:
    blocked: bool
    stage: str                      # "structural" | "semantic" | "clean"
    reason: Optional[str]
    structural: VetoResult
    semantic: Optional[SemanticResult]
    semantic_skipped: bool
    total_elapsed_ms: float


def run(
    trace: list[dict],
    schema: WorkflowSchema,
    semantic_check: Optional[SemanticCheck] = None,
) -> CascadeResult:
    t0 = time.perf_counter()
    tokens = tokenize(trace)

    struct_result = structural_check(tokens, schema)

    if struct_result.flagged:
        # Blocked on structure alone. Semantic check never runs —
        # this is where the cost saving actually happens.
        total = (time.perf_counter() - t0) * 1000
        return CascadeResult(
            blocked=True,
            stage="structural",
            reason=struct_result.reason,
            structural=struct_result,
            semantic=None,
            semantic_skipped=True,
            total_elapsed_ms=total,
        )

    if semantic_check is not None:
        sem_result = semantic_check(trace, tokens)
        total = (time.perf_counter() - t0) * 1000
        if sem_result.flagged:
            return CascadeResult(
                blocked=True,
                stage="semantic",
                reason=sem_result.reason,
                structural=struct_result,
                semantic=sem_result,
                semantic_skipped=False,
                total_elapsed_ms=total,
            )
        return CascadeResult(
            blocked=False,
            stage="clean",
            reason=None,
            structural=struct_result,
            semantic=sem_result,
            semantic_skipped=False,
            total_elapsed_ms=total,
        )

    total = (time.perf_counter() - t0) * 1000
    return CascadeResult(
        blocked=False,
        stage="clean",
        reason=None,
        structural=struct_result,
        semantic=None,
        semantic_skipped=True,
        total_elapsed_ms=total,
    )
