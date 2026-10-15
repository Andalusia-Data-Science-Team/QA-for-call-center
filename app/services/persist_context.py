"""
A context-local record of what the last DWH persist actually did.

WHY THIS EXISTS
───────────────
Persistence is a LangGraph node (app/agent/nodes.py:save_to_database), not
something the caller does after `QAAgent.analyze()` returns. So by the time
the automation worker gets its QAAnalysisResult, the row is already in DWH and
the worker has no idea which Analysis_Version landed — which is exactly what
the QA_ResultLinks protocol needs in order to be idempotent.

The alternatives were worse. Changing `analyze()` to return a tuple breaks
every existing caller. Restructuring the graph to persist outside itself
changes manual behaviour for no benefit. A context variable keeps the result
type untouched, costs nothing when unused, and is naturally scoped to one
evaluation.

contextvars, not a module global: `asyncio.gather` over several calls would
have them overwrite each other's outcome under a plain global. A ContextVar
set inside a task is visible only to that task.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PersistOutcome:
    """What insert_qa_result did, for the caller that needs to record it."""

    call_id: str
    analysis_version: int
    rows_inserted: int
    job_id: Optional[str] = None


_current: ContextVar[Optional[PersistOutcome]] = ContextVar("qa_persist_outcome", default=None)


def reset() -> None:
    """Clear any outcome before starting an evaluation."""
    _current.set(None)


def record(outcome: PersistOutcome) -> None:
    """Called by insert_qa_result once the rows are committed."""
    _current.set(outcome)


def last() -> Optional[PersistOutcome]:
    """
    The outcome of the most recent persist in this context, or None.

    None means no DWH write happened — the result was an error assessment, the
    insert failed, or persistence was skipped. The caller must treat that as
    "not persisted" and must not assume success.
    """
    return _current.get()
