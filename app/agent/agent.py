"""
QAAgent — the public interface for the LangGraph QA pipeline.

Drop-in replacement for the old CallAnalyzer.  The FastAPI layer only needs to:
    agent = QAAgent(llm_client)
    result = await agent.analyze(call)

Internals:
  - Compiles the LangGraph StateGraph once at construction time.
  - Runs the graph asynchronously for every call.
  - Surfaces the final QAAnalysisResult or raises AnalysisError on failure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from app.agent.graph import build_qa_graph
from app.agent.state import AgentState
from app.config import settings
from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _summarize_chat_usage(usage_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate token and USD usage across every LLM node for one call."""
    entries = [entry for entry in usage_entries if isinstance(entry, dict) and entry.get("node")]
    known_costs = [entry.get("cost_usd") for entry in entries if entry.get("cost_usd") is not None]
    cost_complete = bool(entries) and all(entry.get("cost_complete") for entry in entries)
    return {
        "request_count": sum(int(entry.get("requests") or 0) for entry in entries),
        "prompt_tokens": sum(int(entry.get("prompt_tokens") or 0) for entry in entries),
        "completion_tokens": sum(int(entry.get("completion_tokens") or 0) for entry in entries),
        "total_tokens": sum(int(entry.get("total_tokens") or 0) for entry in entries),
        "cost_usd": round(sum(float(cost) for cost in known_costs), 12) if cost_complete else None,
        "known_cost_usd": round(sum(float(cost) for cost in known_costs), 12),
        "cost_complete": cost_complete,
        "nodes": entries,
    }


def _build_token_usage_logger() -> tuple[logging.Logger, Path]:
    """Create a dedicated rotating JSONL logger for per-call LLM usage."""
    configured_path = Path(settings.TOKEN_USAGE_LOG_PATH)
    project_root = Path(__file__).resolve().parents[2]
    log_path = configured_path if configured_path.is_absolute() else project_root / configured_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    usage_logger = logging.getLogger("token_usage")
    usage_logger.setLevel(logging.INFO)
    usage_logger.propagate = False
    resolved_path = str(log_path.resolve())
    if not any(
        isinstance(handler, RotatingFileHandler)
        and getattr(handler, "baseFilename", None) == resolved_path
        for handler in usage_logger.handlers
    ):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=settings.TOKEN_USAGE_LOG_MAX_BYTES,
            backupCount=settings.TOKEN_USAGE_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        usage_logger.addHandler(handler)
    return usage_logger, log_path


class AnalysisError(Exception):
    """Raised when the LangGraph pipeline cannot produce a valid QAAnalysisResult."""


class QAAgent:
    """
    Async QA agent backed by a LangGraph StateGraph.

    Usage
    -----
    >>> agent = QAAgent(llm_client=LLMClient(provider="openrouter", model="..."))
    >>> result: QAAnalysisResult = await agent.analyze(call_transcript)

    Graph topology is defined in app/agent/graph.py.
    Nodes are defined in app/agent/nodes.py.
    State schema is defined in app/agent/state.py.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client
        self._token_usage_logger, self._token_usage_log_path = _build_token_usage_logger()
        # Compile the graph once — reused across all requests
        self._graph = build_qa_graph(llm_client)
        logger.info(
            "QAAgent initialised | provider=%s model=%s",
            llm_client.provider,
            llm_client.model,
        )

    async def analyze(self, call: CallTranscript) -> QAAnalysisResult:
        """
        Run the full QA pipeline for a single call transcript.

        Parameters
        ----------
        call : CallTranscript
            The validated inbound call object.

        Returns
        -------
        QAAnalysisResult
            Always returns a well-typed result.  On pipeline failure an
            'error' assessment result is returned (never raises for batch use).

        Raises
        ------
        AnalysisError
            Only raised if the graph itself produces no result at all
            (i.e. a bug in the graph — should never happen in normal operation).
        """
        initial_state: AgentState = {
            "call": call,
            "node_trace": [],   # Annotated[list, operator.add] — must be [] not None
            "error": None,
            "error_node": None,
            "usage_list": [],
        }

        final_state: AgentState = await self._graph.ainvoke(initial_state)

        usage = _summarize_chat_usage(final_state.get("usage_list") or [])
        usage_record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "call_id": call.call_id,
            "provider": self.llm_client.provider,
            "model": self.llm_client.model,
            "request_count": usage["request_count"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cost_usd": usage["cost_usd"],
            "known_cost_usd": usage["known_cost_usd"],
            "cost_complete": usage["cost_complete"],
            "currency": "USD",
            "nodes": usage["nodes"],
        }
        self._token_usage_logger.info(
            json.dumps(usage_record, ensure_ascii=False, default=str)
        )
        logger.info(
            "LLM usage | call_id=%s provider=%s model=%s requests=%d "
            "prompt_tokens=%d completion_tokens=%d total_tokens=%d "
            "cost_usd=%s cost_complete=%s file=%s",
            call.call_id,
            self.llm_client.provider,
            self.llm_client.model,
            usage["request_count"],
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
            usage["cost_usd"] if usage["cost_usd"] is not None else "unavailable",
            usage["cost_complete"],
            self._token_usage_log_path,
        )

        result = final_state.get("result")
        if result is None:
            raise AnalysisError(
                f"Graph produced no result for call_id={call.call_id}. "
                f"Trace: {final_state.get('node_trace')}"
            )

        return result
