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

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent.graph import build_qa_graph
from app.agent.state import AgentState
from app.agent.telemetry import (
    begin_node_execution_tracking,
    build_jsonl_logger,
    end_node_execution_tracking,
    write_jsonl,
)
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


def _summarize_node_usage(
    node_trace: list[str],
    usage_entries: list[dict[str, Any]],
    *,
    default_provider: str,
    default_model: str,
) -> list[dict[str, Any]]:
    """Build one consumption record for every graph node in the execution trace."""
    ordered_nodes: list[str] = []
    execution_counts: dict[str, int] = {}
    for trace_entry in node_trace:
        node_name = str(trace_entry).split("[", 1)[0]
        execution_counts[node_name] = execution_counts.get(node_name, 0) + 1
        if node_name not in ordered_nodes:
            ordered_nodes.append(node_name)

    usage_by_node: dict[str, list[dict[str, Any]]] = {}
    for entry in usage_entries:
        if isinstance(entry, dict) and entry.get("node"):
            usage_by_node.setdefault(str(entry["node"]), []).append(entry)

    records: list[dict[str, Any]] = []
    for sequence, node_name in enumerate(ordered_nodes, start=1):
        entries = usage_by_node.get(node_name, [])
        known_costs = [
            entry.get("cost_usd")
            for entry in entries
            if entry.get("cost_usd") is not None
        ]
        uses_llm = bool(entries)
        cost_complete = not uses_llm or all(
            bool(entry.get("cost_complete")) for entry in entries
        )
        records.append(
            {
                "node": node_name,
                "sequence": sequence,
                "execution_count": execution_counts[node_name],
                "provider": (
                    entries[-1].get("provider") or default_provider
                    if uses_llm else None
                ),
                "model": (
                    entries[-1].get("model") or default_model
                    if uses_llm else None
                ),
                "request_count": sum(int(entry.get("requests") or 0) for entry in entries),
                "prompt_tokens": sum(int(entry.get("prompt_tokens") or 0) for entry in entries),
                "completion_tokens": sum(int(entry.get("completion_tokens") or 0) for entry in entries),
                "total_tokens": sum(int(entry.get("total_tokens") or 0) for entry in entries),
                "cost_usd": (
                    round(sum(float(cost) for cost in known_costs), 12)
                    if cost_complete else None
                ),
                "known_cost_usd": round(sum(float(cost) for cost in known_costs), 12),
                "cost_complete": cost_complete,
                "uses_llm": uses_llm,
            }
        )
    return records


def _build_consumption_logger(
    logger_name: str, configured_path: str
) -> tuple[logging.Logger, Path]:
    """Create a dedicated rotating JSONL consumption logger."""
    return build_jsonl_logger(logger_name, configured_path)


def _build_token_usage_logger() -> tuple[logging.Logger, Path]:
    """Backward-compatible builder for the former combined usage log."""
    configured_path = settings.TOKEN_USAGE_LOG_PATH or settings.OVERALL_CONSUMPTION_LOG_PATH
    return build_jsonl_logger("token_usage", configured_path)


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
        self._node_consumption_logger, self._node_consumption_log_path = (
            _build_consumption_logger(
                "node_consumption", settings.NODE_CONSUMPTION_LOG_PATH
            )
        )
        self._overall_consumption_logger, self._overall_consumption_log_path = (
            _build_consumption_logger(
                "overall_consumption", settings.OVERALL_CONSUMPTION_LOG_PATH
            )
        )
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

        tracking_token = begin_node_execution_tracking()
        try:
            final_state: AgentState = await self._graph.ainvoke(initial_state)
        finally:
            executed_nodes = end_node_execution_tracking(tracking_token)

        usage_entries = final_state.get("usage_list") or []
        usage = _summarize_chat_usage(usage_entries)
        node_usage = _summarize_node_usage(
            executed_nodes or final_state.get("node_trace") or [],
            usage_entries,
            default_provider=self.llm_client.provider,
            default_model=self.llm_client.model,
        )
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        for node_record in node_usage:
            write_jsonl(
                self._node_consumption_logger,
                {"timestamp_utc": timestamp_utc, "call_id": call.call_id, **node_record},
            )
        usage_record = {
            "timestamp_utc": timestamp_utc,
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
            "node_count": len(node_usage),
            "llm_node_count": sum(1 for record in node_usage if record["uses_llm"]),
        }
        write_jsonl(self._overall_consumption_logger, usage_record)
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
            self._overall_consumption_log_path,
        )

        result = final_state.get("result")
        if result is None:
            raise AnalysisError(
                f"Graph produced no result for call_id={call.call_id}. "
                f"Trace: {final_state.get('node_trace')}"
            )

        return result
