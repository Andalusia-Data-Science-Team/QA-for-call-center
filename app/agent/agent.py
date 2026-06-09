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

from app.agent.graph import build_qa_graph
from app.agent.state import AgentState
from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


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
        }

        final_state: AgentState = await self._graph.ainvoke(initial_state)

        result = final_state.get("result")
        if result is None:
            raise AnalysisError(
                f"Graph produced no result for call_id={call.call_id}. "
                f"Trace: {final_state.get('node_trace')}"
            )

        return result
