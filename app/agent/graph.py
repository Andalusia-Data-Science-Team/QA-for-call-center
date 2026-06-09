"""
LangGraph Pipeline Graph for the Call QA Analysis System.

Graph topology (happy path):
─────────────────────────────────────────────────────────────────────────────
  START
    │
    ▼
  load_call ──(error)──────────────────────────────────────────────────────┐
    │                                                                       │
    ├─────────────────────────────────────────────────────┐                │
    │                       (fan-out: 4 parallel loaders) │                │
    ▼                        ▼                   ▼        ▼                │
  load_behavioral_criteria  load_compliance_pillars      load_script_templates
  load_scoring_weights                                                      │
    │                        │                   │        │                │
    └────────────────────────┴───────────────────┴────────┘                │
                                    │ (fan-in)                              │
                                    ▼                                       │
                              build_prompt ──(error)──────────────────────┤│
                                    │                                       ││
                                    ▼                                       ││
                             llm_inference ──(error)──────────────────────┤│
                                    │                                       ││
                                    ▼                                       ││
                            parse_response ──(error)──────────────────────┤│
                                    │                                       ││
                                    ▼                                       ││
                           validate_output ──(error)──────────────────────┤│
                                    │                                       ││
                                    ▼                                       ││
                           integrity_check                                  ││
                                    │                                       ││
                                    ▼                                       ││
                                 finalize                                   ││
                                    │                             ┌─────────┘│
                                   END ◄─── handle_error ◄───────┘          │
                                               ▲                             │
                                               └─────────────────────────────┘
─────────────────────────────────────────────────────────────────────────────

The 4 criteria-loader nodes run after load_call using LangGraph's fan-out/fan-in
pattern (parallel edges + a barrier node).  They are pure I/O nodes (YAML reads
from lru_cache) so the concurrency overhead is negligible, but the graph topology
makes it trivial to add, remove, or replace individual criteria sources.

Adding a new criteria node:
  1. Write the async node function in nodes.py.
  2. Import it below.
  3. Add `builder.add_node("my_criteria_node", my_criteria_node)`.
  4. Add `builder.add_edge("load_call", "my_criteria_node")` (fan-out).
  5. Add `builder.add_edge("my_criteria_node", "build_prompt")` (fan-in).
  6. Add the new field to AgentState and consume it in build_prompt.

Adding any other new node (e.g. human_review, re_rank):
  1. Write it in nodes.py.
  2. Import + register it.
  3. Insert it into the sequential chain after validate_output or wherever needed.
"""

from __future__ import annotations

import functools
import logging
from typing import Literal

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.nodes import (
    load_call,
    load_behavioral_criteria,
    load_compliance_pillars,
    load_script_templates,
    load_scoring_weights,
    build_prompt,
    llm_inference,
    parse_response,
    validate_output,
    integrity_check,
    finalize,
    handle_error,
)
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Router — continue happy path or jump to handle_error
# ─────────────────────────────────────────────────────────────────────────────

def _error_router(state: AgentState) -> Literal["continue", "handle_error"]:
    """Conditional edge after every fallible node."""
    if state.get("error"):
        return "handle_error"
    return "continue"


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def build_qa_graph(llm_client: LLMClient) -> StateGraph:
    """
    Compile and return the LangGraph StateGraph for QA analysis.

    LLMClient is injected via functools.partial so the graph is provider-agnostic
    and can be rebuilt with a different provider at any time.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────

    # Stage 1: entry + validation
    builder.add_node("load_call", load_call)

    # Stage 2: criteria loaders (run in parallel after load_call)
    builder.add_node("load_behavioral_criteria", load_behavioral_criteria)
    builder.add_node("load_compliance_pillars", load_compliance_pillars)
    builder.add_node("load_script_templates", load_script_templates)
    builder.add_node("load_scoring_weights", load_scoring_weights)

    # Stage 3: prompt assembly (barrier — waits for all 4 loaders)
    builder.add_node("build_prompt", build_prompt)

    # Stage 4: LLM + response processing chain
    builder.add_node(
        "llm_inference",
        functools.partial(llm_inference, llm_client=llm_client),
    )
    builder.add_node("parse_response", parse_response)
    builder.add_node("validate_output", validate_output)
    builder.add_node("integrity_check", integrity_check)
    builder.add_node("finalize", finalize)

    # Error sink
    builder.add_node("handle_error", handle_error)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "load_call")

    # load_call → error check (conditional) then fan-out to 4 parallel loaders.
    # Pattern: conditional edge handles the error path; direct edges handle the
    # success fan-out.  LangGraph only executes the nodes that load_call actually
    # routes to — if _error_router returns "handle_error" the loaders are skipped.
    builder.add_conditional_edges(
        "load_call",
        _error_router,
        {"continue": "load_behavioral_criteria", "handle_error": "handle_error"},
    )
    # The remaining 3 loaders are wired with direct edges so they run in parallel
    # with load_behavioral_criteria on the success path.
    builder.add_edge("load_call", "load_compliance_pillars")
    builder.add_edge("load_call", "load_script_templates")
    builder.add_edge("load_call", "load_scoring_weights")

    # fan-in: all 4 loaders → build_prompt (LangGraph waits for all four to finish)
    builder.add_edge("load_behavioral_criteria", "build_prompt")
    builder.add_edge("load_compliance_pillars",  "build_prompt")
    builder.add_edge("load_script_templates",    "build_prompt")
    builder.add_edge("load_scoring_weights",     "build_prompt")

    # Sequential chain: build_prompt → llm_inference → parse → validate → …
    builder.add_conditional_edges(
        "build_prompt",
        _error_router,
        {"continue": "llm_inference", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "llm_inference",
        _error_router,
        {"continue": "parse_response", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "parse_response",
        _error_router,
        {"continue": "validate_output", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "validate_output",
        _error_router,
        {"continue": "integrity_check", "handle_error": "handle_error"},
    )

    # Safe tail nodes (no error conditions possible)
    builder.add_edge("integrity_check", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("handle_error", END)

    return builder.compile()
