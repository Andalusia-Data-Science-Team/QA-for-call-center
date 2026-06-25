"""
LangGraph Pipeline Graph for the Call QA Analysis System.

Graph topology (happy path):
─────────────────────────────────────────────────────────────────────────────
  START
    │
    ▼
  load_call ──(error)─────────────────────────────────────────────────────┐
    │                                                                      │
    │  fan-out: 4 parallel criteria loaders                                │
    ├──→ load_behavioral_criteria ──┐                                      │
    ├──→ load_compliance_pillars  ──┤                                      │
    ├──→ load_script_templates    ──┤                                      │
    └──→ load_scoring_weights     ──┘                                      │
                 (all 4 loaders fan-in into the 3 focused inference nodes) │
                                                                           │
    fan-out: 3 parallel focused LLM calls                                  │
    ├──→ infer_behavioral_evaluation ──(error)────────────────────────────┤│
    ├──→ infer_compliance_evaluation ──(error)────────────────────────────┤│
    └──→ infer_script_matching       ──(error)────────────────────────────┤│
                                │ (fan-in)                                 ││
                                ▼                                          ││
                   infer_overall_scoring ──(error)────────────────────────┤│
                                │                                          ││
                                ▼                                          ││
                       aggregate_results ──(error)────────────────────────┤│
                                │                                          ││
                                ▼                                          ││
                       integrity_check                                     ││
                                │                                          ││
                                ▼                                          ││
                             finalize                                      ││
                                │                            ┌─────────────┘│
                               END ◄─── handle_error ◄──────┘               │
                                            ▲                                │
                                            └────────────────────────────────┘
─────────────────────────────────────────────────────────────────────────────

Topology details
────────────────
Stage 1 (criteria loaders, parallel):
  load_behavioral_criteria, load_compliance_pillars,
  load_script_templates, load_scoring_weights
  → Pure YAML reads (lru_cached). Negligible overhead.

Stage 2 (focused LLM calls, parallel):
  infer_behavioral_evaluation  — tone, empathy, professionalism, red flags
  infer_compliance_evaluation  — 15 compliance pillars (C2Com/C2C/C2B/NC)
  infer_script_matching        — greeting / closing script adherence
  Each node calls the LLM with a narrow prompt and stores a partial result.

Stage 3 (sequential synthesis):
  infer_overall_scoring   — synthesises the three sub-results + scoring weights
  aggregate_results       — merges all four dicts → QAAnalysisResult (Pydantic)
  integrity_check         — fixes escalation_required ↔ overall_assessment
  finalize                — logs summary, closes trace

Error path:
  Any node that sets state["error"] is immediately routed to handle_error → END.

Adding a new criteria node:
  1. Write the async loader in nodes.py.
  2. Import + register it.
  3. Add fan-out edge from load_call and fan-in edge to infer_behavioral_evaluation
     (or infer_compliance_evaluation / infer_script_matching as appropriate).
  4. Add the new field to AgentState and consume it in the relevant prompt builder.

Adding a new focused inference node:
  1. Write it in nodes.py + add a prompt builder in qa_prompt.py.
  2. Import + register it.
  3. Wire fan-out edges from all criteria loaders it needs.
  4. Wire a fan-in edge from it to infer_overall_scoring.
  5. Pass its result into build_scoring_prompt() in nodes.py.
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
    infer_behavioral_evaluation,
    infer_compliance_evaluation,
    infer_script_matching,
    infer_overall_scoring,
    aggregate_results,
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
    builder.add_node("load_compliance_pillars",  load_compliance_pillars)
    builder.add_node("load_script_templates",    load_script_templates)
    builder.add_node("load_scoring_weights",     load_scoring_weights)

    # Stage 3: focused LLM inference nodes (run in parallel after all loaders)
    builder.add_node(
        "infer_behavioral_evaluation",
        functools.partial(infer_behavioral_evaluation, llm_client=llm_client),
    )
    builder.add_node(
        "infer_compliance_evaluation",
        functools.partial(infer_compliance_evaluation, llm_client=llm_client),
    )
    builder.add_node(
        "infer_script_matching",
        functools.partial(infer_script_matching, llm_client=llm_client),
    )

    # Stage 4: scoring synthesis (fan-in barrier — waits for all 3 focused nodes)
    builder.add_node(
        "infer_overall_scoring",
        functools.partial(infer_overall_scoring, llm_client=llm_client),
    )

    # Stage 5: aggregate + validate merged result
    builder.add_node("aggregate_results", aggregate_results)

    # Stage 6: post-processing chain
    builder.add_node("integrity_check", integrity_check)
    builder.add_node("finalize", finalize)

    # Error sink
    builder.add_node("handle_error", handle_error)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "load_call")

    # load_call → error check (conditional) then fan-out to all 4 criteria loaders.
    builder.add_conditional_edges(
        "load_call",
        _error_router,
        {"continue": "load_behavioral_criteria", "handle_error": "handle_error"},
    )
    builder.add_edge("load_call", "load_compliance_pillars")
    builder.add_edge("load_call", "load_script_templates")
    builder.add_edge("load_call", "load_scoring_weights")

    # ── Fan-in from criteria loaders → 3 focused inference nodes ─────────
    #
    # Each focused node needs only a subset of criteria, but we wire ALL four
    # loaders to ALL three inference nodes so LangGraph's barrier waits for
    # every loader to finish before any inference node starts.  The extra
    # criteria keys are simply ignored by each focused prompt builder.
    #
    # Behavioral inference needs: behavioral_criteria
    builder.add_edge("load_behavioral_criteria", "infer_behavioral_evaluation")
    builder.add_edge("load_compliance_pillars",  "infer_behavioral_evaluation")
    builder.add_edge("load_script_templates",    "infer_behavioral_evaluation")
    builder.add_edge("load_scoring_weights",     "infer_behavioral_evaluation")

    # Compliance inference needs: compliance_pillars
    builder.add_edge("load_behavioral_criteria", "infer_compliance_evaluation")
    builder.add_edge("load_compliance_pillars",  "infer_compliance_evaluation")
    builder.add_edge("load_script_templates",    "infer_compliance_evaluation")
    builder.add_edge("load_scoring_weights",     "infer_compliance_evaluation")

    # Script inference needs: script_templates
    builder.add_edge("load_behavioral_criteria", "infer_script_matching")
    builder.add_edge("load_compliance_pillars",  "infer_script_matching")
    builder.add_edge("load_script_templates",    "infer_script_matching")
    builder.add_edge("load_scoring_weights",     "infer_script_matching")

    # ── Focused inference nodes → error check → infer_overall_scoring ────
    builder.add_conditional_edges(
        "infer_behavioral_evaluation",
        _error_router,
        {"continue": "infer_overall_scoring", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_compliance_evaluation",
        _error_router,
        {"continue": "infer_overall_scoring", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_script_matching",
        _error_router,
        {"continue": "infer_overall_scoring", "handle_error": "handle_error"},
    )

    # ── infer_overall_scoring → aggregate_results → integrity_check ───────
    builder.add_conditional_edges(
        "infer_overall_scoring",
        _error_router,
        {"continue": "aggregate_results", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "aggregate_results",
        _error_router,
        {"continue": "integrity_check", "handle_error": "handle_error"},
    )

    # Safe tail nodes (no error conditions possible)
    builder.add_edge("integrity_check", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("handle_error", END)

    return builder.compile()
