"""
LangGraph Pipeline Graph for the Call QA Analysis System.

Graph topology (happy path):
─────────────────────────────────────────────────────────────────────────────
  START
    │
    ▼
  load_call ──(error)──────────────────────────────────────────────────────┐
    │                                                                       │
    │  fan-out: 5 parallel criteria loaders                                 │
    ├──→ load_behavioral_criteria ──┐                                       │
    ├──→ load_compliance_pillars  ──┤                                       │
    ├──→ load_script_templates    ──┤ fan-in                                │
    ├──→ load_reservation_pillars ──┤                                       │
    └──→ load_scoring_weights     ──┘                                       │
                                   │                                        │
                            criteria_ready  (barrier / no-op)               │
                                   │                                        │
    └──→ detect_intent                                                        │
              │                                                               │
              ├──(booking)──→ extract_appointment_details                     │
              │                        │                                      │
              │               verify_appointment_in_db                       │
              │                        │                                      │
              │               infer_reservation_evaluation ──(error)─────────┤
              │                        │                                      │
              └──(skip_booking)────────┘                                      │
                               │  (sequential — completes before LLM calls)  │
                               ▼                                              │
    fan-out: 2 parallel LLM calls (after booking branch fully done)           │
    ├──→ infer_behavioral_evaluation ──(error)──────────────────────────┐    │
    └──→ infer_compliance_evaluation ──(error)──────────────────────────┤    │
                                                    │ fan-in (2 edges)  │    │
                                                    ▼                   │    │
                                            inference_ready             │    │
                                     (barrier — waits for exactly 2)    │    │
                                                    │                   │    │
                                                    ▼                   │    │
                  infer_overall_scoring ──(error)───────────────────────┘    │
                               │                                         │  │
                               ▼                                         │  │
                      aggregate_results ──(error)────────────────────────┤  │
                               │                                         │  │
                               ▼                                         │  │
                      integrity_check                                    │  │
                               │                                         │  │
                               ▼                                         │  │
                            finalize                          ┌───────────┘  │
                               │                             │               │
                              END ◄─── handle_error ◄────────┘               │
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
    load_reservation_pillars,
    load_script_templates,
    load_scoring_weights,
    infer_behavioral_evaluation,
    infer_compliance_evaluation,
    infer_reservation_evaluation,
    infer_script_matching,
    infer_overall_scoring,
    aggregate_results,
    integrity_check,
    finalize,
    handle_error,
    detect_intent,
    extract_appointment_details,
    verify_appointment_in_db,
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


def _booking_router(state: AgentState) -> Literal["booking", "skip_booking"]:
    """Route to the booking sub-flow when a booking intent was detected."""
    if state.get("is_booking_intent"):
        return "booking"
    return "skip_booking"


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
    builder.add_node("load_reservation_pillars", load_reservation_pillars)
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
    # infer_script_matching is currently disabled (returns pass) — excluded
    # from the graph so it does not corrupt the inference_ready barrier count.
    # Re-add it here + wire it when the script-matching LLM call is enabled.

    builder.add_node(
        "infer_reservation_evaluation",
        functools.partial(infer_reservation_evaluation, llm_client=llm_client),
    )

    # Stage 4: scoring synthesis (fan-in barrier — waits for all 4 focused nodes)
    builder.add_node(
        "infer_overall_scoring",
        functools.partial(infer_overall_scoring, llm_client=llm_client),
    )

    # Barrier node 1 — fan-in for all criteria loaders, fan-out to inference.
    builder.add_node("criteria_ready", lambda state: {})

    # Barrier node 2 — fan-in that waits for:
    #   • infer_behavioral_evaluation
    #   • infer_compliance_evaluation
    # These two are the only unconditional predecessors. The booking branch
    # runs sequentially BEFORE these two start (detect_intent is gated by
    # criteria_ready too, but wired separately — see edge section below).
    builder.add_node("inference_ready", lambda state: {})

    # Stage 5: aggregate + validate merged result
    builder.add_node("aggregate_results", aggregate_results)

    # Stage 6: post-processing chain
    builder.add_node("integrity_check", integrity_check)
    builder.add_node("finalize", finalize)

    # Error sink
    builder.add_node("handle_error", handle_error)

    # ── Booking intent sub-flow (inserted after infer_behavioral_evaluation) ──
    builder.add_node("detect_intent", detect_intent)
    builder.add_node(
        "extract_appointment_details",
        functools.partial(extract_appointment_details, llm_client=llm_client),
    )
    builder.add_node("verify_appointment_in_db", verify_appointment_in_db)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "load_call")

    # load_call → error check (conditional) then fan-out to all 5 criteria loaders.
    builder.add_conditional_edges(
        "load_call",
        _error_router,
        {"continue": "load_behavioral_criteria", "handle_error": "handle_error"},
    )
    builder.add_edge("load_call", "load_compliance_pillars")
    #builder.add_edge("load_call", "load_script_templates")
    builder.add_edge("load_call", "load_reservation_pillars")
    builder.add_edge("load_call", "load_scoring_weights")

    # ── All 5 loaders fan-in to the barrier node ──────────────────────────
    #
    # criteria_ready is a no-op that LangGraph uses as a synchronisation
    # point: it fires only after every loader has written its state key.
    # From there we fan-out to the 4 parallel inference nodes + detect_intent.
    # This replaces the previous 5×5 = 25 repeated edges with 5 + 5 = 10.
    builder.add_edge("load_behavioral_criteria", "criteria_ready")
    builder.add_edge("load_compliance_pillars",  "criteria_ready")
    #builder.add_edge("load_script_templates",    "criteria_ready")
    builder.add_edge("load_reservation_pillars", "criteria_ready")
    builder.add_edge("load_scoring_weights",     "criteria_ready")

    # ── Step 1: detect_intent runs first (sequential, after all loaders) ──
    #
    # detect_intent is the first thing that fires after criteria_ready.
    # Its result (is_booking_intent) determines which branch runs next.
    builder.add_edge("criteria_ready", "detect_intent")

    # ── Step 2: booking sub-flow OR skip — both end at infer_behavioral ───
    #
    # BOOKING:    detect_intent → extract → verify → infer_reservation_evaluation
    #             → infer_behavioral_evaluation (fan-in start)
    # SKIP:       detect_intent → infer_behavioral_evaluation directly
    #
    # This ensures infer_reservation_evaluation ALWAYS completes (or is
    # skipped) BEFORE the parallel behavioral+compliance LLM calls begin.
    builder.add_conditional_edges(
        "detect_intent",
        _booking_router,
        {
            "booking":      "extract_appointment_details",
            "skip_booking": "infer_behavioral_evaluation",
        },
    )
    builder.add_edge("extract_appointment_details", "verify_appointment_in_db")
    builder.add_edge("verify_appointment_in_db",    "infer_reservation_evaluation")
    builder.add_conditional_edges(
        "infer_reservation_evaluation",
        _error_router,
        {"continue": "infer_behavioral_evaluation", "handle_error": "handle_error"},
    )

    # ── Step 3: behavioral + compliance run in parallel ────────────────────
    #
    # Both receive a single incoming edge from detect_intent/reservation branch.
    # criteria_ready also fans into them so they wait for all loaders.
    builder.add_edge("criteria_ready", "infer_behavioral_evaluation")
    builder.add_edge("criteria_ready", "infer_compliance_evaluation")
    # infer_script_matching excluded (disabled) — add back here when re-enabled

    # ── Step 4: behavioral + compliance fan-in → inference_ready ──────────
    #
    # Exactly 2 unconditional predecessors: behavioral + compliance.
    # LangGraph fires inference_ready only after both complete.
    builder.add_conditional_edges(
        "infer_behavioral_evaluation",
        _error_router,
        {"continue": "inference_ready", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_compliance_evaluation",
        _error_router,
        {"continue": "inference_ready", "handle_error": "handle_error"},
    )

    # ── Step 5: inference_ready → infer_overall_scoring ───────────────────
    builder.add_edge("inference_ready", "infer_overall_scoring")
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
