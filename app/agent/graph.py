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
                               │  (both paths converge here)                  │
                               ▼                                              │
                        inference_gate  (barrier — single fan-out point)      │
                               │                                              │
    fan-out: 2 parallel LLM calls                                             │
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
  infer_offer_evaluation       — offer recommendation (via fetch_crm_offers_for_call)
  infer_service_evaluation     — service recommendation (via fetch_crm_services_for_call)
  infer_package_evaluation     — package recommendation (via fetch_crm_packages_for_call)
  Each node calls the LLM with a narrow prompt and stores a partial result.

Stage 3 (sequential synthesis):
  infer_overall_scoring   — synthesises all sub-results + scoring weights
  aggregate_results       — merges all dicts → QAAnalysisResult (Pydantic)
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
    load_offer_pillars,
    load_script_templates,
    load_scoring_weights,
    infer_behavioral_evaluation,
    infer_compliance_evaluation,
    infer_reservation_evaluation,
    enforce_ineligible_reservation_violation,
    infer_offer_evaluation,
    infer_service_evaluation,
    infer_package_evaluation,
    fetch_crm_offers_for_call,
    fetch_crm_services_for_call,
    fetch_crm_packages_for_call,
    infer_script_matching,
    infer_overall_scoring,
    aggregate_results,
    integrity_check,
    save_to_database,
    finalize,
    handle_error,
    detect_intent,
    detect_insurance_intent,
    extract_appointment_details,
    verify_appointment_in_db,
    check_patient_eligibility,
    handle_ineligible_patient,
    _eligibility_router,
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


def _booking_router(state: AgentState) -> Literal["booking", "offer_only", "skip_booking"]:
    """Route the general booking intent without inspecting insurance status."""
    if state.get("is_booking_intent"):
        return "booking"
    if state.get("is_offer_intent"):
        return "offer_only"
    return "skip_booking"


def _insurance_router(state: AgentState) -> Literal["insurance", "continue"]:
    """Prioritize a separately detected insurance intent before booking routing."""
    if state.get("is_insurance_intent"):
        return "insurance"
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
    builder.add_node("load_reservation_pillars", load_reservation_pillars)
    builder.add_node("load_offer_pillars",       load_offer_pillars)
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
    builder.add_node(
        "infer_reservation_evaluation",
        functools.partial(infer_reservation_evaluation, llm_client=llm_client),
    )
    builder.add_node("fetch_crm_offers_for_call",   fetch_crm_offers_for_call)
    builder.add_node("fetch_crm_services_for_call",  fetch_crm_services_for_call)
    builder.add_node("fetch_crm_packages_for_call",  fetch_crm_packages_for_call)
    builder.add_node(
        "infer_offer_evaluation",
        functools.partial(infer_offer_evaluation, llm_client=llm_client),
    )
    builder.add_node(
        "infer_service_evaluation",
        functools.partial(infer_service_evaluation, llm_client=llm_client),
    )
    builder.add_node(
        "infer_package_evaluation",
        functools.partial(infer_package_evaluation, llm_client=llm_client),
    )

    # Stage 4: scoring synthesis (fan-in barrier — waits for all 3 focused nodes)
    builder.add_node(
        "infer_overall_scoring",
        functools.partial(infer_overall_scoring, llm_client=llm_client),
    )

    # Barrier node 1 — fan-in for all criteria loaders, fan-out to inference.
    builder.add_node("criteria_ready", lambda state: {})

    # Barrier node 1b — fan-in that waits for the booking/skip branch to
    # fully complete before the three parallel LLM calls start.  This is the
    # SINGLE entry point into behavioral + compliance + script_matching, preventing any second
    # trigger from arriving via a different path.
    builder.add_node("inference_gate", lambda state: {})

    # Barrier node 2 — fan-in that waits for:
    #   • infer_behavioral_evaluation     (direct from inference_gate)
    #   • infer_compliance_evaluation     (direct from inference_gate)
    #   • infer_offer_evaluation          (via fetch_crm_offers_for_call)
    #   • infer_service_evaluation        (via fetch_crm_services_for_call)
    #   • infer_package_evaluation        (via fetch_crm_packages_for_call)
    #   • infer_script_matching           (direct from inference_gate)
    # Exactly 6 unconditional predecessors.
    builder.add_node("inference_ready", lambda state: {})

    # Stage 5: aggregate + validate merged result
    builder.add_node("aggregate_results", aggregate_results)

    # Stage 6: post-processing chain
    builder.add_node("integrity_check", integrity_check)
    builder.add_node("save_to_database", save_to_database)
    builder.add_node("finalize", finalize)

    # Error sink
    builder.add_node("handle_error", handle_error)

    # ── Booking intent sub-flow (inserted after infer_behavioral_evaluation) ──
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("detect_insurance_intent", detect_insurance_intent)
    builder.add_node("route_booking_intent", lambda state: {})
    builder.add_node(
        "extract_appointment_details",
        functools.partial(extract_appointment_details, llm_client=llm_client),
    )
    builder.add_node("verify_appointment_in_db", verify_appointment_in_db)
    builder.add_node("enforce_ineligible_reservation_violation", enforce_ineligible_reservation_violation)
    builder.add_node("check_patient_eligibility", check_patient_eligibility)
    builder.add_node("handle_ineligible_patient", handle_ineligible_patient)

    # ── Edges ─────────────────────────────────────────────────────────────

    # Entry
    builder.add_edge(START, "load_call")

    # load_call → error check (conditional) then fan-out to all 6 criteria loaders.
    builder.add_conditional_edges(
        "load_call",
        _error_router,
        {"continue": "load_behavioral_criteria", "handle_error": "handle_error"},
    )
    builder.add_edge("load_call", "load_compliance_pillars")
    builder.add_edge("load_call", "load_script_templates")
    builder.add_edge("load_call", "load_reservation_pillars")
    builder.add_edge("load_call", "load_offer_pillars")
    builder.add_edge("load_call", "load_scoring_weights")

    # ── All 6 loaders fan-in to the barrier node ──────────────────────────
    #
    # criteria_ready is a no-op that LangGraph uses as a synchronisation
    # point: it fires only after every loader has written its state key.
    # From there we fan-out to the inference nodes + detect_intent.
    builder.add_edge("load_behavioral_criteria", "criteria_ready")
    builder.add_edge("load_compliance_pillars",  "criteria_ready")
    builder.add_edge("load_script_templates",    "criteria_ready")
    builder.add_edge("load_reservation_pillars", "criteria_ready")
    builder.add_edge("load_offer_pillars",       "criteria_ready")
    builder.add_edge("load_scoring_weights",     "criteria_ready")

    # ── Step 1: detect_intent runs first (sequential, after all loaders) ──
    #
    # detect_intent is the first thing that fires after criteria_ready.
    # Its result (is_booking_intent) determines which branch runs next.
    builder.add_edge("criteria_ready", "detect_intent")

    # ── Step 2: independent insurance intent, then booking routing ────────
    #
    # Identity/IQAMA requests are insurance intent unless the patient explicitly
    # selected cash or stated they are uninsured. The detector always runs; a
    # successful check returns to the unchanged booking router.
    builder.add_edge("detect_intent", "detect_insurance_intent")
    builder.add_conditional_edges(
        "detect_insurance_intent",
        _insurance_router,
        {
            "insurance": "check_patient_eligibility",
            "continue":  "route_booking_intent",
        },
    )
    builder.add_conditional_edges(
        "check_patient_eligibility",
        _eligibility_router,
        {
            "eligible":     "route_booking_intent",
            "not_eligible": "handle_ineligible_patient",
            "error":        "handle_error",
        },
    )
    builder.add_conditional_edges(
        "route_booking_intent",
        _booking_router,
        {
            "booking":      "extract_appointment_details",
            "offer_only":   "extract_appointment_details",
            "skip_booking": "inference_gate",
        },
    )

    # Ineligible bookings are still extracted and verified to catch an improper reservation.
    builder.add_edge("handle_ineligible_patient", "extract_appointment_details")
    # Booking path: extract → verify DB → infer_reservation → inference_gate
    # Offer-only path: extract → inference_gate (skips DB + reservation eval)
    builder.add_node("offer_extraction_done", lambda state: {})  # no-op barrier
    builder.add_conditional_edges(
        "extract_appointment_details",
        lambda s: "booking" if s.get("is_booking_intent") else "offer_only",
        {
            "booking":    "verify_appointment_in_db",
            "offer_only": "offer_extraction_done",
        },
    )
    builder.add_edge("offer_extraction_done",       "inference_gate")
    builder.add_edge("verify_appointment_in_db",    "infer_reservation_evaluation")
    builder.add_conditional_edges(
        "infer_reservation_evaluation",
        _error_router,
        {"continue": "enforce_ineligible_reservation_violation", "handle_error": "handle_error"},
    )
    builder.add_edge("enforce_ineligible_reservation_violation", "inference_gate")

    # ── Step 3: inference_gate fan-out ────────────────────────────────────
    #
    # Three nodes run in PARALLEL directly from inference_gate:
    #   behavioral, compliance, script_matching.
    #
    # The offer/service/package paths each have one extra sequential step:
    #   inference_gate → fetch_crm_offers_for_call → infer_offer_evaluation
    #   inference_gate → fetch_crm_services_for_call → infer_service_evaluation
    #   inference_gate → fetch_crm_packages_for_call → infer_package_evaluation
    # This ensures live CRM data is fetched BEFORE the LLM prompt runs.
    builder.add_edge("inference_gate", "infer_behavioral_evaluation")
    builder.add_edge("inference_gate", "infer_compliance_evaluation")
    builder.add_edge("inference_gate", "infer_script_matching")
    builder.add_edge("inference_gate", "fetch_crm_offers_for_call")
    builder.add_edge("fetch_crm_offers_for_call", "infer_offer_evaluation")
    # Services and packages run with their fetch → infer chain in parallel
    builder.add_edge("inference_gate", "fetch_crm_services_for_call")
    builder.add_edge("fetch_crm_services_for_call", "infer_service_evaluation")
    builder.add_edge("inference_gate", "fetch_crm_packages_for_call")
    builder.add_edge("fetch_crm_packages_for_call", "infer_package_evaluation")

    # ── Step 4: all four evaluations fan-in → inference_ready ─────────────
    #
    # LangGraph fires inference_ready only after all four complete.
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
    builder.add_conditional_edges(
        "infer_offer_evaluation",
        _error_router,
        {"continue": "inference_ready", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_script_matching",
        _error_router,
        {"continue": "inference_ready", "handle_error": "handle_error"},
    )
    builder.add_edge("fetch_crm_services_for_call",  "inference_ready")
    builder.add_edge("fetch_crm_packages_for_call",  "inference_ready")

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
    builder.add_edge("integrity_check", "save_to_database")
    builder.add_edge("save_to_database", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("handle_error", END)

    return builder.compile()
