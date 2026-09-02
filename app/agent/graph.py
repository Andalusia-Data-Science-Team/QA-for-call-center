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
              │  bank-intent router (conditional edge, reuses                │
              │  bank_validation_needed — the SAME gate the node itself      │
              │  uses, not a duplicate check):                               │
              ├──(intent present)────→ validate_bank_information ──┐         │
              ├──(no intent)──────────→ skip_bank_validation ───────┤        │
              │                        (no CRM fetch, no BU          │       │
              │                         resolution, no node_trace)   │       │
              │                                                      │       │
              │  location-intent router (independent conditional edge,       │
              │  same pattern, reuses location_validation_needed):           │
              ├──(intent present)────→ validate_location ────────────┤ fan-in│
              └──(no intent)──────────→ skip_location_validation ────┤       │
                                        (no CRM fetch, no             │       │
                                         node_trace entry)            │       │
                                                                       │       │
                                                       loc_bank_ready (barrier)│
                                               │                               │
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
                     fetch_crm_offers_for_call                                │
              (shared 1-hop pass-through — see below)                        │
                               │                                              │
    fan-out: 5 parallel branches, ALL exactly one hop past the fetch —        │
    this equal hop-count is required so inference_ready's fan-in merges      │
    all five arrivals into ONE superstep (see Step 3 comment in code: a      │
    mismatched hop-count here previously fired inference_ready — and every-  │
    thing downstream of it — TWICE per call). The doctor-scope branch is    │
    conditional (_doctor_scope_intent_router): EITHER                       │
    infer_doctor_scope_validation OR skip_doctor_scope_validation runs,     │
    never both — so it's still exactly ONE of the five arrivals.            │
    ├──→ infer_behavioral_evaluation ──(error)──────────────────────────┐    │
    ├──→ infer_compliance_evaluation ──(error)──────────────────────────┤    │
    ├──→ infer_script_matching       ──(error)──────────────────────────┤    │
    ├──→ infer_offer_evaluation      ──(error)──────────────────────────┤    │
    └──(doctor resolved + clinical need)→ infer_doctor_scope_validation ┤    │
       (else)──────────────────────────→ skip_doctor_scope_validation  ┤    │
                                                    │ fan-in (5 edges)  │    │
                                                    ▼                   │    │
                                            inference_ready             │    │
                                     (barrier — waits for exactly 5)    │    │
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
    load_offer_pillars,
    load_script_templates,
    load_scoring_weights,
    infer_behavioral_evaluation,
    infer_compliance_evaluation,
    infer_reservation_evaluation,
    infer_offer_evaluation,
    fetch_crm_offers_for_call,
    infer_script_matching,
    infer_overall_scoring,
    aggregate_results,
    integrity_check,
    save_to_database,
    finalize,
    handle_error,
    detect_intent,
    extract_appointment_details,
    verify_appointment_in_db,
    validate_bank_information_node,
    validate_location_node,
    skip_location_validation,
    skip_bank_validation,
    validate_doctor_node,
    skip_doctor_validation,
    infer_doctor_scope_validation,
    skip_doctor_scope_validation,
)
from app.service_hub.bank_validation import detect_bank_signals, bank_validation_needed
from app.service_hub.location_validation import detect_location_signals, location_validation_needed
from app.service_hub.doctor_validation import (
    classify_doctor_context,
    describe_doctor_extraction_evidence,
    detect_doctor_signals,
    doctor_scope_skip_reason,
    doctor_scope_validation_needed,
    patient_describes_medical_complaint,
    raw_doctor_title_tails,
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


def _bank_intent_router(state: AgentState) -> Literal["validate_bank_information", "skip_bank"]:
    """Route to validate_bank_information only when there is actual bank
    intent — reusing app.service_hub.bank_validation.bank_validation_needed /
    detect_bank_signals, the SAME deterministic gate the node itself uses
    internally, not a second copy of the logic (supported business unit AND
    a bank request or agent-supplied financial identifier). When it doesn't
    hold, validate_bank_information is skipped entirely at the graph level:
    it never executes, never appears in node_trace, never triggers a CRM
    bank-account fetch, and never resolves a business unit against CRM
    bank data."""
    call = state["call"]
    signals = detect_bank_signals(call)
    needed = bank_validation_needed(call, signals)
    logger.info("bank intent routing | call_id=%s bank_validation_needed=%s", call.call_id, needed)
    if needed:
        return "validate_bank_information"
    return "skip_bank"


def _location_intent_router(state: AgentState) -> Literal["validate_location", "skip_location"]:
    """Route to validate_location only when there is actual location
    intent — patient_has_location_intent OR
    agent_has_location_information (app.service_hub.location_validation.
    location_validation_needed / detect_location_intent) — reusing the
    SAME deterministic gate the node itself uses internally, not a second
    copy of the logic. When neither holds, validate_location is skipped
    entirely at the graph level: it never executes, never appears in
    node_trace, and never triggers a CRM location fetch."""
    call = state["call"]
    signals = detect_location_signals(call)
    if location_validation_needed(call, signals):
        return "validate_location"
    return "skip_location"


def _doctor_intent_router(state: AgentState) -> Literal["validate_doctor", "skip_doctor"]:
    """Route to validate_doctor only when the ACTIVE conversational intent
    is specifically about a named doctor — reusing app.service_hub.
    doctor_validation.classify_specific_doctor_intent (via
    doctor_validation_needed/classify_doctor_context), the SAME
    deterministic gate the node itself uses internally, not a second copy
    of the logic. A named doctor merely mentioned — as the ordering/
    referring physician for a different service, an existing/follow-up
    relationship, or any other incidental reference — never triggers a CRM
    fetch; only booking/rescheduling/availability WITH that doctor, a
    booking confirmation naming them, an Agent recommendation, or an
    inquiry ABOUT them does. Independent of the bank/location routers
    above — all three are separate conditional edges off detect_intent, so
    every combination (none/bank only/location only/doctor only/any pair/
    all three) is supported without one suppressing another.

    Logs the full pre-routing breakdown BEFORE any CRM doctor fetch could
    happen, so a production call that skipped (or ran) doctor validation is
    always explainable from this one line — see classify_specific_doctor_
    intent's docstring for what each field means.

    This router only decides which edge to take — it is a cheap,
    regex-only pre-check, not the authoritative doctor-name extraction.
    The clean "[doctor] extraction:"/"[doctor] routing:"/"[doctor]
    resolution:"/"[doctor] outcome:" blocks (INFO level) are printed
    exactly once, downstream, by validate_doctor_node (when this router
    sends the call there) or skip_doctor_validation (when it doesn't) —
    never here, and never twice. The raw per-fragment candidate/rejection
    dump this used to print unconditionally at INFO is now DEBUG-only."""
    call = state["call"]
    ctx = classify_doctor_context(call)
    needed = ctx["doctor_intent"] != "not_applicable"
    call_bu = getattr(call, "business_unit", None)
    logger.info(
        "doctor pre-routing | call_id=%s business_unit=%s doctor_role=%s doctor_validation_needed=%s reason=%s",
        call.call_id, call_bu, ctx["doctor_role"], needed, ctx["reason"],
    )
    if logger.isEnabledFor(logging.DEBUG):
        raw_patient, raw_agent = raw_doctor_title_tails(call)
        evidence = describe_doctor_extraction_evidence(call)
        logger.debug(
            "doctor pre-routing diagnostics | call_id=%s raw_patient_candidates=%s raw_agent_candidates=%s "
            "rejected=%s patient_candidates=%s agent_candidates=%s recommended_doctors=%s "
            "patient_selected_doctor=%s booking_target=%s inquiry_target=%s",
            call.call_id, raw_patient, raw_agent, evidence["rejected"], ctx["patient_candidates"],
            ctx["agent_candidates"], ctx.get("recommended_doctors"), ctx.get("patient_selected_doctor"),
            ctx["booking_target"], ctx["inquiry_target"],
        )
    return "validate_doctor" if needed else "skip_doctor"


def _doctor_scope_intent_router(state: AgentState) -> Literal["infer_doctor_scope_validation", "skip_doctor_scope"]:
    """Route to the semantic infer_doctor_scope_validation node ONLY when a
    doctor was successfully resolved deterministically AND the patient
    described a genuine clinical need — reusing
    app.service_hub.doctor_validation.doctor_scope_validation_needed /
    doctor_scope_skip_reason, the SAME gate infer_doctor_scope_validation
    itself uses internally as a defensive fallback, not a second copy of
    the logic. Placed on the conditional edge out of
    fetch_crm_offers_for_call (not detect_intent) because doctor_validation
    — written earlier by validate_doctor_node/skip_doctor_validation before
    the booking split — must already be in state before this decision can
    be made. When the gate fails, infer_doctor_scope_validation never
    executes at all: no clinical-need extraction, no scope-reference JSON,
    no LLM prompt, no LLM call, and no node_trace entry for it — see
    skip_doctor_scope_validation. Both branches are exactly one hop past
    fetch_crm_offers_for_call (mirroring _bank_intent_router/
    _location_intent_router/_doctor_intent_router's either/or pattern), so
    this preserves the equal-hop-count invariant inference_ready's fan-in
    depends on (see the Step 3 comment below)."""
    call = state["call"]
    doctor_result = state.get("doctor_validation")
    _p_mentions, _a_mentions, patient_text, _agent_text = detect_doctor_signals(call)
    needed = doctor_scope_validation_needed(doctor_result, patient_text, call)
    reason = doctor_scope_skip_reason(doctor_result, patient_text, call)
    doctor_role = classify_doctor_context(call)["doctor_role"]
    logger.info(
        "doctor_scope routing | call_id=%s needed=%s doctor_key=%s doctor_role=%s reason=%s",
        call.call_id, needed, (doctor_result or {}).get("doctor_key"), doctor_role, reason,
    )
    print(
        f"[doctor_scope] routing: doctor_key={(doctor_result or {}).get('doctor_key')} "
        f"doctor_role={doctor_role} clinical_need_detected={patient_describes_medical_complaint(patient_text)} "
        f"needed={needed} reason={reason}",
        flush=True,
    )
    return "infer_doctor_scope_validation" if needed else "skip_doctor_scope"


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
    builder.add_node("fetch_crm_offers_for_call", fetch_crm_offers_for_call)
    builder.add_node(
        "infer_offer_evaluation",
        functools.partial(infer_offer_evaluation, llm_client=llm_client),
    )
    # Semantic doctor-recommendation-suitability check — separate from the
    # deterministic validate_doctor node above (which already ran and wrote
    # state["doctor_validation"] before the booking split). Applicability
    # (doctor resolved + scope evidence + patient complaint) is decided at
    # the GRAPH level by _doctor_scope_intent_router below — when it's not
    # applicable, this node never executes at all (see
    # skip_doctor_scope_validation). Its own internal gate remains as a
    # defensive fallback only, mirroring validate_doctor_node.
    builder.add_node(
        "infer_doctor_scope_validation",
        functools.partial(infer_doctor_scope_validation, llm_client=llm_client),
    )
    builder.add_node("skip_doctor_scope_validation", skip_doctor_scope_validation)

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
    #   • infer_script_matching           (direct from inference_gate)
    # Exactly 4 unconditional predecessors.
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
    builder.add_node(
        "extract_appointment_details",
        functools.partial(extract_appointment_details, llm_client=llm_client),
    )
    builder.add_node("verify_appointment_in_db", verify_appointment_in_db)
    # Two fully independent deterministic nodes — each cheaply gates on its
    # own Arabic detection before reading CRM data, and each runs in
    # parallel with the other (neither depends on the other's result).
    builder.add_node("validate_bank_information", validate_bank_information_node)
    builder.add_node("validate_location", validate_location_node)
    builder.add_node(
        "validate_doctor",
        functools.partial(validate_doctor_node, llm_client=llm_client),
    )
    # Graph-level skip paths for validate_bank_information / validate_location
    # / validate_doctor — see _bank_intent_router / _location_intent_router /
    # _doctor_intent_router below. Deliberately do not add to node_trace:
    # each is a routing decision, not a validation step.
    builder.add_node("skip_bank_validation", skip_bank_validation)
    builder.add_node("skip_location_validation", skip_location_validation)
    builder.add_node("skip_doctor_validation", skip_doctor_validation)
    # Barrier — fan-in for EITHER validate_bank_information or its skip path,
    # EITHER validate_location or its skip path, AND EITHER validate_doctor
    # or its skip path, fan-out to the booking router. None of the three is
    # on the booking-detection critical path individually; the router just
    # needs all three branches to have finished, independently of each
    # other (any combination of the three must be supported).
    builder.add_node("loc_bank_ready", lambda state: {})

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

    # ── Step 2: booking sub-flow OR skip — both end at infer_behavioral ───
    #
    # BOOKING:    detect_intent → extract → verify → infer_reservation_evaluation
    #             → infer_behavioral_evaluation (fan-in start)
    # SKIP:       detect_intent → infer_behavioral_evaluation directly
    #
    # This ensures infer_reservation_evaluation ALWAYS completes (or is
    # skipped) BEFORE the parallel behavioral+compliance LLM calls begin.
    # BOOKING:  detect_intent → extract → verify → infer_reservation → inference_gate
    # SKIP:     detect_intent → inference_gate
    #
    # inference_gate is the SINGLE fan-out point for both parallel LLM calls.
    # Nothing else feeds into infer_behavioral_evaluation or
    # infer_compliance_evaluation — this guarantees inference_ready receives
    # exactly 2 triggers per run, preventing duplicate tail execution.
    # Route through bank + location + doctor validation (parallel) before
    # the booking split. All three states survive every branch combination
    # and are included in final scoring and aggregation.
    builder.add_conditional_edges(
        "detect_intent",
        _bank_intent_router,
        {"validate_bank_information": "validate_bank_information", "skip_bank": "skip_bank_validation"},
    )
    builder.add_conditional_edges(
        "detect_intent",
        _location_intent_router,
        {"validate_location": "validate_location", "skip_location": "skip_location_validation"},
    )
    builder.add_conditional_edges(
        "detect_intent",
        _doctor_intent_router,
        {"validate_doctor": "validate_doctor", "skip_doctor": "skip_doctor_validation"},
    )
    builder.add_edge("validate_bank_information", "loc_bank_ready")
    builder.add_edge("skip_bank_validation", "loc_bank_ready")
    builder.add_edge("validate_location", "loc_bank_ready")
    builder.add_edge("skip_location_validation", "loc_bank_ready")
    builder.add_edge("validate_doctor", "loc_bank_ready")
    builder.add_edge("skip_doctor_validation", "loc_bank_ready")
    builder.add_conditional_edges(
        "loc_bank_ready",
        _booking_router,
        {
            "booking":      "extract_appointment_details",
            "skip_booking": "inference_gate",
        },
    )
    builder.add_edge("extract_appointment_details", "verify_appointment_in_db")
    builder.add_edge("verify_appointment_in_db",    "infer_reservation_evaluation")
    builder.add_conditional_edges(
        "infer_reservation_evaluation",
        _error_router,
        {"continue": "inference_gate", "handle_error": "handle_error"},
    )

    # ── Step 3: inference_gate fan-out ────────────────────────────────────
    #
    # behavioral/compliance/script_matching/offer are all exactly TWO hops
    # from inference_gate, via fetch_crm_offers_for_call as a shared (fast,
    # never-erroring) pass-through — not direct edges for some plus a
    # two-hop path for others. That asymmetry used to be the root cause of
    # inference_ready (and everything downstream of it) firing TWICE per
    # call: LangGraph's fan-in only merges arrivals that land in the SAME
    # superstep, so when some branches reached inference_ready a superstep
    # earlier than one that took the extra fetch_crm_offers_for_call hop,
    # the barrier fired once for the fast ones and again when the last
    # branch caught up — replaying infer_overall_scoring →
    # aggregate_results → integrity_check → save_to_database → finalize a
    # second time. Routing every branch through fetch_crm_offers_for_call
    # equalises the hop count so all arrive in the same superstep and the
    # barrier fires exactly once.
    #
    # The doctor-scope branch is a CONDITIONAL edge
    # (_doctor_scope_intent_router) rather than a direct one: it goes to
    # EITHER infer_doctor_scope_validation OR skip_doctor_scope_validation,
    # both exactly one hop past fetch_crm_offers_for_call — i.e. still the
    # same two-hop depth from inference_gate as the other four branches, so
    # the equal-hop-count invariant above is preserved (this is the exact
    # same either/or-same-depth pattern already used for
    # validate_bank_information/skip_bank_validation etc. converging on
    # loc_bank_ready). Only ONE of the two ever executes per call, so
    # inference_ready still receives exactly 5 arrivals in one superstep —
    # never 4, never 6. fetch_crm_offers_for_call never sets state["error"],
    # and behavioral/compliance/script_matching don't read its output (only
    # infer_offer_evaluation and infer_doctor_scope_validation do — the
    # latter reads state["doctor_validation"], written earlier by
    # validate_doctor_node/skip_doctor_validation before the booking split,
    # not from fetch_crm_offers_for_call itself).
    builder.add_edge("inference_gate", "fetch_crm_offers_for_call")
    builder.add_edge("fetch_crm_offers_for_call", "infer_behavioral_evaluation")
    builder.add_edge("fetch_crm_offers_for_call", "infer_compliance_evaluation")
    builder.add_edge("fetch_crm_offers_for_call", "infer_script_matching")
    builder.add_edge("fetch_crm_offers_for_call", "infer_offer_evaluation")
    builder.add_conditional_edges(
        "fetch_crm_offers_for_call",
        _doctor_scope_intent_router,
        {
            "infer_doctor_scope_validation": "infer_doctor_scope_validation",
            "skip_doctor_scope": "skip_doctor_scope_validation",
        },
    )

    # ── Step 4: all five branches fan-in → inference_ready ────────────────
    #
    # LangGraph fires inference_ready only after all five (behavioral,
    # compliance, script_matching, offer, and EITHER doctor_scope or its
    # skip) complete.
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
    builder.add_conditional_edges(
        "infer_doctor_scope_validation",
        _error_router,
        {"continue": "inference_ready", "handle_error": "handle_error"},
    )
    builder.add_edge("skip_doctor_scope_validation", "inference_ready")

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
