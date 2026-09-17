"""
LangGraph Pipeline Graph for the Call QA Analysis System.

Graph topology (happy path):
─────────────────────────────────────────────────────────────────────────────
  START
    │
    ▼
  load_call ──(error)──────────────────────────────────────────────────────┐
    │                                                                       │
    │  fan-out: 6 parallel criteria loaders                                 │
    ├──→ load_behavioral_criteria ──┐                                       │
    ├──→ load_compliance_pillars  ──┤                                       │
    ├──→ load_script_templates    ──┤ fan-in → criteria_ready               │
    ├──→ load_reservation_pillars ──┤                                       │
    ├──→ load_offer_pillars       ──┤                                       │
    └──→ load_scoring_weights     ──┘                                       │
                                   │                                        │
                            criteria_ready (barrier)                        │
                                   │                                        │
                             detect_intent                                   │
                                   │                                        │
    ┌──────────────────────────────┼──────────────────────────┐            │
    │ (bank intent router)         │ (location intent router)  │            │
    ▼                              │                           ▼            │
validate_bank_information          │           validate_location            │
  OR skip_bank_validation          │           OR skip_location_validation  │
    │                              │                           │            │
    └──────────────────────────────┼───────────────────────────┘            │
                                   │  fan-in (exactly 2 arrivals)           │
                             loc_bank_ready (barrier)                       │
                                   │                                        │
                      detect_insurance_intent                               │
                                   │                                        │
       ┌───────────────────────────┤                                        │
       │(insurance)                │(continue)                              │
       ▼                           ▼                                        │
check_patient_eligibility  route_booking_intent                             │
       │                           │                                        │
       │ (eligible)                │ _doctor_intent_router                  │
       └──────────────────────────►│                                        │
                (not_eligible)     ├──(validate_doctor)──→ validate_doctor  │
                     │             │  (skip_doctor)──────→ skip_doctor_val. │
                     │             │              │                         │
                     │             │        booking_ready (barrier,         │
                     │             │        exactly 1 arrival)              │
                     │             │              │                         │
                     │             │  _booking_router                       │
                     │             ├──(booking/offer_only)──────────────────┼──────────────────┐
                     │             │  (skip_booking)────→ inference_gate    │                  │
                     │             │                                        │                  ▼
                     └────────────►│                              extract_appointment_details  │
                    (ineligible)   │                                        │                  │
                    also goes      │                              ┌─────────┴─────────┐        │
                    here ──────────►                              │                   │        │
                                                         (booking)│           (offer) │        │
                                                                  ▼                   ▼        │
                                                    verify_appointment_in_db  offer_done       │
                                                                  │                   │        │
                                                    infer_reservation_eval     inference_gate  │
                                                                  │                            │
                                                    enforce_ineligible_viol                    │
                                                                  │                            │
                                                            inference_gate ◄───────────────────┘
                                                                  │
                    ┌─────────────────────────────────────────────┼────────────────────────────────────┐
                    │                    fetch_crm_offers_for_call │                                    │
                    │                 ┌──────────┬────────────┬────┴────┬───────────┬────────────┐     │
                    │                 │          │            │         │           │            │     │
                    │           infer_beh  infer_comp  infer_script infer_offer doc_scope  infer_coe   │
                    │                 │          │            │         │           │            │     │
                    │           beh_done  comp_done   script_done off_done  doc_scope_done coe_done    │
                    │                                                                                  │
                    │   fetch_crm_services → infer_service → service_done                             │
                    │   fetch_crm_packages → infer_package → package_done                             │
                    │                                                                                  │
                    └──────────────────────────────── inference_ready (barrier, 9 arrivals) ──────────┘
                                                                  │
                                                         validate_crm_lead
                                                                  │
                                                      detect_faq_escalation
                                                                  │
                                                       infer_overall_scoring
                                                                  │
                                                         aggregate_results
                                                                  │
                                                          integrity_check
                                                                  │
                                                          save_to_database
                                                                  │
                                                              finalize
                                                                  │
                     handle_error ◄──── (any error edge) ───     END
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
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.telemetry import (
    build_jsonl_logger,
    record_node_execution,
    write_jsonl,
)
from app.config import settings
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
    validate_crm_lead,
    detect_faq_escalation,
    validate_faq_record,
    handle_error,
    detect_intent,
    detect_insurance_intent,
    extract_appointment_details,
    verify_appointment_in_db,
    check_patient_eligibility,
    handle_ineligible_patient,
    _eligibility_router,
    validate_bank_information_node,
    validate_location_node,
    skip_location_validation,
    skip_bank_validation,
    validate_doctor_node,
    skip_doctor_validation,
    infer_doctor_scope_validation,
    skip_doctor_scope_validation,
    infer_coe_validation,
    skip_coe_validation,
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
from app.service_hub.coe_validation import classify_coe_trigger
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _with_node_output_logging(
    node_name: str,
    action: Callable[[AgentState], Any],
    output_logger: logging.Logger,
) -> Callable[[AgentState], Any]:
    """Wrap a graph node and record its returned state update or exception."""
    async def logged_node(state: AgentState) -> Any:
        record_node_execution(node_name)
        call = state.get("call") if isinstance(state, dict) else None
        call_id = getattr(call, "call_id", "UNKNOWN")
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            output = action(state)
            if inspect.isawaitable(output):
                output = await output
        except Exception as exc:
            write_jsonl(
                output_logger,
                {
                    "timestamp_utc": timestamp_utc,
                    "call_id": call_id,
                    "node": node_name,
                    "status": "error",
                    "output": None,
                    "error": str(exc),
                },
            )
            raise

        write_jsonl(
            output_logger,
            {
                "timestamp_utc": timestamp_utc,
                "call_id": call_id,
                "node": node_name,
                "status": "success",
                "output": output,
                "error": None,
            },
        )
        return output

    return logged_node


class _NodeOutputLoggingGraph:
    """StateGraph proxy that instruments every registered node."""

    def __init__(self, graph: StateGraph, output_logger: logging.Logger) -> None:
        self._graph = graph
        self._output_logger = output_logger

    def add_node(self, name: str, action: Callable, *args: Any, **kwargs: Any) -> Any:
        wrapped = _with_node_output_logging(name, action, self._output_logger)
        return self._graph.add_node(name, wrapped, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


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


def _faq_router(state: AgentState) -> Literal["validate", "skip"]:
    """Route escalation claims through FAQ validation."""
    return "validate" if state.get("is_faq_escalation") else "skip"


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


def _coe_intent_router(state: AgentState) -> Literal["infer_coe_validation", "skip_coe"]:
    """Route to infer_coe_validation ONLY when there is clear transcript
    evidence of a COE/specialized-center trigger — reusing
    app.service_hub.coe_validation.classify_coe_trigger, the SAME
    deterministic gate infer_coe_validation itself uses defensively, so the
    decision is never duplicated. A qualifying medical complaint alone (no
    COE/specialized-center discussion), a COE doctor mentioned only during
    an ordinary booking, or a customer mention the agent never responds to
    are all correctly excluded here — see classify_coe_trigger's docstring.

    A detected COE campaign marker identifies only a CANDIDATE COE — it
    never, by itself, establishes that this is a COE conversation (see
    classify_coe_trigger's CAMPAIGN DETECTION IS NOT CAMPAIGN ENGAGEMENT
    section). classify_coe_trigger itself already resolves campaign
    relevance internally (via classify_campaign_relevance, computed
    exactly once and attached here as ctx["campaign_info"]) before
    deciding triggered/trigger_path, so a diverted/pending/uncertain
    campaign with no OTHER independently eligible COE need correctly
    routes to skip_coe — never entering infer_coe_validation, and
    therefore never fetching CRM data or calling the COE LLM.

    Placed on the conditional edge out of fetch_crm_offers_for_call (same
    hop depth as the other five branches — behavioral/compliance/script/
    offer/doctor_scope) so the equal-hop-count invariant inference_ready's
    fan-in depends on is preserved (see the Step 3 comment below)."""
    call = state["call"]
    ctx = classify_coe_trigger(call)
    campaign_info = ctx.get("campaign_info") or {}
    if campaign_info.get("campaign_detected"):
        # Classification fields only — never the raw patient inquiry text
        # (see "avoid logging raw patient names/IDs/full complaints").
        print(
            f"[coe] campaign | detected={campaign_info.get('campaign_detected')} "
            f"candidate={campaign_info.get('campaign_candidate_coe')} "
            f"relevance={campaign_info.get('campaign_relevance')} "
            f"active={campaign_info.get('active_campaign_coe')}",
            flush=True,
        )
    logger.info(
        "coe intent routing | call_id=%s triggered=%s trigger_path=%s reason=%s",
        call.call_id, ctx["triggered"], ctx["trigger_path"], ctx["trigger_reason"],
    )
    return "infer_coe_validation" if ctx["triggered"] else "skip_coe"


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def build_qa_graph(llm_client: LLMClient) -> StateGraph:
    """
    Compile and return the LangGraph StateGraph for QA analysis.

    LLMClient is injected via functools.partial so the graph is provider-agnostic
    and can be rebuilt with a different provider at any time.

    Loop-free invariants enforced by this wiring
    ─────────────────────────────────────────────
    1. Every barrier node has a FIXED, KNOWN number of predecessors.
       LangGraph fires a barrier exactly once — when ALL predecessors have
       delivered exactly one arrival in the same superstep.

    2. loc_bank_ready has exactly 2 predecessors:
         bank branch  (validate_bank_information OR skip_bank_validation)
         location branch (validate_location OR skip_location_validation)
       Doctor validation runs AFTER loc_bank_ready in a dedicated
       booking_ready barrier so it never inflates loc_bank_ready's count.

    3. booking_ready has exactly 2 predecessors:
         doctor branch (validate_doctor OR skip_doctor_validation)
         eligibility/ineligible merge (route_booking_intent via booking_merge)
       booking_ready is the SINGLE entry into the booking router.

    4. inference_gate has exactly ONE predecessor per call path:
         booking path:    enforce_ineligible_reservation_violation
         offer-only path: offer_extraction_done
         skip path:       booking_ready (skip_booking edge)
       These are mutually exclusive per call — exactly one fires.

    5. inference_ready has exactly 9 fixed predecessors — one *_done
       barrier per parallel branch:
         behavioral_done, compliance_done, script_done,
         offer_done, service_done, package_done,
         doctor_scope_done, coe_done, (none for services/packages — see below)
       doctor_scope and coe each contribute exactly ONE arrival via their
       own done-barrier (EITHER the infer_* path OR the skip_* path).

    6. handle_ineligible_patient feeds into booking_merge (not directly
       into extract_appointment_details), so it never bypasses booking_ready
       and cannot create a second path into inference_gate.
    """
    output_logger, _output_log_path = build_jsonl_logger(
        "node_output", settings.NODE_OUTPUT_LOG_PATH
    )
    builder = _NodeOutputLoggingGraph(
        StateGraph(AgentState), output_logger
    )

    # ── Stage 1: entry ────────────────────────────────────────────────────
    builder.add_node("load_call", load_call)

    # ── Stage 2: criteria loaders (6 parallel) ───────────────────────────
    builder.add_node("load_behavioral_criteria", load_behavioral_criteria)
    builder.add_node("load_compliance_pillars",  load_compliance_pillars)
    builder.add_node("load_reservation_pillars", load_reservation_pillars)
    builder.add_node("load_offer_pillars",       load_offer_pillars)
    builder.add_node("load_script_templates",    load_script_templates)
    builder.add_node("load_scoring_weights",     load_scoring_weights)
    # Barrier: fires once all 6 loaders complete (6 predecessors).
    builder.add_node("criteria_ready", lambda state: {})

    # ── Stage 3: intent detection + parallel bank/location validation ─────
    builder.add_node("detect_intent", detect_intent)
    builder.add_node("validate_bank_information", validate_bank_information_node)
    builder.add_node("skip_bank_validation",      skip_bank_validation)
    builder.add_node("validate_location",         validate_location_node)
    builder.add_node("skip_location_validation",  skip_location_validation)
    # Barrier: fires once bank branch AND location branch both complete (2 predecessors).
    builder.add_node("loc_bank_ready", lambda state: {})

    # ── Stage 4: insurance / eligibility → doctor validation ─────────────
    builder.add_node("detect_insurance_intent", detect_insurance_intent)
    builder.add_node("check_patient_eligibility", check_patient_eligibility)
    builder.add_node("handle_ineligible_patient", handle_ineligible_patient)
    # no-op barrier that merges the insurance→eligible path and the
    # no-insurance path into a single token before doctor routing.
    # Also receives handle_ineligible_patient so ineligible calls still
    # reach doctor validation (to catch an improper reservation) without
    # bypassing booking_ready.
    builder.add_node("booking_merge", lambda state: {})
    builder.add_node(
        "validate_doctor",
        functools.partial(validate_doctor_node, llm_client=llm_client),
    )
    builder.add_node("skip_doctor_validation", skip_doctor_validation)
    # Barrier: fires once doctor branch completes (1 predecessor).
    # Single entry point into the booking router — prevents any second
    # path from re-triggering extract_appointment_details.
    builder.add_node("booking_ready", lambda state: {})

    # ── Stage 5: booking branch ───────────────────────────────────────────
    builder.add_node(
        "extract_appointment_details",
        functools.partial(extract_appointment_details, llm_client=llm_client),
    )
    builder.add_node("verify_appointment_in_db",   verify_appointment_in_db)
    builder.add_node(
        "infer_reservation_evaluation",
        functools.partial(infer_reservation_evaluation, llm_client=llm_client),
    )
    builder.add_node("enforce_ineligible_reservation_violation", enforce_ineligible_reservation_violation)
    # no-op barrier: offer-only path joins here before inference_gate.
    builder.add_node("offer_extraction_done", lambda state: {})
    # Single fan-out point for all parallel inference branches.
    builder.add_node("inference_gate", lambda state: {})

    # ── Stage 6: CRM data fetches + parallel inference ────────────────────
    builder.add_node("fetch_crm_offers_for_call",  fetch_crm_offers_for_call)
    builder.add_node("fetch_crm_services_for_call", fetch_crm_services_for_call)
    builder.add_node("fetch_crm_packages_for_call", fetch_crm_packages_for_call)

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
    builder.add_node(
        "infer_doctor_scope_validation",
        functools.partial(infer_doctor_scope_validation, llm_client=llm_client),
    )
    builder.add_node("skip_doctor_scope_validation", skip_doctor_scope_validation)
    builder.add_node(
        "infer_coe_validation",
        functools.partial(infer_coe_validation, llm_client=llm_client),
    )
    builder.add_node("skip_coe_validation", skip_coe_validation)

    # Per-branch done-barriers — each has exactly 1 predecessor so
    # inference_ready always receives exactly 9 arrivals per call.
    builder.add_node("behavioral_done",   lambda state: {})
    builder.add_node("compliance_done",   lambda state: {})
    builder.add_node("script_done",       lambda state: {})
    builder.add_node("offer_done",        lambda state: {})
    builder.add_node("service_done",      lambda state: {})
    builder.add_node("package_done",      lambda state: {})
    builder.add_node("doctor_scope_done", lambda state: {})
    builder.add_node("coe_done",          lambda state: {})

    # Barrier: fires once all 9 *_done nodes deliver (9 predecessors).
    # Note: services and packages use their own fetch→infer→done chain
    # rather than routing through fetch_crm_offers_for_call, so they
    # are independent parallel branches at equal hop depth.
    builder.add_node("inference_ready", lambda state: {})

    # ── Stage 7: post-inference sequential chain ──────────────────────────
    builder.add_node(
        "validate_crm_lead",
        functools.partial(validate_crm_lead, llm_client=llm_client),
    )
    builder.add_node("detect_faq_escalation", detect_faq_escalation)
    builder.add_node(
        "validate_faq_record",
        functools.partial(validate_faq_record, llm_client=llm_client),
    )
    builder.add_node(
        "infer_overall_scoring",
        functools.partial(infer_overall_scoring, llm_client=llm_client),
    )
    builder.add_node("aggregate_results", aggregate_results)
    builder.add_node("integrity_check",   integrity_check)
    builder.add_node("save_to_database",  save_to_database)
    builder.add_node("finalize",          finalize)
    builder.add_node("handle_error",      handle_error)

    # ════════════════════════════════════════════════════════════════════
    # EDGES
    # ════════════════════════════════════════════════════════════════════

    builder.add_edge(START, "load_call")

    # ── load_call → fan-out to 6 criteria loaders ─────────────────────────
    builder.add_conditional_edges(
        "load_call", _error_router,
        {"continue": "load_behavioral_criteria", "handle_error": "handle_error"},
    )
    builder.add_edge("load_call", "load_compliance_pillars")
    builder.add_edge("load_call", "load_script_templates")
    builder.add_edge("load_call", "load_reservation_pillars")
    builder.add_edge("load_call", "load_offer_pillars")
    builder.add_edge("load_call", "load_scoring_weights")

    # ── 6 loaders → criteria_ready (6 predecessors) ──────────────────────
    builder.add_edge("load_behavioral_criteria", "criteria_ready")
    builder.add_edge("load_compliance_pillars",  "criteria_ready")
    builder.add_edge("load_script_templates",    "criteria_ready")
    builder.add_edge("load_reservation_pillars", "criteria_ready")
    builder.add_edge("load_offer_pillars",       "criteria_ready")
    builder.add_edge("load_scoring_weights",     "criteria_ready")

    # ── criteria_ready → detect_intent → parallel bank + location ────────
    builder.add_edge("criteria_ready", "detect_intent")

    builder.add_conditional_edges(
        "detect_intent", _bank_intent_router,
        {"validate_bank_information": "validate_bank_information",
         "skip_bank": "skip_bank_validation"},
    )
    builder.add_conditional_edges(
        "detect_intent", _location_intent_router,
        {"validate_location": "validate_location",
         "skip_location": "skip_location_validation"},
    )

    # ── bank + location branches → loc_bank_ready (2 predecessors) ───────
    builder.add_edge("validate_bank_information", "loc_bank_ready")
    builder.add_edge("skip_bank_validation",      "loc_bank_ready")
    builder.add_edge("validate_location",         "loc_bank_ready")
    builder.add_edge("skip_location_validation",  "loc_bank_ready")

    # ── loc_bank_ready → detect_insurance_intent (sequential) ────────────
    builder.add_edge("loc_bank_ready", "detect_insurance_intent")

    builder.add_conditional_edges(
        "detect_insurance_intent", _insurance_router,
        {"insurance": "check_patient_eligibility",
         "continue":  "booking_merge"},
    )
    builder.add_conditional_edges(
        "check_patient_eligibility", _eligibility_router,
        {"eligible":     "booking_merge",
         "not_eligible": "handle_ineligible_patient",
         "error":        "handle_error"},
    )
    # Ineligible patients still go through booking_merge → doctor validation
    # → booking_ready so the booking router sees them exactly once.
    builder.add_edge("handle_ineligible_patient", "booking_merge")

    # ── booking_merge → doctor validation → booking_ready ────────────────
    builder.add_conditional_edges(
        "booking_merge", _doctor_intent_router,
        {"validate_doctor": "validate_doctor",
         "skip_doctor":     "skip_doctor_validation"},
    )
    builder.add_edge("validate_doctor",       "booking_ready")
    builder.add_edge("skip_doctor_validation","booking_ready")

    # ── booking_ready → booking router (single entry point) ──────────────
    builder.add_conditional_edges(
        "booking_ready", _booking_router,
        {"booking":      "extract_appointment_details",
         "offer_only":   "extract_appointment_details",
         "skip_booking": "inference_gate"},
    )

    # ── extract_appointment_details → booking or offer-only fork ─────────
    builder.add_conditional_edges(
        "extract_appointment_details",
        lambda s: "booking" if s.get("is_booking_intent") else "offer_only",
        {"booking":    "verify_appointment_in_db",
         "offer_only": "offer_extraction_done"},
    )

    # Offer-only path: skip DB + reservation eval.
    builder.add_edge("offer_extraction_done", "inference_gate")

    # Booking path: DB verify → reservation eval → ineligibility enforcement.
    builder.add_edge("verify_appointment_in_db", "infer_reservation_evaluation")
    builder.add_conditional_edges(
        "infer_reservation_evaluation", _error_router,
        {"continue": "enforce_ineligible_reservation_violation",
         "handle_error": "handle_error"},
    )
    builder.add_edge("enforce_ineligible_reservation_violation", "inference_gate")

    # ── inference_gate → 3 independent fetch → infer chains (parallel) ───
    #
    # Each chain is fetch → infer → *_done → inference_ready.
    # All chains are equal-depth so inference_ready fires exactly once
    # when all 9 *_done nodes have delivered in the same superstep.
    #
    # Chain A: offers  (fetch_crm_offers → behavioral/compliance/script/offer/doctor_scope/coe)
    builder.add_edge("inference_gate", "fetch_crm_offers_for_call")
    builder.add_edge("fetch_crm_offers_for_call", "infer_behavioral_evaluation")
    builder.add_edge("fetch_crm_offers_for_call", "infer_compliance_evaluation")
    builder.add_edge("fetch_crm_offers_for_call", "infer_script_matching")
    builder.add_edge("fetch_crm_offers_for_call", "infer_offer_evaluation")
    builder.add_conditional_edges(
        "fetch_crm_offers_for_call", _doctor_scope_intent_router,
        {"infer_doctor_scope_validation": "infer_doctor_scope_validation",
         "skip_doctor_scope":             "skip_doctor_scope_validation"},
    )
    builder.add_conditional_edges(
        "fetch_crm_offers_for_call", _coe_intent_router,
        {"infer_coe_validation": "infer_coe_validation",
         "skip_coe":             "skip_coe_validation"},
    )

    # Chain B: services
    builder.add_edge("inference_gate",             "fetch_crm_services_for_call")
    builder.add_edge("fetch_crm_services_for_call", "infer_service_evaluation")

    # Chain C: packages
    builder.add_edge("inference_gate",              "fetch_crm_packages_for_call")
    builder.add_edge("fetch_crm_packages_for_call", "infer_package_evaluation")

    # ── Each infer node → its own *_done barrier → inference_ready ────────
    builder.add_conditional_edges(
        "infer_behavioral_evaluation", _error_router,
        {"continue": "behavioral_done", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_compliance_evaluation", _error_router,
        {"continue": "compliance_done", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_script_matching", _error_router,
        {"continue": "script_done", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_offer_evaluation", _error_router,
        {"continue": "offer_done", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_service_evaluation", _error_router,
        {"continue": "service_done", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_package_evaluation", _error_router,
        {"continue": "package_done", "handle_error": "handle_error"},
    )
    # doctor_scope: EITHER infer_doctor_scope_validation OR skip → doctor_scope_done
    builder.add_conditional_edges(
        "infer_doctor_scope_validation", _error_router,
        {"continue": "doctor_scope_done", "handle_error": "handle_error"},
    )
    builder.add_edge("skip_doctor_scope_validation", "doctor_scope_done")
    # coe: EITHER infer_coe_validation OR skip → coe_done
    builder.add_conditional_edges(
        "infer_coe_validation", _error_router,
        {"continue": "coe_done", "handle_error": "handle_error"},
    )
    builder.add_edge("skip_coe_validation", "coe_done")

    # ── 9 *_done barriers → inference_ready (9 predecessors) ─────────────
    builder.add_edge(
        ["behavioral_done", "compliance_done", "script_done",
         "offer_done", "service_done", "package_done",
         "doctor_scope_done", "coe_done"],
        "inference_ready",
    )
    # Note: service_done and package_done are already in the list above.
    # inference_ready now has exactly 8 named predecessors. We keep it at
    # 8 because service and package share equal hop depth with the others
    # via their own fetch→infer→done chains.

    # ── inference_ready → CRM lead → FAQ → scoring → aggregation ─────────
    builder.add_edge("inference_ready", "validate_crm_lead")
    builder.add_conditional_edges(
        "validate_crm_lead", _error_router,
        {"continue": "detect_faq_escalation", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "detect_faq_escalation", _faq_router,
        {"validate": "validate_faq_record", "skip": "infer_overall_scoring"},
    )
    builder.add_conditional_edges(
        "validate_faq_record", _error_router,
        {"continue": "infer_overall_scoring", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "infer_overall_scoring", _error_router,
        {"continue": "aggregate_results", "handle_error": "handle_error"},
    )
    builder.add_conditional_edges(
        "aggregate_results", _error_router,
        {"continue": "integrity_check", "handle_error": "handle_error"},
    )

    # ── tail chain (non-fatal) ────────────────────────────────────────────
    builder.add_edge("integrity_check",  "save_to_database")
    builder.add_edge("save_to_database", "finalize")
    builder.add_edge("finalize",         END)
    builder.add_edge("handle_error",     END)

    return builder.compile()
