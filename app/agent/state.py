from __future__ import annotations

import operator
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult


class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────
    call: CallTranscript

    # ── Criteria blocks (loaded in parallel) ──────────────────────────────
    behavioral_criteria: str        # written by load_behavioral_criteria
    compliance_pillars: str         # written by load_compliance_pillars
    reservation_pillars: str        # written by load_reservation_pillars
    offer_pillars: str              # written by load_offer_pillars
    script_templates: str           # written by load_script_templates
    scoring_weights: str            # written by load_scoring_weights

    # ── Per-evaluation LLM sub-results (raw JSON dicts) ───────────────────
    # Each focused inference node stores its parsed output here so the
    # aggregate_results node can merge them into a single QAAnalysisResult.
    behavioral_eval: dict[str, Any]   # from infer_behavioral_evaluation
    compliance_eval: dict[str, Any]   # from infer_compliance_evaluation
    reservation_eval: dict[str, Any]  # from infer_reservation_evaluation
    offer_eval: dict[str, Any]        # from infer_offer_evaluation
    script_eval: dict[str, Any]       # from infer_script_matching
    scoring_eval: dict[str, Any]      # from infer_overall_scoring

    # ── Per-node usage tracking (list so all 4 LLM calls are preserved) ───
    usage_list: Annotated[list[dict[str, Any]], operator.add]

    # ── Final merged result ───────────────────────────────────────────────
    parsed_data: dict[str, Any]
    # Use a last-write-wins reducer so that when multiple parallel branches
    # all fan into handle_error in the same step, the concurrent writes to
    # `result` do not raise InvalidUpdateError.
    result: Annotated[Optional[QAAnalysisResult], lambda _old, new: new]
    is_booking_intent: Optional[bool]
    intent_label: Optional[str]
    appointment_details: Optional[dict[str, Any]]
    appointment_verification: Optional[dict[str, Any]]
    crm_offers_context: Optional[str]   # written by fetch_crm_offers_for_call
    # Bank and location are separate graph nodes (app/bank_node/,
    # app/location_node/) with separate state keys — each is independently
    # NOT_APPLICABLE when its own request type isn't present in the call.
    bank_validation: Optional[dict[str, Any]]      # written by validate_bank_information_node
    location_validation: Optional[dict[str, Any]]  # written by validate_location_node
    # Masked deterministic results retained so scoring and aggregation share
    # the same conclusion without passing financial identifiers to the LLM.

    # ── Error handling ────────────────────────────────────────────────────
    # Last-write-wins, for the same reason `result` above needs it: when
    # multiple parallel LLM nodes (behavioral/compliance/offer/script) fail
    # in the same graph step — e.g. no LLM API key configured at all, so
    # every concurrent LLM call fails together — they each try to write
    # `error`/`error_node` in that same step. Without a reducer here,
    # LangGraph raises InvalidUpdateError ("Can receive only one value per
    # step") instead of cleanly routing to handle_error.
    error: Annotated[Optional[str], lambda _old, new: new]
    error_node: Annotated[Optional[str], lambda _old, new: new]

    # ── Booking / intent sub-flow ─────────────────────────────────────────
    # is_booking_intent / intent_label  — written by detect_intent
    # appointment_details               — written by extract_appointment_details
    #   keys: appointment_date, doctor_name, specialty_name, patient_name
    # appointment_verification          — written by verify_appointment_in_db
    #   keys: found (bool), record (dict|None), message (str)

    # ── Execution trace ───────────────────────────────────────────────────
    node_trace: Annotated[list[str], operator.add]
