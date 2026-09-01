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
    service_eval: dict[str, Any]      # from infer_service_evaluation
    package_eval: dict[str, Any]      # from infer_package_evaluation
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
    is_booking_intent: Optional[bool]  # booking/appointment keywords matched
    is_offer_intent: Optional[bool]    # offer/package keywords matched (no DB verify needed)
    patient_is_insured: Optional[bool] # patient chose insurance after cash/insured prompt
    patient_declined_insurance: Optional[bool] # patient explicitly selected cash or stated no insurance
    is_insurance_intent: Optional[bool] # insurance flow detected unless the patient declined insurance
    intent_label: Optional[str]
    appointment_details: Optional[dict[str, Any]]
    appointment_verification: Optional[dict[str, Any]]
    crm_offers_context: Optional[str]    # written by fetch_crm_offers_for_call
    crm_services_context: Optional[str]  # written by fetch_crm_services_for_call
    crm_matched_services: list[dict[str, str]]  # CRM records that matched services mentioned by the agent
    crm_packages_context: Optional[str]  # written by fetch_crm_packages_for_call

    # ── Eligibility check sub-flow ────────────────────────────────────────
    # iqama_number        — injected by the API layer from the booking request
    # eligibility_result  — written by check_patient_eligibility
    #   keys: iqama_number (int), http_status (int|None), api_status (str),
    #         is_eligible (bool), insurance (dict|None), error_code (str|None),
    #         transaction_name (str|None), checked_at (str), reason (str|None)
    # ineligible_reason   — written by handle_ineligible_patient
    # final_response      — written by handle_ineligible_patient (bilingual rejection message)
    iqama_number: Optional[str]
    eligibility_result: Optional[dict[str, Any]]
    ineligible_reason: Optional[str]
    final_response: Optional[str]

    # ── Error handling ────────────────────────────────────────────────────
    error: Optional[str]
    error_node: Optional[str]

    # ── Booking / intent sub-flow ─────────────────────────────────────────
    # is_booking_intent / intent_label  — written by detect_intent
    # appointment_details               — written by extract_appointment_details
    #   keys: appointment_date, doctor_name, specialty_name, patient_name
    # appointment_verification          — written by verify_appointment_in_db
    #   keys: found (bool), record (dict|None), message (str)

    # ── Execution trace ───────────────────────────────────────────────────
    node_trace: Annotated[list[str], operator.add]