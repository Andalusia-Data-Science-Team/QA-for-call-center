"""Tests for the graph-level bank/location-intent routing (app.agent.graph).

Neither validate_bank_information nor validate_location may execute at
all — no CRM fetch, no node_trace entry — when there is no corresponding
intent. Both decisions are made by dedicated conditional-edge routers
(_bank_intent_router / _location_intent_router) BEFORE the graph ever
reaches the respective node, each reusing the exact same deterministic
gate the node itself uses defensively
(bank_validation_needed/detect_bank_signals,
location_validation_needed/detect_location_signals) — not a second copy of
either. Bank and location routing are fully independent conditional edges
off the same detect_intent node, so all four combinations (neither/bank
only/location only/both) must be supported.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.graph import _bank_intent_router, _location_intent_router, _doctor_intent_router, build_qa_graph
from app.models.input import CallTranscript


class _StubLLMClient:
    """Minimal async LLM stub — enough for the graph to run past the
    location-routing segment without raising. Downstream LLM-dependent
    nodes may still degrade to handle_error on this stub's placeholder
    output; that's irrelevant to these tests, which only assert on the
    node_trace/state produced up through and around loc_bank_ready."""

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return "{}", {"prompt_tokens": 0, "completion_tokens": 0}


def _call(transcript: str, call_id: str = "graph-route-test") -> CallTranscript:
    return CallTranscript(
        call_id=call_id, agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-08-26", call_duration_seconds=1, department="Scheduling",
        transcript=transcript,
    )


def _run(transcript: str, call_id: str = "graph-route-test") -> dict:
    graph = build_qa_graph(_StubLLMClient())
    return asyncio.run(graph.ainvoke({"call": _call(transcript, call_id)}))


NO_INTENT_TRANSCRIPT = "Patient: عندي استفسار عن الأسعار\nAgent: تفضل"
HOME_CARE_TRANSCRIPT = "Patient: تمام تحت أمرك\nAgent: وبترسل لهم الموقع"
PATIENT_REQUEST_TRANSCRIPT = "Patient: فين فرع السنابل؟\nAgent: العنوان شارع السنابل العام بعد محطه نفط حي السنابل جده"
PROACTIVE_AGENT_TRANSCRIPT = "Patient: تمام الحجز مناسب\nAgent: العنوان فرع السنابل شارع السنابل العام بعد محطة نفط، حي السنابل، جدة"
BOOKING_TRANSCRIPT = "Patient: عايز احجز كشف بكرة\nAgent: تمام هحجزلك بكرة الساعة 5"

# ── Bank routing fixtures ───────────────────────────────────────────────────
BANK_ACCOUNT_NUMBER_TRANSCRIPT = "Patient: ممكن رقم حساب فرع BU-AKW؟\nAgent: تمام هبعتلك"
BANK_IBAN_TRANSCRIPT = "Patient: ممكن رقم الايبان بتاع فرع AKW؟\nAgent: تمام هجيبلك"
BANK_DETAILS_FOR_BU_TRANSCRIPT = "Patient: عايز بيانات الحساب البنكي بتاع فرع BU-ALW\nAgent: حاضر"
GENERIC_NUMBERS_TRANSCRIPT = "Patient: الموعد بكرة الساعة 5 والسعر 350 جنيه\nAgent: تمام"
BOTH_BANK_AND_LOCATION_TRANSCRIPT = (
    "Patient: عايز اعرف رقم الايبان بتاع فرع AHJ\n"
    "Patient: وكمان فين مستشفى أندلسية جدة؟\n"
    "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
)
BANK_PROACTIVE_AGENT_TRANSCRIPT = "Patient: أنا من فرع BU-AKW\nAgent: تمام الايبان بتاعنا هو SA1234567890123456789012"


# ── 1/2. No location intent → validate_location does not run / is absent ───

def test_no_intent_validate_location_absent_from_trace():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]


def test_no_intent_both_validations_skipped_but_state_present():
    """NO_INTENT_TRANSCRIPT has neither bank nor location intent — both
    nodes are skipped, but both states are still populated."""
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]
    assert "validate_location" not in result["node_trace"]
    assert result["bank_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"


# ── 3. No location intent → CRM location fetch is not called ───────────────

def test_no_intent_crm_fetch_never_called(monkeypatch):
    import app.service_hub.crm_location as crm_location

    calls = {"count": 0}

    def _record(*_a, **_k):
        calls["count"] += 1
        return []

    monkeypatch.setattr(crm_location, "fetch_ksa_locations", _record)
    _run(NO_INTENT_TRANSCRIPT)
    assert calls["count"] == 0


# ── 4. No location intent → location_validation still ends NOT_APPLICABLE ──

def test_no_intent_location_validation_is_not_applicable():
    result = _run(NO_INTENT_TRANSCRIPT)
    loc = result["location_validation"]
    assert loc["outcome"] == "NOT_APPLICABLE"
    assert loc["applicable"] is False
    assert loc["is_violation"] is False


# ── 5/6. Real intent → validate_location executes and is in the trace ──────

def test_patient_request_routes_to_validate_location():
    assert _location_intent_router({"call": _call(PATIENT_REQUEST_TRANSCRIPT)}) == "validate_location"
    result = _run(PATIENT_REQUEST_TRANSCRIPT)
    assert "validate_location" in result["node_trace"]
    assert "skip_location_validation" not in result["node_trace"]


def test_proactive_agent_location_routes_to_validate_location():
    assert _location_intent_router({"call": _call(PROACTIVE_AGENT_TRANSCRIPT)}) == "validate_location"
    result = _run(PROACTIVE_AGENT_TRANSCRIPT)
    assert "validate_location" in result["node_trace"]


# ── 7. Customer/home-care location phrasing does not execute validate_location ──

def test_home_care_customer_location_does_not_execute_validate_location(monkeypatch):
    import app.service_hub.crm_location as crm_location

    calls = {"count": 0}
    monkeypatch.setattr(crm_location, "fetch_ksa_locations", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    assert _location_intent_router({"call": _call(HOME_CARE_TRANSCRIPT)}) == "skip_location"
    result = _run(HOME_CARE_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"
    assert calls["count"] == 0


# ── 8. Bank validation behavior remains unchanged ───────────────────────────

def test_bank_validation_unaffected_by_location_routing_change():
    """None of these transcripts contain bank intent — bank routing must be
    unaffected by the location-intent router: bank_validation is still
    always populated (NOT_APPLICABLE here) regardless of which location
    branch was taken."""
    for transcript in (NO_INTENT_TRANSCRIPT, PATIENT_REQUEST_TRANSCRIPT, HOME_CARE_TRANSCRIPT):
        result = _run(transcript)
        assert result.get("bank_validation") is not None
        assert result["bank_validation"]["outcome"] == "NOT_APPLICABLE"
        assert "validate_bank_information" not in result["node_trace"]


# ── 9. Booking and non-booking paths still work correctly ──────────────────

def test_booking_path_still_reaches_extract_appointment_details():
    result = _run(BOOKING_TRANSCRIPT)
    assert "extract_appointment_details" in result["node_trace"]


def test_non_booking_path_skips_appointment_extraction():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "extract_appointment_details" not in result["node_trace"]
    assert "infer_behavioral_evaluation" in result["node_trace"]


def test_booking_path_with_location_intent_still_runs_both():
    """Booking + location intent together must still hit both
    validate_location and the booking sub-flow — the two conditional
    routers (location-intent, booking) are independent."""
    transcript = "Patient: عايز احجز في فرع السنابل بكرة\nPatient: فين فرع السنابل؟\nAgent: العنوان شارع السنابل العام بعد محطه نفط حي السنابل جده"
    result = _run(transcript)
    assert "validate_location" in result["node_trace"]
    assert "extract_appointment_details" in result["node_trace"]


# ── 10. Graph compilation succeeds ──────────────────────────────────────────

def test_graph_compiles():
    graph = build_qa_graph(_StubLLMClient())
    node_names = set(graph.get_graph().nodes.keys())
    assert "validate_location" in node_names
    assert "skip_location_validation" in node_names
    assert "validate_bank_information" in node_names
    assert "skip_bank_validation" in node_names


# ═════════════════════════════════════════════════════════════════════════
# Bank-intent routing — mirrors the location-intent routing tests above,
# using the same graph-level skipping approach (_bank_intent_router),
# independent of location routing.
# ═════════════════════════════════════════════════════════════════════════

# ── 1/2. No bank intent → validate_bank_information does not run / is absent ──

def test_no_bank_intent_validate_bank_information_absent_from_trace():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]


def test_no_bank_intent_skip_node_present_instead():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]
    assert "validate_location" not in result["node_trace"]


# ── 3. No bank intent → CRM bank fetch is not called ────────────────────────

def test_no_bank_intent_crm_fetch_never_called(monkeypatch):
    import app.service_hub.crm_bank as crm_bank

    calls = {"count": 0}

    def _record(*_a, **_k):
        calls["count"] += 1
        return []

    monkeypatch.setattr(crm_bank, "fetch_bank_accounts", _record)
    _run(NO_INTENT_TRANSCRIPT)
    assert calls["count"] == 0


# ── 4. No bank intent → bank_validation is still NOT_APPLICABLE ────────────

def test_no_bank_intent_bank_validation_is_not_applicable():
    result = _run(NO_INTENT_TRANSCRIPT)
    bank = result["bank_validation"]
    assert bank["outcome"] == "NOT_APPLICABLE"
    assert bank["applicable"] is False
    assert bank["is_violation"] is False
    assert bank["provided_identifiers"] == []


# ── 5/6/7. Real bank requests route to validate_bank_information ───────────

def test_account_number_request_routes_to_validate_bank_information():
    assert _bank_intent_router({"call": _call(BANK_ACCOUNT_NUMBER_TRANSCRIPT)}) == "validate_bank_information"
    result = _run(BANK_ACCOUNT_NUMBER_TRANSCRIPT)
    assert "validate_bank_information" in result["node_trace"]


def test_iban_request_routes_to_validate_bank_information():
    assert _bank_intent_router({"call": _call(BANK_IBAN_TRANSCRIPT)}) == "validate_bank_information"
    result = _run(BANK_IBAN_TRANSCRIPT)
    assert "validate_bank_information" in result["node_trace"]


def test_bank_details_for_business_unit_routes_to_validate_bank_information():
    assert _bank_intent_router({"call": _call(BANK_DETAILS_FOR_BU_TRANSCRIPT)}) == "validate_bank_information"
    result = _run(BANK_DETAILS_FOR_BU_TRANSCRIPT)
    assert "validate_bank_information" in result["node_trace"]


# ── 8. Agent proactively provides valid bank identifiers ───────────────────

def test_agent_proactive_bank_identifier_routes_to_validate_bank_information():
    assert _bank_intent_router({"call": _call(BANK_PROACTIVE_AGENT_TRANSCRIPT)}) == "validate_bank_information"
    result = _run(BANK_PROACTIVE_AGENT_TRANSCRIPT)
    assert "validate_bank_information" in result["node_trace"]


# ── 9. Generic unrelated numbers do not trigger bank validation ────────────

def test_generic_unrelated_numbers_do_not_trigger_bank_validation(monkeypatch):
    import app.service_hub.crm_bank as crm_bank

    calls = {"count": 0}
    monkeypatch.setattr(crm_bank, "fetch_bank_accounts", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    assert _bank_intent_router({"call": _call(GENERIC_NUMBERS_TRANSCRIPT)}) == "skip_bank"
    result = _run(GENERIC_NUMBERS_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]
    assert result["bank_validation"]["outcome"] == "NOT_APPLICABLE"
    assert calls["count"] == 0


# ── 10/11/12/13. All four bank/location combinations ────────────────────────

def test_location_only_skips_bank_but_executes_location():
    result = _run(PATIENT_REQUEST_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]
    assert "validate_location" in result["node_trace"]


def test_bank_only_skips_location_but_executes_bank():
    result = _run(BANK_IBAN_TRANSCRIPT)
    assert "validate_bank_information" in result["node_trace"]
    assert "validate_location" not in result["node_trace"]


def test_neither_bank_nor_location_both_nodes_absent():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_bank_information" not in result["node_trace"]
    assert "validate_location" not in result["node_trace"]
    assert result["bank_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"


def test_both_bank_and_location_intents_both_nodes_execute_exactly_once():
    result = _run(BOTH_BANK_AND_LOCATION_TRANSCRIPT)
    assert result["node_trace"].count("validate_bank_information") == 1
    assert result["node_trace"].count("validate_location") == 1
    assert "skip_bank_validation" not in result["node_trace"]
    assert "skip_location_validation" not in result["node_trace"]


# ── Logging format ───────────────────────────────────────────────────────

def test_bank_intent_routing_log_format(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="app.agent.graph"):
        _run(NO_INTENT_TRANSCRIPT)
    messages = [r.message for r in caplog.records if "bank intent routing" in r.message]
    assert any("bank_validation_needed=False" in m for m in messages)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app.agent.graph"):
        _run(BANK_IBAN_TRANSCRIPT)
    messages = [r.message for r in caplog.records if "bank intent routing" in r.message]
    assert any("bank_validation_needed=True" in m for m in messages)


# ═════════════════════════════════════════════════════════════════════════
# Home-care/customer location must not trigger validate_location
# ═════════════════════════════════════════════════════════════════════════

HOME_LOCATION_QUESTION_TRANSCRIPT = "Agent: ممكن لوكيشن المنزل ؟\nPatient: https://maps.google.com/?q=21.606500,39.240455"
HOME_SEND_LOCATION_TRANSCRIPT = "Patient: تمام\nAgent: ابعت موقعك"
HOME_SHARE_LOCATION_TRANSCRIPT = "Patient: تمام\nAgent: شارك اللوكيشن معايا"
BRANCH_LOCATION_ONLY_TRANSCRIPT = "Patient: موقع فرع السنابل ايه؟\nAgent: العنوان شارع السنابل العام بعد محطه نفط حي السنابل جده"


def test_home_location_question_does_not_execute_validate_location():
    assert _location_intent_router({"call": _call(HOME_LOCATION_QUESTION_TRANSCRIPT)}) == "skip_location"
    result = _run(HOME_LOCATION_QUESTION_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]


def test_map_coordinates_after_home_location_request_does_not_execute_validate_location():
    result = _run(HOME_LOCATION_QUESTION_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"


def test_home_care_send_your_location_does_not_execute_validate_location():
    assert _location_intent_router({"call": _call(HOME_SEND_LOCATION_TRANSCRIPT)}) == "skip_location"
    result = _run(HOME_SEND_LOCATION_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]


def test_home_care_share_location_does_not_execute_validate_location():
    assert _location_intent_router({"call": _call(HOME_SHARE_LOCATION_TRANSCRIPT)}) == "skip_location"
    result = _run(HOME_SHARE_LOCATION_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]


def test_branch_location_question_executes_validate_location():
    assert _location_intent_router({"call": _call(BRANCH_LOCATION_ONLY_TRANSCRIPT)}) == "validate_location"
    result = _run(BRANCH_LOCATION_ONLY_TRANSCRIPT)
    assert "validate_location" in result["node_trace"]


def test_agent_provided_andalusia_address_executes_validate_location():
    result = _run(PROACTIVE_AGENT_TRANSCRIPT)
    assert "validate_location" in result["node_trace"]


def test_normal_conversation_skips_location_validation():
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_location" not in result["node_trace"]
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"


def test_home_service_crm_location_fetch_never_called(monkeypatch):
    import app.service_hub.crm_location as crm_location

    calls = {"count": 0}
    monkeypatch.setattr(crm_location, "fetch_ksa_locations", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    for transcript in (HOME_LOCATION_QUESTION_TRANSCRIPT, HOME_SEND_LOCATION_TRANSCRIPT, HOME_SHARE_LOCATION_TRANSCRIPT):
        _run(transcript)
    assert calls["count"] == 0


def test_home_service_skip_reason_is_distinguished_in_log(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="app.agent.nodes"):
        _run(HOME_LOCATION_QUESTION_TRANSCRIPT)
    messages = [r.message for r in caplog.records if "location validation skipped" in r.message]
    assert any("reason=home_service_location" in m for m in messages)


# ═════════════════════════════════════════════════════════════════════════
# LangGraph single-execution regression — every downstream node after the
# parallel inference phase must fire exactly once per call, regardless of
# which branches (booking/offer/bank/location) were active.
# ═════════════════════════════════════════════════════════════════════════

_SINGLE_EXECUTION_NODES = (
    "infer_overall_scoring", "aggregate_results", "integrity_check",
    "save_to_database", "finalize",
)

OFFER_RELATED_TRANSCRIPT = "Patient: عندي عرض ايه على كشف الأسنان؟\nAgent: عندنا خصم 20% على كشف الأسنان"
MULTI_BRANCH_TRANSCRIPT = (
    "Patient: عايز احجز كشف بكرة\n"
    "Patient: وممكن رقم الايبان بتاع فرع AKW؟\n"
    "Patient: وفين فرع السنابل؟\n"
    "Agent: تمام هحجزلك، وده العنوان شارع السنابل العام بعد محطه نفط حي السنابل جده"
)


@pytest.mark.parametrize("transcript", [
    NO_INTENT_TRANSCRIPT,
    BOOKING_TRANSCRIPT,
    OFFER_RELATED_TRANSCRIPT,
    PATIENT_REQUEST_TRANSCRIPT,
    BANK_IBAN_TRANSCRIPT,
    MULTI_BRANCH_TRANSCRIPT,
])
def test_downstream_nodes_execute_exactly_once(transcript):
    result = _run(transcript, call_id="single-exec-test")
    trace = result["node_trace"]
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_no_node_appears_more_than_once_in_trace_for_multi_branch_call():
    """Every node in node_trace — not just the five downstream ones — must
    appear at most once (skip_bank_validation/skip_location_validation are
    mutually exclusive with their validate_* counterparts, so this holds
    for the whole trace, not just the tail)."""
    result = _run(MULTI_BRANCH_TRANSCRIPT, call_id="no-dup-test")
    trace = result["node_trace"]
    from collections import Counter
    counts = Counter(n.split("[")[0] for n in trace)  # strip handle_error[...] suffixes
    duplicates = {node: c for node, c in counts.items() if c > 1}
    assert duplicates == {}, f"trace={trace}"


# ═════════════════════════════════════════════════════════════════════════
# Doctor-intent routing — mirrors the bank/location routing tests above,
# using the same graph-level skipping approach (_doctor_intent_router),
# independent of bank/location routing. All combinations (none/bank only/
# location only/doctor only/any pair/all three) must be supported.
# ═════════════════════════════════════════════════════════════════════════

DOCTOR_INFO_TRANSCRIPT = "Patient: دكتور محمد الألفي تخصصه ايه؟\nAgent: دكتور محمد الألفي استشاري عظام"
DOCTOR_RECOMMEND_TRANSCRIPT = "Patient: عندي طفل عنده تأخر نمو وعايز اكشف\nAgent: ممكن تحجز مع دكتورة خيرية محمد"
GENERIC_SPECIALTY_TRANSCRIPT = "Patient: محتاج دكتور عظام\nAgent: ممكن أوضح لك المواعيد المتاحة"


def test_no_doctor_intent_validate_doctor_absent_from_trace():
    assert _doctor_intent_router({"call": _call(NO_INTENT_TRANSCRIPT)}) == "skip_doctor"
    result = _run(NO_INTENT_TRANSCRIPT)
    assert "validate_doctor" not in result["node_trace"]
    assert result["doctor_validation"]["outcome"] == "NOT_APPLICABLE"


def test_generic_specialty_request_does_not_execute_validate_doctor():
    assert _doctor_intent_router({"call": _call(GENERIC_SPECIALTY_TRANSCRIPT)}) == "skip_doctor"
    result = _run(GENERIC_SPECIALTY_TRANSCRIPT)
    assert "validate_doctor" not in result["node_trace"]


def test_doctor_intent_routes_to_validate_doctor_exactly_once():
    assert _doctor_intent_router({"call": _call(DOCTOR_INFO_TRANSCRIPT)}) == "validate_doctor"
    result = _run(DOCTOR_INFO_TRANSCRIPT)
    assert result["node_trace"].count("validate_doctor") == 1
    assert "skip_doctor_validation" not in result["node_trace"]


def test_doctor_recommendation_routes_to_validate_doctor():
    assert _doctor_intent_router({"call": _call(DOCTOR_RECOMMEND_TRANSCRIPT)}) == "validate_doctor"
    result = _run(DOCTOR_RECOMMEND_TRANSCRIPT)
    assert "validate_doctor" in result["node_trace"]


def test_doctor_crm_fetch_never_called_when_no_doctor_intent(monkeypatch):
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])
    _run(NO_INTENT_TRANSCRIPT)
    _run(GENERIC_SPECIALTY_TRANSCRIPT)
    assert calls["count"] == 0


def test_bank_plus_doctor_both_execute_exactly_once():
    transcript = (
        "Patient: عايز اعرف رقم الايبان بتاع فرع AKW\n"
        "Patient: وكمان دكتور محمد الألفي تخصصه ايه؟\n"
        "Agent: دكتور محمد الألفي استشاري عظام"
    )
    result = _run(transcript)
    assert result["node_trace"].count("validate_bank_information") == 1
    assert result["node_trace"].count("validate_doctor") == 1
    assert "validate_location" not in result["node_trace"]


def test_location_plus_doctor_both_execute_exactly_once():
    transcript = (
        "Patient: فين فرع السنابل؟\n"
        "Patient: وكمان دكتور محمد الألفي تخصصه ايه؟\n"
        "Agent: العنوان شارع السنابل العام بعد محطه نفط حي السنابل جده\n"
        "Agent: دكتور محمد الألفي استشاري عظام"
    )
    result = _run(transcript)
    assert result["node_trace"].count("validate_location") == 1
    assert result["node_trace"].count("validate_doctor") == 1
    assert "validate_bank_information" not in result["node_trace"]


BANK_LOCATION_DOCTOR_TRANSCRIPT = (
    # Deliberately uses AHJ/"مستشفى أندلسية جدة" (not AKW/"السنابل") for the
    # bank + location parts: mentioning "السنابل" alongside an explicit BU
    # code makes resolve_business_unit() pick up the (unrelated) SNB BU
    # keyword instead, which is outside bank-validation scope — a known,
    # pre-existing BU-resolution characteristic, not a routing bug. AHJ/the
    # hospital avoids that conflict entirely.
    "Patient: عايز اعرف رقم الايبان بتاع فرع AHJ\n"
    "Patient: وفين مستشفى أندلسية جدة؟\n"
    "Patient: وكمان دكتور محمد الألفي تخصصه ايه؟\n"
    "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
    "Agent: دكتور محمد الألفي استشاري عظام"
)


def test_bank_location_and_doctor_all_execute_exactly_once():
    result = _run(BANK_LOCATION_DOCTOR_TRANSCRIPT)
    for node in ("validate_bank_information", "validate_location", "validate_doctor"):
        assert result["node_trace"].count(node) == 1, f"{node} count={result['node_trace'].count(node)}"


def test_doctor_routing_does_not_introduce_duplicate_downstream_execution():
    result = _run(BANK_LOCATION_DOCTOR_TRANSCRIPT, call_id="doctor-no-dup-test")
    trace = result["node_trace"]
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_doctor_scope_validation_state_present_when_not_applicable():
    """infer_doctor_scope_validation always executes (equal hop count with
    the other 4 inference branches — see graph.py Step 3), but degrades to
    NOT_APPLICABLE internally without an LLM call when its own gate fails,
    exactly like infer_offer_evaluation already does for NO_OFFER_AVAILABLE."""
    result = _run(NO_INTENT_TRANSCRIPT)
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_scope_validation"]["applicable"] is False


def test_graph_compiles_with_doctor_nodes():
    graph = build_qa_graph(_StubLLMClient())
    node_names = set(graph.get_graph().nodes.keys())
    for node in ("validate_doctor", "skip_doctor_validation", "infer_doctor_scope_validation"):
        assert node in node_names


# ═════════════════════════════════════════════════════════════════════════
# Real-call regressions — false-positive doctor detection (Agent
# self-introductions must never route to validate_doctor) and the
# corresponding true-positive (a genuinely named, resolvable doctor must
# still route through it exactly once), run through the actual compiled
# graph, not just the standalone detector functions.
# ═════════════════════════════════════════════════════════════════════════

# Real regression call 4D0155E7 — MRI booking; the only doctor-looking
# phrase is the Agent's own self-introduction.
MRI_BOOKING_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايز احجز اشعة رنين مغناطيسي MRI\n"
    "Agent: تمام هوضحلك الاسعار والمواعيد المتاحة\n"
)

# Real regression call 8486A57A — home-care/nursing conversation, same
# Agent self-introduction.
HOME_CARE_INTRO_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايز اطلب خدمة تمريض منزلي لوالدتي\n"
    "Agent: تمام هوضحلك تفاصيل الخدمة والاسعار\n"
)

# Real regression call 4B77DCA1 — branch-location/mammography conversation.
# Doctor validation must be skipped while location validation is completely
# unaffected: resolution_source=provided_address, branch=مستشفى أندلسية
# جدة, outcome=PASS.
MAMMOGRAPHY_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايزة اعمل ماموجرام, فين اقرب فرع في جدة؟\n"
    "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
    "Patient: تمام هروح هناك\n"
)

# Most important true-positive regression — Patient explicitly names the
# doctor, Agent later confirms the same doctor in a booking-confirmation
# message ("مع د/ ( محمد الالفي )") that previously got captured almost in
# full and produced AMBIGUOUS_DOCTOR.
MOHAMED_ELALFY_TRANSCRIPT = (
    "Patient: عايز احجز مع دكتور محمد الالفي\n"
    "Agent: تمام يا فندم هوضحلك التفاصيل\n"
    "Patient: تمام, الدكتور محمد الالفي عايز اعرف كام الكشف عنده\n"
    "Agent: نعم هوضحلك\n"
    "Agent: تم تأكيد حجزك يوم الخميس الساعه 5 مع د/ ( محمد الالفي ) في فرع مستشفى أندلسية الحجاز\n"
)


@pytest.mark.parametrize("transcript", [
    MRI_BOOKING_TRANSCRIPT,
    HOME_CARE_INTRO_TRANSCRIPT,
    MAMMOGRAPHY_TRANSCRIPT,
])
def test_agent_self_introduction_calls_skip_doctor_validation(transcript):
    """None of these real regression calls name a specific doctor — they
    only contain the Agent's own self-introduction — so validate_doctor
    must never appear in node_trace. skip_doctor_validation deliberately
    never calls _trace() either (see README_doctor.md), so it is likewise
    absent from node_trace, not present. infer_doctor_scope_validation must
    be absent too — with no resolved doctor at all, the semantic scope node
    must never even execute to discover it's not applicable (see
    _doctor_scope_intent_router)."""
    result = _run(transcript, call_id="self-intro-skip-test")
    assert "validate_doctor" not in result["node_trace"]
    assert "infer_doctor_scope_validation" not in result["node_trace"]
    assert result["doctor_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_scope_validation"]["applicable"] is False


def test_mammography_call_skips_doctor_while_location_still_passes():
    """The mammography regression specifically requires location validation
    to keep working exactly as before — this must not regress alongside
    the doctor-detection fix."""
    result = _run(MAMMOGRAPHY_TRANSCRIPT, call_id="mammography-location-test")
    assert "validate_doctor" not in result["node_trace"]
    assert "infer_doctor_scope_validation" not in result["node_trace"]
    assert result["location_validation"]["outcome"] == "PASS"
    assert "أندلسية" in (result["location_validation"].get("requested_branch") or "")


def test_mohamed_elalfy_transcript_routes_to_validate_doctor_exactly_once():
    result = _run(MOHAMED_ELALFY_TRANSCRIPT, call_id="mohamed-elalfy-test")
    trace = result["node_trace"]
    assert trace.count("validate_doctor") == 1
    assert "skip_doctor_validation" not in trace
    doctor_result = result["doctor_validation"]
    assert doctor_result["outcome"] != "AMBIGUOUS_DOCTOR"
    assert doctor_result["doctor_resolved"] is True
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"
    # This transcript never states an actual clinical complaint (it's a
    # price/booking conversation about an already-named doctor) — the
    # semantic scope node must therefore be skipped even though the doctor
    # itself resolved successfully.
    assert "infer_doctor_scope_validation" not in trace
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"


# ═════════════════════════════════════════════════════════════════════════
# infer_doctor_scope_validation graph-level routing (Problem 1 above) —
# the semantic scope node must be ABSENT from node_trace whenever it is not
# applicable (no resolved doctor, or a resolved doctor but no genuine
# patient clinical need), not merely return NOT_APPLICABLE after executing.
# ═════════════════════════════════════════════════════════════════════════

# Real regression call 4177708C — the patient reschedules an existing
# appointment with an already-named doctor and separately asks an MRI
# pricing question; neither is a clinical complaint about that doctor.
# Doctor resolution must stay correct while scope validation is skipped.
MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT = (
    "Patient: اختي ممكن اجل حجزي لبكره حصل معي امر طارئ\n"
    "Patient: كان عندي موعد اليوم مع الدكتور\n"
    "Patient: محمد الالفي\n"
    "Patient: وابغى اجل الموعد لبكره\n"
    "Agent: تمام هوضحلك\n"
    "Patient: الدكتور محمد الالفي\n"
    "Agent: نعم هوضحلك\n"
    "Patient: موجود الدكتور\n"
    "Agent: تم تأكيد حجزك بكرة مع د/ ( محمد الالفي )\n"
    "Patient: عندي ملف عندكم موجود\n"
    "Patient: أنا ماعندي هويه عندي جواز سفر أختي أنا زياره\n"
    "Patient: كم سعر الرنين المغناطيسي عندكم لليد\n"
)

# Genuine clinical-need call — the patient does NOT name a specific doctor
# (only "a suitable doctor"), the AGENT proposes the named doctor in direct
# response to the stated complaint (doctor_role="agent_recommended"), so
# the semantic scope node must execute exactly once.
DOCTOR_WITH_CLINICAL_NEED_TRANSCRIPT = (
    "Patient: طفلي عنده صرع وتشنجات متكررة\n"
    "Patient: أبغى أحجز مع دكتور مناسب\n"
    "Agent: ممكن نحجز مع دكتورة خيرية محمد, استشارية غدد صماء أطفال\n"
)

# Additional real-batch no-doctor transcripts (radiology price inquiry,
# pregnancy ultrasound, CT scan pricing) — none name a specific doctor.
RADIOLOGY_PRICE_TRANSCRIPT = "Patient: عايز اعرف سعر اشعة الصدر\nAgent: تمام هوضحلك الاسعار المتاحة"
PREGNANCY_ULTRASOUND_TRANSCRIPT = "Patient: عايزة اعمل سونار للحمل, كام السعر؟\nAgent: تمام هوضحلك المواعيد والاسعار"
CT_SCAN_PRICING_TRANSCRIPT = "Patient: كام سعر الأشعة المقطعية CT؟\nAgent: تمام هوضحلك التفاصيل"


def test_mohamed_elalfy_reschedule_scope_absent_from_trace_at_graph_level():
    """Doctor resolution must remain correct while the semantic scope node
    itself never executes — this is the exact real regression from Problem
    2/Problem 1 combined, verified through the actual compiled graph."""
    result = _run(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT, call_id="elalfy-reschedule-test")
    trace = result["node_trace"]
    assert trace.count("validate_doctor") == 1
    doctor_result = result["doctor_validation"]
    assert doctor_result["outcome"] == "PASS"
    assert doctor_result["doctor_key"] == "110685"
    assert "infer_doctor_scope_validation" not in trace
    scope_result = result["doctor_scope_validation"]
    assert scope_result["outcome"] == "NOT_APPLICABLE"
    assert scope_result["applicable"] is False
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_genuine_clinical_need_routes_to_scope_validation_exactly_once():
    """The positive counterpart — a resolved doctor PLUS a genuine clinical
    complaint must route through infer_doctor_scope_validation exactly
    once, never through skip_doctor_scope_validation."""
    result = _run(DOCTOR_WITH_CLINICAL_NEED_TRANSCRIPT, call_id="clinical-need-test")
    trace = result["node_trace"]
    assert trace.count("validate_doctor") == 1
    assert result["doctor_validation"]["doctor_resolved"] is True
    assert trace.count("infer_doctor_scope_validation") == 1
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


@pytest.mark.parametrize("transcript", [
    RADIOLOGY_PRICE_TRANSCRIPT,
    PREGNANCY_ULTRASOUND_TRANSCRIPT,
    CT_SCAN_PRICING_TRANSCRIPT,
])
def test_batch_no_doctor_transcripts_never_reach_scope_node(transcript):
    result = _run(transcript, call_id="batch-no-doctor-test")
    trace = result["node_trace"]
    assert "validate_doctor" not in trace
    assert "infer_doctor_scope_validation" not in trace
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_doctor_scope_skip_case_all_downstream_nodes_execute_exactly_once():
    """For a skipped-scope case, everything except
    infer_doctor_scope_validation must still execute exactly once —
    confirms the equal-hop-count invariant survived adding a conditional
    edge on the doctor-scope branch."""
    result = _run(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT, call_id="skip-scope-single-exec-test")
    trace = result["node_trace"]
    for node in (
        "infer_behavioral_evaluation", "infer_compliance_evaluation",
        "infer_offer_evaluation", *_SINGLE_EXECUTION_NODES,
    ):
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"
    assert trace.count("infer_doctor_scope_validation") == 0


# ═════════════════════════════════════════════════════════════════════════
# Real regression call 42714555-0C7E-F111-B337-000D3AA9D4A7 — أحمد أفندي is
# the ORDERING/referring doctor for a CT scan; a later, unrelated kidney-
# failure complaint must never trigger a scope comparison against him.
# Doctor information validation must still resolve him correctly (PASS).
# ═════════════════════════════════════════════════════════════════════════

AHMED_AFANDI_TRANSCRIPT = (
    "Patient: الدكتور احمد افندي طلب اشعه مقطعيه ، هل تطلع في التطبيق ؟\n"
    "Patient: دكتور احمد افندي\n"
    "Patient: عندكم\n"
    "Patient: كاش\n"
    "Patient: ابغى اشوف نسخه من request الاشعه\n"
    "Agent: الاشعة المطلوبة لحضرتك هي تصوير الأوعية الدموية بالتصوير المقطعي للذراع\n"
    "Patient: لانه انا عندي غسيل كلى ولازم تكون قبل موعد جلسة الغسيل\n"
    "Patient: لانها بالصبغه فا لازم اغسل بعد الاشعه على طول\n"
    "Patient: انا مريض فشل كلوي لي 15 سنه فتحليل وظائف الكلى مرتفع دائما\n"
)


def test_ahmed_afandi_ordering_doctor_never_triggers_doctor_validation(monkeypatch):
    """Real regression, refined: أحمد أفندي is ONLY the ordering/referring
    physician for a CT scan — the active conversational intent throughout
    is about the imaging request/application/price/dialysis timing, never
    about him specifically. This is stricter than an earlier version of
    this test that expected doctor identity validation to still run
    (PASS): a valid, resolvable doctor name does NOT by itself make the
    conversation doctor-related — see classify_specific_doctor_intent.
    Doctor validation must be skipped ENTIRELY: no CRM fetch, no
    validate_doctor, no scope validation."""
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    result = _run(AHMED_AFANDI_TRANSCRIPT, call_id="ahmed-afandi-test")
    trace = result["node_trace"]
    assert calls["count"] == 0
    assert "validate_doctor" not in trace
    doctor_result = result["doctor_validation"]
    assert doctor_result["outcome"] == "NOT_APPLICABLE"
    assert "infer_doctor_scope_validation" not in trace
    scope_result = result["doctor_scope_validation"]
    assert scope_result["outcome"] == "NOT_APPLICABLE"
    assert scope_result["applicable"] is False
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


# Real regression call B1498491-ED7D-F111-B337-000D3AA9D4A7 — the booking
# targets an MRI; the named doctor (امير) is only an existing/referring
# relationship, and a later follow-up-policy/timing question is still not
# a doctor inquiry.
B1498491_TRANSCRIPT = (
    "Patient: عندنا حجز مع الدكتور الساعة ٦ لازم ندخل المراجعة ومعانا صورة الاشعة\n"
    "Patient: عايزة اعمل حجز اشعة رنين مغناطيسي اليوم قبل الساعة ٦ لان عندنا مراجعة مع دكتور امير وهو طالب صورة من الاشعة\n"
    "Patient: بالنسبة للمراجعة مع الدكتور يوم الاربعاء راح يكون ١٦ يوم من يوم الكشفية عادي يعني؟\n"
    "Agent: لا لازم 14 يوم\n"
)


def test_b1498491_existing_doctor_mri_booking_never_triggers_doctor_validation(monkeypatch):
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    result = _run(B1498491_TRANSCRIPT, call_id="b1498491-test")
    trace = result["node_trace"]
    assert calls["count"] == 0
    assert "validate_doctor" not in trace
    assert result["doctor_validation"]["outcome"] == "NOT_APPLICABLE"
    assert "infer_doctor_scope_validation" not in trace
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


# ── Further doctor-routing regressions (cross-turn linkage, booking
# confirmation-only, and service-category booking targets) — graph-level,
# with explicit CRM-fetch-count verification per the strong invariant:
# doctor_is_semantic_target=False must always mean fetch_doctors() is
# never called and validate_doctor never appears in node_trace.
# ═════════════════════════════════════════════════════════════════════════

def test_cross_turn_which_doctor_reply_triggers_validate_doctor(monkeypatch):
    """The booking verb and the doctor's name are on separate turns,
    linked only by the Agent's clarifying 'which doctor?' question — this
    must still resolve to a doctor booking and run validate_doctor."""
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    orig = crm_doctors.fetch_doctors

    def counting_fetch(*a, **k):
        calls["count"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(crm_doctors, "fetch_doctors", counting_fetch)

    transcript = "Patient: ابغى احجز موعد\nAgent: مع دكتور مين؟\nPatient: محمد الالفي\n"
    result = _run(transcript, call_id="which-doctor-test")
    trace = result["node_trace"]
    assert calls["count"] >= 1
    assert trace.count("validate_doctor") == 1
    assert result["doctor_validation"]["doctor_resolved"] is True


@pytest.mark.parametrize("transcript", [
    "Patient: الدكتور أحمد أفندي طلب أشعة مقطعية\nPatient: أبغى أحجز الأشعة يوم الاثنين\n",
    "Patient: الدكتور طلب تحليل\nPatient: ممكن احجز موعد للتحليل؟\n",
    "Patient: الدكتور كتب علاج طبيعي\nPatient: أبغى أحجز زيارة منزلية\n",
    "Patient: أبغى أحجز الرنين قبل موعدي مع الدكتور أمير\n",
])
def test_doctor_named_other_service_booking_never_fetches_crm(transcript, monkeypatch):
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    result = _run(transcript, call_id="service-target-test")
    trace = result["node_trace"]
    assert calls["count"] == 0
    assert "validate_doctor" not in trace
    assert result["doctor_validation"]["outcome"] == "NOT_APPLICABLE"
    assert "infer_doctor_scope_validation" not in trace


def test_booking_confirmation_only_runs_validate_doctor_but_not_scope(monkeypatch):
    """The Agent's booking confirmation is the ONLY doctor mention in the
    call — identity validation should still run, but this must never be
    treated as a clinical recommendation eligible for scope validation."""
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    orig = crm_doctors.fetch_doctors

    def counting_fetch(*a, **k):
        calls["count"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(crm_doctors, "fetch_doctors", counting_fetch)

    transcript = "Patient: تمام احجزوه\nAgent: تم تأكيد الموعد مع الدكتور محمد الألفي\n"
    result = _run(transcript, call_id="booking-confirmation-test")
    trace = result["node_trace"]
    assert calls["count"] >= 1
    assert trace.count("validate_doctor") == 1
    assert "infer_doctor_scope_validation" not in trace
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


# ── Agent-is-a-doctor self-introduction — graph-level, CRM-fetch-count
# verification (both English and Arabic self-introduction phrasing) ────────

@pytest.mark.parametrize("transcript", [
    "Agent: Hello, this is Dr. Mohamed Ahmed from Andalusia.\nPatient: I need physical therapy.\n",
    "Agent: السلام عليكم مع حضرتك دكتور يوسف محمد من أندلسية.\nPatient: محتاج علاج طبيعي.\n",
])
def test_agent_doctor_self_introduction_never_fetches_crm(transcript, monkeypatch):
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])

    result = _run(transcript, call_id="agent-self-intro-doctor-test")
    trace = result["node_trace"]
    assert calls["count"] == 0
    assert "validate_doctor" not in trace
    assert "infer_doctor_scope_validation" not in trace
    assert result["doctor_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_external_doctor_named_after_agent_self_introduction_still_resolves(monkeypatch):
    """The self-introduction exclusion must not swallow a genuinely
    DIFFERENT, external doctor named later in the same call — end-to-end
    through the compiled graph, not just the standalone classifier (see
    app.service_hub.doctor_validation.classify_specific_doctor_intent's
    docstring section on distinguishing self-introduction from an external
    doctor reference)."""
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    orig = crm_doctors.fetch_doctors

    def counting_fetch(*a, **k):
        calls["count"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(crm_doctors, "fetch_doctors", counting_fetch)

    transcript = (
        "Agent: Hello, this is Dr. Mohamed Ahmed from Andalusia.\n"
        "Patient: I need an appointment with Dr Ahmed Afandi.\n"
    )
    result = _run(transcript, call_id="self-intro-plus-external-doctor-test")
    trace = result["node_trace"]
    assert calls["count"] >= 1
    assert trace.count("validate_doctor") == 1
    doctor_result = result["doctor_validation"]
    assert doctor_result["doctor_resolved"] is True
    # The resolved doctor must be the externally-named one, never the
    # self-introducing Agent's own asserted identity.
    assert "mohamed ahmed" not in (doctor_result.get("doctor_name_en") or "").lower()
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def test_english_direct_doctor_booking_runs_validate_doctor(monkeypatch):
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    orig = crm_doctors.fetch_doctors

    def counting_fetch(*a, **k):
        calls["count"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(crm_doctors, "fetch_doctors", counting_fetch)

    transcript = "Patient: I want an appointment with Dr Ahmed Afandi.\n"
    result = _run(transcript, call_id="english-direct-booking-test")
    trace = result["node_trace"]
    assert calls["count"] >= 1
    assert trace.count("validate_doctor") == 1
    assert result["doctor_validation"]["doctor_resolved"] is True
    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"


def _fabricated_doctor(key: str, name_en: str, name_ar: str, scope_en: str) -> dict:
    return {
        "cr301_doctorkey": key, "servhub_doctornameen": name_en, "cr301_doctornamear": name_ar,
        "cr301_degreename": "Consultant", "cr301_specialtyname": "Orthopedic", "cr301_subspecialtyname": None,
        "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
        "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ",
        "statuscodename": "Active", "cr301_opdflag": "OPD",
        "cr301_drnotes": None, "cr301_scopeofservice": scope_en, "cr301_scopeofservicear": None,
        "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
        "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
    }


def test_multi_doctor_recommendation_single_node_execution(monkeypatch):
    """The central multi-doctor regression: an Agent turn recommending FOUR
    named doctors for one clinical need must resolve/validate/scope-check
    all four INTERNALLY within validate_doctor/infer_doctor_scope_validation
    — never by branching the graph into four paths. node_trace must show
    each of those two nodes exactly ONCE, and every downstream single-
    execution node exactly once too, regardless of how many doctors were
    evaluated inside them."""
    import app.service_hub.crm_doctors as crm_doctors

    fake_pool = [
        _fabricated_doctor("g1", "Mohamed Adel Belkadhi", "محمد عادل بلقاضي", "Hip and leg orthopedic surgery"),
        _fabricated_doctor("g2", "Ameer Elsayed", "امير السيد", "Hip and leg orthopedic surgery"),
        _fabricated_doctor("g3", "Elsayed Shaheen", "السيد شاهين", "Hip and leg orthopedic surgery"),
        _fabricated_doctor("g4", "Mahmoud Elsayed Elbadawy Ismail", "محمود السيد البدوي اسماعيل", "Hip and leg orthopedic surgery"),
    ]
    fetch_calls = {"count": 0}

    def fake_fetch(*a, **k):
        fetch_calls["count"] += 1
        return fake_pool

    monkeypatch.setattr(crm_doctors, "fetch_doctors", fake_fetch)

    transcript = (
        "Patient: I have hip pain radiating to my leg.\n"
        "Agent: I can recommend:\n"
        "DR \\ Mohamed Adel Belkadhi\n"
        "DR \\ Ameer Elsayed\n"
        "DR \\ Elsayed Shaheen\n"
        "DR \\ Mahmoud Elsayed Elbadawy Ismail\n"
        "Agent: Who is the doctor you want to book with ?\n"
        "Patient: I will choose later.\n"
    )
    result = _run(transcript, call_id="multi-doctor-recommendation-test")
    trace = result["node_trace"]

    assert fetch_calls["count"] == 1  # CRM fetched once regardless of recommendation size
    assert trace.count("validate_doctor") == 1
    doctor_result = result["doctor_validation"]
    assert doctor_result.get("recommended_doctor_count") == 4
    assert len(doctor_result.get("doctors", [])) == 4
    assert all(d["doctor_resolved"] for d in doctor_result["doctors"])
    # The false candidate must never appear among the resolved doctors.
    assert not any("you want to book" in (d.get("input_name") or "") for d in doctor_result["doctors"])

    for node in _SINGLE_EXECUTION_NODES:
        assert trace.count(node) == 1, f"{node} executed {trace.count(node)} times, expected 1 | trace={trace}"
