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

from app.agent.graph import _bank_intent_router, _location_intent_router, build_qa_graph
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
