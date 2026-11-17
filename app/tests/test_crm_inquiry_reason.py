"""The inquiry-only CRM reason must reflect the patient's main question."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.agent.nodes import validate_crm_lead
from app.models.input import CallTranscript
from app.prompts.qa_prompt import build_crm_lead_validation_prompt
from app.service_hub.crm_leads_validation import (
    _normalize_web_api_lead,
    normalize_crm_lead_evaluation,
)
from app.services.regulation_catalog import matched_regulation_flags, policy_outcome


def test_web_api_uses_formatted_inquiry_reason_choice():
    record = _normalize_web_api_lead({
        "new_inquiryonlyreason": 100000002,
        "new_inquiryonlyreason@OData.Community.Display.V1.FormattedValue": "Asking about Offers",
    })
    assert record["new_inquiryonlyreason"] == "Asking about Offers"


def test_crm_prompt_requires_reason_check_against_patient_inquiry():
    call = CallTranscript(
        call_id="offer-inquiry", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-11", call_duration_seconds=60, department="Care",
        transcript="Patient: أريد تفاصيل عرض الجمعة للولادة الطبيعية\nAgent: العرض متاح",
    )
    prompt = build_crm_lead_validation_prompt(call, {"record": {
        "new_lastcallresult": "Inquiry only [01]",
        "new_inquiryonlyreason": "Inquiry about services",
    }})
    assert "new_inquiryonlyreason" in prompt
    assert "Asking about Offers" in prompt
    assert "Inquiry about services" in prompt
    assert "Inquire about price" in prompt
    assert "patient's primary inquiry" in prompt
    assert "Only when new_lastcallresult is Inquiry only" in prompt


def test_wrong_inquiry_reason_is_system_action_c2b():
    evaluation = normalize_crm_lead_evaluation({
        "crm_lead_status": "violation",
        "field_checks": [{
            "field": "new_inquiryonlyreason", "matches": False,
            "expected": "Asking about Offers", "actual": "Inquiry about services",
            "reason": "The patient asked about a Friday maternity offer.",
        }],
        "crm_leads_flags": [],
    })
    flags = matched_regulation_flags(evaluation["crm_leads_flags"])
    assert len(flags) == 1
    assert flags[0]["type"] == "C2B"
    assert flags[0]["attribute"] == "Displays positive company image"
    assert flags[0]["comment"] == "Didn’t take the proper action on system such linking CST Contract etc."
    assert policy_outcome(flags, "pass", False) == ("needs_review", False, "C")


def test_matching_inquiry_reason_has_no_violation():
    evaluation = normalize_crm_lead_evaluation({
        "crm_lead_status": "match",
        "field_checks": [{
            "field": "new_inquiryonlyreason", "matches": True,
            "expected": "Asking about Offers", "actual": "Asking about Offers",
        }],
        "crm_leads_flags": [],
    })
    assert evaluation["crm_lead_status"] == "match"
    assert evaluation["crm_leads_flags"] == []


def test_inquiry_only_evaluation_retries_when_reason_check_is_omitted():
    call = CallTranscript(
        call_id="offer-inquiry", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-11", call_duration_seconds=60, department="Care",
        transcript="Patient: أريد تفاصيل عرض الجمعة للولادة الطبيعية",
    )
    lookup = {"status": "found", "record": {
        "new_lastcallresult": "Inquiry only [01]",
        "new_inquiryonlyreason": "Inquiry about services",
    }}
    first = {"crm_lead_status": "match", "field_checks": [], "crm_leads_flags": []}
    corrected = {
        "crm_lead_status": "violation",
        "field_checks": [{
            "field": "new_inquiryonlyreason", "matches": False,
            "expected": "Asking about Offers", "actual": "Inquiry about services",
            "reason": "The patient asked about the offer.",
        }],
        "crm_leads_flags": [],
    }
    llm = AsyncMock(side_effect=[(first, None), (corrected, None)])
    with patch("app.agent.nodes.fetch_crm_lead", return_value=lookup), patch(
        "app.agent.nodes._focused_llm_call", llm
    ):
        result = asyncio.run(validate_crm_lead({"call": call, "node_trace": []}, None))
    assert llm.await_count == 2
    assert result["crm_lead_eval"]["field_checks"][0]["expected"] == "Asking about Offers"
    assert result["crm_lead_eval"]["crm_leads_flags"][0]["comment"] == (
        "Didn’t take the proper action on system such linking CST Contract etc."
    )


def test_inquiry_only_evaluation_cannot_pass_without_reason_check():
    call = CallTranscript(
        call_id="offer-inquiry", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-11", call_duration_seconds=60, department="Care",
        transcript="Patient: أريد تفاصيل عرض الجمعة للولادة الطبيعية",
    )
    lookup = {"status": "found", "record": {
        "new_lastcallresult": "Inquiry only [01]",
        "new_inquiryonlyreason": "Inquiry about services",
    }}
    incomplete = {"crm_lead_status": "match", "field_checks": [], "crm_leads_flags": []}
    llm = AsyncMock(side_effect=[(incomplete.copy(), None), (incomplete.copy(), None)])
    with patch("app.agent.nodes.fetch_crm_lead", return_value=lookup), patch(
        "app.agent.nodes._focused_llm_call", llm
    ):
        result = asyncio.run(validate_crm_lead({"call": call, "node_trace": []}, None))
    assert llm.await_count == 2
    assert result["error_node"] == "validate_crm_lead"
    assert "new_inquiryonlyreason" in result["error"]


def test_blank_inquiry_reason_is_a_violation_even_if_model_says_match():
    call = CallTranscript(
        call_id="offer-inquiry", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-11", call_duration_seconds=60, department="Care",
        transcript="Patient: أريد تفاصيل عرض الجمعة للولادة الطبيعية",
    )
    lookup = {"status": "found", "record": {
        "new_lastcallresult": "Inquiry only [01]",
        "new_inquiryonlyreason": None,
    }}
    incorrect = {
        "crm_lead_status": "match",
        "field_checks": [{
            "field": "new_inquiryonlyreason", "matches": True,
            "expected": "Asking about Offers", "actual": None,
        }],
        "crm_leads_flags": [],
    }
    with patch("app.agent.nodes.fetch_crm_lead", return_value=lookup), patch(
        "app.agent.nodes._focused_llm_call", AsyncMock(return_value=(incorrect, None))
    ):
        result = asyncio.run(validate_crm_lead({"call": call, "node_trace": []}, None))
    assert result["crm_lead_eval"]["field_checks"][0]["matches"] is False
    assert result["crm_lead_eval"]["crm_leads_flags"][0]["comment"] == (
        "Didn’t take the proper action on system such linking CST Contract etc."
    )
