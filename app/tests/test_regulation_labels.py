"""A violation's policy labels come only from the regulation YAML catalog."""

import asyncio
from fastapi.testclient import TestClient

from app.agent.nodes import aggregate_results, integrity_check
from app import main
from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.prompts.qa_prompt import build_behavioral_prompt
from app.services.criteria_loader import CriteriaLoader
from app.services.regulation_catalog import canonicalize_flag, regulation_items


def test_regulation_prompts_include_exact_attribute_item_and_id():
    compliance = CriteriaLoader().compliance_pillars()
    reservation = CriteriaLoader().reservation_pillars()
    assert "Misleading the Privacy Policy" in compliance
    assert "C2Com_001: Released customer personal data" in compliance
    assert "Submitting the right action on the system" in reservation
    assert "C2C_005: Made wrong reservation" in reservation


def test_behavioral_prompt_uses_catalog_for_labels_only():
    call = CallTranscript(
        call_id="chat-1", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-05", call_duration_seconds=120,
        department="Care", transcript="Agent: Hello",
    )
    prompt = build_behavioral_prompt(
        call, regulation_catalog=CriteriaLoader().compliance_pillars())
    assert "C2C_014: Bad Handling skills" in prompt
    assert '"attribute":' in prompt and '"comment":' in prompt
    assert "label existing behavioral violations only" in prompt


def test_catalog_canonicalizes_exact_yaml_pair_without_touching_evidence():
    flag = {
        "type": "C2Com",
        "attribute": " misleading the privacy policy ",
        "comment": "released customer personal data",
        "description": "Agent shared identifying details.",
        "transcript_excerpt": "Agent: The number is ...",
    }
    actual = canonicalize_flag(flag)
    assert actual["attribute"] == "Misleading the Privacy Policy"
    assert actual["comment"] == "Released customer personal data"
    assert actual["description"] == flag["description"]
    assert actual["transcript_excerpt"] == flag["transcript_excerpt"]
    assert flag["attribute"] == " misleading the privacy policy "


def test_catalog_recovers_exact_item_id_from_existing_description():
    actual = canonicalize_flag({
        "type": "C2C", "description": "C2C_005: The agent booked an ineligible patient.",
        "transcript_excerpt": "Agent: booked",
    })
    assert actual["attribute"] == "Submitting the right action on the system"
    assert actual["comment"] == "Made wrong reservation"


def test_catalog_never_invents_labels_for_unmatched_flag():
    flag = {
        "type": "C2B", "attribute": "Invented policy", "comment": "Invented item",
        "description": "The bank account did not match.", "transcript_excerpt": "Agent: IBAN",
    }
    actual = canonicalize_flag(flag)
    assert actual["attribute"] is None
    assert actual["comment"] is None
    assert actual["description"] == flag["description"]


def test_catalog_includes_workbook_rules_from_top_level_yaml():
    actual = canonicalize_flag({
        "type": "NC", "attribute": "Voice tone", "comment": "Unclear Voice Tone",
        "description": "Agent was hard to hear.",
    })
    assert actual["attribute"] == "Voice tone"
    assert actual["comment"] == "Unclear Voice Tone"


def test_ambiguous_or_generic_item_id_cannot_create_an_escalation():
    actual = canonicalize_flag({
        "type": "C2C", "description": "C2C_030: eligibility misinformation",
    })
    assert actual["attribute"] is None
    assert actual["comment"] is None


def test_catalog_has_no_labels_removed_by_workbook_cleanup():
    actual = canonicalize_flag({
        "type": "C2B", "outcome": "OFFER_SKIPPED",
        "description": "A qualifying offer was available but never discussed.",
    })
    assert actual["attribute"] is None
    assert actual["comment"] is None
    assert all(item["comment"] == "Other" for item in regulation_items() if item["id"] == "C2C_030")
    assert all(not item["id"].endswith("_OFF_001") for item in regulation_items())
    assert next(item["comment"] for item in regulation_items() if item["id"] == "C2C_032") == (
        "Released / Closed / Aviod the call intentionaly"
    )


def test_eligibility_check_uses_existing_workbook_item():
    call = CallTranscript(
        call_id="chat-eligibility", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-05", call_duration_seconds=120,
        department="Care", transcript="Agent: You are eligible.",
    )
    result = QAAnalysisResult(
        call_id=call.call_id, agent_name=call.agent_name,
        overall_assessment="needs_review", assessment_reasoning="Original reasoning.",
        compliance_flags=[{
            "type": "C2C", "severity": "critical",
            "attribute": "Provide complete accurate and relevant information",
            "comment": "Described wrong Information",
            "description": "The agent stated an incorrect eligibility outcome.",
            "transcript_excerpt": "Agent: You are eligible.",
        }],
        agent_performance={"professionalism_score": 0.8, "Agent Classification": "C"},
        escalation_required=False,
    )
    checked = asyncio.run(integrity_check({"call": call, "result": result, "node_trace": []}))["result"]
    assert checked.overall_assessment == "escalate"
    assert checked.assessment_reasoning == "Original reasoning."


def test_aggregation_keeps_assessment_and_canonicalizes_flag_labels():
    call = CallTranscript(
        call_id="12345678-1234-1234-1234-123456789abc", agent_name="Agent",
        Patient_Phone="501234567", call_date="2026-11-05", call_duration_seconds=120,
        department="Care", transcript="Agent: The number is ...",
    )
    state = {
        "call": call,
        "compliance_eval": {"compliance_flags": [{
            "type": "C2Com", "severity": "critical",
            "attribute": "Misleading the Privacy Policy",
            "comment": "Released customer personal data",
            "description": "Agent shared identifying details.",
            "transcript_excerpt": "Agent: The number is ...",
        }]},
        "scoring_eval": {
            "overall_assessment": "escalate",
            "assessment_reasoning": "Original assessment reasoning must remain intact.",
            "agent_performance": {"Agent Classification": "D"},
            "escalation_required": True,
        },
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))["result"]
    assert result.assessment_reasoning == "Original assessment reasoning must remain intact."
    assert result.compliance_flags[0].attribute == "Misleading the Privacy Policy"
    assert result.compliance_flags[0].comment == "Released customer personal data"
    assert result.compliance_flags[0].description == "Agent shared identifying details."


def test_aggregation_discards_unmatched_escalation_and_resets_policy_status():
    call = CallTranscript(
        call_id="chat-unmatched", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-05", call_duration_seconds=120,
        department="Care", transcript="Agent: Hello",
    )
    state = {
        "call": call,
        "offer_eval": {"offer_outcome": "OFFER_SKIPPED", "offer_flags": [{
            "type": "C2B", "description": "The agent skipped an offer.",
            "transcript_excerpt": "N/A",
        }]},
        "scoring_eval": {
            "overall_assessment": "escalate", "escalation_required": True,
            "escalation_reason": "Unmatched offer finding",
            "assessment_reasoning": "Original assessment reasoning must remain intact.",
            "agent_performance": {"Agent Classification": "D"},
        },
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))["result"]
    assert result.compliance_flags == []
    assert result.overall_assessment == "pass"
    assert result.escalation_required is False
    assert result.escalation_reason is None
    assert result.agent_performance.agent_classification == "A"
    assert result.assessment_reasoning == "Original assessment reasoning must remain intact."


def test_only_noncritical_match_cannot_escalate_because_of_unmatched_critical():
    call = CallTranscript(
        call_id="chat-mixed", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-05", call_duration_seconds=120,
        department="Care", transcript="Agent: Hello",
    )
    state = {
        "call": call,
        "compliance_eval": {"compliance_flags": [{
            "type": "NC", "attribute": "Voice tone", "comment": "Unclear Voice Tone",
            "description": "The agent's voice was unclear.",
            "transcript_excerpt": "Agent: Hello",
        }, {
            "type": "C2B", "description": "Unmatched bank error.",
            "transcript_excerpt": "N/A",
        }]},
        "scoring_eval": {
            "overall_assessment": "escalate", "escalation_required": True,
            "assessment_reasoning": "Original assessment.",
            "agent_performance": {"Agent Classification": "D"},
        },
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))["result"]
    assert [(flag.attribute, flag.comment) for flag in result.compliance_flags] == [
        ("Voice tone", "Unclear Voice Tone")
    ]
    assert result.overall_assessment == "needs_review"
    assert result.escalation_required is False
    assert result.agent_performance.agent_classification == "B"


def test_supervisor_page_contains_only_current_catalog_pairs():
    response = TestClient(main.app).get("/qa-supervisor")
    assert response.status_code == 200
    assert 'id="regulationPairs"' in response.text
    assert "Unclear Voice Tone" in response.text
    assert "Offer Recommendation Quality" not in response.text


def test_escalation_endpoint_rejects_unmatched_regulation(monkeypatch):
    updates = []
    monkeypatch.setattr(main, "update_escalation_row", lambda *args, **kwargs: updates.append(kwargs) or 1)
    client = TestClient(main.app)
    invalid = client.post("/escalations", json={
        "call_id": "chat-1", "flag_type": "C2B",
        "attribute": "Offer Recommendation Quality", "comment": "Skipped an offer",
        "flag_description": "The agent skipped an offer.",
    })
    assert invalid.status_code == 400
    assert updates == []

    valid = client.post("/escalations", json={
        "call_id": "chat-1", "flag_type": "C2Com",
        "attribute": "Misleading the Privacy Policy",
        "comment": "Released customer personal data",
        "flag_description": "Agent shared identifying details.",
    })
    assert valid.status_code == 200
    assert updates[0]["flag_type"] == "C2Com"
    assert updates[0]["attribute"] == "Misleading the Privacy Policy"
    assert updates[0]["comment"] == "Released customer personal data"


def test_stored_results_keep_each_rows_attribute_and_comment(monkeypatch):
    rows = [{
        "Call_ID": "chat-1", "Overall_Assessment": "escalate",
        "Compliance_Flag_Type": "C2Com", "Severity": "critical",
        "Attribute": "Misleading the Privacy Policy",
        "Comment": "Released customer personal data",
        "Flag_Description": "Agent shared identifying details.",
        "Transcript_Excerpt": "Agent: The number is ...",
    }, {
        "Call_ID": "chat-1", "Overall_Assessment": "escalate",
        "Compliance_Flag_Type": "C2Com", "Severity": "critical",
        "Attribute": "Misleading the Privacy Policy",
        "Comment": "Released customer personal data",
        "Flag_Description": "The same policy item from another evaluation pass.",
        "Transcript_Excerpt": "Agent: Different excerpt.",
    }, {
        "Call_ID": "chat-1", "Overall_Assessment": "escalate",
        "Compliance_Flag_Type": "C2C", "Severity": "critical",
        "Attribute": "Submitting the right action on the system",
        "Comment": "Made wrong reservation",
        "Flag_Description": "An invalid reservation was saved.",
        "Transcript_Excerpt": "Agent: Done.",
    }, {
        "Call_ID": "chat-1", "Overall_Assessment": "escalate",
        "Compliance_Flag_Type": "C2C", "Severity": "critical",
        "Attribute": None, "Comment": None,
        "Flag_Description": "C2C_014: The agent used bad handling skills.",
        "Transcript_Excerpt": "Agent: Wait.",
    }, {
        "Call_ID": "chat-2", "Overall_Assessment": "escalate",
        "Escalation_Required": True, "Escalation_Reason": "Offer skipped",
        "Compliance_Flag_Type": "C2B", "Severity": "critical",
        "Attribute": "Offer Recommendation Quality", "Comment": "Old offer item",
        "Flag_Description": "The agent skipped an offer.",
        "Transcript_Excerpt": "N/A",
    }]

    class FakeConnection:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, *args):
            class Result:
                def mappings(self): return self
                def all(self): return rows
            return Result()

    class FakeEngine:
        def connect(self): return FakeConnection()

    monkeypatch.setattr("sqlalchemy.create_engine", lambda *args, **kwargs: FakeEngine())
    monkeypatch.setattr(main.json, "load", lambda _: {"DB_NAMES": {"QA_DB": {
        "driver": "test", "Server": "test", "Database": "test", "UID": "test", "PWD": "test",
    }}})
    response = TestClient(main.app).get("/api/qa-results")
    assert response.status_code == 200
    calls = response.json()["results"]
    assert len(calls) == 2
    assert [(f["attribute"], f["comment"]) for f in calls[0]["compliance_flags"]] == [
        ("Misleading the Privacy Policy", "Released customer personal data"),
        ("Submitting the right action on the system", "Made wrong reservation"),
        ("Effective trouble shooting", "Bad Handling skills"),
    ]
    assert calls[0]["compliance_flags"][0]["description"] == "Agent shared identifying details."
    assert calls[1]["compliance_flags"] == []
    assert calls[1]["overall_assessment"] == "pass"
    assert calls[1]["escalation_required"] is False
    assert calls[1]["escalation_reason"] is None
    assert response.json()["summary"]["escalate"] == 1
