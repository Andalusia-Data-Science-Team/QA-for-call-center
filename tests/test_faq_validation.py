import asyncio
import csv
import json
from pathlib import Path

import pytest
from app.agent import graph as agent_graph
from app.agent import nodes as agent_nodes

from app.FAQs.faq_validation import (
    detect_faq_escalation,
    lookup_faq_record,
    missing_faq_evaluation,
    normalize_faq_evaluation,
    normalize_faq_phone,
    normalize_faq_text,
)
from app.models.input import CallTranscript
from app.prompts.qa_prompt import (
    build_faq_validation_prompt,
    build_scoring_prompt,
)


FAQ_HEADERS = [
    "ID", "CustomerName", "mobile_phone", "BU", "Date", "Inquiry",
    "Speciality", "Response", "AgentName", "AgentEmail", "End Call Result",
    "Modified by ", "Response Time", "Injected to CRM",
]


def faq_row(**overrides):
    row = {
        "ID": "1",
        "CustomerName": "عبد الرحمن",
        "mobile_phone": "0555123456",
        "BU": "AHJ",
        "Date": "2026-09-14",
        "Inquiry": "استفسار عن تقرير طبي",
        "Speciality": "Urology",
        "Response": "",
        "AgentName": "Mahmoud Atef Helmy",
        "AgentEmail": "Mahmoud.Atef@Andalusiagroup.net",
        "End Call Result": "In Progress",
        "Modified by ": "Andalusia SharePoint",
        "Response Time": "",
        "Injected to CRM": "Yes",
    }
    row.update(overrides)
    return row


def write_faq_csv(tmp_path: Path, rows, headers=FAQ_HEADERS) -> Path:
    path = tmp_path / "faq.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize(
    "transcript",
    [
        "Agent: تم رفع الطلب للقسم المختص وسيتم التواصل معك",
        "Agent: رفعنا استفسارك للجهة المعنية",
        "Agent: أرسلت الشكوى إلى الفريق المسؤول",
        "Agent: Your request was escalated to the concerned department.",
    ],
)
def test_detect_faq_escalation_accepts_semantic_phrase_variants(transcript):
    assert detect_faq_escalation(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: أريد رفع التقرير الطبي",
        "Patient: أريد رفع الطلب للقسم المختص\nAgent: حاضر",
        "Agent: تم تحويلك إلى قسم الحجز",
        "Agent: القسم المختص متاح من التاسعة",
    ],
)
def test_detect_faq_escalation_rejects_unrelated_language(transcript):
    assert detect_faq_escalation(transcript) is False


def test_normalizers_align_arabic_names_and_saudi_phone_formats():
    assert normalize_faq_text("  أحمَد   عليّ ") == "احمد علي"
    assert normalize_faq_phone("+966 50-123-4567") == "501234567"
    assert normalize_faq_phone("0501234567") == "501234567"


def test_lookup_requires_same_day_phone_and_agent_identity(tmp_path):
    csv_path = write_faq_csv(tmp_path, [
        faq_row(ID="10", Date="2026-09-13"),
        faq_row(ID="11", AgentName="Other Agent", AgentEmail="other@example.com"),
        faq_row(ID="12", mobile_phone="0555999999"),
    ])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="0555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
    )
    assert result["status"] == "not_found"
    assert result["record"] is None


def test_lookup_prefers_email_then_highest_id(tmp_path):
    csv_path = write_faq_csv(tmp_path, [
        faq_row(ID="20", AgentEmail=""),
        faq_row(ID="21", AgentName="Different Spelling", AgentEmail="Mahmoud.Atef@Andalusiagroup.net"),
        faq_row(ID="22", AgentName="Different Spelling", AgentEmail="Mahmoud.Atef@Andalusiagroup.net"),
    ])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
    )
    assert result["status"] == "found"
    assert result["record"]["ID"] == "22"
    assert result["match"]["identity"] == "email"


def test_lookup_matches_normalized_agent_name_when_call_email_is_missing(tmp_path):
    csv_path = write_faq_csv(tmp_path, [
        faq_row(ID="30", AgentName="  MAHMOUD   ATEF HELMY ", AgentEmail="")
    ])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="+966555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email=None,
    )
    assert result["status"] == "found"
    assert result["record"]["ID"] == "30"
    assert result["match"]["identity"] == "name"


def test_lookup_returns_unavailable_when_required_columns_are_missing(tmp_path):
    csv_path = write_faq_csv(tmp_path, [faq_row()], headers=["ID", "Date"])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="0555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email=None,
    )
    assert result["status"] == "unavailable"
    assert "required columns" in result["message"]


def test_lookup_returns_unavailable_when_csv_does_not_exist(tmp_path):
    result = lookup_faq_record(
        csv_path=tmp_path / "missing.csv",
        patient_phone="0555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email=None,
    )
    assert result["status"] == "unavailable"
    assert result["record"] is None

def test_missing_faq_record_is_deterministic_c2b_017():
    result = missing_faq_evaluation("No same-day FAQ record.")

    assert result["faq_status"] == "violation"
    assert result["summary"] == "No same-day FAQ record."
    assert result["faq_flags"] == [
        {
            "type": "C2B",
            "severity": "moderate",
            "description": "C2B_017: FAQ request was not recorded on the call date.",
            "transcript_excerpt": "N/A",
        }
    ]


def test_normalize_faq_evaluation_enforces_rules_and_deduplicates_evidence():
    evaluation = normalize_faq_evaluation(
        {
            "faq_status": "violation",
            "summary": "FAQ values conflict with the chat.",
            "field_checks": [
                {
                    "field": "Response",
                    "matches": False,
                    "expected": "No response yet",
                    "actual": "Completed",
                    "rule_id": "C2C_023",
                    "reason": "The agent said the request was still pending.",
                    "transcript_excerpt": "لسه تحت الإجراء",
                },
                {
                    "field": "Response",
                    "matches": False,
                    "expected": "No response yet",
                    "actual": "Completed",
                    "rule_id": "C2C_023",
                    "reason": "The agent said the request was still pending.",
                    "transcript_excerpt": "لسه تحت الإجراء",
                },
                {
                    "field": "Escalation",
                    "matches": False,
                    "expected": "Required",
                    "actual": "Not performed",
                    "rule_id": "C2C_024",
                    "reason": "Required escalation was not performed.",
                    "transcript_excerpt": "لن أرفع طلب",
                },
                {
                    "field": "BU",
                    "matches": False,
                    "expected": "AHJ",
                    "actual": "MKR",
                    "rule_id": "C9_UNKNOWN",
                    "reason": "The business unit is wrong.",
                    "transcript_excerpt": "فرع حي الجامعة",
                },
            ],
            "faq_flags": [
                {
                    "rule_id": "C2C_023",
                    "description": "C2C_023: duplicated response mismatch.",
                    "transcript_excerpt": "لسه تحت الإجراء",
                }
            ],
        }
    )

    assert [flag["type"] for flag in evaluation["faq_flags"]] == [
        "C2C",
        "C2C",
        "C2B",
    ]
    assert [flag["severity"] for flag in evaluation["faq_flags"]] == [
        "critical",
        "critical",
        "moderate",
    ]
    assert [flag["description"].split(":", 1)[0] for flag in evaluation["faq_flags"]] == [
        "C2C_023",
        "C2C_024",
        "C2B_021",
    ]


def test_faq_prompt_contains_call_record_fields_and_existing_rules():
    call = CallTranscript(
        call_id="call-faq-1",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
        Patient_Phone="0555123456",
        call_date="2026-09-14",
        call_duration_seconds=120,
        department="Helpdesk",
        business_unit="LIVE",
        transcript=(
            "Patient: أنا عبد الرحمن وأستفسر عن التقرير الطبي\n"
            "Agent: تم رفع الطلب للقسم المختص"
        ),
    )
    record = faq_row(
        ID="44",
        CustomerName="عبد الرحمن",
        Response="سيتم الرد لاحقاً",
        **{"End Call Result": "In Progress"},
    )

    prompt = build_faq_validation_prompt(
        call,
        {"status": "found", "record": record},
        compliance_pillars="C2B_017 existing-regulation excerpt",
    )

    for column in (
        "AgentName",
        "AgentEmail",
        "mobile_phone",
        "Date",
        "CustomerName",
        "BU",
        "Inquiry",
        "Response",
        "End Call Result",
    ):
        assert column in prompt
    for rule_id in ("C2B_017", "C2B_021", "C2C_023", "C2C_024"):
        assert rule_id in prompt
    assert "mahmoud.atef@andalusiagroup.net" in prompt
    assert "C2B_017 existing-regulation excerpt" in prompt


def make_call(transcript="Agent: تم رفع الطلب للقسم المختص"):
    return CallTranscript(
        call_id="call-faq-node",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
        Patient_Phone="0555123456",
        call_date="2026-09-14",
        call_duration_seconds=120,
        department="Helpdesk",
        business_unit="LIVE",
        transcript=transcript,
    )


def test_detect_faq_escalation_node_sets_route_and_neutral_state():
    detected = asyncio.run(
        agent_nodes.detect_faq_escalation({"call": make_call()})
    )
    skipped = asyncio.run(
        agent_nodes.detect_faq_escalation(
            {"call": make_call("Agent: القسم المختص متاح من التاسعة")}
        )
    )

    assert detected["is_faq_escalation"] is True
    assert detected["faq_eval"]["faq_status"] == "pending"
    assert detected["node_trace"] == ["detect_faq_escalation"]
    assert skipped["is_faq_escalation"] is False
    assert skipped["faq_eval"]["faq_status"] == "skipped"


def test_validate_faq_record_missing_does_not_call_llm(monkeypatch):
    monkeypatch.setattr(
        agent_nodes,
        "lookup_faq_record",
        lambda **kwargs: {
            "status": "not_found",
            "record": None,
            "message": "No same-day FAQ record.",
        },
    )

    class RejectingLLM:
        async def complete(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for a missing FAQ row")

    result = asyncio.run(
        agent_nodes.validate_faq_record(
            {"call": make_call(), "compliance_pillars": "rules"},
            RejectingLLM(),
        )
    )

    assert result["faq_lookup"]["status"] == "not_found"
    assert result["faq_eval"]["faq_flags"][0]["description"].startswith("C2B_017:")
    assert "usage_list" not in result


def test_validate_faq_record_routes_unavailable_csv_to_error(monkeypatch):
    monkeypatch.setattr(
        agent_nodes,
        "lookup_faq_record",
        lambda **kwargs: {
            "status": "unavailable",
            "record": None,
            "message": "FAQ CSV could not be read.",
        },
    )

    result = asyncio.run(
        agent_nodes.validate_faq_record(
            {"call": make_call(), "compliance_pillars": "rules"},
            object(),
        )
    )

    assert result["error_node"] == "validate_faq_record"
    assert "FAQ CSV could not be read" in result["error"]


def test_validate_faq_record_normalizes_found_record_evaluation(monkeypatch):
    lookup = {
        "status": "found",
        "record": faq_row(ID="77"),
        "match": {"identity": "email", "faq_id": "77"},
        "message": "Same-day FAQ record found.",
    }
    monkeypatch.setattr(agent_nodes, "lookup_faq_record", lambda **kwargs: lookup)

    class FakeLLM:
        async def complete(self, system_prompt, user_prompt):
            assert "C2B_017 existing rules" in user_prompt
            return (
                json.dumps(
                    {
                        "faq_status": "violation",
                        "summary": "Wrong business unit.",
                        "field_checks": [
                            {
                                "field": "BU",
                                "matches": False,
                                "expected": "AHJ",
                                "actual": "MKR",
                                "rule_id": "C2B_021",
                                "reason": "The business unit is wrong.",
                                "transcript_excerpt": "N/A",
                            }
                        ],
                        "faq_flags": [],
                    }
                ),
                {
                    "provider": "test",
                    "model": "fake",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            )

    result = asyncio.run(
        agent_nodes.validate_faq_record(
            {
                "call": make_call(),
                "compliance_pillars": "C2B_017 existing rules",
            },
            FakeLLM(),
        )
    )

    assert result["faq_lookup"]["record"]["ID"] == "77"
    assert result["faq_eval"]["faq_flags"][0]["description"].startswith("C2B_021:")
    assert result["usage_list"][0]["total_tokens"] == 15
    assert result["node_trace"] == ["validate_faq_record"]


def test_scoring_prompt_includes_faq_validation_summary():
    summary = '{"faq_status":"violation","faq_flags":[{"type":"C2B"}]}'

    prompt = build_scoring_prompt(make_call(), faq_summary=summary)

    assert "FAQ VALIDATION" in prompt
    assert summary in prompt


def test_infer_overall_scoring_passes_faq_summary_to_prompt(monkeypatch):
    captured = {}

    async def fake_focused_call(
        node_name,
        call_id,
        user_prompt,
        llm_client,
        state,
        max_tokens=None,
    ):
        captured["prompt"] = user_prompt
        return (
            {
                "overall_assessment": "needs_review",
                "assessment_reasoning": "FAQ record missing.",
                "compliance_flags": [],
                "agent_performance": {
                    "professionalism_score": 0.8,
                    "Agent Classification": "C",
                    "Profiling Comment": "Poor Report",
                    "strengths": [],
                    "improvements": [],
                },
                "escalation_required": False,
                "escalation_reason": None,
                "_usage": {"total_tokens": 3},
            },
            None,
        )

    monkeypatch.setattr(agent_nodes, "_focused_llm_call", fake_focused_call)
    result = asyncio.run(
        agent_nodes.infer_overall_scoring(
            {
                "call": make_call(),
                "faq_eval": {
                    "faq_status": "violation",
                    "faq_flags": [{"description": "C2B_017 missing record"}],
                },
            },
            object(),
        )
    )

    assert '"faq_status": "violation"' in captured["prompt"]
    assert "C2B_017 missing record" in captured["prompt"]
    assert result["usage_list"] == [{"total_tokens": 3}]


def test_aggregate_results_merges_faq_flags_into_final_result():
    faq_flag = {
        "type": "C2B",
        "severity": "moderate",
        "description": "C2B_017: FAQ request was not recorded on the call date.",
        "transcript_excerpt": "N/A",
    }
    second_faq_flag = {
        "type": "C2B",
        "severity": "moderate",
        "description": "C2B_021: FAQ mismatch in BU: wrong business unit.",
        "transcript_excerpt": "N/A",
    }
    state = {
        "call": make_call(),
        "behavioral_eval": {
            "professionalism_score": 0.8,
            "strengths": [],
            "improvements": [],
        },
        "faq_eval": {
            "faq_status": "violation",
            "faq_flags": [faq_flag, second_faq_flag],
        },
        "scoring_eval": {
            "overall_assessment": "needs_review",
            "assessment_reasoning": "A required FAQ record is missing.",
            "compliance_flags": [],
            "agent_performance": {
                "professionalism_score": 0.8,
                "Agent Classification": "C",
                "Profiling Comment": "Poor Report",
                "strengths": [],
                "improvements": [],
            },
            "escalation_required": False,
            "escalation_reason": None,
        },
    }

    output = asyncio.run(agent_nodes.aggregate_results(state))

    assert "error" not in output
    assert [flag.description for flag in output["result"].compliance_flags] == [
        faq_flag["description"],
        second_faq_flag["description"],
    ]


def test_faq_router_only_validates_detected_escalation_claims():
    assert agent_graph._faq_router({"is_faq_escalation": True}) == "validate"
    assert agent_graph._faq_router({"is_faq_escalation": False}) == "skip"
    assert agent_graph._faq_router({}) == "skip"


def test_graph_wires_faq_branch_between_crm_validation_and_scoring():
    compiled = agent_graph.build_qa_graph(object())
    edge_pairs = {
        (edge.source, edge.target)
        for edge in compiled.get_graph().edges
    }

    assert {
        ("validate_crm_lead", "detect_faq_escalation"),
        ("detect_faq_escalation", "validate_faq_record"),
        ("detect_faq_escalation", "infer_overall_scoring"),
        ("validate_faq_record", "infer_overall_scoring"),
        ("validate_faq_record", "handle_error"),
    } <= edge_pairs
