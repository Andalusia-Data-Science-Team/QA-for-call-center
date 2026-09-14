import csv
from pathlib import Path

import pytest

from app.FAQs.faq_validation import (
    detect_faq_escalation,
    lookup_faq_record,
    missing_faq_evaluation,
    normalize_faq_evaluation,
    normalize_faq_phone,
    normalize_faq_text,
)
from app.models.input import CallTranscript
from app.prompts.qa_prompt import build_faq_validation_prompt


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
