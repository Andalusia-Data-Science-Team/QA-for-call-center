import csv
from pathlib import Path

import pytest

from app.FAQs.faq_validation import (
    detect_faq_escalation,
    lookup_faq_record,
    normalize_faq_phone,
    normalize_faq_text,
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
