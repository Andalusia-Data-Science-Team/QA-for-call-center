"""The customer-care export reflects the already displayed QA call list."""

from io import BytesIO
from math import nan
from datetime import timedelta

from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app import main
from app.services.qa_results_export import build_customer_care_workbook
from app.services.regulation_catalog import regulation_items


def _sheet(calls):
    return load_workbook(BytesIO(build_customer_care_workbook(calls))).active


def _flag(item, number):
    return {
        "type": item["type"], "attribute": item["attribute"],
        "comment": item["comment"], "description": f"Issue {number}",
        "severity": "critical", "transcript_excerpt": f"Evidence {number}",
        "escalated": number % 2 == 0,
        "qa_reviewed": number == 6,
        "qa_review_comment": "Confirmed" if number == 6 else None,
    }


def test_export_keeps_one_chat_per_row_and_all_of_its_flags():
    flags = [_flag(item, i) for i, item in enumerate(
        (item for item in regulation_items() if item["comment"] != "Other"), 1) if i <= 6]
    sheet = _sheet([{
        "call_id": "chat-1", "agent_id": "123", "agent_name": "Ahmed",
        "bu": "Care", "supervisor_name": "Sara", "call_date": "2026-11-04T14:35:00",
        "call_duration_seconds": 120, "customer_phone_number": "01234567890",
        "overall_assessment": "escalate", "agent_classification": "B",
        "profiling_comment": "Needs coaching", "coaching_status": "pending",
        "conversation_link": "https://example.test/chat-1", "compliance_flags": flags,
    }])

    assert sheet.title == "Customer Care"
    assert sheet.max_row == 3
    assert [sheet[f"{col}1"].value for col in ("A", "J", "L", "T", "AY", "BF")] == [
        "Month", "Agent", "Call Date", "Pillar", "Agent Classification", "Action Plan",
    ]
    assert sheet["T2"].value == "Error 1"
    assert sheet["AJ2"].value == "Error 5"
    assert sheet["E3"].value == "123"
    assert sheet["J3"].value == "Ahmed"
    assert sheet["L3"].value == "2026-11-04"
    assert sheet["M3"].value == "14:35:00"
    assert sheet["N3"].value == timedelta(seconds=120)
    assert sheet["O3"].value == "01234567890"
    assert sheet["T3"].value == flags[0]["type"]
    assert sheet["U3"].value == flags[0]["attribute"]
    assert sheet["V3"].value == flags[0]["comment"]
    assert "Issue 1" in sheet["W3"].value
    assert sheet["AJ3"].value == flags[4]["type"]
    assert "Issue 6" in sheet["BG3"].value
    assert "Confirmed" in sheet["BG3"].value
    assert sheet["BH3"].value == "https://example.test/chat-1"
    assert sheet["BI3"].value == "chat-1"


def test_export_leaves_missing_and_nonfinite_values_empty():
    sheet = _sheet([{
        "call_id": "chat-2", "agent_name": None, "weighted_score": nan,
        "profiling_comment": "NaN", "compliance_flags": [],
    }])
    assert sheet.max_row == 3
    assert sheet["J3"].value is None
    assert sheet["L3"].value is None
    assert sheet["AY3"].value == "A"
    assert sheet["AZ3"].value is None
    assert sheet["BG3"].value is None
    assert sheet["BH3"].value is None
    assert sheet["BI3"].value == "chat-2"


def test_error_counts_match_every_flag_type_for_each_chat():
    items = regulation_items()
    flag_types = ["C2C", "C2B", "C2C", "NC", "C2Com", "NC"]
    flags = [_flag(next(item for item in items if item["type"] == flag_type
                        and item["comment"] != "Other"), i)
             for i, flag_type in enumerate(flag_types, 1)]
    sheet = _sheet([
        {"call_id": "chat-1", "compliance_flags": flags},
        {"call_id": "chat-2", "compliance_flags": []},
    ])
    assert [sheet[f"{column}3"].value for column in "PQRS"] == [2, 1, 1, 2]
    assert [sheet[f"{column}4"].value for column in "PQRS"] == [0, 0, 0, 0]
    assert sheet["BG3"].value.count("Description:") == 6


def test_export_does_not_include_unmatched_escalations():
    sheet = _sheet([{
        "call_id": "chat-3", "overall_assessment": "escalate",
        "escalation_required": True, "escalation_reason": "Offer issue",
        "compliance_flags": [{
            "type": "C2B", "attribute": "Offer Recommendation Quality",
            "comment": "Old offer item", "description": "Offer skipped",
        }],
    }])
    assert sheet["AU3"].value == "pass"
    assert sheet["AY3"].value == "A"
    assert [sheet[f"{column}3"].value for column in "PQRS"] == [0, 0, 0, 0]
    assert sheet["T3"].value is None
    assert sheet["BG3"].value is None


def test_export_endpoint_uses_only_the_displayed_snapshot():
    response = TestClient(main.app).post("/api/qa-results/export", json=[{
        "call_id": "visible-chat", "agent_name": "Displayed", "compliance_flags": [],
    }])
    assert response.status_code == 200
    assert "spreadsheetml.sheet" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    sheet = load_workbook(BytesIO(response.content)).active
    assert sheet.max_row == 3
    assert sheet["J3"].value == "Displayed"
    assert sheet["BI3"].value == "visible-chat"


def test_agent_session_results_export_nested_performance_and_one_row_per_chat():
    item = next(item for item in regulation_items()
                if item["type"] == "C2Com" and item["comment"] == "Released customer personal data")
    response = TestClient(main.app).post("/api/qa-results/export", json=[{
        "call_id": "session-chat-1", "agent_name": "Rania",
        "business_unit": "Customer Care", "supervisor_name": "Sara",
        "conversation_link": "https://example.test/session-chat-1",
        "overall_assessment": "escalate", "escalation_required": True,
        "agent_performance": {
            "professionalism_score": 0.7,
            "Agent Classification": "C",
            "Profiling Comment": "Poor Process",
            "improvements": ["Check policy before replying"],
        },
        "compliance_flags": [_flag(item, 1)],
    }, {
        "call_id": "session-chat-2", "agent_name": "Rania",
        "overall_assessment": "pass", "escalation_required": False,
        "agent_performance": {
            "professionalism_score": 0.9, "Agent Classification": "A",
        },
        "compliance_flags": [],
    }])
    assert response.status_code == 200
    sheet = load_workbook(BytesIO(response.content)).active
    assert sheet.max_row == 4
    assert sheet["F3"].value == "Customer Care"
    assert sheet["H3"].value == "Sara"
    assert sheet["J3"].value == "Rania"
    assert sheet["R3"].value == 1
    assert sheet["U3"].value == item["attribute"]
    assert sheet["V3"].value == item["comment"]
    assert sheet["AY3"].value == "C"
    assert sheet["AZ3"].value == "Poor Process"
    assert sheet["BF3"].value == "Check policy before replying"
    assert sheet["BH3"].value == "https://example.test/session-chat-1"
    assert sheet["BI4"].value == "session-chat-2"
    assert sheet["L3"].value is None
    assert sheet["N3"].value is None
