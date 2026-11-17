"""Missed booking scripts must be represented by a regulation-backed card."""

import asyncio

from app.agent.nodes import aggregate_results, normalize_script_evaluation
from app.models.input import CallTranscript


MOBILE_SCRIPT = "رقم الموبايل حضرتك المسجل على الملف هو نفس الرقم اللي بتكلمني منه تمام ؟"


def _mobile_script_finding():
    return {
        "type": "NC",
        "attribute": "If Customer Need Booking / Accept The Offer",
        "comment": MOBILE_SCRIPT,
        "description": "The agent did not verify the patient's registered mobile number while rescheduling.",
        "transcript_excerpt": "Agent: تم تأكيد الحجز يوم الأربعاء الساعة 12",
    }


def test_missing_mobile_confirmation_uses_an_existing_policy_item():
    result = normalize_script_evaluation({
        "accuracy_score": 0.9,
        "script_flags": [_mobile_script_finding()],
    })
    flag = result["script_flags"][0]
    assert flag["type"] == "C2B"
    assert flag["attribute"] == "Displays positive company image"
    assert flag["comment"] == "Didn't Commit to call script"
    assert "registered mobile number" in flag["description"]


def test_missing_mobile_confirmation_survives_final_aggregation():
    call = CallTranscript(
        call_id="script-mobile-test", agent_name="Agent", Patient_Phone="500000000",
        call_date="2026-11-09", call_duration_seconds=60,
        department="Call Center",
        transcript="Patient: أغير الموعد\nAgent: تم تأكيد الحجز يوم الأربعاء الساعة 12",
    )
    state = {
        "call": call,
        "script_eval": {"accuracy_score": 0.9, "script_flags": [_mobile_script_finding()]},
        "scoring_eval": {
            "overall_assessment": "needs_review",
            "assessment_reasoning": "The agent did not verify the registered mobile number.",
            "agent_performance": {"Agent Classification": "C"},
            "escalation_required": False,
        },
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))["result"]
    assert len(result.compliance_flags) == 1
    assert result.compliance_flags[0].type == "C2B"
    assert result.compliance_flags[0].comment == "Didn't Commit to call script"
