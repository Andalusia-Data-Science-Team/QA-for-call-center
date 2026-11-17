"""Missing required MPI fields must survive profile validation and aggregation."""

import asyncio

from app.agent.patient_profile_nodes import normalize_patient_profile_evaluation
from app.agent.nodes import aggregate_results
from app.models.input import CallTranscript
from app.services.regulation_catalog import matched_regulation_flags


def _missing(field, asked=False, criticality="important"):
    return {
        "field_name": field,
        "field_value": None,
        "agent_asked": asked,
        "criticality": criticality,
        "reason": f"{field} is empty in MPI",
    }


def test_unasked_important_field_cannot_pass_without_a_flag():
    result = normalize_patient_profile_evaluation({
        "applicable": True,
        "validation_status": "pass",
        "missing_fields": [_missing("Nationality")],
        "profile_updates": [],
        "compliance_flags": [],
    })

    assert result["validation_status"] == "incomplete"
    flags = matched_regulation_flags(result["compliance_flags"])
    assert len(flags) == 1
    assert flags[0]["type"] == "C2C"
    assert flags[0]["comment"] == "Described Inaccurate, Incomplete Information-None Medical"
    assert "Nationality" in flags[0]["description"]


def test_all_unasked_fields_are_named_in_one_profile_finding():
    result = normalize_patient_profile_evaluation({
        "validation_status": "pass",
        "missing_fields": [
            _missing("MaritalStatus"),
            _missing("IdentificationType"),
            _missing("DateOfBirth", criticality="critical"),
        ],
        "compliance_flags": [],
    })
    flags = matched_regulation_flags(result["compliance_flags"])
    assert len(flags) == 1
    assert all(field in flags[0]["description"] for field in (
        "MaritalStatus", "IdentificationType", "DateOfBirth",
    ))


def test_agent_asked_for_missing_field_does_not_create_a_violation():
    result = normalize_patient_profile_evaluation({
        "validation_status": "pass",
        "missing_fields": [_missing("Nationality", asked=True)],
        "compliance_flags": [],
    })
    assert result["validation_status"] == "pass"
    assert result["compliance_flags"] == []


def test_existing_profile_flag_is_merged_not_duplicated():
    result = normalize_patient_profile_evaluation({
        "validation_status": "incomplete",
        "missing_fields": [_missing("Nationality"), _missing("MaritalStatus")],
        "compliance_flags": [{
            "type": "C2C",
            "attribute": "Provide complete accurate and relevant information",
            "comment": "Described Inaccurate, Incomplete Information-None Medical",
            "description": "C2C_021: Nationality was not collected.",
            "transcript_excerpt": "N/A",
        }],
    })
    flags = matched_regulation_flags(result["compliance_flags"])
    assert len(flags) == 1
    assert "Nationality" in flags[0]["description"]
    assert "MaritalStatus" in flags[0]["description"]


def test_profile_field_survives_same_policy_item_from_other_evaluator():
    profile = normalize_patient_profile_evaluation({
        "validation_status": "pass",
        "missing_fields": [_missing("Nationality")],
        "compliance_flags": [],
    })
    call = CallTranscript(
        call_id="profile-merge-test", agent_name="Agent", Patient_Phone="500000000",
        call_date="2026-11-09", call_duration_seconds=30,
        department="Scheduling", transcript="Patient: hello\nAgent: hello",
    )
    result = asyncio.run(aggregate_results({
        "call": call,
        "node_trace": [],
        "compliance_eval": {"compliance_flags": [{
            "type": "C2C",
            "attribute": "Provide complete accurate and relevant information",
            "comment": "Described Inaccurate, Incomplete Information-None Medical",
            "description": "C2C_021: Wrong nonmedical information was provided.",
            "transcript_excerpt": "Agent: hello",
        }]},
        "patient_profile_eval": profile,
        "scoring_eval": {
            "overall_assessment": "pass",
            "assessment_reasoning": "No violations found.",
            "agent_performance": {"Agent Classification": "A"},
            "escalation_required": False,
        },
    }))
    flags = result["parsed_data"]["compliance_flags"]
    assert len(flags) == 1
    assert "Nationality" in flags[0]["description"]
    assert "Wrong nonmedical information" in flags[0]["description"]
