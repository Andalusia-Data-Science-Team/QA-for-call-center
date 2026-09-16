"""Tests for app.agent.nodes.aggregate_results' flag-merging/normalisation
logic — specifically the "positive" type/severity mix-up regression.

Real regression: call 57C946E6-1F85-F111-B337-000D3AA9D4A7 crashed
aggregate_results' Pydantic validation with
    compliance_flags.0.type: Input should be 'C2C', 'C2B', 'C2Com' or 'NC'
    (input_value='positive')
The ComplianceFlag schema's `type` field only accepts {C2C, C2B, C2Com,
NC} (app.models.output.FlagType) — "positive" is a valid SEVERITY value
instead (app.models.output.Severity). Every focused-evaluation prompt
shows the LLM "positive" as an example severity, and the offer-evaluation
prompt specifically also shows it as an example type, so ANY of the
behavioral/compliance/reservation/offer/script/scoring LLM calls can echo
"positive" into `type` instead of `severity` — not just the offer node,
which is what a narrower, offer-only fix previously missed.
"""
from __future__ import annotations

import asyncio

from app.agent.nodes import _normalize_flag_type, aggregate_results
from app.models.input import CallTranscript


def call() -> CallTranscript:
    return CallTranscript(
        call_id="57C946E6-1F85-F111-B337-000D3AA9D4A7", agent_name="Agent",
        Patient_Phone="501234567", call_date="2026-09-14", call_duration_seconds=1,
        department="Scheduling", transcript="Patient: hi\nAgent: hello",
    )


def base_scoring_eval() -> dict:
    """Minimal scoring_eval shape aggregate_results reads unconditionally
    (overall_assessment/agent_performance/Agent Classification) so a test
    state doesn't need to supply every possible upstream node."""
    return {
        "overall_assessment": "pass",
        "assessment_reasoning": "No violations found.",
        "agent_performance": {"Agent Classification": "A"},
        "escalation_required": False,
        "escalation_reason": None,
    }


# ── _normalize_flag_type (pure function) ────────────────────────────────────

def test_normalize_flag_type_maps_positive_type_to_nc_with_positive_severity():
    flag = {"type": "positive", "severity": "critical", "description": "d", "transcript_excerpt": "e"}
    normalized = _normalize_flag_type(flag)
    assert normalized["type"] == "NC"
    assert normalized["severity"] == "positive"
    # Original dict is untouched — callers must never rely on in-place mutation.
    assert flag["type"] == "positive"


def test_normalize_flag_type_leaves_valid_type_unchanged():
    flag = {"type": "C2B", "severity": "moderate", "description": "d", "transcript_excerpt": "e"}
    assert _normalize_flag_type(flag) == flag


# ── aggregate_results: the exact regression, from a NON-offer source ───────

def test_positive_type_from_behavioral_flags_no_longer_crashes_aggregation():
    """The exact reported regression: a 'positive'-typed flag from
    BEHAVIORAL evaluation (not the offer node, which already had its own
    narrower fix) must not crash Pydantic validation."""
    state = {
        "call": call(),
        "behavioral_eval": {
            "professionalism_score": 0.9,
            "behavioral_flags": [{
                "type": "positive", "severity": "critical",
                "description": "Agent greeted the patient warmly and professionally.",
                "transcript_excerpt": "Agent: hello",
            }],
            "strengths": ["Warm greeting"], "improvements": [],
        },
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    assert result["result"] is not None
    flags = result["parsed_data"]["compliance_flags"]
    assert len(flags) == 1
    assert flags[0]["type"] == "NC"
    assert flags[0]["severity"] == "positive"


def test_positive_type_from_compliance_reservation_script_scoring_all_normalize():
    """Every focused-evaluation flag list can independently carry a
    'positive'-typed flag — none of them may bypass normalisation."""
    positive_flag = {
        "type": "positive", "severity": "critical",
        "description": "d", "transcript_excerpt": "e",
    }
    state = {
        "call": call(),
        "compliance_eval": {"compliance_flags": [dict(positive_flag, transcript_excerpt="compliance")]},
        "reservation_eval": {"reservation_flags": [dict(positive_flag, transcript_excerpt="reservation")]},
        "script_eval": {"script_flags": [dict(positive_flag, transcript_excerpt="script")]},
        "scoring_eval": {**base_scoring_eval(), "compliance_flags": [dict(positive_flag, transcript_excerpt="scoring")]},
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    flags = result["parsed_data"]["compliance_flags"]
    assert len(flags) == 4
    assert all(f["type"] == "NC" and f["severity"] == "positive" for f in flags)


def test_positive_type_from_offer_flags_still_normalizes():
    """Regression guard for the ORIGINAL (narrower) fix this generalises —
    the offer node's own 'positive' type must keep working exactly as
    before."""
    state = {
        "call": call(),
        "offer_eval": {"offer_flags": [{
            "type": "positive", "severity": "critical",
            "description": "Agent recommended a matching offer.",
            "transcript_excerpt": "Agent: offer",
        }]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    flags = result["parsed_data"]["compliance_flags"]
    assert flags[0]["type"] == "NC"
    assert flags[0]["severity"] == "positive"


def test_valid_flag_types_are_unaffected_by_normalization():
    state = {
        "call": call(),
        "compliance_eval": {"compliance_flags": [{
            "type": "C2Com", "severity": "critical",
            "description": "d", "transcript_excerpt": "e",
        }]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    flags = result["parsed_data"]["compliance_flags"]
    assert flags[0]["type"] == "C2Com"
    assert flags[0]["severity"] == "critical"


# ── Doctor-validation failure_details/warning_details -> compliance flags
# (Part: doctor-validation results/logging/persistence/UI presentation) ──
# warning_details (e.g. name_completeness) must NEVER become a C2B
# compliance flag, never affect overall_assessment/escalation; genuine
# failure_details must become a DETAILED C2B flag, not just the old
# generic "N of M recommended doctor(s) failed validation: names."
# sentence.

def _warning_only_doctor_validation() -> dict:
    return {
        "outcome": "PASS", "reason": "All 1 recommended doctors resolved and passed validation.",
        "is_violation": False, "result_shape": "multi_doctor",
        "doctors": [{
            "input_name": "بدريه", "doctor_resolved": True, "doctor_key": "110411",
            "doctor_name_ar": "بدرية البيروتي", "doctor_name_en": "Badria Bairuti",
            "outcome": "PASS", "is_violation": False,
            "failure_details": [],
            "warning_details": [{
                "field": "name_completeness", "label": "Doctor name incomplete",
                "outcome": "WARNING", "is_violation": False,
                "chat_value": "بدريه", "crm_value": "بدرية البيروتي",
                "reason": "The agent stated only the doctor's first name. The preferred practice is to provide at least the first and second name.",
                "transcript_excerpt": "متاح الطبيبه بدريه",
            }],
        }],
    }


def test_warning_only_doctor_validation_retains_pass_assessment():
    """Item 13."""
    state = {
        "call": call(),
        "doctor_validation": _warning_only_doctor_validation(),
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    assert result["parsed_data"]["overall_assessment"] == "pass"


def test_warning_only_doctor_validation_creates_no_c2b_flag():
    """Item 14 — the warning must appear on the separate doctor_warnings
    surface, never in compliance_flags as a C2B (or any other) flag."""
    state = {
        "call": call(),
        "doctor_validation": _warning_only_doctor_validation(),
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    flags = result["parsed_data"]["compliance_flags"]
    assert not any(f["type"] == "C2B" for f in flags)
    warnings = result["parsed_data"]["doctor_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["type"] != "C2B"
    assert "بدرية البيروتي" in warnings[0]["description"]


def test_warning_card_contains_no_escalation_action():
    """Item 15 — a non-punitive warning must never set escalation_
    required/reason, and its own card carries no escalation action."""
    state = {
        "call": call(),
        "doctor_validation": _warning_only_doctor_validation(),
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert result["parsed_data"]["escalation_required"] is False
    warning = result["parsed_data"]["doctor_warnings"][0]
    assert "action" not in warning
    assert "escalate" not in warning.get("description", "").lower()


def test_genuine_doctor_failure_produces_detailed_c2b_flag():
    """A real field-level failure must produce a DETAILED C2B flag built
    from failure_details (Agent-stated value, CRM value, reason, and
    local transcript evidence) — not just the old generic sentence as the
    only explanation."""
    doctor_validation = {
        "outcome": "FAIL", "reason": "1 of 1 recommended doctor(s) failed validation: اميره بركات.",
        "is_violation": True, "result_shape": "multi_doctor",
        "doctors": [{
            "input_name": "اميره بركات", "doctor_resolved": True, "doctor_key": "11011216",
            "doctor_name_ar": "أميرة بركات", "doctor_name_en": "Amira Barakat",
            "outcome": "FAIL", "is_violation": True,
            "failure_details": [{
                "field": "degree", "label": "Degree/title mismatch",
                "chat_value": "اخصاييه", "crm_value": "Senior Registrar",
                "reason": "The professional degree stated by the agent does not match the authoritative CRM record.",
                "transcript_excerpt": "Patient: استشاري؟\nAgent: اخصاييه",
            }],
            "warning_details": [],
        }],
    }
    state = {
        "call": call(),
        "doctor_validation": doctor_validation,
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    flags = result["parsed_data"]["compliance_flags"]
    c2b = [f for f in flags if f["type"] == "C2B"]
    assert len(c2b) == 1
    assert "أميرة بركات" in c2b[0]["description"]
    assert "اخصاييه" in c2b[0]["description"]
    assert "Senior Registrar" in c2b[0]["description"]
    assert not result["parsed_data"]["doctor_warnings"]


def test_doctor_validation_persists_failure_and_warning_details():
    """Item 23 — the saved/serialized doctor_validation dict must retain
    both failure_details and warning_details on every per-doctor entry
    (loosely typed as Optional[dict[str, Any]] on QAAnalysisResult, so
    any additive key is carried through unchanged)."""
    doctor_validation = {
        "outcome": "FAIL", "reason": "reason", "is_violation": True,
        "doctors": [{
            "input_name": "اميره بركات", "outcome": "FAIL",
            "failure_details": [{"field": "degree"}],
            "warning_details": [],
        }],
    }
    state = {
        "call": call(),
        "doctor_validation": doctor_validation,
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    persisted = result["parsed_data"]["doctor_validation"]
    assert persisted["doctors"][0]["failure_details"] == [{"field": "degree"}]
    assert persisted["doctors"][0]["warning_details"] == []
