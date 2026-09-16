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
