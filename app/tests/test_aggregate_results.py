"""Tests for app.agent.nodes.aggregate_results' flag-merging/normalisation
logic — specifically the "positive" type/severity mix-up regression.

Flags are violations only: a positive/no-violation finding from any source
is dropped during aggregation, and every remaining flag is critical.

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


def test_aggregate_result_exposes_detected_business_unit():
    state = {
        "call": call().model_copy(update={"business_unit": "SNB"}),
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert result["parsed_data"]["business_unit"] == "SNB"
    assert result["result"].business_unit == "SNB"


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
    assert result["parsed_data"]["compliance_flags"] == []


def test_positive_type_from_compliance_reservation_script_scoring_all_dropped():
    """Every focused-evaluation flag list can independently carry a
    'positive'-typed flag — none of them may reach the final flags."""
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
    assert result["parsed_data"]["compliance_flags"] == []


def test_positive_type_from_offer_flags_is_dropped():
    """A suitable offer is a pass — the offer node's 'positive' flag must
    not crash aggregation and must not be reported."""
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
    assert result["parsed_data"]["compliance_flags"] == []


def test_valid_flag_types_are_unaffected_by_normalization():
    state = {
        "call": call(),
        "compliance_eval": {"compliance_flags": [{
            "type": "C2Com", "severity": "critical",
            "attribute": "Misleading the Privacy Policy",
            "comment": "Released customer personal data",
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


def test_positive_severity_dropped_and_other_severities_become_critical():
    state = {
        "call": call(),
        "compliance_eval": {"compliance_flags": [
            {"type": "NC", "severity": "positive", "description": "good", "transcript_excerpt": "a"},
            {"type": "C2B", "severity": "moderate",
             "attribute": "Intentionally waste Call time", "comment": "Used Hold with no need/Reason",
             "description": "d", "transcript_excerpt": "b"},
            {"type": "NC", "severity": "minor",
             "attribute": "Voice tone", "comment": "Unclear Voice Tone",
             "description": "d", "transcript_excerpt": "c"},
        ]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    flags = result["parsed_data"]["compliance_flags"]
    assert [(f["type"], f["severity"]) for f in flags] == [("C2B", "critical"), ("NC", "critical")]


def test_flag_without_severity_is_stored_as_critical():
    """The prompts only ask for the flag type — a missing severity must
    still validate and be stored as critical."""
    state = {
        "call": call(),
        "compliance_eval": {"compliance_flags": [
            {"type": "NC", "attribute": "Voice tone", "comment": "Unclear Voice Tone",
             "description": "d", "transcript_excerpt": "e"},
        ]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }
    result = asyncio.run(aggregate_results(state))
    assert "error" not in result
    flags = result["parsed_data"]["compliance_flags"]
    assert [(f["type"], f["severity"]) for f in flags] == [("NC", "critical")]


def test_same_policy_item_from_behavioral_and_compliance_is_one_flag():
    chat = CallTranscript(
        call_id="chat-missing-name", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-11-14", call_duration_seconds=120, department="Approval",
        transcript=("Patient: مرحبا\nAgent: ممكن رقم الهويه ؟\n"
                    "Patient: 1234567890\nAgent: تم تأكيد الحجز باسم عادل"),
    )
    state = {
        "call": chat,
        "behavioral_eval": {"behavioral_flags": [{
            "type": "NC", "attribute": "Professional Personalization",
            "comment": "Didn’t ask for customer name",
            "description": "Asked for ID but never asked for the patient's name.",
            "transcript_excerpt": "ممكن رقم الهويه ؟",
        }]},
        "compliance_eval": {"compliance_flags": [{
            "type": "NC", "attribute": "Professional Personalization",
            "comment": "Didn’t ask for customer name",
            "description": "NC_001: No agent requested the patient's name.",
            "transcript_excerpt": "تم تأكيد الحجز باسم عادل",
        }]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }

    result = asyncio.run(aggregate_results(state))["result"]

    assert len(result.compliance_flags) == 1
    assert result.compliance_flags[0].comment == "Didn’t ask for customer name"
    assert result.compliance_flags[0].transcript_excerpt == "ممكن رقم الهويه ؟"
    assert result.agent_performance.agent_classification == "B"
    assert result.overall_assessment == "needs_review"


def test_distinct_policy_items_with_same_evidence_remain_separate():
    state = {
        "call": call(),
        "behavioral_eval": {"behavioral_flags": [{
            "type": "NC", "attribute": "Professional Personalization",
            "comment": "Didn’t ask for customer name",
            "description": "NC_001: Name was not requested.",
            "transcript_excerpt": "Agent: hello",
        }]},
        "compliance_eval": {"compliance_flags": [{
            "type": "NC", "attribute": "Voice tone", "comment": "Unclear Voice Tone",
            "description": "The response was difficult to hear.",
            "transcript_excerpt": "Agent: hello",
        }]},
        "scoring_eval": base_scoring_eval(),
        "node_trace": [],
    }

    result = asyncio.run(aggregate_results(state))["result"]

    assert {(flag.attribute, flag.comment) for flag in result.compliance_flags} == {
        ("Professional Personalization", "Didn’t ask for customer name"),
        ("Voice tone", "Unclear Voice Tone"),
    }


def test_scoring_llm_context_never_contains_severity():
    from app.agent.nodes import _llm_context_json
    payload = {"crm_leads_flags": [{"type": "C2B", "severity": "critical", "description": "d"}],
               "nested": {"severity": "x", "items": [{"severity": "y", "keep": 1}]}}
    out = _llm_context_json(payload)
    assert "severity" not in out
    assert '"keep": 1' in out
