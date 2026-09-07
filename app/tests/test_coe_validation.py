"""Tests for COE (Center of Excellence) validation:
  - app.service_hub.coe_validation — deterministic trigger detection,
    complaint mapping, and authoritative primary-doctor matching.
  - app.agent.nodes.infer_coe_validation / skip_coe_validation — the graph
    node wrappers (deterministic gate + LLM semantic extraction +
    deterministic safety nets), unit-tested here with a stubbed LLM client
    exactly like test_doctor_scope_validation.py's pattern, so the gating/
    safety-net logic this module owns is verified deterministically.
  - app.agent.graph._coe_intent_router / graph wiring.

Mocks the LLM and never touches the production CRM database — see
crm_coe.fetch_coe_reference (monkeypatched or left unconfigured, which
degrades safely to DEFAULT_SCRIPTS_AR).
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models.input import CallTranscript
from app.service_hub.coe_validation import (
    AUTHORITATIVE_PRIMARY_DOCTORS,
    COE_SPECIALTIES,
    DEFAULT_SCRIPTS_AR,
    PRIMARY_DOCTOR_ALIASES,
    SPECIALTY_ALIASES,
    SPECIALTY_TO_COES,
    WEAK_SPECIALTY_ALIASES,
    _agent_turn_supports_category,
    ambiguous_specialty_mentions,
    build_coe_contexts,
    build_coe_evaluations,
    build_coe_reference,
    campaign_origin_evidence,
    classify_coe_trigger,
    clean_extracted_doctor_name,
    coe_validation_needed,
    detect_specialty_mentions,
    detect_weak_specialty_mentions,
    evaluate_context_doctors,
    existing_patient_exception_evidence,
    extract_doctor_context_associations,
    ground_llm_coe_value,
    is_plausible_coe_doctor_candidate,
    match_primary_doctor,
    normalize_doctor_name_for_match,
    resolve_campaign_coe,
    resolve_canonical_specialty,
    resolve_primary_complaint,
    resolve_primary_doctor_identity,
    resolve_recommended_coe,
    resolve_specialty_coes,
    unassociated_initial_doctors,
)
from app.agent.nodes import infer_coe_validation, skip_coe_validation
from app.agent.graph import _coe_intent_router, build_qa_graph


def call(transcript: str, call_id: str = "coe-test") -> CallTranscript:
    return CallTranscript(
        call_id=call_id, agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-08-30", call_duration_seconds=1, department="Scheduling",
        transcript=transcript,
    )


class _StubLLM:
    """Returns a fixed JSON response regardless of the prompt — used to
    test infer_coe_validation's own gating/safety-net logic in isolation
    from real LLM semantic reasoning (mirrors test_doctor_scope_validation.
    py's _StubLLM)."""

    def __init__(self, response: dict | None = None):
        self._response = response if response is not None else {}
        self.called = False
        self.last_user_prompt: str | None = None

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.called = True
        self.last_user_prompt = user_prompt
        return json.dumps(self._response, ensure_ascii=False), {"prompt_tokens": 0, "completion_tokens": 0}


def run_coe_node(transcript: str, llm_response: dict | None = None, call_id: str = "coe-test") -> tuple[dict, _StubLLM]:
    c = call(transcript, call_id)
    stub = _StubLLM(llm_response)
    state = {"call": c, "node_trace": []}
    result = asyncio.run(infer_coe_validation(state, stub))
    return result["coe_validation"], stub


# ═════════════════════════════════════════════════════════════════════════
# Trigger detection (pure, deterministic — classify_coe_trigger)
# ═════════════════════════════════════════════════════════════════════════

def test_no_coe_discussion_not_triggered():
    """Item 13 — an eligible complaint with no COE/specialized-center
    discussion at all must never trigger this validator."""
    c = call("Patient: عندي صداع مستمر من كذا يوم\nAgent: تمام هحجزلك عيادة مخ واعصاب")
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False
    assert coe_validation_needed(c, ctx) is False


def test_coe_doctor_mentioned_in_ordinary_booking_not_triggered():
    """Item 14 — a COE doctor mentioned only during an ordinary doctor
    booking (no COE/specialized-center language at all) must not trigger."""
    c = call("Patient: عايز احجز مع Dr. Dalinda بكرة\nAgent: تمام هحجزلك بكرة الساعة 5")
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False


def test_generic_center_word_alone_not_triggered():
    """A bare 'مركز'/branch mention without COE-specific language must
    never trigger this validator."""
    c = call("Patient: فين اقرب مركز لكم؟\nAgent: مركز أندلسية في جدة")
    assert classify_coe_trigger(c)["triggered"] is False


def test_customer_inquiry_specialized_center_triggers_path_b():
    """Item 15 — the customer asks about 'مركز متخصص' and the agent
    responds/confirms -> triggered via customer_inquiry."""
    c = call(
        "Patient: في عندكم مركز متخصص للصداع؟\n"
        "Agent: أيوه، هيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع"
    )
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is True
    assert ctx["trigger_path"] == "customer_inquiry"


def test_customer_mentions_center_agent_never_responds_not_triggered():
    """Item 16 — the customer mentions a specialized center, but the agent
    never responds to or confirms it -> no recommendation is attributed to
    the agent, and the node is not triggered."""
    c = call(
        "Patient: سمعت إن في مركز متخصص للسكر\n"
        "Agent: تمام، حابب تحجز كشف عادي امتى؟"
    )
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False
    assert "customer" in ctx["trigger_reason"].lower() or "specialized" in ctx["trigger_reason"].lower()


def test_proactive_agent_recommendation_triggers_path_a():
    """The agent proactively introduces a COE without the customer asking
    first -> triggered via proactive_recommendation."""
    c = call(
        "Patient: عندي الم في بطني من فتره وحابب اعرف مواعيد متاحه\n"
        "Agent: لضمان تحقيق أقصى استفادة، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الجهاز الهضمي"
    )
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is True
    assert ctx["trigger_path"] == "proactive_recommendation"


def test_faithful_paraphrase_of_official_script_triggers():
    """Item 17 — a faithful paraphrase (not the exact script) of the
    approved Arabic script still triggers and is still correctly matched to
    its COE by resolve_recommended_coe."""
    c = call(
        "Patient: عندي كحه وضيق في التنفس من مده\n"
        "Agent: عشان تاخد افضل تشخيص لحالتك هنحجزلك في مركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي"
    )
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is True
    assert resolve_recommended_coe(c) == "Asthma"


def test_speaker_attribution_customer_statement_not_agent_recommendation():
    """Item 28 — a customer statement must never be interpreted as if the
    HUMAN AGENT made the recommendation. Here only the Patient ever
    mentions the COE; the Agent's only turn is unrelated, so trigger must
    stay False and resolve_recommended_coe must not attribute anything to
    the agent."""
    transcript = (
        "Patient: انا حابب احجز في مركز التميز المتخصص في علاج امراض السكر والغدد الصماء\n"
        "Agent: تمام، عايز تحجز كشف عادي امتى؟"
    )
    c = call(transcript)
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False
    assert resolve_recommended_coe(c) is None


# ═════════════════════════════════════════════════════════════════════════
# Campaign/post-origin trigger (Path C, "campaign_origin")
#
# A conversation may begin with an automatically populated Patient message
# from clicking a COE ad/post — a STRUCTURED campaign identifier containing
# "COE" (e.g. "BU-AHJ-COE-...") is stronger evidence than a bare "مركز
# تميز" mention, and TRIGGERS the check immediately even when the agent
# never repeats COE language afterward.
# ═════════════════════════════════════════════════════════════════════════

BU_AHJ_COE_HEADACHE_TRANSCRIPT = (
    "Patient: BU-AHJ-COE- أضغطي علي أرسال للأستفادة بعروضنا في مركز تميز الصداع\n"
    "Patient: انا عندي صداع مزمن عايزه احجز اون لاين عند اخصائي الألم\n"
    "Agent: بيكون كشفية اون لاين مع دكتور محمود الحوراني بعيادة المخ والاعصاب"
)


def test_campaign_origin_marker_triggers_even_without_agent_repeating_coe_language():
    """Item 1 — the exact BU-AHJ-COE headache campaign scenario: triggered
    via campaign_origin even though the agent's reply never says 'مركز
    تميز' or otherwise mentions a COE."""
    c = call(BU_AHJ_COE_HEADACHE_TRANSCRIPT)
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is True
    assert ctx["trigger_path"] == "campaign_origin"
    assert "BU-AHJ-COE" in ctx["evidence"]
    # The primary complaint still resolves correctly from the patient's own
    # words (both the campaign line's "الصداع" and the follow-up complaint
    # line agree on Headache).
    primary, categories = resolve_primary_complaint(c)
    assert primary == "Headache"


def test_campaign_origin_evidence_helper_matches_only_the_structured_identifier():
    assert campaign_origin_evidence(call(BU_AHJ_COE_HEADACHE_TRANSCRIPT)) is not None
    # A bare "مركز تميز" mention alone — no structured campaign code — must
    # NOT be picked up by the campaign-origin path (see item 3 below); it
    # stays governed by the existing proactive/customer_inquiry paths.
    plain = call("Patient: عايز احجز في مركز تميز للسكر\nAgent: تمام هحجزلك")
    assert campaign_origin_evidence(plain) is None


def test_campaign_origin_followed_by_agent_discussing_clinic_without_coe_wording():
    """Item 2 — campaign-origin followed by an agent response that
    discusses the matching complaint/clinic (booking into the Neurology/
    'عيادة المخ والاعصاب' clinic that IS the Headache COE's first clinic)
    without ever saying 'مركز تميز' — still triggered, and the established
    COE context is still usable for the doctor check."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي أرسال للأستفادة بعروضنا في مركز تميز الصداع\n"
        "Patient: بعاني من صداع نصفي متكرر\n"
        "Agent: تمام، هحجزلك في عيادة المخ والاعصاب مع دكتور عمر أيوب"
    )
    result, stub = run_coe_node(transcript)
    assert result["triggered"] is True
    assert result["trigger_path"] == "campaign_origin"
    assert result["expected_coe"] == "Headache"


def test_ordinary_customer_only_coe_mention_still_does_not_trigger():
    """Item 3 — an ordinary Patient-only 'مركز متخصص' mention with an
    unrelated normal-booking Agent reply must remain untriggered; this
    must NOT be reclassified as campaign_origin."""
    c = call("Patient: سمعت إن في مركز متخصص للسكر\nAgent: تمام، حابب تحجز كشف عادي امتى؟")
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False
    assert ctx["trigger_path"] is None


def test_complaint_without_campaign_marker_or_center_discussion_does_not_trigger():
    """Item 4 — a plain medical complaint with no campaign marker and no
    specialized-center discussion at all must not trigger."""
    c = call("Patient: عندي صداع بسيط من ساعتين\nAgent: تمام هحجزلك كشف عادي بكرة")
    ctx = classify_coe_trigger(c)
    assert ctx["triggered"] is False
    assert campaign_origin_evidence(c) is None


def test_campaign_text_never_attributed_to_the_agent():
    """Item 5 — the campaign message is Patient-side marketing/system
    context; resolve_recommended_coe (Agent-only) must never pick it up,
    and the trigger_reason must explicitly say it is not agent wording."""
    c = call(BU_AHJ_COE_HEADACHE_TRANSCRIPT)
    ctx = classify_coe_trigger(c)
    assert "not something the human agent wrote" in ctx["trigger_reason"] or "human agent" in ctx["trigger_reason"]
    # The agent's actual turn ("بيكون كشفية اون لاين مع دكتور محمود
    # الحوراني...") contains no explicit COE-name marker and is not
    # script-similar enough on its own — resolve_recommended_coe must not
    # fabricate a match from the campaign (Patient-side) text.
    assert resolve_recommended_coe(c) is None

    # Node-level: the campaign excerpt appears in evidence (as context),
    # never disguised as something the agent said.
    result, stub = run_coe_node(BU_AHJ_COE_HEADACHE_TRANSCRIPT)
    assert any("BU-AHJ-COE" in ev for ev in result["evidence"])


def test_campaign_origin_node_extracts_agent_offered_doctor_and_evaluates_eligibility():
    """Item 7 (node-level) — for the supplied regression conversation: the
    node runs (never skipped), 'محمود الحوراني' is extracted as the
    initial doctor via the deterministic transcript extraction, and
    primary-doctor eligibility is evaluated against the authoritative
    Headache list (he is not on it, so this must not be a false pass)."""
    result, stub = run_coe_node(
        BU_AHJ_COE_HEADACHE_TRANSCRIPT,
        {
            "primary_complaint_category": "Headache",
            "recommended_coe": "Headache",
            "initial_doctors": ["محمود الحوراني"],
        },
    )
    assert result["applicable"] is True
    assert result["triggered"] is True
    assert result["trigger_path"] == "campaign_origin"
    assert result["expected_coe"] == "Headache"
    assert result["booking_discussed"] is True
    joined = " ".join(result["recommended_or_selected_doctors"])
    assert "حوراني" in joined
    # Not one of the four authoritative Headache doctors -> correctly not
    # matched as an approved primary doctor.
    assert result["primary_doctor_status"] == "fail"
    # Never falsely credited as a match — the approved names may still
    # appear in the reason as the (unmet) approved-list context, but never
    # as matched_primary_doctors.
    assert result["matched_primary_doctors"] == []


def test_campaign_message_not_presented_as_agent_quotation_in_reason():
    """Item 7 — the campaign message must not be presented as an agent
    quotation anywhere in the human-readable reason/coe_reason text."""
    result, stub = run_coe_node(
        BU_AHJ_COE_HEADACHE_TRANSCRIPT,
        {"primary_complaint_category": "Headache", "recommended_coe": "Headache"},
    )
    assert "BU-AHJ-COE" not in result["reason"]


# ═════════════════════════════════════════════════════════════════════════
# campaign_coe / recommended_coe / validation_coe separation
#
# Regression coverage for a campaign-origin conversation where the agent
# books a doctor within the already-established COE context WITHOUT
# repeating COE language — expected_coe, campaign_coe, recommended_coe,
# and validation_coe must never be conflated, an ungrounded LLM
# recommended_coe (e.g. a hallucinated "IBD") must be rejected, and the
# doctor name must be extracted cleanly (no trailing clinic phrase).
# ═════════════════════════════════════════════════════════════════════════

MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT = (
    "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
    "Patient: انا عندي صداع مزمن عايزه احجز اون لاين عند اخصائي الألم\n"
    "Agent: بيكون كشفية اون لاين مع دكتور محمود الحوراني بعيادة المخ والاعصاب"
)


def test_full_regression_conversation_produces_expected_result():
    """Item 1 — the complete supplied conversation, with a stub LLM
    reproducing the reported bug (hallucinated recommended_coe='IBD'),
    must produce the exact corrected result."""
    result, stub = run_coe_node(
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT,
        {
            "primary_complaint_category": "Headache",
            "recommended_coe": "IBD",
            "initial_doctors": ["محمود الحوراني"],
        },
    )
    assert result["triggered"] is True
    assert result["trigger_path"] == "campaign_origin"
    assert result["expected_coe"] == "Headache"
    assert result["campaign_coe"] == "Headache"
    assert result["recommended_coe"] is None
    assert result["validation_coe"] == "Headache"
    assert result["recommended_or_selected_doctors"] == ["محمود الحوراني"]
    assert result["matched_primary_doctors"] == []
    assert result["primary_doctor_status"] == "fail"
    assert result["is_violation"] is True
    # coe_match_status truthfully reflects that the COE came from the
    # campaign, not from an explicit agent recommendation.
    assert result["coe_match_status"] in ("not_applicable", "uncertain")
    assert "IBD" not in result["reason"]


def test_campaign_phrase_resolves_deterministically_to_headache():
    """Item 2 — "مركز تميز الصداع" resolves deterministically to
    campaign_coe = Headache."""
    c = call(
        "Patient: BU-AHJ-COE- اضغط هنا مركز تميز الصداع\n"
        "Agent: تمام"
    )
    assert resolve_campaign_coe(c) == "Headache"


@pytest.mark.parametrize("neurology_phrase", [
    "مخ واعصاب", "المخ والاعصاب", "مخ وأعصاب", "مخ و اعصاب", "اعصاب", "أعصاب",
])
def test_neurology_language_supports_headache_in_campaign_context(neurology_phrase):
    """Item 3 — neurology expressions and their spelling variants support
    Headache when used as campaign/COE context."""
    c = call(f"Patient: BU-AHJ-COE- استفسار عن عيادة {neurology_phrase}\nAgent: تمام")
    assert resolve_campaign_coe(c) == "Headache"


def test_unsupported_llm_recommended_coe_is_rejected():
    """Item 4 — an LLM response containing unsupported recommended_coe=IBD
    (no IBD/gastrointestinal evidence anywhere in an Agent turn) is
    rejected, never producing a false mismatch."""
    c = call(MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT)
    assert ground_llm_coe_value(c, "IBD") is None
    assert _agent_turn_supports_category(c, "IBD") is False


def test_reference_data_alone_cannot_ground_an_llm_selected_coe():
    """Item 5 — reference data shown to the LLM lists all four COEs; an
    LLM claiming a category with NO real transcript evidence (here:
    Diabetes, never mentioned anywhere) must be discarded regardless of
    what the reference section contained."""
    result, stub = run_coe_node(
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT,
        {"primary_complaint_category": "Headache", "recommended_coe": "Diabetes"},
    )
    assert result["recommended_coe"] is None
    assert result["coe_match_status"] != "fail"


def test_doctor_name_extraction_stops_before_clinic_connector():
    """Item 6 — "دكتور محمود الحوراني بعيادة المخ والاعصاب" extracts
    exactly "محمود الحوراني", never including "بعيادة ..."."""
    assert clean_extracted_doctor_name("محمود الحوراني بعياده المخ") == "محمود الحوراني"
    assert clean_extracted_doctor_name("محمود الحوراني بعيادة المخ والاعصاب") == "محمود الحوراني"
    # Idempotent on an already-clean name.
    assert clean_extracted_doctor_name("محمود الحوراني") == "محمود الحوراني"
    # A genuine multi-token compound name with no clinic connector must
    # never be truncated.
    assert clean_extracted_doctor_name("عبدالرحمن الشهري") == "عبدالرحمن الشهري"

    result, stub = run_coe_node(
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT,
        {"primary_complaint_category": "Headache", "initial_doctors": ["محمود الحوراني"]},
    )
    assert result["recommended_or_selected_doctors"] == ["محمود الحوراني"]
    assert not any("بعياده" in d or "بعيادة" in d for d in result["recommended_or_selected_doctors"])


def test_mahmoud_elhorany_fails_headache_primary_doctor_check():
    """Item 7 — Mahmoud El Horany (any common spelling) is NOT an approved
    Headache primary doctor."""
    assert resolve_primary_doctor_identity("محمود الحوراني", "Headache") is None
    assert match_primary_doctor("محمود الحوراني", "Headache") is False


def test_mahmoud_elhorany_not_added_to_authoritative_list():
    """Item 8 (part) — the authoritative Headache primary-doctor list is
    unchanged; Mahmoud El Horany must never appear in it."""
    headache_list = AUTHORITATIVE_PRIMARY_DOCTORS["Headache"]
    assert headache_list == ["Osama Abdel Salam", "Abdelrhman Alshehri", "Omar Ayoub", "Abdulrahman Bogus"]
    assert not any("حوراني" in d or "horany" in d.lower() for d in headache_list)


@pytest.mark.parametrize("doctor_name", [
    "Osama Abdel Salam", "Abdelrhman Alshehri", "Omar Ayoub", "Abdulrahman Bogus",
])
def test_approved_headache_primary_doctors_still_pass_in_campaign_context(doctor_name):
    """Item 8 — approved Headache primary doctors still pass, including
    within a campaign-origin conversation."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        f"Agent: يبدأ الحجز أولاً في عيادة المخ والاعصاب مع دكتور {doctor_name}"
    )
    result, stub = run_coe_node(transcript, {
        "primary_complaint_category": "Headache",
        "initial_doctors": [doctor_name],
    })
    assert result["validation_coe"] == "Headache"
    assert result["primary_doctor_status"] == "pass"
    assert doctor_name in result["matched_primary_doctors"]
    assert result["is_violation"] is False


def test_existing_proactive_and_customer_inquiry_routing_unchanged():
    """Item 9 — the pre-existing proactive_recommendation and
    customer_inquiry paths still work exactly as before: recommended_coe
    is set directly from the agent's own explicit COE language, and
    validation_coe/coe_match_status behave as a plain match/mismatch."""
    proactive_transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Omar Ayoub"
    )
    result, stub = run_coe_node(proactive_transcript)
    assert result["trigger_path"] == "proactive_recommendation"
    assert result["campaign_coe"] is None
    assert result["recommended_coe"] == "Headache"
    assert result["validation_coe"] == "Headache"
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"

    inquiry_transcript = (
        "Patient: في عندكم مركز متخصص للسكر؟\n"
        "Patient: عندي مرض السكر وعايز اتابع حالتي\n"
        "Agent: أيوه، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض السكر والغدد الصماء مع Dr. Badri Bairuti"
    )
    result2, stub2 = run_coe_node(inquiry_transcript)
    assert result2["trigger_path"] == "customer_inquiry"
    assert result2["campaign_coe"] is None
    assert result2["recommended_coe"] == "Diabetes"
    assert result2["validation_coe"] == "Diabetes"
    assert result2["coe_match_status"] == "pass"
    assert result2["primary_doctor_status"] == "pass"


def test_campaign_text_never_attributed_to_human_agent_in_regression():
    """Item 10 — the campaign text is never attributed to the human agent:
    resolve_recommended_coe (Agent-only) ignores it, and the final
    recommended_coe never equals a value ONLY the campaign text supports
    when no actual Agent turn corroborates it."""
    c = call(MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT)
    assert resolve_recommended_coe(c) is None
    result, stub = run_coe_node(
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT,
        {"primary_complaint_category": "Headache", "recommended_coe": "IBD"},
    )
    assert result["recommended_coe"] is None
    assert "BU-AHJ-COE" not in result["reason"]
    assert "BU-AHJ-COE" not in (result.get("coe_match_status") or "")


# ═════════════════════════════════════════════════════════════════════════
# Multi-context COE evaluation — independent per-COE contexts, each with
# its own independently-validated doctor(s), never collapsed into one
# scalar and never letting one approved doctor stand in for another
# unapproved one discussed under a DIFFERENT (or even the same) COE.
# ═════════════════════════════════════════════════════════════════════════

def test_one_coe_one_approved_doctor():
    """Item 1."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Omar Ayoub"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    assert evaluations[0]["coe"] == "Headache"
    assert evaluations[0]["primary_doctor_status"] == "pass"
    assert evaluations[0]["is_violation"] is False


def test_one_coe_one_unapproved_doctor():
    """Item 2."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Samir Youssef"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    assert evaluations[0]["primary_doctor_status"] == "fail"
    assert evaluations[0]["is_violation"] is True


def test_one_coe_two_approved_doctors():
    """Item 3."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع، "
        "المتاح دكتور أسامة عبد السلام أو دكتور عمر أيوب"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    ctx = evaluations[0]
    assert ctx["primary_doctor_status"] == "pass"
    statuses = {d["primary_doctor_status"] for d in ctx["doctors"]}
    assert statuses == {"pass"}
    assert {d["canonical_name"] for d in ctx["doctors"]} == {"Osama Abdel Salam", "Omar Ayoub"}


def test_one_coe_mixed_approved_and_unapproved_initial_doctors_fails():
    """Item 4 — one COE with one approved and one unapproved initial
    doctor offered as alternatives: the context must FAIL."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: المتاح لحضرتك دكتور أسامة عبد السلام أو دكتور محمود الحوراني بعيادة المخ والاعصاب"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    ctx = evaluations[0]
    assert ctx["coe"] == "Headache"
    assert ctx["primary_doctor_status"] == "fail"
    assert ctx["is_violation"] is True
    statuses = {(d["extracted_name"], d["primary_doctor_status"]) for d in ctx["doctors"]}
    assert ("محمود الحوراني", "fail") in statuses
    assert any(status == "pass" for _name, status in statuses)


def test_headache_and_ibd_both_discussed_one_doctor_each():
    """Item 5 — Headache and IBD both explicitly discussed, each with its
    own associated doctor, evaluated as two independent contexts.

    "داليندا عرفاوي" is an explicitly-requested alias of the existing
    canonical IBD doctor "Dalinda" (surname attached in some transcripts,
    never a second/unapproved doctor — see PRIMARY_DOCTOR_ALIASES), so the
    IBD context correctly PASSES here rather than failing."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
        "Patient: انا عندي صداع مزمن\n"
        "Agent: المتاح دكتور اسامه عبدالسلام بعيادة المخ والاعصاب\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الجهاز الهضمي مع دكتور داليندا عرفاوي\n"
    )
    evaluations = build_coe_evaluations(call(transcript))
    by_coe = {e["coe"]: e for e in evaluations}
    assert set(by_coe) == {"Headache", "IBD"}
    assert by_coe["Headache"]["primary_doctor_status"] == "pass"
    assert by_coe["Headache"]["doctors"][0]["canonical_name"] == "Osama Abdel Salam"
    assert by_coe["IBD"]["primary_doctor_status"] == "pass"
    assert by_coe["IBD"]["doctors"][0]["extracted_name"] == "داليندا عرفاوي"
    assert by_coe["IBD"]["doctors"][0]["canonical_name"] == "Dalinda"


def test_doctors_correctly_associated_through_nearby_turn_evidence():
    """Item 6 — a specialty mentioned in one turn and the doctor offered in
    the very next turn are still correctly associated via 'nearby turn'
    evidence."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: هيبدأ الحجز في عيادة المخ والاعصاب\n"
        "Agent: المتاح لحضرتك دكتور عمر أيوب"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    ctx = evaluations[0]
    assert ctx["coe"] == "Headache"
    assert ctx["primary_doctor_status"] == "pass"
    assert ctx["doctors"][0]["canonical_name"] == "Omar Ayoub"


def test_doctor_not_cross_validated_against_unrelated_coe():
    """Item 7 — a doctor approved for Headache must not pass when the
    grounded evidence connects him to a DIFFERENT COE (here IBD)."""
    transcript = (
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الجهاز الهضمي مع دكتور عمر أيوب"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    ctx = evaluations[0]
    assert ctx["coe"] == "IBD"
    assert normalize_doctor_name_for_match(ctx["doctors"][0]["extracted_name"]) == normalize_doctor_name_for_match("عمر أيوب")
    assert ctx["doctors"][0]["canonical_name"] is None
    assert ctx["primary_doctor_status"] == "fail"
    # Confirm independently: he WOULD be approved for Headache — just not
    # for the COE actually grounded by this turn's evidence (IBD).
    assert match_primary_doctor("عمر أيوب", "Headache") is True
    assert match_primary_doctor("عمر أيوب", "IBD") is False


def test_unsupported_llm_ibd_context_is_discarded():
    """Item 8 — an LLM-produced recommended_coe='IBD' with zero IBD
    evidence anywhere in the transcript must never surface as an IBD
    context; the deterministic multi-context builder never even consults
    LLM output, so a hallucinated category simply never appears."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Omar Ayoub"
    )
    result, stub = run_coe_node(transcript, {"recommended_coe": "IBD"})
    assert all(e["coe"] != "IBD" for e in result["coe_evaluations"])
    assert ground_llm_coe_value(call(transcript), "IBD") is None


def test_crm_reference_data_alone_cannot_create_a_coe_context():
    """Item 9 — build_coe_contexts/build_coe_evaluations never consult CRM
    reference data or an LLM prompt at all; a transcript with genuinely no
    textual evidence for any COE produces zero contexts, regardless of
    what reference scripts exist."""
    transcript = "Patient: عايز اعرف اسعار الكشف العادي\nAgent: الكشف العادي بـ 300 ريال"
    contexts = build_coe_contexts(call(transcript), scripts=DEFAULT_SCRIPTS_AR)
    assert contexts == {}
    assert build_coe_evaluations(call(transcript)) == []


def test_campaign_headache_context_distinct_from_agent_ibd_recommendation():
    """Item 10 — a campaign-established Headache context remains distinct
    from a separate, explicit Agent IBD recommendation elsewhere in the
    same call (the exact regression scenario, generalised)."""
    result, stub = run_coe_node(
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT,
        {"primary_complaint_category": "Headache", "recommended_coe": "IBD"},
    )
    by_coe = {e["coe"]: e for e in result["coe_evaluations"]}
    assert "Headache" in by_coe
    assert "campaign" in by_coe["Headache"]["context_sources"]
    assert by_coe["Headache"]["campaign_evidence"] is not None


def test_referral_only_doctors_do_not_affect_primary_doctor_validation():
    """Item 11 — a referral-only doctor mention must not, on its own,
    produce a fail or pass — it is excluded from the aggregation."""
    transcript = (
        "Patient: عندي ربو وضيق شديد في التنفس\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي مع Dr. Eid Elajmi، "
        "وممكن بعد التقييم الأولي دكتور الانف والاذن والحنجرة يتابع معاك لو احتجت"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    ctx = evaluations[0]
    assert ctx["primary_doctor_status"] == "pass"
    assert ctx["doctors"][0]["role"] == "initial_primary"
    assert ctx["doctors"][0]["canonical_name"] == "Eid Elajmi"


def test_ambiguous_doctor_to_coe_association_is_uncertain_not_guessed():
    """Item 12 — a doctor offered with NO same-turn or nearby COE evidence,
    while two DIFFERENT COEs are simultaneously active, must never be
    guessed into either one — it is reported as an unassociated/uncertain
    doctor instead."""
    transcript = (
        "Patient: عندي صداع مزمن\n"
        "Patient: وعندي كمان مرض السكر\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض السكر والغدد الصماء\n"
        "Patient: تمام\n"
        "Agent: تمام هحجزلك مع دكتور فلان الفلاني"
    )
    c = call(transcript)
    associations = extract_doctor_context_associations(c)
    matching = [a for a in associations if "فلان" in a["extracted_name"]]
    assert matching, "expected the doctor mention to be extracted at all"
    assert matching[0]["associated_coes"] == []
    assert matching[0]["association_certainty"] == "unclear"
    unassociated = unassociated_initial_doctors(c)
    assert any("فلان" in a["extracted_name"] for a in unassociated)


def test_passing_context_does_not_hide_a_failing_context():
    """Item 13 — a passing Headache context must not hide a failing IBD
    context in the aggregate result.

    Uses a genuinely unapproved IBD doctor name — "داليندا عرفاوي" is now
    a recognised alias of the approved IBD doctor "Dalinda" (see
    PRIMARY_DOCTOR_ALIASES) and would legitimately pass, which is not what
    this test is checking."""
    transcript = (
        MAHMOUD_ELHORANY_REGRESSION_TRANSCRIPT.replace("محمود الحوراني", "عمر أيوب")
        + "\nAgent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الجهاز الهضمي مع دكتور سالم الغامدي"
    )
    result, stub = run_coe_node(transcript)
    by_coe = {e["coe"]: e for e in result["coe_evaluations"]}
    assert by_coe["Headache"]["is_violation"] is False
    assert by_coe["IBD"]["is_violation"] is True
    assert result["is_violation"] is True
    assert result["overall_coe_status"] == "fail"
    assert result["overall_coe_status"] == "fail"


def test_existing_single_coe_behavior_remains_compatible():
    """Item 14 — a genuinely single-COE call keeps its EXACT pre-existing
    scalar-field behaviour, with coe_evaluations containing exactly one
    matching entry."""
    transcript = (
        "Patient: عندي اسهال مزمن وآلام في الجهاز الهضمي من فترة طويلة\n"
        "Agent: لضمان تحقيق أقصى استفادة، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في "
        "علاج أمراض الجهاز الهضمي مع Dr. Dalinda"
    )
    result, stub = run_coe_node(transcript)
    assert result["expected_coe"] == "IBD"
    assert result["recommended_coe"] == "IBD"
    assert result["validation_coe"] == "IBD"
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"
    assert result["is_violation"] is False
    assert len(result["coe_evaluations"]) == 1
    assert result["coe_evaluations"][0]["coe"] == "IBD"
    assert result["coe_evaluations"][0]["primary_doctor_status"] == "pass"
    assert result["overall_coe_status"] == "pass"


# ═════════════════════════════════════════════════════════════════════════
# Primary-complaint -> expected-COE mapping (pure, deterministic)
# ═════════════════════════════════════════════════════════════════════════

def test_single_complaint_resolves_unambiguously():
    c = call("Patient: عندي صداع نصفي شديد من فتره\nAgent: تمام")
    primary, all_cats = resolve_primary_complaint(c)
    assert primary == "Headache"
    assert all_cats == ["Headache"]


def test_multiple_unrelated_complaints_no_primary_returns_uncertain():
    """Item 18 — several complaints with no clear primary complaint ->
    None (caller reports 'uncertain'), never a fabricated failure."""
    c = call(
        "Patient: عندي صداع من ايام وعندي كمان مشاكل في السكر\n"
        "Agent: تمام هوضحلك"
    )
    primary, all_cats = resolve_primary_complaint(c)
    assert primary is None
    assert set(all_cats) == {"Headache", "Diabetes"}


def test_last_single_category_turn_disambiguates_multiple_complaints():
    c = call(
        "Patient: كان عندي صداع الاسبوع اللي فات بس خف\n"
        "Patient: دلوقتي المشكلة الاساسية عندي انها السكر مرتفع جدا وعايز اتابع\n"
        "Agent: تمام هوضحلك"
    )
    primary, all_cats = resolve_primary_complaint(c)
    assert primary == "Diabetes"
    assert set(all_cats) == {"Headache", "Diabetes"}


def test_unsupported_complaint_returns_no_category():
    """Item 29 — an unsupported/unrelated complaint must never be
    force-mapped into one of the four supported COEs."""
    c = call("Patient: عندي الم في الركبة من اصابة رياضية\nAgent: تمام")
    primary, all_cats = resolve_primary_complaint(c)
    assert primary is None
    assert all_cats == []


# ═════════════════════════════════════════════════════════════════════════
# Authoritative primary-doctor matching (pure, deterministic)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,coe", [
    ("Dr. Dalinda", "IBD"),
    ("Dalinda", "IBD"),
    ("Dr. Osama Abdel Salam", "Headache"),
    ("Doctor Osama Abdel Salam", "Headache"),
    ("Dr. Abdelrhman Alshehri", "Headache"),
    ("Dr. Omar Ayoub", "Headache"),
    ("Dr. Abdulrahman Bogus", "Headache"),
    ("Dr. Nagwa Elhalawani", "Asthma"),
    ("Dr. Eid Elajmi", "Asthma"),
    ("Dr. Badri Bairuti", "Diabetes"),
])
def test_match_primary_doctor_confirms_approved_doctors(name, coe):
    assert match_primary_doctor(name, coe) is True


@pytest.mark.parametrize("name,wrong_coe", [
    ("Dr. Dalinda", "Headache"),
    ("Dr. Omar Ayoub", "Asthma"),
    ("Dr. Badri Bairuti", "IBD"),
])
def test_match_primary_doctor_rejects_wrong_coe(name, wrong_coe):
    assert match_primary_doctor(name, wrong_coe) is False


def test_match_primary_doctor_rejects_unapproved_doctor():
    assert match_primary_doctor("Dr. Samir Youssef", "Headache") is False
    assert match_primary_doctor("Dr. Samir Youssef", "Asthma") is False


def test_match_primary_doctor_common_transliteration_variants():
    """Item 24 — safe matching of common spelling/spacing/title/ASR
    variants of the same doctor."""
    variants = [
        "dr osama abdel salam", "DR. OSAMA  ABDEL   SALAM",
        "Doctor Osama Abdel-Salam", "دكتور Osama Abdel Salam",
    ]
    for v in variants:
        assert match_primary_doctor(v, "Headache") is True, v


def test_match_primary_doctor_does_not_force_ambiguous_match():
    """Item 25 — a name that is only vaguely/generically similar to an
    approved doctor must never be force-matched."""
    assert match_primary_doctor("Dr. Ahmed", "Headache") is False
    assert match_primary_doctor("Dr. Sam", "Asthma") is False


def test_normalize_doctor_name_strips_titles_and_case():
    assert normalize_doctor_name_for_match("Dr. Dalinda") == normalize_doctor_name_for_match("dalinda")
    assert normalize_doctor_name_for_match("Doctor Dalinda") == normalize_doctor_name_for_match("DALINDA")


# ═════════════════════════════════════════════════════════════════════════
# Bilingual (Arabic <-> English) primary-doctor identity resolution
#
# Regression coverage for the cross-script matching bug: an Arabic
# extraction (e.g. "اسامه عبد السلام") must resolve to its canonical
# English identity ("Osama Abdel Salam") via the explicit
# PRIMARY_DOCTOR_ALIASES table — never via fuzzy string similarity between
# an Arabic string and a Latin one (which is meaningless and previously
# caused correct Arabic doctor names to be reported as unapproved).
# ═════════════════════════════════════════════════════════════════════════

# Items 1-10 — each Arabic alias resolves to its exact canonical identity.
@pytest.mark.parametrize("arabic_name,coe,expected_canonical", [
    ("اسامه عبد السلام", "Headache", "Osama Abdel Salam"),          # Item 1
    ("أسامة عبد السلام", "Headache", "Osama Abdel Salam"),          # Item 2
    ("اسامة عبدالسلام", "Headache", "Osama Abdel Salam"),           # Item 3
    ("عمر ايوب", "Headache", "Omar Ayoub"),                          # Item 4
    ("عمر أيوب", "Headache", "Omar Ayoub"),                          # Item 5
    ("عبدالرحمن الشهري", "Headache", "Abdelrhman Alshehri"),         # Item 6
    ("عبد الرحمن بوقس", "Headache", "Abdulrahman Bogus"),            # Item 7
    ("نجوى الحلواني", "Asthma", "Nagwa Elhalawani"),                 # Item 8
    ("عيد العجمي", "Asthma", "Eid Elajmi"),                          # Item 9
    ("بدري بيروتي", "Diabetes", "Badri Bairuti"),                    # Item 10
])
def test_arabic_alias_resolves_to_canonical_english_identity(arabic_name, coe, expected_canonical):
    assert resolve_primary_doctor_identity(arabic_name, coe) == expected_canonical
    assert match_primary_doctor(arabic_name, coe) is True


def test_arabic_doctor_titles_do_not_affect_matching():
    """Item 11."""
    bare = resolve_primary_doctor_identity("اسامة عبد السلام", "Headache")
    with_title = resolve_primary_doctor_identity("دكتور اسامة عبد السلام", "Headache")
    with_female_title = resolve_primary_doctor_identity("دكتورة نجوى الحلواني", "Asthma")
    assert bare == "Osama Abdel Salam"
    assert with_title == "Osama Abdel Salam"
    assert with_female_title == "Nagwa Elhalawani"


def test_english_doctor_titles_do_not_affect_matching():
    """Item 12."""
    bare = resolve_primary_doctor_identity("Osama Abdel Salam", "Headache")
    with_dr = resolve_primary_doctor_identity("Dr. Osama Abdel Salam", "Headache")
    with_doctor = resolve_primary_doctor_identity("Doctor Osama Abdel Salam", "Headache")
    assert bare == with_dr == with_doctor == "Osama Abdel Salam"


def test_arabic_doctor_rejected_for_wrong_coe():
    """Item 13 — the same Arabic doctor name that resolves correctly for
    Headache must not resolve at all for a different COE."""
    assert resolve_primary_doctor_identity("اسامة عبد السلام", "Headache") == "Osama Abdel Salam"
    assert resolve_primary_doctor_identity("اسامة عبد السلام", "Asthma") is None
    assert resolve_primary_doctor_identity("اسامة عبد السلام", "Diabetes") is None
    assert resolve_primary_doctor_identity("اسامة عبد السلام", "IBD") is None


def test_ambiguous_partial_arabic_name_is_rejected():
    """Item 14 — "عبد الرحمن" is the shared first+middle name of TWO
    different approved Headache doctors (Abdelrhman Alshehri and
    Abdulrahman Bogus); the bare partial must never be force-matched to
    either."""
    assert resolve_primary_doctor_identity("عبد الرحمن", "Headache") is None
    assert resolve_primary_doctor_identity("عبدالرحمن", "Headache") is None


def test_unrelated_arabic_name_is_rejected():
    """Item 15."""
    assert resolve_primary_doctor_identity("محمد إبراهيم", "Headache") is None
    assert resolve_primary_doctor_identity("خالد المصري", "Asthma") is None


def test_secondary_specialty_arabic_doctor_remains_unapproved():
    """Item 16 — an Arabic name for a doctor who simply isn't on any
    approved-primary-doctor list must never resolve, regardless of how
    cleanly it normalises."""
    assert resolve_primary_doctor_identity("سمير يوسف", "Headache") is None
    assert resolve_primary_doctor_identity("دكتور سمير يوسف", "Asthma") is None


def test_two_valid_arabic_headache_doctors_produce_pass():
    """Item 17 — node-level: two Arabic-named approved Headache doctors
    offered for the initial appointment yield primary_doctor_status='pass'."""
    transcript = (
        "Patient: عندي صداع نصفي متكرر من فترة طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع\n"
        "Agent: يبدأ الحجز أولاً في عيادة المخ والأعصاب، والمتاح لحضرتك دكتور أسامة عبد "
        "السلام أو دكتور عمر أيوب."
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"
    assert set(result["matched_primary_doctors"]) == {"Osama Abdel Salam", "Omar Ayoub"}
    assert result["is_violation"] is False


def test_mixture_of_invalid_and_valid_initial_doctor_fails():
    """Item 18 — SUPERSEDED by the multi-context work's item 8: when two
    doctors are offered as alternatives for the SAME initial appointment
    and one is unapproved, the context must FAIL — one approved doctor
    must never make an unapproved alternative pass (previously this
    asserted the opposite, "any one approved is enough"; that rule is
    intentionally reversed here)."""
    transcript = (
        "Patient: عندي صداع نصفي متكرر من فترة طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع\n"
        "Agent: المتاح لحضرتك دكتور سمير يوسف أو دكتور عمر أيوب."
    )
    result, stub = run_coe_node(transcript)
    assert result["primary_doctor_status"] == "fail"
    assert result["is_violation"] is True


def test_original_arabic_extracted_names_preserved_in_output():
    """Item 19 — the original Arabic transcript wording must remain
    present in both recommended_or_selected_doctors and the evidence,
    never silently replaced by the translated/canonical English form."""
    transcript = (
        "Patient: عندي صداع نصفي متكرر من فترة طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع\n"
        "Agent: يبدأ الحجز أولاً في عيادة المخ والأعصاب، والمتاح لحضرتك دكتور أسامة عبد "
        "السلام أو دكتور عمر أيوب."
    )
    result, stub = run_coe_node(transcript)
    joined_doctors = " ".join(result["recommended_or_selected_doctors"])
    assert "اسامة" in joined_doctors or "أسامة" in joined_doctors or normalize_doctor_name_for_match("أسامة عبد السلام") in normalize_doctor_name_for_match(joined_doctors)
    assert any("أسامة" in ev or "اسامة" in ev for ev in result["evidence"])
    assert any("أيوب" in ev or "ايوب" in ev for ev in result["evidence"])
    # Never replaced with the English canonical spelling in the evidence.
    assert not any("Osama" in ev for ev in result["evidence"])


def test_alias_table_covers_every_authoritative_doctor_bilingually():
    """Sanity check on the alias data itself: every authoritative primary
    doctor has at least one Arabic AND one English alias registered."""
    _ar_range = "؀-ۿ"
    for coe, doctors in PRIMARY_DOCTOR_ALIASES.items():
        for canonical, aliases in doctors.items():
            has_arabic = any(any(ch.isalpha() and "؀" <= ch <= "ۿ" for ch in a) for a in aliases)
            has_english = any(a.isascii() for a in aliases)
            assert has_arabic, f"{canonical} ({coe}) has no Arabic alias"
            assert has_english, f"{canonical} ({coe}) has no English alias"


# ═════════════════════════════════════════════════════════════════════════
# coe_test.json direct-execution scenario (Item 20)
# ═════════════════════════════════════════════════════════════════════════

def test_coe_test_json_scenario_matches_expected_result():
    """Item 20 — the exact regression scenario reported: two Arabic-named
    approved Headache doctors offered for the initial appointment must
    yield a full pass with no violation."""
    transcript = (
        "Agent: السلام عليكم، مع حضرتك أحمد من مجموعة أندلسية صحة. أتشرف باسم حضرتك؟\n"
        "Patient: وعليكم السلام، معك محمد.\n"
        "Agent: أهلاً وسهلاً أستاذ محمد، كيف أقدر أساعد حضرتك؟\n"
        "Patient: أعاني من صداع نصفي متكرر من فترة وأريد أحجز عند طبيب متخصص.\n"
        "Agent: لضمان تحقيق أقصى استفادة والوصول إلى تشخيص دقيق لحالتك من جميع الجوانب "
        "الطبية، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع، والذي "
        "يضم نخبة من أفضل الاستشاريين والأخصائيين في هذا المجال.\n"
        "Patient: تمام، أريد الحجز في مركز التميز.\n"
        "Agent: يبدأ الحجز أولاً في عيادة المخ والأعصاب، والمتاح لحضرتك دكتور أسامة عبد "
        "السلام أو دكتور عمر أيوب.\n"
        "Patient: أحجز مع دكتور عمر أيوب لو سمحت.\n"
        "Agent: تم تأكيد حجز موعدك مع دكتور عمر أيوب في عيادة المخ والأعصاب ضمن مركز "
        "التميز لعلاج الصداع.\n"
        "Patient: شكراً."
    )
    result, stub = run_coe_node(transcript, call_id="COE-TEST-HEADACHE-001")
    assert result["expected_coe"] == "Headache"
    assert result["recommended_coe"] == "Headache"
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"
    assert result["is_violation"] is False
    assert set(result["matched_primary_doctors"]) == {"Osama Abdel Salam", "Omar Ayoub"}


# ═════════════════════════════════════════════════════════════════════════
# Existing-patient exception (pure, deterministic)
# ═════════════════════════════════════════════════════════════════════════

def test_existing_patient_followup_language_detected():
    c = call("Patient: انا بتابع مع دكتور رامي من زمان في عيادة الروماتيزم\nAgent: تمام")
    assert existing_patient_exception_evidence(c) is not None


def test_vague_existing_patient_label_alone_not_treated_as_exception():
    """A bare self-label ('مريض قديم') without naming an ongoing treating-
    doctor relationship must not, by itself, trigger the exception — see
    Item 23 (a new COE journey requested by an existing patient)."""
    c = call("Patient: انا مريض قديم بس حابب ابدأ رحلة جديدة\nAgent: تمام")
    assert existing_patient_exception_evidence(c) is None


# ═════════════════════════════════════════════════════════════════════════
# HTML handling in the Asthma Script_AR CRM field
# ═════════════════════════════════════════════════════════════════════════

def test_asthma_script_html_is_stripped_and_entities_decoded():
    """Item 27 — the Asthma Script_AR CRM value may contain raw HTML; it
    must be safely stripped/decoded before use."""
    html_script = (
        "<p>لضمان تحقيق أقصى استفادة &amp; الوصول إلى تشخيص دقيق لحالتك من جميع الجوانب "
        "الطبية، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الصدر والجهاز "
        "التنفسي، والذي يضم نخبة من أفضل الاستشاريين والأخصائيين.</p>"
    )
    rows = [{
        "Clinic_Name": "Asthma", "Clinic_Leader": "Dr. X", "Specialty": "Pulmonology",
        "BU": "AHJ", "Clinic_Coordinator": "Coordinator", "Member": "Dr. X, Dr. Y",
        "Script_AR": html_script,
    }]
    reference = build_coe_reference(rows)
    assert "<p>" not in reference["Asthma"]["script_ar"]
    assert "&amp;" not in reference["Asthma"]["script_ar"]
    assert "&" in reference["Asthma"]["script_ar"]  # entity decoded, not dropped
    assert "مركز التميز" in reference["Asthma"]["script_ar"]


def test_build_coe_reference_handles_missing_and_malformed_rows():
    """Item 26 — a missing row, a duplicate row, null fields, and an
    unsupported COE name must all be handled safely, never crashing."""
    rows = [
        {"Clinic_Name": "Headache", "Script_AR": None, "Member": None},
        {"Clinic_Name": "Headache", "Script_AR": "  ", "Clinic_Leader": "Dr. Leader"},
        {"Clinic_Name": "UnsupportedCOE", "Script_AR": "some text"},
        None,  # malformed row
        "not-a-dict",  # malformed row
    ]
    reference = build_coe_reference(rows)
    assert set(reference.keys()) == {"IBD", "Headache", "Asthma", "Diabetes"}
    # Falls back to the default canonical script when CRM data is null/blank.
    assert reference["Headache"]["script_ar"] == DEFAULT_SCRIPTS_AR["Headache"]
    assert reference["Headache"]["clinic_leader"] == "Dr. Leader"
    # IBD/Asthma/Diabetes rows were never present at all — still safe.
    assert reference["IBD"]["script_ar"] == DEFAULT_SCRIPTS_AR["IBD"]


def test_build_coe_reference_handles_empty_or_none_rows_list():
    for rows in ([], None):
        reference = build_coe_reference(rows)
        assert set(reference.keys()) == {"IBD", "Headache", "Asthma", "Diabetes"}
        for key in reference:
            assert reference[key]["script_ar"] == DEFAULT_SCRIPTS_AR[key]


def test_coe_crm_fetch_failure_does_not_crash_node(monkeypatch):
    """A COE lookup failure (CRM error) must not crash the pipeline — the
    node degrades to using DEFAULT_SCRIPTS_AR and still produces a result."""
    import app.service_hub.crm_coe as crm_coe

    def _boom(*a, **k):
        raise RuntimeError("simulated CRM outage")

    monkeypatch.setattr(crm_coe, "fetch_coe_reference", _boom)
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: هيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Omar Ayoub"
    )
    result, stub = run_coe_node(transcript)
    assert result["applicable"] is True
    assert result["coe_match_status"] == "pass"


# ═════════════════════════════════════════════════════════════════════════
# infer_coe_validation node — end-to-end scenarios (deterministic + stub LLM)
# ═════════════════════════════════════════════════════════════════════════

def test_ibd_complaint_and_dalinda_both_pass():
    """Item 1 — IBD complaint, IBD COE recommendation, Dr. Dalinda as the
    initial doctor: both checks pass."""
    transcript = (
        "Patient: عندي اسهال مزمن وآلام في الجهاز الهضمي من فترة طويلة\n"
        "Agent: لضمان تحقيق أقصى استفادة، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في "
        "علاج أمراض الجهاز الهضمي مع Dr. Dalinda"
    )
    result, stub = run_coe_node(transcript)
    assert result["triggered"] is True
    assert result["expected_coe"] == "IBD"
    assert result["recommended_coe"] == "IBD"
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"
    assert result["is_violation"] is False


@pytest.mark.parametrize("doctor_name", [
    "Osama Abdel Salam", "Abdelrhman Alshehri", "Omar Ayoub", "Abdulrahman Bogus",
])
def test_headache_complaint_and_coe_with_each_approved_doctor_passes(doctor_name):
    """Items 2-5 — Headache complaint + Headache COE with each of the four
    approved primary doctors: both checks pass."""
    transcript = (
        "Patient: عندي صداع نصفي شديد ومتكرر من فترة طويلة\n"
        f"Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. {doctor_name}"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"


@pytest.mark.parametrize("doctor_name", ["Nagwa Elhalawani", "Eid Elajmi"])
def test_asthma_complaint_and_coe_with_each_approved_doctor_passes(doctor_name):
    """Items 6-7."""
    transcript = (
        "Patient: عندي ربو وضيق شديد في التنفس من مدة طويلة\n"
        f"Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي مع Dr. {doctor_name}"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"


def test_diabetes_complaint_and_coe_with_badri_bairuti_passes():
    """Item 8."""
    transcript = (
        "Patient: عندي مرض السكر ومحتاج متابعة دقيقة لحالتي\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض السكر والغدد الصماء مع Dr. Badri Bairuti"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"


def test_asthma_coe_starts_with_ent_doctor_fails_primary_doctor_only():
    """Item 9 — COE check passes, primary-doctor check fails."""
    transcript = (
        "Patient: عندي ربو وضيق شديد في التنفس\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي مع Dr. Samir Youssef"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "fail"
    assert result["is_violation"] is True


def test_diabetes_coe_starts_with_wrong_doctor_fails_primary_doctor_only():
    """Item 10 — diabetic educator/orthopedic doctor as the initial doctor:
    COE check passes, primary-doctor check fails."""
    transcript = (
        "Patient: عندي مرض السكر ومحتاج متابعة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض السكر والغدد الصماء مع Dr. Farid Spine"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "fail"


def test_headache_coe_starts_with_ophthalmology_doctor_fails_primary_doctor_only():
    """Item 11."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Yousef Kamal"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "fail"


def test_diabetes_complaint_but_agent_recommends_headache_coe_fails():
    """Item 12 — COE check fails."""
    transcript = (
        "Patient: عندي مرض السكر وعايز اتابع حالتي\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "fail"
    assert result["expected_coe"] == "Diabetes"
    assert result["recommended_coe"] == "Headache"
    assert result["is_violation"] is True


def test_customer_inquiry_specialized_center_correct_recommendation_validated():
    """Item 15 (node-level) — triggered via customer inquiry, correct COE
    validated as a pass."""
    transcript = (
        "Patient: في عندكم مركز متخصص للصداع؟\n"
        "Patient: عندي صداع نصفي شديد من فترة طويلة\n"
        "Agent: أيوه، سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع"
    )
    result, stub = run_coe_node(transcript)
    assert result["triggered"] is True
    assert result["trigger_path"] == "customer_inquiry"
    assert result["coe_match_status"] == "pass"


def test_faithful_paraphrase_recommendation_accepted():
    """Item 17 (node-level) — a faithful paraphrase (not the exact script)
    is still accepted as a valid COE recommendation."""
    transcript = (
        "Patient: عندي كحه وضيق في التنفس من مده طويلة جدا\n"
        "Agent: عشان تاخد افضل تشخيص لحالتك من كل النواحي الطبية هنحجزلك موعد بمركز "
        "التميز المتخصص في علاج امراض الصدر والجهاز التنفسي عندنا نخبة من افضل الاستشاريين"
    )
    result, stub = run_coe_node(transcript)
    assert result["triggered"] is True
    assert result["coe_match_status"] == "pass"
    assert result["expected_coe"] == "Asthma"


def test_multiple_complaints_no_clear_primary_is_uncertain_not_fabricated_fail():
    """Item 18 (node-level) — uncertain, not a fabricated failure."""
    transcript = (
        "Patient: عندي صداع وعندي كمان مشاكل في السكر\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "uncertain"
    assert result["is_violation"] is False


def test_coe_discussed_but_no_doctor_selection_is_not_applicable_for_doctor_check():
    """Item 19 — COE check is evaluated; primary-doctor check is
    not_applicable because the call never reached doctor selection."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع، هرجعلك بمواعيد الدكاترة لاحقا"
    )
    result, stub = run_coe_node(transcript)
    assert result["coe_match_status"] == "pass"
    assert result["booking_discussed"] is False
    assert result["primary_doctor_status"] == "not_applicable"


def test_multiple_doctors_offered_one_unapproved_fails():
    """Item 20 — SUPERSEDED by the multi-context work's item 8: multiple
    doctors offered as alternatives for the initial clinic, one of them
    unapproved, must FAIL the context (previously asserted "pass" under
    the old "any one approved is enough" rule; that rule is intentionally
    reversed here — see test_mixture_of_invalid_and_valid_initial_doctor_
    fails and app.service_hub.coe_validation.evaluate_context_doctors)."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع، "
        "ممكن نحجزلك مع Dr. Yousef Kamal او Dr. Omar Ayoub"
    )
    result, stub = run_coe_node(transcript)
    assert result["primary_doctor_status"] == "fail"
    assert result["is_violation"] is True


def test_secondary_doctor_mentioned_only_as_later_referral_does_not_fail():
    """Item 21 — a secondary doctor mentioned only as a possible later
    referral must not fail primary-doctor validation. The stub LLM
    explicitly classifies the mentioned doctor as referral-only."""
    transcript = (
        "Patient: عندي ربو وضيق شديد في التنفس\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز "
        "التنفسي مع Dr. Nagwa Elhalawani، وممكن بعد التقييم الأولي دكتور الانف والاذن "
        "والحنجرة Dr. Samir Youssef يتابع معاك لو احتجت"
    )
    result, stub = run_coe_node(transcript, {
        "primary_complaint_category": "Asthma",
        "recommended_coe": "Asthma",
        "initial_doctors": ["Nagwa Elhalawani"],
        "referral_only_doctors": ["Samir Youssef"],
    })
    assert result["primary_doctor_status"] != "fail"
    assert result["primary_doctor_status"] == "pass"


def test_existing_patient_continues_with_established_doctor_exception_applies():
    """Item 22 — existing-patient exception applied; primary-doctor
    validation does not fail."""
    transcript = (
        "Patient: انا بتابع مع دكتور رامي من زمان بس سمعت عندكم مركز متخصص للصداع\n"
        "Patient: عندي صداع نصفي شديد بردو\n"
        "Agent: أيوه عندنا مركز التميز المتخصص في تشخيص وعلاج الصداع، بما انك بتابع مع "
        "دكتور رامي هنكمل معاه المتابعة العادية"
    )
    result, stub = run_coe_node(transcript)
    assert result["existing_patient_exception"] is True
    assert result["primary_doctor_status"] != "fail"


def test_new_coe_journey_requested_by_existing_patient_applies_normal_rules():
    """Item 23 — the existing-patient label alone (no ongoing treating-
    doctor relationship named) does not exempt a NEW COE journey from
    normal primary-doctor rules."""
    transcript = (
        "Patient: انا مريض قديم بس حابب ابدأ رحلة جديدة، عندي مرض السكر\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض السكر والغدد "
        "الصماء مع Dr. Farid Spine"
    )
    result, stub = run_coe_node(transcript)
    assert result["existing_patient_exception"] is False
    assert result["primary_doctor_status"] == "fail"


def test_ambiguous_doctor_similarity_does_not_force_match():
    """Item 25 (node-level) — an ambiguous/generic doctor name must not be
    force-matched to an approved doctor."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Ahmed"
    )
    result, stub = run_coe_node(transcript)
    assert result["primary_doctor_status"] != "pass"


def test_unsupported_coe_or_complaint_returns_not_applicable_or_uncertain():
    """Item 29 — a complaint outside the four supported COEs must never be
    force-mapped; with no COE trigger at all, the node returns
    not_applicable without inventing a mapping."""
    transcript = "Patient: عندي الم في الركبة من اصابة رياضية\nAgent: هوصلك بدكتور عظام"
    result, stub = run_coe_node(transcript)
    assert result["applicable"] is False
    assert result["coe_match_status"] == "not_applicable"
    assert stub.called is False  # LLM never called when not triggered


# ═════════════════════════════════════════════════════════════════════════
# skip_coe_validation node
# ═════════════════════════════════════════════════════════════════════════

def test_skip_coe_validation_node_returns_not_applicable_without_node_trace():
    """Mirrors skip_doctor_scope_validation: the skip node itself must
    never write a node_trace entry (see app.agent.graph._coe_intent_router)."""
    c = call("Patient: عندي استفسار عن الأسعار\nAgent: تفضل")
    state = {"call": c, "node_trace": []}
    result = skip_coe_validation(state)
    assert result["coe_validation"]["coe_match_status"] == "not_applicable"
    assert result["coe_validation"]["applicable"] is False
    assert "node_trace" not in result


def test_infer_coe_validation_llm_never_called_when_not_triggered():
    """LLM must never be called when the deterministic trigger is absent."""
    result, stub = run_coe_node("Patient: عايز احجز كشف بكرة\nAgent: تمام هحجزلك بكرة الساعة 5")
    assert result["applicable"] is False
    assert stub.called is False


def test_malformed_empty_llm_response_still_produces_safe_result():
    """A COE LLM failure/empty response must never crash the node — it
    degrades safely, using deterministic signals alone."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع مع Dr. Omar Ayoub"
    )
    result, stub = run_coe_node(transcript, {})  # empty/degenerate LLM response
    assert result["applicable"] is True
    assert result["coe_match_status"] == "pass"
    assert result["primary_doctor_status"] == "pass"


def test_llm_call_failure_degrades_safely_without_crashing_pipeline(monkeypatch):
    """A COE lookup/LLM failure must not crash the complete QA pipeline —
    see module docstring."""
    transcript = (
        "Patient: عندي صداع نصفي شديد جدا من فتره طويلة\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع"
    )
    c = call(transcript)

    class _FailingLLM:
        async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
            raise RuntimeError("simulated LLM outage")

    state = {"call": c, "node_trace": []}
    result = asyncio.run(infer_coe_validation(state, _FailingLLM()))
    assert "coe_validation" in result
    assert result["node_trace"] == ["infer_coe_validation"]
    # Deterministic signals alone still resolve the COE match correctly.
    assert result["coe_validation"]["coe_match_status"] == "pass"


# ═════════════════════════════════════════════════════════════════════════
# Graph-level routing
# ═════════════════════════════════════════════════════════════════════════

def test_coe_intent_router_routes_correctly():
    triggered = call("Patient: عايز مركز متخصص للصداع\nAgent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع")
    not_triggered = call("Patient: عايز احجز كشف بكرة\nAgent: تمام هحجزلك بكرة الساعة 5")
    assert _coe_intent_router({"call": triggered}) == "infer_coe_validation"
    assert _coe_intent_router({"call": not_triggered}) == "skip_coe"


class _StubLLMClient:
    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return "{}", {"prompt_tokens": 0, "completion_tokens": 0}


def test_graph_compiles_with_coe_nodes():
    graph = build_qa_graph(_StubLLMClient())
    node_names = set(graph.get_graph().nodes.keys())
    assert "infer_coe_validation" in node_names
    assert "skip_coe_validation" in node_names


# ═════════════════════════════════════════════════════════════════════════
# PHASE 9 — centralized COE_SPECIALTIES/SPECIALTY_ALIASES taxonomy,
# phrase-aware shared-specialty disambiguation, multi-context association,
# and doctor-extraction/identity safeguards.
# ═════════════════════════════════════════════════════════════════════════

# ── 1. Specialty -> COE mapping (taxonomy) ──────────────────────────────────

def test_git_specialty_maps_to_ibd():
    assert SPECIALTY_TO_COES["GIT"] == ["IBD"]
    assert resolve_canonical_specialty("الجهاز الهضمي") == "GIT"


def test_nutrition_specialty_maps_to_ibd():
    assert SPECIALTY_TO_COES["Nutrition"] == ["IBD"]
    assert resolve_canonical_specialty("تغذية علاجية") == "Nutrition"


def test_general_surgery_specialty_maps_to_ibd():
    assert SPECIALTY_TO_COES["General Surgery"] == ["IBD"]
    assert resolve_canonical_specialty("جراحة عامة") == "General Surgery"


def test_neurology_specialty_maps_to_headache():
    assert SPECIALTY_TO_COES["Neurology"] == ["Headache"]
    assert resolve_canonical_specialty("المخ والاعصاب") == "Neurology"


def test_ophthalmology_specialty_maps_to_headache():
    assert SPECIALTY_TO_COES["Ophthalmology"] == ["Headache"]
    assert resolve_canonical_specialty("طب العيون") == "Ophthalmology"


def test_cardiology_specialty_maps_to_headache():
    assert SPECIALTY_TO_COES["Cardiology"] == ["Headache"]
    assert resolve_canonical_specialty("طب القلب") == "Cardiology"


def test_psychiatry_specialty_maps_to_headache():
    assert SPECIALTY_TO_COES["Psychiatry"] == ["Headache"]
    assert resolve_canonical_specialty("الصحة النفسية") == "Psychiatry"


def test_dental_specialty_maps_to_headache():
    assert SPECIALTY_TO_COES["Dental"] == ["Headache"]
    assert resolve_canonical_specialty("طب الأسنان") == "Dental"


def test_diabetes_specialty_maps_to_diabetes():
    assert SPECIALTY_TO_COES["Diabetes"] == ["Diabetes"]
    assert resolve_canonical_specialty("مرض السكر") == "Diabetes"


def test_diabetic_educator_specialty_maps_to_diabetes():
    assert SPECIALTY_TO_COES["Diabetic Educator"] == ["Diabetes"]
    assert resolve_canonical_specialty("التثقيف السكري") == "Diabetic Educator"


def test_orthopedics_specialty_maps_to_diabetes():
    assert SPECIALTY_TO_COES["Orthopedics"] == ["Diabetes"]
    assert resolve_canonical_specialty("جراحة العظام") == "Orthopedics"


def test_pulmonology_specialty_maps_to_asthma():
    assert SPECIALTY_TO_COES["Pulmonology"] == ["Asthma"]
    assert resolve_canonical_specialty("الجهاز التنفسي") == "Pulmonology"


def test_allergy_immunology_specialty_maps_to_asthma():
    assert SPECIALTY_TO_COES["Allergy & Immunology"] == ["Asthma"]
    assert resolve_canonical_specialty("الحساسية والمناعة") == "Allergy & Immunology"


def test_ent_specialty_is_shared_between_headache_and_asthma():
    """Item 14 — ENT is organizationally SHARED, unlike every other
    specialty, which maps to exactly one COE."""
    assert set(SPECIALTY_TO_COES["ENT"]) == {"Headache", "Asthma"}


def test_every_coe_has_a_specialties_entry_in_the_taxonomy():
    assert set(COE_SPECIALTIES.keys()) == {"IBD", "Headache", "Asthma", "Diabetes"}
    for coe, specialties in COE_SPECIALTIES.items():
        assert specialties, f"{coe} must list at least one specialty"


def test_resolve_canonical_specialty_is_phrase_aware_not_naive_substring():
    """Item 16 — "قلب" (heart/Cardiology) must not match inside unrelated
    words like "انقلاب" (accident/overturn) via naive substring search."""
    assert resolve_canonical_specialty("حصل انقلاب للسيارة في الطريق") is None
    assert resolve_canonical_specialty("عندي الم في القلب") == "Cardiology"


def test_resolve_canonical_specialty_recognises_english_aliases():
    assert resolve_canonical_specialty("Gastroenterology follow-up") == "GIT"
    assert resolve_canonical_specialty("Pulmonology clinic") == "Pulmonology"


def test_weak_specialty_manazeer_alone_is_not_strong_evidence():
    """Item 18 — "مناظير" (scopes/endoscopy) alone, with no corroborating
    GIT/IBD context, must never by itself create or confirm a context."""
    assert detect_specialty_mentions("عايز احجز مناظير") == []
    assert resolve_specialty_coes("عايز احجز مناظير") == []


def test_weak_specialty_manazeer_confirmed_when_git_context_already_active():
    """Item 19 — "مناظير" becomes valid GIT/IBD evidence only once an IBD
    context is already active (e.g. from an earlier GIT/liver mention)."""
    assert detect_weak_specialty_mentions("عايز احجز مناظير") == [("GIT", ["IBD"])]
    assert resolve_specialty_coes("عايز احجز مناظير", active_coes=["IBD"]) == ["IBD"]


# ── 2. Shared-specialty disambiguation priority order ───────────────────────

def test_shared_specialty_resolved_by_campaign_coe():
    """Item 20 — priority 1: an explicit campaign-established COE resolves
    a later shared-specialty (ENT) mention."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
        "Agent: تمام هحولك لعيادة انف واذن وحنجرة لتقييم اولي"
    )
    contexts = build_coe_contexts(call(transcript))
    assert set(contexts) == {"Headache"}
    assert "ENT" in [s["canonical_specialty"] for s in contexts["Headache"]["specialties"]]


def test_shared_specialty_resolved_by_explicit_agent_coe_name():
    """Item 21 — priority 2: an explicit agent-named COE (no campaign
    involved) resolves a same-turn shared-specialty (ENT) mention."""
    transcript = (
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي "
        "مع دكتور في عيادة انف واذن وحنجرة"
    )
    contexts = build_coe_contexts(call(transcript))
    assert set(contexts) == {"Asthma"}
    assert "ENT" in [s["canonical_specialty"] for s in contexts["Asthma"]["specialties"]]


def test_shared_specialty_resolved_by_patient_complaint():
    """Item 22 — priority 3: the patient's own complaint resolves a later
    shared-specialty (ENT) mention when no campaign/explicit-name evidence
    exists."""
    transcript = "Patient: عندي ربو وضيق تنفس\nAgent: هحولك لعيادة انف واذن وحنجرة"
    contexts = build_coe_contexts(call(transcript))
    assert set(contexts) == {"Asthma"}
    assert "ENT" in [s["canonical_specialty"] for s in contexts["Asthma"]["specialties"]]


def test_shared_specialty_unresolved_is_uncertain_not_guessed():
    """Item 23 — with no campaign, explicit COE name, complaint, or active
    context to disambiguate it, a bare ENT mention must be surfaced as
    ambiguous/uncertain rather than guessed into either Headache or
    Asthma."""
    transcript = "Agent: هحولك لعيادة انف واذن وحنجرة"
    c = call(transcript)
    assert build_coe_contexts(c) == {}
    assert resolve_specialty_coes("عيادة انف واذن وحنجرة") == []
    ambiguous = ambiguous_specialty_mentions("عيادة انف واذن وحنجرة")
    assert ambiguous and ambiguous[0][0] == "ENT"
    assert set(ambiguous[0][1]) == {"Headache", "Asthma"}


# ── 3. Multi-context behavior ────────────────────────────────────────────────

def test_multi_context_headache_and_ibd_each_get_own_correct_doctor():
    """Item 24 — the exact multi-context regression: a campaign-established
    Headache context and a patient-initiated IBD context in the same call
    each independently resolve to their own correctly-approved doctor."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
        "Patient: انا عندي صداع مزمن\n"
        "Agent: المتاح دكتور اسامه عبدالسلام بعيادة المخ والاعصاب\n"
        "Patient: وبعده بدي جهاز هضمي\n"
        "Agent: تمام هوصلك لدكتور جهاز هضمي، وبيتم التحويل بعد ذلك\n"
        "Agent: تم تأكيد حجزك مع داليندا عرفاوي استشاري امراض الجهاز الهضمي والكبد والمناظير\n"
    )
    by_coe = {e["coe"]: e for e in build_coe_evaluations(call(transcript))}
    assert set(by_coe) == {"Headache", "IBD"}
    assert by_coe["Headache"]["primary_doctor_status"] == "pass"
    assert by_coe["Headache"]["doctors"][0]["canonical_name"] == "Osama Abdel Salam"
    assert by_coe["IBD"]["primary_doctor_status"] == "pass"
    assert by_coe["IBD"]["doctors"][0]["canonical_name"] == "Dalinda"


def test_one_approved_doctor_never_validates_an_unrelated_coe_context():
    """Item 25 — an approved Headache doctor discussed in one context must
    never make an unrelated, unapproved doctor discussed under a DIFFERENT
    COE (here IBD) pass — each context is independently validated."""
    transcript = (
        "Patient: عندي صداع مزمن\n"
        "Agent: هيبدأ الحجز بعيادة المخ والاعصاب مع دكتور اسامة عبدالسلام\n"
        "Patient: وبعده عايز جهاز هضمي\n"
        "Agent: تمام هحولك لعيادة الجهاز الهضمي مع دكتور فلان الفلاني\n"
    )
    by_coe = {e["coe"]: e for e in build_coe_evaluations(call(transcript))}
    assert by_coe["Headache"]["primary_doctor_status"] == "pass"
    assert by_coe["IBD"]["primary_doctor_status"] == "fail"
    assert by_coe["IBD"]["doctors"][0]["canonical_name"] is None


def test_multiple_specialties_within_the_same_coe_all_recorded():
    """Item 26 — GIT and Nutrition, both IBD specialties, mentioned in the
    same call, are both recorded under the single IBD context rather than
    creating separate contexts."""
    transcript = (
        "Patient: عايز اعرف عن الجهاز الهضمي\n"
        "Agent: تمام، عندنا كمان قسم تغذية علاجية بيتابع مع نفس البرنامج"
    )
    contexts = build_coe_contexts(call(transcript))
    assert set(contexts) == {"IBD"}
    specialties = {s["canonical_specialty"] for s in contexts["IBD"]["specialties"]}
    assert {"GIT", "Nutrition"} <= specialties


def test_topic_change_marker_begins_a_new_service_context():
    """Item 27 — "وبعده" (and after that) correctly separates an earlier
    Headache discussion from a later, unrelated IBD discussion instead of
    merging them into one context."""
    transcript = (
        "Patient: عندي صداع مزمن\n"
        "Agent: هيبدأ الحجز بعيادة المخ والاعصاب\n"
        "Patient: وبعده بدي جهاز هضمي\n"
        "Agent: تمام هحولك لعيادة الجهاز الهضمي\n"
    )
    contexts = build_coe_contexts(call(transcript))
    assert set(contexts) == {"Headache", "IBD"}


def test_context_sources_records_every_contributing_evidence_type():
    """Item 28 — context_sources accumulates every distinct kind of
    evidence (campaign, complaint, specialty, recommendation) that
    contributed to a context, not just the first one found."""
    transcript = (
        "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
        "Patient: انا عندي صداع مزمن\n"
        "Agent: هيبدأ الحجز بعيادة المخ والاعصاب\n"
    )
    contexts = build_coe_contexts(call(transcript))
    sources = set(contexts["Headache"]["context_sources"])
    assert "campaign" in sources
    assert "patient_complaint" in sources


def test_specialties_entries_are_structured_with_canonical_specialty_key():
    """Item 29 — each specialties[] entry is a structured dict carrying at
    least the canonical specialty name and its supporting evidence, not a
    flat string."""
    transcript = "Patient: عندي صداع مزمن\nAgent: هيبدأ الحجز بعيادة المخ والاعصاب"
    contexts = build_coe_contexts(call(transcript))
    entry = contexts["Headache"]["specialties"][0]
    assert entry["canonical_specialty"] == "Neurology"
    assert "speaker" in entry and "evidence" in entry


def test_no_grounded_coe_produces_zero_contexts_and_evaluations():
    """Item 30 — a call with no COE-relevant language at all must produce
    zero contexts/evaluations, never a spurious default."""
    transcript = "Patient: عايز اعرف اسعار الكشف العادي\nAgent: الكشف العادي بـ 300 ريال"
    assert build_coe_contexts(call(transcript)) == {}
    assert build_coe_evaluations(call(transcript)) == []


# ── 4. Doctor extraction / identity ──────────────────────────────────────────

@pytest.mark.parametrize("false_positive", [
    "تحويل طبي", "استشارة طبيب", "موعد مع الدكتور", "وبيتم التحويل بعد ذلك",
])
def test_known_false_positive_phrases_are_rejected_as_doctor_candidates(false_positive):
    """Items 31-32 — generic referral/procedural phrases must never be
    accepted as doctor-name candidates."""
    assert is_plausible_coe_doctor_candidate(false_positive) is False


def test_specialty_phrase_rejected_as_doctor_candidate():
    """Item 33 — "جهاز هضمي" extracted from "لدكتور جهاز هضمي" is a
    specialty/clinic phrase, not a personal name, and must be rejected."""
    assert is_plausible_coe_doctor_candidate("جهاز هضمي") is False


def test_doctor_name_extracted_when_title_trails_the_name():
    """Item 34 — "داليندا عرفاوي استشاري ..." names the doctor BEFORE the
    title/role word, which the shared title-anchored engine cannot find on
    its own; the COE-local supplementary extraction must still find it."""
    transcript = (
        "Agent: تم تأكيد حجزك مع داليندا عرفاوي استشاري امراض الجهاز الهضمي والكبد والمناظير"
    )
    ctx = build_coe_evaluations(call(transcript))
    assert len(ctx) == 1
    assert ctx[0]["doctors"][0]["extracted_name"] == "داليندا عرفاوي"


def test_dalinda_arfawy_resolves_to_existing_canonical_dalinda_identity():
    """Item 35 — "داليندا عرفاوي" is an alias of the EXISTING canonical
    "Dalinda" doctor, never a second/new doctor identity."""
    assert resolve_primary_doctor_identity("داليندا عرفاوي", "IBD") == "Dalinda"
    assert match_primary_doctor("داليندا عرفاوي", "IBD") is True


def test_shortened_confirmation_does_not_create_a_second_doctor_entry():
    """Item 36 — a shortened confirmation of an already-offered doctor's
    name must not surface as a second, separate doctor entry after
    deduplication."""
    transcript = (
        "Patient: عندي صداع مزمن\n"
        "Agent: المتاح دكتور اسامة عبدالسلام بعيادة المخ والاعصاب\n"
        "Patient: تمام موافقه على اسامه\n"
    )
    evaluations = build_coe_evaluations(call(transcript))
    assert len(evaluations) == 1
    assert len(evaluations[0]["doctors"]) == 1
    assert evaluations[0]["doctors"][0]["canonical_name"] == "Osama Abdel Salam"


def test_is_plausible_coe_doctor_candidate_accepts_real_doctor_names():
    """Item 37 — the plausibility filter must never reject genuine doctor
    names, only known false-positive patterns."""
    for name in ("اسامة عبدالسلام", "داليندا عرفاوي", "عمر أيوب", "Dr. Eid Elajmi"):
        assert is_plausible_coe_doctor_candidate(name) is True


def test_is_plausible_coe_doctor_candidate_rejects_none_and_empty():
    """Item 38 — the plausibility filter degrades safely on missing input
    rather than raising."""
    assert is_plausible_coe_doctor_candidate(None) is False
    assert is_plausible_coe_doctor_candidate("") is False


# ── 5. Grounding safeguards ──────────────────────────────────────────────────

def test_llm_specialty_claim_without_transcript_evidence_is_discarded():
    """Item 39 — an LLM-claimed recommended_coe with no real transcript
    evidence anywhere is discarded, mirroring the existing grounding rule
    now that specialty vocabulary is much wider."""
    transcript = "Patient: عايز اعرف اسعار الكشف العادي\nAgent: الكشف العادي بـ 300 ريال"
    assert ground_llm_coe_value(call(transcript), "IBD") is None


def test_crm_reference_data_cannot_substitute_transcript_evidence_for_specialty():
    """Item 40 — passing the full COE reference (which necessarily
    contains every specialty name) must never itself create a context;
    only real transcript text does."""
    transcript = "Patient: عايز اعرف اسعار الكشف العادي\nAgent: الكشف العادي بـ 300 ريال"
    contexts = build_coe_contexts(call(transcript), scripts=DEFAULT_SCRIPTS_AR)
    assert contexts == {}


def test_campaign_message_never_attributed_to_agent_speaker():
    """Item 41 — a campaign/marketing identifier persisted under the
    Patient speaker must contribute "campaign" evidence, never
    "agent_recommendation" evidence."""
    transcript = "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع"
    contexts = build_coe_contexts(call(transcript))
    assert "campaign" in contexts["Headache"]["context_sources"]
    assert "agent_recommendation" not in contexts["Headache"]["context_sources"]


def test_doctor_membership_in_coe_specialty_does_not_imply_approval():
    """Item 42 — being discussed within a COE's specialty context is not
    the same thing as being an approved primary doctor for that COE; an
    unapproved name in a correctly-grounded context still fails."""
    transcript = "Agent: هيبدأ الحجز بعيادة المخ والاعصاب مع دكتور فلان الفلاني"
    evaluations = build_coe_evaluations(call(transcript))
    assert evaluations[0]["coe"] == "Headache"
    assert evaluations[0]["primary_doctor_status"] == "fail"
    assert evaluations[0]["doctors"][0]["canonical_name"] is None


def test_generic_referral_language_excluded_from_role_initial_primary():
    """Item 43 — a purely referral-language doctor mention must be tagged
    referral_only, never initial_primary, so it cannot affect the primary-
    doctor pass/fail aggregation."""
    transcript = (
        "Patient: عندي ربو وضيق شديد في التنفس\n"
        "Agent: سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج امراض الصدر والجهاز التنفسي مع Dr. Eid Elajmi، "
        "وممكن بعد التقييم الأولي دكتور الانف والاذن والحنجرة يتابع معاك لو احتجت"
    )
    evaluations = build_coe_evaluations(call(transcript))
    doctors = evaluations[0]["doctors"]
    # The referral-language clause names no personal doctor (only a bare
    # specialty description), so it contributes no doctor entry at all —
    # only the genuine initial offer (Eid Elajmi) is present, and it is
    # correctly tagged initial_primary rather than referral_only.
    assert len(doctors) == 1
    assert doctors[0]["role"] == "initial_primary"
    assert doctors[0]["canonical_name"] == "Eid Elajmi"


# ── 6. Full regression (exact consolidated-prompt scenario) ────────────────

_TAXONOMY_REGRESSION_TRANSCRIPT = (
    "Patient: BU-AHJ-COE- أضغطي علي إرسال للاستفادة بعروضنا في مركز تميز الصداع\n"
    "Patient: ابغى احجز باقة الصداع\n"
    "Agent: برنامج مركز التميز للصداع، هيبدأ الحجز في عيادة المخ والاعصاب مع دكتور اسامة عبدالسلام\n"
    "Patient: تمام موافقه على اسامه\n"
    "Patient: وبعده بدي جهاز هضمي\n"
    "Agent: تمام هوصلك لدكتور جهاز هضمي، وبيتم التحويل بعد ذلك\n"
    "Agent: تم تأكيد حجزك مع داليندا عرفاوي استشاري امراض الجهاز الهضمي والكبد والمناظير\n"
)


def test_regression_produces_exactly_two_contexts():
    """Item 44 — the exact consolidated-prompt regression conversation
    produces exactly two contexts: Headache and IBD."""
    evaluations = build_coe_evaluations(call(_TAXONOMY_REGRESSION_TRANSCRIPT))
    assert {e["coe"] for e in evaluations} == {"Headache", "IBD"}


def test_regression_headache_doctor_passes_cleanly():
    """Item 45 — the Headache context resolves to exactly one doctor,
    Osama Abdel Salam, passing."""
    by_coe = {e["coe"]: e for e in build_coe_evaluations(call(_TAXONOMY_REGRESSION_TRANSCRIPT))}
    headache_doctors = by_coe["Headache"]["doctors"]
    assert len(headache_doctors) == 1
    assert headache_doctors[0]["canonical_name"] == "Osama Abdel Salam"
    assert by_coe["Headache"]["primary_doctor_status"] == "pass"


def test_regression_ibd_doctor_passes_cleanly():
    """Item 46 — the IBD context resolves to exactly one doctor, Dalinda
    (via her "داليندا عرفاوي" alias), passing."""
    by_coe = {e["coe"]: e for e in build_coe_evaluations(call(_TAXONOMY_REGRESSION_TRANSCRIPT))}
    ibd_doctors = by_coe["IBD"]["doctors"]
    assert len(ibd_doctors) == 1
    assert ibd_doctors[0]["canonical_name"] == "Dalinda"
    assert by_coe["IBD"]["primary_doctor_status"] == "pass"


def test_regression_no_false_doctor_candidates_leak_into_either_context():
    """Item 47 — none of "تحويل طبي", "وبيتم التحويل بعد ذلك", or "جهاز
    هضمي" ever appear as extracted doctor names in either context."""
    evaluations = build_coe_evaluations(call(_TAXONOMY_REGRESSION_TRANSCRIPT))
    all_names = {d["extracted_name"] for e in evaluations for d in e["doctors"]}
    for bad_name in ("تحويل طبي", "وبيتم التحويل بعد ذلك", "جهاز هضمي", "التحويل بعد ذلك"):
        assert bad_name not in all_names


def test_regression_via_full_node_path_overall_status_pass():
    """Item 48 — run through the real infer_coe_validation node (not just
    the pure builder functions) and confirm the aggregate outcome is a
    clean pass with no violation."""
    result, stub = run_coe_node(_TAXONOMY_REGRESSION_TRANSCRIPT)
    assert result["overall_coe_status"] == "pass"
    assert result["is_violation"] is False
    assert {e["coe"] for e in result["coe_evaluations"]} == {"Headache", "IBD"}


def test_regression_evaluations_carry_specialties_for_logging():
    """Item 49 — each context's specialties are available in the
    structured form the node's logging/reporting relies on (canonical
    specialty name extractable from each entry)."""
    evaluations = build_coe_evaluations(call(_TAXONOMY_REGRESSION_TRANSCRIPT))
    for e in evaluations:
        specialty_names = [s["canonical_specialty"] for s in e["specialties"]]
        assert specialty_names, f"{e['coe']} context must carry at least one specialty"


def test_graph_run_no_coe_trigger_skips_and_state_present():
    """infer_coe_validation always executes-or-skips (equal hop count with
    the other five inference branches), degrading to NOT_APPLICABLE
    internally without an LLM call when its gate fails — exactly like
    infer_doctor_scope_validation already does."""
    graph = build_qa_graph(_StubLLMClient())
    c = call("Patient: عايز احجز كشف بكرة\nAgent: تمام هحجزلك بكرة الساعة 5", call_id="graph-coe-skip")
    result = asyncio.run(graph.ainvoke({"call": c}))
    assert "infer_coe_validation" not in result["node_trace"]
    assert result["coe_validation"]["coe_match_status"] == "not_applicable"
    assert result["coe_validation"]["applicable"] is False
