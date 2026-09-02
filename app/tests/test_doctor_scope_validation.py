"""Tests for the SECOND, LLM-based doctor validation: clinical scope/
recommendation-suitability (app.agent.nodes.infer_doctor_scope_validation +
app.service_hub.doctor_validation's applicability/evidence helpers).

Deliberately separate from test_doctor_validation.py's deterministic
doctor-information checks — these two validators must stay independent
(see README_doctor.md). Unit tests here use a stubbed LLM client so the
gating/evidence-hierarchy/safety-net logic this module owns is verified
deterministically; the LLM's own semantic judgment quality is a runtime
concern, not something a unit test can meaningfully assert.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models.input import CallTranscript
from app.service_hub.doctor_validation import (
    check_age_eligibility,
    classify_doctor_context,
    doctor_is_selected_for_clinical_need,
    doctor_scope_skip_reason,
    doctor_scope_validation_needed,
    extract_patient_clinical_need,
    extract_patient_stated_age,
    has_detailed_scope_evidence,
    patient_describes_medical_complaint,
)
from app.agent.nodes import infer_doctor_scope_validation, skip_doctor_scope_validation


def call(transcript: str, call_id: str = "scope-test") -> CallTranscript:
    return CallTranscript(
        call_id=call_id, agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-08-30", call_duration_seconds=1, department="Scheduling",
        transcript=transcript,
    )


def doctor_result(
    outcome: str = "PASS",
    doctor_resolved: bool = True,
    scope_reference: dict | None = None,
) -> dict:
    """Minimal stand-in for validate_doctor_information()'s output shape —
    only the fields doctor_scope_validation_needed()/
    infer_doctor_scope_validation() actually read."""
    return {
        "applicable": True, "outcome": outcome, "doctor_resolved": doctor_resolved,
        "doctor_key": (scope_reference or {}).get("doctor_key"),
        "scope_reference": scope_reference,
    }


SCOPE_EPILEPSY = {
    "doctor_key": "700700111", "doctor_name_ar": "خيرية محمد", "doctor_name_en": "Khairya Mohamed",
    "business_unit": "AKW", "specialty": "Pediatrics", "subspecialty": "Pediatric Neurology",
    "manual_specialty": None, "manual_subspecialty": None,
    "scope_of_service": "Treatment of epilepsy and neurological disorders in children. Perform an EEG.",
    "scope_of_service_ar": "علاج الصرع والاضطرابات العصبية عند الأطفال. إجراء تخطيط للمخ.",
    "doctor_notes": None, "examination_age": "Age: 0-16 years Gender : All",
    "qualifications": None, "qualifications_ar": None,
}

SCOPE_ACL_SPORTS = {
    "doctor_key": "700700222", "doctor_name_ar": "رامي سمرقندي", "doctor_name_en": "Rami Samarkandi",
    "business_unit": "AHJ", "specialty": "Orthopedic", "subspecialty": "Sports Medicine",
    "manual_specialty": None, "manual_subspecialty": None,
    "scope_of_service": "Sports injuries, knee arthroscopy, ACL reconstruction.",
    "scope_of_service_ar": "اصابات رياضية، مناظير الركبة، اعادة بناء الرباط الصليبي",
    "doctor_notes": None, "examination_age": "18 years and above",
    "qualifications": None, "qualifications_ar": None,
}

SCOPE_SPINE_ONLY = {
    "doctor_key": "700700333", "doctor_name_ar": "فريد العمود", "doctor_name_en": "Farid Spine",
    "business_unit": "AHJ", "specialty": "Orthopedic", "subspecialty": "Spine Surgery",
    "manual_specialty": None, "manual_subspecialty": None,
    "scope_of_service": "Cervical and lumbar disc surgery, spinal deformity correction.",
    "scope_of_service_ar": "جراحة الديسك العنقي والقطني وتصحيح تشوهات العمود الفقري",
    "doctor_notes": None, "examination_age": None,
    "qualifications": None, "qualifications_ar": None,
}

SCOPE_SPECIALTY_ONLY = {
    "doctor_key": "700700444", "doctor_name_ar": "محمد سالم", "doctor_name_en": "Mohamed Salem",
    "business_unit": "AHJ", "specialty": "Orthopedic", "subspecialty": None,
    "manual_specialty": None, "manual_subspecialty": None,
    "scope_of_service": None, "scope_of_service_ar": None,
    "doctor_notes": None, "examination_age": None,
    "qualifications": "Consultant, 15 years experience", "qualifications_ar": None,
}

SCOPE_WITH_NOTES_ONLY = {
    **SCOPE_SPECIALTY_ONLY,
    "doctor_notes": "يستقبل حالات اصابات الرياضة والمفاصل فقط، لا يستقبل حالات العمود الفقري",
}


class _StubLLM:
    """Returns a fixed JSON response regardless of the prompt — used to
    test infer_doctor_scope_validation's own gating/safety-net logic in
    isolation from real LLM semantic reasoning."""

    def __init__(self, response: dict):
        self._response = response
        self.last_user_prompt: str | None = None

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.last_user_prompt = user_prompt
        return json.dumps(self._response, ensure_ascii=False), {"prompt_tokens": 0, "completion_tokens": 0}


def run_scope_node(transcript: str, doctor_res: dict | None, llm_response: dict) -> tuple[dict, _StubLLM]:
    c = call(transcript)
    stub = _StubLLM(llm_response)
    state = {"call": c, "doctor_validation": doctor_res, "node_trace": []}
    result = asyncio.run(infer_doctor_scope_validation(state, stub))
    return result["doctor_scope_validation"], stub


# ── Applicability gate (pure function) ──────────────────────────────────────

def test_no_doctor_resolved_skips_scope_validation():
    assert doctor_scope_validation_needed(None, "عندي وجع في الركبة") is False
    unresolved = doctor_result(outcome="DOCTOR_UNRESOLVED", doctor_resolved=False, scope_reference=None)
    assert doctor_scope_validation_needed(unresolved, "عندي وجع في الركبة") is False


def test_no_patient_clinical_need_skips_scope_validation():
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    assert doctor_scope_validation_needed(resolved, "عايز اعرف المواعيد المتاحة") is False
    assert doctor_scope_validation_needed(resolved, "") is False


def test_generic_specialty_request_without_resolved_doctor_skips():
    """Item 20 — AMBIGUOUS_DOCTOR/NOT_APPLICABLE deterministic results must
    never reach the scope validator, regardless of what the patient said."""
    ambiguous = doctor_result(outcome="AMBIGUOUS_DOCTOR", doctor_resolved=False, scope_reference=None)
    not_applicable = doctor_result(outcome="NOT_APPLICABLE", doctor_resolved=False, scope_reference=None)
    for res in (ambiguous, not_applicable):
        assert doctor_scope_validation_needed(res, "عندي دكتور عظام محتاج له") is False


def test_booking_only_no_complaint_is_not_applicable_at_node_level():
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    data, stub = run_scope_node(
        "Patient: عايز احجز مع دكتور رامي سمرقندي بكرة\nAgent: تمام هحجزلك",
        resolved, {"outcome": "SUITABLE"},
    )
    assert data["outcome"] == "NOT_APPLICABLE"
    assert data["applicable"] is False
    assert stub.last_user_prompt is None  # LLM never called


# ── Pure evidence-hierarchy helpers ─────────────────────────────────────────

def test_has_detailed_scope_evidence():
    assert has_detailed_scope_evidence(SCOPE_EPILEPSY) is True
    assert has_detailed_scope_evidence(SCOPE_SPECIALTY_ONLY) is False
    assert has_detailed_scope_evidence(None) is False
    assert has_detailed_scope_evidence({"scope_of_service": "", "scope_of_service_ar": "  "}) is False


def test_extract_patient_clinical_need_is_turn_and_speaker_aware():
    transcript = (
        "Patient: عايز احجز كشف بكرة\n"
        "Agent: تمام هوضحلك المواعيد\n"
        "Patient: طفلي عنده صرع وتشنجات متكررة\n"
        "Agent: هوصلك بدكتورة خيرية\n"
    )
    need = extract_patient_clinical_need(call(transcript))
    assert "صرع" in need
    assert "المواعيد" not in need  # Agent explanation must never leak in
    assert "كشف بكرة" not in need  # unrelated Patient turn excluded


def test_extract_patient_stated_age_explicit_vs_ambiguous():
    assert extract_patient_stated_age("طفلي عمره 5 سنين وعنده صرع") == 5.0
    assert extract_patient_stated_age("عمرها 8 شهور") == round(8 / 12, 2)
    assert extract_patient_stated_age("طفلي الصغير عنده صرع") is None  # no explicit number


def test_check_age_eligibility():
    assert check_age_eligibility(5.0, "Age: 0-16 years Gender : All") == "within_range"
    assert check_age_eligibility(25.0, "Age: 0-16 years Gender : All") == "outside_range"
    assert check_age_eligibility(None, "Age: 0-16 years Gender : All") is None
    assert check_age_eligibility(25.0, None) is None
    assert check_age_eligibility(25.0, "weird format with no numbers") is None


# ── Node-level: evidence hierarchy / safety nets (stubbed LLM) ──────────────

def test_clear_scope_match_passes_through_as_suitable():
    """Item 6 — exact doctor + clear documented scope match: the LLM's
    SUITABLE verdict must stand unmodified since detailed scope text
    genuinely backs it."""
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    data, _ = run_scope_node(
        "Patient: طفلي عنده صرع وتشنجات وعايز دكتور مناسب\nAgent: هوصلك بدكتورة خيرية محمد",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "SUITABLE"
    assert data["is_violation"] is False


def test_arabic_complaint_english_scope_field_present_not_downgraded():
    """Item 7 — scope_of_service (English) is populated even when
    scope_of_service_ar is not; has_detailed_scope_evidence must still be
    True so a genuine SUITABLE verdict is never downgraded for missing
    scope data."""
    scope = {**SCOPE_ACL_SPORTS, "scope_of_service_ar": None}
    resolved = doctor_result(scope_reference=scope)
    data, _ = run_scope_node(
        "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور رامي سمرقندي",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "SUITABLE"


def test_english_complaint_arabic_scope_field_present_not_downgraded():
    """Item 8 — the reverse: only scope_of_service_ar populated must still
    count as detailed scope evidence."""
    scope = {**SCOPE_ACL_SPORTS, "scope_of_service": None}
    resolved = doctor_result(scope_reference=scope)
    data, _ = run_scope_node(
        "Patient: I have a torn ACL ligament in my knee\nAgent: connecting you with Dr. Rami",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "SUITABLE"


def test_wrong_orthopedic_subscope_mismatch_passes_through():
    """Items 9 & 10 — same specialty (Orthopedic) as the ACL case, but this
    doctor's documented scope is spine-only. The LLM's UNSUITABLE verdict
    must pass through unmodified — the safety net only ever intervenes on a
    false SUITABLE, never on a correct UNSUITABLE."""
    resolved = doctor_result(scope_reference=SCOPE_SPINE_ONLY)
    data, _ = run_scope_node(
        "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور فريد العمود",
        resolved, {"outcome": "UNSUITABLE", "is_violation": True},
    )
    assert data["outcome"] == "UNSUITABLE"
    assert data["is_violation"] is True


def test_missing_scope_data_downgrades_false_suitable_to_unclear():
    """Item 11 (the key safety net) — the doctor has ONLY a bare specialty
    label (no scope text, no notes). Even if the LLM incorrectly returns a
    confident SUITABLE, the deterministic safety net must downgrade it to
    UNCLEAR rather than let specialty alone stand as proof."""
    resolved = doctor_result(scope_reference=SCOPE_SPECIALTY_ONLY)
    data, _ = run_scope_node(
        "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور محمد سالم",
        resolved, {"outcome": "SUITABLE", "is_violation": False, "reasoning": "specialty matches"},
    )
    assert data["outcome"] == "UNCLEAR"
    assert data["is_violation"] is False
    assert "specialty" in data["reasoning"].lower() or "scope" in data["reasoning"].lower()


def test_explicit_doctor_note_protects_suitable_from_downgrade():
    """Item 12 — the doctor has no detailed scope_of_service text, but an
    explicit doctor note directly confirms the fit. The note counts as
    real evidence (tier 2, above bare specialty) so SUITABLE must NOT be
    downgraded here, unlike the bare-specialty-only case above."""
    resolved = doctor_result(scope_reference=SCOPE_WITH_NOTES_ONLY)
    data, _ = run_scope_node(
        "Patient: عندي اصابة رياضية في الركبة\nAgent: هوصلك بدكتور محمد سالم",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "SUITABLE"


def test_examination_age_violation_downgrades_false_suitable_to_unsuitable():
    """Item 13 — patient's explicit age falls outside the doctor's
    documented eligibility; even an LLM SUITABLE verdict must be
    overridden to UNSUITABLE."""
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)  # eligibility: 0-16 years
    data, _ = run_scope_node(
        "Patient: عمري 40 سنة وعندي صرع وتشنجات\nAgent: هوصلك بدكتورة خيرية محمد",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "UNSUITABLE"
    assert data["is_violation"] is True


def test_missing_patient_age_does_not_incorrectly_fail():
    """Item 14 — no explicit age stated at all: the age safety net must
    never fire, and a genuine SUITABLE verdict must stand."""
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    data, _ = run_scope_node(
        "Patient: طفلي الصغير عنده صرع وتشنجات\nAgent: هوصلك بدكتورة خيرية محمد",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert data["outcome"] == "SUITABLE"


def test_scope_prompt_uses_exact_resolved_doctor_only():
    """Item 15 & 16 — the doctor_reference handed to the LLM must be built
    solely from the ALREADY-resolved doctor's own scope_reference (a single
    doctor_key), never a pool of candidates the LLM could pick among."""
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    _, stub = run_scope_node(
        "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور رامي سمرقندي",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert stub.last_user_prompt is not None
    assert "700700222" in stub.last_user_prompt  # the exact resolved doctor_key
    assert "700700333" not in stub.last_user_prompt  # a different doctor never leaks in
    assert "700700111" not in stub.last_user_prompt


def test_llm_never_fetches_or_selects_a_doctor(monkeypatch):
    """Item 16 (continued) — infer_doctor_scope_validation must never touch
    the CRM doctor dataset itself; doctor identity comes only from the
    already-resolved doctor_validation state."""
    import app.service_hub.crm_doctors as crm_doctors

    calls = {"count": 0}
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: calls.__setitem__("count", calls["count"] + 1) or [])
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    run_scope_node(
        "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور رامي سمرقندي",
        resolved, {"outcome": "SUITABLE", "is_violation": False},
    )
    assert calls["count"] == 0


# ── Clinical-need detection: ownership/existence phrases are NOT evidence ──
# Real regression — "عندي" alone (an ownership/existence expression, not
# medical evidence) was previously enough on its own to trigger the
# semantic scope check for purely administrative/booking mentions.

NON_CLINICAL_OWNERSHIP_PHRASES = [
    "كان عندي موعد اليوم مع الدكتور",
    "عندي ملف عندكم موجود",
    "أنا ماعندي هويه عندي جواز سفر",
    "عندي حجز بكرة",
    "عندي تأمين",
    "عندي وصفة",
    "عندي اشعة رنين",
    "الدكتور طلب الأشعة",
    "نزلها لي الدكتور",
    "عندي استفسار",
    "عندي موعد",
    "عندي رقم",
    "عندي تحويل",
    "عندي تقرير",
    "عندي نتيجة",
    "عندي موعد بكرة",
    "هل عندكم دكتور؟",
    "ودكتور او دكتوره؟",
    "محتاج دكتور",
    "عايز احجز مع الدكتور",
    "الدكتور موجود؟",
    "عندي موعد مع الدكتور",
]


def test_ownership_phrases_are_not_clinical_need():
    for text in NON_CLINICAL_OWNERSHIP_PHRASES:
        assert patient_describes_medical_complaint(text) is False, text


CLINICAL_NEED_PHRASES = [
    "عندي ألم شديد في الركبة",
    "بعاني من ألم في الظهر",
    "عندي صرع",
    "طفلي عنده صرع وتشنجات متكررة",
    "عندي حصوة في الكلى",
    "عندي قطع في الرباط الصليبي",
    "عندي تمزق في الرباط",
    "عندي التهاب في الأعصاب",
    "عندي مشكلة في الغدة",
    "عندي صداع مستمر",
    "عندي دوخة مستمرة",
    "عندي حساسية شديدة",
    "عندي تأخر حمل",
    "عندي أعراض في القلب",
    "عندي كسر في اليد",
    "طفلي عنده تأخر في النمو",
    "المريض عنده مرض سرطان",
    "I have severe knee pain",
    "I have a torn ACL",
    "My child has epilepsy and seizures",
]


def test_genuine_clinical_phrases_are_detected():
    for text in CLINICAL_NEED_PHRASES:
        assert patient_describes_medical_complaint(text) is True, text


def test_ownership_phrases_inside_realistic_transcript_yield_no_clinical_need():
    """The same 'عندي X' phrases, spoken as actual Patient turns inside a
    realistic multi-turn transcript, must still not be picked up by the
    turn-aware extractor."""
    transcript = "\n".join(f"Patient: {p}" for p in NON_CLINICAL_OWNERSHIP_PHRASES)
    need = extract_patient_clinical_need(call(transcript))
    assert need == ""


# ── Real regression: Mohamed Elalfy reschedule/booking call ────────────────
# Call 4177708C-B17E-F111-B337-000D3AA9D4A7 — the patient reschedules an
# existing appointment with an already-named doctor and separately asks an
# MRI pricing question; neither is a clinical complaint about that doctor.
# Doctor resolution must stay correct (PASS, doctor_key=110685) while scope
# validation must be NOT_APPLICABLE — and, critically, the LLM must never
# even be called.
MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT = (
    "Patient: اختي ممكن اجل حجزي لبكره حصل معي امر طارئ\n"
    "Patient: كان عندي موعد اليوم مع الدكتور\n"
    "Patient: محمد الالفي\n"
    "Patient: وابغى اجل الموعد لبكره\n"
    "Agent: تمام هوضحلك\n"
    "Patient: الدكتور محمد الالفي\n"
    "Agent: نعم هوضحلك\n"
    "Patient: موجود الدكتور\n"
    "Agent: تم تأكيد حجزك بكرة مع د/ ( محمد الالفي )\n"
    "Patient: عندي ملف عندكم موجود\n"
    "Patient: أنا ماعندي هويه عندي جواز سفر أختي أنا زياره\n"
    "Patient: كم سعر الرنين المغناطيسي عندكم لليد\n"
)


def test_mohamed_elalfy_reschedule_call_has_no_clinical_need():
    need = extract_patient_clinical_need(call(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT))
    assert need == ""


def test_mohamed_elalfy_reschedule_scope_validation_skips_without_llm_call():
    resolved = {
        "applicable": True, "outcome": "PASS", "doctor_resolved": True,
        "doctor_key": "110685",
        "scope_reference": {
            "doctor_key": "110685", "doctor_name_ar": "محمد الألفي", "doctor_name_en": "Mohamed Elalfy",
            "business_unit": "AHJ", "specialty": "Orthopedic", "subspecialty": None,
            "scope_of_service": None, "scope_of_service_ar": None,
            "doctor_notes": None, "examination_age": None,
        },
    }
    assert doctor_scope_skip_reason(resolved, extract_patient_clinical_need(call(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT))) == "no_patient_clinical_need"
    # This is now correctly recognised as a genuine doctor-targeted
    # rescheduling action (an active "اجل الموعد" verb inherits the doctor
    # established earlier via "كان عندي موعد مع الدكتور") — doctor
    # validation (identity) legitimately runs and resolves PASS; only
    # scope validation is skipped, because rescheduling your OWN chosen
    # doctor's appointment is not a request to judge that doctor's clinical
    # fit for anything.
    assert classify_doctor_context(call(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT))["doctor_role"] == "patient_selected"

    data, stub = run_scope_node(MOHAMED_ELALFY_RESCHEDULE_TRANSCRIPT, resolved, {"outcome": "SUITABLE"})
    assert data["outcome"] == "NOT_APPLICABLE"
    assert data["applicable"] is False
    assert stub.last_user_prompt is None  # the scope LLM must never be called


# ── skip_doctor_scope_validation node (graph-level skip path) ──────────────

def test_skip_doctor_scope_validation_node_returns_not_applicable_without_node_trace():
    """Mirrors skip_doctor_validation/skip_bank_validation/
    skip_location_validation: the skip node itself must never write a
    node_trace entry (see app.agent.graph._doctor_scope_intent_router)."""
    resolved = {"applicable": True, "outcome": "NOT_APPLICABLE", "doctor_resolved": False, "doctor_key": None, "scope_reference": None}
    state = {"call": call("Patient: عايز اعرف المواعيد\nAgent: تمام"), "doctor_validation": resolved, "node_trace": []}
    result = skip_doctor_scope_validation(state)
    assert result["doctor_scope_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_scope_validation"]["applicable"] is False
    assert "node_trace" not in result


# ── Doctor-to-clinical-need relationship (Condition #4) ─────────────────────
# Real regression call 42714555-0C7E-F111-B337-000D3AA9D4A7 — أحمد أفندي is
# correctly resolved as a real, named doctor (deterministic doctor
# information validation may run), but he is only the ORDERING/referring
# doctor for a CT scan; a completely unrelated kidney-failure complaint
# elsewhere in the same call must never be compared against his scope.

AHMED_AFANDI_ORDERING_TRANSCRIPT = (
    "Patient: الدكتور احمد افندي طلب اشعه مقطعيه ، هل تطلع في التطبيق ؟\n"
    "Patient: دكتور احمد افندي\n"
    "Patient: عندكم\n"
    "Patient: كاش\n"
    "Patient: ابغى اشوف نسخه من request الاشعه\n"
    "Agent: الاشعة المطلوبة لحضرتك هي تصوير الأوعية الدموية بالتصوير المقطعي للذراع\n"
    "Patient: لانه انا عندي غسيل كلى ولازم تكون قبل موعد جلسة الغسيل\n"
    "Patient: لانها بالصبغه فا لازم اغسل بعد الاشعه على طول\n"
    "Patient: انا مريض فشل كلوي لي 15 سنه فتحليل وظائف الكلى مرتفع دائما\n"
)

AHMED_AFANDI_RESOLVED = {
    "applicable": True, "outcome": "PASS", "doctor_resolved": True, "doctor_key": "110432",
    "scope_reference": {
        "doctor_key": "110432", "doctor_name_ar": "أحمد أفندي", "doctor_name_en": "Ahmed Afandi",
        "business_unit": "AHJ", "specialty": "Radiology", "subspecialty": None,
        "scope_of_service": None, "scope_of_service_ar": None,
        "doctor_notes": None, "examination_age": None,
    },
}


def test_ordering_doctor_with_unrelated_condition_skips_scope_validation():
    assert doctor_is_selected_for_clinical_need(call(AHMED_AFANDI_ORDERING_TRANSCRIPT)) is False
    assert classify_doctor_context(call(AHMED_AFANDI_ORDERING_TRANSCRIPT))["doctor_role"] == "ordering_or_referring"
    need = extract_patient_clinical_need(call(AHMED_AFANDI_ORDERING_TRANSCRIPT))
    assert "فشل كلوي" in need  # the clinical need genuinely exists...
    data, stub = run_scope_node(AHMED_AFANDI_ORDERING_TRANSCRIPT, AHMED_AFANDI_RESOLVED, {"outcome": "SUITABLE"})
    assert data["outcome"] == "NOT_APPLICABLE"  # ...but must never be compared to this doctor's scope
    assert data["applicable"] is False
    assert stub.last_user_prompt is None


def test_ordering_doctor_skip_reason_is_specific():
    c = call(AHMED_AFANDI_ORDERING_TRANSCRIPT)
    patient_text = extract_patient_clinical_need(c)
    reason = doctor_scope_skip_reason(AHMED_AFANDI_RESOLVED, patient_text, c)
    assert reason == "doctor_reference_is_ordering_or_referring_context"


@pytest.mark.parametrize("transcript", [
    "Patient: الدكتور احمد طلب الأشعة\nPatient: عندي فشل كلوي مزمن\n",
    "Patient: دكتور محمد كتب لي التحليل\nPatient: عندي فشل كلوي مزمن\n",
    "Patient: الدكتور احمد حولني للأشعة\nPatient: عندي فشل كلوي مزمن\n",
    "Patient: معايا طلب من دكتور أحمد\nPatient: عندي فشل كلوي مزمن\n",
])
def test_ordering_doctor_terminology_variants_do_not_select_doctor_for_need(transcript):
    """The resolved doctor may still be validated deterministically if
    sufficiently named — this only asserts the RELATIONSHIP check, not
    doctor-information validation."""
    assert doctor_is_selected_for_clinical_need(call(transcript)) is False


def test_existing_followup_doctor_does_not_select_for_unrelated_need():
    transcript = "Patient: عندي مراجعة مع دكتور أمير\nPatient: عندي وجع شديد في الظهر\n"
    assert doctor_is_selected_for_clinical_need(call(transcript)) is False


def test_administrative_named_doctor_reference_skips_scope_but_not_info():
    """Item 10 — doctor information validation may run for a purely
    administrative named-doctor mention, but scope validation must not."""
    transcript = "Patient: عايز أغير موعد دكتور محمد سالم\n"
    c = call(transcript)
    assert doctor_is_selected_for_clinical_need(c) is False
    # No clinical need is described either — belt and suspenders, scope is
    # excluded via BOTH Condition #3 and Condition #4.
    resolved = {
        "applicable": True, "outcome": "PASS", "doctor_resolved": True, "doctor_key": "500500088",
        "scope_reference": {"doctor_key": "500500088", "specialty": "Cardiology", "scope_of_service": None, "scope_of_service_ar": None},
    }
    reason = doctor_scope_skip_reason(resolved, extract_patient_clinical_need(c) or "", c)
    assert reason in ("no_patient_clinical_need", "doctor_reference_is_ordering_or_referring_context")


def test_true_recommendation_for_stated_need_still_applies():
    """Item 8 — Agent recommends/books a NEW doctor directly in response to
    a stated clinical need; no ordering marker anywhere disqualifies it."""
    transcript = (
        "Patient: طفلي عنده صرع وتشنجات متكررة\n"
        "Patient: مين دكتور مناسب؟\n"
        "Agent: نحجز مع دكتورة خيرية محمد\n"
    )
    assert doctor_is_selected_for_clinical_need(call(transcript)) is True
    assert classify_doctor_context(call(transcript))["doctor_role"] == "agent_recommended"
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    data, stub = run_scope_node(transcript, resolved, {"outcome": "SUITABLE", "is_violation": False})
    assert data["outcome"] == "SUITABLE"
    assert stub.last_user_prompt is not None


def test_explicit_suitability_question_still_applies():
    """Item 9 — the patient directly asks whether the named doctor suits
    their condition."""
    transcript = "Patient: عندي قطع في الرباط الصليبي\nPatient: هل دكتور رامي مناسب للحالة دي؟\n"
    assert doctor_is_selected_for_clinical_need(call(transcript)) is True
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    data, stub = run_scope_node(transcript, resolved, {"outcome": "SUITABLE", "is_violation": False})
    assert data["outcome"] == "SUITABLE"
    assert stub.last_user_prompt is not None


# ── classify_doctor_context — patient-selected vs agent-recommended ─────────
# The required decision-flow refactor: scope validation must distinguish a
# patient choosing their own doctor from an agent recommending/selecting
# one on the patient's behalf, not merely "doctor mentioned + medical words
# found somewhere in the transcript".

def test_case1_patient_names_doctor_with_separate_unrelated_disease_mention():
    """A patient who books a SPECIFIC doctor by name, and separately
    mentions an unrelated condition elsewhere, must NOT imply a suitability
    judgment for that condition — this was the residual gap the role-based
    default closes (real spec example: 'ابغى احجز دكتور أحمد' + 'أنا مريض
    سكر' must never imply 'is Dr Ahmed suitable for diabetes?')."""
    transcript = "Patient: ابغى احجز دكتور أحمد\nPatient: أنا مريض سكر\n"
    ctx = classify_doctor_context(call(transcript))
    assert ctx["doctor_role"] == "patient_selected"
    assert ctx["scope_applicable"] is False


def test_case1_explicit_named_doctor_booking_no_clinical_need():
    transcript = "Patient: أبغى أحجز مع الدكتور محمد الألفي بكرة\n"
    resolved = {
        "applicable": True, "outcome": "PASS", "doctor_resolved": True, "doctor_key": "110685",
        "scope_reference": {"doctor_key": "110685", "specialty": "Orthopedic", "scope_of_service": None, "scope_of_service_ar": None},
    }
    reason = doctor_scope_skip_reason(resolved, extract_patient_clinical_need(call(transcript)) or "", call(transcript))
    assert reason in ("no_patient_clinical_need", "doctor_reference_is_ordering_or_referring_context")


def test_case2_patient_names_doctor_and_asks_suitability_question():
    """The patient picks a specific doctor by name AND explicitly asks
    whether that doctor suits their condition — this refines the base
    'patient_selected' role to 'doctor_inquiry' (a clinical suitability
    question asked about an already-named doctor), and scope validation
    must run."""
    transcript = (
        "Patient: عندي إصابة في الرباط الصليبي\n"
        "Patient: هل دكتور رامي مناسب؟\n"
        "Patient: لو مناسب أبغى أحجز معه\n"
    )
    ctx = classify_doctor_context(call(transcript))
    assert ctx["doctor_role"] == "doctor_inquiry"
    assert ctx["scope_applicable"] is True
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    data, stub = run_scope_node(transcript, resolved, {"outcome": "SUITABLE", "is_violation": False})
    assert data["outcome"] == "SUITABLE"
    assert stub.last_user_prompt is not None


def test_case5_non_booking_explicit_clinical_scope_inquiry():
    """A pure information inquiry (no booking language at all) that
    directly asks a treatment-capability question must still run scope
    validation."""
    transcript = "Patient: هل دكتورة خيرية محمد تعالج الصرع عند الأطفال؟\n"
    ctx = classify_doctor_context(call(transcript))
    assert ctx["doctor_role"] == "doctor_inquiry"
    assert ctx["scope_applicable"] is True
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    data, stub = run_scope_node(transcript, resolved, {"outcome": "SUITABLE", "is_violation": False})
    assert data["outcome"] == "SUITABLE"
    assert stub.last_user_prompt is not None


def test_case6_non_booking_administrative_inquiry_skips_scope():
    """A purely administrative availability question about a named doctor
    must validate doctor identity but never run scope suitability."""
    transcript = "Patient: دكتور محمد موجود بكرة؟\n"
    ctx = classify_doctor_context(call(transcript))
    assert ctx["doctor_role"] == "administrative_reference"
    assert ctx["scope_applicable"] is False
    resolved = {
        "applicable": True, "outcome": "PASS", "doctor_resolved": True, "doctor_key": "500500088",
        "scope_reference": {"doctor_key": "500500088", "specialty": "Cardiology", "scope_of_service": None, "scope_of_service_ar": None},
    }
    data, stub = run_scope_node(transcript, resolved, {"outcome": "SUITABLE"})
    assert data["outcome"] == "NOT_APPLICABLE"
    assert stub.last_user_prompt is None


# ── Malformed/empty LLM response normalization ──────────────────────────────

def test_malformed_empty_llm_response_normalizes_to_safe_unclear():
    """Item 13 — when scope validation IS applicable but the LLM returns an
    empty/malformed JSON object, the node must never leak outcome=None/
    applicable=None into state; it must normalize to a safe, non-punitive
    UNCLEAR result."""
    resolved = doctor_result(scope_reference=SCOPE_EPILEPSY)
    transcript = "Patient: طفلي عنده صرع وتشنجات متكررة\nAgent: هوصلك بدكتورة خيرية محمد\n"
    data, stub = run_scope_node(transcript, resolved, {})
    assert data["outcome"] == "UNCLEAR"
    assert data["applicable"] is True
    assert data["is_violation"] is False
    assert data["reasoning"]
    assert data["patient_need_summary"] is None
    assert data["doctor_scope_summary"] is None
    assert data["matched_scope_evidence"] == []


def test_malformed_response_with_partial_fields_still_normalized():
    """A response missing only some fields (but with a valid outcome) must
    keep that outcome and just fill in the rest safely."""
    resolved = doctor_result(scope_reference=SCOPE_ACL_SPORTS)
    transcript = "Patient: عندي قطع في الرباط الصليبي\nAgent: هوصلك بدكتور رامي سمرقندي\n"
    data, stub = run_scope_node(transcript, resolved, {"outcome": "UNSUITABLE"})
    assert data["outcome"] == "UNSUITABLE"
    assert data["is_violation"] is True  # inferred from outcome when not explicitly provided
    assert data["applicable"] is True


# ── Multi-doctor recommendation sets ────────────────────────────────────────
# Regression: a recommendation naming several doctors for the SAME clinical
# need must get an INDEPENDENT scope verdict per resolved doctor — never one
# LLM call applied to all of them, and never the first doctor's verdict
# standing in for the whole set. See validate_doctor_information's "doctors"
# list (app.service_hub.doctor_validation) and infer_doctor_scope_
# validation's multi-doctor path (app.agent.nodes).

class _StubLLMPerDoctor:
    """Varies its response based on which doctor's scope evidence appears
    in the prompt — lets a test assert each recommended doctor gets its
    OWN independent LLM judgment, not a single shared one."""

    def __init__(self, responses_by_marker: dict[str, dict]):
        self._responses = responses_by_marker
        self.prompts: list[str] = []

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        self.prompts.append(user_prompt)
        for marker, response in self._responses.items():
            if marker in user_prompt:
                return json.dumps(response, ensure_ascii=False), {"prompt_tokens": 0, "completion_tokens": 0}
        return json.dumps({"outcome": "UNCLEAR"}), {"prompt_tokens": 0, "completion_tokens": 0}


def multi_doctor_result(entries: list[dict]) -> dict:
    """Minimal stand-in for validate_doctor_information()'s multi-doctor
    output shape ("doctors" list) — only the fields doctor_scope_skip_
    reason()/infer_doctor_scope_validation() actually read."""
    first = entries[0]
    return {
        "applicable": True, "outcome": first["outcome"], "doctor_resolved": first["doctor_resolved"],
        "doctor_key": (first.get("scope_reference") or {}).get("doctor_key"),
        "scope_reference": first.get("scope_reference"),
        "doctors": entries,
        "recommended_doctor_count": len(entries),
    }


def test_multi_doctor_scope_gives_each_doctor_an_independent_verdict():
    entries = [
        {"input_name": "rami", "doctor_resolved": True, "outcome": "PASS", "scope_reference": SCOPE_ACL_SPORTS},
        {"input_name": "farid", "doctor_resolved": True, "outcome": "PASS", "scope_reference": SCOPE_SPINE_ONLY},
    ]
    dr = multi_doctor_result(entries)
    transcript = "Patient: عندي قطع في الرباط الصليبي من اصابة رياضية\nAgent: هرشحلك دكتور رامي او دكتور فريد\n"
    c = call(transcript)
    stub = _StubLLMPerDoctor({
        "700700222": {"outcome": "SUITABLE", "reasoning": "Sports/ACL matches."},
        "700700333": {"outcome": "UNSUITABLE", "reasoning": "Spine surgeon does not treat ACL injuries."},
    })
    state = {"call": c, "doctor_validation": dr, "node_trace": []}
    result = asyncio.run(infer_doctor_scope_validation(state, stub))
    scope = result["doctor_scope_validation"]

    # Both doctors were actually asked about independently — not one shared call.
    assert len(stub.prompts) == 2
    by_key = {d["doctor_key"]: d for d in scope["doctors"]}
    assert by_key["700700222"]["outcome"] == "SUITABLE"
    assert by_key["700700333"]["outcome"] == "UNSUITABLE"
    # One UNSUITABLE doctor must be reflected in the aggregate — never
    # masked by the other doctor's SUITABLE verdict.
    assert scope["outcome"] == "UNSUITABLE"
    assert scope["is_violation"] is True
    # Exactly one graph-node-level trace entry regardless of how many
    # doctors were internally evaluated.
    assert result["node_trace"] == ["infer_doctor_scope_validation"]


def test_multi_doctor_scope_all_suitable_aggregates_to_suitable():
    entries = [
        {"input_name": "rami", "doctor_resolved": True, "outcome": "PASS", "scope_reference": SCOPE_ACL_SPORTS},
        {"input_name": "khairya", "doctor_resolved": True, "outcome": "PASS", "scope_reference": SCOPE_EPILEPSY},
    ]
    dr = multi_doctor_result(entries)
    transcript = "Patient: طفلي عنده صرع\nAgent: هرشحلك دكتور رامي او دكتورة خيرية\n"
    c = call(transcript)
    stub = _StubLLMPerDoctor({
        "700700222": {"outcome": "SUITABLE"},
        "700700111": {"outcome": "SUITABLE"},
    })
    state = {"call": c, "doctor_validation": dr, "node_trace": []}
    result = asyncio.run(infer_doctor_scope_validation(state, stub))
    scope = result["doctor_scope_validation"]
    assert scope["outcome"] == "SUITABLE"
    assert scope["is_violation"] is False
    assert len(scope["doctors"]) == 2


def test_multi_doctor_scope_unresolved_doctor_stays_explicit_not_dropped():
    """An unresolved doctor in the recommendation set must still appear in
    the scope 'doctors' list (as NOT_APPLICABLE), never silently omitted,
    while the resolved doctor still gets a real, independent verdict."""
    entries = [
        {"input_name": "rami", "doctor_resolved": True, "outcome": "PASS", "scope_reference": SCOPE_ACL_SPORTS},
        {"input_name": "unknown doctor", "doctor_resolved": False, "outcome": "DOCTOR_UNRESOLVED", "scope_reference": None},
    ]
    dr = multi_doctor_result(entries)
    transcript = "Patient: عندي قطع في الرباط الصليبي\nAgent: هرشحلك دكتور رامي او دكتور تاني\n"
    c = call(transcript)
    stub = _StubLLMPerDoctor({"700700222": {"outcome": "SUITABLE"}})
    state = {"call": c, "doctor_validation": dr, "node_trace": []}
    result = asyncio.run(infer_doctor_scope_validation(state, stub))
    scope = result["doctor_scope_validation"]
    assert len(scope["doctors"]) == 2
    names = {d.get("input_name") for d in scope["doctors"]}
    assert "unknown doctor" in names
    unresolved_entry = next(d for d in scope["doctors"] if d.get("input_name") == "unknown doctor")
    assert unresolved_entry["outcome"] == "NOT_APPLICABLE"
    resolved_entry = next(d for d in scope["doctors"] if d.get("doctor_key") == "700700222")
    assert resolved_entry["outcome"] == "SUITABLE"
