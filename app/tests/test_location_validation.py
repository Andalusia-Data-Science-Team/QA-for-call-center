import asyncio

import pytest

from app.models.input import CallTranscript
from app.service_hub.location_validation import (
    detect_location_intent,
    detect_location_request,
    location_validation_needed,
    resolve_branch_candidates,
    validate_location_request,
)

# Real branch examples from the KSA location dataset (Step 5 of the spec).
LOC_SULTAN = {
    "cr301_branchname": "عيادات أندلسية فرع الأمير سلطان",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "شارع الأمير سلطان - حي المحمدية - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_SANABEL = {
    "cr301_branchname": "عيادات أندلسية فرع السنابل",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "شارع السنابل العام بعد محطه نفط - حي السنابل - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_SARI = {
    "cr301_branchname": "عيادات أندلسية فرع صاري",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "شارع صارى متفرع من الخطيب التبريزى - حي البوادي - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_DENTAL_SULTAN = {
    "cr301_branchname": "عيادات أندلسية لطب الأسنان - فرع الأمير سلطان",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "شارع الأمير سلطان - حي المحمدية - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_MACARONA = {
    "cr301_branchname": "عيادات أندلسية للأسنان فرع المكرونة",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "المكرونة - حي مشرفة - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_HOSPITAL = {
    "cr301_branchname": "مستشفى أندلسية جدة",
    "cr301_area": "جدة", "cr301_country": "السعودية",
    "cr301_description": "تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة",
    "cr18c_region": "KSA", "statecodename": "Active",
}
LOC_CAIRO = {
    "cr301_branchname": "فرع القاهرة", "cr301_area": "القاهرة", "cr301_country": "مصر",
    "cr301_description": "شارع التحرير - القاهرة", "cr18c_region": "EGY", "statecodename": "Active",
}
ALL_LOCATIONS = [LOC_SULTAN, LOC_SANABEL, LOC_SARI, LOC_DENTAL_SULTAN, LOC_MACARONA, LOC_HOSPITAL, LOC_CAIRO]


def call(transcript):
    return CallTranscript(call_id="loc-1", agent_name="Agent", Patient_Phone="501234567", call_date="2026-08-24", call_duration_seconds=1, department="Scheduling", transcript=transcript)


def _turns(transcript: str) -> tuple[str, str]:
    """Split a "Patient: .. / Agent: .." transcript fixture into
    (patient_text, agent_text) — mirrors bank_validation._turns() closely
    enough for these tests without importing a private helper across modules."""
    patient_lines, agent_lines = [], []
    for line in transcript.splitlines():
        if line.startswith("Patient:"):
            patient_lines.append(line.split(":", 1)[1].strip())
        elif line.startswith("Agent:"):
            agent_lines.append(line.split(":", 1)[1].strip())
    return "\n".join(patient_lines), "\n".join(agent_lines)


def validate(transcript: str, locations=ALL_LOCATIONS):
    patient_text, agent_text = _turns(transcript)
    return validate_location_request(call(transcript), locations, patient_text=patient_text, agent_text=agent_text)


def test_detect_location_request_context_aware():
    """Step 17: bare 'فرع'/'مكان' without request context must not trigger."""
    assert detect_location_request("فين فرع الأمير سلطان؟")
    assert detect_location_request("عنوان فرع صاري")
    assert detect_location_request("ممكن لوكيشن الفرع؟")
    assert not detect_location_request("أنا في الفرع دلوقتي")  # mentions فرع, no request
    assert not detect_location_request("عندي مكان فاضي بكرة الساعة 5")  # مكان, no request


def test_correct_exact_location_passes():
    result = validate("Patient: وين فرع الأمير سلطان؟\nAgent: شارع الأمير سلطان حي المحمدية جدة")
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"


def test_minor_formatting_differences_still_pass():
    """Dashes/hamza differences between DB and agent phrasing must not fail."""
    result = validate("Patient: عنوان فرع الأمير سلطان؟\nAgent: شارع الامير سلطان - حي المحمديه - جده")
    assert result["outcome"] == "PASS"


def test_wrong_branch_address_fails():
    result = validate("Patient: فين فرع الأمير سلطان؟\nAgent: شارع صاري حي البوادي")
    assert result["outcome"] == "WRONG_LOCATION"
    assert result["is_violation"] is True


def test_no_address_provided_fails():
    result = validate("Patient: عنوان فرع صاري؟\nAgent: لحظة من فضلك")
    assert result["outcome"] == "NO_ADDRESS_PROVIDED"


def test_area_only_answer_is_incomplete_not_pass():
    """'بس في جدة' — every branch here is in جدة, so that alone isn't specific
    enough to count as the correct address (ubiquitous tokens are filtered)."""
    result = validate("Patient: فين فرع السنابل؟\nAgent: في جدة")
    assert result["outcome"] != "PASS"


def test_correct_but_incomplete_address_is_partial():
    """Names the street but omits the district — partial credit, not a hard fail."""
    result = validate("Patient: عنوان فرع صاري؟\nAgent: شارع صارى")
    assert result["outcome"] == "INCOMPLETE_ADDRESS"


def test_dental_vs_non_dental_branch_disambiguated_by_context():
    """Two branches share the exact same street address text; mentioning
    'الأسنان' should resolve to the dental one specifically."""
    result = validate("Patient: فين عيادة الأسنان فرع الأمير سلطان؟\nAgent: شارع الأمير سلطان حي المحمدية جدة")
    assert result["requested_branch"] == "عيادات أندلسية لطب الأسنان - فرع الأمير سلطان"
    assert result["outcome"] == "PASS"


def test_ambiguous_when_multiple_records_tie():
    """Synthetic duplicate CRM records — must not guess between them."""
    dup_a = {"cr301_branchname": "فرع تجريبي", "cr301_area": "جدة", "cr301_country": "السعودية", "statecodename": "Active"}
    dup_b = {"cr301_branchname": "فرع تجريبي", "cr301_area": "الرياض", "cr301_country": "السعودية", "statecodename": "Active"}
    candidates = resolve_branch_candidates("فين فرع تجريبي؟", [dup_a, dup_b])
    assert len(candidates) == 2


def test_egypt_branch_not_resolved_ksa_only():
    candidates = resolve_branch_candidates("فين فرع القاهرة؟", ALL_LOCATIONS, ksa_only=True)
    assert candidates == []


def test_location_and_bank_request_in_same_chat():
    """Both a location request and a bank request appear in one transcript —
    the location side still resolves and validates independently (the bank
    side of this same scenario is covered in test_bank_validation.py)."""
    transcript = (
        "Patient: فين فرع الأمير سلطان؟\n"
        "Agent: شارع الأمير سلطان حي المحمدية جدة\n"
        "Patient: تمام، ممكن رقم الحساب؟\n"
        "Agent: هنبعتلك بعدين"
    )
    assert detect_location_request("فين فرع الأمير سلطان؟")
    result = validate(transcript)
    assert result["outcome"] == "PASS"


# ── Applicability ─────────────────────────────────────────────────────────────

def test_branch_mentioned_for_booking_only_is_not_applicable():
    result = validate("Patient: عايز احجز كشف في فرع السنابل\nAgent: متاح الساعة 6")
    assert result["outcome"] == "NOT_APPLICABLE"


def test_patient_recalling_a_past_branch_visit_is_not_applicable():
    """Mentioning a branch in passing ('I was there before') must not trigger
    validation just because the agent then says something unrelated."""
    result = validate("Patient: أنا كنت في فرع السنابل قبل كده\nAgent: تمام")
    assert result["outcome"] == "NOT_APPLICABLE"


def test_no_location_or_branch_discussion_at_all_is_not_applicable():
    result = validate("Patient: عايز اسأل عن الأسعار\nAgent: تحت أمرك")
    assert result["outcome"] == "NOT_APPLICABLE"


def test_generic_area_word_alone_does_not_resolve_a_specific_branch():
    """'جدة' alone is ubiquitous across every current KSA row — it must not
    resolve to any one specific branch on its own."""
    assert resolve_branch_candidates("جدة", ALL_LOCATIONS) == []


# ── Validation — PASS for every known branch ────────────────────────────────

def test_correct_address_passes_for_every_branch():
    cases = [
        ("Patient: فين فرع السنابل؟\nAgent: شارع السنابل العام بعد محطة نفط حي السنابل جدة", "عيادات أندلسية فرع السنابل"),
        ("Patient: عنوان فرع صاري؟\nAgent: شارع صارى متفرع من الخطيب التبريزي حي البوادي جدة", "عيادات أندلسية فرع صاري"),
        ("Patient: فين فرع المكرونة؟\nAgent: المكرونة حي مشرفة جدة", "عيادات أندلسية للأسنان فرع المكرونة"),
        ("Patient: فين مستشفى أندلسية جدة؟\nAgent: تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا جدة", "مستشفى أندلسية جدة"),
    ]
    for transcript, expected_branch in cases:
        result = validate(transcript)
        assert result["outcome"] == "PASS", f"expected PASS for {expected_branch!r}, got {result}"
        assert result["requested_branch"] == expected_branch


def test_country_only_answer_is_insufficient():
    result = validate("Patient: فين فرع صاري؟\nAgent: في السعودية")
    assert result["outcome"] != "PASS"
    assert result["is_violation"] is True


def test_partial_but_distinctive_address_passes():
    """Enough distinctive detail (street + district), even without the
    trailing city name, should still be treated as sufficient."""
    result = validate("Patient: فين فرع السنابل؟\nAgent: شارع السنابل العام بعد محطة نفط حي السنابل")
    assert result["outcome"] == "PASS"


def test_patient_request_answered_with_only_branch_name_is_insufficient():
    result = validate("Patient: فين فرع السنابل؟\nAgent: فرع السنابل في جدة")
    assert result["outcome"] != "PASS"
    assert result["is_violation"] is True


# ── Trigger B — agent proactively provides a location ───────────────────────

def test_proactive_agent_address_is_applicable_and_passes():
    """No patient request at all — the agent volunteers a correct address."""
    transcript = "Patient: تمام الحجز مناسب\nAgent: العنوان فرع السنابل شارع السنابل العام بعد محطة نفط، حي السنابل، جدة"
    assert not detect_location_request("تمام الحجز مناسب")
    assert location_validation_needed(call(transcript))
    result = validate(transcript)
    assert result["outcome"] == "PASS"
    assert result["request_detected"] is False
    assert result["applicable"] is True


def test_proactive_agent_wrong_address_fails():
    """The agent names the RIGHT branch (السنابل) but describes a DIFFERENT
    branch's street — must resolve to the branch actually named (not get
    hijacked by the wrong street's overlapping words) and must fail either
    way (WRONG_LOCATION or INCOMPLETE_ADDRESS are both valid classifications
    for "named correctly, address doesn't hold up" — the precise boundary
    isn't the point, failing is)."""
    transcript = "Patient: تمام\nAgent: فرع السنابل موجود في شارع الأمير سلطان حي المحمدية"
    result = validate(transcript)
    assert result["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert result["outcome"] != "PASS"
    assert result["is_violation"] is True
    assert result["request_detected"] is False


def test_proactive_branch_name_without_address_is_not_applicable():
    """Naming a branch with no address, and no patient request either, is
    not enough to trigger — Trigger B requires an actual address, not just
    a place-noun mention."""
    transcript = "Patient: تمام\nAgent: هنكمل معاك في فرع السنابل"
    result = validate(transcript)
    assert result["outcome"] == "NOT_APPLICABLE"


def test_proactive_address_via_full_transcript_fallback():
    """The branch name only appears in the AGENT's own turn (not the
    patient's) — resolution must still work via the full-transcript fallback."""
    transcript = "Patient: تمام، الحجز مناسب\nAgent: تمام، الفرع في شارع الأمير سلطان حي المحمدية في جدة"
    result = validate(transcript)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"


# ── KSA scope ────────────────────────────────────────────────────────────────

def test_egypt_branch_request_is_non_punitive():
    result = validate("Patient: فين فرع القاهرة؟\nAgent: شارع التحرير القاهرة")
    assert result["is_violation"] is False
    assert result["outcome"] in ("BRANCH_UNRESOLVED", "NOT_APPLICABLE")


def test_ksa_region_field_records_are_included():
    candidates = resolve_branch_candidates("فين فرع السنابل؟", ALL_LOCATIONS, ksa_only=True)
    assert len(candidates) == 1
    assert candidates[0]["cr18c_region"] == "KSA"


# ── Infrastructure ───────────────────────────────────────────────────────────

def test_crm_unavailable_is_not_an_agent_violation():
    """Empty locations (as if the CRM fetch failed/returned nothing) must
    degrade to a system-data outcome, never a compliance violation."""
    result = validate("Patient: فين فرع السنابل؟\nAgent: شارع السنابل حي السنابل", locations=[])
    assert result["outcome"] == "BRANCH_UNRESOLVED"
    assert result["is_violation"] is False


def test_ambiguous_branches_sharing_identical_address_still_validate():
    """Step 8: if every tied candidate shares the exact same authoritative
    address, validation proceeds with it instead of refusing as ambiguous."""
    dup_a = {"cr301_branchname": "فرع تجريبي أ", "cr301_area": "جدة", "cr301_country": "السعودية",
             "cr301_description": "شارع تجريبي مشترك حي تجريبي جدة", "cr18c_region": "KSA", "statecodename": "Active"}
    dup_b = {"cr301_branchname": "فرع تجريبي ب", "cr301_area": "جدة", "cr301_country": "السعودية",
             "cr301_description": "شارع تجريبي مشترك حي تجريبي جدة", "cr18c_region": "KSA", "statecodename": "Active"}
    result = validate("Patient: فين فرع تجريبي؟\nAgent: شارع تجريبي مشترك حي تجريبي جدة", locations=[dup_a, dup_b])
    assert result["outcome"] == "PASS"


# ── Location Intent Detection Layer ─────────────────────────────────────────
# The dedicated gate (detect_location_intent) that must run BEFORE any CRM
# lookup, branch resolution, or address matching. See app.agent.nodes.
# validate_location_node, which now logs "location intent | ..." on every
# call and skips straight to NOT_APPLICABLE — no CRM fetch at all — when
# neither side shows intent.

def test_intent_patient_asks_for_branch_location():
    patient_intent, _agent_intent = detect_location_intent("فين فرع السنابل؟", "")
    assert patient_intent is True


def test_intent_patient_asks_for_branch_address():
    patient_intent, _agent_intent = detect_location_intent("ممكن عنوان فرع صاري؟", "")
    assert patient_intent is True


def test_intent_patient_asks_for_branch_moqe():
    patient_intent, _agent_intent = detect_location_intent("وين موقع الأمير سلطان؟", "")
    assert patient_intent is True


def test_intent_agent_proactively_gives_full_address():
    _patient_intent, agent_intent = detect_location_intent(
        "تمام الحجز مناسب",
        "العنوان فرع السنابل شارع السنابل العام بعد محطة نفط حي السنابل جدة",
    )
    assert agent_intent is True


def test_intent_agent_sends_location_with_branch_context():
    _patient_intent, agent_intent = detect_location_intent("تمام", "تم إرسال الموقع الخاص بفرع السنابل")
    assert agent_intent is True


def test_intent_booking_request_mentioning_branch_only_is_false():
    patient_intent, agent_intent = detect_location_intent(
        "عايز احجز في فرع السنابل", "تمام هحجزلك دلوقتي",
    )
    assert patient_intent is False
    assert agent_intent is False


def test_intent_doctor_question_mentioning_branch_only_is_false():
    patient_intent, agent_intent = detect_location_intent(
        "فيه دكتور نساء متاح في فرع صاري؟", "أيوه متاح بكرة الساعة 5",
    )
    assert patient_intent is False
    assert agent_intent is False


def test_intent_generic_makan_alone_is_false():
    """'مكان' is deliberately a WEAK signal (too overloaded colloquially —
    e.g. 'عندي مكان فاضي' = 'I have a free slot') and must not trigger
    without a place-noun ('فرع'/'عيادة'/'مستشفى') alongside it."""
    patient_intent, _agent_intent = detect_location_intent("في مكان فاضي بكرة؟", "")
    assert patient_intent is False


def test_intent_no_location_vocabulary_is_false():
    patient_intent, agent_intent = detect_location_intent(
        "عايز أأكد الحجز بتاعي", "تمام تم التأكيد",
    )
    assert patient_intent is False
    assert agent_intent is False


def test_intent_false_skips_crm_fetch_entirely(monkeypatch):
    """When neither side shows location intent, fetch_ksa_locations() must
    never even be called — no CRM lookup, no branch resolution, no address
    matching."""
    import app.service_hub.crm_location as crm_location
    from app.agent.nodes import validate_location_node

    calls = {"count": 0}

    def _record_and_fail(*_args, **_kwargs):
        calls["count"] += 1
        return []

    monkeypatch.setattr(crm_location, "fetch_ksa_locations", _record_and_fail)

    transcript = call("Patient: عايز أأكد الحجز بتاعي\nAgent: تمام تم التأكيد")
    result = asyncio.run(validate_location_node({"call": transcript}))

    assert calls["count"] == 0
    assert result["location_validation"]["outcome"] == "NOT_APPLICABLE"
    assert result["location_validation"]["applicable"] is False


def test_intent_true_still_runs_full_validation_flow(monkeypatch):
    """When intent IS present, the existing CRM-backed validation flow still
    runs exactly as before — the intent layer is a gate in front of it, not
    a replacement for it."""
    import app.service_hub.crm_location as crm_location
    from app.agent.nodes import validate_location_node

    monkeypatch.setattr(crm_location, "fetch_ksa_locations", lambda *a, **k: ALL_LOCATIONS)

    transcript = call("Patient: فين فرع السنابل؟\nAgent: شارع السنابل العام بعد محطة نفط حي السنابل")
    result = asyncio.run(validate_location_node({"call": transcript}))

    assert result["location_validation"]["outcome"] == "PASS"
    assert result["location_validation"]["requested_branch"] == "عيادات أندلسية فرع السنابل"


def test_intent_layer_does_not_affect_bank_validation():
    """The location-intent layer only touches location_validation.py — bank
    validation, an independent sibling feature (same app.service_hub
    package), must behave identically."""
    from app.service_hub.bank_validation import bank_validation_needed

    transcript = call("Patient: فين فرع السنابل؟\nAgent: العنوان فرع السنابل شارع السنابل حي السنابل")
    assert bank_validation_needed(transcript) is False


# ── CRM/chat location exposed on the result (debug visibility) ─────────────
# The authoritative CRM record and the chat-extracted location must be
# readable from the returned dict itself, not just console/log output.

def test_crm_location_and_provided_location_are_exposed_on_pass():
    result = validate("Patient: ممكن أعرف موقع عيادات أندلسية فرع الأمير سلطان؟\nAgent: العنوان: شارع الأمير سلطان، حي المحمدية، جدة")
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"
    assert result["crm_location"] == LOC_SULTAN["cr301_description"]
    assert result["provided_location"] == "العنوان: شارع الأمير سلطان، حي المحمدية، جدة"


def test_crm_location_reflects_the_resolved_branch_not_the_wrong_one():
    """Section 4/6: an agent giving a real-but-WRONG branch's address must
    have crm_location reflect the branch that was actually ASKED about
    (السنابل), never the branch whose address happens to appear in the
    reply (الأمير سلطان) — the authoritative direction is always
    resolved branch → its own CRM location → compare against chat text."""
    result = validate("Patient: فين فرع السنابل؟\nAgent: العنوان شارع الأمير سلطان، حي المحمدية، جدة")
    assert result["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert result["crm_location"] == LOC_SANABEL["cr301_description"]
    assert result["outcome"] == "WRONG_LOCATION"
    assert result["is_violation"] is True


def test_provided_location_extracted_for_proactive_agent_share():
    transcript = "Patient: تمام الحجز مناسب\nAgent: العنوان فرع السنابل شارع السنابل العام بعد محطة نفط، حي السنابل، جدة"
    result = validate(transcript)
    _patient_text, agent_text = _turns(transcript)
    assert result["provided_location"] == agent_text.strip()
    assert result["outcome"] == "PASS"


def test_not_applicable_and_branch_unresolved_still_expose_null_location_fields():
    """Schema is consistent across every outcome — crm_location/
    provided_location are always present keys, even when None."""
    not_applicable = validate("Patient: تمام شكرا\nAgent: تمام مع السلامة")
    assert not_applicable["outcome"] == "NOT_APPLICABLE"
    assert not_applicable["crm_location"] is None
    assert not_applicable["provided_location"] is None

    unresolved = validate("Patient: فين الفرع اللي في مكان تاني خالص؟\nAgent: العنوان مش معايا دلوقتي", locations=[])
    assert unresolved["outcome"] == "BRANCH_UNRESOLVED"
    assert unresolved["crm_location"] is None


# ── Local-context extraction refinements ────────────────────────────────────
# provided_location must be the location-relevant Agent turn(s) actually
# tied to the request/trigger, never the whole concatenated Agent transcript
# and never a coincidental real CRM address mentioned in an unrelated turn.

def test_entire_agent_transcript_is_not_returned_as_provided_location():
    """Section 1/9.1: irrelevant surrounding Agent turns (branch-only note,
    an unrelated prescription remark) must not leak into provided_location."""
    transcript = (
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
        "Agent: يلزم وصفة طبية"
    )
    result = validate(transcript)
    assert result["provided_location"] != _turns(transcript)[1]
    assert "يلزم وصفة طبية" not in result["provided_location"]


def test_only_location_related_agent_turns_are_extracted():
    """Section 9.2: the extracted text is exactly the location-carrying
    turn(s) — the branch-context line plus the address line, not more."""
    transcript = (
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
        "Agent: يلزم وصفة طبية"
    )
    result = validate(transcript)
    assert result["provided_location"] == (
        "في فرع المستشفى الرئيسي فقط\n"
        "العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )


def test_hospital_address_resolves_to_hospital_branch():
    """Section 9.3."""
    transcript = (
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript)
    assert result["requested_branch"] == "مستشفى أندلسية جدة"


def test_muhammadiyah_alone_does_not_resolve_to_hospital():
    """Section 9.4/2: 'المحمدية' belongs to Prince Sultan's ADDRESS, not the
    hospital's name or address — it must never end up as resolution
    evidence for the hospital branch."""
    candidates = resolve_branch_candidates("المحمدية", ALL_LOCATIONS, ksa_only=True)
    resolved_names = {c["cr301_branchname"] for c in candidates}
    assert "مستشفى أندلسية جدة" not in resolved_names


def test_muhammadiyah_plus_sultan_resolves_to_prince_sultan_branch():
    """Section 9.5."""
    result = validate("Patient: عنوان فرع الأمير سلطان؟\nAgent: العنوان شارع الأمير سلطان حي المحمدية جدة")
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"
    assert result["outcome"] == "PASS"


def test_correct_hospital_address_passes():
    """Section 9.6, matching the spec's exact worked example (Section 5)."""
    transcript = (
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert result["crm_location"] == LOC_HOSPITAL["cr301_description"]
    assert result["provided_location"] == (
        "في فرع المستشفى الرئيسي فقط\n"
        "العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    assert result["is_violation"] is False


def test_correct_prince_sultan_address_passes():
    """Section 9.7."""
    result = validate("Patient: ممكن أعرف موقع عيادات أندلسية فرع الأمير سلطان؟\nAgent: العنوان: شارع الأمير سلطان، حي المحمدية، جدة")
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"


def test_wrong_branchs_valid_address_fails():
    """Section 9.8, the spec's exact wrong-branch worked example (Section 6)."""
    result = validate("Patient: فين فرع السنابل؟\nAgent: العنوان شارع الأمير سلطان، حي المحمدية، جدة")
    assert result["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert result["crm_location"] == LOC_SANABEL["cr301_description"]
    assert result["is_violation"] is True


def test_correct_crm_address_elsewhere_in_transcript_cannot_cause_false_pass():
    """Section 6/9.9: the actual (wrong) answer sits right after the
    request; a real CRM address mentioned in an unrelated LATER Agent turn
    must not be picked up and must not flip this to PASS."""
    transcript = (
        "Patient: فين فرع السنابل؟\n"
        "Agent: فرع السنابل في شارع الأمير سلطان حي المحمدية\n"
        "Patient: تمام شكرا\n"
        "Agent: تحت أمرك\n"
        "Agent: مستشفى أندلسية جدة موجود أمام مول الجامعة بلازا"
    )
    result = validate(transcript)
    assert result["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert result["provided_location"] == "فرع السنابل في شارع الأمير سلطان حي المحمدية"
    assert "مول الجامعة بلازا" not in result["provided_location"]
    assert result["is_violation"] is True


def test_consecutive_location_related_agent_messages_are_grouped():
    """Section 9.10, matching Section 7's proactive-grouping example."""
    transcript = (
        "Agent: هتروحي فرع المستشفى الرئيسي\n"
        "Agent: العنوان تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript)
    assert result["outcome"] == "PASS"
    assert "هتروحي فرع المستشفى الرئيسي" in result["provided_location"]
    assert "العنوان تقاطع" in result["provided_location"]


def test_unrelated_agent_messages_are_excluded_from_comparison():
    """Section 9.11: greetings/closing/booking remarks around the real
    address turn never enter provided_location."""
    transcript = (
        "Agent: أهلاً بيك معانا\n"
        "Agent: العنوان شارع الأمير سلطان حي المحمدية جدة\n"
        "Agent: تحت أمرك في أي وقت"
    )
    result = validate(transcript)
    for noise in ("أهلاً بيك معانا", "تحت أمرك في أي وقت"):
        assert noise not in result["provided_location"]


def test_ksa_filtering_remains_correct_after_extraction_refinements():
    """Section 9.12: Egypt records still never resolve/participate."""
    candidates = resolve_branch_candidates("فرع القاهرة شارع التحرير", ALL_LOCATIONS, ksa_only=True)
    assert all(c["cr18c_region"] != "EGY" for c in candidates)
    result = validate("Patient: فين فرع القاهرة؟\nAgent: شارع التحرير القاهرة")
    assert result["is_violation"] is False


# ── Regression: customer-location vs. Andalusia branch-location intent ─────
# A home-care/delivery conversation where the agent explains that the
# CUSTOMER will send/share their own location must never be mistaken for a
# statement about an Andalusia facility's address — the mere presence of
# "موقع"/"لوكيشن" is not enough evidence on its own.

def test_customer_sends_own_location_does_not_trigger_intent():
    """Real regression: 'وبترسل لهم الموقع' (the customer will later send
    THEM — the delivery/home-care team — their own location) incorrectly
    flagged agent_provided=True before this fix."""
    patient_intent, agent_intent = detect_location_intent("تمام تحت أمرك", "وبترسل لهم الموقع")
    assert patient_intent is False
    assert agent_intent is False


def test_customer_sends_own_location_is_not_applicable_end_to_end():
    result = validate("Patient: تمام تحت أمرك\nAgent: وبترسل لهم الموقع")
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


@pytest.mark.parametrize("customer_location_text", [
    "ابعت موقعك",
    "شارك اللوكيشن",
    "المنسق هيطلب موقعك",
    "نحتاج موقع حضرتك",
])
def test_other_customer_location_phrasings_do_not_trigger_intent(customer_location_text):
    _patient_intent, agent_intent = detect_location_intent("", customer_location_text)
    assert agent_intent is False


@pytest.mark.parametrize("branch_location_text", [
    "عنوان الفرع",
    "موقع الفرع",
    "العنوان : شارع الأمير سلطان",
    "هشارك مع حضرتك لوكيشن فرع السنابل",
    "تم إرسال الموقع الخاص بفرع السنابل",
])
def test_branch_location_phrasings_still_trigger_intent(branch_location_text):
    """The fix must not overcorrect — genuine Andalusia facility-location
    statements (category A) still count, including ones that share a verb
    ('شارك') with a category-B phrasing but explicitly name a branch."""
    _patient_intent, agent_intent = detect_location_intent("", branch_location_text)
    assert agent_intent is True


# ── Regression: provided-address branch resolution on a real conversation ──

def test_real_hospital_conversation_resolves_via_provided_address():
    """Real regression: patient books at 'أندلسية اللي في المحمديه' (not a
    location REQUEST — a booking statement) and never asks a location
    question; the agent proactively clarifies branch + address across two
    turns. Explicit branch resolution has nothing to anchor on (the patient
    never names a real branch), so this must fall through to
    provided-address resolution against the extracted Agent location
    text — not BRANCH_UNRESOLVED, and not a comparison against the whole
    transcript."""
    transcript = (
        "Patient: ابغى احجز في أندلسية اللي في المحمديه\n"
        "Patient: هذا في فرع المحمديه\n"
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript)
    assert "العنوان : تقاطع شارع عبدالله سليمان" in result["provided_location"]
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert result["crm_location"] == LOC_HOSPITAL["cr301_description"]
    assert result["outcome"] == "PASS"
    assert result["is_violation"] is False


def test_prince_sultan_still_resolves_correctly_after_intent_fix():
    result = validate("Patient: عنوان فرع الأمير سلطان؟\nAgent: شارع الامير سلطان حي المحمديه جده")
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"
    assert result["outcome"] == "PASS"


def test_sanabil_still_resolves_correctly_after_intent_fix():
    result = validate("Patient: فين فرع السنابل؟\nAgent: شارع السنابل العام بعد محطه نفط حي السنابل جده")
    assert result["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert result["outcome"] == "PASS"


def test_normal_booking_conversation_mentioning_branch_does_not_trigger():
    """A plain booking exchange that names a branch, with no location
    vocabulary and no address content, must stay NOT_APPLICABLE."""
    result = validate("Patient: عايز احجز في فرع السنابل بكرة الساعة 5\nAgent: تمام تم الحجز في فرع السنابل بكرة الساعة 5")
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


# ── Regression: home-care/customer location must not trigger branch validation ──
# Real regression: "Agent: ممكن لوكيشن المنزل ؟ / Patient: <maps.google.com link>"
# was incorrectly detected as agent_provided=True and routed into
# validate_location, which then fetched CRM data and returned
# BRANCH_UNRESOLVED — the location here is the patient's home/service-
# delivery address, never an Andalusia branch.

def test_home_location_question_does_not_trigger_intent():
    _patient_intent, agent_intent = detect_location_intent("", "ممكن لوكيشن المنزل ؟")
    assert agent_intent is False


def test_map_link_after_home_location_question_does_not_trigger_intent():
    patient_intent, agent_intent = detect_location_intent(
        "https://maps.google.com/?q=21.606500,39.240455", "ممكن لوكيشن المنزل ؟",
    )
    assert patient_intent is False
    assert agent_intent is False


def test_home_location_question_is_not_applicable_end_to_end():
    result = validate(
        "Agent: ممكن لوكيشن المنزل ؟\nPatient: https://maps.google.com/?q=21.606500,39.240455",
    )
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


@pytest.mark.parametrize("home_service_text", [
    "ابعت موقعك",
    "شارك اللوكيشن",
    "ممكن موقع حضرتك؟",
    "المنسق هيطلب موقعك",
    "وبترسل لهم الموقع",
    "نحتاج لوكيشن المريض",
    "عنوان المنزل",
    "موقع المنزل",
])
def test_home_service_phrasings_do_not_trigger_intent(home_service_text):
    _patient_intent, agent_intent = detect_location_intent("", home_service_text)
    assert agent_intent is False


@pytest.mark.parametrize("branch_text", [
    "ممكن لوكيشن فرع السنابل؟",
    "عنوان فرع الأمير سلطان؟",
    "فين موقع المستشفى؟",
    "العنوان : شارع الأمير سلطان - حي المحمدية - جدة",
])
def test_branch_location_phrasings_are_unaffected_by_home_service_fix(branch_text):
    """The home-service exclusion must stay narrow — موقع/لوكيشن/عنوان keep
    triggering normally for real Andalusia branch/facility references."""
    patient_intent, _agent_intent = detect_location_intent(branch_text, "")
    assert patient_intent is True


def test_home_location_crm_fetch_never_called_via_direct_validate_call(monkeypatch):
    """Even if validate_location_request() were ever reached for a
    home-service call (defensive path), it must not resolve/compare
    against CRM data — the top-of-function gate already returns
    NOT_APPLICABLE before any candidate resolution happens."""
    result = validate(
        "Agent: ممكن لوكيشن المنزل ؟\nPatient: https://maps.google.com/?q=21.606500,39.240455",
        locations=ALL_LOCATIONS,
    )
    assert result["outcome"] == "NOT_APPLICABLE"


# ── Regression: coincidental token overlap must not select the wrong turn ──
# Real regression (call 4B77DCA1-AF7E-F111-B337-000D3AA9D4A7): an agent's
# own name ("د محمد") coincidentally shared ONE word with an unrelated
# branch's street address, causing the greeting/department-introduction
# turn ("...من قسم الرعاية المنزلية...") to be wrongly picked as "the"
# location-bearing turn instead of the real address given two turns
# later — even though "قسم الرعاية المنزلية" is home-care VOCABULARY, the
# root cause here was never keyword-based exclusion; it was
# _agent_mentions_known_location() accepting a single coincidental token
# match as "evidence" of a location statement. The fix (requiring >=2
# overlapping tokens, or a >=60% overlap of the turn's own tokens) targets
# that directly, without any home-service keyword list.

LOC_WITH_COINCIDENTAL_NAME = {
    "cr301_branchname": "مستشفى أندلسية الشلالات", "cr301_area": "Alex", "cr301_country": "مصر",
    "cr301_description": "7 شارع محمد محمد مطاوع متفرع من ش السلطان حسين - الازاريطه",
    "cr18c_region": "EGY", "statecodename": "Active",
}


def test_af7e_regression_department_introduction_does_not_hijack_extraction():
    transcript = (
        "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
        "Patient: اهلا دكتور\n"
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript, locations=ALL_LOCATIONS + [LOC_WITH_COINCIDENTAL_NAME])
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert result["crm_location"] == LOC_HOSPITAL["cr301_description"]
    assert "الرعاية المنزلية" not in result["provided_location"]
    assert "العنوان : تقاطع شارع عبدالله سليمان" in result["provided_location"]
    assert result["is_violation"] is False


def test_short_single_word_address_still_qualifies_after_overlap_fix():
    """The stricter overlap bar must not regress the existing partial-
    credit case: a short reply that IS the branch's own street name (a
    single surviving token after filler-stripping) must still reach
    scoring, not be dropped as if nothing was said."""
    result = validate("Patient: عنوان فرع صاري؟\nAgent: شارع صارى")
    assert result["outcome"] == "INCOMPLETE_ADDRESS"
    assert result["is_violation"] is True


# ── Regression: Agent evidence must dominate weaker/older Patient context ──
# Root cause: the resolution-attempt loop tried Patient-side sources
# (explicit anchor, then whole patient_text) BEFORE Agent-side sources, and
# stopped at the first non-empty (even ambiguous) match — so an ambiguous
# or stale Patient signal could win before the Agent's own unambiguous
# evidence was ever tried. Fixed by reordering to Agent-anchor → Patient-
# anchor → Agent-address (now also resolvable via address CONTENT, not
# just branch NAME — see resolve_branch_by_address_content) → Patient
# whole-text, plus recognizing that Patient's EXPLICIT branch anchor still
# must catch a genuinely WRONG bare Agent address (existing Sanabil/Prince
# Sultan/Hospital wrong-branch tests, unaffected below).

def test_5c2d54e5_regression_agent_address_overrides_patient_locality_alias():
    """Real regression (call 5C2D54E5): the Patient never uses an explicit
    branch-anchor phrase, only locality/area aliases ("اندلسية جدة",
    "اندلسية السنابل") — historical context that must not override the
    Agent's later, concrete branch + address."""
    transcript = (
        "Patient: انتوا في حي الجامعه ؟ لاني اخترت اندلسية جده\n"
        "Agent: بالفعل\n"
        "Agent: العنوان: تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
        "Agent: الموقع : https://maps.app.goo.gl/VpKdwFauiiubFmVi7\n"
        "Patient: في اندلسية السنابل كنت\n"
        "Agent: انا كدة اكدت لحضرتك الحجز في فرع مستشفي اندلسية جدة يوم الاربعاء"
    )
    result = validate(transcript, locations=ALL_LOCATIONS)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert result["match_confidence"] >= 0.6
    assert result["is_violation"] is False
    assert "السنابل" not in result["provided_location"]


def test_af7e_regression_agent_branch_redirection_overrides_patient_branch():
    """Real regression (call 4B77DCA1-AF7E-F111-B337-000D3AA9D4A7): the
    Patient explicitly names one branch ("فرع المحمديه") but the Agent
    redirects to the main hospital and gives its address — the Agent's
    redirection must win."""
    transcript = (
        "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
        "Patient: اهلا دكتور\n"
        "Patient: هذا في فرع المحمديه\n"
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript, locations=ALL_LOCATIONS + [LOC_WITH_COINCIDENTAL_NAME])
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert result["match_confidence"] >= 0.6
    assert result["is_violation"] is False
    assert "الرعاية المنزلية" not in result["provided_location"]


def test_agent_textual_address_plus_map_url_resolves_correctly():
    """A map link supplied alongside a textual address must stay associated
    with the same block but never become (or pollute) provided_location —
    the textual address alone must resolve and score correctly."""
    transcript = (
        "Agent: العنوان: تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة\n"
        "Agent: الموقع : https://maps.app.goo.gl/VpKdwFauiiubFmVi7"
    )
    result = validate(transcript, locations=ALL_LOCATIONS)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "مستشفى أندلسية جدة"
    assert "https://" not in result["provided_location"]
    assert "maps.app.goo.gl" not in result["provided_location"]


def test_ambiguous_patient_text_becomes_unambiguous_via_agent_address():
    """Patient text alone ties between two branches (each contributes
    exactly one matching distinctive token); the Agent's concrete address
    must resolve it uniquely rather than reporting AMBIGUOUS_BRANCH."""
    assert len(resolve_branch_candidates("مش متاكد السنابل ولا صاري", ALL_LOCATIONS, ksa_only=True)) == 2
    transcript = "Patient: مش متاكد السنابل ولا صاري\nAgent: العنوان شارع صارى متفرع من الخطيب التبريزى حي البوادي جدة"
    result = validate(transcript, locations=ALL_LOCATIONS)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع صاري"


def test_department_introduction_is_never_extracted_as_location():
    transcript = (
        "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة"
    )
    result = validate(transcript, locations=ALL_LOCATIONS)
    assert "قسم الرعاية المنزلية" not in result["provided_location"]
    assert "د محمد" not in result["provided_location"]
    assert result["outcome"] == "PASS"


def test_patient_home_map_location_still_skips_branch_validation():
    result = validate("Agent: ممكن لوكيشن المنزل؟\nPatient: https://maps.google.com/?q=21.606500,39.240455")
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


def test_agent_branch_address_in_home_care_conversation_still_validates():
    """Home-service vocabulary earlier in the call must never hard-stop
    validation of a real branch address given later — the decision is
    based on WHO provided the actual location, not on stop words."""
    transcript = (
        "Agent: السلام عليكم انا من قسم الرعاية المنزلية\n"
        "Agent: العنوان شارع الأمير سلطان حي المحمدية جدة"
    )
    result = validate(transcript, locations=ALL_LOCATIONS)
    assert result["outcome"] == "PASS"
    assert result["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"


def test_sanabil_prince_sultan_hospital_cases_remain_unchanged():
    """Section 8: existing wrong-branch detection must still work — a
    Patient's EXPLICIT branch anchor still catches a genuinely wrong bare
    Agent address (no explicit facility naming of its own)."""
    wrong = validate("Patient: فين فرع السنابل؟\nAgent: العنوان شارع الأمير سلطان، حي المحمدية، جدة")
    assert wrong["requested_branch"] == "عيادات أندلسية فرع السنابل"
    assert wrong["is_violation"] is True

    sultan = validate("Patient: عنوان فرع الأمير سلطان؟\nAgent: شارع الامير سلطان حي المحمديه جده")
    assert sultan["outcome"] == "PASS"
    assert sultan["requested_branch"] == "عيادات أندلسية فرع الأمير سلطان"

    sanabil = validate("Patient: فين فرع السنابل؟\nAgent: شارع السنابل العام بعد محطه نفط حي السنابل جده")
    assert sanabil["outcome"] == "PASS"
    assert sanabil["requested_branch"] == "عيادات أندلسية فرع السنابل"

    hospital = validate(
        "Agent: في فرع المستشفى الرئيسي فقط\n"
        "Agent: العنوان : تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا - جدة",
    )
    assert hospital["outcome"] == "PASS"
    assert hospital["requested_branch"] == "مستشفى أندلسية جدة"


def test_no_regression_in_graph_level_skip_behavior():
    """Section 9: the graph-level location-intent gate (used by
    app.agent.graph's router) must remain unaffected by this resolution-
    priority change — home-service text still yields no intent at all."""
    patient_intent, agent_intent = detect_location_intent("", "ممكن لوكيشن المنزل؟")
    assert patient_intent is False
    assert agent_intent is False
    assert location_validation_needed(call("Agent: ممكن لوكيشن المنزل؟\nPatient: https://maps.google.com/?q=1,2")) is False
