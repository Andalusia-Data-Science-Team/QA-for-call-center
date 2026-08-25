from app.models.input import CallTranscript
from app.location_node.location_validation import (
    detect_location_request,
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
