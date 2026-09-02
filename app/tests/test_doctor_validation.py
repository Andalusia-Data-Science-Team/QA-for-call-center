import asyncio

import pytest

from app.models.input import CallTranscript
from app.service_hub.doctor_validation import (
    authoritative_doctor_pool,
    classify_specific_doctor_intent,
    dedupe_doctors,
    describe_doctor_extraction_evidence,
    describe_excluded_doctor_candidates,
    detect_doctor_mention,
    detect_doctor_signals,
    doctor_scope_validation_needed,
    doctor_validation_needed,
    extract_doctor_context_specialty,
    extract_doctor_turn_candidates,
    has_detailed_scope_evidence,
    is_agent_self_introduction,
    is_plausible_person_name,
    is_supported_doctor_bu,
    parse_examination_age,
    patient_describes_medical_complaint,
    resolve_doctor_candidates,
    validate_doctor_information,
)

# Fixtures mirror the real cr301_newdoctordataset shape (verified against a
# LIVE Dynamics 365 query this session — same doctors, same field names).
DOC_KHAIRYA = {
    "cr301_doctorkey": "100100044",
    "servhub_doctornameen": "khairya Mohamed Aly Mousa",
    "cr301_doctornamear": "خيرية محمد علي موسي",
    "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Pediatrics",
    "cr301_subspecialtyname": "Pediatric Endocrinology",
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AKW", "cr301_businessunitname": "AKW",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None,
    "cr301_scopeofservice": "Diagnosis and treatment of developmental disorders",
    "cr301_scopeofservicear": "تشخيص وعلاج اضطرابات النمو",
    "cr301_qualificationsandexperience": "Consultant in pediatric endocrinology",
    "cr301_qualificationsandexperiencear": "استشاري غدد الصماء والسكري للأطفال",
    "servhub_examinationage": "Age: 0-16 years Gender : All",
    "cr301_walkinconsultationfees": 400,
}
DOC_ALIAA = {
    "cr301_doctorkey": "100110053",
    "servhub_doctornameen": "Aliaa Mahmoud Elmadbouly",
    "cr301_doctornamear": "علياء محمود المدبولي",
    "cr301_degreename": "Specialist",
    "cr301_specialtyname": "E.N.T",
    "cr301_subspecialtyname": "Audiological medicine",
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AKW", "cr301_businessunitname": "AKW",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": "لا يوجد لدينا عملية زراعة قوقعة الأذن ولكن يوجد تركيب سماعات",
    "cr301_scopeofservice": "Treatment of hearing loss with a hearing aid",
    "cr301_scopeofservicear": "علاج ضعف السمع عن طريق السماعة",
    "cr301_qualificationsandexperience": None,
    "cr301_qualificationsandexperiencear": "اخصائى سمعيات",
    "servhub_examinationage": "Age: 0 and above Gender : All",
    "cr301_walkinconsultationfees": 345,
}
DOC_RAMI = {
    "cr301_doctorkey": "200200055",
    "servhub_doctornameen": "Rami Samarkandi",
    "cr301_doctornamear": "رامي سمرقندي",
    "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Orthopedic",
    "cr301_subspecialtyname": "Sports Medicine",
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None,
    "cr301_scopeofservice": "Sports injuries, knee arthroscopy, joint problems",
    "cr301_scopeofservicear": "اصابات رياضية، مناظير الركبة، مشاكل المفاصل",
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": "18 years and above",
    "cr301_walkinconsultationfees": 500,
}
DOC_INACTIVE = {
    "cr301_doctorkey": "300300066",
    "servhub_doctornameen": "Nabil Retired",
    "cr301_doctornamear": "نبيل متقاعد",
    "cr301_degreename": "Consultant", "cr301_specialtyname": "General Surgery",
    "cr301_subspecialtyname": None, "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Inactive", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
DOC_UNSUPPORTED_BU = {
    "cr301_doctorkey": "400400077",
    "servhub_doctornameen": "Farid Outside Scope",
    "cr301_doctornamear": "فريد خارج النطاق",
    "cr301_degreename": "Specialist", "cr301_specialtyname": "Dermatology",
    "cr301_subspecialtyname": None, "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AMH", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
# Active, supported-BU, but explicitly NOT flagged OPD — a legitimate
# home-care/non-OPD doctor. Must still be searchable/resolvable.
DOC_NON_OPD_HOME_CARE = {
    "cr301_doctorkey": "500500111",
    "servhub_doctornameen": "Salma Non OPD",
    "cr301_doctornamear": "سلمى غير عيادات",
    "cr301_degreename": "Consultant", "cr301_specialtyname": "Home Care",
    "cr301_subspecialtyname": None, "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "Non-OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": "Home-visit consultations",
    "cr301_scopeofservicear": "استشارات منزلية",
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
# Two doctors sharing a common first name — first-name-alone must not resolve.
DOC_MOHAMED_A = {
    "cr301_doctorkey": "500500088", "servhub_doctornameen": "Mohamed Salem",
    "cr301_doctornamear": "محمد سالم", "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Cardiology", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
DOC_MOHAMED_B = {
    "cr301_doctorkey": "500500099", "servhub_doctornameen": "Mohamed Nasser",
    "cr301_doctornamear": "محمد ناصر", "cr301_degreename": "Specialist",
    "cr301_specialtyname": "Dermatology", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AKW", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
# Duplicate rows, same doctor key, one with a gap the other fills.
DOC_DUP_1 = {
    "cr301_doctorkey": "600600011", "servhub_doctornameen": "Gihad Bekheet",
    "cr301_doctornamear": "جهاد بخيت", "cr301_degreename": "Specialist",
    "cr301_specialtyname": None, "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
DOC_DUP_2 = {**DOC_DUP_1, "cr301_specialtyname": "General Surgery"}

# Real doctor from the live CRM dataset (verified this session) — used for
# the false-positive-intent/extraction regression tests below, since the
# real transcripts that triggered the bug all involve this doctor.
DOC_MOHAMED_ELALFY = {
    "cr301_doctorkey": "110685",
    "servhub_doctornameen": "Mohamed Elalfy",
    "cr301_doctornamear": "محمد الألفي",
    "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Orthopedic",
    "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}

# External doctor used for the self-introduction-vs-external-recommendation
# regression tests: has BOTH a specialty and documented scope-of-service
# text, so a resolved-doctor test can assert the mandatory doctor evidence
# (name/specialty/scope) is actually carried, not just a bare CRM key.
DOC_MOHAMED_ADEL = {
    "cr301_doctorkey": "700700022",
    "servhub_doctornameen": "Mohamed Adel",
    "cr301_doctornamear": "محمد عادل",
    "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Orthopedic",
    "cr301_subspecialtyname": "Spine Surgery",
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None,
    "cr301_scopeofservice": "Back pain, spine disorders, joint injuries",
    "cr301_scopeofservicear": "الم الظهر، امراض العمود الفقري، اصابات المفاصل",
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": "18 years and above",
    "cr301_walkinconsultationfees": 450,
}

# Real-CRM-shape fixture for the 86CE896E-6C79-F111-B337-000D3AA9D4A7
# regression: an agent recommendation using the "الاستشاري <name>" title
# (not دكتور/طبيب) for a neurology home-visit case.
DOC_OSAMA = {
    "cr301_doctorkey": "800800033",
    "servhub_doctornameen": "Osama Abdelsalam",
    "cr301_doctornamear": "اسامة عبد السلام",
    "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Neurology",
    "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ",
    "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None,
    "cr301_scopeofservice": "Neurological assessment, home-visit consultations",
    "cr301_scopeofservicear": "تقييم اعصاب، استشارات منزلية",
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None,
    "cr301_walkinconsultationfees": 1000,
}
DOC_MOHAMED_AHMED_2 = {
    "cr301_doctorkey": "800800044", "servhub_doctornameen": "Mohamed Ahmed Kamal",
    "cr301_doctornamear": "محمد أحمد كامل", "cr301_degreename": "Consultant",
    "cr301_specialtyname": "Neurology", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}

ALL_DOCTORS = [
    DOC_KHAIRYA, DOC_ALIAA, DOC_RAMI, DOC_INACTIVE, DOC_UNSUPPORTED_BU,
    DOC_MOHAMED_A, DOC_MOHAMED_B, DOC_MOHAMED_ELALFY, DOC_MOHAMED_ADEL,
    DOC_OSAMA, DOC_MOHAMED_AHMED_2,
]


def call(transcript: str) -> CallTranscript:
    return CallTranscript(
        call_id="doc-1", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-08-27", call_duration_seconds=1, department="Scheduling",
        transcript=transcript,
    )


def validate(transcript: str, doctors=ALL_DOCTORS):
    c = call(transcript)
    signals = detect_doctor_signals(c)
    return validate_doctor_information(c, doctors, signals)


# ── Applicability ────────────────────────────────────────────────────────────

def test_no_doctor_mentioned_is_not_applicable():
    result = validate("Patient: عايز اعرف الاسعار\nAgent: تمام")
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


def test_generic_specialty_request_alone_is_not_applicable():
    result = validate("Patient: محتاج دكتور عظام\nAgent: ممكن أوضح لك المواعيد المتاحة")
    assert result["outcome"] == "NOT_APPLICABLE"


def test_patient_asks_named_doctor_and_agent_answers_runs_validation():
    result = validate("Patient: دكتور محمد الألفي تخصصه ايه؟\nAgent: دكتور محمد الألفي استشاري عظام")
    assert result["applicable"] is True
    assert result["outcome"] != "NOT_APPLICABLE"


def test_agent_recommends_named_doctor_runs_validation():
    result = validate("Patient: عندي طفل عنده تأخر نمو\nAgent: دكتورة خيرية محمد استشارية غدد صماء أطفال")
    assert result["applicable"] is True


def test_patient_only_mention_with_no_name_is_not_an_agent_claim():
    """'قالولي الدكتور متخصص قلب' names no doctor at all — it may still be
    read as SOME mention of "the doctor" (hence non-punitive
    DOCTOR_UNRESOLVED, not a hard error), but it must never be treated as
    an Agent claim/fabricate any validated field."""
    result = validate("Patient: قالولي الدكتور متخصص قلب\nAgent: تمام هوضحلك")
    assert result["outcome"] in ("NOT_APPLICABLE", "DOCTOR_UNRESOLVED")
    assert result["validated_fields"] == {}
    assert result["is_violation"] is False


# ── Doctor resolution ────────────────────────────────────────────────────────

def test_exact_arabic_doctor_name_resolves():
    result = validate("Agent: دكتورة خيرية محمد استشارية أطفال")
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == "100100044"


def test_exact_english_doctor_name_resolves():
    candidates = resolve_doctor_candidates("Khairya Mohamed", ALL_DOCTORS)
    assert len(candidates) == 1
    assert candidates[0]["cr301_doctorkey"] == "100100044"


def test_title_prefix_ignored_safely():
    """Dr./دكتور/د. prefixes must not prevent resolution, wherever in the
    sentence they appear."""
    for phrasing in ("دكتور رامي سمرقندي", "Dr. Rami Samarkandi", "د. رامي سمرقندي"):
        candidates = resolve_doctor_candidates(phrasing, ALL_DOCTORS)
        assert len(candidates) == 1
        assert candidates[0]["cr301_doctorkey"] == "200200055"


def test_multi_token_partial_match_resolves_shortened_name():
    """Agent says only part of the full CRM name ('خيرية محمد') — must still
    resolve to the doctor whose full CRM name is 'خيرية محمد علي موسي'."""
    candidates = resolve_doctor_candidates("خيرية محمد", ALL_DOCTORS)
    assert len(candidates) == 1
    assert candidates[0]["cr301_doctorkey"] == "100100044"


def test_ambiguous_first_name_only_does_not_resolve():
    candidates = resolve_doctor_candidates("محمد", ALL_DOCTORS)
    assert candidates == []
    result = validate("Agent: دكتور محمد هيكشفلك")
    assert result["outcome"] == "DOCTOR_UNRESOLVED"


def test_duplicate_crm_rows_same_key_are_deduplicated():
    deduped = dedupe_doctors([DOC_DUP_1, DOC_DUP_2])
    assert len(deduped) == 1
    # Non-null value from the second row fills the gap in the first.
    assert deduped[0]["cr301_specialtyname"] == "General Surgery"


# ── Record filtering ─────────────────────────────────────────────────────────

def test_active_opd_supported_bu_is_valid():
    pool = authoritative_doctor_pool(ALL_DOCTORS)
    keys = {d["cr301_doctorkey"] for d in pool}
    assert "100100044" in keys


def test_inactive_doctor_recommendation_fails():
    result = validate("Agent: دكتور نبيل متقاعد استشاري جراحة عامة", doctors=[DOC_INACTIVE])
    assert result["outcome"] == "FAIL"
    assert result["is_violation"] is True


def test_unsupported_bu_doctor_not_authoritative():
    pool = authoritative_doctor_pool([DOC_UNSUPPORTED_BU])
    assert pool == []
    assert is_supported_doctor_bu("AMH") is False
    assert is_supported_doctor_bu("AKW") is True


# ── OPD is informational metadata only, never a searchability gate ─────────
# Regression: call 86CE896E-6C79-F111-B337-000D3AA9D4A7's home-visit doctor
# recommendation kept returning DOCTOR_UNRESOLVED because the authoritative
# pool required cr301_opdflag == 'OPD' — but a legitimate home-care doctor
# may not carry that flag at all. Fixed generally: authoritative_doctor_pool
# now requires only Active status + supported BU; cr301_opdflag is still
# fetched/carried on every record purely as informational metadata.

def test_active_non_opd_doctor_is_authoritative():
    pool = authoritative_doctor_pool([DOC_NON_OPD_HOME_CARE])
    keys = {d["cr301_doctorkey"] for d in pool}
    assert DOC_NON_OPD_HOME_CARE["cr301_doctorkey"] in keys


def test_home_care_recommendation_resolves_despite_non_opd_flag():
    transcript = (
        "Agent: قد نحتاج لزيارة طبيب مخ واعصاب\n"
        "Agent: متوفر معنا الاستشاري سلمى غير عيادات وهي ممتازة في تلك الحالات\n"
    )
    result = validate(transcript, doctors=[DOC_NON_OPD_HOME_CARE])
    assert result["outcome"] != "DOCTOR_UNRESOLVED"
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_NON_OPD_HOME_CARE["cr301_doctorkey"]
    # opd_flag is still carried as metadata, never a gate.
    assert result["opd_flag"] == "Non-OPD"
    assert has_detailed_scope_evidence(result["scope_reference"]) is True


def test_inactive_non_opd_doctor_still_rejected():
    """Removing the OPD gate must not accidentally remove the ACTIVE-status
    gate too — an inactive doctor stays excluded regardless of opd_flag."""
    inactive_non_opd = {**DOC_NON_OPD_HOME_CARE, "statuscodename": "Inactive"}
    pool = authoritative_doctor_pool([inactive_non_opd])
    assert pool == []


def test_unsupported_bu_doctor_recommendation_fails():
    result = validate("Agent: دكتور فريد خارج النطاق أخصائي جلدية", doctors=[DOC_UNSUPPORTED_BU])
    assert result["outcome"] == "FAIL"
    assert result["is_violation"] is True


# ── Information validation ───────────────────────────────────────────────────

def test_correct_degree_passes():
    result = validate("Agent: دكتورة خيرية محمد استشارية أطفال")
    assert result["validated_fields"]["degree"]["outcome"] == "PASS"


def test_incorrect_degree_fails():
    result = validate("Agent: دكتورة خيرية محمد أخصائية أطفال")
    assert result["validated_fields"]["degree"]["outcome"] == "FAIL"
    assert result["outcome"] == "FAIL"


def test_correct_specialty_passes():
    result = validate("Agent: دكتورة خيرية محمد استشارية أطفال")
    assert result["validated_fields"]["specialty"]["outcome"] == "PASS"


def test_incorrect_specialty_fails():
    result = validate("Agent: دكتورة خيرية محمد استشارية قلب")
    assert result["validated_fields"]["specialty"]["outcome"] == "FAIL"


def test_correct_subspecialty_recorded_separately_from_specialty():
    """'غدد صماء' describes the SUBSPECIALTY (Pediatric Endocrinology), not
    the general specialty (Pediatrics) — must not be reported as a failed
    specialty claim."""
    result = validate("Agent: دكتورة خيرية محمد استشارية غدد صماء أطفال")
    assert result["validated_fields"]["subspecialty"]["outcome"] == "PASS"
    assert "specialty" not in result["validated_fields"] or result["validated_fields"].get("specialty", {}).get("outcome") != "FAIL"
    assert result["outcome"] == "PASS"


def test_correct_business_unit_recorded():
    result = validate("Agent: دكتور رامي سمرقندي في فرع AHJ")
    assert result["business_unit"] == "AHJ"


def test_correct_walkin_fee_passes():
    result = validate("Patient: كام كشفية دكتورة خيرية محمد؟\nAgent: كشفية دكتورة خيرية محمد 400 ريال")
    assert result["validated_fields"]["walkin_fee"]["outcome"] == "PASS"
    assert result["outcome"] == "PASS"


def test_incorrect_walkin_fee_fails():
    result = validate("Patient: كام كشفية دكتورة خيرية محمد؟\nAgent: كشفية دكتورة خيرية محمد 900 ريال")
    assert result["validated_fields"]["walkin_fee"]["outcome"] == "FAIL"
    assert result["outcome"] == "FAIL"


def test_correct_examination_age_statement_passes():
    result = validate("Agent: دكتورة خيرية محمد يستقبل أطفال")
    assert result["validated_fields"]["examination_age"]["outcome"] == "PASS"


def test_incorrect_age_eligibility_statement_fails():
    """CRM age range is 0-16; claiming 'لا يستقبل أقل من 18' contradicts it."""
    result = validate("Agent: دكتورة خيرية محمد لا يستقبل اقل من 18")
    assert result["validated_fields"]["examination_age"]["outcome"] == "FAIL"
    assert result["outcome"] == "FAIL"


def test_scope_of_service_factual_statement_validated():
    result = validate("Patient: عندي اصابة رياضية في الركبة\nAgent: دكتور رامي سمرقندي يعالج اصابات رياضية ومناظير الركبة")
    assert result["validated_fields"]["scope_of_service"]["outcome"] == "PASS"


def test_qualification_statement_validated():
    result = validate("Agent: دكتورة خيرية محمد استشارية غدد صماء وسكري للأطفال")
    assert result.get("validated_fields", {}).get("qualifications", {}).get("outcome") in (None, "PASS")


def test_doctor_note_restriction_validated():
    result = validate("Agent: دكتورة علياء المدبولي لا يوجد لدينا عملية زراعة قوقعة الأذن")
    assert result["validated_fields"]["doctor_notes"]["outcome"] == "PASS"


def test_optional_missing_field_does_not_cause_fail():
    """Qualifications is NULL (both AR and EN) for DOC_RAMI — a claim about
    it must read as NEEDS_REVIEW (no authoritative reference at all),
    never a hard FAIL, and must not drag down an otherwise-correct result."""
    result = validate("Agent: دكتور رامي سمرقندي معاه بورد سعودي", doctors=[DOC_RAMI])
    assert result["validated_fields"]["qualifications"]["outcome"] == "NEEDS_REVIEW"
    assert result["outcome"] != "FAIL"


# ── Semantic-validation applicability gate (deterministic side only —
# the LLM call itself is exercised via app.agent.nodes in test_graph_routing) ──

def test_scope_gate_true_when_all_conditions_met():
    result = validate("Patient: عندي اصابة رياضية في الركبة\nAgent: دكتور رامي سمرقندي يعالج اصابات رياضية")
    assert doctor_scope_validation_needed(result, "عندي اصابة رياضية في الركبة شديدة") is True


def test_scope_gate_false_when_doctor_unresolved():
    result = validate("Agent: دكتور محمد هيشوفلك")
    assert result["outcome"] == "DOCTOR_UNRESOLVED"
    assert doctor_scope_validation_needed(result, "عندي وجع في الركبة") is False


def test_scope_gate_false_when_no_patient_complaint():
    result = validate("Agent: دكتور رامي سمرقندي يعالج اصابات رياضية")
    assert doctor_scope_validation_needed(result, "عايز اعرف المواعيد المتاحة") is False


def test_patient_complaint_detection():
    assert patient_describes_medical_complaint("عندي طفل عنده تأخر نمو") is True
    assert patient_describes_medical_complaint("عايز اعرف المواعيد المتاحة") is False


# ── Examination-age parsing ──────────────────────────────────────────────────

def test_parse_examination_age_range():
    assert parse_examination_age("Age: 0-16 years Gender : All") == (0, 16.0)


def test_parse_examination_age_and_above():
    assert parse_examination_age("18 years and above") == (18, float("inf"))


def test_parse_examination_age_malformed_returns_none():
    """Multi-BU strings this parser can't confidently split must not
    silently invent a rule — the caller treats None as needs-review."""
    assert parse_examination_age("weird format with no numbers at all !!") is None


def test_doctor_validation_needed_gate():
    assert doctor_validation_needed(call("Agent: دكتور محمد الألفي استشاري عظام")) is True
    assert doctor_validation_needed(call("Patient: محتاج دكتور عظام\nAgent: تمام")) is False


def test_detect_doctor_mention_generic_vs_named():
    assert detect_doctor_mention("محتاج دكتور عظام") is False


# ── False-positive regression: generic/function references to "the doctor" ──
# Real batch-run false positives — none of these name a specific doctor, so
# none of them may register as a doctor mention at all (items 1-4 of the
# user's 15-item regression list).

def test_generic_reference_doctor_requested_xray_no_intent():
    assert detect_doctor_mention("للاسف دا سعر الاشعة اللي الدكتور طلبها لحضرتك") is False


def test_generic_reference_prescription_from_a_doctor_no_intent():
    assert detect_doctor_mention("معاك وصفة من طبيب؟") is False


def test_generic_either_gender_doctor_no_intent():
    assert detect_doctor_mention("ودكتور او دكتوره وكم السعر إذا تعملوها عندكم") is False


def test_generic_treating_physician_no_intent():
    assert detect_doctor_mention("الطبيب المعالج") is False
    assert detect_doctor_mention("الدكتور المختص") is False
    assert detect_doctor_mention("لازم الطبيب يكتب الطلب") is False


# ── False-positive regression: Agent self-introductions (items 5-6) ─────────

def test_agent_self_introduction_mohamed_not_a_doctor_candidate():
    text = "السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة"
    assert is_agent_self_introduction(text) is True
    c = call(f"Agent: {text}")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert agent_c == []
    assert patient_c == []
    assert ignored == ["محمد"]
    assert doctor_validation_needed(c) is False


def test_agent_self_introduction_hisham_not_a_doctor_candidate():
    text = "مع حضرتك دكتور هشام من مجموعة اندلسية"
    assert is_agent_self_introduction(text) is True
    c = call(f"Agent: {text}")
    assert doctor_validation_needed(c) is False


# ── True-positive regression: entity-based detection must still fire (7-8) ──

def test_patient_named_doctor_registers_intent():
    c = call("Patient: الدكتور محمد الالفي")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == ["محمد الالفي"]
    assert agent_c == []
    assert ignored == []
    # A completely bare name mention, with no booking/inquiry action ever
    # attached to it, is NOT itself a doctor-related intent — see
    # classify_specific_doctor_intent's docstring ("a named doctor merely
    # appearing somewhere in the conversation is not enough"). The name
    # DOES register as a raw candidate (above), but doctor_validation_needed
    # requires it to be the target of an actual booking/inquiry action.
    assert doctor_validation_needed(c) is False


def test_patient_booking_action_attached_to_named_doctor_registers_intent():
    c = call("Patient: ابغى احجز مع الدكتور محمد الالفي")
    assert doctor_validation_needed(c) is True


def test_agent_named_doctor_with_slash_prefix_registers_intent():
    c = call("Agent: مع د/ محمد الالفي")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert agent_c == ["محمد الالفي"]
    assert doctor_validation_needed(c) is True


def test_more_should_trigger_entity_based_examples():
    for text in [
        "مع دكتور محمد الألفي",
        "الدكتورة وصال محمد",
        "احجز مع د/ محمد الالفي",
        "دكتور حاتم تخصصه ايه؟",
        "متاح مع دكتورة نانسي رند",
        "أنصحك بدكتور محمود الألفي",
    ]:
        assert detect_doctor_mention(text) is True, text


def test_more_should_not_trigger_generic_examples():
    for text in [
        "الدكتور طلب الأشعة",
        "معاك وصفة من طبيب؟",
        "لازم الطبيب يكتب الطلب",
        "ودكتور أو دكتورة؟",
        "الطبيب المعالج",
        "الدكتور المختص",
    ]:
        assert detect_doctor_mention(text) is False, text
    # "مع حضرتك د محمد" alone DOES look like a named doctor to the
    # speaker-agnostic, single-message detect_doctor_mention() gate (it
    # deliberately knows nothing about self-introductions — see its
    # docstring); the self-introduction exclusion is speaker+message-aware
    # and is exercised separately via is_agent_self_introduction() /
    # extract_doctor_turn_candidates() in the tests above.
    assert detect_doctor_mention("مع حضرتك د محمد") is True
    assert is_agent_self_introduction("مع حضرتك د محمد") is True


# ── Real transcript regressions (items 9-12) ─────────────────────────────────

MOHAMED_ELALFY_TRANSCRIPT = (
    "Patient: عايز احجز مع دكتور محمد الالفي\n"
    "Agent: تمام يا فندم هوضحلك التفاصيل\n"
    "Patient: تمام, الدكتور محمد الالفي عايز اعرف كام الكشف عنده\n"
    "Agent: نعم هوضحلك\n"
    "Agent: تم تأكيد حجزك يوم الخميس الساعه 5 مع د/ ( محمد الالفي ) في فرع مستشفى أندلسية الحجاز\n"
)


def test_mohamed_elalfy_full_transcript_resolves_to_single_doctor():
    """The most important true-positive regression: this transcript
    previously produced candidate_count=5 / AMBIGUOUS_DOCTOR because the
    whole Agent transcript was matched as one blob. It must now resolve
    uniquely to the single CRM doctor محمد الألفي / Mohamed Elalfy."""
    c = call(MOHAMED_ELALFY_TRANSCRIPT)
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == ["محمد الالفي", "محمد الالفي"]
    assert agent_c == ["محمد الالفي"]
    assert ignored == []
    assert doctor_validation_needed(c) is True

    result = validate(MOHAMED_ELALFY_TRANSCRIPT)
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == "110685"
    assert result["outcome"] != "AMBIGUOUS_DOCTOR"


MRI_BOOKING_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايز احجز اشعة رنين مغناطيسي MRI\n"
    "Agent: تمام هوضحلك الاسعار والمواعيد المتاحة\n"
    "Patient: تمام كويس\n"
    "Agent: هيتم تأكيد الحجز وهيتواصل معاك زميلي بالتفاصيل\n"
)


def test_mri_booking_call_skips_doctor_validation():
    """Real regression call 4D0155E7 — the only doctor-looking phrase is
    the Agent's own self-introduction; doctor validation must be skipped
    entirely, not resolved ambiguously."""
    c = call(MRI_BOOKING_TRANSCRIPT)
    assert doctor_validation_needed(c) is False
    result = validate(MRI_BOOKING_TRANSCRIPT)
    assert result["outcome"] == "NOT_APPLICABLE"


HOME_CARE_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايز اطلب خدمة تمريض منزلي لوالدتي\n"
    "Agent: تمام هوضحلك تفاصيل الخدمة والاسعار\n"
    "Patient: كويس تمام\n"
    "Agent: هيتم التنسيق مع فريق الرعاية المنزلية\n"
)


def test_home_care_call_skips_doctor_validation():
    """Real regression call 8486A57A — nursing/home-care conversation with
    the same Agent self-introduction; must also be skipped."""
    c = call(HOME_CARE_TRANSCRIPT)
    assert doctor_validation_needed(c) is False
    result = validate(HOME_CARE_TRANSCRIPT)
    assert result["outcome"] == "NOT_APPLICABLE"


MAMMOGRAPHY_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك د محمد من قسم الرعاية المنزلية والاشعة\n"
    "Patient: عايزة اعمل ماموجرام, فين اقرب فرع في جدة؟\n"
    "Agent: اقرب فرع ليكي هو تقاطع شارع عبدالله سليمان مع طريق الجامعة أمام مول الجامعة بلازا جدة\n"
    "Patient: تمام هروح هناك\n"
)


def test_mammography_call_skips_doctor_but_location_still_runs(monkeypatch):
    """Real regression call 4B77DCA1 — doctor validation must be skipped
    while location validation is completely unaffected (still resolves to
    مستشفى أندلسية جدة and PASSes)."""
    c = call(MAMMOGRAPHY_TRANSCRIPT)
    assert doctor_validation_needed(c) is False
    result = validate(MAMMOGRAPHY_TRANSCRIPT)
    assert result["outcome"] == "NOT_APPLICABLE"

    import app.service_hub.crm_location as crm_location
    from app.agent.nodes import validate_location_node
    from app.tests.test_location_validation import ALL_LOCATIONS

    monkeypatch.setattr(crm_location, "fetch_ksa_locations", lambda *a, **k: ALL_LOCATIONS)
    loc_result = asyncio.run(validate_location_node({"call": c}))["location_validation"]
    assert loc_result["outcome"] == "PASS"
    assert "أندلسية جدة" in (loc_result.get("requested_branch") or "")


# ── False-positive regression: generic specialty/role is not a person name ──
# Real regression call 9ABA561E-1E7E-F111-B337-000D3AA9D4A7 — "دكتور
# التخدير" ("the anesthesiologist") was extracted as if "التخدير" were a
# person's name, because the generic-specialty blocklist only recognised
# "تخدير" (without the attached "ال" definite article). Fixed generally via
# _is_non_name_word's ال/و-stripping, not by adding "التخدير" as one more
# hardcoded entry.

def test_generic_anesthesia_doctor_no_intent():
    text = "مطلوب تحديد موعد دكتور التخدير وموعد أشعة رنين على الدماغ بدون صبغه لطفل عمره 7 سنوات"
    assert detect_doctor_mention(text) is False
    c = call(f"Patient: {text}")
    assert doctor_validation_needed(c) is False


def test_agent_generic_anesthesia_reference_no_candidate():
    text = "حضرتك بعد زيارة دكتور التخدير بالعيادة بتشرفنا في استقبال قسم الاشعة لحجز موعد الاشعة"
    c = call(f"Agent: {text}")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert agent_c == []
    assert patient_c == []
    assert doctor_validation_needed(c) is False


def test_full_9aba_transcript_has_no_named_doctor():
    """The full real regression call: only 'دكتور التخدير' mentions appear
    (never a named doctor) — validate_doctor must be entirely skippable."""
    transcript = (
        "Patient: مطلوب تحديد موعد دكتور التخدير وموعد أشعة رنين على الدماغ بدون صبغه لطفل عمره 7 سنوات\n"
        "Agent: حضرتك محتاج اي موعد في البداية\n"
        "Patient: تخدير\n"
        "Agent: تم تأكيد حجزك ( عيادة التخدير )\n"
        "Agent: حضرتك بعد زيارة دكتور التخدير بالعيادة بتشرفنا في استقبال قسم الاشعة لحجز موعد الاشعة\n"
    )
    c = call(transcript)
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == []
    assert agent_c == []
    assert doctor_validation_needed(c) is False
    result = validate(transcript)
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["applicable"] is False


def test_9aba_transcript_with_agent_self_introduction_still_has_no_named_doctor():
    """Full real regression combined with an Agent self-introduction naming
    a DIFFERENT person ('احمد') in a different department — that name must
    remain ignored (it's the Agent, not a doctor being discussed), and the
    'دكتور التخدير'/specialty-only mentions still never become candidates
    (no AMBIGUOUS_DOCTOR)."""
    transcript = (
        "Agent: السلام عليكم مع حضرتك د احمد من قسم الاشعة\n"
        "Patient: مطلوب تحديد موعد دكتور التخدير وموعد أشعة رنين على الدماغ بدون صبغه لطفل عمره 7 سنوات\n"
        "Patient: تخدير\n"
        "Agent: تم تأكيد حجزك ( عيادة التخدير )\n"
        "Agent: حضرتك بعد زيارة دكتور التخدير بالعيادة بتشرفنا في استقبال قسم الاشعة لحجز موعد الاشعة\n"
    )
    c = call(transcript)
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == []
    assert agent_c == []
    assert ignored == ["احمد"]
    assert doctor_validation_needed(c) is False
    result = validate(transcript)
    assert result["outcome"] == "NOT_APPLICABLE"


@pytest.mark.parametrize("text", [
    "دكتور الأشعة",
    "دكتور التخدير",
    "دكتور القلب",
    "دكتور العظام",
    "طبيب الأطفال",
    "الدكتور المختص",
    "الدكتور المعالج",
    "دكتور النساء",
    "دكتور الجلدية",
    "دكتور أو دكتورة",
    "دكتور مين",
    "أي دكتور",
])
def test_specialty_and_generic_role_phrases_never_yield_a_candidate(text):
    assert detect_doctor_mention(text) is False, text


# ── False-positive regression: administrative/temporal words never seed a
# name, and a plausible first name still stops cleanly before a following
# pronoun/verb continuation — real regression call
# B1498491-ED7D-F111-B337-000D3AA9D4A7.

def test_doctor_followed_by_time_yields_no_candidate():
    c = call("Patient: عندنا حجز مع الدكتور الساعة 6")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == []
    assert doctor_validation_needed(c) is False


def test_first_name_boundary_stops_before_pronoun_continuation():
    c = call("Patient: عندنا مراجعة مع دكتور امير وهو طالب صورة من الاشعة")
    patient_c, agent_c, ignored = extract_doctor_turn_candidates(c)
    assert patient_c == ["امير"]
    assert detect_doctor_mention("دكتورة وصال هتكشف عليك") is True


# ── Business-Unit-scoped doctor resolution ──────────────────────────────────
# The call's already-detected Business Unit (CallTranscript.business_unit)
# must be preferred over independently guessing a BU when a doctor's name
# matches records in more than one BU — real-world quirk: the same
# physical doctor can exist under two different doctor keys at different
# BUs (see README_doctor.md).

_BU_DOC_AHJ = {
    "cr301_doctorkey": "9001", "cr301_doctornamear": "محمد الالفي", "servhub_doctornameen": "Mohamed Elalfy",
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_degreename": "Consultant", "cr301_specialtyname": "Orthopedic", "cr301_subspecialtyname": None,
}
_BU_DOC_AKW = {
    "cr301_doctorkey": "9002", "cr301_doctornamear": "محمد الالفي", "servhub_doctornameen": "Mohamed Elalfy",
    "cr18c_buname": "AKW", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_degreename": "Specialist", "cr301_specialtyname": "Dermatology", "cr301_subspecialtyname": None,
}


def test_business_unit_disambiguates_same_name_across_two_bus():
    c = CallTranscript(
        call_id="bu-test", agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-08-27", call_duration_seconds=1, department="Scheduling",
        business_unit="AHJ", transcript="Patient: ابغى احجز مع دكتور محمد الالفي",
    )
    from app.service_hub.doctor_validation import validate_doctor_information
    result = validate_doctor_information(c, [_BU_DOC_AHJ, _BU_DOC_AKW])
    assert result["outcome"] == "PASS"
    assert result["doctor_key"] == "9001"
    assert result["business_unit"] == "AHJ"


def test_without_known_business_unit_ambiguity_is_still_reported():
    c = call("Patient: ابغى احجز مع دكتور محمد الالفي")
    from app.service_hub.doctor_validation import validate_doctor_information
    result = validate_doctor_information(c, [_BU_DOC_AHJ, _BU_DOC_AKW])
    assert result["outcome"] == "AMBIGUOUS_DOCTOR"
    assert result["candidate_count"] == 2


# ── classify_specific_doctor_intent — strict doctor-intent gate ─────────────
# The applicability question is "is the ACTIVE booking/inquiry TARGET the
# named doctor", not "does a named doctor appear anywhere in the
# transcript". See classify_specific_doctor_intent's docstring.

def _ctx(transcript: str) -> dict:
    return classify_specific_doctor_intent(call(transcript))


def test_case_a_ordering_doctor_plus_imaging_inquiry_not_applicable():
    """Case A — ordering doctor + imaging inquiry."""
    transcript = (
        "Patient: الدكتور احمد افندي طلب اشعه مقطعيه ، هل تطلع في التطبيق ؟\n"
        "Patient: ابغى اشوف نسخه من request الاشعه\n"
    )
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] == "ordering_or_referring"
    assert doctor_validation_needed(call(transcript)) is False


AHMED_AFANDI_FULL_TRANSCRIPT = (
    "Patient: الدكتور احمد افندي طلب اشعه مقطعيه ، هل تطلع في التطبيق ؟\n"
    "Patient: دكتور احمد افندي\n"
    "Patient: عندكم\n"
    "Patient: كاش\n"
    "Patient: ابغى اشوف نسخه من request الاشعه\n"
    "Agent: الاشعة المطلوبة لحضرتك هي تصوير الأوعية الدموية بالتصوير المقطعي للذراع\n"
    "Agent: هل تحب نحجز موعد للاشعة ؟\n"
    "Patient: متى المواعيد ؟\n"
    "Patient: لانه انا عندي غسيل كلى ولازم تكون قبل موعد جلسة الغسيل\n"
    "Patient: لانها بالصبغه فا لازم اغسل بعد الاشعه على طول\n"
    "Patient: انا مريض فشل كلوي لي 15 سنه فتحليل وظائف الكلى مرتفع دائما\n"
)


def test_case_b_ordering_doctor_plus_service_booking_plus_unrelated_disease():
    """Case B — the full real 42714555-0C7E... scenario: ordering doctor +
    service booking + an unrelated disease mentioned later must NEVER be
    joined into a doctor-scope question."""
    ctx = _ctx(AHMED_AFANDI_FULL_TRANSCRIPT)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] == "ordering_or_referring"
    assert ctx["booking_target"] == "radiology"  # finer-grained category, not just "service"
    assert doctor_validation_needed(call(AHMED_AFANDI_FULL_TRANSCRIPT)) is False


B1498491_FULL_TRANSCRIPT = (
    "Patient: عندنا حجز مع الدكتور الساعة ٦ لازم ندخل المراجعة ومعانا صورة الاشعة\n"
    "Patient: عايزة اعمل حجز اشعة رنين مغناطيسي اليوم قبل الساعة ٦ لان عندنا مراجعة مع دكتور امير وهو طالب صورة من الاشعة\n"
    "Patient: بالنسبة للمراجعة مع الدكتور يوم الاربعاء راح يكون ١٦ يوم من يوم الكشفية عادي يعني؟\n"
    "Agent: لا لازم 14 يوم\n"
)


def test_case_c_existing_doctor_plus_mri_booking_not_applicable():
    """Case C — existing doctor + MRI booking: the booking targets a
    service (MRI), the doctor is only an existing/referring relationship;
    a follow-up-policy/timing question afterward is still not a doctor
    inquiry."""
    ctx = _ctx(B1498491_FULL_TRANSCRIPT)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["booking_target"] == "radiology"  # finer-grained category, not just "service"
    assert doctor_validation_needed(call(B1498491_FULL_TRANSCRIPT)) is False


def test_case_d_direct_doctor_booking():
    ctx = _ctx("Patient: ابغى احجز مع الدكتور محمد الالفي بكرة\n")
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["booking_target"] == "doctor"
    assert ctx["doctor_role"] == "patient_selected"


def test_case_e_direct_doctor_rescheduling():
    transcript = (
        "Patient: كان عندي موعد مع الدكتور محمد الالفي اليوم\n"
        "Patient: ابغى أغير موعد الدكتور لبكرة\n"
    )
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"


def test_case_f_doctor_information_inquiry():
    ctx = _ctx("Patient: ما تخصص الدكتور أحمد أفندي؟\n")
    assert ctx["doctor_intent"] == "specific_doctor_inquiry"
    assert ctx["scope_applicable"] is False  # a factual specialty question, not a suitability one


def test_case_g_doctor_scope_inquiry():
    transcript = "Patient: عندي إصابة في الرباط الصليبي\nPatient: هل الدكتور رامي أحمد مناسب للحالة؟\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_inquiry"
    assert ctx["doctor_role"] == "doctor_inquiry"
    assert ctx["scope_applicable"] is True


def test_case_h_agent_recommendation():
    transcript = "Patient: طفلي عنده صرع\nPatient: مين دكتور مناسب؟\nAgent: ممكن نحجز مع دكتورة خيرية محمد\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_intent"] == "specific_doctor_booking"


def test_case_i_doctor_requested_lab_service_not_applicable():
    transcript = "Patient: الدكتور محمد طلب تحليل كرياتينين\nPatient: هل التحليل يحتاج صيام؟\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"


def test_case_j_doctor_requested_radiology_not_applicable():
    transcript = "Patient: الدكتور محمد طلب رنين\nPatient: كم سعر الرنين؟\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"


# ── Further name-extraction / cross-turn / booking-target regressions ──────
# Real gap found on review: a doctor-attribute marker (يعالج/مناسب/تخصص/...)
# right after a real name was being swallowed INTO the extracted name
# itself, because the continuation-stop blocklist and the attribute-marker
# sets used by the intent classifier were two separate, unreconciled word
# lists.

def test_suitability_marker_is_not_swallowed_into_the_extracted_name():
    assert detect_doctor_mention("هل الدكتور أحمد يعالج الرباط الصليبي؟") is True
    ctx = _ctx("Patient: هل الدكتور أحمد يعالج الرباط الصليبي؟\n")
    assert ctx["named_doctor"] == "احمد"
    assert ctx["doctor_intent"] == "specific_doctor_inquiry"
    assert ctx["scope_applicable"] is True


def test_cross_turn_which_doctor_reply_is_attached_to_booking():
    """The booking verb and the doctor's name need not share a turn: an
    Agent's clarifying 'which doctor?' question, answered with a bare name
    (no دكتور/طبيب title at all), must still resolve to a doctor booking."""
    transcript = "Patient: ابغى احجز موعد\nAgent: مع دكتور مين؟\nPatient: محمد الألفي\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["doctor_role"] == "patient_selected"
    assert ctx["named_doctor"] == "محمد الالفي"
    assert doctor_validation_needed(call(transcript)) is True


def test_which_doctor_question_without_prior_ambiguous_booking_does_not_misfire():
    """A stray 'دكتور مين؟'-shaped phrase with no preceding, still-
    unattached booking verb must not spuriously attach an unrelated later
    name to a booking action."""
    transcript = "Agent: مع دكتور مين؟\nPatient: محمد الألفي\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"


def test_booking_confirmation_only_is_not_scope_eligible():
    """Example C — the Agent's turn is the ONLY doctor mention, and it is a
    transactional confirmation, not a clinical recommendation: identity
    validation should apply, but this must never be treated the same as a
    genuine agent recommendation for scope purposes."""
    transcript = "Patient: تمام احجزوه\nAgent: تم تأكيد الموعد مع الدكتور محمد الألفي\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["doctor_role"] == "doctor_booking_confirmed"
    assert ctx["scope_applicable"] is False
    assert doctor_validation_needed(call(transcript)) is True


@pytest.mark.parametrize("transcript,expected_target", [
    ("Patient: الدكتور أحمد أفندي طلب أشعة مقطعية\nPatient: أبغى أحجز الأشعة يوم الاثنين\n", "radiology"),
    ("Patient: الدكتور طلب تحليل\nPatient: ممكن احجز موعد للتحليل؟\n", "laboratory"),
    ("Patient: الدكتور كتب علاج طبيعي\nPatient: أبغى أحجز زيارة منزلية\n", "home_care"),
    ("Patient: الدكتور طلب علاج طبيعي\nPatient: ابغى احجز موعد علاج طبيعي\n", "physiotherapy"),
])
def test_doctor_named_but_booking_targets_another_service_category(transcript, expected_target):
    """Section 8/17 — doctor name + radiology/lab/home-care/physiotherapy
    booking must all skip Doctor Validation, with the finer service
    category correctly identified for observability."""
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["booking_target"] == expected_target
    assert doctor_validation_needed(call(transcript)) is False


def test_future_appointment_with_doctor_while_booking_another_service():
    """Section 12 Example E — the doctor is only mentioned as timing
    context for an unrelated MRI booking."""
    transcript = "Patient: أبغى أحجز الرنين قبل موعدي مع الدكتور أمير\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["booking_target"] == "radiology"
    assert doctor_validation_needed(call(transcript)) is False


# ── Stage-1 lexical/person-name quality — real production regressions ──────
# (E37A/E47A/EE7A-shaped, not the literal call IDs) — arbitrary text
# extracted after دكتور/طبيب was being treated as a person's name purely
# because it satisfied a loose token-count check. is_plausible_person_name
# (and _doctor_name_candidate, which now consults it) must reject all of
# these outright.

@pytest.mark.parametrize("phrase", [
    "طبيب في المنزل",
    "طبيب في المستشفى",
    "دكتور في البداية",
    "دكتور للتقييم وتحديد الخطة",
    "دكتور و فيتامين ب12",
    "دكتور 375 ريال للكشفية",
    "دكتور يقرر بعدها كم جلسة",
    "دكتور أقرب",
    "دكتور مش",
])
def test_false_candidate_phrases_are_rejected(phrase):
    assert detect_doctor_mention(phrase) is False


def test_e37a_style_home_or_hospital_visit_question_yields_no_doctor():
    """Regression 1 — 'تحتاج زيارة طبيب في المنزل ولا المستشفى؟' extracted
    the entire prepositional phrase as a 'doctor name'."""
    transcript = "Agent: تحتاج زيارة طبيب في المنزل ولا المستشفي ؟\n"
    ctx = _ctx(transcript)
    assert ctx["agent_candidates"] == []
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


def test_e47a_style_price_and_vitamin_fragments_yield_no_doctor():
    """Regression 2 — price/vitamin/session fragments after a title must
    never become doctor candidates."""
    transcript = (
        "Agent: دكتور في البداية هيشوف حالتك\n"
        "Agent: هيوصف دكتور و فيتامين ب12\n"
        "Agent: دكتور 375 ريال للكشفية\n"
    )
    ctx = _ctx(transcript)
    assert ctx["agent_candidates"] == []
    assert ctx["patient_candidates"] == []
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


def test_ee7a_style_bare_which_doctor_question_does_not_leak_patient_name():
    """Regression 3 — a bare 'مع مين؟' (not 'مع دكتور مين؟') must never arm
    the cross-turn linkage; the next Patient turn naming a family
    member/patient ('يوسف') must not become a doctor candidate."""
    transcript = "Patient: ابغى احجز موعد\nAgent: حاجز مع مين؟\nPatient: يوسف\n"
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] is None
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


def test_patient_name_collection_reply_is_never_a_doctor_candidate():
    """'اتشرف بالاسم' collecting the Patient's OWN name must never be
    confused with a doctor-naming exchange."""
    transcript = "Agent: اتشرف بالاسم\nPatient: يوسف\n"
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] is None
    assert ctx["doctor_intent"] == "not_applicable"


def test_agent_self_introduction_with_department_is_not_a_doctor_candidate():
    transcript = "Agent: مع حضرتك محمد من مجموعة أندلسية\n"
    ctx = _ctx(transcript)
    assert ctx["agent_candidates"] == []
    assert ctx["doctor_intent"] == "not_applicable"


@pytest.mark.parametrize("candidate,expected", [
    ("احمد افندي", True),
    ("خيريه محمد", True),
    ("رامي احمد", True),
    ("امير", True),
    ("في المنزل ولا المستشفي", False),
    ("في البدايه", False),
    ("و فيتامين ب12", False),
    ("375 ريال للكشفيه", False),
    ("للتقييم وتحديد الخطه", False),
    ("يقرر بعدها كم جلسه", False),
    ("اقرب", False),
    ("مش", False),
    ("", False),
    (None, False),
])
def test_is_plausible_person_name(candidate, expected):
    assert is_plausible_person_name(candidate) is expected


@pytest.mark.parametrize("phrase", [
    "دكتور أحمد",
    "دكتور أحمد أفندي",
    "دكتورة خيرية محمد",
    "مع د/ رامي أحمد",
])
def test_positive_doctor_name_controls_still_valid(phrase):
    assert detect_doctor_mention(phrase) is True


# ── Agent-is-a-doctor self-introduction (English + Arabic) ─────────────────
# The agent handling the call may themselves be a doctor and introduce
# themselves as "Dr <name>" — that name refers to the SPEAKER, never a
# physician being discussed with the patient, and must never enter the
# doctor candidate pool used for CRM resolution.

@pytest.mark.parametrize("transcript", [
    "Agent: Hello, this is Dr. Mohamed Ahmed from Andalusia.\nPatient: I need physical therapy.\n",
    "Agent: Dr. Ahmed speaking, how may I help you?\nPatient: I need physical therapy.\n",
    "Agent: السلام عليكم مع حضرتك دكتور يوسف محمد من أندلسية.\nPatient: محتاج علاج طبيعي.\n",
    "Agent: د/ محمد أحمد مع حضرتك من قسم الرعاية المنزلية.\nPatient: عايز موعد رعاية منزلية.\n",
])
def test_agent_doctor_self_introduction_never_triggers_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] == "agent_self_introduction"
    assert doctor_validation_needed(call(transcript)) is False


# ── Patient addressing the agent as "doctor" (vocative, not a named
# physician) ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("transcript", [
    "Patient: Thank you doctor.\n",
    "Patient: معليش يا دكتور مش عارف اخر قرار الان اعطيني وقت كدا اشاور الاهل وارد عليكم\n",
    "Patient: دكتور ممكن تساعدني؟\n",
])
def test_patient_addressing_agent_as_doctor_never_triggers_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


# ── Generic doctor references (no specific named person) ───────────────────

@pytest.mark.parametrize("transcript", [
    "Patient: The doctor will decide how many sessions I need.\n",
    "Patient: نشوف الدكتور يقرر بعدها كم جلسة نحتاج\n",
    "Patient: بيتم حجز موعد مع الطبيب للتقييم\n",
    "Patient: محتاج زيارة دكتور عظام\n",
    "Patient: I need an orthopedic doctor.\n",
    "Patient: I want to see a female doctor.\n",
])
def test_generic_doctor_reference_never_triggers_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


# ── English ordering/referring and existing-context doctor references ──────

@pytest.mark.parametrize("transcript", [
    "Patient: Dr Ahmed ordered a CT scan. Can I book the CT tomorrow?\n",
    "Patient: Dr Mohamed requested an MRI. What is the earliest MRI appointment?\n",
    "Patient: I have a follow-up with Dr Amir at 6, so I need an MRI before then.\n",
])
def test_english_ordering_or_existing_doctor_never_triggers_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] in ("ordering_or_referring", "existing_doctor_reference")
    assert doctor_validation_needed(call(transcript)) is False


# ── English positive controls — must continue to RUN ────────────────────────

@pytest.mark.parametrize("transcript,expected_role", [
    ("Patient: I want an appointment with Dr Ahmed Afandi.\n", "patient_selected"),
    ("Patient: When is Dr Ahmed Afandi available?\n", "administrative_reference"),
    ("Patient: What is Dr Ahmed Afandi's specialty?\n", "doctor_inquiry"),
    ("Patient: Is Dr Ahmed Afandi a consultant?\n", "doctor_inquiry"),
])
def test_english_positive_doctor_cases_run_validation(transcript, expected_role):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] != "not_applicable"
    assert ctx["doctor_role"] == expected_role
    assert doctor_validation_needed(call(transcript)) is True


def test_english_agent_confirms_booking_with_named_doctor_runs_validation():
    transcript = "Agent: Your appointment with Dr Ahmed Afandi is confirmed tomorrow.\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["doctor_role"] == "doctor_booking_confirmed"
    assert doctor_validation_needed(call(transcript)) is True


# ── Generalized semantic-role tests ──────────────────────────────────────
# These deliberately use names, phrasings, and structures that do NOT
# appear anywhere else in this test file (or in any resolved case), to
# prove the classifier generalizes on ROLE/STRUCTURE rather than on
# memorized strings/names.

@pytest.mark.parametrize("transcript", [
    # First-person self-identification, varied names/languages/spelling.
    "Agent: أنا دكتور فهد العتيبي من فرع الرياض.\nPatient: عايز اعمل تحليل دم.\n",
    "Agent: انا الدكتوره سارة خالد.\nPatient: محتاج موعد سونار.\n",
    "Agent: I am Dr. John Smith, how can I help?\nPatient: I need a blood test.\n",
    "Agent: I'm Dr Priya Nair.\nPatient: I need a CT scan.\n",
    # "Speaking"/"this is" anchor, unseen names.
    "Agent: Doctor Karim Haddad speaking, good morning.\nPatient: I need physiotherapy.\n",
    "Agent: Hi, this is Dr Layla Al-Otaibi from the clinic.\nPatient: I want to schedule a checkup.\n",
    # Arabic greeting-style self-intro with department, unseen names.
    "Agent: السلام عليكم مع حضرتك د/ عبدالله ناصر من قسم الأشعة.\nPatient: عايز احجز اشعة.\n",
    "Agent: معاك دكتورة هدى إبراهيم من خدمة العملاء.\nPatient: عندي استفسار عن الفاتورة.\n",
])
def test_generalized_self_introduction_variants_never_trigger_validation(transcript):
    """Self-introduction detection must generalize across unseen names,
    both languages, and multiple phrasing anchors — never a specific
    hard-coded name/phrase."""
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_self_introduction"
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


@pytest.mark.parametrize("transcript", [
    # Vocative/direct-address, unseen names, both languages, with and
    # without an explicit title.
    "Patient: Dr Priya, is an ultrasound available today?\n",
    "Patient: Doctor Karim, can you check my insurance coverage?\n",
    "Patient: يا دكتور عبدالله, ممكن اعرف سعر التحليل؟\n",
    "Patient: دكتور هدى, في أي فرع ممكن اروح؟\n",
])
def test_generalized_addressee_variants_never_trigger_validation(transcript):
    """A named person being directly ADDRESSED, with the actual request
    about an unrelated service/entity in a separate clause, must never
    become a doctor-validation target — regardless of the specific name."""
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "conversation_addressee"
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


@pytest.mark.parametrize("transcript", [
    # Ordering/referring, unseen names/services, both languages.
    "Patient: الدكتور عبدالله طلب لي تحليل فيتامين د، هل يحتاج صيام؟\n",
    "Patient: Dr Priya Nair referred me for a bone density scan. When is the earliest slot?\n",
    "Patient: دكتورة هدى حولتني لعمل أشعة صدر، فين اقرب فرع؟\n",
])
def test_generalized_ordering_referring_variants_never_trigger_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] == "ordering_or_referring"
    assert doctor_validation_needed(call(transcript)) is False


@pytest.mark.parametrize("transcript", [
    # Historical/existing relationship, unseen names.
    "Patient: كنت بتابع مع دكتور عبدالله من زمان، بس دلوقتي عايز اعمل اشعة رنين.\n",
    "Patient: I used to follow up with Dr Karim Haddad. I now need to book a CT scan.\n",
])
def test_generalized_historical_or_existing_doctor_never_triggers_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


@pytest.mark.parametrize("transcript,expected_role", [
    # Genuine booking with unseen names/spellings, both languages,
    # single-word and full names, with typographic variety.
    ("Patient: ابغى احجز مع د. عبدالله ناصر بكرة.\n", "patient_selected"),
    ("Patient: ممكن موعد مع الدكتورة هدى إبراهيم؟\n", "patient_selected"),
    ("Patient: I'd like to book with DR. Priya Nair please.\n", "patient_selected"),
    ("Patient: Can I get an appointment with dr karim haddad?\n", "patient_selected"),
])
def test_generalized_direct_booking_variants_run_validation(transcript, expected_role):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["doctor_role"] == expected_role
    assert doctor_validation_needed(call(transcript)) is True


@pytest.mark.parametrize("transcript", [
    # Genuine inquiries about the named doctor themselves, unseen names.
    "Patient: هل دكتور عبدالله ناصر استشاري؟\n",
    "Patient: What is Dr Priya Nair's specialty?\n",
    "Patient: هل دكتورة هدى إبراهيم تعالج مشاكل الغدة الدرقية؟\n",
    "Patient: Is Dr Karim Haddad available on Sunday?\n",
])
def test_generalized_doctor_inquiry_variants_run_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_inquiry"
    assert doctor_validation_needed(call(transcript)) is True


def test_generalized_agent_recommendation_with_unseen_name_runs_validation():
    transcript = (
        "Patient: طفلي عنده حساسية شديدة في الجلد\n"
        "Patient: مين دكتور مناسب للحالة؟\n"
        "Agent: ممكن نحجز مع دكتور عبدالله ناصر\n"
    )
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert doctor_validation_needed(call(transcript)) is True


def test_generalized_cross_turn_which_doctor_with_unseen_name_runs_validation():
    """The doctor-slot-filling exchange must generalize to any name — not
    just the ones used to originally build this mechanism."""
    transcript = "Patient: ابغى احجز موعد\nAgent: مع دكتور مين؟\nPatient: عبدالله ناصر\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["named_doctor"] == "عبدالله ناصر"
    assert doctor_validation_needed(call(transcript)) is True


# ── Core doctor evidence: name + specialty + scope of service ───────────────
# The doctor feature must not stop at "a name resolved" — for every
# genuinely applicable case it must also carry the CRM's authoritative
# specialty and scope-of-service as part of the resolved evidence (never
# invented when CRM has none — see has_detailed_scope_evidence).

def test_agent_is_doctor_self_introduction_arabic_excludes_and_blocks_validation():
    """Section 13 case 1: the agent handling the call is themselves a
    doctor and opens with a self-introduction — that identity must never
    become the doctor being validated, even though "دكتور"+a name is
    present."""
    transcript = "Agent: السلام عليكم مع حضرتك د محمد من مجموعة أندلسية\nPatient: محتاج علاج طبيعي\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_self_introduction"
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False
    result = validate(transcript)
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["doctor_resolved"] is False


def test_english_doctor_agent_self_introduction_blocks_validation():
    """Section 13 case 2: an English-language self-introduction must be
    excluded exactly like the Arabic equivalent."""
    transcript = "Agent: Hello, this is Dr Ahmed from Andalusia.\nPatient: Is ultrasound available today?\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_self_introduction"
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


def test_external_doctor_recommended_after_self_introduction_resolves_with_specialty_and_scope():
    """Section 13 case 3: the agent introduces themselves as Dr Ahmed, then
    SEPARATELY recommends an external doctor (Dr Mohamed Adel) for the
    patient's complaint. Ahmed must be excluded as a self-introduction;
    Mohamed Adel must be accepted, resolved, and carry both specialty and
    scope-of-service as part of the resolved evidence."""
    transcript = (
        "Agent: This is Dr Ahmed.\n"
        "Patient: I have back pain.\n"
        "Agent: I recommend Dr Mohamed Adel.\n"
    )
    excluded = describe_excluded_doctor_candidates(call(transcript))
    assert any(e["reason"] == "agent_self_introduction" for e in excluded)
    assert not any("adel" in e["name"].lower() for e in excluded)

    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert doctor_validation_needed(call(transcript)) is True

    result = validate(transcript)
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_MOHAMED_ADEL["cr301_doctorkey"]
    scope_ref = result["scope_reference"]
    assert scope_ref["specialty"] == "Orthopedic"
    assert has_detailed_scope_evidence(scope_ref) is True
    assert scope_ref["scope_of_service"] == DOC_MOHAMED_ADEL["cr301_scopeofservice"]


def test_specific_named_doctor_booking_loads_specialty_and_scope():
    """Section 13 case 4: booking with an explicitly named doctor must
    resolve exactly and carry mandatory specialty/scope evidence."""
    transcript = "Patient: I want an appointment with Dr Mohamed Adel.\n"
    result = validate(transcript)
    assert result["outcome"] in ("PASS", "FAIL")
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_MOHAMED_ADEL["cr301_doctorkey"]
    scope_ref = result["scope_reference"]
    assert scope_ref["specialty"] == "Orthopedic"
    assert scope_ref["subspecialty"] == "Spine Surgery"
    assert has_detailed_scope_evidence(scope_ref) is True


def test_doctor_inquiry_validates_against_crm_specialty_and_loads_scope():
    """Section 13 case 5: a factual inquiry about a named doctor must
    resolve them and load scope-of-service evidence even when the specific
    field the patient asked about (specialty) is never itself claimed by
    the Agent in this exchange — the mandatory evidence is attached to
    every resolved doctor, not only when a claim happens to be validated."""
    transcript = "Patient: What specialty is Dr Mohamed Adel?\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "specific_doctor_inquiry"
    result = validate(transcript)
    assert result["doctor_resolved"] is True
    scope_ref = result["scope_reference"]
    assert scope_ref["specialty"] == "Orthopedic"
    assert has_detailed_scope_evidence(scope_ref) is True


def test_clinical_suitability_missing_detailed_scope_stays_explicit_not_invented():
    """Section 13 case 6 / Section 3's mandatory-but-not-invented rule: a
    doctor with only a bare specialty label (no documented
    scope_of_service/_ar text) must resolve normally, but
    has_detailed_scope_evidence must say so plainly rather than treating
    the specialty alone as if it were detailed scope evidence."""
    transcript = (
        "Patient: طفلي عنده مشكلة معينة، هل دكتور محمد الألفي مناسب لحالته؟\n"
    )
    result = validate(transcript)
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_MOHAMED_ELALFY["cr301_doctorkey"]
    scope_ref = result["scope_reference"]
    assert scope_ref["specialty"] == "Orthopedic"
    assert not scope_ref["scope_of_service"]
    assert not scope_ref["scope_of_service_ar"]
    assert has_detailed_scope_evidence(scope_ref) is False


# ── Line-separated doctor lists within a single turn ────────────────────────

def test_extract_doctor_turn_candidates_supports_line_separated_list():
    """A single Agent turn may present more than one named doctor as
    separate lines (e.g. a short list of options) rather than one doctor
    per turn — every plausible name must be extracted, not just the first
    line's."""
    transcript = "Agent: عندنا دكتور خيرية محمد\nدكتور رامي سمرقندي\n"
    patient_candidates, agent_candidates, _ignored = extract_doctor_turn_candidates(call(transcript))
    # "خيرية" normalises (teh-marbuta -> ه) to "خيريه" — compare against the
    # already-normalised candidate form, not the raw transcript spelling.
    assert any("خيريه" in c for c in agent_candidates)
    assert any("رامي" in c for c in agent_candidates)


def test_line_separated_doctor_list_addressee_and_booking_still_generalize():
    """Newline-splitting must not break the existing comma-based
    addressee/booking-attachment semantics — each line is simply another
    clause, evaluated the same way."""
    transcript = "Patient: دكتور رامي سمرقندي\nهل الاشعة متاحة اليوم؟\n"
    ctx = _ctx(transcript)
    # The doctor is being addressed on the first line; the actual question
    # (imaging availability) is a separate line/clause — never itself about
    # the named doctor.
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["doctor_role"] == "conversation_addressee"


# ── Title-anchored NAME extraction vs. specialty/generic-role rejection ─────
# Regression: call 86CE896E-6C79-F111-B337-000D3AA9D4A7 — "الاستشاري اسامة
# عبد السلام" (a title the extractor never even recognised) was silently
# dropped while unrelated "طبيب مخ واعصاب" mentions produced "مختص" garbage,
# which then failed CRM resolution entirely. Fixed generally: استشاري/
# اخصائي/consultant/specialist are now valid name-anchoring titles, and
# "مختص" (plus other degree/generic-role words) can never itself become a
# name. All assertions below use names/wording NOT present in the original
# regression report to prove the fix generalizes.

def test_must_extract_real_doctor_after_consultant_title_not_generic_role():
    """Section: 'Must extract real doctor'."""
    transcript = "Agent: متوفر معنا الاستشاري اسامة عبد السلام وهو ممتاز في تلك الحالات\n"
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] == "اسامه عبد السلام"
    assert "مختص" not in ctx["agent_candidates"]
    assert "مخ واعصاب" not in ctx["agent_candidates"]
    assert "استشاري" not in ctx["agent_candidates"]


def test_must_separate_specialty_context_from_doctor_name():
    """Section: 'Must separate specialty and name' — a bare specialty
    reference with no actual person named must yield doctor_name=None and
    the specialty phrase as separate CONTEXT evidence, never a name
    candidate."""
    transcript = "Patient: نحتاج زيارة طبيب مخ واعصاب\n"
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] is None
    assert ctx["agent_candidates"] == [] and ctx["patient_candidates"] == []
    assert ctx["doctor_context_specialty"] == "مخ واعصاب"
    assert doctor_validation_needed(call(transcript)) is False


def test_must_handle_recommendation_after_specialty_context_unseen_name():
    """Section: 'Must handle recommendation' — an unseen name, never used
    to build this mechanism."""
    transcript = (
        "Patient: نحتاج طبيب مخ واعصاب\n"
        "Agent: متوفر معنا الاستشاري محمد أحمد\n"
    )
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] == "محمد احمد"
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_context_specialty"] == "مخ واعصاب"
    assert doctor_validation_needed(call(transcript)) is True


def test_must_reject_generic_role_mokhtass():
    """Section: 'Must reject generic role'."""
    transcript = "Patient: هل يوجد طبيب مختص؟\n"
    ctx = _ctx(transcript)
    assert ctx["agent_candidates"] == [] and ctx["patient_candidates"] == []
    assert doctor_validation_needed(call(transcript)) is False


def test_self_introduction_plus_external_doctor_unseen_names():
    """Section: 'Self-introduction + external doctor' — unseen names."""
    transcript = (
        "Agent: مع حضرتك دكتور يوسف\n"
        "Patient: عندي ألم شديد\n"
        "Agent: متوفر معنا الاستشاري أحمد محمد للحالة\n"
    )
    evidence = describe_doctor_extraction_evidence(call(transcript))
    assert "يوسف" in evidence["accepted"] or any(
        r["candidate"] == "يوسف" and r["reason"] == "agent_self_introduction" for r in evidence["rejected"]
    )
    assert "يوسف" not in evidence["accepted"]
    assert any("احمد محمد" in c for c in evidence["accepted"])


def test_generic_role_and_specialty_words_never_enter_accepted_candidates():
    """Cross-check every explicitly listed reject example from the task
    generalizes, not just the exact regression phrase."""
    for transcript in [
        "Patient: طبيب مختص\n",
        "Patient: طبيب مخ واعصاب\n",
        "Patient: دكتور عظام\n",
        "Patient: استشاري الأشعة\n",
        "Patient: الطبيب يحدد الخطة\n",
    ]:
        ctx = _ctx(transcript)
        assert ctx["agent_candidates"] == [] and ctx["patient_candidates"] == [], transcript


def test_unseen_recommendation_wordings_all_extract_cleanly():
    """Generalization requirement — arbitrary future phrasings, none of
    which were used to build the fix."""
    cases = [
        ("Agent: متوفر معنا الاستشاري أحمد محمد وهو مناسب للحالة\n", "احمد محمد"),
        ("Agent: أنصحك بدكتورة منى سعيد\n", "مني سعيد"),
        ("Patient: ممكن نحجز لحضرتك مع دكتور رامي أحمد\n", "رامي احمد"),
        ("Agent: We have consultant John Smith available for this case.\n", "john smith"),
    ]
    for transcript, expected_name in cases:
        ctx = _ctx(transcript)
        assert ctx["named_doctor"] == expected_name, transcript


# ── Full regression scenario: call 86CE896E-6C79-F111-B337-000D3AA9D4A7 ────
# The transcript below mirrors the reported production call's structure
# (two agent self-introductions, an ordering/context specialty mention, and
# an agent recommendation using the "الاستشاري" title) purely as a
# regression example — the underlying fix is general (see the unseen-name
# tests above), never keyed to this call ID or these exact names.

REGRESSION_86CE896E_TRANSCRIPT = (
    "Agent: السلام عليكم مع حضرتك دكتور يوسف من مجموعة اندلسية صحة\n"
    "Patient: عندي جلطة واحتاج تقرير\n"
    "Agent: السلام عليكم مع حضرتك د ابانوب من قسم الرعاية المنزلية\n"
    "Agent: في هذه الحالة قد نحتاج لزيارة تقييمية لكتابة تقرير من طبيب مخ واعصاب بناءا على الفحص وتقارير الاشعات\n"
    "ف اذا حابب ممكن نبدا اولا بزيارة طبيب مخ واعصاب للمنزل\n"
    "متوفر معنا الاستشاري اسامة عبد السلام وهو ممتاز في تلك الحالات\n"
    "متوفر عرض حالي على كشفيته بالمنزل تكون ب 1000 ريال سعودي بدلا من 1150 ريال\n"
)


def test_regression_86ce896e_extracts_named_doctor_not_mokhtass():
    ctx = _ctx(REGRESSION_86CE896E_TRANSCRIPT)
    assert ctx["named_doctor"] == "اسامه عبد السلام"
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["doctor_context_specialty"] == "مخ واعصاب"
    assert doctor_validation_needed(call(REGRESSION_86CE896E_TRANSCRIPT)) is True


def test_regression_86ce896e_excludes_both_self_introductions():
    evidence = describe_doctor_extraction_evidence(call(REGRESSION_86CE896E_TRANSCRIPT))
    reasons = {r["candidate"]: r["reason"] for r in evidence["rejected"]}
    assert reasons.get("يوسف") == "agent_self_introduction"
    assert reasons.get("ابانوب") == "agent_self_introduction"
    assert "اسامه عبد السلام" in evidence["accepted"]


def test_regression_86ce896e_resolves_from_crm_not_unresolved():
    result = validate(REGRESSION_86CE896E_TRANSCRIPT)
    assert result["outcome"] != "DOCTOR_UNRESOLVED"
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_OSAMA["cr301_doctorkey"]
    scope_ref = result["scope_reference"]
    assert scope_ref["specialty"] == "Neurology"
    assert has_detailed_scope_evidence(scope_ref) is True
    # Supporting evidence only — never confused with, and never overwriting,
    # the CRM specialty above.
    assert result["doctor_context_specialty"] == "مخ واعصاب"


# ── Follow-up regression: candidate extraction must produce exactly ONE
# genuine name, never a fistful of unrelated sentence fragments pulled from
# elsewhere in a longer call (grammatical particles/verb forms/preposition-
# pronoun fusions, not an enumerable list of "bad phrases") ────────────────

def test_full_semantic_pattern_extracts_single_clean_candidate():
    """The exact semantic pattern reported: a bare specialty mention
    ('طبيب مخ واعصاب') immediately followed by a genuine agent
    recommendation using the 'الاستشاري' title. Must yield exactly ONE
    accepted candidate — the real name — never the specialty phrase, the
    generic title word, or any sentence-fragment noise."""
    transcript = (
        "Agent: قد نحتاج لزيارة طبيب مخ واعصاب\n"
        "Agent: متوفر معنا الاستشاري اسامة عبد السلام وهو ممتاز في تلك الحالات\n"
    )
    ctx = _ctx(transcript)
    assert ctx["named_doctor"] == "اسامه عبد السلام"
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_context_specialty"] == "مخ واعصاب"
    assert ctx["agent_candidates"] == ["اسامه عبد السلام"]

    evidence = describe_doctor_extraction_evidence(call(transcript))
    assert evidence["accepted"] == ["اسامه عبد السلام"]
    rejected_candidates = {r["candidate"] for r in evidence["rejected"]}
    for must_not_be_candidate in ("مخ واعصاب", "استشاري", "مختص"):
        assert must_not_be_candidate not in rejected_candidates or not any(
            c == must_not_be_candidate for c in evidence["accepted"]
        )


@pytest.mark.parametrize("filler_sentence", [
    # Grammatical-particle / verb-form noise that must never survive as a
    # name candidate, regardless of what unrelated title word precedes it —
    # each exercises a DIFFERENT closed grammatical category (subordinating
    # conjunction, preposition+pronoun fusion, purpose-lam + verb), not one
    # specific phrase.
    "Agent: هيتم تحديد الاستشاري حسب الحالة بالتنسيق مع الفرع\n",
    "Agent: هنرسل الحالة للاخصائي لتتطلع عليها\n",
    "Agent: الاخصائي اذا تحب هيتواصل معاك\n",
    "Agent: الدكتور لتحديد الخطة العلاجية المناسبة\n",
])
def test_grammatical_filler_sentences_never_produce_a_candidate(filler_sentence):
    ctx = _ctx(filler_sentence)
    assert ctx["agent_candidates"] == []
    assert ctx["patient_candidates"] == []


def test_full_semantic_pattern_generalizes_to_unseen_names_and_specialties():
    """Same structural pattern (specialty mention -> consultant-title
    recommendation), completely different names/specialty/wording than any
    example used to build the fix."""
    cases = [
        (
            "Agent: هنحتاج زيارة دكتورة نساء وولادة\n"
            "Agent: متوفر معنا الاستشاري أحمد محمود وهو ممتاز في الحالات دي\n",
            "احمد محمود", "نساء وولاده",
        ),
        (
            "Agent: أنصح حضرتك بدكتورة منى سعيد\n",
            "مني سعيد", None,
        ),
        (
            "Patient: ممكن نبدأ بزيارة دكتور رامي أحمد\n",
            "رامي احمد", None,
        ),
        (
            "Agent: We have consultant John Smith available for this condition.\n",
            "john smith", None,
        ),
    ]
    for transcript, expected_name, expected_specialty in cases:
        ctx = _ctx(transcript)
        assert ctx["named_doctor"] == expected_name, transcript
        assert ctx["agent_candidates"] + ctx["patient_candidates"] == [expected_name], transcript
        if expected_specialty is not None:
            assert ctx["doctor_context_specialty"] == expected_specialty, transcript


# ── CRM resolution layer: shared-family-name collision must not outvote a
# genuine exact match (regression: call 86CE896E-6C79-F111-B337-000D3AA9D4A7,
# "اسامة عبد السلام" resolved to source=outside_authoritative_scope,
# candidate_count=4, outcome=AMBIGUOUS_DOCTOR — the extraction had already
# produced the single correct candidate; the CRM resolver's tier-3 overlap
# rule was matching unrelated doctors who merely shared the "عبد السلام"
# family-name component). ───────────────────────────────────────────────────

DOC_KHALED_ABDELSALAM = {
    "cr301_doctorkey": "900900011", "servhub_doctornameen": "Khaled Abdelsalam",
    "cr301_doctornamear": "خالد عبد السلام", "cr301_degreename": "Specialist",
    "cr301_specialtyname": "Cardiology", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AKW", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
DOC_MONA_ABDELSALAM = {
    "cr301_doctorkey": "900900022", "servhub_doctornameen": "Mona Abdelsalam",
    "cr301_doctornamear": "منى عبد السلام", "cr301_degreename": "Specialist",
    "cr301_specialtyname": "Dermatology", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AKW", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
DOC_TAREK_ABDELSALAM = {
    "cr301_doctorkey": "900900033", "servhub_doctornameen": "Tarek Abdelsalam",
    "cr301_doctornamear": "طارق عبد السلام", "cr301_degreename": "Specialist",
    "cr301_specialtyname": "Orthopedic", "cr301_subspecialtyname": None,
    "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
}
FAMILY_NAME_COLLISION_POOL = [DOC_OSAMA, DOC_KHALED_ABDELSALAM, DOC_MONA_ABDELSALAM, DOC_TAREK_ABDELSALAM]


def test_shared_family_name_does_not_outvote_exact_match():
    """Four CRM doctors share the family-name component 'عبد السلام'; only
    one ('اسامة عبد السلام') is an EXACT match for the query. The other
    three must never compete via the weaker overlap-based partial-match
    tier just because they happen to share 2 of 3 tokens."""
    transcript = "Agent: متوفر معنا الاستشاري اسامة عبد السلام وهو ممتاز في تلك الحالات\n"
    result = validate(transcript, doctors=FAMILY_NAME_COLLISION_POOL)
    assert result["outcome"] != "AMBIGUOUS_DOCTOR"
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == DOC_OSAMA["cr301_doctorkey"]
    assert result["candidate_count"] == 1


def test_resolve_doctor_candidates_partial_tier_requires_query_first_token():
    """Direct unit check on the resolver: the family-name-only doctors
    must not appear in the tier-3 partial-match result at all."""
    found = resolve_doctor_candidates("اسامة عبد السلام", FAMILY_NAME_COLLISION_POOL, allow_fuzzy=False)
    keys = {r["cr301_doctorkey"] for r in found}
    assert keys == {DOC_OSAMA["cr301_doctorkey"]}


def test_genuine_same_name_different_doctors_still_reports_ambiguous():
    """Case B — two GENUINELY distinct doctor keys share the exact same
    full name with no BU discriminator: AMBIGUOUS_DOCTOR must still be
    reported, never forced to an arbitrary pick."""
    duplicate_name_doctors = [
        {**DOC_OSAMA, "cr301_doctorkey": "111", "cr18c_buname": "AHJ", "cr301_specialtyname": "Neurology"},
        {**DOC_OSAMA, "cr301_doctorkey": "222", "cr18c_buname": "AKW", "cr301_specialtyname": "Cardiology"},
    ]
    transcript = "Agent: متوفر معنا الاستشاري اسامة عبد السلام\n"
    result = validate(transcript, doctors=duplicate_name_doctors)
    assert result["outcome"] == "AMBIGUOUS_DOCTOR"
    assert result["candidate_count"] == 2


# ── General person-name matching regression tests (not this-doctor-specific)
# Regression: "اسامة عبد السلام" wrongly matched "أسامة عبد المقصود" /
# "أسامة عبد المقصود تحضيرية طارق الجمل" purely because the given name
# ("اسامة") and the common compound-name component ("عبد") overlapped —
# the actual FAMILY name never agreed. Fixed generally via ordered-prefix
# and given+family-name-endpoint matching (see _is_plausible_same_person_
# name) — these tests use arbitrary names to prove it, not the one
# production doctor.

def _minimal_doctor(key: str, name_ar: str, name_en: str = "") -> dict:
    return {
        "cr301_doctorkey": key, "servhub_doctornameen": name_en, "cr301_doctornamear": name_ar,
        "cr301_degreename": "Consultant", "cr301_specialtyname": "General", "cr301_subspecialtyname": None,
        "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
        "cr18c_buname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
        "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
        "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
        "servhub_examinationage": None, "cr301_walkinconsultationfees": None,
    }


@pytest.mark.parametrize("requested,candidate_name", [
    ("اسامة عبد السلام", "أسامة عبد المقصود"),
    ("محمد احمد", "محمد علي"),
    ("احمد عبد الرحمن", "احمد عبد العزيز"),
    ("يوسف محمد علي", "يوسف محمد حسن"),
])
def test_conflicting_discriminating_token_never_matches(requested, candidate_name):
    doctor = _minimal_doctor("k1", candidate_name)
    found = resolve_doctor_candidates(requested, [doctor], allow_fuzzy=True)
    assert found == []


@pytest.mark.parametrize("requested,candidate_name", [
    ("أسامة عبد السلام", "اسامه عبد السلام"),   # hamza + teh-marbuta variant
    ("اسامة عبد السلام", "أسامة عبد السلام"),   # reverse direction
])
def test_normalization_variants_still_match(requested, candidate_name):
    doctor = _minimal_doctor("k1", candidate_name)
    found = resolve_doctor_candidates(requested, [doctor], allow_fuzzy=True)
    assert len(found) == 1 and found[0]["cr301_doctorkey"] == "k1"


def test_coherent_expanded_name_matches_via_ordered_containment():
    """'محمد أحمد' (given + father's name only) must still resolve
    'محمد أحمد علي' (the fuller legal name) — a genuine ordered prefix."""
    doctor = _minimal_doctor("k1", "محمد أحمد علي")
    found = resolve_doctor_candidates("محمد أحمد", [doctor], allow_fuzzy=True)
    assert len(found) == 1 and found[0]["cr301_doctorkey"] == "k1"


def test_omitted_middle_name_still_matches_via_given_and_family_endpoints():
    """A genuinely omitted MIDDLE name ('علياء المدبولي' skipping the
    father's name 'محمود') must still resolve, since both the given name
    and the actual family name agree."""
    doctor = _minimal_doctor("k1", "علياء محمود المدبولي")
    found = resolve_doctor_candidates("علياء المدبولي", [doctor], allow_fuzzy=True)
    assert len(found) == 1 and found[0]["cr301_doctorkey"] == "k1"


def test_unrelated_extra_tokens_never_create_false_positive_match():
    """A candidate name padded with unrelated extra tokens after a
    genuinely conflicting discriminating token must still be rejected —
    more noise must never make a bad match look better."""
    doctor = _minimal_doctor("k1", "أسامة عبد المقصود تحضيرية طارق الجمل")
    found = resolve_doctor_candidates("اسامة عبد السلام", [doctor], allow_fuzzy=True)
    assert found == []


# ═════════════════════════════════════════════════════════════════════════
# Multi-doctor recommendation sets — regression: an Agent turn recommending
# SEVERAL named doctors for the SAME clinical need was silently collapsed
# to a single named_doctor (the extractor even leaked "you want to book"
# from an unrelated later turn as the "chosen" one). Fixed generally at the
# classifier level (named_doctor_candidates/recommended_doctors/
# patient_selected_doctor) and threaded through resolution/validation
# (validate_doctor_information's "doctors" list) and scope evaluation
# (infer_doctor_scope_validation's per-doctor LLM calls). None of the
# names/wording below are the production call's — every case here proves
# the mechanism generalizes.
# ═════════════════════════════════════════════════════════════════════════

MULTI_DOCTOR_DR_BACKSLASH_TRANSCRIPT = (
    "Patient: I have hip pain radiating to my leg.\n"
    "Agent: I can recommend:\n"
    "DR \\ Mohamed Adel Belkadhi\n"
    "DR \\ Ameer Elsayed\n"
    "DR \\ ELSAYED SHAHEEN\n"
    "DR \\ Mahmoud ElSayed Elbadawy Ismail\n"
    "Agent: Who is the doctor you want to book with ?\n"
    "Patient: I will choose later.\n"
)


def test_multi_doctor_recommendation_extracts_all_four_not_first_only():
    ctx = _ctx(MULTI_DOCTOR_DR_BACKSLASH_TRANSCRIPT)
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["doctor_intent"] == "specific_doctor_booking"
    assert ctx["recommended_doctors"] == [
        "mohamed adel belkadhi", "ameer elsayed", "elsayed shaheen", "mahmoud elsayed elbadawy ismail",
    ]
    assert ctx["named_doctor_candidates"] == ctx["recommended_doctors"]
    assert ctx["patient_selected_doctor"] is None
    assert "you want to book" not in ctx["named_doctor_candidates"]
    assert doctor_validation_needed(call(MULTI_DOCTOR_DR_BACKSLASH_TRANSCRIPT)) is True


@pytest.mark.parametrize("transcript,expected_names", [
    (
        "Agent: Here are the options:\nDr Ahmed Ali\nDr Mohamed Hassan\nDr Khaled Adel\n",
        ["ahmed ali", "mohamed hassan", "khaled adel"],
    ),
    (
        "Agent: Here are the options:\n1. Dr Ahmed Ali\n2. Dr Mohamed Hassan\n3. Dr Khaled Adel\n",
        ["ahmed ali", "mohamed hassan", "khaled adel"],
    ),
    (
        # "علي" is a pre-existing Arabic homograph limitation (also reads as
        # the preposition "على"/"علي", already blocked as a non-name
        # continuation word) — "سالم علي" truncates to "سالم" here, unrelated
        # to the multi-doctor mechanism itself, which correctly still finds
        # all THREE separate list entries.
        "Agent: نرشح لك:\nد. أحمد محمد\nد. خالد حسن\nد. سالم علي\n",
        ["احمد محمد", "خالد حسن", "سالم"],
    ),
])
def test_multi_doctor_various_list_formats_extract_all_names(transcript, expected_names):
    ctx = _ctx(transcript)
    assert ctx["doctor_role"] == "agent_recommended"
    assert ctx["recommended_doctors"] == expected_names


def test_multi_doctor_resolves_and_validates_each_independently():
    """One doctor unresolved, two resolved — the unresolved one must
    surface explicitly, never silently dropped, and must never turn the
    whole set's outcome into a false PASS."""
    doctors = [
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m1", "servhub_doctornameen": "Ahmed Ali", "cr301_doctornamear": "احمد علي"},
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m2", "servhub_doctornameen": "Mohamed Hassan", "cr301_doctornamear": "محمد حسان"},
        # "Khaled Adel" intentionally absent from CRM.
    ]
    transcript = "Agent: Here are the options:\nDr Ahmed Ali\nDr Mohamed Hassan\nDr Khaled Adel\n"
    result = validate(transcript, doctors=doctors)
    assert result["recommended_doctor_count"] == 3
    assert len(result["doctors"]) == 3
    by_name = {d["input_name"]: d for d in result["doctors"]}
    assert by_name["ahmed ali"]["doctor_resolved"] is True
    assert by_name["mohamed hassan"]["doctor_resolved"] is True
    assert by_name["khaled adel"]["doctor_resolved"] is False
    assert by_name["khaled adel"]["outcome"] == "DOCTOR_UNRESOLVED"
    # The unresolved doctor must be reflected in the AGGREGATE outcome —
    # never silently dropped, never overridden by the first doctor's PASS.
    assert result["outcome"] == "DOCTOR_UNRESOLVED"
    assert "khaled adel" in result["reason"]


def test_multi_doctor_all_resolved_and_passing_aggregates_to_pass():
    doctors = [
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m1", "servhub_doctornameen": "Ahmed Ali", "cr301_doctornamear": "احمد علي"},
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m2", "servhub_doctornameen": "Mohamed Hassan", "cr301_doctornamear": "محمد حسان"},
    ]
    transcript = "Agent: Here are the options:\nDr Ahmed Ali\nDr Mohamed Hassan\n"
    result = validate(transcript, doctors=doctors)
    assert result["outcome"] == "PASS"
    assert all(d["outcome"] == "PASS" for d in result["doctors"])


def test_patient_selecting_one_doctor_afterward_narrows_to_single_target():
    transcript = (
        "Agent: I can recommend:\nDr Ahmed Ali\nDr Mohamed Hassan\nDr Khaled Adel\n"
        "Patient: Book Dr Mohamed Hassan please.\n"
    )
    ctx = _ctx(transcript)
    assert ctx["patient_selected_doctor"] == "mohamed hassan"
    assert ctx["named_doctor_candidates"] == ["mohamed hassan"]
    assert ctx["doctor_role"] == "patient_selected"
    # Recommendation history is preserved even after narrowing.
    assert ctx["recommended_doctors"] == ["ahmed ali", "mohamed hassan", "khaled adel"]

    doctors = [{**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m2", "servhub_doctornameen": "Mohamed Hassan", "cr301_doctornamear": "محمد حسان"}]
    result = validate(transcript, doctors=doctors)
    # Only ONE doctor is validated — the selected one, not all three.
    assert "doctors" not in result or len(result.get("doctors", [result])) <= 1
    assert result["doctor_resolved"] is True
    assert result["doctor_key"] == "m2"


def test_no_selection_yet_still_validates_full_recommendation_set():
    ctx = _ctx(MULTI_DOCTOR_DR_BACKSLASH_TRANSCRIPT)
    assert ctx["patient_selected_doctor"] is None
    assert len(ctx["recommended_doctors"]) == 4
    assert doctor_validation_needed(call(MULTI_DOCTOR_DR_BACKSLASH_TRANSCRIPT)) is True


@pytest.mark.parametrize("transcript", [
    "Patient: I need MRI\nPatient: CT\nPatient: Ultrasound\n",
    "Agent: We offer MRI, CT and Ultrasound services.\n",
])
def test_non_doctor_service_list_never_enters_doctor_validation(transcript):
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert ctx["recommended_doctors"] == []
    assert doctor_validation_needed(call(transcript)) is False


@pytest.mark.parametrize("transcript", [
    "Patient: I need an orthopedic doctor.\n",
    "Patient: I need a female doctor.\n",
    "Patient: I need a male doctor.\n",
    "Patient: I need a physiotherapist.\n",
    "Agent: Please clarify so I can refer you to the appropriate doctor.\n",
    "Agent: Who is the doctor?\n",
])
def test_generic_role_phrases_never_become_named_doctors(transcript):
    ctx = _ctx(transcript)
    assert ctx["agent_candidates"] == [] and ctx["patient_candidates"] == []


def test_existing_ordering_semantic_target_regression_preserved():
    """Preserve the pre-existing semantic-target skip: a doctor who merely
    ORDERED a service must not trigger doctor validation just because a
    multi-doctor detector now also exists."""
    transcript = "Patient: Dr Ahmed ordered an MRI; I want to book the MRI.\n"
    ctx = _ctx(transcript)
    assert ctx["doctor_intent"] == "not_applicable"
    assert doctor_validation_needed(call(transcript)) is False


def test_multi_doctor_crm_fetched_once(monkeypatch):
    """validate_doctor_information itself never re-fetches/re-filters the
    CRM pool per recommended doctor — the caller passes the already-fetched
    doctor_records list once; this test asserts dedupe_doctors (the pool-
    building step) runs exactly once regardless of recommendation size."""
    import app.service_hub.doctor_validation as dv

    calls = {"count": 0}
    orig = dv.dedupe_doctors

    def counting_dedupe(records):
        calls["count"] += 1
        return orig(records)

    monkeypatch.setattr(dv, "dedupe_doctors", counting_dedupe)

    doctors = [
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m1", "servhub_doctornameen": "Ahmed Ali", "cr301_doctornamear": "احمد علي"},
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m2", "servhub_doctornameen": "Mohamed Hassan", "cr301_doctornamear": "محمد حسان"},
        {**DOC_MOHAMED_ADEL, "cr301_doctorkey": "m3", "servhub_doctornameen": "Khaled Adel", "cr301_doctornamear": "خالد عادل"},
    ]
    transcript = "Agent: Here are the options:\nDr Ahmed Ali\nDr Mohamed Hassan\nDr Khaled Adel\n"
    dv.validate_doctor_information(call(transcript), doctors)
    assert calls["count"] == 1
