from app.models.input import CallTranscript
from app.service_hub.bank_validation import (
    bank_validation_needed,
    detect_arabic_bank_request,
    is_supported_bank_bu,
    resolve_business_unit,
    validate_ksa_bank_information,
)

# Fixtures mirror the real cr301_bankaccounts shape verified against a LIVE
# CRM query this session: cr301_bu / servhub_bulookupname / cr18c_bunname
# use short codes (AHJ, not LIVE); owningbusinessunitname is a constant
# ("ERS") on every row and is never used for BU filtering.
BANK_AKW = {
    "cr301_bu": "AKW", "servhub_bulookupname": "AKW", "cr18c_bunname": "AKW", "owningbusinessunitname": "ERS",
    "cr301_bankname": "البنك السعودي الفرنسي", "cr301_accountnumber": "99453100124",
    "cr301_ibannumber": "SA5655000000099453100124", "cr301_accountowner": "Andalusia Arab medical clinic",
    "statecodename": "Active",
}
BANK_ALW = {
    "cr301_bu": "ALW", "servhub_bulookupname": "ALW", "cr18c_bunname": "ALW", "owningbusinessunitname": "ERS",
    "cr301_bankname": "بنك الخليج الدولي", "cr301_ibannumber": "SA4590000000020000004551",
    "cr301_accountnumber": "20000004551", "cr301_accountowner": "Arabian Andalusia CO",
    "statecodename": "Active",
}
# AHJ / LIVE — two accounts at different banks (a BU is a LIST of eligible
# accounts, never assumed to be exactly one).
BANK_AHJ_ALAHLI = {
    "cr301_bu": "AHJ", "servhub_bulookupname": "AHJ", "cr18c_bunname": "AHJ", "owningbusinessunitname": "ERS",
    "cr301_bankname": "البنك الاهلي", "cr301_accountnumber": "62212527000103",
    "cr301_ibannumber": "SA7910000062212527000103", "cr301_accountowner": "مستشفى أندلسية",
    "statecodename": "Active",
}
BANK_AHJ_RIYAD = {
    "cr301_bu": "AHJ", "servhub_bulookupname": "AHJ", "cr18c_bunname": "AHJ", "owningbusinessunitname": "ERS",
    "cr301_bankname": "بنك الرياض", "cr301_ibannumber": "SA9320000001251109369901",
    "cr301_accountowner": "مستشفى أندلسية", "statecodename": "Active",
}
BANK_AFW = {
    "cr301_bu": "AFW", "servhub_bulookupname": "AFW", "cr18c_bunname": "AFW", "owningbusinessunitname": "ERS",
    "cr301_bankname": "بنك الراجحي", "cr301_ibannumber": "SA9880000176608010866661",
    "cr301_accountowner": "شركة اندلسية العربية للخدمات الطبية", "statecodename": "Active",
}
ALL_BANKS = [BANK_AKW, BANK_ALW, BANK_AHJ_ALAHLI, BANK_AHJ_RIYAD, BANK_AFW]


def call(transcript):
    return CallTranscript(call_id="bank-1", agent_name="Agent", Patient_Phone="501234567", call_date="2026-08-24", call_duration_seconds=1, department="Scheduling", transcript=transcript)


# ── Business Unit Detection ─────────────────────────────────────────────────

def test_bu_detection_akw():
    assert resolve_business_unit("صحة الطفل") == "AKW"
    assert resolve_business_unit("الطفل") == "AKW"
    assert resolve_business_unit("BU-AKW") == "AKW"


def test_bu_detection_alw():
    assert resolve_business_unit("صحة المرأة") == "ALW"
    assert resolve_business_unit("المرأة") == "ALW"
    assert resolve_business_unit("BU-ALW") == "ALW"


def test_bu_detection_ahj_resolves_to_live():
    """Per BUSINESS_UNIT_KEYWORD_MAP's own convention, AHJ's app-level
    canonical value is "LIVE" — the CRM's own value is "AHJ" (verified via a
    live query); the validator bridges the two via _APP_BU_TO_CRM_BU, not by
    forcing "AHJ" as the canonical app value."""
    assert resolve_business_unit("جدة") == "LIVE"
    assert resolve_business_unit("حي الجامعة") == "LIVE"
    assert resolve_business_unit("BU-AHJ") == "LIVE"


def test_bu_detection_typo_tolerance_via_fuzzy_fallback():
    """Multi-word alias with a tatweel inserted — char-level normalisation
    alone doesn't fix this, only the fuzzy fallback does."""
    assert resolve_business_unit("صحة الطفـل") == "AKW"


def test_bu_detection_unrelated_text_does_not_resolve():
    assert resolve_business_unit("عايز احجز موعد بكرة الساعة خمسة") is None


def test_bu_detection_afw_explicit_forms():
    """AFW has no Arabic alias anywhere in the project (verified) — only
    explicit forms are supported, per Step 3's explicit instruction not to
    invent Arabic terminology."""
    assert resolve_business_unit("BU-AFW") == "AFW"
    assert resolve_business_unit("AFW") == "AFW"
    assert is_supported_bank_bu("AFW")


# ── Bank-validation scope ───────────────────────────────────────────────────

def test_scope_excludes_lch_mkr_snb_but_keeps_them_valid_bus_elsewhere():
    from app.models.input import BUSINESS_UNIT_KEYWORD_MAP
    assert resolve_business_unit("فرع سلطان") == "LCH"
    assert not is_supported_bank_bu("LCH")
    assert not is_supported_bank_bu("MKR")
    assert not is_supported_bank_bu("SNB")
    # LCH/MKR/SNB remain untouched in the global mapping used elsewhere.
    assert BUSINESS_UNIT_KEYWORD_MAP["فرع سلطان"] == "LCH"


def test_unsupported_bu_is_not_applicable_not_a_violation():
    result = validate_ksa_bank_information(
        call("Patient: ابغى رقم حساب فرع سلطان\nAgent: رقم الحساب 123456"), ALL_BANKS,
    )
    assert result["outcome"] == "NOT_APPLICABLE"
    assert result["is_violation"] is False


# ── Business Unit mention alone is not a bank request (Step 7) ─────────────

def test_bu_mention_without_bank_intent_is_not_applicable():
    c = call("Patient: عايز احجز موعد في صحة الطفل\nAgent: تمام، امتى يناسبك؟")
    assert resolve_business_unit("عايز احجز موعد في صحة الطفل") == "AKW"
    assert not bank_validation_needed(c)
    result = validate_ksa_bank_information(c, ALL_BANKS)
    assert result["outcome"] == "NOT_APPLICABLE"


# ── Intent Detection ─────────────────────────────────────────────────────────

def test_arabic_dialect_bank_intents_and_non_bank_payment():
    for phrase in ("ابي الايبان", "أبغى الآيبان", "وش رقم الحساب؟", "وين احول؟", "على اي حساب احول؟", "ممكن ترسل IBAN حق فرع الرياض؟", "اسم البنك ايه"):
        assert detect_arabic_bank_request(phrase)
    assert not detect_arabic_bank_request("هل أقدر أدفع فيزا؟")


def test_generic_account_word_without_bank_context_is_not_detected():
    """Bare 'حساب' (e.g. a social-media account) must not false-positive."""
    assert not detect_arabic_bank_request("عندي حساب في تويتر مش شغال")


# ── Validation — identifiers, bank name, account owner ──────────────────────

def test_correct_akw_iban_passes_and_is_masked():
    result = validate_ksa_bank_information(
        call("Patient: ممكن ايبان صحة الطفل؟\nAgent: SA5655000000099453100124"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"
    assert result["requested_business_unit"] == "AKW"
    assert result["provided_identifiers"] == ["********************0124"]


def test_wrong_akw_iban_and_one_digit_difference_both_fail():
    wrong = validate_ksa_bank_information(
        call("Patient: ممكن ايبان صحة الطفل؟\nAgent: SA9999999999999999999999"), ALL_BANKS,
    )
    one_digit_off = validate_ksa_bank_information(
        call("Patient: ممكن ايبان صحة الطفل؟\nAgent: SA5655000000099453100125"), ALL_BANKS,  # last digit 4->5
    )
    assert wrong["outcome"] == "INCORRECT_BANK_INFORMATION"
    assert one_digit_off["outcome"] == "INCORRECT_BANK_INFORMATION"


def test_valid_alw_iban_used_for_akw_request_is_wrong_business_unit():
    """Step 14 — never validate against another BU. This IBAN is genuinely
    valid, just for ALW, not the requested AKW."""
    result = validate_ksa_bank_information(
        call("Patient: ممكن ايبان صحة الطفل؟\nAgent: SA4590000000020000004551"), ALL_BANKS,
    )
    assert result["outcome"] == "WRONG_BUSINESS_UNIT_ACCOUNT"
    assert result["is_violation"] is True


def test_correct_and_incorrect_account_number():
    correct = validate_ksa_bank_information(
        call("Patient: ممكن رقم حساب صحة الطفل؟\nAgent: رقم الحساب 99453100124"), ALL_BANKS,
    )
    incorrect = validate_ksa_bank_information(
        call("Patient: ممكن رقم حساب صحة الطفل؟\nAgent: رقم الحساب 11111111111"), ALL_BANKS,
    )
    assert correct["outcome"] == "PASS"
    assert incorrect["outcome"] == "INCORRECT_BANK_INFORMATION"


def test_correct_and_wrong_bank_name():
    correct = validate_ksa_bank_information(
        call("Patient: اسم البنك ايه لصحة الطفل؟\nAgent: البنك السعودي الفرنسي"), ALL_BANKS,
    )
    wrong = validate_ksa_bank_information(
        call("Patient: اسم البنك ايه لصحة الطفل؟\nAgent: بنك الراجحي"), ALL_BANKS,
    )
    assert correct["outcome"] == "PASS"
    assert wrong["outcome"] == "INCORRECT_BANK_INFORMATION"


def test_correct_account_owner():
    result = validate_ksa_bank_information(
        call("Patient: الحساب باسم مين لصحة الطفل؟\nAgent: الحساب باسم Andalusia Arab medical clinic"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"
    wrong = validate_ksa_bank_information(
        call("Patient: الحساب باسم مين لصحة الطفل؟\nAgent: الحساب باسم شركة تانية خالص"), ALL_BANKS,
    )
    assert wrong["outcome"] == "INCORRECT_BANK_INFORMATION"


# ── Requested Field ──────────────────────────────────────────────────────────

def test_iban_requested_and_provided_validates_normally():
    result = validate_ksa_bank_information(
        call("Patient: ممكن الايبان بتاع صحة الطفل؟\nAgent: SA5655000000099453100124"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"


def test_iban_requested_bank_name_only_is_wrong_field_type():
    """Step 18 — naming the right bank does not answer an IBAN request."""
    result = validate_ksa_bank_information(
        call("Patient: ممكن رقم الآيبان لصحة الطفل؟\nAgent: الحساب في البنك السعودي الفرنسي"), ALL_BANKS,
    )
    assert result["outcome"] == "WRONG_FIELD_TYPE_PROVIDED"
    assert result["is_violation"] is True


def test_account_number_requested_iban_only_is_wrong_field_type():
    result = validate_ksa_bank_information(
        call("Patient: ابغى رقم حساب صحة الطفل\nAgent: الآيبان SA5655000000099453100124"), ALL_BANKS,
    )
    assert result["outcome"] == "WRONG_FIELD_TYPE_PROVIDED"


def test_requested_bank_data_no_answer_at_all():
    result = validate_ksa_bank_information(
        call("Patient: ممكن رقم حساب صحة الطفل؟\nAgent: لحظة من فضلك"), ALL_BANKS,
    )
    assert result["outcome"] == "BANK_INFO_REQUESTED_NOT_PROVIDED"
    assert result["is_violation"] is True


# ── Multiple Accounts (AHJ / LIVE has two) ──────────────────────────────────

def test_multiple_accounts_exact_iban_resolves():
    result = validate_ksa_bank_information(
        call("Patient: ابغى ايبان حي الجامعة\nAgent: SA9320000001251109369901"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"


def test_multiple_accounts_exact_account_number_resolves():
    result = validate_ksa_bank_information(
        call("Patient: ابغى رقم حساب حي الجامعة\nAgent: رقم الحساب 62212527000103"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"


def test_multiple_accounts_bank_name_disambiguates():
    result = validate_ksa_bank_information(
        call("Patient: ابغى ايبان بنك الرياض لحي الجامعة\nAgent: SA9320000001251109369901"), ALL_BANKS,
    )
    assert result["outcome"] == "PASS"


def test_multiple_accounts_ambiguous_when_unresolved():
    result = validate_ksa_bank_information(
        call("Patient: ابغى ايبان حي الجامعة\nAgent: تمام لحظة من فضلك"), ALL_BANKS,
    )
    assert result["outcome"] == "AMBIGUOUS_MULTIPLE_ACCOUNTS"
    assert result["is_violation"] is False  # a system-data condition, not agent misconduct


# ── Infrastructure / context ─────────────────────────────────────────────────

def test_multi_message_agent_answer_is_aggregated():
    transcript = (
        "Patient: ابغى ايبان بنك الرياض لحي الجامعة\n"
        "Agent: تمام لحظة\n"
        "Agent: البنك المطلوب هو بنك الرياض\n"
        "Agent: والآيبان SA9320000001251109369901"
    )
    result = validate_ksa_bank_information(call(transcript), ALL_BANKS)
    assert result["outcome"] == "PASS"


def test_crm_unavailable_is_not_an_agent_violation():
    """Empty bank_records (as if the CRM fetch failed/returned nothing) must
    degrade to a system-data outcome, never a compliance violation."""
    result = validate_ksa_bank_information(
        call("Patient: ممكن ايبان صحة الطفل؟\nAgent: SA5655000000099453100124"), [],
    )
    assert result["outcome"] == "NO_APPROVED_BANK_ACCOUNT_FOUND"
    assert result["is_violation"] is False


def test_no_bank_intent_is_not_applicable():
    c = call("Patient: عايزة اسأل عن مواعيد صحة الطفل بس\nAgent: تحت أمرك")
    assert not bank_validation_needed(c)


def test_unrelated_phone_number_does_not_trigger_bank_validation():
    """An ordinary phone number must never look like a bank request."""
    c = call("Patient: ابغى اتواصل معاكم بخصوص صحة الطفل\nAgent: تقدر تتواصل معنا على 0551234567")
    assert not bank_validation_needed(c)
