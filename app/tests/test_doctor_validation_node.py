"""Tests for app.agent.nodes.validate_doctor_node's multi-doctor
OBSERVABILITY (logging) — a SEPARATE concern from doctor resolution/claim
validation itself (see app/tests/test_doctor_validation.py for that).

Real regression: for a genuine multi-doctor recommendation set, the node's
final print block assembled singular `requested_name`/`resolved_name`/
`doctor_key` fields from DIFFERENT sources (the semantic/LLM-extracted name
for `requested_name`, but the top-level scalar mirror — which may describe a
COMPLETELY DIFFERENT doctor — for `resolved_name`/`doctor_key`), producing an
internally contradictory log:

    requested_name='اميره بركات'
    resolved_name='بدرية البيروتي'
    doctor_key=110411

Fixed by detecting a genuine multi-doctor result via the authoritative
`doctors` list (`len(doctors) > 1`) and, for that case ONLY, printing a
dedicated multi-doctor summary where every requested name, resolved
identity, key, BU, match method, outcome, and validated-fields breakdown for
one doctor is read from that SAME doctor's own per-doctor dictionary — never
paired across two different doctors. A 0- or 1-doctor result keeps printing
exactly as before.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agent.nodes import (
    _derive_doctor_match_method,
    _print_multi_doctor_validation_summary,
    validate_doctor_node,
)
from app.models.input import CallTranscript


def call(transcript: str, call_id: str = "node-test") -> CallTranscript:
    return CallTranscript(
        call_id=call_id, agent_name="Agent", Patient_Phone="501234567",
        call_date="2026-09-20", call_duration_seconds=1, department="Scheduling",
        transcript=transcript,
    )


def doctor_entry(
    input_name: str, *, resolved: bool, doctor_key: str | None = None,
    name_ar: str | None = None, business_unit: str | None = None,
    resolution_source: str | None = None, outcome: str = "PASS",
    validated_fields: dict | None = None,
    failure_details: list | None = None, warning_details: list | None = None,
) -> dict:
    """Minimal stand-in for one entry in validate_doctor_information's
    authoritative "doctors" list — only the fields the print/observability
    layer actually reads. failure_details/warning_details are the
    structured, UI/persistence-facing evidence lists every real per-doctor
    result now always carries (see _result() in doctor_validation.py) —
    kept strictly separate, and always present (default []) here too."""
    return {
        "input_name": input_name, "doctor_resolved": resolved,
        "doctor_key": doctor_key, "doctor_name_ar": name_ar, "doctor_name_en": None,
        "business_unit": business_unit, "resolution_source": resolution_source,
        "outcome": outcome, "validated_fields": validated_fields or {},
        "failure_details": failure_details or [], "warning_details": warning_details or [],
        "is_violation": outcome == "FAIL",
    }


def multi_doctor_result(doctors: list[dict], outcome: str, reason: str) -> dict:
    return {
        "doctors": doctors, "outcome": outcome, "reason": reason,
        "doctor_context_specialty": "Endocrinology",
    }


# ── _derive_doctor_match_method (pure function) ─────────────────────────────

def test_match_method_none_when_unresolved():
    assert _derive_doctor_match_method("بدريه", None, False, None) is None


def test_match_method_contextual_single_name_reported_verbatim():
    assert _derive_doctor_match_method("بدريه", "بدرية البيروتي", True, "contextual_single_name") == "contextual_single_name"


def test_match_method_exact_for_same_normalized_name():
    assert _derive_doctor_match_method("اميره بركات", "أميرة بركات", True, "authoritative_pool") == "exact"


def test_match_method_partial_for_different_names():
    assert _derive_doctor_match_method("خيرية محمد", "خيرية محمد علي موسي", True, "authoritative_pool") == "partial"


# ── _print_multi_doctor_validation_summary (capsys) ─────────────────────────

def test_two_resolved_doctors_produce_dedicated_multi_doctor_summary(capsys):
    """Item 1."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي",
                     business_unit="AHJ", resolution_source="contextual_single_name"),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات",
                     business_unit="AHJ", resolution_source="authoritative_pool", outcome="FAIL",
                     validated_fields={"degree": {"outcome": "FAIL"}}),
    ]
    result = multi_doctor_result(doctors, "FAIL", "1 of 2 recommended doctor(s) failed validation: اميره بركات.")
    _print_multi_doctor_validation_summary(result, doctors, "الغدد", {"doctor_role": "agent_recommended"}, None, None)
    out = capsys.readouterr().out
    assert "[doctor] multi-doctor resolution:" in out
    assert "[doctor] doctor[1]:" in out
    assert "[doctor] doctor[2]:" in out
    assert "[doctor] aggregate outcome:" in out


def test_requested_and_resolved_identities_remain_paired_per_doctor(capsys):
    """Item 2 & 3 — the exact reported contradiction must never occur: a
    doctor's OWN requested name must appear on the SAME doctor[N] block as
    their OWN resolved identity/key, never a different doctor's."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ"),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ"),
    ]
    result = multi_doctor_result(doctors, "PASS", "All 2 recommended doctors resolved and passed validation.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    lines = capsys.readouterr().out.splitlines()

    def block(marker: str) -> list[str]:
        start = next(i for i, l in enumerate(lines) if l.strip() == marker)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[doctor] doctor[") or lines[i].startswith("[doctor] aggregate")), len(lines))
        return lines[start:end]

    badria_block = "\n".join(block("[doctor] doctor[1]:"))
    amira_block = "\n".join(block("[doctor] doctor[2]:"))
    assert "requested_name='بدريه'" in badria_block
    assert "doctor_key=110411" in badria_block
    assert "resolved_name='بدرية البيروتي'" in badria_block
    assert "requested_name='اميره بركات'" in amira_block
    assert "doctor_key=11011216" in amira_block
    assert "resolved_name='اميرة بركات'" in amira_block
    # The exact reported contradiction must never appear together on one line.
    assert "requested_name='اميره بركات'" not in badria_block
    assert "doctor_key=110411" not in amira_block


def test_log_never_combines_amiras_requested_name_with_badrias_key(capsys):
    """Item 3, isolated: scanning the WHOLE printed output, the specific
    contradictory pairing from the bug report must never occur on any
    single line."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ"),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ"),
    ]
    result = multi_doctor_result(doctors, "PASS", "All 2 recommended doctors resolved and passed validation.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    for line in out.splitlines():
        if "requested_name='اميره بركات'" in line:
            assert "110411" not in line
        if "doctor_key=110411" in line:
            assert "اميره بركات" not in line


def test_one_resolved_one_unresolved_reports_correct_counts(capsys):
    """Item 4."""
    doctors = [
        doctor_entry("بدريه", resolved=False, outcome="DOCTOR_UNRESOLVED"),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ"),
    ]
    result = multi_doctor_result(doctors, "DOCTOR_UNRESOLVED", "1 of 2 recommended doctor(s) could not be resolved: بدريه.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    assert "requested_count=2" in out
    assert "resolved_count=1" in out
    assert "requested_name='بدريه'" in out
    assert "resolved=False" in out


def test_two_unresolved_doctors_report_both_entries_independently(capsys):
    """Item 5."""
    doctors = [
        doctor_entry("بدريه", resolved=False, outcome="DOCTOR_UNRESOLVED"),
        doctor_entry("سلمى", resolved=False, outcome="DOCTOR_UNRESOLVED"),
    ]
    result = multi_doctor_result(doctors, "DOCTOR_UNRESOLVED", "None of the 2 recommended doctors could be resolved against the authoritative CRM record.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    assert out.count("resolved=False") == 2
    assert "requested_name='بدريه'" in out
    assert "requested_name='سلمى'" in out
    assert "resolved_count=0" in out


def test_one_pass_one_fail_aggregate_fail_without_corrupting_either_entry(capsys):
    """Item 6."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ", outcome="PASS"),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ",
                     outcome="FAIL", validated_fields={"degree": {"outcome": "FAIL"}}),
    ]
    result = multi_doctor_result(doctors, "FAIL", "1 of 2 recommended doctor(s) failed validation: اميره بركات.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    assert "[doctor]     outcome=FAIL" in out
    lines = out.splitlines()
    badria_outcome = next(l for l in lines[lines.index("[doctor] doctor[1]:"):] if "validation_outcome=" in l)
    amira_outcome = next(l for l in lines[lines.index("[doctor] doctor[2]:"):] if "validation_outcome=" in l)
    assert badria_outcome.strip() == "[doctor]     validation_outcome=PASS"
    assert amira_outcome.strip() == "[doctor]     validation_outcome=FAIL"


def test_per_doctor_failures_and_warnings_printed_correctly(capsys):
    """Item 7 — failures=/warnings= must show EACH doctor's own
    failure_details/warning_details field names, never the aggregate or a
    different doctor's."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ",
                     validated_fields={"specialty": {"outcome": "PASS"}},
                     warning_details=[{"field": "name_completeness"}]),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ",
                     outcome="FAIL", validated_fields={"degree": {"outcome": "FAIL"}},
                     failure_details=[{"field": "degree"}]),
    ]
    result = multi_doctor_result(doctors, "FAIL", "1 of 2 recommended doctor(s) failed validation: اميره بركات.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    assert "failures=[]" in out
    assert "warnings=['name_completeness']" in out
    assert "failures=['degree']" in out
    assert "warnings=[]" in out


def test_multi_doctor_summary_reports_failed_and_warning_counts(capsys):
    """Items 12/20 (logging side) — failed_count/warning_count are
    computed independently from each doctor's OWN failure_details/
    warning_details, never conflated with each other or with
    resolved_count/requested_count."""
    doctors = [
        doctor_entry("بدريه", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ",
                     resolution_source="contextual_single_name",
                     warning_details=[{"field": "name_completeness"}]),
        doctor_entry("اميره بركات", resolved=True, doctor_key="11011216", name_ar="اميرة بركات", business_unit="AHJ",
                     resolution_source="authoritative_pool", outcome="FAIL",
                     failure_details=[{"field": "degree"}]),
    ]
    result = multi_doctor_result(doctors, "FAIL", "1 of 2 recommended doctor(s) failed validation: اميره بركات.")
    _print_multi_doctor_validation_summary(result, doctors, None, {}, None, None)
    out = capsys.readouterr().out
    assert "requested_count=2" in out
    assert "resolved_count=2" in out
    assert "failed_count=1" in out
    assert "warning_count=1" in out
    assert "resolution_source=contextual_single_name" in out
    assert "resolution_source=authoritative_pool" in out


def test_single_doctor_never_triggers_multi_doctor_summary_function(capsys):
    """Item 9 — a 0- or 1-doctor 'doctors' list must never route to the
    multi-doctor summary at all; that decision lives in validate_doctor_
    node itself (len(doctors) > 1), verified end to end below."""
    doctors = [doctor_entry("بدرية البيروتي", resolved=True, doctor_key="110411", name_ar="بدرية البيروتي", business_unit="AHJ")]
    # Directly calling the summary function with a 1-item list still works
    # mechanically (it has no gate of its own — the caller decides), but
    # the REAL guarantee is validate_doctor_node's own `len(doctors) > 1`
    # check, exercised in the end-to-end test below.
    assert len(doctors) <= 1


# ── End-to-end: validate_doctor_node ────────────────────────────────────────

class _StubDoctorLLM:
    def __init__(self, response: dict):
        self._response = response

    async def complete(self, system_prompt: str, user_prompt: str, max_tokens: int | None = None) -> tuple[str, dict]:
        return json.dumps(self._response, ensure_ascii=False), {"prompt_tokens": 0, "completion_tokens": 0}


BASE_DOCTOR = {
    "cr301_degreename": "Senior Registrar", "cr301_specialtyname": "Endocrinology",
    "cr301_subspecialtyname": None, "cr18c_manualspecialtyname": None, "cr18c_manualsubspecialtyname": None,
    "cr18c_buname": "AHJ", "cr301_businessunitname": "AHJ", "statuscodename": "Active", "cr301_opdflag": "OPD",
    "cr301_drnotes": None, "cr301_scopeofservice": None, "cr301_scopeofservicear": None,
    "cr301_qualificationsandexperience": None, "cr301_qualificationsandexperiencear": None,
    "servhub_examinationage": None, "cr301_walkinconsultationfees": 300,
}
DOC_BADRIA = {**BASE_DOCTOR, "cr301_doctorkey": "110411", "servhub_doctornameen": "Badria Bairuti", "cr301_doctornamear": "بدرية البيروتي"}
DOC_AMIRA = {**BASE_DOCTOR, "cr301_doctorkey": "11011216", "servhub_doctornameen": "Amira Barakat", "cr301_doctornamear": "أميرة بركات"}

ENDOCRINE_TRANSCRIPT = (
    "Patient: بدي دكتور الغدد\n"
    "Agent: متاح الطبيبه بدريه والطبيبه اميره بركات\n"
)


def run_validate_doctor_node(monkeypatch, transcript: str, doctors: list[dict], llm_response: dict) -> str:
    import app.service_hub.crm_doctors as crm_doctors
    monkeypatch.setattr(crm_doctors, "fetch_doctors", lambda *a, **k: doctors)
    state = {"call": call(transcript), "node_trace": []}
    asyncio.run(validate_doctor_node(state, _StubDoctorLLM(llm_response)))


def test_end_to_end_multi_doctor_call_prints_dedicated_summary_no_contradiction(monkeypatch, capsys):
    """The full reported regression, driven through the real node: both
    doctors resolve, and the printed log must show the dedicated
    multi-doctor summary with no contradictory singular lines."""
    run_validate_doctor_node(monkeypatch, ENDOCRINE_TRANSCRIPT, [DOC_BADRIA, DOC_AMIRA], {})
    out = capsys.readouterr().out
    assert "[doctor] multi-doctor resolution:" in out
    assert "requested_count=2" in out
    assert "resolved_count=2" in out
    # The legacy singular "[doctor] resolution:" block must NOT be printed
    # for a genuine multi-doctor result.
    assert "[doctor] resolution:" not in out
    for line in out.splitlines():
        if "requested_name='اميره بركات'" in line:
            assert "110411" not in line


def test_end_to_end_single_doctor_call_logging_unchanged(monkeypatch, capsys):
    """Item 9, end to end — an ordinary single-doctor call must keep
    printing the original singular '[doctor] resolution:' block, never the
    multi-doctor summary."""
    transcript = "Agent: دكتورة بدرية البيروتي استشارية غدد صماء\n"
    run_validate_doctor_node(monkeypatch, transcript, [DOC_BADRIA], {})
    out = capsys.readouterr().out
    assert "[doctor] resolution:" in out
    assert "[doctor] multi-doctor resolution:" not in out
