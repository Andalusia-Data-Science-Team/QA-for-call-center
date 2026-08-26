"""Deterministic, Arabic-first validation of KSA bank account information.

Bank validation as its own feature — see location_validation.py (same
app/service_hub/ package) for the independent (but sibling) location-
request validator. They only share the transcript-turn split
(app.services.text_helpers.split_transcript_by_speaker) and financial-
identifier normalisation (app.services.text_helpers), not any
bank/location-specific logic.

Flow (Business Unit first):
    resolve BU (BUSINESS_UNIT_KEYWORD_MAP, exact → fuzzy fallback)
        -> in bank-validation scope?
        -> bank-information intent detected?
        -> fetch/filter CRM bank records to that BU
        -> disambiguate multiple accounts (named bank, if any)
        -> detect requested field (IBAN / account number / bank name / owner)
        -> exact validation of agent-supplied information
"""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz as _rfuzz

from app.models.input import BUSINESS_UNIT_KEYWORD_MAP, CallTranscript
from app.services.text_helpers import (
    mask_financial_identifier,
    normalize_arabic_text,
    normalize_financial_identifier,
    split_transcript_by_speaker,
)

# Bank records are the only authority for correctness; LLM scoring receives
# the masked deterministic outcome, never a full financial identifier.
BANK_FAILURES = {
    "WRONG_BUSINESS_UNIT_ACCOUNT", "INCORRECT_BANK_INFORMATION",
    "BANK_INFO_REQUESTED_NOT_PROVIDED", "WRONG_FIELD_TYPE_PROVIDED",
}

# ── Bank-information intent + identifier extraction ─────────────────────────
# Arabic text is normalised before these patterns run so common spelling
# variants resolve together. Transfer alone is deliberately insufficient.
_BANK_INTENT = re.compile(r"(?:ايبان|iban|رقم\s*(?:ال)?حساب|الحساب\s*البنكي|بيانات\s*(?:ال)?(?:بنك|حساب)|حساب\s*(?:ال)?تحويل|اسم\s*(?:ال)?بنك|باسم\s*مين|اسم\s*صاحب\s*(?:ال)?حساب)", re.I)
_TRANSFER_CONTEXT = re.compile(r"(?:وين|فين)\s*احول|(?:علي\s*اي\s*حساب\s*احول)|(?:كيف|ابغى|ابي|ممكن|احتاج).{0,30}(?:احول|تحويل)|(?:احول|تحويل)\s*(?:المبلغ|لكم|علي\s*(?:اي|ايش)\s*حساب)", re.I)
_IBAN = re.compile(r"(?<![A-Z0-9])S\s*A(?:[\s\-]*[0-9٠-٩۰-۹]){8,32}(?![A-Z0-9])", re.I)
_ACCOUNT = re.compile(r"(?:رقم\s*(?:ال)?حساب|الحساب(?:\s*رقمه)?|account(?:\s*number)?)[^0-9٠-٩۰-۹]{0,20}([0-9٠-٩۰-۹][0-9٠-٩۰-۹\s\-]{5,40})", re.I)


def detect_arabic_bank_request(text: str) -> bool:
    # Normalisation collapses hamza/diacritic variants before matching Arabic chat language.
    """High-confidence terms plus contextual transfer wording; never bare تحويل."""
    text = normalize_arabic_text(text)
    return bool(_BANK_INTENT.search(text) or _TRANSFER_CONTEXT.search(text))


def extract_agent_financial_identifiers(text: str) -> list[str]:
    """Extract exact IBAN/account candidates from agent-only message text."""
    # Preserve leading zeroes and never fuzzy-match financial identifiers.
    values = [normalize_financial_identifier(m.group(0)) for m in _IBAN.finditer(text or "")]
    values += [normalize_financial_identifier(m.group(1)) for m in _ACCOUNT.finditer(text or "")]
    return list(dict.fromkeys(v for v in values if v))


def extract_agent_financial_identifiers_typed(text: str) -> list[dict[str, str]]:
    """Like extract_agent_financial_identifiers, but tags each value with its
    detected kind ('iban' | 'account_number'). Used only when the patient
    asked for one specific field: a valid IBAN must not silently satisfy an
    account-number-only request just because it's a valid identifier for the
    same business unit."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in _IBAN.finditer(text or ""):
        v = normalize_financial_identifier(m.group(0))
        if v and v not in seen:
            seen.add(v)
            out.append({"value": v, "kind": "iban"})
    for m in _ACCOUNT.finditer(text or ""):
        v = normalize_financial_identifier(m.group(1))
        if v and v not in seen:
            seen.add(v)
            out.append({"value": v, "kind": "account_number"})
    return out


# ── Per-field request detection ────────────────────────────────────────────
# Distinguishes WHICH banking field the patient actually asked for, so an
# IBAN answer doesn't silently satisfy an account-number request (or vice
# versa), and so a bare bank-name/owner question doesn't require any
# identifier at all. Falls back to the undifferentiated _BANK_INTENT/
# _TRANSFER_CONTEXT check above when none of these match — generic-request
# behaviour (Step 9's GENERAL_BANK_INFORMATION is this empty-set fallback,
# not a distinct code path).
_IBAN_FIELD_RE = re.compile(r"ايبان|iban", re.I)
_ACCOUNT_FIELD_RE = re.compile(r"رقم\s*(?:ال)?حساب|account\s*number", re.I)
_BANK_NAME_FIELD_RE = re.compile(r"اسم\s*(?:ال)?بنك|أي\s*بنك|انهي\s*بنك|ايه\s*البنك|which\s*bank|bank\s*name", re.I)
_ACCOUNT_OWNER_FIELD_RE = re.compile(r"باسم\s*مين|اسم\s*صاحب\s*(?:ال)?حساب|account\s*owner|account\s*holder", re.I)


def _detect_requested_fields(patient_text: str) -> set[str]:
    """Return the subset of {'iban','account_number','bank_name','account_owner'}
    the patient text explicitly asks for. Empty set means a generic/
    undifferentiated request (e.g. "بيانات التحويل") — the general check
    applies to all of it."""
    norm = normalize_arabic_text(patient_text)
    fields: set[str] = set()
    if _IBAN_FIELD_RE.search(norm):
        fields.add("iban")
    if _ACCOUNT_FIELD_RE.search(norm):
        fields.add("account_number")
    if _BANK_NAME_FIELD_RE.search(norm):
        fields.add("bank_name")
    if _ACCOUNT_OWNER_FIELD_RE.search(norm):
        fields.add("account_owner")
    return fields


# ── Bank-name normalisation ─────────────────────────────────────────────────
# Aliases only, for STRING COMPARISON — the DB (cr301_bankname) remains the
# source of truth for which bank is actually approved; this table never
# decides correctness on its own, it only lets "الاهلي" and "البنك الاهلي"
# compare equal to whatever spelling the CRM/agent happens to use.
_BANK_NAME_CANON: dict[str, str] = {
    "الاهلي": "alahli", "البنك الاهلي": "alahli", "بنك الاهلي": "alahli",
    "الرياض": "riyad", "بنك الرياض": "riyad",
    "السعودي الفرنسي": "fransi", "البنك السعودي الفرنسي": "fransi", "الفرنسي": "fransi",
    "الراجحي": "rajhi", "بنك الراجحي": "rajhi", "مصرف الراجحي": "rajhi",
    "الخليج الدولي": "gib", "بنك الخليج الدولي": "gib",
    "ساب": "sabb", "بنك ساب": "sabb",
    "الانماء": "alinma", "بنك الانماء": "alinma", "مصرف الانماء": "alinma",
}


def _canon_bank_name(text: str | None) -> str | None:
    """Longest-alias-wins canonicalisation (same pattern as
    offer_search.py's resolve_specialty, same package) — falls back to the
    plain normalised string so an unlisted bank name still compares via
    substring equality against the DB value rather than being ignored."""
    norm = normalize_arabic_text(text or "")
    if not norm:
        return None
    best, best_len = None, 0
    for alias, canon in _BANK_NAME_CANON.items():
        alias_norm = normalize_arabic_text(alias)
        if alias_norm and alias_norm in norm and len(alias_norm) > best_len:
            best, best_len = canon, len(alias_norm)
    return best or norm


def _bank_names_mentioned(text: str) -> set[str]:
    """All canonical bank names mentioned anywhere in *text*."""
    norm = normalize_arabic_text(text)
    found: set[str] = set()
    for alias, canon in _BANK_NAME_CANON.items():
        if normalize_arabic_text(alias) in norm:
            found.add(canon)
    return found


def _active(record: dict[str, Any]) -> bool:
    # Numeric state/status values are intentionally not interpreted without tenant evidence.
    names = [normalize_arabic_text(str(record.get(k) or "")) for k in ("statecodename", "statuscodename")]
    return not any(v in {"inactive", "disabled", "draft", "غير نشط", "موقوف"} for v in names if v)


# ── Business Unit resolution ────────────────────────────────────────────────
# BUSINESS_UNIT_KEYWORD_MAP (app.models.input) is the project's single source
# of truth for BU aliases — reused here, not duplicated. This module only
# adds what bank validation specifically needs on top of it: a fuzzy fallback
# for spelling variants, and a canonical→CRM BU translation layer (see
# _APP_BU_TO_CRM_BU below).

# Multi-word aliases only, for the fuzzy fallback — deliberately excludes
# short single-word aliases ("الطفل", "المرأة", "سلطان", ...): fuzzy-matching
# a short, common word against arbitrary chat text risks resolving unrelated
# generic language to a BU. Short aliases stay reachable via exact match only.
_FUZZY_BU_ALIASES: dict[str, str] = {
    alias: code for alias, code in BUSINESS_UNIT_KEYWORD_MAP.items()
    if len(normalize_arabic_text(alias).split()) >= 2
}
_BU_FUZZY_THRESHOLD = 85  # stricter than location's 80 — a BU misresolution
                          # decides which CRM records get validated against


def resolve_business_unit(text: str) -> str | None:
    """Resolve a canonical application BU code (e.g. "AKW", "LIVE") from free
    Arabic text via BUSINESS_UNIT_KEYWORD_MAP.

    1. Exact normalised-substring match — longest alias wins (so "صحة
       الطفل" outscores a coincidental bare "الطفل" elsewhere in the text),
       same pattern as offer_search.py's resolve_specialty / this module's
       own _canon_bank_name.
    2. Fuzzy fallback (rapidfuzz) — multi-word aliases only, high threshold —
       for spelling variants exact character-normalisation doesn't cover
       (e.g. a tatweel-inserted "الطفـل"). Never applied to short aliases or
       literal "BU-XXX" codes, so it can't resolve an unrelated generic word
       to a BU.
    """
    norm_text = normalize_arabic_text(text)
    if not norm_text:
        return None

    best_code, best_len = None, 0
    for alias, code in BUSINESS_UNIT_KEYWORD_MAP.items():
        alias_norm = normalize_arabic_text(alias)
        if alias_norm and alias_norm in norm_text and len(alias_norm) > best_len:
            best_code, best_len = code, len(alias_norm)
    if best_code:
        return best_code

    best_score, best_fuzzy_code = 0.0, None
    for alias, code in _FUZZY_BU_ALIASES.items():
        alias_norm = normalize_arabic_text(alias)
        score = _rfuzz.partial_ratio(alias_norm, norm_text)
        if score > best_score:
            best_score, best_fuzzy_code = score, code
    if best_score >= _BU_FUZZY_THRESHOLD:
        return best_fuzzy_code
    return None


# Bank-validation scope: AFW / AHJ (internally "LIVE" — see below) / AKW / ALW
# only. LCH/MKR/SNB remain valid BUs elsewhere in the project (booking,
# offers, ...) — this restriction is local to bank validation, not a change
# to BUSINESS_UNIT_KEYWORD_MAP or CallTranscript.business_unit.
_SUPPORTED_BANK_BUS = {"AFW", "LIVE", "AKW", "ALW"}

# Canonical APP BU code → CRM BU code, only where they differ. Verified
# against LIVE cr301_bankaccounts data (this session, via a real Dynamics
# 365 query): real CRM rows use cr301_bu / servhub_bulookupname /
# cr18c_bunname = "AHJ", never "LIVE" — "LIVE" is this QA system's own
# application-level label for that branch (BUSINESS_UNIT_KEYWORD_MAP maps
# "جدة" / "حي الجامعة" / "BU-AHJ" → "LIVE"). AFW/AKW/ALW already match their
# CRM values identically, so no entry is needed for them — this dict exists
# so AHJ⇄LIVE is translated in exactly ONE place instead of scattered
# special-casing throughout the validator.
_APP_BU_TO_CRM_BU: dict[str, str] = {"LIVE": "AHJ"}

# owningbusinessunitname deliberately excluded from CRM-side BU filtering:
# verified (live CRM query, this session) to hold a constant tenant-level
# value ("ERS") on every single row — it provides zero discriminating signal
# for per-branch filtering, unlike the other three.
_CRM_BU_FIELDS = ("cr301_bu", "servhub_bulookupname", "cr18c_bunname")


def is_supported_bank_bu(bu: str | None) -> bool:
    return bu in _SUPPORTED_BANK_BUS


def _bank_rows_for_bu(bu: str, banks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return active CRM accounts whose BU field matches the resolved BU,
    after canonical→CRM normalisation. Any ONE of the three CRM BU fields
    matching is enough — real CRM data has been observed (this session) to
    disagree between cr301_bu and servhub_bulookupname/cr18c_bunname on the
    SAME row (one live Gulf International Bank record: cr301_bu="ALW" but
    servhub_bulookupname/cr18c_bunname="AKW"). Treating any field match as
    sufficient is the safer failure mode here — it risks over-including a
    genuinely ambiguous record rather than silently excluding it from its
    true BU.
    """
    crm_bu = _APP_BU_TO_CRM_BU.get(bu, bu)
    return [
        row for row in banks
        if _active(row) and any(str(row.get(f) or "").strip().upper() == crm_bu for f in _CRM_BU_FIELDS)
    ]


def _owner_names_match(agent_text: str, crm_owner: str | None) -> bool:
    """Approximate match for account-owner free text (company/branch names,
    e.g. "مستشفى أندلسية" vs CRM's "Andalusia Arab medical clinic") — unlike
    financial identifiers, owner names are naturally paraphrased, so exact
    equality is too strict. Either normalised string containing the other is
    treated as a match; this is deliberately looser than bank-name
    canonicalisation because there's no fixed alias table for owner names."""
    crm_norm = normalize_arabic_text(crm_owner or "")
    agent_norm = normalize_arabic_text(agent_text)
    if not crm_norm or not agent_norm:
        return False
    return crm_norm in agent_norm or agent_norm in crm_norm


def _ids(row: dict[str, Any]) -> set[str]:
    """Read both supported CRM identifiers without exposing them externally."""
    return {v for v in (normalize_financial_identifier(row.get("cr301_ibannumber")), normalize_financial_identifier(row.get("cr301_accountnumber"))) if v}


def _ids_of_kind(row: dict[str, Any], kind: str) -> str:
    field = "cr301_ibannumber" if kind == "iban" else "cr301_accountnumber"
    return normalize_financial_identifier(row.get(field))


def _result(outcome: str, reason: str, request: bool, ids: list[str], bu: str | None = None, applicable: bool = False) -> dict[str, Any]:
    """Build an API-safe result with masked identifiers only."""
    return {"outcome": outcome, "applicable": applicable, "request_detected": request,
            "requested_business_unit": bu,
            "provided_identifiers": [mask_financial_identifier(v) for v in ids], "reason": reason,
            "is_violation": outcome in BANK_FAILURES}


def detect_bank_signals(call: CallTranscript) -> tuple[bool, list[str], str, str, str | None]:
    """Parse the transcript once into (patient_requested, agent_supplied_ids,
    patient_text, agent_text, resolved_business_unit).

    Shared by bank_validation_needed() and validate_ksa_bank_information() so
    a call's transcript is split into turns and regex-scanned once per
    pipeline run, instead of once for the cheap gate and again for the full
    validation. BU is resolved from the patient's own text first, falling
    back to the full transcript — a patient's current request overrides
    stale/earlier conversation context, the same priority already used by
    this project's location resolution.
    """
    patient, agent = split_transcript_by_speaker(call.transcript)
    bu = resolve_business_unit(patient) or resolve_business_unit(call.transcript)
    return detect_arabic_bank_request(patient), extract_agent_financial_identifiers(agent), patient, agent, bu


def _check_bank_name_only(agent_text: str, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bank-name-only request path (e.g. "اسم البنك؟") — independent of any
    identifier, since the answer here is a bank name, not a number."""
    approved = {_canon_bank_name(row.get("cr301_bankname")) for row in expected_rows}
    approved.discard(None)
    mentioned = _bank_names_mentioned(agent_text)
    if not mentioned:
        return {"outcome": "BANK_INFO_REQUESTED_NOT_PROVIDED",
                "reason": "The patient asked for the bank name, but the agent did not name a bank."}
    if mentioned & approved:
        return {"outcome": "PASS", "reason": "The agent named an approved bank for the requested business unit."}
    return {"outcome": "INCORRECT_BANK_INFORMATION",
            "reason": "The agent named a bank that is not approved for the requested business unit."}


def _check_account_owner_only(agent_text: str, expected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Account-owner-only request ("الحساب باسم مين؟") — free-text name
    match, not an identifier and not a fixed alias table (owner names are
    naturally paraphrased, unlike bank names from a small closed set)."""
    if not agent_text.strip():
        return {"outcome": "BANK_INFO_REQUESTED_NOT_PROVIDED",
                "reason": "The patient asked who the account is registered to, but the agent gave no answer."}
    owners = [r.get("cr301_accountowner") for r in expected_rows if r.get("cr301_accountowner")]
    if any(_owner_names_match(agent_text, o) for o in owners):
        return {"outcome": "PASS", "reason": "The agent named the approved account owner for this business unit."}
    return {"outcome": "INCORRECT_BANK_INFORMATION",
            "reason": "The agent's stated account owner does not match any approved CRM record for this business unit."}


def validate_ksa_bank_information(
    call: CallTranscript,
    bank_records: list[dict[str, Any]],
    signals: tuple[bool, list[str], str, str, str | None] | None = None,
) -> dict[str, Any]:
    """Validate agent-supplied bank information against CRM-authoritative
    records for the business unit resolved from the conversation.

    Flow (Business Unit first, per this module's design):
      resolve BU -> in scope? -> bank intent detected? -> filter CRM by BU ->
      narrow by named bank -> detect requested field -> exact validation.
    """
    requested, supplied, patient, agent_text, bu = signals if signals is not None else detect_bank_signals(call)

    # Step 1 — Business Unit scope gate. Out-of-scope/unresolved BU takes the
    # request out of scope entirely, independent of bank intent (Step 7: a
    # booking request that happens to mention "صحة الطفل" is not a bank
    # request just because AKW resolves).
    if not bu or not is_supported_bank_bu(bu):
        reason = (
            "No business unit within bank-validation scope (AFW/AHJ/AKW/ALW) could be resolved."
            if not bu else
            f"Resolved business unit '{bu}' is outside bank-validation scope (AFW/AHJ/AKW/ALW)."
        )
        return _result("NOT_APPLICABLE", reason, False, [])

    # Step 2 — Bank-information intent gate.
    if not requested and not supplied:
        return _result("NOT_APPLICABLE", "No bank request or agent account identifier was detected.", False, [], bu)

    # Step 3 — CRM lookup, filtered to the resolved BU. Missing candidate
    # rows is a reference-data condition, not evidence of agent misconduct.
    expected_rows = _bank_rows_for_bu(bu, bank_records)
    if not expected_rows:
        return _result(
            "NO_APPROVED_BANK_ACCOUNT_FOUND",
            f"The resolved business unit '{bu}' has no active approved identifier in CRM.",
            requested, supplied, bu, True,
        )

    # Step 4 — disambiguate multiple accounts. A BU can have multiple
    # accounts at different banks. If the patient named a specific bank,
    # narrow to just that bank's row(s) before anything else —
    # "ابغى ايبان بنك الرياض لمستشفى أندلسية" resolves via BU + bank name,
    # not by picking one arbitrarily.
    requested_banks = _bank_names_mentioned(patient)
    if requested_banks:
        narrowed = [r for r in expected_rows if _canon_bank_name(r.get("cr301_bankname")) in requested_banks]
        if narrowed:
            expected_rows = narrowed

    requested_fields = _detect_requested_fields(patient)

    # Bank-name-only / account-owner-only requests — no identifier expected.
    if requested_fields == {"bank_name"}:
        outcome = _check_bank_name_only(agent_text, expected_rows)
        return _result(outcome["outcome"], outcome["reason"], requested, supplied, bu, True)
    if requested_fields == {"account_owner"}:
        outcome = _check_account_owner_only(agent_text, expected_rows)
        return _result(outcome["outcome"], outcome["reason"], requested, supplied, bu, True)

    if not supplied:
        # `requested` is guaranteed True here: the top-of-function guard above
        # already returns NOT_APPLICABLE for the case where neither is true.
        #
        # A specific field (IBAN/account number) was asked for, no
        # identifier was given, but the agent answered with a DIFFERENT
        # recognisable field (a bank name) — that's a wrong-field-type
        # answer, not a non-answer, even if the bank named happens to be
        # correct for this BU (Step 18: naming the right bank doesn't
        # answer "what's the IBAN?").
        single_field = requested_fields & {"iban", "account_number"}
        if len(single_field) == 1 and _bank_names_mentioned(agent_text):
            kind = next(iter(single_field))
            return _result(
                "WRONG_FIELD_TYPE_PROVIDED",
                f"The patient asked for the {kind.replace('_', ' ')}, but the agent only named a bank — "
                f"that does not answer what was asked, even if the bank itself is correct for this business unit.",
                requested, [], bu, True,
            )
        distinct_banks = {_canon_bank_name(r.get("cr301_bankname")) for r in expected_rows} - {None}
        if not requested_banks and len(distinct_banks) > 1:
            return _result(
                "AMBIGUOUS_MULTIPLE_ACCOUNTS",
                f"This business unit has {len(distinct_banks)} approved bank accounts and the patient did not "
                f"specify which one — the agent should have asked, not guessed (and neither should this check).",
                requested, [], bu, True,
            )
        return _result("BANK_INFO_REQUESTED_NOT_PROVIDED", "The patient requested bank information but the agent provided no account identifier.", requested, [], bu, True)

    # Step 5 — exact validation. Every identifier sent by the agent must be
    # authorised for THIS BU, regardless of which specific field was asked —
    # volunteering an unauthorised identifier is a violation even if it
    # wasn't the one asked for.
    expected = set().union(*(_ids(r) for r in expected_rows))
    all_ids = set().union(*(_ids(r) for r in bank_records if _active(r)))
    if not expected:
        return _result("NO_APPROVED_BANK_ACCOUNT_FOUND", f"The resolved business unit '{bu}' has no active approved identifier in CRM.", requested, supplied, bu, True)
    # Check all approved accounts outside the requested BU before treating a
    # value as unknown. This distinguishes a valid-but-wrong-BU account from
    # fabricated or mistyped bank information.
    wrong, unknown = [v for v in supplied if v in all_ids - expected], [v for v in supplied if v not in all_ids]
    if wrong:
        return _result("WRONG_BUSINESS_UNIT_ACCOUNT", "An identifier belongs to another approved business unit.", requested, supplied, bu, True)
    if unknown:
        return _result("INCORRECT_BANK_INFORMATION", "An identifier is not an approved active CRM account.", requested, supplied, bu, True)

    # Every supplied identifier is approved for this BU. Now check whether the
    # SPECIFICALLY requested field type was actually among them — an agent
    # who only gave the IBAN when asked for the account number gave a
    # valid-but-wrong-field answer, which must not read as a silent PASS.
    single_field = requested_fields & {"iban", "account_number"}
    if len(single_field) == 1:
        kind = next(iter(single_field))
        typed = extract_agent_financial_identifiers_typed(agent_text)
        supplied_of_kind = {d["value"] for d in typed if d["kind"] == kind}
        expected_of_kind = {_ids_of_kind(r, kind) for r in expected_rows} - {""}
        if not (supplied_of_kind & expected_of_kind):
            other = "account number" if kind == "iban" else "IBAN"
            return _result(
                "WRONG_FIELD_TYPE_PROVIDED",
                f"The patient asked for the {kind.replace('_', ' ')}, but the agent only provided a {other} — "
                f"the value given is valid for this business unit, just not the field that was requested.",
                requested, supplied, bu, True,
            )

    return _result("PASS", "Every supplied identifier exactly matches an approved account for the requested business unit.", requested, supplied, bu, True)


def bank_validation_needed(call: CallTranscript, signals: tuple[bool, list[str], str, str, str | None] | None = None) -> bool:
    """Return whether a call needs bank reference data before graph evaluation.

    Requires BOTH a supported business unit AND bank-information context —
    a booking request that happens to mention a supported BU is not, by
    itself, a bank request (Step 7). This also means calls about
    out-of-scope BUs (LCH/MKR/SNB) or with no resolvable BU never trigger a
    CRM fetch at all.
    """
    requested, supplied, _patient, _agent, bu = signals if signals is not None else detect_bank_signals(call)
    return is_supported_bank_bu(bu) and (requested or bool(supplied))
