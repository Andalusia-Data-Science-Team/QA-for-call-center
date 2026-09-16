"""Deterministic, Arabic-first validation of doctor information given by the
Agent to the Patient.

Doctor validation as its own feature — see bank_validation.py /
location_validation.py (same app/service_hub/ package) for the independent
sibling validators. Shares only the transcript-turn split
(app.services.text_helpers.split_transcript_by_speaker) and Arabic text
normalisation, not any bank/location/doctor-specific logic.

Flow (mirrors bank_validation.py's "resolve identity first" philosophy):
    detect a named-doctor mention anywhere in the call
        -> doctor validation applicable at all?
        -> fetch/filter CRM doctor records (Active + supported BU)
        -> deduplicate by doctor key
        -> resolve the specific doctor (exact name -> distinctive partial -> fuzzy)
        -> extract ONLY the factual claims the Agent actually made
        -> validate each claimed field against the resolved CRM record

Doctor searchability is Active + supported-BU ONLY — cr301_opdflag is
deliberately NOT part of this gate (see _is_active's docstring): a doctor
who isn't flagged OPD (e.g. a home-care/other non-OPD service context) can
still be a genuine, resolvable doctor. cr301_opdflag continues to be
fetched and carried as informational metadata on every doctor record.

See doctor_scope_validation for the separate, LLM-based semantic
"is this doctor's CRM scope of service compatible with what the patient
described" check — that one is intentionally NOT part of this deterministic
module (see its own docstring for why).

KNOWN LIMITATION (documented per design, not silently papered over): the
CRM doctor dataset also contains non-human operational rows (e.g. "MRI .",
"X Ray Male", "Procedure Room", "TEST Doctor") that pass the same
Active + supported-BU filter real doctors do — there is no reliable
CRM field available to exclude them (degree/specialty/BU look identical to
real physician rows). This is handled defensively, not by inventing a
filter: name resolution stays conservative (exact/high-confidence match
only, never a bare generic word), so such rows only ever get selected when
the transcript explicitly names them — which essentially never happens in
a real conversation.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any

from rapidfuzz import fuzz as _rfuzz

from app.models.input import CallTranscript
from app.services.text_helpers import normalize_arabic_text, split_transcript_by_speaker, split_transcript_turns
# Reuse the project's single canonical APP-BU→CRM-BU translation table
# rather than defining a second one here — see canonical_doctor_bu()'s
# docstring below and app.service_hub.bank_validation's own comment beside
# _APP_BU_TO_CRM_BU for the full rationale (verified against real Dynamics
# 365 data: CRM rows use "AHJ", never the app-level "LIVE" label).
from app.service_hub.bank_validation import _APP_BU_TO_CRM_BU, resolve_business_unit
# Reuse the project's single canonical Arabic/English specialty-alias
# table rather than a second, duplicate copy of it — see
# _resolve_specialty_category's docstring below.
from app.service_hub.offer_search import _AR_ALIAS
# The shared, formal EN<->AR specialty/subspecialty NAME taxonomy —
# distinct from (and merged with, never duplicating) offer_search._AR_ALIAS
# above; see app.service_hub.specialty_taxonomy's own module docstring for
# why the two tables are kept separate.
from app.service_hub.specialty_taxonomy import (
    SPECIALTY_EN_TO_AR,
    SPECIALTY_TAXONOMY_ALIASES,
    canonicalize_specialty_name,
    is_pediatric_specialty,
)

logger = logging.getLogger(__name__)


# ── Terminal debug output ────────────────────────────────────────────────────
# A concise, human-readable "[doctor] ..." terminal view — sibling to the
# existing "[offers] ..." / "[crm_location] ..." / "[location] ..." print-
# style logs. Purely additive local-dev/testing observability: never
# replaces the structured logger.info() calls beside it.
#
# DEBUG-gated, deliberately: this is where every raw candidate/rejection/
# CRM-record-count diagnostic gets printed while resolving a doctor — at
# INFO level this presented obvious non-name garbage ("بعدين", "بتكتب", a
# specialty-contaminated fragment like "وصال بعياده") as if it were
# legitimate extracted doctor evidence. The single clean summary a normal
# INFO-level run shows (extraction/routing/resolution/outcome) is emitted
# once, separately, by app.agent.nodes.validate_doctor_node /
# skip_doctor_validation — never by this helper.
def _doc_print(text: str = "") -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    print(f"[doctor] {text}" if text else "", flush=True)


def _doc_print_section(title: str, **fields: Any) -> None:
    _doc_print()
    _doc_print(f"{title}:")
    for key, value in fields.items():
        if value is not None:
            _doc_print(f"  {key}: {value}")


DOCTOR_FAILURES = {"FAIL"}

# ── Authoritative filtering (Part B) ────────────────────────────────────────
# cr18c_buname is the primary BU authority for RECORD ELIGIBILITY/
# searchability (is_supported_doctor_bu / authoritative_doctor_pool below) —
# NOT cr301_businessunitname, which real CRM data has been observed to
# disagree with on the same row (e.g. cr301_businessunitname="AHJ" while
# cr18c_buname="AKW" for the same doctor). Only these 7 BUs are in scope for
# THAT policy. This is a separate question from whether a chat-stated BU
# CLAIM about an already-eligible doctor matches — see
# _doctor_business_units/_match_doctor_business_unit below, which
# deliberately check BOTH fields for that, precisely BECAUSE the two fields
# can disagree: a claim matching either one is genuine evidence, and a
# disagreement between the two CRM fields alone must never fail it.
_SUPPORTED_DOCTOR_BUS = {"AKW", "AHJ", "HJH", "ALW", "ADC", "LCH", "AFW"}


def is_supported_doctor_bu(bu: str | None) -> bool:
    return bool(bu) and str(bu).strip().upper() in _SUPPORTED_DOCTOR_BUS


def canonical_doctor_bu(bu: str | None) -> str | None:
    """Translate an application-level business-unit code (e.g. "LIVE") to
    the CRM business-unit code doctor records actually carry (e.g. "AHJ"),
    BEFORE any BU-scoped filtering/comparison happens. Reuses
    app.service_hub.bank_validation._APP_BU_TO_CRM_BU — the project's one
    canonical APP-BU→CRM-BU translation table (verified against real
    Dynamics 365 data) — rather than defining a second, inconsistent
    mapping here. A BU whose app-level and CRM codes already match 1:1
    (AKW/ALW/AFW/...) simply passes through unchanged.

    Doctor Validation's own CRM records (cr18c_buname/cr301_businessunitname)
    always use the CRM code, so every comparison against them — building
    bu_scoped_pool, the BU tie-break, is_supported_doctor_bu — must run on
    this canonical value, never on the raw conversation-level business_unit."""
    if not bu:
        return None
    bu_upper = str(bu).strip().upper()
    return _APP_BU_TO_CRM_BU.get(bu_upper, bu_upper)


def _normalize_bu(value: Any) -> str | None:
    """Trim + uppercase a raw business-unit value for COMPARISON purposes
    only ("mkr", "MKR", " MKR " must all be recognised as the same code) —
    never used to alter what gets stored/returned as a reference value.
    Returns None for empty/missing input so callers can treat "no BU on
    this field" and "BU present but didn't match" distinctly."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _doctor_business_units(record: dict[str, Any]) -> set[str]:
    """Every normalized BU code carried on *record*, drawn from EITHER of
    the two CRM business-unit fields a doctor record exposes
    (cr301_businessunitname, cr18c_buname). The two fields are NOT required
    to agree with each other (see _SUPPORTED_DOCTOR_BUS' comment above for
    the real-data disagreement this generalises from) — this set is the
    single place that OR condition lives, so no call site duplicates it.

    NEVER used to decide record eligibility/searchability — that remains
    cr18c_buname alone (is_supported_doctor_bu / authoritative_doctor_pool),
    a deliberately separate, narrower policy this set does not replace."""
    return {
        normalized
        for normalized in (
            _normalize_bu(record.get("cr301_businessunitname")),
            _normalize_bu(record.get("cr18c_buname")),
        )
        if normalized
    }


def _match_doctor_business_unit(claimed_bu: Any, record: dict[str, Any]) -> str | None:
    """Does *claimed_bu* (already canonicalised to the CRM code space via
    canonical_doctor_bu) match EITHER of *record*'s two CRM BU fields?
    Comparison is exact, case-insensitive, and whitespace-trimmed only —
    deliberately never substring/fuzzy, so "MK" must never match "MKR".

    Returns the NAME of the CRM field that matched
    ("cr301_businessunitname" or "cr18c_buname" — cr301 is checked first,
    so a doctor whose two fields disagree but both happen to equal the
    claim reports a single, deterministic field), or None when there is no
    match at all (including when claimed_bu is empty/None). A disagreement
    between the doctor's own two CRM fields is, by itself, never a reason
    to fail a claim that matches ONE of them."""
    claimed_norm = _normalize_bu(claimed_bu)
    if not claimed_norm:
        return None
    for field_name in ("cr301_businessunitname", "cr18c_buname"):
        if _normalize_bu(record.get(field_name)) == claimed_norm:
            return field_name
    return None


def _is_active(record: dict[str, Any]) -> bool:
    """Active-status check ONLY. cr301_opdflag is deliberately NOT part of
    this gate — a doctor who isn't flagged OPD (e.g. a home-care or other
    non-OPD service context, like "اسامة عبد السلام" in the home-visit
    regression this generalises from) can still be a genuine, resolvable
    doctor. cr301_opdflag continues to be fetched and carried on every
    doctor record as informational metadata (surfaced in logs/diagnostics
    and in the resolved doctor evidence) — it just no longer determines
    whether a doctor is SEARCHABLE. Named _is_active (not the historical
    _active_opd) precisely because OPD is no longer part of what it
    checks."""
    status = normalize_arabic_text(str(record.get("statuscodename") or ""))
    return status == "active"


def authoritative_doctor_pool(doctors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Active + supported-BU records only — the mandatory searchability
    conditions. A record outside the supported BU list or inactive must
    never be treated as a valid doctor for this validation. cr301_opdflag
    is NOT a condition here (see _is_active's docstring) — the pool spans
    both OPD and non-OPD doctors."""
    return [r for r in doctors if _is_active(r) and is_supported_doctor_bu(r.get("cr18c_buname"))]


def _merge_duplicate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge rows that share a doctor key: take the first non-null/non-empty
    value for each field across the group (first row wins ties), so
    scattered nulls in one duplicate don't shadow real data in another.
    Logs (does not silently hide) when two rows materially disagree on a
    field that matters for validation."""
    merged: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            if merged.get(key) in (None, "") and value not in (None, ""):
                merged[key] = value
    conflict_fields = ("cr301_degreename", "cr301_specialtyname", "cr18c_buname")
    for field in conflict_fields:
        values = {str(r.get(field) or "").strip() for r in rows if r.get(field)}
        if len(values) > 1:
            logger.warning(
                "doctor duplicate rows conflict | doctor_key=%s field=%s values=%s — using first non-empty",
                rows[0].get("cr301_doctorkey"), field, sorted(values),
            )
    return merged


def dedupe_doctors(doctors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by cr301_doctorkey — the dataset contains duplicate/near-
    duplicate rows for the same physical doctor. Rows with no doctor key at
    all are kept as-is (can't be grouped)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    unkeyed: list[dict[str, Any]] = []
    order: list[str] = []
    for row in doctors:
        key = row.get("cr301_doctorkey")
        if not key:
            unkeyed.append(row)
            continue
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(row)
    return [_merge_duplicate_rows(groups[k]) for k in order] + unkeyed


# ── Doctor-mention detection (applicability gate) ───────────────────────────
# The generic specialty words below must NOT, by themselves, register as a
# "named doctor" mention — "محتاج دكتور عظام" is a specialty request, not a
# recommendation/claim about a specific physician. This list is deliberately
# extensible (mirrors bank_validation.py's _BANK_NAME_CANON — a non-
# exhaustive alias table, not a closed enum) rather than exhaustive.
_GENERIC_SPECIALTY_WORDS = {
    # Specialties/subspecialties — "حشو"/"حشوات" (dental fillings) added so
    # a claim like "الحشوات" is recognised as a SPECIALTY/SERVICE phrase
    # (see _first_specialty_phrase_in_text -> doctor_context_specialty),
    # even though it has no entry of its own in the specialty-alias tables
    # (_AR_ALIAS/_SPECIALTY_ALIAS_SUPPLEMENT/specialty_taxonomy) — it is
    # deliberately left OUT of those, so such a claim reaches the CRM
    # scope-of-service fallback (see _match_specialty_against_scope)
    # instead of coarsely resolving to "Dental Services".
    "عظام", "اسنان", "أسنان", "حشو", "حشوات", "اطفال", "أطفال", "نساء", "ولاده", "ولادة",
    "جلديه", "جلدية", "قلب", "مخ", "اعصاب", "أعصاب", "عيون", "انف", "أنف",
    "اذن", "أذن", "حنجره", "حنجرة", "باطنه", "باطنة", "مسالك", "بوليه",
    "بولية", "نفسيه", "نفسية", "تغذيه", "تغذية", "جراحه", "جراحة", "عام", "عامه",
    "عامة", "تجميل", "روماتيزم", "غدد", "صماء", "سكري", "اورام", "أورام",
    "طوارئ", "سمعيات", "تخاطب", "نطق", "امراض", "أمراض",
    "تناسليه", "تناسلية", "كلي", "كلى", "مفاصل", "عمود", "فقري", "فقرى",
    "صدر", "تنفس", "ربو", "سمنه", "سمنة", "تكميم", "قسطره", "قسطرة",
    # Imaging/procedures/labs — mirrors the same core vocabulary already
    # used for offer-search relevance in app.service_hub.crm_offers
    # (_CORE_MEDICAL) and app.service_hub.offer_search (_AR_ALIAS) — kept
    # as a separate copy here rather than importing those private, offer-
    # context structures, but drawn from the same source vocabulary so this
    # blocklist doesn't rely on inventing (and re-discovering) its own list
    # of imaging/procedure terms one regression at a time. Real regression:
    # "دكتور الرنين" ("the MRI doctor") wasn't recognised as a role/service
    # reference because "رنين" alone was missing from the specialty set.
    "اشعه", "أشعة", "تخدير", "رنين", "مقطعي", "مقطعيه", "مقطعية", "سونار",
    "ايكو", "إيكو", "منظار", "تنظير", "تحاليل", "تحليل", "مختبر",
    "عمليات", "عمليه", "علاج", "فحص", "كشف", "متابعه", "متابعة",
    "استشاره", "استشارة", "تطعيم", "تلقيح", "غسيل", "تصوير",
    # Department/team names — "التمريض" (nursing) is the team, never a
    # doctor's name; real regression: "التمريض بيكون ..." extracted the
    # whole fragment as a candidate.
    "التمريض", "تمريض",
    # Specialty-MODIFIER adjectives that combine with a preceding specialty
    # noun to name a full field ("العلاج الطبيعي" = physiotherapy, "الطب
    # النفسي" = psychiatry) — same open, non-exhaustive class as the nouns
    # above, not a standalone name-blocking word list.
    "طبيعي", "طبيعيه", "طبيعية", "نفسي",

    # English generic-role/specialty words — "orthopedic doctor", "female
    # doctor", "consultant", "specialist", "physiotherapist" are all
    # specialty/role references, never a person's name, in either
    # language. Non-exhaustive, same philosophy as the Arabic set above.
    "orthopedic", "female", "male", "consultant", "specialist",
    "physiotherapist", "pediatrician", "cardiologist", "dermatologist",
    "surgeon", "physician",
}

# Extended with every taxonomy specialty/subspecialty name's own FIRST
# word (EN and AR, from the shared app.service_hub.specialty_taxonomy
# table) — lets a literal specialty/subspecialty NAME an Agent states
# verbatim (e.g. "Pediatric Cardiology", "NICU", "Pedodontic", "طب أسنان
# الأطفال") be recognised as a specialty/service phrase the same way the
# hardcoded colloquial words above already are (see _rejected_candidate_
# reason / _first_specialty_phrase_in_text — this is what feeds
# doctor_context_specialty / _scoped_specialty_text in the first place),
# even though it is not itself one of those hand-picked conversational
# words. Computed once from the shared taxonomy, never a second,
# hand-maintained list of specialty names to keep in sync.
_GENERIC_SPECIALTY_WORDS |= {
    normalize_arabic_text(name).split()[0]
    for name in SPECIALTY_TAXONOMY_ALIASES
    if normalize_arabic_text(name)
}

# Function/generic words that must NOT be mistaken for the start of a
# proper name — real regression: "الدكتور طلبها لحضرتك" ("the doctor
# requested it"), "دكتور أو دكتورة" (asking whether a male/female doctor
# exists), "الطبيب المعالج"/"الطبيب المختص" (generic references to
# "the treating/specialist physician", not a name), "لازم الطبيب يكتب
# الطلب" ("the doctor must write the request") were all previously
# extracted as if the word right after the title were a doctor's name.
# Extensible, non-exhaustive — same philosophy as _GENERIC_SPECIALTY_WORDS.
_NON_NAME_FIRST_WORDS = {
    "طلبها", "طلب", "المعالج", "المختص", "او", "أو", "من", "يكتب", "كتب",
    "هيشوف", "هيكشف", "هيكشفلك", "حضرتك", "اللي", "الي", "التي", "بيقول", "قال",
    "قالت", "قالوا", "بيطلب", "يطلب", "طالب", "المشرف", "المسؤول",
    "المسئول", "المناوب", "ايه", "أيه", "مين", "كام", "فين", "وين",
    "المتابع", "المناسب", "مناسب", "مناسبه", "يناسب", "تناسب",
    "متاح", "متاحه", "متاحة", "حولني", "حولتني", "موجود",
    "هو", "هي",
    # Prepositions/particles — real regression: "طبيب في المنزل ولا
    # المستشفى؟" extracted "في المنزل ولا المستشفي" as a "name" purely
    # because "في" (a preposition) was never on any blocklist. A person's
    # name can never grammatically start with one of these.
    "في", "من", "على", "علي", "الى", "إلى", "عن", "مع", "و", "لكن", "بس",
    "حتى", "او", "ثم",
    # Negations — "مش" ("not", colloquial) can never start a name.
    "مش", "لا", "ما", "مو", "مب",
    # Verbs/decision phrases — "الدكتور يقرر بعدها كام جلسة" was extracted
    # as candidate "يقرر بعدها كم جلسه" because "يقرر" (he decides) wasn't
    # blocked either. "تحديد" is the verbal-noun form of the same حدد root
    # ("لتحديد الخطة" = "to determine/specify the plan").
    "يقرر", "تقرر", "قرر", "يحدد", "تحدد", "حدد", "تحديد", "حول",
    # Review/follow-up verb family (root ط-ل-ع) — "دكتور يطلع على
    # التقرير"/"...لتتطلع عليها" ("...for [them] to review it") are always
    # an action described ABOUT a report/case, never a person's name.
    "يطلع", "تطلع", "اطلع", "إطلع", "يتطلع", "تتطلع", "اطلعت",
    # Adjectives/quantifiers that routinely follow a title in a scheduling
    # context but are never themselves a name ("دكتور اقرب" = "nearest
    # doctor slot", not a name called "اقرب").
    "اقرب", "أقرب", "ابعد", "أبعد", "احسن", "أحسن", "افضل", "أفضل",
    # Recommendation/selection markers (mirrors _RECOMMENDATION_CONTEXT_
    # MARKERS further below — folded in here too so a bare "دكتور انصح"
    # can't seed a name either).
    "انصح", "انصحك", "ننصح", "نرشح", "رشح",
    # Vocative/help phrases — the PATIENT addressing the agent as "يا
    # دكتور" is not naming a physician at all; real regression: "دكتور
    # ممكن تساعدني؟" extracted "ممكن تساعدني" as a "name".
    "ممكن", "تقدر", "تقدري", "احتاج", "عايزك", "عاوزك",
    # Restrictive/quantifying word attached to a title — "دكتور فقط"
    # means "male doctors ONLY" (a gender/availability qualifier), never a
    # name; real regression: a line-separated list "دكتور فقط / دكتور
    # شريف / ودكتور احمد" extracted "فقط" as if it were the first doctor's
    # name alongside the two genuine ones that follow.
    "فقط",
    # Copula/"to be" verb forms (root ك-و-ن) — a closed grammatical class,
    # never a person's name — real regression: "دكتور بتكون عمليه مو
    # جلسات" / "التمريض بيكون ..." extracted the verb itself as if it were
    # a name, since no form of "to be" was ever on this blocklist.
    "بتكون", "هتكون", "بيكون", "يكون", "تكون", "هيكون", "حيكون", "كانت",
}

# Arabic subordinating conjunctions / adverbial connectors — UNLIKE every
# other blocklist in this module, this is a genuinely CLOSED, FINITE
# grammatical category (Arabic has a fixed, small inventory of these
# particles; there is no open-ended vocabulary to keep discovering one
# regression at a time). A name can never grammatically START with one of
# these, because they only ever introduce a subordinate clause or an
# adverbial qualifier, never a noun phrase. Listed here as a complete
# category rather than added piecemeal after each new observed phrase —
# real regressions this closes generally (not by matching these exact
# strings elsewhere): "دكتور حسب الحالة بالتنسيق..." ("a doctor, depending
# on the case, in coordination with...") and "طبيب اذا حابب..." ("a doctor,
# if you'd like...") are both a BARE, UNNAMED title followed by a
# conditional/qualifying clause — not a name at all — which no amount of
# rejecting specific downstream words can fully cover, since the qualifying
# clause that follows is unbounded free text. Rejecting the SUBORDINATOR
# itself closes the whole category at once.
_SUBORDINATING_CONJUNCTIONS = {
    "اذا", "إذا", "لو", "لولا", "حيث", "بينما", "عندما", "كي", "لكي",
    "بناء", "بناءا", "حسب", "نظرا", "طبقا", "وفقا", "ريثما",
    "كأن", "كأنه", "إلا", "الا", "سوى", "اما", "أما", "إما", "لان", "لأن",
    "عشان", "علشان",
}

# Preposition-plus-attached-pronoun fusions ("عليها" = على+ها, "معه" =
# مع+ه, "بها" = ب+ها, ...) — a STRUCTURAL/grammatical pattern, not a word
# list: built by combining the small, closed set of core Arabic
# prepositions with the closed set of attached object-pronoun suffixes, so
# adding a new preposition or suffix here completes a category rather than
# patching one observed phrase. A person's name can never itself be a
# preposition fused with a pronoun — real regression: "...لتتطلع عليها"
# ("...for [them] to review it") left "عليها" as a trailing "name" token
# once the actual title+name span had already ended.
_PREP_PRONOUN_FUSION_RE = re.compile(
    r"^(?:علي|عن|من|في|الي|إلي|مع|عند|ب|ل)(?:ه|ها|هم|هن|كم|كن|نا|ني)$"
)

# Currency, medication/consumable, and treatment-plan nouns — another
# closed, non-exhaustive class that can never itself be a person's name,
# even though it routinely follows a title in real scheduling/consultation
# chat (real regressions: "دكتور 375 ريال للكشفية", "دكتور و فيتامين ب12",
# "دكتور للتقييم وتحديد الخطة").
_NON_NAME_MISC_WORDS = {
    "ريال", "جنيه", "دولار", "درهم", "دينار",
    "فيتامين", "فيتامينات", "دواء", "دوا", "ادويه", "أدوية",
    "حبوب", "اقراص", "أقراص", "مكمل", "مكملات",
    "جلسه", "جلسة", "جلسات", "التقييم", "تقييم", "الخطه", "خطه", "خطة",
    "بدايه", "بداية",
    # Clinical/administrative ACTION or SERVICE nouns — the PURPOSE of a
    # doctor visit ("طبيب لكتابة وصفة" = "a doctor IN ORDER TO write a
    # prescription", "طبيب لعمل الكشف" = "...to do the exam", "طبيب لقياس
    # العلامات الحيوية" = "...to measure vital signs"), never a person's
    # name. These BARE (non-prefixed) forms cover the rare case the word
    # appears with no attached "لـ" at all; the far more common PREFIXED
    # form ("لكتابة"/"لعمل"/"لإجراء"/"لقياس"/"لتقرير") is recognised
    # separately via _LAM_PURPOSE_ACTION_WORDS/
    # _strip_lam_prefix_for_purpose_words below, without enumerating every
    # prefixed surface form here too. "كشف"/"تحديد"/"خطة" are already
    # covered elsewhere (_NON_NAME_ADMIN_WORDS/_NON_NAME_FIRST_WORDS/
    # above) — only the words not already blocked are added here.
    "كتابة", "كتابه", "وصفة", "وصفه", "عمل", "إجراء", "اجراء",
    "قياس", "تقرير",
    # English function/connector words — real regression: an English
    # self-introduction ("this is Dr. Mohamed Ahmed FROM Andalusia") or a
    # bare "the doctor WILL DECIDE how many sessions..." extracted the
    # trailing filler as if it were part of a person's name, since no
    # English word was ever on any blocklist before. Deliberately a small,
    # closed set of the specific connector/filler words observed in real
    # transcripts — not an attempt at general English stop-word coverage.
    "this", "is", "are", "speaking", "from", "will", "would", "the",
    "how", "may", "help", "calling", "today", "now", "please", "thank",
    "thanks", "regards", "care", "customer", "decide", "confirmed",
    # Bare articles and a stray possessive "s" — "Dr Ahmed Afandi's
    # specialty" and "Is Dr Ahmed Afandi A consultant?" were leaving a
    # single leftover token ("s"/"a") in the extracted candidate once
    # normalize_arabic_text() strips the apostrophe. Safe to block as a
    # standalone token: a genuine name is never JUST "a"/"an"/"s".
    "a", "an", "s",
    # English personal pronouns and WH-question words — a genuinely
    # CLOSED, finite grammatical class (mirrors the Arabic pronoun/
    # question-word entries already in _NON_NAME_FIRST_WORDS: هو، هي، مين،
    # ايه، فين), completed here rather than discovered one phrase at a
    # time. Real regression: "Who is the doctor you want to book with?"
    # extracted "you want to book" as a person's name purely because no
    # English pronoun or question word was ever blocked — a title
    # (bare "doctor") at the END of a question, with an active verb
    # elsewhere in the same clause, must never seed a name from whatever
    # pronoun/question-word happens to follow.
    "you", "he", "she", "they", "we", "it", "i",
    "who", "what", "which", "whom", "whose", "when", "where", "why",
    # English prepositions — completing the same closed class already
    # partly covered ("from" above) — real regression: "a female
    # specialist FOR the therapy sessions" extracted "for" as a name once
    # "specialist" was recognised as a title anchor (see _DOCTOR_TITLE_RE),
    # since no other English preposition besides "from" was ever blocked.
    "for", "with", "to", "of", "by", "at", "on", "as", "about", "into",
    "onto",
    # Generic-role adjective — "the appropriate doctor" is a role
    # reference, not a name (mirrors _GENERIC_SPECIALTY_WORDS' orthopedic/
    # female/male/... entries).
    "appropriate",
}

# Booking/administrative/temporal nouns that must NOT be mistaken for the
# start of (or a continuation of) a proper name — real regression:
# "حجز مع الدكتور الساعة ٦" extracted "الساعة 6" as a doctor candidate
# purely because these words carry no medical/specialty meaning and so
# weren't covered by _GENERIC_SPECIALTY_WORDS. Deliberately a closed,
# small class (days of the week, "hour/appointment/booking/clinic/
# department") rather than an attempt at exhaustive vocabulary coverage —
# same non-exhaustive philosophy as the other blocklists here.
_NON_NAME_ADMIN_WORDS = {
    "موعد", "حجز", "كشف", "عياده", "عيادة", "قسم", "ساعه", "ساعة",
    "بكره", "بكرة", "اليوم", "يوم", "الاسبوع", "اسبوع", "الشهر", "شهر",
    "اثنين", "ثلاثاء", "اربعاء", "خميس", "جمعه", "جمعة", "سبت", "احد",
}

# Degree/generic-role title words — distinct from _GENERIC_SPECIALTY_WORDS
# (medical fields/procedures): these are credential/role words that
# legitimately anchor a name search of their own ("الاستشاري اسامة عبد
# السلام" — see _DOCTOR_TITLE_RE below) but must never themselves be
# mistaken for (the start of) a person's name when they show up as the
# FIRST word of a candidate instead — real regression: "طبيب مختص؟" ("a
# specialist doctor?" — a generic, unnamed role reference) extracted
# "مختص" as if it were a doctor's name, and "دكتور استشاري محمد" (title
# immediately followed by a second credential word before the real name)
# would otherwise fold "استشاري" into the captured name. Deliberately a
# small, closed, non-exhaustive class — same philosophy as every other
# blocklist here — not an attempt to enumerate every clinical credential.
#
# دكتور/دكتوره/دكتورة/طبيب/طبيبه/طبيبة (the primary title words
# _DOCTOR_TITLE_RE itself anchors on) are included here too — NOT
# redundant with _DOCTOR_TITLE_RE despite both recognising the same words:
# _DOCTOR_TITLE_RE only ever consumes the title as the ANCHOR each
# candidate search starts AFTER, so it safely removes a title at the very
# START of a message/occurrence, but does nothing to stop an ALREADY-
# STARTED candidate's tail from swallowing a SECOND title word that shows
# up later, mid-sentence, before the next doctor's actual name — real
# regression: "الدكتورة بدريه والطبيبه اميره بركات" (two doctors offered
# in one turn) extracted "بدريه والطبيبه اميره بركات" as ONE candidate for
# the FIRST doctor, because "والطبيبه" ("و" + "الطبيبة", fused with no
# space — ordinary Arabic conjunction attachment) was never recognised as
# a continuation-stopping word: _doctor_name_candidates_in_text's own
# multi-occurrence scan DOES find "طبيبه" inside "والطبيبه" as a fresh
# title anchor for a SECOND candidate, but that never trims the FIRST
# candidate's already-in-progress tail, which needs this blocklist (via
# _is_non_name_word's prefix-stripping — see _strip_leading_wa/_strip_
# contracted_prefix) to stop at "والطبيبه" instead of running through it.
_DEGREE_TITLE_WORDS = {
    "استشاري", "استشاريه", "استشارية",
    "اخصائي", "أخصائي", "اخصائيه", "أخصائية",
    "مختص", "مختصه", "مختصة",
    "دكتور", "دكتوره", "دكتورة", "طبيب", "طبيبه", "طبيبة",
    "consultant", "specialist", "doctor", "dr",
}

# Doctor UNAVAILABILITY/departure/non-existence vocabulary — real
# regression: "الطبيب حاتم غادر اندلسيه" ("Dr Hatem has LEFT Andalusia")
# extracted "حاتم غادر اندلسيه" as if the whole phrase were a name, and
# "الدكتور غير متاح"/"الطبيب لم يعد يعمل معنا"/"لا يوجد دكتور بهذا الاسم"
# (no real name in any of them at all) extracted "غير"/"لم يعد يعمل"/
# "بهذا الاسم" as if THOSE were names. Split into two tiers:
#   _DOCTOR_DEPARTURE_ACTION_WORDS — strong action verbs describing a
#     SPECIFIC named doctor's own status (غادر/استقال/انتقل/توقف/رفض, all
#     conjugations used here) — when one of these immediately follows an
#     otherwise-plausible name, _name_candidate_from_title_match doesn't
#     just STOP there (like any other continuation-stop word); it
#     suppresses the WHOLE candidate, because the mention describes a
#     doctor who is no longer an active recommendation at all, never a
#     genuine candidate with a merely-truncated name (see that function's
#     own use of this set below).
#   The remaining negation/non-existence words (غير/لم/يعد/يعمل/معنا/يوجد/
#     بهذا/هذا/اسم/مفيش) cover the "no real name was ever given" phrasings
#     — ordinary first-word/continuation blocking (via _NON_NAME_WORDS,
#     same as every other entry there) already yields no candidate at all
#     for those, since the negation word is the very first thing after the
#     title with nothing name-shaped before it.
_DOCTOR_DEPARTURE_ACTION_WORDS = {
    "غادر", "غادرت", "غادروا",
    "استقال", "استقالت", "استقالوا",
    "انتقل", "انتقلت", "انتقلوا",
    "توقف", "توقفت",
    "رفض", "رفضت",
}
# Deliberately NOT a bare "غير" (unlike, say, "لم"/"يعد" below): "غير" alone
# is an ordinary Arabic word ("other/non-") that can legitimately appear as
# part of a real (or test-fixture) name/surname — blocking it unconditionally
# would reject a genuine candidate. Only the specific 2-word phrase "غير
# متاح[ة/ه]" ("not available") is a reliable unavailability signal — see
# _unavailable_phrase_len below, which checks this multi-word pattern
# directly rather than adding "غير" to the single-word blocklist.
_DOCTOR_UNAVAILABLE_PHRASE_STARTS: dict[str, set[str]] = {
    "غير": {"متاح", "متاحه", "متاحة"},
}
_DOCTOR_UNAVAILABLE_WORDS = _DOCTOR_DEPARTURE_ACTION_WORDS | {
    "لم", "يعد", "يعمل", "معنا", "يوجد", "بهذا", "هذا", "اسم", "مفيش",
}


def _unavailable_phrase_len(words: list[str], i: int) -> int:
    """Length (in words), or 0, of a doctor-unavailability PHRASE (see
    _DOCTOR_UNAVAILABLE_PHRASE_STARTS) starting at words[i] — unlike the
    single-word _DOCTOR_UNAVAILABLE_WORDS/_DOCTOR_DEPARTURE_ACTION_WORDS
    sets, this only ever matches the exact multi-word sequence, so a word
    like "غير" that is ambiguous on its own (could start "غير متاح" OR be
    an ordinary word/name-fragment elsewhere) is never treated as a
    rejection signal in isolation."""
    continuations = _DOCTOR_UNAVAILABLE_PHRASE_STARTS.get(words[i])
    if continuations and i + 1 < len(words) and words[i + 1] in continuations:
        return 2
    return 0

# Combined "this token can never be (part of) a doctor's proper name"
# blocklist, and the single check function used for BOTH the first-word
# gate and mid-candidate continuation cutoff in _doctor_name_candidate
# below — one general rule instead of two subtly different ones.
_NON_NAME_WORDS = (
    _GENERIC_SPECIALTY_WORDS | _NON_NAME_FIRST_WORDS | _NON_NAME_ADMIN_WORDS
    | _NON_NAME_MISC_WORDS | _DEGREE_TITLE_WORDS | _SUBORDINATING_CONJUNCTIONS
    | _DOCTOR_UNAVAILABLE_WORDS
)


def _strip_leading_al(word: str) -> str:
    """Strip a leading Arabic definite article 'ال' from a single already-
    tokenised word, for blocklist membership checks only — 'التخدير' and
    'تخدير' must both be recognised as the same generic word without
    listing every definite-article variant of every specialty/admin term
    twice (real regression: "دكتور التخدير" wasn't caught because the
    blocklist only had "تخدير", not "التخدير"). Never applied to arbitrary
    running text or to a name once accepted — 'ال' is a normal, productive
    Arabic prefix that would be unsafe to strip outside this narrow,
    closed-class-membership check."""
    return word[2:] if word.startswith("ال") and len(word) >= 4 else word


def _strip_leading_wa(word: str) -> str:
    """Strip a leading Arabic conjunction 'و' (and) from a single already-
    tokenised word, for the same narrow blocklist-membership purpose as
    _strip_leading_al — real regression: "دكتور امير وهو طالب صورة"
    over-captured "امير وهو" because "وهو" (bare pronoun "هو" with an
    attached "و") wasn't recognised as a continuation-stopping word."""
    return word[1:] if word.startswith("و") and len(word) >= 3 else word


def _strip_lam_taa_prefix(word: str) -> str:
    """Strip a leading Arabic 'ل' (lam of purpose, "in order to/for") ONLY
    when immediately followed by 'ت' — the "لـ + تفعّل/تفعيل" construction
    (a subordinated verb or verbal noun: "لتحديد" = "in order to
    determine/specify", "لتتطلع" = "in order to look at/review") — for the
    same narrow blocklist-membership purpose as _strip_leading_al/
    _strip_leading_wa. Deliberately narrower than stripping a bare leading
    'ل' in general: several genuine Arabic names begin with 'ل' followed by
    a DIFFERENT letter (e.g. "لمياء", "لبنى", "ليلى"), so only the specific
    'لت' combination is treated as this grammatical prefix rather than part
    of a name — real regression: "...للاخصائي لتتطلع عليها" ("...to the
    specialist, for [them] to review it") left "لتتطلع" as a stray one-word
    'name' once its trailing "عليها" (a preposition+pronoun fusion — see
    _PREP_PRONOUN_FUSION_RE) was already correctly cut off. Checked
    against the FULL _NON_NAME_WORDS vocabulary (see _is_non_name_word) —
    safe specifically because "لت..." is a narrow enough prefix that it
    essentially never collides with an ordinary dative/benefactive "ل"
    attached to some OTHER already-blocked word (contrast with
    _strip_lam_prefix_for_purpose_words below, which is intentionally
    checked against a much narrower vocabulary for exactly that reason)."""
    return word[1:] if word.startswith("لت") and len(word) >= 4 else word


# Purpose/action nouns commonly attached via the "ل" (purpose, "in order
# to/for") preposition directly after a generic doctor title — "لكتابة" =
# "in order to write", "لعمل" = "in order to do/perform", "لإجراء" = "in
# order to conduct", "لقياس" = "in order to measure". See
# _strip_lam_prefix_for_purpose_words.
_LAM_PURPOSE_ACTION_WORDS = {
    "كتابة", "كتابه", "وصفة", "وصفه", "عمل", "إجراء", "اجراء",
    "كشف", "قياس", "تقييم", "تحديد", "تقرير", "خطة", "خطه",
}


def _strip_lam_prefix_for_purpose_words(word: str) -> str:
    """Strip ANY leading Arabic 'ل' from *word* — deliberately UNCONDITIONAL
    (unlike _strip_lam_taa_prefix, which only fires for 'لت...') — but used
    ONLY to check membership against the narrow _LAM_PURPOSE_ACTION_WORDS
    set above, NEVER the full _NON_NAME_WORDS vocabulary (see
    _is_non_name_word).

    This separation is the actual fix for a real regression this
    generalisation introduced during development: stripping a bare
    leading 'ل' and checking it against the FULL blocklist also matches
    the ordinary dative/benefactive "ل" ("لبكرة" = "for tomorrow", part
    of "غير موعد الدكتور لبكرة" = "change the doctor's appointment to
    tomorrow") purely because "بكرة" is ALSO separately blocked (as a
    temporal/admin word) — that reclassified a perfectly ordinary
    rescheduling clause as a role reference. Checking the general strip
    ONLY against this small, closed purpose/action vocabulary (never the
    admin/temporal/pronoun/preposition words also in _NON_NAME_WORDS)
    avoids that collision entirely: "لبكرة" strips to "بكرة", which is
    not in _LAM_PURPOSE_ACTION_WORDS, so it never matches here — only
    "لكتابة"/"لعمل"/"لإجراء"/etc. do.

    A genuine Arabic name beginning with 'ل' (e.g. "لمياء", "لبنى",
    "ليلى") is unaffected either way: the stripped remainder ("مياء"/
    "بنى"/"يلى") is not itself in _LAM_PURPOSE_ACTION_WORDS."""
    return word[1:] if word.startswith("ل") and len(word) >= 3 else word


def _strip_contracted_prefix(word: str) -> str:
    """Strip a contracted preposition+article prefix ('لل' = ل+ال "for
    the", or 'بال'/'كال'/'فال'/'وال' = ب/ك/ف/و + ال) from a single
    already-tokenised word, for the same narrow blocklist-membership
    purpose as _strip_leading_al/_strip_leading_wa — real regression:
    "دكتور للتقييم وتحديد الخطة" ("doctor, for the assessment...") wasn't
    caught because only a bare 'ال' prefix was ever stripped, never 'لل'."""
    for prefix in ("لل", "بال", "كال", "فال", "وال"):
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            return word[len(prefix):]
    return word


def _is_non_name_word(word: str) -> bool:
    """True when *word* (or that word with a leading ال/و/لل/بال/... morpheme
    stripped) is a specialty/department/service/generic-role/booking/
    temporal/function word, OR structurally matches a closed Arabic
    grammatical pattern that can never be a person's name (a preposition
    fused with an attached pronoun — see _PREP_PRONOUN_FUSION_RE) — i.e.
    can never itself be, or continue, a person's proper name. See
    _NON_NAME_WORDS and the strip helpers above for why the vocabulary side
    of this generalises rather than hardcoding every surface form, and
    _PREP_PRONOUN_FUSION_RE's docstring for why the grammatical side is a
    closed structural pattern rather than a word list."""
    if not word:
        return False
    if _PREP_PRONOUN_FUSION_RE.match(word):
        return True
    # Checked against the NARROW purpose/action vocabulary only — see
    # _strip_lam_prefix_for_purpose_words' docstring for why a general "ل"
    # strip must never be checked against the full _NON_NAME_WORDS set.
    if _strip_lam_prefix_for_purpose_words(word) in _LAM_PURPOSE_ACTION_WORDS:
        return True
    return bool(
        {
            word, _strip_leading_al(word), _strip_leading_wa(word),
            _strip_leading_al(_strip_leading_wa(word)), _strip_contracted_prefix(word),
            _strip_lam_taa_prefix(word),
        }
        & _NON_NAME_WORDS
    )

# IMPORTANT: this must search the RAW (pre-normalisation) text, never
# normalize_arabic_text()'s output — that shared normaliser already strips
# a LEADING دكتور/دكتورة/د prefix itself (app.services.text_helpers's own
# _strip_dr_prefix, used elsewhere for doctor/specialty-name DB lookups),
# but only when the prefix sits at position 0 of the string. Searching for
# the title inside already-normalised text is unreliable — it disappears
# silently when anchored at the start, and is never found at all mid-
# sentence either way. Matching the raw text directly and normalising only
# the extracted remainder avoids depending on that anchoring quirk.
# (?<![\w]) before bare "د" approximates a left word-boundary for Arabic so
# a bare "د" only counts as the title when it starts its own word. "طبيب"
# (physician) is recognised too — filtered by the exact same generic-word
# logic below, so a bare "من طبيب؟"/"الطبيب المعالج" still correctly
# yields no candidate.
#
# استشاري/اخصائي/أخصائي/consultant/specialist are ALSO valid name-anchoring
# titles, not just دكتور/طبيب/Dr — real regression: "متوفر معنا الاستشاري
# اسامة عبد السلام" never even attempted name extraction because "استشاري"
# wasn't recognised as a title at all, so the actual named doctor was
# silently dropped while an unrelated "طبيب" mention elsewhere in the same
# turn produced noise instead. Widening the anchor is safe: a bare
# "استشاري"/"اخصائي" with no real name after it (e.g. "استشاري الأشعة" — a
# generic role/specialty reference) still correctly yields no candidate,
# since _DEGREE_TITLE_WORDS/_GENERIC_SPECIALTY_WORDS reject the tail the
# same way they already do for دكتور/طبيب.
_DOCTOR_TITLE_RE = re.compile(
    r"(?:دكتور[ةه]?|طبيب[ةه]?|استشاري[ةه]?|اخصائي[ةه]?|أخصائي[ةه]?|"
    r"dr\.?|doctor|consultant|specialist|(?<![\w])د(?:[./\\-])?)(?![\w])\s*[\(\[]?\s*", re.I,
)

# PRIMARY doctor-referring titles only (دكتور/طبيب/Dr/د) — deliberately
# EXCLUDES the credential/degree words استشاري/اخصائي/consultant/specialist
# that _DOCTOR_TITLE_RE also anchors on. Used ONLY for "is this turn
# actually REFERRING TO a doctor (possibly without naming them, e.g. a
# negated/unavailable mention)" boundary checks — see _turn_is_doctor_
# context_boundary/_agent_turns_text_for_doctor — never for name
# extraction itself (_doctor_name_candidate and friends keep using the
# full _DOCTOR_TITLE_RE, unchanged). Real regression: a BARE degree claim
# like "اخصائيه" (stated on its own line, answering "استشاري؟") itself
# matches _DOCTOR_TITLE_RE (via its own "اخصائي[ةه]?" alternative) with an
# empty tail — using the full title regex for the boundary check wrongly
# treated that legitimate bare follow-up claim as if it were a doctor
# reference with no name, resetting the active doctor and losing the
# claim entirely, exactly backwards from what "a bare follow-up claim
# stays attached to the active doctor" is supposed to do.
_PRIMARY_DOCTOR_TITLE_RE = re.compile(
    r"(?:دكتور[ةه]?|طبيب[ةه]?|dr\.?|doctor|(?<![\w])د(?:[./\\-])?)(?![\w])\s*[\(\[]?\s*", re.I,
)

# Extraction boundary: stop at punctuation/parentheses, or at a preposition/
# structural marker that signals the sentence has moved on from a name to
# branch/appointment/context detail (mirrors location_validation.py's
# _BRANCH_ANCHOR_RE "stop at the next structural marker" approach). Real
# regression: "تم تأكيد حجزك ... مع د/ ( محمد الالفي ) في فرع ..." was
# previously captured almost in full instead of stopping at "في فرع".
_NAME_STOP_RE = re.compile(
    r"[\)\]\.,،؟!:؛\n]|\s+(?:في|علي|على|يوم|الساعه|الساعة|بتاريخ|رقم|فرع|"
    r"مستشفى|مستشفي|عياده|عيادة|من\s+قسم|من\s+مجموع[ةه])\b",
    re.I,
)
_NAME_MAX_TOKENS = 4  # defensive cap even when no stop marker is found

# Continuation cutoff: a name candidate must also stop the moment a later
# token stops looking like part of a proper name — e.g. Patient: "الدكتور
# محمد الالفي عايز اعرف كام الكشف عنده" previously captured "محمد الالفي
# عايز اعرف" in full because "عايز"/"اعرف" contain no punctuation or
# structural marker for _NAME_STOP_RE to anchor on. Reuses _NON_NAME_WORDS
# (a function/admin/specialty word is never part of a name, wherever it
# appears — checked via _is_non_name_word, so ال/و-prefixed forms are
# covered automatically) plus a few common conversational verbs/participles
# that routinely follow a name mid-sentence but are never themselves part
# of one.
_NAME_CONTINUATION_EXTRA_STOP_WORDS = {
    "عايز", "عايزة", "عاوز", "عاوزة", "محتاج", "محتاجة", "حابب", "حابة",
    "نفسي", "عارف", "عارفة", "يقدر", "تقدر", "ممكن", "لازم",
}


def _contains_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


def is_plausible_person_name(candidate: str | None) -> bool:
    """Stage-1 lexical/person-name quality gate — the SINGLE authoritative
    check for "does this assembled string actually resemble a person's
    name", applied as a final defensive pass by both _doctor_name_candidate
    and _bare_reply_name_candidate (belt-and-suspenders alongside their own
    incremental first-word/continuation-stop logic, which builds the
    candidate token-by-token in the first place).

    Real regressions this closes: "في المنزل ولا المستشفي" (a prepositional
    phrase), "375 ريال للكشفية" (a price), "يقرر بعدها كم جلسة" (a verb
    fragment) were all being returned as "doctor names" purely because
    _is_non_name_word's blocklist didn't yet cover prepositions/negations/
    currency/session-plan vocabulary, or because a digit anywhere in the
    candidate was never checked at all. This function rejects a candidate
    when:
      - it is empty;
      - ANY token contains a digit (a name is never a price/quantity);
      - the first token is a non-name word (_is_non_name_word);
      - ANY subsequent token is a non-name word — a genuine multi-token
        name should never contain a preposition/verb/currency word
        partway through it either.
    This is deliberately a REJECTION filter, not a positive name-vocabulary
    list — Arabic personal names are far too open-ended to enumerate, so
    only the well-defined negative cases (see _NON_NAME_WORDS and its
    docstring) are ever excluded.
    """
    if not candidate:
        return False
    words = candidate.split()
    if not words:
        return False
    if any(_contains_digit(w) for w in words):
        return False
    if _is_non_name_word(words[0]):
        return False
    return not any(_is_non_name_word(w) for w in words[1:])


def _name_candidate_from_title_match(text: str, m: re.Match) -> str | None:
    """Shared per-occurrence extraction logic behind BOTH
    _doctor_name_candidate (first title occurrence only) and
    _doctor_name_candidates_in_text (every occurrence) — kept as one
    function so the two can never drift into subtly different extraction
    rules. See _doctor_name_candidate's docstring for the extraction
    contract."""
    tail = text[m.end():]
    stop = _NAME_STOP_RE.search(tail)
    raw = tail[:stop.start()] if stop else tail
    rest = normalize_arabic_text(raw)
    words = rest.split()[:_NAME_MAX_TOKENS]
    if not words or _is_non_name_word(words[0]) or _contains_digit(words[0]):
        return None
    if _unavailable_phrase_len(words, 0):
        # "الدكتور غير متاح" ("the doctor is not available") — no name is
        # given at all here (the phrase starts immediately after the
        # title), so this must yield no candidate, exactly like the
        # single-word rejections just above — see _unavailable_phrase_len.
        return None
    for i, word in enumerate(words[1:], start=1):
        if word in _DOCTOR_DEPARTURE_ACTION_WORDS or _unavailable_phrase_len(words, i):
            # Not an ordinary continuation-stop: "حاتم غادر اندلسيه" ("Dr
            # Hatem LEFT Andalusia") must never be reported as a plausible
            # candidate named "حاتم" either — a name immediately followed
            # by a departure/resignation/rejection verb describes a doctor
            # who is no longer an active recommendation at all, so the
            # WHOLE mention is suppressed here rather than merely trimmed
            # (contrast with an ordinary stop word, e.g. "دكتور محمد
            # للتقييم", which correctly keeps "محمد"). Callers needing this
            # excluded-but-real name for diagnostics/history still get it
            # via the existing "ignored non-name doctor phrase" logging
            # path (see _rejected_candidate_reason), which already fires
            # whenever a title match yields no candidate.
            return None
        if _is_non_name_word(word) or word in _NAME_CONTINUATION_EXTRA_STOP_WORDS or _contains_digit(word):
            words = words[:i]
            break
    candidate = " ".join(words)
    return candidate if is_plausible_person_name(candidate) else None


def _doctor_name_candidate(text: str) -> str | None:
    """Return a TIGHTLY-bounded name candidate immediately following the
    FIRST دكتور/طبيب/Dr title within THIS message only, or None when what
    follows is (or starts with) a generic specialty/department/service/
    booking/temporal/function word rather than a plausible proper name (see
    _is_non_name_word — this is a general entity-quality rule, not a
    hardcoded per-specialty list: "دكتور التخدير", "دكتور الأشعة", "حجز مع
    الدكتور الساعة ٦" all correctly yield no candidate because "التخدير"/
    "الأشعة"/"الساعة" are recognised as non-name words regardless of their
    attached ال). Used both as the applicability gate and as the first
    stage of name-candidate extraction for resolution — callers must pass
    ONE message/turn at a time (see extract_doctor_turn_candidates), never
    the whole concatenated transcript, or a stop marker several sentences
    away would swallow everything in between.

    Only ever returns the FIRST occurrence — see
    _doctor_name_candidates_in_text for a turn that may name more than one
    doctor (e.g. a line-separated list of options)."""
    if not text:
        return None
    m = _DOCTOR_TITLE_RE.search(text)
    if not m:
        return None
    return _name_candidate_from_title_match(text, m)


def _doctor_name_candidates_in_text(text: str) -> list[str]:
    """All plausible doctor-name candidates in *text*, one per دكتور/طبيب/Dr
    title occurrence — the multi-candidate sibling of _doctor_name_candidate
    (which only ever returns the first). Exists to support a single Agent/
    Patient turn that names more than one doctor, e.g. a short line-
    separated list of options ("دكتور أحمد\\nدكتور سالم") — real transcripts
    occasionally present doctor choices this way rather than one doctor per
    turn. Each occurrence is bounded independently by the exact same
    _NAME_STOP_RE/_is_non_name_word rules _doctor_name_candidate uses (via
    the shared _name_candidate_from_title_match), so a later occurrence's
    tail is never fused into an earlier one's candidate, and duplicate
    candidates within the same turn are not repeated."""
    if not text:
        return []
    candidates: list[str] = []
    pos = 0
    while True:
        m = _DOCTOR_TITLE_RE.search(text, pos)
        if not m:
            break
        pos = m.end()
        cand = _name_candidate_from_title_match(text, m)
        if cand and cand not in candidates:
            candidates.append(cand)
    return candidates


def detect_doctor_mention(text: str) -> bool:
    """Cheap, vocabulary-based gate: does this SINGLE message name/reference
    a specific doctor (as opposed to a generic specialty/physician
    reference)? For whole-call detection see detect_doctor_signals(), which
    is turn-aware and also excludes Agent self-introductions — this
    function alone does not know about self-introductions."""
    return _doctor_name_candidate(text) is not None


# Andalusia Agent self-introduction patterns — "السلام عليكم مع حضرتك د
# محمد من قسم الرعاية المنزلية والاشعة" is the Agent introducing THEMSELVES,
# not recommending a doctor to the patient. A doctor-like token inside such
# a turn must never become a doctor candidate. Deliberately turn-scoped: a
# LATER, separate Agent message with a genuine recommendation is never
# excluded by this.
_SELF_INTRO_RE = re.compile(
    r"مع\s*حضرتك|السلام\s*عليكم|من\s*مجموع[ةه]\s*اندلسي[ةه]|من\s*اندلسي[ةه]|"
    r"اتشرف\s*بالاسم|معاك\s*(?:في\s*)?خدم[ةه]\s*العملاء|معاك\s*(?:دكتور|طبيب|د[./\\-]?)|"
    # First-person self-identification ("أنا دكتور X" — literally "I am
    # Dr X") — the SAME structural signal as English "I am/I'm Dr X"
    # below: the SPEAKER is asserting that the title+name refers to
    # themselves. General pattern (first-person pronoun + title), not tied
    # to any specific name/organization. Word-bounded ((?<![\w])/(?![\w]))
    # — real regression: "متاح معانا دكتورة وصال" ("Dr Wesal is available
    # WITH US") was misclassified as a self-introduction purely because
    # "معانا" ends in the literal substring "انا", which an unbounded
    # "انا" pattern matched as if the Agent had said "أنا دكتورة".
    r"(?<![\w])انا(?![\w])\s*(?:ال)?(?:دكتور[ةه]?|طبيب[ةه]?|د[./\\-]?)|"
    # English self-introduction anchors — an Agent naming THEMSELVES as
    # "Dr <name>" is an employee identity, not a physician being discussed
    # with the patient. General first-person/self-identification patterns
    # (not tied to any specific name/organization) — "this is Dr X", "Dr X
    # speaking", "I am/I'm Dr X", "my name is Dr X", "you are speaking
    # with Dr X".
    # NOTE: this regex runs against normalize_arabic_text()'s OUTPUT (see
    # is_agent_self_introduction below), which lowercases and strips ALL
    # punctuation (replacing it with a space) — so a contraction like
    # "I'm"/"you're" is never a literal apostrophe by the time this regex
    # sees it; it is two separate, space-separated tokens ("i m"/"you re").
    # Patterns below are written to match that post-normalisation form,
    # not the raw apostrophe form.
    r"this\s+is\s+dr|(?:dr\.?|doctor)(?:\s+\w+){1,4}\s+speaking|speaking\s+with\s+you|"
    r"you\s+(?:re|are)\s+speaking\s+with|my\s+name\s+is\s+dr|with\s+you\s*dr|"
    r"(?<![\w])i\s+(?:am|m)\s*(?:a\s*)?(?:dr\.?|doctor)",
    re.I,
)

# General organisational-affiliation anchor — "من قسم/فرع/إدارة/رعاية/
# مجموعة X" is the Agent stating WHICH team/department/company they call
# from, the same structural role as "من مجموعة اندلسية"/"من قسم" above but
# generalised beyond one company name or the single word "قسم". Real
# regression: "د. أحمد من الرعاية المنزلية" ("Dr. Ahmed from home care
# [team]") was missed because only the literal "من قسم" was ever
# recognised — a closed, extensible set of organisational-unit nouns
# generalises this without hardcoding any specific department/company name.
#
# Kept SEPARATE from _SELF_INTRO_RE (rather than folded into it) because,
# unlike "أنا دكتور X"/"معاك دكتور X"/"مع حضرتك", an organisational-
# affiliation phrase alone is not an unambiguous first-person marker — an
# Agent can just as easily use it while recommending or referring to a
# DIFFERENT, external doctor ("أنصحك بدكتور أحمد من فرع الرياض"). So this
# anchor only counts as self-introduction when the SAME turn contains no
# explicit recommendation/referral marker — see
# is_agent_self_introduction().
_SELF_INTRO_ORG_AFFILIATION_RE = re.compile(
    r"من\s*(?:ال)?(?:قسم|فرع|إدار[ةه]|ادار[ةه]|رعاي[ةه]|مجموع[ةه])", re.I,
)
_RECOMMENDATION_OR_REFERRAL_MARKERS = {
    "انصح", "انصحك", "ننصح", "نرشح", "رشح", "recommend", "recommended",
}


def is_agent_self_introduction(text: str) -> bool:
    """Does this Agent turn look like the Agent introducing themselves
    (name/department/company), rather than talking about a patient's
    doctor? See _SELF_INTRO_RE for the unambiguous first-person patterns.

    The organisational-affiliation anchor (_SELF_INTRO_ORG_AFFILIATION_RE)
    is checked separately and only counts when the turn carries no
    recommendation/referral marker of its own (_RECOMMENDATION_OR_REFERRAL_
    MARKERS, plus the existing ordering/referring vocabulary) — otherwise
    "أنصحك بدكتور أحمد من فرع الرياض" (recommending a DIFFERENT, external
    doctor "from" a branch) would be wrongly swallowed as the Agent
    introducing themselves."""
    norm = normalize_arabic_text(text or "")
    if _SELF_INTRO_RE.search(norm):
        return True
    if not _SELF_INTRO_ORG_AFFILIATION_RE.search(norm):
        return False
    turn_tokens = set(norm.split())
    if turn_tokens & _RECOMMENDATION_OR_REFERRAL_MARKERS:
        return False
    if turn_tokens & _ORDERING_VERB_MARKERS:
        return False
    return True


def _rejected_candidate_reason(raw_tail: str) -> str:
    """Best-effort, observability-only classification of WHY a title's tail
    failed to become a name candidate — never consulted by any routing or
    validation decision (those only ever care THAT it was rejected, via
    is_plausible_person_name), purely so a rejection reads as "a specialty
    word"/"a generic role word"/"a non-person phrase" in the logs instead of
    one undifferentiated "not_a_plausible_person_name" for every case."""
    words = normalize_arabic_text(raw_tail).split()
    if not words:
        return "empty_after_title"
    # Checked FIRST, across every word (not just the first) and BEFORE the
    # degree-role/specialty checks below — deliberately: "للغدد بهذا
    # الاسم" ("for the glands, by this name") starts with a specialty
    # word ("غدد"/"للغدد"), but the phrase as a WHOLE is fundamentally an
    # UNAVAILABILITY statement (the "بهذا الاسم" tail is what actually
    # carries the meaning here), not a legitimate specialty-context
    # signal — a genuine name can also legitimately precede the
    # departure/unavailability marker instead ("حاتم غادر اندلسيه"). Real
    # regression: classifying this "specialty_not_person_name" FIRST (the
    # original order) let _first_specialty_phrase_in_text treat the whole
    # negated mention as trustworthy specialty context and attribute it to
    # an unrelated, later-named doctor (see _turn_is_doctor_context_
    # boundary, which also relies on this specific ordering) — purely so
    # the mention stays identifiable as diagnostic/historical evidence in
    # the logs either way, never lumped into the generic "not_a_plausible_
    # person_name" bucket.
    if any(w in _DOCTOR_UNAVAILABLE_WORDS for w in words):
        return "doctor_departed_or_unavailable"
    if any(_unavailable_phrase_len(words, i) for i in range(len(words))):
        return "doctor_departed_or_unavailable"
    first = words[0]
    variants = {first, _strip_leading_al(first), _strip_leading_wa(first), _strip_contracted_prefix(first)}
    if variants & _DEGREE_TITLE_WORDS:
        return "generic_role_not_person_name"
    if variants & _GENERIC_SPECIALTY_WORDS:
        return "specialty_not_person_name"
    if _contains_digit(first):
        return "non_person_phrase"
    return "not_a_plausible_person_name"


def _trim_to_specialty_phrase(raw_tail: str) -> str:
    """Bound a specialty-rejected tail down to just its CONTIGUOUS leading
    run of specialty-vocabulary tokens (e.g. "مخ واعصاب بناءا على الفحص..."
    -> "مخ واعصاب", not the whole rest of the clause) — _NAME_STOP_RE only
    stops at punctuation/structural markers, which is the right boundary
    for a NAME candidate but too loose for reporting a clean specialty
    phrase as supporting evidence. Falls back to the original tail
    (unchanged) if no specialty word is found at all, e.g. a caller that
    already confirmed _rejected_candidate_reason == "specialty_not_person_
    name" for a single-token tail."""
    words = normalize_arabic_text(raw_tail).split()
    kept: list[str] = []
    for word in words:
        # Same variant set _rejected_candidate_reason itself checks (see
        # its own {first, _strip_leading_al(first), _strip_leading_wa(
        # first), _strip_contracted_prefix(first)}) — previously missing
        # _strip_contracted_prefix here specifically, so a "لل"-prefixed
        # specialty word ("للغدد" = لل + غدد) was correctly recognised as
        # a specialty word BY _rejected_candidate_reason (which is what
        # let this whole tail qualify as "specialty_not_person_name" in
        # the first place) but then NOT recognised here, leaving `kept`
        # empty and falling back to returning the WHOLE raw tail
        # unchanged — e.g. "للغدد بهذا الاسم" instead of just "للغدد".
        variants = {
            word, _strip_leading_al(word), _strip_leading_wa(word),
            _strip_leading_al(_strip_leading_wa(word)), _strip_contracted_prefix(word),
        }
        if variants & _GENERIC_SPECIALTY_WORDS:
            kept.append(word)
            continue
        if kept:
            break
    return " ".join(kept) if kept else raw_tail


# A transcript-only stand-in for CallTranscript, used solely to feed the
# memoised _*_cached() functions below a plain, hashable string key while
# letting the (unchanged) *_impl() functions keep reading `call.transcript`
# exactly as before — see extract_doctor_turn_candidates/
# classify_specific_doctor_intent's caching notes for why this exists.
class _TranscriptOnly:
    __slots__ = ("transcript",)

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript


def _extract_doctor_turn_candidates_impl(call: CallTranscript) -> tuple[list[str], list[str], list[str]]:
    """Message-by-message doctor-name-candidate extraction — never the
    whole concatenated Patient/Agent transcript (that was the root cause of
    both the false-positive AMBIGUOUS_DOCTOR regressions and the "swallows
    half the conversation" extraction bug: a real recommendation in one
    turn must not be diluted by, or fused with, an unrelated turn).

    Uses _doctor_name_candidates_in_text (not the single-result
    _doctor_name_candidate) so a turn that names MORE THAN ONE doctor — a
    short line-separated list of options ("دكتور أحمد\\nدكتور سالم") is the
    real-world case this supports — contributes every plausible candidate
    it contains, not just the first.

    Returns (patient_candidates, agent_candidates, ignored_self_intro_candidates).
    Patient mentions help resolve WHICH doctor is being discussed; only
    Agent candidates (from non-self-introduction turns) are ever treated as
    factual claims to validate.

    Not called directly outside this module — see extract_doctor_turn_
    candidates(), the public, memoised entry point every other caller uses.
    """
    patient_candidates: list[str] = []
    agent_candidates: list[str] = []
    ignored: list[str] = []
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker == "agent" and is_agent_self_introduction(text):
            ignored.extend(_doctor_name_candidates_in_text(text))
            continue
        cands = _doctor_name_candidates_in_text(text)
        if not cands:
            # A دكتور/طبيب/Dr title was present but yielded no candidate —
            # log it ONLY when the title actually matched (not for every
            # ordinary turn with no doctor mention at all), so a real
            # rejected fragment (a specialty/role phrase, a preposition, a
            # price, a verb fragment, ...) is visible in the logs as a
            # deliberate rejection, not silence.
            title_match = _DOCTOR_TITLE_RE.search(text)
            if title_match:
                raw_tail = text[title_match.end():].strip()[:80]
                _doc_print_section(
                    "ignored non-name doctor phrase",
                    speaker=speaker, raw_candidate=raw_tail or None,
                    turn=text.strip()[:120], reason=_rejected_candidate_reason(raw_tail),
                )
            continue
        (agent_candidates if speaker == "agent" else patient_candidates).extend(cands)
    return patient_candidates, agent_candidates, ignored


@functools.lru_cache(maxsize=512)
def _extract_doctor_turn_candidates_cached(
    transcript: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    patient, agent, ignored = _extract_doctor_turn_candidates_impl(_TranscriptOnly(transcript))
    return tuple(patient), tuple(agent), tuple(ignored)


def extract_doctor_turn_candidates(call: CallTranscript) -> tuple[list[str], list[str], list[str]]:
    """Public entry point — memoised on the transcript TEXT (a pure
    function of it; see _extract_doctor_turn_candidates_impl for the actual
    extraction logic). Doctor extraction is otherwise re-run independently
    by the graph router, validate_doctor_information, the doctor-scope
    router, and assorted diagnostics (raw_doctor_title_tails,
    describe_excluded_doctor_candidates) — without this cache the same
    "ignored non-name doctor phrase" lines print repeatedly for every one of
    those call sites on the same call. Safe indefinitely (no staleness
    risk): identical transcript text always yields an identical result: a
    bounded LRU, not a per-call_id cache, so distinct calls (or the same
    call_id reused with different transcript text, as some tests do) never
    collide."""
    patient, agent, ignored = _extract_doctor_turn_candidates_cached(call.transcript or "")
    return list(patient), list(agent), list(ignored)


def describe_excluded_doctor_candidates(call: CallTranscript) -> list[dict[str, str]]:
    """Structured, per-candidate exclusion evidence for observability (see
    module-level logging: "[doctor] extraction:") — one
    {"name": ..., "reason": ...} entry for every doctor-shaped candidate
    that was found but deliberately NOT carried forward as a genuine
    specific-doctor target. Purely additive/diagnostic: never consulted by
    any routing or validation decision, which continue to rely solely on
    classify_specific_doctor_intent/extract_doctor_turn_candidates. Two
    exclusion categories are surfaced:
      - "agent_self_introduction" — every candidate ignored because it
        appeared in an Agent turn that was introducing the Agent themselves
        (see is_agent_self_introduction).
      - the overall doctor_role's exclusion reason (e.g.
        "conversation_addressee", "ordering_or_referring",
        "existing_doctor_reference", "administrative_reference") — applied
        to the specific named_doctor classify_specific_doctor_intent
        identified for that role, when doctor_intent stayed
        "not_applicable". The classifier currently tracks one active role
        per call rather than a per-candidate role for every mention, so
        this is a best-effort label on the one candidate it did track, not
        an exhaustive per-mention audit trail.
    """
    _patient, _agent, ignored_self_intros = extract_doctor_turn_candidates(call)
    excluded = [{"name": name, "reason": "agent_self_introduction"} for name in ignored_self_intros]
    ctx = classify_specific_doctor_intent(call)
    if ctx["doctor_intent"] == "not_applicable" and ctx["doctor_role"] not in (None, "agent_self_introduction"):
        named = ctx.get("named_doctor")
        if named:
            excluded.append({"name": named, "reason": ctx["doctor_role"]})
    return excluded


def describe_doctor_extraction_evidence(call: CallTranscript) -> dict[str, list]:
    """Full per-occurrence extraction evidence for observability — every
    دكتور/طبيب/استشاري/اخصائي/Dr/consultant/specialist title occurrence in
    the call, classified as accepted / rejected(with a specific reason),
    regardless of whether OTHER occurrences in the same turn succeeded.

    This is intentionally richer than describe_excluded_doctor_candidates()
    (which only reports one exclusion per call at the doctor_role level):
    a single Agent turn recommending a doctor often ALSO contains an
    earlier, unrelated specialty/generic-role phrase after a different
    title word in the SAME turn (e.g. "...من طبيب مخ واعصاب...متوفر معنا
    الاستشاري اسامة عبد السلام...") — the turn-level "ignored non-name
    doctor phrase" log only fires when a whole turn yields NO candidate at
    all, so it would silently miss "مخ واعصاب" once "اسامة عبد السلام" is
    also found in the same turn. This function reports every occurrence
    independently instead.

    Purely additive/diagnostic: never consulted by any routing or
    validation decision, which continue to rely solely on
    classify_specific_doctor_intent()/extract_doctor_turn_candidates().
    """
    accepted: list[str] = []
    rejected: list[dict[str, str]] = []
    for speaker, text in split_transcript_turns(call.transcript):
        is_self_intro = speaker == "agent" and is_agent_self_introduction(text)
        pos = 0
        while True:
            m = _DOCTOR_TITLE_RE.search(text, pos)
            if not m:
                break
            pos = m.end()
            tail = text[m.end():]
            stop = _NAME_STOP_RE.search(tail)
            raw = (tail[:stop.start()] if stop else tail).strip()
            if not raw:
                continue
            if is_self_intro:
                cand = _name_candidate_from_title_match(text, m)
                rejected.append({"candidate": (cand or raw)[:80], "reason": "agent_self_introduction"})
                continue
            cand = _name_candidate_from_title_match(text, m)
            if cand:
                if cand not in accepted:
                    accepted.append(cand)
            else:
                reason = _rejected_candidate_reason(raw)
                shown = _trim_to_specialty_phrase(raw) if reason == "specialty_not_person_name" else raw
                rejected.append({"candidate": shown[:80], "reason": reason})
    return {"accepted": accepted, "rejected": rejected}


def _first_specialty_phrase_in_text(text: str) -> str | None:
    """Scan a SINGLE turn's text for the first title-tail that was rejected
    specifically as a specialty reference (see _rejected_candidate_reason)
    and return it trimmed to just the specialty phrase itself (see
    _trim_to_specialty_phrase). Helper for extract_doctor_context_specialty
    — never called on the whole transcript at once."""
    pos = 0
    while True:
        m = _DOCTOR_TITLE_RE.search(text, pos)
        if not m:
            return None
        pos = m.end()
        tail = text[m.end():]
        stop = _NAME_STOP_RE.search(tail)
        raw = (tail[:stop.start()] if stop else tail).strip()
        if raw and _rejected_candidate_reason(raw) == "specialty_not_person_name":
            return _trim_to_specialty_phrase(raw)


def _turn_is_doctor_context_boundary(text: str, resolved_tokens: list[str]) -> bool:
    """True when *text* is a boundary that per-doctor claim scoping (see
    extract_doctor_context_specialty/_specialty_context_for_resolved_name/
    _agent_turns_text_for_doctor, the three callers of this shared check)
    must never cross or attribute to *resolved_tokens*'s doctor — either
    of two cases:
      1. A DIFFERENT, incompatible doctor is actually named in this turn
         (a real title+name candidate that does not match resolved_tokens
         via _is_plausible_same_person_name).
      2. A دكتور/طبيب title is present, NO name candidate survived, AND
         the tail was rejected SPECIFICALLY as a negated/unavailable/
         departed/rejected doctor reference (_rejected_candidate_reason ==
         "doctor_departed_or_unavailable" — see _DOCTOR_UNAVAILABLE_WORDS/
         _unavailable_phrase_len/_DOCTOR_DEPARTURE_ACTION_WORDS). Real
         regression: "غير متاح طبيب للغدد بهذا الاسم" (about the earlier,
         REJECTED "حاتم" request) was being walked straight through and
         its specialty-shaped tail ("للغدد...") attributed to Amira, the
         doctor named in a LATER turn, purely because the backward walk
         had no notion of "this earlier turn is about someone/something
         else entirely".

    Deliberately does NOT treat as a boundary: an ordinary title-LESS turn
    (price/offer/closing/scheduling chatter, or a genuine bare follow-up
    claim like "اخصائيه" — see _PRIMARY_DOCTOR_TITLE_RE's own docstring
    for why a bare credential word never counts as a title reference
    here), OR a title-bearing tail rejected for any OTHER reason —
    crucially, a bare SPECIALTY/service request with no name at all
    ("نحتاج طبيب مخ واعصاب") is *_rejected_candidate_reason ==
    "specialty_not_person_name"*, never "doctor_departed_or_unavailable",
    and is exactly the legitimate context extract_doctor_context_specialty
    is searching the backward walk FOR — it must never be treated as a
    boundary, or every "specialty mention, then a later recommendation"
    call this module already correctly supports would silently break."""
    turn_candidates = _doctor_name_candidates_in_text(text)
    if turn_candidates:
        return not any(
            _name_tokens(candidate)
            and _is_plausible_same_person_name(resolved_tokens, _name_tokens(candidate))
            for candidate in turn_candidates
        )
    m = _PRIMARY_DOCTOR_TITLE_RE.search(text)
    if not m:
        return False
    tail = text[m.end():]
    stop = _NAME_STOP_RE.search(tail)
    raw = (tail[:stop.start()] if stop else tail).strip()
    return _rejected_candidate_reason(raw) == "doctor_departed_or_unavailable"


def extract_doctor_context_specialty(call: CallTranscript, named_doctor: str | None = None) -> str | None:
    """Best-effort SUPPORTING evidence only: the clinical specialty/field
    phrase a title's tail was rejected as (see _rejected_candidate_reason's
    "specialty_not_person_name") — e.g. "نحتاج طبيب مخ واعصاب" yields "مخ
    واعصاب". This is the specialty CONTEXT a doctor recommendation was
    framed around in the conversation, NEVER the doctor's authoritative CRM
    specialty (only the resolved CRM record is authoritative for that —
    see validate_doctor_information's scope_reference/"specialty"). Kept
    strictly separate from app.agent.nodes.extract_appointment_details'
    LLM-extracted "specialty_name" too, which describes the requested
    SERVICE/appointment specialty, not necessarily the recommended doctor's
    field — the two must never be conflated or overwrite one another.

    When *named_doctor* is given (the doctor classify_specific_doctor_intent
    actually accepted), the search is SCOPED to the turn where that name was
    found, then walks BACKWARDS through earlier turns — never forwards past
    it, and never the whole transcript indiscriminately. This matters
    because a real call can mention more than one specialty for more than
    one reason (e.g. the PATIENT's originally requested service earlier in
    the call, vs. the FIELD the recommended doctor was introduced with) —
    grabbing "the first specialty phrase anywhere" would silently return
    the wrong one whenever they differ (real regression: a call opening
    with "...علاج طبيعي..." and only later recommending a "...مخ
    واعصاب..." doctor returned "العلاج" — the requested-service phrase —
    as if it were the recommended doctor's own specialty context).
    Without a named_doctor (e.g. no doctor was ever accepted at all), falls
    back to the first specialty phrase found anywhere, since there is no
    specific doctor mention to scope the search to.
    """
    turns = split_transcript_turns(call.transcript)
    if named_doctor:
        target_idx = None
        for i, (speaker, text) in enumerate(turns):
            if speaker == "agent" and is_agent_self_introduction(text):
                continue
            if named_doctor in _doctor_name_candidates_in_text(text):
                target_idx = i  # last occurrence wins — closest to the actual recommendation
        if target_idx is not None:
            resolved_tokens = _name_tokens(named_doctor)
            for i in range(target_idx, -1, -1):
                text_i = turns[i][1]
                if i < target_idx and _turn_is_doctor_context_boundary(text_i, resolved_tokens):
                    # An earlier turn names a DIFFERENT doctor, or is
                    # itself a negated/unavailable doctor reference — the
                    # backward walk must never cross it (see
                    # _turn_is_doctor_context_boundary's own regression
                    # note): whatever specialty-shaped text lies beyond it
                    # belongs to that OTHER context, never named_doctor.
                    break
                found = _first_specialty_phrase_in_text(text_i)
                if found:
                    return found
            return None
    for _speaker, text in turns:
        found = _first_specialty_phrase_in_text(text)
        if found:
            return found
    return None


def _specialty_context_for_resolved_name(call: CallTranscript, resolved_name: str) -> str | None:
    """Sibling of extract_doctor_context_specialty, scoped the exact same
    way (the doctor's own turn, walking BACKWARDS — never forward past it,
    never the whole transcript indiscriminately), but anchored via a SAFE
    same-person name match (_is_plausible_same_person_name) instead of
    requiring *resolved_name* to appear as an EXACT member of that turn's
    regex-extracted candidate list. Needed because that per-turn candidate
    can be a noisier superset of the actual name (e.g. "وصال بعياده" for a
    turn genuinely about "وصال") — a CRM-resolved or semantic-vetted name
    is trusted here as the anchor. Never changes what extract_doctor_
    context_specialty itself returns for any existing caller — this is an
    additive sibling, used only for per-doctor specialty-CLAIM validation
    (see _resolve_and_validate_one_doctor)."""
    resolved_tokens = _name_tokens(resolved_name)
    if not resolved_tokens:
        return None
    turns = split_transcript_turns(call.transcript)
    target_idx = None
    for i, (speaker, text) in enumerate(turns):
        if speaker == "agent" and is_agent_self_introduction(text):
            continue
        for candidate in _doctor_name_candidates_in_text(text):
            candidate_tokens = _name_tokens(candidate)
            if candidate_tokens and _is_plausible_same_person_name(resolved_tokens, candidate_tokens):
                target_idx = i  # last occurrence wins — closest to the actual recommendation
                break
    if target_idx is None:
        return None
    for i in range(target_idx, -1, -1):
        text_i = turns[i][1]
        if i < target_idx and _turn_is_doctor_context_boundary(text_i, resolved_tokens):
            break  # see _turn_is_doctor_context_boundary's regression note
        found = _first_specialty_phrase_in_text(text_i)
        if found:
            return found
    return None


def _agent_turns_text_for_doctor(call: CallTranscript, resolved_name: str) -> str:
    """Concatenated text of every AGENT turn safely linked to
    *resolved_name* (same safe same-person name match as
    _specialty_context_for_resolved_name — _is_plausible_same_person_name
    against that turn's own regex-extracted doctor-name candidates), used
    to scope RANK/DEGREE-claim (and every other optional-field) extraction
    to the doctor actually being validated. Never the whole call's
    agent_text: in a multi-doctor conversation an explicit degree claim
    about a DIFFERENT recommended doctor ("...ومتواجد دكتور احمد استشاري
    عظام") must never be attributed to this one. Self-introduction turns
    are excluded, same as the specialty-scoping sibling above.

    ALSO includes a BARE (name-less) Agent turn — one with no دكتور/طبيب-
    anchored candidate of its own at all, e.g. "اخصائيه" stated on its own
    line right after naming the doctor — as long as the MOST RECENT prior
    Agent turn that DID name a doctor was this same *resolved_name* (real
    regression: "متاح الطبيبه اميره بركات" / "اخصائيه" on the very next
    Agent turn, replying to the Patient's own follow-up question, with
    Amira's name never repeated in that second turn — her degree claim
    must still be scoped to her). This tracks the single MOST RECENTLY
    named doctor only: it resets the moment a LATER Agent turn names a
    DIFFERENT doctor (or the same doctor again, which simply keeps
    matching), so a bare follow-up claim is never misattributed once the
    conversation has moved on to someone else. Patient turns never affect
    this tracking either way — only an Agent turn that itself names (or
    fails to name) a doctor changes which doctor is "active".

    A title-bearing turn that yields NO valid name candidate at all — most
    commonly a negated/unavailable/departed/rejected doctor reference (see
    _turn_is_doctor_context_boundary) — is a BOUNDARY, not a bare
    continuation: it resets "active" to False rather than extending the
    previous doctor's scope. Real regression: "غير متاح طبيب للغدد بهذا
    الاسم" (about an earlier, rejected doctor request) was being treated
    exactly like a genuine bare follow-up ("اخصائيه") purely because
    _doctor_name_candidates_in_text also returns [] for it, letting its
    specialty/fee-shaped text keep flowing to whichever doctor happened to
    be active — see _turn_is_doctor_context_boundary's own regression note.

    Returns "" (never None) when no turn matches, since _extract_degree_
    claim already treats an empty string as "no claim" — this remains an
    additive helper, only used for optional-field scoping; it does not
    change extract_doctor_context_specialty or _specialty_context_for_
    resolved_name."""
    return " ".join(text for _, text, _ in _agent_turn_matches_for_doctor(call, resolved_name))


def _agent_turn_matches_for_doctor(
    call: CallTranscript, resolved_name: str,
) -> list[tuple[int, str, bool]]:
    """Turn-level companion to _agent_turns_text_for_doctor (same exact
    "most recently named doctor" tracking — see that function's docstring
    for the full regression history), returning (turn_index, text,
    names_doctor) for every Agent turn scoped to *resolved_name* instead
    of just the joined string. turn_index indexes into the SAME
    split_transcript_turns(call.transcript) list a caller can recompute
    deterministically (e.g. to look up the immediately preceding turn for
    a local transcript excerpt — see _field_transcript_excerpt). names_
    doctor is True only for a turn whose OWN extracted candidate(s)
    actually matched *resolved_name* (i.e. a turn that itself STATES the
    doctor's name, as opposed to a bare follow-up folded in via the
    active-doctor tracking) — used by the name_completeness WARNING check
    to find the turn where an incomplete name was actually stated."""
    resolved_tokens = _name_tokens(resolved_name)
    if not resolved_tokens:
        return []
    turns = split_transcript_turns(call.transcript)
    matched: list[tuple[int, str, bool]] = []
    _this_doctor_is_active = False
    for idx, (speaker, text) in enumerate(turns):
        if speaker != "agent" or is_agent_self_introduction(text):
            continue
        turn_candidates = _doctor_name_candidates_in_text(text)
        if turn_candidates:
            _this_doctor_is_active = any(
                _name_tokens(candidate)
                and _is_plausible_same_person_name(resolved_tokens, _name_tokens(candidate))
                for candidate in turn_candidates
            )
            if _this_doctor_is_active:
                matched.append((idx, text, True))
        elif _turn_is_doctor_context_boundary(text, resolved_tokens):
            # A negated/unavailable/departed doctor reference (never a
            # bare credential word or a bare specialty request — see
            # _turn_is_doctor_context_boundary's own docstring) — boundary.
            _this_doctor_is_active = False
        elif _this_doctor_is_active:
            matched.append((idx, text, False))
    return matched


def _field_transcript_excerpt(
    call: CallTranscript | None, resolved_name: str | None, field_predicate: Any,
) -> str | None:
    """Best-effort LOCAL transcript excerpt for one FAILED field on ONE
    resolved doctor (see the "failure_details" schema in validate_doctor_
    information's docstring: "transcript_excerpt" must be local to the
    relevant doctor/field, never cross-attributed). Picks the specific
    Agent turn — from THIS doctor's own scoped turns only, see
    _agent_turn_matches_for_doctor, never the whole call's agent_text —
    whose OWN text, checked in isolation, independently satisfies
    *field_predicate* (a text->bool check using the SAME extractor the
    field's own validation block already used), paired with the
    immediately preceding turn when it is a Patient turn so the excerpt
    reads exactly as the exchange actually happened (e.g. "Patient:
    استشاري؟\\nAgent: اخصائيه"). Falls back to this doctor's LAST scoped
    turn when no single turn's own text satisfies the predicate (a claim
    can legitimately be assembled across more than one turn) — this is
    still always a turn belonging to THIS doctor, never a blind scan."""
    if not call or not resolved_name:
        return None
    matches = _agent_turn_matches_for_doctor(call, resolved_name)
    if not matches:
        return None
    chosen: tuple[int, str] | None = None
    for idx, text, _ in matches:
        try:
            if field_predicate(text):
                chosen = (idx, text)
                break
        except Exception:
            continue
    if chosen is None:
        last_idx, last_text, _ = matches[-1]
        chosen = (last_idx, last_text)
    idx, agent_turn_text = chosen
    lines = [f"Agent: {agent_turn_text}"]
    if idx > 0:
        turns = split_transcript_turns(call.transcript)
        prev_speaker, prev_text = turns[idx - 1]
        if prev_speaker == "patient":
            lines.insert(0, f"Patient: {prev_text}")
    return "\n".join(lines)


def raw_doctor_title_tails(call: CallTranscript) -> tuple[list[str], list[str]]:
    """Diagnostic-only companion to extract_doctor_turn_candidates(): the
    UNFILTERED text immediately following EVERY دكتور/طبيب/Dr title
    occurrence (not just the first per turn — see
    _doctor_name_candidates_in_text), regardless of whether
    is_plausible_person_name() would ever accept it. Never used for any
    routing decision; exists solely so a production pre-routing log can
    show what raw text was seen next to "accepted" candidates, making a
    rejection (e.g. a preposition/price/verb fragment) as visible as an
    acceptance."""
    patient_raw: list[str] = []
    agent_raw: list[str] = []
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker == "agent" and is_agent_self_introduction(text):
            continue
        pos = 0
        while True:
            m = _DOCTOR_TITLE_RE.search(text, pos)
            if not m:
                break
            pos = m.end()
            tail = text[m.end():].strip()[:80]
            if tail:
                (agent_raw if speaker == "agent" else patient_raw).append(tail)
    return patient_raw, agent_raw


def detect_doctor_signals(call: CallTranscript) -> tuple[bool, bool, str, str]:
    """Parse the transcript once into (patient_mentions_doctor,
    agent_mentions_doctor, patient_text, agent_text) — the doctor-side
    equivalent of detect_bank_signals()/detect_location_signals().
    patient_mentions_doctor/agent_mentions_doctor are computed from
    extract_doctor_turn_candidates() (turn-aware, self-introduction-aware —
    reused by BOTH the graph router and validate_doctor_node's defensive
    guard, so there is exactly one doctor-evidence detector, not a looser
    one in graph.py and a stricter one here). patient_text/agent_text
    remain the simple joined-by-speaker strings, used elsewhere (e.g. the
    patient-complaint check for the separate semantic scope validation)."""
    patient, agent = split_transcript_by_speaker(call.transcript)
    patient_candidates, agent_candidates, _ignored = extract_doctor_turn_candidates(call)
    return bool(patient_candidates), bool(agent_candidates), patient, agent


def doctor_validation_needed(call: CallTranscript, signals: tuple[bool, bool, str, str] | None = None) -> bool:
    """The doctor-intent gate, reused by BOTH the graph router
    (_doctor_intent_router) and validate_doctor_node's/
    validate_doctor_information's own defensive fallback — there is exactly
    ONE doctor-intent classifier (classify_specific_doctor_intent), never a
    looser check here and a stricter one elsewhere.

    Applicable only when the ACTIVE conversational intent is specifically
    about a named doctor (booking/rescheduling/checking availability WITH
    them, or an inquiry ABOUT them) — NOT merely because a named doctor
    happens to appear somewhere in the transcript. A doctor who only
    ordered/referred a different service, or who is only an existing/
    follow-up relationship mentioned as context for a different active
    booking, never triggers a CRM fetch (see
    classify_specific_doctor_intent's docstring for the full reasoning).

    `signals` is accepted only for call-site backward compatibility
    (detect_doctor_signals's patient_text/agent_text are still used
    elsewhere for the separate scope-clinical-need check) — it is NOT
    consulted for this decision, since "does a name-shaped phrase appear
    anywhere" is exactly the over-broad rule this function replaced.
    """
    return classify_specific_doctor_intent(call)["doctor_intent"] != "not_applicable"


# ── Doctor identity resolution ───────────────────────────────────────────────
# Arabic hamza/tashkeel/teh-marbuta folding already happens in
# normalize_arabic_text(); here we additionally strip title prefixes so
# "دكتور محمد الألفي" and "محمد الألفي" compare equal.
def _strip_title(text: str) -> str:
    """Strip title prefixes from RAW text (see _DOCTOR_TITLE_RE's docstring
    note on why this must run before normalize_arabic_text, not after),
    then normalise the remainder."""
    stripped = _DOCTOR_TITLE_RE.sub("", text or "", count=1)
    return normalize_arabic_text(stripped)


def _name_tokens(name: str) -> list[str]:
    return [t for t in normalize_arabic_text(name or "").split() if len(t) > 1]


def _meaningful_name_token_count(candidate: str | None) -> int:
    """Count of GENUINE person-name tokens in an already-extracted doctor-
    name candidate — used only by the name_completeness WARNING check
    (see _name_completeness_warning), never by identity resolution itself
    (which already has its own established prefix/given-family matching).
    Reuses the SAME closed blocklist every other part of this module
    already trusts to tell a real name token apart from a title
    (دكتور/دكتورة/طبيب/طبيبة/Dr/Doctor), a degree/rank word (استشاري/
    اخصائي), an attached conjunction (و/والدكتورة), or a specialty/
    schedule/branch/administrative word (_is_non_name_word) — so "دكتورة
    بدرية" correctly counts as ONE meaningful token (the title isn't a
    second name) while "Dr. Ahmed Ali" correctly counts as TWO."""
    return len([t for t in _name_tokens(candidate) if not _is_non_name_word(t)])


def _name_completeness_warning(
    input_name: str | None, doctor: dict[str, Any],
    call: CallTranscript | None = None, resolved_name: str | None = None,
) -> dict[str, Any] | None:
    """Non-punitive WARNING (never a validation failure — see this
    module's docstring on the name_completeness quality notice) raised
    when the Agent only ever stated the resolved doctor's bare first
    name, never at least a first+second name, anywhere in the call.

    *input_name* is the caller's already-enriched _input_name (see
    _resolve_and_validate_one_doctor and _enrich_candidates_with_fuller_
    agent_names) — the MOST COMPLETE name the Agent themselves stated for
    this doctor anywhere in the transcript, since enrichment already
    widens a bare early mention (e.g. "اميره") to a fuller one found in a
    LATER Agent turn (e.g. "اميره بركات") before resolution ever runs.
    That is exactly why a later Agent-stated full name already clears
    this warning here for free, while a Patient-only full name never
    does: enrichment only ever draws from Agent-turn candidates (see
    _enrichment_sources in validate_doctor_information), never Patient
    turns.

    Returns None once *input_name* already carries >=2 meaningful
    person-name tokens (title/degree/conjunction/specialty/admin words
    never count — see _meaningful_name_token_count), or when there is no
    input_name at all (nothing to warn about)."""
    if not input_name or _meaningful_name_token_count(input_name) >= 2:
        return None
    crm_value = doctor.get("cr301_doctornamear") or doctor.get("servhub_doctornameen") or "Not on file"
    excerpt = None
    if call is not None and resolved_name:
        matches = _agent_turn_matches_for_doctor(call, resolved_name)
        naming_turn = next((t for t in matches if t[2]), None)
        if naming_turn is not None:
            excerpt = naming_turn[1]
    return {
        "field": "name_completeness",
        "label": "Doctor name incomplete",
        "outcome": "WARNING",
        "is_violation": False,
        "chat_value": input_name,
        "crm_value": crm_value,
        "reason": (
            "The agent stated only the doctor's first name. The preferred "
            "practice is to provide at least the first and second name."
        ),
        "transcript_excerpt": excerpt,
    }


# First names alone are too common to trust — "محمد"/"أحمد"/"سارة" must not
# resolve a doctor on their own (Part on doctor identity resolution).
_SINGLE_TOKEN_MATCH_MIN_LEN = 2  # a query must supply >=2 real name tokens
                                  # (or an exact single-token CRM name) to
                                  # resolve — see resolve_doctor_candidates.


def _ordered_prefix_match(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """True when the SHORTER of two ordered name-token sequences is an
    exact, ORDERED prefix of the longer one — the only form of partial
    person-name matching this module accepts (see resolve_doctor_candidates
    tier 3). Both Arabic and English personal names are conventionally
    stated in a FIXED order (given name, then father's/family name), so a
    real partial mention is always a PREFIX of the full legal name — never
    a same-length name with a token swapped out, and never an unordered
    bag of shared words. This is what makes "اسامة عبد السلام" (a full
    name) correctly NOT match "اسامة عبد المقصود" (same length, same first
    two tokens, but the FINAL discriminating token conflicts) while still
    matching "خيرية محمد" against "خيرية محمد علي موسى" (a genuine prefix).
    No compound-name dictionary is needed: "عبد" carrying almost no
    discriminative value on its own is handled automatically, because
    whatever token immediately follows it must ALSO align for the prefix
    relationship to hold."""
    shorter, longer = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
    if not shorter:
        return False
    return longer[:len(shorter)] == shorter


def _given_and_family_name_match(tokens_a: list[str], tokens_b: list[str]) -> bool:
    """True when BOTH the first token (given name) and the last token
    (family name) agree between two ordered name-token sequences of length
    >=2 each — regardless of length difference or what the MIDDLE tokens
    are. This is the second, equally strict acceptance path alongside
    _ordered_prefix_match: real Arabic speech routinely OMITS the middle
    father's/grandfather's name while still stating the actual family name
    ("علياء المدبولي" for a CRM record of "علياء محمود المدبولي" — the
    middle "محمود" is skipped, but "علياء" and "المدبولي" both anchor the
    same person). Anchoring on BOTH ends is what keeps this safe: "اسامة
    عبد السلام" vs "اسامة عبد المقصود" still correctly FAILS here too
    (first tokens agree, but the last tokens "السلام"/"المقصود" conflict),
    since only the middle is allowed to differ, never the two endpoints."""
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return False
    return tokens_a[0] == tokens_b[0] and tokens_a[-1] == tokens_b[-1]


def _is_plausible_same_person_name(query_tokens: list[str], name_tokens: list[str]) -> bool:
    """The single acceptance rule for tier-3 partial person-name matching —
    true if EITHER _ordered_prefix_match (a genuine truncation: the shorter
    sequence is stated in full, just fewer of the LATER components) or
    _given_and_family_name_match (a genuine omission: both endpoints — the
    given name and the actual family name — agree, only a middle
    father's/grandfather's name differs or is missing) holds. Both paths
    independently reject a same-length name with any discriminating token
    swapped out ("اسامة عبد السلام" vs "اسامة عبد المقصود", "محمد احمد" vs
    "محمد علي") and reject a shared given name alone with nothing else
    agreeing ("محمد احمد" vs "محمد علي" again — the family-name endpoints
    differ too)."""
    return _ordered_prefix_match(query_tokens, name_tokens) or _given_and_family_name_match(query_tokens, name_tokens)


# ── Cross-turn recommendation-set identity enrichment ───────────────────────
# Real regression: an Agent turn offers "بدريه والطبيبه اميره" (a genuine
# 2-candidate recommendation set), then a LATER turn — replying to the
# Patient's own follow-up question about availability — states the fuller
# "اميره بركات". The Patient never repeats the name themselves, but the
# Agent's own later turn is exactly the kind of transcript evidence CRM
# resolution needs: a bare first-name-only candidate should be superseded by
# a more complete, name-COMPATIBLE mention found anywhere else the Agent
# spoke, in either direction of the conversation (never guessed — only ever
# a strictly more informative, compatible name already present in the
# transcript). See validate_doctor_information's use of this below.

def _enrich_candidates_with_fuller_agent_names(
    candidates: list[str], fuller_name_sources: list[str],
) -> list[str]:
    """For each name in *candidates* (a recommendation-set's own entries,
    possibly bare first-name-only), look for a LONGER, name-compatible
    candidate among *fuller_name_sources* (every OTHER name-shaped string
    available as evidence — typically every Agent-turn candidate across
    the whole call, see extract_doctor_turn_candidates, optionally with a
    grounded semantic name appended — see _semantic_name_is_grounded) and,
    when found, REPLACE the shorter entry with the fuller one.

    Compatibility is the SAME strict rule CRM resolution itself already
    uses end-to-end (_is_plausible_same_person_name: an ordered prefix, OR
    agreement at both the given-name and family-name endpoints) — the
    shared normaliser it builds on (_name_tokens -> normalize_arabic_text)
    already treats ه/ة and every alef/yeh variant as equivalent, so a bare
    "اميره" is recognised as compatible with a fuller "أميرة بركات" just as
    readily as with "اميره بركات". A candidate that DISCRIMINATINGLY
    conflicts with a longer name (e.g. a genuinely different surname) is
    never touched by it — this only ever WIDENS an already-compatible
    name, never guesses a replacement.

    Never invents a new candidate and never changes how many candidates
    exist: only replaces an EXISTING entry, only with a compatible name
    that adds EXACTLY ONE token (a plausible single missing name
    component — a surname added to a bare first name, e.g. "اميره" ->
    "اميره بركات" — never an open-ended amount of extra text), and only
    when exactly ONE existing candidate qualifies for a given fuller name.
    The exact "+1 token" bound is deliberate, not merely a simplification:
    extraction is not perfect elsewhere in this module — a name candidate
    can end up with a claim glued onto it with no punctuation boundary
    ("Dr Ahmed Ali بورد سعودي" extracted as one 4-token string when the
    Agent stated a qualification right after re-naming an already-known
    doctor) — and that IS still an ordered-prefix-compatible "longer"
    string, but it must never be mistaken for a genuinely fuller NAME and
    overwrite a clean, correct 2-token candidate ("ahmed ali") with claim
    text. A real missing name component is always exactly one token; a
    claim/specialty/degree phrase accidentally glued onto a name never is
    (see the regression this specific bound closes).

    A fuller name that is compatible with more than one existing
    candidate at once is ambiguous and is left alone, since guessing
    which one it enriches would be exactly the kind of unsafe widening
    this module avoids everywhere else (e.g. the same-first-name-alone
    safety net)."""
    if not candidates or not fuller_name_sources:
        return candidates
    enriched = list(candidates)
    for fuller in fuller_name_sources:
        fuller_tokens = _name_tokens(fuller)
        if len(fuller_tokens) < 2:
            continue  # only a genuinely MORE COMPLETE name can enrich anything
        compatible_indices = [
            i for i, existing in enumerate(enriched)
            if len(_name_tokens(existing)) == len(fuller_tokens) - 1
            and _is_plausible_same_person_name(_name_tokens(existing), fuller_tokens)
        ]
        if len(compatible_indices) == 1:
            enriched[compatible_indices[0]] = fuller
    return enriched


def _semantic_name_is_grounded(semantic_name: str | None, agent_candidates: list[str]) -> bool:
    """True when *semantic_name* (an LLM-produced doctor name) is actually
    compatible with SOME name an Agent turn's own deterministic extraction
    independently found (see extract_doctor_turn_candidates) — the safety
    net a multi-doctor recommendation set needs before trusting an LLM
    name at all: an LLM name with no correspondence to anything the Agent
    was ever transcript-recorded as saying must never be allowed to
    enrich, replace, or otherwise stand in for real deterministic
    evidence (see _enrich_candidates_with_fuller_agent_names, the only
    caller of this check)."""
    if not semantic_name:
        return False
    semantic_tokens = _name_tokens(semantic_name)
    if not semantic_tokens:
        return False
    return any(
        _is_plausible_same_person_name(semantic_tokens, _name_tokens(a))
        for a in agent_candidates
    )


# Degree/title words that sometimes end up glued onto an already-extracted
# name candidate with no punctuation boundary between the claim and the
# name ("خيرية محمد أخصائية أطفال" — the degree claim runs straight into
# the name) — stripped from the END of a query's token sequence ONLY for
# CRM name-MATCHING purposes here in the resolution layer; this never
# changes what extraction itself returns (extract_doctor_turn_candidates'
# output — the accepted candidate string — is untouched). A small, local,
# already-normalised set kept separate from the extraction-side
# _DEGREE_TITLE_WORDS specifically so no extraction behaviour changes.
_TRAILING_DEGREE_WORDS_FOR_MATCHING = {
    "استشاري", "استشاريه", "اخصايي", "اخصاييه", "مختص", "مختصه",
    "consultant", "specialist",
}


def _strip_trailing_degree_words(tokens: list[str]) -> list[str]:
    trimmed = list(tokens)
    while len(trimmed) > 1 and trimmed[-1] in _TRAILING_DEGREE_WORDS_FOR_MATCHING:
        trimmed.pop()
    return trimmed


def resolve_doctor_candidates(
    query_text: str, doctors: list[dict[str, Any]], *, allow_fuzzy: bool = True,
    allow_single_token: bool = False,
) -> list[dict[str, Any]]:
    """Resolve a doctor from free text against an authoritative (already
    active/supported-BU-filtered, deduplicated) pool — cr301_opdflag is
    NOT a filtering condition (see _is_active's docstring).

    Priority:
      1. Exact normalised Arabic name (cr301_doctornamear) — full-string match
         after stripping title prefixes.
      2. Exact normalised English name (servhub_doctornameen).
      3. Safe ORDERED-PREFIX partial match — the shorter of the query's/CRM
         name's token sequences must be an exact, ORDERED prefix of the
         longer one (see _ordered_prefix_match), never merely "some tokens
         overlap somewhere". This is what lets "خيرية محمد" resolve
         "خيرية محمد علي موسى" while correctly REFUSING "اسامة عبد السلام"
         against "اسامة عبد المقصود" — same length, same first two tokens,
         but the final DISCRIMINATING token conflicts, so there is no valid
         prefix relationship in either direction. A single-token query is
         only attempted here when *allow_single_token* is True (see below);
         a single-token CRM name is always reachable when it is itself a
         prefix of a longer query, with no separate special case needed.
      4. High-confidence fuzzy fallback (rapidfuzz `ratio` — the plain
         full-string similarity, not `token_set_ratio`, which is
         deliberately insensitive to a differing/extra token and would
         happily score "اسامة عبد السلام" vs "اسامة عبد المقصود" far too
         high; threshold 90, stricter than location's 80, since a doctor
         misresolution directly drives which factual claims get
         validated). Only a single clear winner accepted. Never relaxed by
         *allow_single_token* — a single token is too easily and too
         plausibly close (by edit-distance alone) to an unrelated short
         name/operational-row label to trust a fuzzy score on it.

    allow_single_token : when False (the default — every existing caller),
        a single-token query can only ever resolve via an EXACT full-string
        match (tier 1/2), exactly as before — "محمد"/"أحمد"/"سارة" alone
        must not partially resolve a doctor, since a bare first name is far
        too common to trust from unvetted text. When True, a single-token
        query MAY also resolve via tier 3 (_ordered_prefix_match against a
        longer CRM name — e.g. "وصال" against CRM "وصال محمد"), because the
        caller has already established, upstream of this function, that
        the token is being used as an actual human doctor's name (see
        app.agent.nodes' semantic doctor-name extraction) — this is never
        turned on for a raw, unvetted text fragment. Tier 4 (fuzzy) is
        deliberately NOT relaxed by this flag, for the reason in the fuzzy
        tier's own docstring above.

    A tie at any stage is returned as multiple candidates — the caller
    decides PASS vs AMBIGUOUS_DOCTOR, never resolves arbitrarily.
    """
    query_norm = _strip_title(query_text)
    if not query_norm or not doctors:
        return []
    query_token_list = _name_tokens(query_norm)
    if not query_token_list:
        return []

    # 1/2 — exact full-string match (either name field).
    exact: list[dict[str, Any]] = []
    for rec in doctors:
        ar = normalize_arabic_text(str(rec.get("cr301_doctornamear") or ""))
        en = normalize_arabic_text(str(rec.get("servhub_doctornameen") or ""))
        if ar and ar == query_norm:
            exact.append(rec)
        elif en and en == query_norm:
            exact.append(rec)
    if exact:
        return exact

    # 3 — safe partial match: ordered prefix OR given+family-name agreement
    # (see _is_plausible_same_person_name's docstring and this function's
    # docstring above). A trailing degree/title word accidentally glued
    # onto the query by noisy chat text (no punctuation boundary between a
    # degree claim and the name) is stripped for MATCHING purposes only —
    # never attempted for a bare single-token query unless allow_single_
    # token is set, since an UNVETTED first name alone is too common to
    # trust.
    query_for_matching = _strip_trailing_degree_words(query_token_list)
    partial: list[dict[str, Any]] = []
    _min_match_len = 1 if allow_single_token else _SINGLE_TOKEN_MATCH_MIN_LEN
    if len(query_for_matching) >= _min_match_len:
        for rec in doctors:
            for field in ("cr301_doctornamear", "servhub_doctornameen"):
                name_token_list = _name_tokens(str(rec.get(field) or ""))
                if name_token_list and _is_plausible_same_person_name(query_for_matching, name_token_list):
                    partial.append(rec)
                    break
    if partial:
        return partial

    if not allow_fuzzy:
        return []

    # 4 — fuzzy fallback, high threshold (90) on the FULL-NAME similarity
    # (rapidfuzz `ratio`, not `token_set_ratio` — see this function's
    # docstring). Only ever applied against multi-token queries — never a
    # bare single token — so a colloquial nickname/typo can still resolve
    # without letting one common word fuzzy-match an unrelated doctor.
    if len(query_token_list) < _SINGLE_TOKEN_MATCH_MIN_LEN:
        return []
    fuzzy_scored: list[tuple[float, dict]] = []
    for rec in doctors:
        for field in ("cr301_doctornamear", "servhub_doctornameen"):
            name = normalize_arabic_text(str(rec.get(field) or ""))
            if not name:
                continue
            score = _rfuzz.ratio(query_norm, name)
            if score >= 90:
                fuzzy_scored.append((score, rec))
                break
    if not fuzzy_scored:
        return []
    best = max(score for score, _ in fuzzy_scored)
    winners = [rec for score, rec in fuzzy_scored if score == best]
    # De-dup winners that are literally the same record referenced twice.
    seen_keys = set()
    unique_winners = []
    for rec in winners:
        k = rec.get("cr301_doctorkey") or id(rec)
        if k not in seen_keys:
            seen_keys.add(k)
            unique_winners.append(rec)
    return unique_winners if len(unique_winners) == 1 else unique_winners


# ── Safe CONTEXTUAL single-name resolution (multi-doctor recommendation
# sets ONLY) ──────────────────────────────────────────────────────────────
# Real regression: an Agent turn offers two doctors bare-first-name-only
# ("متاح الطبيبه بدريه والطبيبه اميره") — "اميره" later gets enriched to a
# full name from a subsequent Agent turn (see _enrich_candidates_with_
# fuller_agent_names), but "بدريه" never does; she has no surname anywhere
# in the call. resolve_doctor_candidates' own allow_single_token flag
# deliberately never trusts a bare first name via ordinary partial/fuzzy
# matching (a common given name is far too likely to collide) — and this
# module must NOT start doing so globally just to resolve one doctor in one
# call (see resolve_doctor_candidates' own docstring, unchanged).
#
# Instead: a bare first name may resolve ONLY when AUTHORITATIVE, NON-FUZZY
# CONTEXT — the doctor being Active + in a supported BU, the call's own BU
# when known, and the specialty/service the Patient actually asked for
# (e.g. "بدي دكتور الغدد" -> Endocrinology) — narrows the CRM pool down to
# EXACTLY one doctor. Zero matches stays DOCTOR_UNRESOLVED; two or more
# stays AMBIGUOUS_DOCTOR — this NEVER picks a "best" fuzzy-scored guess
# among several plausible doctors (see _resolve_single_token_candidate_
# with_context's own "never fuzzy" contract). Used ONLY as a fallback tier
# inside _resolve_and_validate_one_doctor, and ONLY when the caller
# (validate_doctor_information's multi-doctor branch) explicitly opts in
# via allow_contextual_single_name=True — the ordinary single-doctor path
# never sets this, so a bare first name mentioned outside a genuine
# multi-doctor recommendation set is completely unaffected by this.


def _recommendation_set_specialty_context(call: CallTranscript) -> str | None:
    """The specialty/service category (see _resolve_specialty_category —
    already maps "الغدد"/"الغدد الصماء"/"امراض السكر والغدد الصماء" to
    "Endocrinology", among every other specialty this module already
    recognises) the WHOLE multi-doctor recommendation set is framed
    around — the Patient's own original request, checked across every
    Patient turn FIRST (in order), falling back to Agent turns only when
    no Patient turn resolves anything. Used ONLY to narrow a single-token
    candidate's safe contextual resolution (see _resolve_single_token_
    candidate_with_context) — never to override or replace extract_
    doctor_context_specialty's own per-doctor-anchored scoping used
    elsewhere for specialty CLAIM validation.

    Deliberately scans the WHOLE call rather than being anchored to (and
    boundary-limited by — see _turn_is_doctor_context_boundary) any one
    named doctor's own turn: a recommendation set's shared context
    legitimately comes from the Patient's original, standalone service
    request, which can sit BEFORE an intervening rejected/unavailable-
    doctor mention ("بدي دكتور الغدد" ... "غير متاح طبيب بهذا الاسم" ...
    "متاح الطبيبه بدريه والطبيبه اميره" — all in the SAME call). This
    never widens NAME matching by itself — it is only ever one of several
    filters _resolve_single_token_candidate_with_context requires ALL of
    to agree on before a bare first name is allowed to resolve at all."""
    turns = split_transcript_turns(call.transcript)
    for speaker, text in turns:
        if speaker == "patient":
            category = _resolve_specialty_category(text)
            if category:
                return category
    for speaker, text in turns:
        if speaker == "agent" and not is_agent_self_introduction(text):
            category = _resolve_specialty_category(text)
            if category:
                return category
    return None


def _resolve_single_token_candidate_with_context(
    name: str,
    authoritative_pool: list[dict[str, Any]],
    call_bu: str | None,
    specialty_context: str | None,
) -> list[dict[str, Any]]:
    """Safe, CONTEXTUAL resolution for a single-token doctor-name
    candidate — see this section's own module-level comment for the full
    business rationale and safety contract. Filters applied, in order:
      1. name compatibility — the CRM doctor's own given name (first
         token, either cr301_doctornamear or servhub_doctornameen) equals
         *name*, using the SAME ه/ة, alef, and yeh-normalising comparison
         every other name match in this module already uses (_name_
         tokens -> normalize_arabic_text) — never fuzzy/partial.
      2. *authoritative_pool* only — Active + supported-BU (see
         authoritative_doctor_pool); a doctor outside that scope is never
         reachable through this path at all.
      3. the call's own business unit, when known (*call_bu*, already
         canonicalised) — via _match_doctor_business_unit (either of the
         doctor's two CRM BU fields).
      4. *specialty_context* (already resolved via _resolve_specialty_
         category) — matches EITHER the doctor's specialty OR
         subspecialty field (cr301_specialtyname/cr18c_manualspecialtyname
         /cr301_subspecialtyname/cr18c_manualsubspecialtyname), reusing
         _specialty_claim_result — the SAME comparison this module
         already uses for specialty CLAIM validation elsewhere — never a
         bare substring/fuzzy text match of its own.
    Each of steps 3/4 is applied ONLY when its own input (*call_bu*/
    *specialty_context*) is actually available — never silently skipped
    the other way around (an unavailable filter is never treated as
    "matches everything"; it simply isn't one of the filters this
    particular call applies).

    Returns the filtered list AS-IS — 0 (no context-compatible doctor at
    all), exactly 1 (safely resolved), or 2+ (still genuinely ambiguous
    even after every filter). NEVER picks a "best" candidate by fuzzy
    score among 2+ results — the caller (_resolve_and_validate_one_doctor)
    interprets this exactly like any other resolve_doctor_candidates
    result: 0 -> DOCTOR_UNRESOLVED, 1 -> resolved, 2+ -> AMBIGUOUS_DOCTOR
    (after its own existing same-profile/BU-tiebreak checks, unchanged)."""
    name_tokens = _name_tokens(name)
    if len(name_tokens) != 1:
        return []
    if not call_bu and not specialty_context:
        # No authoritative context available AT ALL — a bare first name
        # must never resolve on name-uniqueness-within-the-pool alone
        # (that would silently degrade into exactly the unrestricted
        # first-name matching this mechanism is explicitly required NOT
        # to enable globally, the moment a call happens to have only one
        # same-named doctor in its pool). At least one real contextual
        # signal (call BU or resolved specialty) must be available before
        # this function does any narrowing at all.
        return []
    given = name_tokens[0]

    candidates = [
        rec for rec in authoritative_pool
        if any(
            _name_tokens(str(rec.get(field) or ""))[:1] == [given]
            for field in ("cr301_doctornamear", "servhub_doctornameen")
        )
    ]
    if not candidates:
        return []

    if call_bu:
        candidates = [rec for rec in candidates if _match_doctor_business_unit(call_bu, rec) is not None]
        if not candidates:
            return []

    if specialty_context:
        candidates = [
            rec for rec in candidates
            if _specialty_claim_result(
                specialty_context, specialty_context,
                rec.get("cr301_specialtyname"), rec.get("cr18c_manualspecialtyname"),
                rec.get("cr301_subspecialtyname"), rec.get("cr18c_manualsubspecialtyname"),
            ) is True
        ]

    return candidates


# ── Resolution diagnostics (ambiguous / outside-authoritative-scope) ────────
# Purely observational — never consulted by any resolution decision itself.
# Re-derives WHICH tier (exact/partial/fuzzy) actually matched each
# candidate, since resolve_doctor_candidates() only returns the winning
# records, not how each one won — a per-candidate breakdown is essential to
# tell apart genuine CRM ambiguity (several real, distinct doctors sharing a
# name) from a resolution-layer weakness (a fuzzy/partial match that should
# never have competed with a real one).
_HOME_CARE_TEXT_RE = re.compile(r"منزل|home\s*care|home\s*visit", re.I)


def _mentions_home_care(rec: dict[str, Any]) -> bool:
    """True when this doctor's own documented scope/notes text mentions
    home-care/home-visit service — informational only, surfaced so a human
    reviewing the diagnostic log can see whether CRM evidence actually
    supports a home-care distinction for THIS doctor. Note that OPD is no
    longer part of the authoritative gate at all (see _is_active) — this
    flag exists purely for observability now, not to justify bypassing a
    filter."""
    text = " ".join(
        str(rec.get(k) or "") for k in
        ("cr301_scopeofservice", "cr301_scopeofservicear", "cr301_drnotes")
    )
    return bool(_HOME_CARE_TEXT_RE.search(text))


def _person_name_match_detail(query_tokens: list[str], name_tokens: list[str]) -> dict[str, Any]:
    """Structured, auditable explanation of whether *query_tokens* and
    *name_tokens* (both ordered, normalised — see _name_tokens) are a
    plausible SAME-PERSON match under _is_plausible_same_person_name (an
    ordered prefix, OR agreement at both the given-name and family-name
    endpoints) — never merely "do they share some tokens". Diagnostic-only:
    identifies the matched (agreeing) PREFIX and, when the sequences are
    the same length and disagree, the exact CONFLICTING token pair that
    rules the match out — e.g. requested=['اسامه','عبد','السلام'] vs
    candidate=['اسامه','عبد','المقصود'] reports matched=['اسامه','عبد'],
    conflicting=['السلام','المقصود'], accepted=False, reason=
    'discriminating_name_token_conflict'."""
    if not query_tokens or not name_tokens:
        return {"accepted": False, "matched_tokens": [], "conflicting_tokens": [], "reason": "empty_name"}
    accepted = _is_plausible_same_person_name(query_tokens, name_tokens)
    shorter, longer = (
        (query_tokens, name_tokens) if len(query_tokens) <= len(name_tokens) else (name_tokens, query_tokens)
    )
    matched: list[str] = []
    conflicting: list[str] = []
    for i in range(len(shorter)):
        if shorter[i] == longer[i]:
            matched.append(shorter[i])
        else:
            conflicting = [shorter[i], longer[i]]
            break
    if accepted:
        reason = "ordered_prefix_match" if _ordered_prefix_match(query_tokens, name_tokens) else "given_and_family_name_match"
    elif not matched:
        reason = "no_shared_given_name"
    elif conflicting:
        reason = "discriminating_name_token_conflict"
    else:
        reason = "insufficient_overlap"
    return {"accepted": accepted, "matched_tokens": matched, "conflicting_tokens": conflicting, "reason": reason}


def _describe_candidate_match(query_norm: str, query_token_list: list[str], rec: dict[str, Any]) -> dict[str, Any]:
    """Re-derive, for ONE candidate record, which resolve_doctor_candidates
    tier actually matched it against *query_norm* (already title-stripped
    and normalised) — 'exact' / 'partial' / 'fuzzy' / 'none' — plus the
    fuzzy score when relevant, the matched/conflicting name-token detail
    (see _person_name_match_detail), and why it is/isn't inside the
    authoritative (Active + supported-BU — cr301_opdflag is informational
    only, never a gate; see _is_active) pool. See module docstring above."""
    ar_raw, en_raw = str(rec.get("cr301_doctornamear") or ""), str(rec.get("servhub_doctornameen") or "")
    ar, en = normalize_arabic_text(ar_raw), normalize_arabic_text(en_raw)
    match_method, match_score = "none", None
    name_detail = {"matched_tokens": [], "conflicting_tokens": [], "reason": "no_shared_given_name"}

    if (ar and ar == query_norm) or (en and en == query_norm):
        match_method = "exact"
        name_detail = {"matched_tokens": query_token_list, "conflicting_tokens": [], "reason": "exact_match"}
    else:
        for field_name in (ar, en):
            name_token_list = _name_tokens(field_name)
            if not name_token_list:
                continue
            detail = _person_name_match_detail(query_token_list, name_token_list)
            # Keep the MOST INFORMATIVE detail seen across both fields —
            # an accepted match wins outright; otherwise prefer whichever
            # field shares the longer matched prefix.
            if detail["accepted"]:
                match_method, name_detail = "partial", detail
                break
            if len(detail["matched_tokens"]) > len(name_detail["matched_tokens"]):
                name_detail = detail
        if match_method == "none":
            best = 0.0
            for field_name in (ar, en):
                if field_name:
                    best = max(best, _rfuzz.ratio(query_norm, field_name))
            if best >= 90:
                match_method, match_score = "fuzzy", round(best, 1)

    authoritative = _is_active(rec) and is_supported_doctor_bu(rec.get("cr18c_buname"))
    exclusion_reasons = []
    if not _is_active(rec):
        status = str(rec.get("statuscodename") or "").strip()
        exclusion_reasons.append(f"status={status!r} (not Active)")
    if not is_supported_doctor_bu(rec.get("cr18c_buname")):
        exclusion_reasons.append(f"business_unit={rec.get('cr18c_buname')!r} (not in supported scope)")
    # opd_flag is informational only — never an exclusion reason (see
    # _is_active's docstring) — but still surfaced below for visibility.

    return {
        "doctor_key": rec.get("cr301_doctorkey"),
        "doctor_name_ar": rec.get("cr301_doctornamear"),
        "doctor_name_en": rec.get("servhub_doctornameen"),
        "business_unit": rec.get("cr18c_buname"),
        "cr301_businessunitname": rec.get("cr301_businessunitname"),
        "status": rec.get("statuscodename"),
        "opd_flag": rec.get("cr301_opdflag"),
        "specialty": rec.get("cr301_specialtyname") or rec.get("cr18c_manualspecialtyname"),
        "subspecialty": rec.get("cr301_subspecialtyname") or rec.get("cr18c_manualsubspecialtyname"),
        "match_method": match_method,
        "match_score": match_score,
        "matched_tokens": name_detail["matched_tokens"],
        "conflicting_tokens": name_detail["conflicting_tokens"],
        "authoritative": authoritative,
        "exclusion_reason": "; ".join(exclusion_reasons) or None,
        "scope_mentions_home_care": _mentions_home_care(rec),
    }


def _log_ambiguous_resolution(
    call_id: str, requested_name: str, requested_business_unit: str | None,
    candidates: list[dict[str, Any]],
) -> None:
    """Full per-candidate breakdown for an ambiguous/outside-scope
    resolution — printed instead of the bare 'candidate_count=N' line, so
    a real production ambiguity is immediately diagnosable: genuine
    distinct-doctors-same-name (Case B — legitimate AMBIGUOUS_DOCTOR) vs. a
    resolution-layer weakness where one exact match is being outvoted by
    weaker partial/fuzzy noise (Case A — should have resolved cleanly)."""
    query_norm = _strip_title(requested_name)
    query_token_list = _strip_trailing_degree_words(_name_tokens(query_norm))
    _doc_print()
    _doc_print("ambiguous CRM resolution:")
    _doc_print(f"  requested_name={requested_name}")
    _doc_print(f"  normalized_requested={query_token_list}")
    _doc_print(f"  requested_business_unit={requested_business_unit}")
    _doc_print(f"  candidate_count={len(candidates)}")
    exact_count = 0
    match_methods = []
    for i, rec in enumerate(candidates, start=1):
        info = _describe_candidate_match(query_norm, query_token_list, rec)
        match_methods.append(info["match_method"])
        if info["match_method"] == "exact":
            exact_count += 1
        _doc_print()
        _doc_print(f"  candidate[{i}]:")
        for key, value in info.items():
            _doc_print(f"      {key}={value}")
    if exact_count == 1 and len(candidates) > 1:
        _doc_print()
        _doc_print(
            f"  NOTE: exactly ONE candidate matched via 'exact' name equality; "
            f"the remaining {len(candidates) - 1} matched via a weaker tier — "
            f"see match_method above. Every candidate here already passed the "
            f"ordered-prefix/exact resolver tiers (resolve_doctor_candidates never "
            f"returns a bare token-overlap match), so this reflects genuine "
            f"CRM ambiguity, not resolution-layer noise.",
        )
    logger.info(
        "doctor ambiguous resolution | call_id=%s requested_name=%s candidate_count=%d "
        "exact_matches=%d candidates=%s",
        call_id, requested_name, len(candidates), exact_count,
        [{"doctor_key": c.get("cr301_doctorkey"), "match_method": m} for c, m in zip(candidates, match_methods)],
    )


def _log_near_miss_candidates(call_id: str, requested_name: str, pool: list[dict[str, Any]]) -> None:
    """When a doctor fails to resolve at all, surface any CRM records that
    share the requested name's given name but were correctly REJECTED by
    the ordered-prefix rule (e.g. same given name, conflicting family
    name) — so a human can confirm the rejection was right (the doctor
    genuinely isn't in the data under a matching name) rather than the
    resolver having silently discarded a real near-match with no visible
    reason. Capped at 5 records — a diagnostic aid, not a full data dump.
    Never used to widen resolution itself."""
    query_norm = _strip_title(requested_name)
    query_token_list = _strip_trailing_degree_words(_name_tokens(query_norm))
    if not query_token_list:
        return
    query_first = query_token_list[0]
    shown = 0
    seen_keys: set[Any] = set()
    for rec in pool:
        if shown >= 5:
            break
        key = rec.get("cr301_doctorkey")
        if key in seen_keys:
            continue
        for raw in (rec.get("cr301_doctornamear"), rec.get("servhub_doctornameen")):
            name_norm = normalize_arabic_text(str(raw or ""))
            if not name_norm or name_norm == query_norm:
                continue
            name_token_list = _name_tokens(name_norm)
            if query_first not in name_token_list:
                continue
            detail = _person_name_match_detail(query_token_list, name_token_list)
            if detail["accepted"]:
                continue  # would already have been an accepted candidate
            seen_keys.add(key)
            _doc_print()
            _doc_print("CRM candidate rejected:")
            _doc_print(f"  requested_name={requested_name}")
            _doc_print(f"  candidate_name={raw}")
            _doc_print(f"  normalized_requested={query_token_list}")
            _doc_print(f"  normalized_candidate={name_token_list}")
            _doc_print("  match_method=partial")
            _doc_print("  accepted=False")
            _doc_print(f"  reason={detail['reason']}")
            logger.info(
                "doctor near-miss CRM candidate rejected | call_id=%s requested_name=%s "
                "candidate_key=%s candidate_name=%r reason=%s",
                call_id, requested_name, key, raw, detail["reason"],
            )
            shown += 1
            break


# ── Field-claim extraction (Agent side only — "validate only what was said") ─
# Every extractor below returns None when the Agent made no claim at all for
# that field, so validate_doctor_information() never penalises a field the
# Agent never mentioned.

# Degree: Arabic clinical titles -> canonical CRM degree value. Order
# matters — "senior/first" variants are checked before the bare title they
# contain ("أخصائي أول" before "أخصائي") so the more specific claim wins.
# Patterns are written against normalize_arabic_text()'s OUTPUT spelling
# (hamza folded away, teh-marbuta -> ه) — e.g. "أخصائي" normalises to
# "اخصايي" (no ئ survives), "نائب" to "نايب". Verified directly against
# normalize_arabic_text() rather than guessed, since guessing the folded
# form wrong silently breaks the match with no error.
_DEGREE_CLAIM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"استشاري[ةه]?", re.I), "consultant"),
    # Word-bounded ((?<![\w])/(?![\w])) — real regression: "و استأذنك
    # الحضور..." ("...and kindly note/allow me...", a routine polite
    # phrase) normalises to "...استاذنك..." and was misdetected as
    # "أستاذ" (Professor) purely because the unbounded pattern matched
    # "استاذ" as a mere PREFIX of "استاذنك", not a standalone word —
    # mirrors the same "معانا" ⊃ "انا" fix already applied to
    # _SELF_INTRO_RE.
    (re.compile(r"(?<![\w])است?اذ(?![\w])", re.I), "professor"),
    (re.compile(r"(?:اخصايي|نايب)\s*اول", re.I), "senior registrar"),
    (re.compile(r"اخصايي[ةه]?", re.I), "specialist"),
    (re.compile(r"نايب", re.I), "register"),
    (re.compile(r"طبيب\s*عام|\bgp\b", re.I), "gp"),
    (re.compile(r"مقيم", re.I), "resident"),
]


def _extract_degree_claim(text: str) -> str | None:
    norm = normalize_arabic_text(text)
    for pattern, canon in _DEGREE_CLAIM_PATTERNS:
        if pattern.search(norm):
            return canon
    return None


def _degree_claim_raw_match(text: str) -> str | None:
    """The raw matched substring (from normalize_arabic_text's output)
    behind _extract_degree_claim's canonical bucket — diagnostic-only,
    used for the "[doctor] field validation" log line, never for the
    PASS/FAIL decision itself (that stays exactly _extract_degree_
    claim's own comparison)."""
    norm = normalize_arabic_text(text)
    for pattern, _canon in _DEGREE_CLAIM_PATTERNS:
        m = pattern.search(norm)
        if m:
            return m.group()
    return None


# Display-only casing for the canonical degree bucket in logs — the
# PASS/FAIL comparison itself stays on the lowercase bucket value
# (_canon_degree_value already lowercases the real CRM value it's
# compared against), this exists purely so a log line reads "Consultant"
# rather than "consultant".
_DEGREE_CANON_DISPLAY = {
    "consultant": "Consultant", "professor": "Professor",
    "senior registrar": "Senior Registrar", "specialist": "Specialist",
    "register": "Register", "gp": "GP", "resident": "Resident",
}


def _canon_degree_value(value: str | None) -> str:
    v = str(value or "").strip().lower()
    return "senior registrar" if v == "senior register" else v


# Specialty canonicalisation reuses app.service_hub.offer_search._AR_ALIAS
# — the project's one canonical Arabic/English specialty-alias table
# (verified against real CRM data; now also carries the Oncology entries
# this module needs) — as the PRIMARY source, rather than a second, full
# duplicate of it here. _SPECIALTY_ALIAS_SUPPLEMENT below is NOT a copy of
# that table: it holds only the handful of phrases genuinely absent from
# it that this module's own regression tests rely on (offer_search was
# built for patient-facing OFFER search, not doctor-CRM-claim validation,
# so its vocabulary is close but not 100% identical) — e.g. a bare, no-
# article "غدد صماء" full phrase, which offer_search only has as the
# shorter "غدد" (loses a longest-alias-wins tie against an unrelated
# longer word like "اطفال" in the same sentence), plus Nutrition/Speech,
# which offer_search has no category for at all.
_SPECIALTY_ALIAS_SUPPLEMENT: dict[str, str] = {
    "غدد صماء": "Endocrinology",
    "تغذية": "Nutrition", "تغذيه": "Nutrition",
    "تخاطب": "Speech", "نطق وتخاطب": "Speech", "علاج النطق": "Speech",
}

# Qualifier words stripped when reducing a specialty DISPLAY NAME (either
# side: the resolved EN category from offer_search, or a raw CRM field
# value like "Medical Oncology") to a comparable "core" token — never a
# second alias table, just a generic reduction so CRM's real-world display
# variants for the SAME specialty (e.g. CRM "Oncology" vs "Medical
# Oncology", or offer_search's "General Pediatrics" vs CRM's bare
# "Pediatrics") compare equal instead of failing on wording alone.
_SPECIALTY_CORE_QUALIFIER_WORDS = {"general", "medical", "clinical"}

# Exact-equivalence CRM display-name variants for the SAME "Speech"
# specialty/subspecialty. Real CRM data spells this subspecialty several
# different ways — including "Phonatics", a misspelling of "Phoniatrics"
# that nonetheless is the literal value stored in production — none of
# which share a common qualifier-word prefix/suffix the generic reduction
# above would strip. Deliberately an EXPLICIT, exact-match table keyed on
# the already-reduced core string (never fuzzy string similarity, which
# risks merging genuinely unrelated specialties) — e.g. this must NOT make
# a bare "E.N.T" match "Speech" just because Phonatics is an ENT
# subspecialty; only these specific spellings are equivalent.
_SPECIALTY_CORE_EQUIVALENTS: dict[str, str] = {
    "speech": "speech",
    "speech therapy": "speech",
    "speech and phonetics": "speech",
    "speech language pathology": "speech",
    "phonatics": "speech",  # CRM spelling (misspelling of "Phoniatrics")
    "phonetics": "speech",
    "phoniatrics": "speech",
}


def _specialty_core(display_name: str | None) -> str:
    """Reduce a specialty/subspecialty display name to a comparable
    lowercase 'core' token — see module comment above. Reuses
    normalize_arabic_text (already lowercases, strips punctuation
    including the "E.N.T" vs "E.N.T." dot difference, collapses
    whitespace) rather than a bespoke normaliser. The result is then run
    through _SPECIALTY_CORE_EQUIVALENTS so every CRM spelling of "Speech"
    (including "Phonatics") collapses onto the same comparable value on
    both sides of a claim/CRM comparison."""
    normalized = normalize_arabic_text(display_name)
    if not normalized:
        return ""
    words = [w for w in normalized.split() if w not in _SPECIALTY_CORE_QUALIFIER_WORDS]
    core = " ".join(words) if words else normalized
    return _SPECIALTY_CORE_EQUIVALENTS.get(core, core)


def _resolve_specialty_category(text: str | None) -> str | None:
    """Canonical Arabic/English specialty-phrase -> EN category name, in
    this precedence order:

      1. Exact full specialty name — the WHOLE (normalised) text IS a
         known specialty name, in either language, from the shared
         app.service_hub.specialty_taxonomy table. Checked first via
         canonicalize_specialty_name (an EXACT, never substring, lookup)
         so a literal, unambiguous name (e.g. "Pediatric Cardiology",
         "Pedodontic", "NICU") is never out-ranked by an unrelated longer
         phrase, and — critically — never accidentally collapsed onto a
         shorter, different, related specialty the way a pure substring
         comparison could (see specialty_taxonomy.canonicalize_specialty_
         name's own docstring for why pediatric specialties must stay
         distinct from their adult equivalents).
      2. Exact normalized alias — a whole-string (not substring) match
         against the legacy alias tables (app.service_hub.offer_search.
         _AR_ALIAS + _SPECIALTY_ALIAS_SUPPLEMENT below).
      3. Longest unambiguous phrase — the original longest-alias-wins
         SUBSTRING search, now also searching specialty_taxonomy's own EN/
         AR names (merged in as SPECIALTY_TAXONOMY_ALIASES) so a taxonomy
         entry embedded in a longer sentence is still found, not just a
         bare exact match. A plain "try one table, else try the other"
         would let an unrelated, merely-longer word elsewhere in the same
         phrase (e.g. "اطفال" in "غدد صماء اطفال") win over the correct,
         shorter match from a DIFFERENT table, so every table is always
         merged into ONE search, never tried in sequence.

    Returns None (not "") when nothing matches at all — including an
    ambiguous, contextless word like bare "ليزر", deliberately absent from
    offer_search's own unambiguous alias table, or a genuine service
    phrase with no top-level category of its own at all (e.g. "الحشوات") —
    so callers can distinguish "no coarse category resolved" from "claim
    contradicts CRM", and can still fall back to matching the RAW claim
    text against CRM scope-of-service (see _match_specialty_against_scope)
    even when this function itself returns None."""
    if not text:
        return None
    norm = normalize_arabic_text(text)
    if not norm:
        return None

    exact = canonicalize_specialty_name(norm)
    if exact:
        return exact

    merged_aliases = {**_AR_ALIAS, **_SPECIALTY_ALIAS_SUPPLEMENT}
    for alias, category in merged_aliases.items():
        if normalize_arabic_text(alias) == norm:
            return category

    best_len, best_category = 0, None
    for alias, category in {**merged_aliases, **SPECIALTY_TAXONOMY_ALIASES}.items():
        alias_norm = normalize_arabic_text(alias)
        if alias_norm and alias_norm in norm and len(alias_norm) > best_len:
            best_len, best_category = len(alias_norm), category
    return best_category


# Marker words/phrases that put a specialty CLAIM (or a CRM display value
# not itself found in specialty_taxonomy) in pediatric context — used ONLY
# as a safety signal for _pediatric_flag below, never to resolve a
# specialty category on its own. Deliberately non-exhaustive, same
# philosophy as every other marker/blocklist in this module. Arabic hamza
# variants (أ/إ/آ/ٱ) already collapse onto plain ا inside
# normalize_arabic_text itself (see app.services.text_helpers._normalize_
# arabic), so e.g. "اطفال"/"أطفال" below are genuinely the same normalized
# token — both spelled out here only for readability, never because the
# normaliser treats them differently.
_PEDIATRIC_CONTEXT_MARKERS = {
    "اطفال", "أطفال", "طفل", "طفله", "طفلة", "رضيع", "رضع", "وليد", "مواليد", "مراهقين",
    "pediatric", "paediatric", "pediatrics", "children", "child",
    "infant", "infants", "newborn", "neonatal", "neonate", "adolescent", "adolescents",
}
_PEDIATRIC_CONTEXT_PHRASES = ("حديثي الولاده", "حديثي الولادة")


def _is_pediatric_qualifier_token(word: str) -> bool:
    """True when a single already-normalized token (or that token with a
    leading ال/لل/بال/... morpheme stripped) is itself a pediatric-context
    marker — reuses _strip_leading_al/_strip_contracted_prefix (defined
    above for the doctor-name blocklist) so definite-article/preposition-
    fused forms ("الاطفال", "الأطفال", "للطفل", "للأطفال") are recognised
    as the same marker as their bare form ("اطفال"/"طفل") without listing
    every prefixed variant separately."""
    if not word:
        return False
    variants = {word, _strip_leading_al(word), _strip_contracted_prefix(word)}
    variants |= {_strip_leading_al(_strip_contracted_prefix(word))}
    return bool(variants & _PEDIATRIC_CONTEXT_MARKERS)


def _pediatric_context_match(text: str | None) -> str | None:
    """Token/phrase-boundary-safe pediatric-context match: returns the
    specific normalized token or phrase from *text* that matched a
    pediatric marker (see _PEDIATRIC_CONTEXT_MARKERS/_PEDIATRIC_CONTEXT_
    PHRASES and _is_pediatric_qualifier_token), or None when nothing
    matches. Matching is always whole-TOKEN-set membership (never a raw
    substring search inside an unrelated word) plus the small, closed set
    of multi-word phrases that aren't reducible to one already-covered
    token — this is what keeps e.g. "أطفال" from ever matching inside an
    unrelated longer word, while still recognising the definite-article/
    prefixed forms this module already needs (see
    _is_pediatric_qualifier_token). _mentions_pediatric_context below is
    the boolean-only convenience wrapper over this same check."""
    if not text:
        return None
    norm = normalize_arabic_text(text)
    if not norm:
        return None
    for token in norm.split():
        if _is_pediatric_qualifier_token(token):
            return token
    for phrase in _PEDIATRIC_CONTEXT_PHRASES:
        if phrase in norm:
            return phrase
    return None


def _mentions_pediatric_context(text: str | None) -> bool:
    """Best-effort check: does *text* mention a pediatric/infant/neonatal
    marker anywhere in it? Used to recognise a genuinely pediatric claim
    even when it was resolved to a non-pediatric-flagged category (e.g.
    "غدد صماء أطفال" resolves to plain "Endocrinology" — see
    _resolve_specialty_category — but the raw claim text still says
    "أطفال"), so _pediatric_flag doesn't have to rely on the resolved
    category alone. See _pediatric_context_match for the underlying,
    phrase-identifying check this is a thin boolean wrapper over."""
    return _pediatric_context_match(text) is not None


def _split_compound_specialty_claim(text: str | None) -> tuple[str, list[str]]:
    """Split a possibly-COMPOUND specialty claim (e.g. "عظام اطفال" =
    base specialty "عظام"/Orthopedics + a pediatric QUALIFIER "اطفال")
    into (base_text_with_qualifier_tokens_removed, detected_qualifier_
    labels). Token-boundary safe — reuses the exact same whole-token
    pediatric-marker check as _pediatric_context_match/_mentions_
    pediatric_context (see _is_pediatric_qualifier_token), never an
    unsafe partial substring.

    This is the fix for the root-cause bug this function exists to close:
    _resolve_specialty_category's own longest-alias-SUBSTRING search has
    no notion of "compound phrase" at all — it just picks whichever known
    alias is the most CHARACTERS long anywhere in the claim text. For
    "عظام اطفال" that accidentally picks the pediatric qualifier "اطفال"
    (5 chars) over the real base specialty "عظام" (4 chars), wrongly
    resolving the whole claim to "Pediatrics" instead of "Orthopedics".
    Stripping the qualifier token(s) out BEFORE resolution removes the
    competing token entirely, so re-resolving the remainder recovers the
    genuine base specialty (see _resolve_and_validate_one_doctor, which
    re-resolves this function's base_text and — only when that succeeds —
    uses it in place of the naive whole-phrase resolution).

    Returns ("", []) for empty input, and (text_unchanged, []) whenever no
    pediatric marker was found at all — a plain, non-compound claim (e.g.
    bare "عظام") is returned completely unchanged, and a pediatric-only
    claim with no base concept left after stripping (e.g. bare "اطفال")
    reports qualifiers but an empty base_text, which callers must treat as
    "not a genuine base+qualifier compound" (there is nothing to validate
    a base specialty against) rather than silently resolving an empty
    string."""
    norm = normalize_arabic_text(text)
    if not norm:
        return "", []
    tokens = norm.split()
    kept = [t for t in tokens if not _is_pediatric_qualifier_token(t)]
    qualifiers = ["pediatric"] if len(kept) != len(tokens) else []
    return " ".join(kept), qualifiers


def _pediatric_flag(canonical_name: str | None, *raw_texts: str | None) -> bool:
    """Is this side of a specialty comparison pediatric? True when the
    canonical taxonomy name itself is pediatric-flagged (is_pediatric_
    specialty), OR when any of *raw_texts* (the raw claim text / raw CRM
    display value, whichever is relevant) mentions a pediatric marker —
    see _mentions_pediatric_context. Combining both signals lets a claim
    like "غدد صماء أطفال" (whose RESOLVED category, "Endocrinology", isn't
    itself a pediatric taxonomy entry) still be recognised as pediatric
    from its own wording."""
    if canonical_name and is_pediatric_specialty(canonical_name):
        return True
    return any(_mentions_pediatric_context(t) for t in raw_texts)


def _specialty_values_match(
    claim_category: str | None, raw_claim_text: str | None, crm_value: str | None,
) -> bool | None:
    """Does ONE CRM specialty/subspecialty value match the claim? Returns
    None when there is nothing to compare (empty CRM value, or no claim
    signal at all) — never a silent False just because data is missing.

    1. Pediatric guard: if exactly one side is pediatric (see
       _pediatric_flag — the canonical taxonomy flag OR a raw pediatric
       marker word) and the other is not, this is an automatic mismatch —
       "Pediatric Cardiology" must never match plain "Cardiology" and
       vice-versa, no matter how the rest of the comparison below would
       otherwise treat them.
    2. Otherwise, compare the CANONICAL EN names when the shared taxonomy
       resolved BOTH sides (canonicalize_specialty_name) — this is what
       correctly bridges e.g. an English "Pedodontic" claim against an
       Arabic CRM value "طب أسنان الأطفال": their raw text shares no
       characters at all, only their canonical name does. Falls back to
       the raw resolved category / raw CRM value when the taxonomy didn't
       resolve one side, so specialties outside the new taxonomy (Speech/
       Phonatics, GIT, Oncology's Medical-Oncology spelling, ...) keep
       comparing exactly as before.
    3. The actual comparison is _specialty_core's qualifier-stripped
       CORE-TOKEN SUBSTRING match, checked in both directions — never
       fuzzy string similarity — so a more verbose name on either side
       (CRM's "Medical Oncology" vs claim "Oncology", taxonomy-bridged
       "Pedodontic" vs "Pedodontic") still safely matches once step 1 has
       already ruled out a pediatric/adult mismatch.
    """
    if not crm_value:
        return None
    if not claim_category and not raw_claim_text:
        return None

    claim_canonical = canonicalize_specialty_name(claim_category) if claim_category else None
    crm_canonical = canonicalize_specialty_name(crm_value)

    claim_pediatric = _pediatric_flag(claim_canonical, raw_claim_text, claim_category)
    crm_pediatric = _pediatric_flag(crm_canonical, crm_value)
    if claim_pediatric != crm_pediatric:
        return False

    left = claim_canonical or claim_category
    right = crm_canonical or crm_value
    claim_core = _specialty_core(left)
    if not claim_core:
        return None
    crm_core = _specialty_core(right)
    if not crm_core:
        return None
    return claim_core in crm_core or crm_core in claim_core


def _specialty_claim_result(
    claim_category: str | None, raw_claim_text: str | None, *crm_values: str | None,
) -> bool | None:
    """True/False once at least one non-empty CRM value was actually
    compared (see _specialty_values_match for the per-value comparison);
    None when every one of *crm_values* is empty (no evidence to confirm
    OR contradict the claim — see _resolve_and_validate_one_doctor's
    NEEDS_REVIEW handling, never a silent FAIL just because CRM lacks
    data)."""
    results = [_specialty_values_match(claim_category, raw_claim_text, v) for v in crm_values if v]
    if not results:
        return None
    return any(r is True for r in results)


# ── Scope-of-service fallback (Step 3 of specialty/subspecialty matching) ──
# Words that must NEVER, on their own, count as a scope-of-service match —
# they are common enough in almost any clinical scope text that a bare
# single-word "match" would be meaningless (Part: "Do not pass solely
# because both strings contain generic medical vocabulary"). A MULTI-word
# candidate phrase containing one of these is still fine (e.g. "علاج
# الحشوات" — only a candidate phrase that reduces to JUST one of these
# words, alone, is rejected). Deliberately non-exhaustive, same philosophy
# as every other marker list in this module.
_SCOPE_FALLBACK_GENERIC_WORDS = {
    "طب", "طبيب", "طبيبه", "جراحه", "جراحة", "اطفال", "أطفال", "علاج",
    "عياده", "عيادة", "خدمات", "خدمه", "خدمة", "مرض", "امراض", "أمراض",
    "عام", "عامه", "عامة", "قسم",
    "general", "medicine", "surgery", "treatment", "clinic", "services",
    "care", "department",
}


def _is_generic_scope_phrase(normalized_phrase: str) -> bool:
    """True when *normalized_phrase* (already normalize_arabic_text'd)
    reduces to NOTHING but generic words (see _SCOPE_FALLBACK_GENERIC_
    WORDS) — including the single-word case explicitly named in the
    requirements ("طب", "جراحة", "أطفال", "علاج" must never match by
    themselves). A genuinely short but SPECIFIC phrase (e.g. "الحشوات")
    is unaffected."""
    words = normalized_phrase.split()
    return bool(words) and all(w in _SCOPE_FALLBACK_GENERIC_WORDS for w in words)


def _scope_fallback_candidate_phrases(raw_claim_text: str | None, specialty_claim: str | None) -> list[str]:
    """Every phrase worth searching for in CRM scope-of-service text, most
    specific first: the raw Agent claim text itself (so a service phrase
    with no top-level category at all, e.g. "الحشوات", is still checked),
    then — when a coarse category WAS resolved — that category's own EN
    name and its taxonomy-mapped Arabic equivalent (so a claim that
    resolved to e.g. "Endocrinology" can also be confirmed via CRM scope
    text written in Arabic). Order only matters for the reported
    matched_phrase/evidence; a match on ANY of these is equally valid."""
    phrases: list[str] = []
    if raw_claim_text:
        phrases.append(raw_claim_text)
    if specialty_claim:
        phrases.append(specialty_claim)
        ar_equivalent = SPECIALTY_EN_TO_AR.get(specialty_claim)
        if ar_equivalent:
            phrases.append(ar_equivalent)
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase and phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)
    return unique


def _match_specialty_against_scope(
    raw_claim_text: str | None, specialty_claim: str | None, doctor: dict[str, Any],
) -> dict[str, Any]:
    """Step 3 of specialty/subspecialty validation (see _resolve_and_
    validate_one_doctor): ONLY tried after specialty/subspecialty
    comparison found no match — searches the resolved doctor's own CRM
    scope-of-service text (cr301_scopeofservicear first, then
    cr301_scopeofservice — see module Required-lookup-order) for evidence
    of the claim, so a genuine service phrase that isn't itself a top-level
    specialty category (e.g. "الحشوات" — dental fillings) can still be
    confirmed against a doctor's own DETAILED scope text.

    Considers, in this order, the raw Agent claim text, the resolved
    category's own EN name, and its taxonomy-mapped Arabic equivalent (see
    _scope_fallback_candidate_phrases) — never a bare word-overlap check
    (that risks a false PASS from shared generic vocabulary alone): a
    candidate phrase must appear as a genuine NORMALIZED SUBSTRING of the
    scope text, and a candidate that reduces to nothing but a generic word
    (_is_generic_scope_phrase — "طب"/"جراحة"/"أطفال"/"علاج" alone, ...) is
    never accepted, matching or not.

    Returns {"matched": bool, "source": field name or None, "matched_
    phrase": the phrase that matched or None, "reference": the raw scope
    text it matched in, or None}."""
    candidate_phrases = _scope_fallback_candidate_phrases(raw_claim_text, specialty_claim)
    for field_name in ("cr301_scopeofservicear", "cr301_scopeofservice"):
        scope_text = doctor.get(field_name)
        norm_scope = normalize_arabic_text(scope_text)
        if not norm_scope:
            continue
        for phrase in candidate_phrases:
            norm_phrase = normalize_arabic_text(phrase)
            if not norm_phrase or _is_generic_scope_phrase(norm_phrase):
                continue
            if norm_phrase in norm_scope:
                return {
                    "matched": True, "source": field_name,
                    "matched_phrase": phrase, "reference": scope_text,
                }
    return {"matched": False, "source": None, "matched_phrase": None, "reference": None}


def _match_pediatric_qualifier_against_scope(doctor: dict[str, Any]) -> dict[str, Any]:
    """Does the resolved doctor's own CRM scope-of-service text carry
    clear evidence that they treat pediatric patients? Used ONLY as the
    second half of a compound base+qualifier specialty claim (e.g. "عظام
    اطفال" — see _split_compound_specialty_claim/_resolve_and_validate_
    one_doctor) — AFTER the base specialty ("Orthopedics") has already
    been confirmed against the doctor's own specialty/subspecialty
    fields; this function never runs on its own and never substitutes for
    that base-specialty check.

    Checked in the same field order as the existing scope fallback
    (_match_specialty_against_scope): cr301_scopeofservicear first, then
    cr301_scopeofservice. Matching is the token/phrase-boundary-safe
    _pediatric_context_match — never a raw substring search — so a
    pediatric marker is never picked up as a partial match inside an
    unrelated word.

    Returns {"matched": bool, "source": field name or None, "phrase": the
    matched pediatric marker/phrase, or None}."""
    for field_name in ("cr301_scopeofservicear", "cr301_scopeofservice"):
        phrase = _pediatric_context_match(doctor.get(field_name))
        if phrase:
            return {"matched": True, "source": field_name, "phrase": phrase}
    return {"matched": False, "source": None, "phrase": None}


# Examination age: the CRM field is semi-structured free text (see
# doctor_validation.py module docstring / Part C examples) — this parser
# handles the common "Age: N-M years" / "N years and above" / "Neonatal and
# above" shapes but deliberately does NOT try to be exhaustive. Returns
# None (not "no restriction") when the text can't be confidently parsed, so
# callers can distinguish "no age claim to check" from "claim exists but we
# can't confirm it" (Part: prefer a non-confirmed outcome over inventing a
# rule).
_AGE_RANGE_RE = re.compile(r"(\d{1,3})\s*[-–]\s*(\d{1,3})\s*(?:years|year|سنه|سنة|سنوات)?", re.I)
_AGE_AND_ABOVE_RE = re.compile(r"(\d{1,3})\s*(?:years|year|سنه|سنة|سنوات)?\s*(?:and\s*above|فأكثر|واكثر|وأكثر)", re.I)
_AGE_NEONATAL_RE = re.compile(r"neonatal", re.I)


def parse_examination_age(raw: str | None) -> tuple[int, float] | None:
    """Best-effort (min_age, max_age) in years from servhub_examinationage's
    semi-structured text. max_age=inf for "and above"/open-ended patterns.
    Returns None when no recognisable range is found (multi-BU strings like
    "AKW: 0 and above // AHJ: Neonatal and 17" are read left-to-right — the
    FIRST recognisable range found is used, since we don't reliably know
    which BU segment applies without more context)."""
    if not raw:
        return None
    text = str(raw)
    if _AGE_NEONATAL_RE.search(text):
        m = re.search(r"neonatal\s*(?:and|to|-)?\s*(\d{1,3})?", text, re.I)
        upper = float(m.group(1)) if m and m.group(1) else float("inf")
        return (0, upper)
    m = _AGE_RANGE_RE.search(text)
    if m:
        return (int(m.group(1)), float(m.group(2)))
    m = _AGE_AND_ABOVE_RE.search(text)
    if m:
        return (int(m.group(1)), float("inf"))
    return None


# Adult-only age threshold used ONLY by _pediatric_scope_contradicted below
# — a doctor whose own structured CRM age-eligibility data starts at this
# age or later is explicit, structured evidence they do NOT see children,
# not a guess from freeform scope wording.
_PEDIATRIC_ADULT_ONLY_AGE_THRESHOLD = 18


def _pediatric_scope_contradicted(doctor: dict[str, Any]) -> bool:
    """True when the resolved doctor's own CRM data EXPLICITLY excludes
    children — used only to decide whether an unconfirmed pediatric
    qualifier (see _match_pediatric_qualifier_against_scope) should read
    as NEEDS_REVIEW (no evidence either way) or a genuine FAIL (evidence
    of exclusion). Deliberately conservative: relies on the already-
    structured, already-parsed servhub_examinationage range (parse_
    examination_age, the same field/parser examination_age validation
    itself uses) rather than guessing at negation phrases in freeform
    scope text, which risks a false FAIL from ordinary adult-specialty
    wording that never actually said children are excluded."""
    age_range = parse_examination_age(doctor.get("servhub_examinationage"))
    return bool(age_range and age_range[0] >= _PEDIATRIC_ADULT_ONLY_AGE_THRESHOLD)


# Agent age-eligibility claims: "يستقبل من عمر N" / "يستقبل أطفال" /
# "لا يستقبل أقل من N" / "من عمر N فأكثر" / an explicit SPECIFIC range
# ("من 5 لحد 15 سنة"). Checked in this order — RANGE before FROM — since
# a range claim never contains the word "عمر" FROM_RE requires, so the two
# patterns never compete for the same text; ordering here is purely
# documentation of that intent, not a tie-break.
_AGE_CLAIM_RANGE_RE = re.compile(r"من\s*(\d{1,3})\s*(?:الي|الى|إلى|لحد|حتى)\s*(\d{1,3})", re.I)
_AGE_CLAIM_FROM_RE = re.compile(r"من\s*عمر\s*(\d{1,3})", re.I)
_AGE_CLAIM_NOT_UNDER_RE = re.compile(r"لا\s*يستقبل\s*اقل\s*من\s*(\d{1,3})|لا\s*يستقبل\s*تحت\s*(\d{1,3})", re.I)
_AGE_CLAIM_CHILDREN_RE = re.compile(r"يستقبل\s*اطفال|يستقبل\s*أطفال", re.I)
_AGE_CLAIM_ADULTS_ONLY_RE = re.compile(r"يستقبل\s*(?:كبار|بالغين)\s*فقط|فوق\s*ال?18", re.I)

# Ordered (pattern, tag-builder) pairs — the single source of truth for
# BOTH _extract_examination_age_claim (the canonical tag) and
# _examination_age_claim_raw_match (the raw matched text for logging), so
# the two can never drift out of sync with each other.
_AGE_CLAIM_PATTERNS: list[tuple[re.Pattern, "Callable[[re.Match], str]"]] = [
    (_AGE_CLAIM_RANGE_RE, lambda m: f"range:{m.group(1)}-{m.group(2)}"),
    (_AGE_CLAIM_FROM_RE, lambda m: f"from:{m.group(1)}"),
    (_AGE_CLAIM_NOT_UNDER_RE, lambda m: f"from:{m.group(1) or m.group(2)}"),
    (_AGE_CLAIM_CHILDREN_RE, lambda _m: "children"),
    (_AGE_CLAIM_ADULTS_ONLY_RE, lambda _m: "adults_only"),
]


def _extract_examination_age_claim(text: str) -> str | None:
    """Returns a normalised claim tag ('children' | 'adults_only' |
    'from:N' | 'range:N-M') or None when the Agent made no age-eligibility
    claim at all — see _AGE_CLAIM_PATTERNS for the recognised shapes."""
    norm = normalize_arabic_text(text)
    for pattern, tag in _AGE_CLAIM_PATTERNS:
        m = pattern.search(norm)
        if m:
            return tag(m)
    return None


def _examination_age_claim_raw_match(text: str) -> str | None:
    """The raw matched substring behind _extract_examination_age_claim's
    canonical tag — diagnostic-only, mirrors _degree_claim_raw_match."""
    norm = normalize_arabic_text(text)
    for pattern, _tag in _AGE_CLAIM_PATTERNS:
        m = pattern.search(norm)
        if m:
            return m.group()
    return None


# Walk-in consultation fee claims: "كشفية الدكتور 300 ريال" — a fee-context
# keyword (كشفية/كشف/consultation) MUST be present near the number, in
# EITHER order. A bare "<number> ريال" with no such keyword anywhere near
# it is deliberately never matched: real regression — an unrelated offer
# price, a procedure cost, or a booking deposit mentioned in the same
# scoped text ("عندنا عرض على الأشعة ب 500 ريال") was previously captured
# as if it were the doctor's own walk-in fee, purely because it was some
# number followed by "ريال" somewhere in scope.
_FEE_CLAIM_RE = re.compile(
    r"(?:كشفي[ةه]|consultation|كشف)\D{0,20}([0-9٠-٩]{2,5})"
    r"|([0-9٠-٩]{2,5})\D{0,10}(?:كشفي[ةه]|consultation\s*fee)",
    re.I,
)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Discount/percentage context — a number found near ANY of these must
# NEVER be treated as the doctor's own standard walk-in fee:
#   - a discount RATE ("خصم 50%"/"50 في الميه") is not a monetary amount
#     at all, it's a percentage;
#   - a discounted PAYABLE amount ("بعد الخصم 113 ريال"/"في حدود 113 ريال")
#     is the price AFTER an offer/promotion is applied, never the doctor's
#     own standard CRM fee (cr301_walkinconsultationfees) — comparing the
#     two directly is meaningless without the authoritative discount
#     percentage/rule itself, which belongs to the offer/discount
#     evaluator (real CRM promotion data), never this deterministic
#     field-by-field doctor-information check.
# Real regression: "خصم 50% على الكشفية" was captured as claimed
# walkin_fee=50 and FAILed against the CRM fee (300) — "%" is stripped by
# normalize_arabic_text's own punctuation removal before _FEE_CLAIM_RE
# ever runs, so a percentage number is otherwise indistinguishable from a
# genuine fee amount purely from the normalized text.
_FEE_DISCOUNT_CONTEXT_RE = re.compile(
    r"%|percent|في\s*ال?ميه|في\s*ال?مائه|في\s*ال?مائة|بال?ميه|بال?مية"
    r"|خصم|discount|في\s*حدود|تقريبا|تقريباً|حوالي",
    re.I,
)


def _discount_tainted_fee_numbers(raw_text: str) -> set[str]:
    """Every digit-string in *raw_text* that appears near an explicit
    percentage/discount indicator (see _FEE_DISCOUNT_CONTEXT_RE) — checked
    on a DIGIT-ONLY-normalised view of the ORIGINAL text (Arabic-Indic
    digits converted to Western form; % and every Arabic percentage/
    discount WORD deliberately left untouched, unlike normalize_arabic_
    text's own aggressive punctuation stripping, which erases % — the
    exact signal this needs — before _FEE_CLAIM_RE ever sees the text).
    Matched by VALUE, not by character position, against the (separately,
    fully) normalized text _extract_fee_claim/_fee_claim_raw_match
    actually search — the two normalisations diverge in string length
    (diacritic/whitespace collapsing), so position alignment between them
    is unreliable; a digit-string is specific enough on its own within one
    doctor's scoped claim text that a value-based match is both simpler
    and safe here."""
    if not raw_text:
        return set()
    local = raw_text.translate(_ARABIC_DIGITS)
    tainted: set[str] = set()
    for m in _FEE_DISCOUNT_CONTEXT_RE.finditer(local):
        window = local[max(0, m.start() - 20):m.end() + 20]
        tainted.update(re.findall(r"[0-9]{1,5}", window))
    return tainted


def _first_untainted_fee_match(norm_text: str, tainted_numbers: set[str]) -> re.Match | None:
    """The first _FEE_CLAIM_RE match in *norm_text* whose number is NOT in
    *tainted_numbers* — never merely the first match overall, so a
    discount rate/discounted amount mentioned before (or after) a genuine
    fee statement in the same scoped text never suppresses the real one,
    and a text with ONLY a discount-tainted number never falls back to
    reporting it as a fee claim."""
    for m in _FEE_CLAIM_RE.finditer(norm_text):
        raw = (m.group(1) or m.group(2)).translate(_ARABIC_DIGITS)
        if raw not in tainted_numbers:
            return m
    return None


def _extract_fee_claim(text: str) -> float | None:
    norm = normalize_arabic_text(text)
    m = _first_untainted_fee_match(norm, _discount_tainted_fee_numbers(text or ""))
    if not m:
        return None
    raw = (m.group(1) or m.group(2)).translate(_ARABIC_DIGITS)
    try:
        return float(raw)
    except ValueError:
        return None


def _fee_claim_raw_match(text: str) -> str | None:
    """The raw matched substring behind _extract_fee_claim's parsed float —
    diagnostic-only, mirrors _degree_claim_raw_match."""
    norm = normalize_arabic_text(text)
    m = _first_untainted_fee_match(norm, _discount_tainted_fee_numbers(text or ""))
    return m.group() if m else None


# Scope-of-service / qualifications / doctor-notes claims: these are open
# free text, so — rather than a fixed keyword table — a claim is any Agent
# sentence introduced by one of these trigger phrases, and it's validated
# by MEANINGFUL Arabic word overlap against the CRM field (same ubiquity-
# free "does the claim's own content appear in the authoritative text"
# idea used elsewhere in this project, just without ubiquity filtering
# since there's no comparable pool of "other doctors" to filter against
# here — ubiquity filtering only makes sense when many candidate records
# share a vocabulary; a single resolved doctor's own reference text doesn't).
_SCOPE_CLAIM_TRIGGER_RE = re.compile(r"يعالج|يعمل|متخصص\s*في|يجري|يقوم\s*ب", re.I)
_QUALIFICATION_CLAIM_TRIGGER_RE = re.compile(r"بورد|زمالة|زماله|دكتوراه|خبرة|خبره|ماجستير|دبلوم", re.I)
_NOTES_CLAIM_TRIGGER_RE = re.compile(
    r"لا\s*يستقبل|لا\s*يوجد(?!\s*(?:مواعيد|معاد|موعد|حجز|مكان|متاح))|فقط\s*يوم|"
    r"حالات\s*جديده|new\s*cases\s*only",
    re.I,
)


def _claim_text_if_triggered(text: str, trigger_re: re.Pattern) -> str | None:
    norm = normalize_arabic_text(text)
    return norm if trigger_re.search(norm) else None


def _meaningful_tokens(text: str) -> set[str]:
    return {t for t in normalize_arabic_text(text or "").split() if len(t) > 2}


def _overlap_supports_claim(claim_text: str, *crm_fields: str | None) -> bool:
    claim_tokens = _meaningful_tokens(claim_text)
    crm_tokens = _meaningful_tokens(" ".join(str(f or "") for f in crm_fields))
    if not claim_tokens or not crm_tokens:
        return False
    return bool(claim_tokens & crm_tokens)


def _summarize_doctor(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "doctor_key": rec.get("cr301_doctorkey"),
        "doctor_name_ar": rec.get("cr301_doctornamear"),
        "doctor_name_en": rec.get("servhub_doctornameen"),
        "business_unit": rec.get("cr18c_buname"),
        "status": rec.get("statuscodename"),
        "opd_flag": rec.get("cr301_opdflag"),
        "degree": rec.get("cr301_degreename"),
        "specialty": rec.get("cr301_specialtyname"),
        "subspecialty": rec.get("cr301_subspecialtyname"),
        # Manual specialty/subspecialty overrides — supporting context only,
        # same evidence tier as specialty/subspecialty (see
        # has_detailed_scope_evidence() and qa_prompt.build_doctor_scope_prompt's
        # evidence hierarchy: scope > doctor_notes > subspecialty > specialty).
        "manual_specialty": rec.get("cr18c_manualspecialtyname"),
        "manual_subspecialty": rec.get("cr18c_manualsubspecialtyname"),
        "scope_of_service": rec.get("cr301_scopeofservice"),
        "scope_of_service_ar": rec.get("cr301_scopeofservicear"),
        "doctor_notes": rec.get("cr301_drnotes"),
        "examination_age": rec.get("servhub_examinationage"),
        # Context only — NEVER proof that a doctor handles a given
        # complaint (a consultant's qualifications say nothing about which
        # specific conditions they book for).
        "qualifications": rec.get("cr301_qualificationsandexperience"),
        "qualifications_ar": rec.get("cr301_qualificationsandexperiencear"),
    }


def has_detailed_scope_evidence(scope_ref: dict[str, Any] | None) -> bool:
    """True only when the doctor has actual documented scope-of-service
    TEXT (cr301_scopeofservice/_ar — the strongest evidence tier), as
    opposed to just a bare specialty/subspecialty category label. Two
    doctors can share the same specialty ("Orthopedics") while handling
    completely different conditions, so a specialty/subspecialty label
    alone is never enough to confidently confirm fit — this flag lets both
    the LLM prompt (qa_prompt.build_doctor_scope_prompt) and a deterministic
    safety net (app.agent.nodes.infer_doctor_scope_validation) recognise
    when they're working from that weaker evidence tier only."""
    if not scope_ref:
        return False
    return bool((scope_ref.get("scope_of_service") or "").strip() or (scope_ref.get("scope_of_service_ar") or "").strip())


def _field(claimed: Any, outcome: str, reference: Any = None, **evidence: Any) -> dict[str, Any]:
    """Build one validated-field evidence dict. The three positional keys
    (claimed/outcome/reference) are the original, load-bearing shape every
    existing caller/consumer relies on — always present, in this order.
    **evidence adds extra, field-specific evidence keys (e.g. business_unit's
    matched_crm_field/crm_business_units) backward-compatibly: additive
    only, never renaming or removing the three keys above."""
    return {"claimed": claimed, "outcome": outcome, "reference": reference, **evidence}


def _log_field_validation(
    field: str, *, raw_claim: Any, canonical_claim: Any,
    crm_reference: Any, canonical_crm: Any, outcome: str,
) -> None:
    """Uniform per-field validation diagnostic, printed for every field
    that was ACTUALLY validated (never for an optional field the Agent
    never mentioned — see _log_skipped_optional_fields for that side).
    Generalises the rich diagnostic block degree validation already used
    to every validated field, so a production PASS/FAIL/NEEDS_REVIEW is
    never an unexplained verdict for ANY field, not just degree/specialty."""
    print(
        f"[doctor] field validation:\n"
        f"[doctor]     field={field!r}\n"
        f"[doctor]     claimed_raw={raw_claim!r}\n"
        f"[doctor]     claimed_canonical={canonical_claim!r}\n"
        f"[doctor]     crm_reference={crm_reference!r}\n"
        f"[doctor]     crm_canonical={canonical_crm!r}\n"
        f"[doctor]     result={outcome}",
        flush=True,
    )


# The Agent-claim-driven fields (Part: "Optional fields") — validated ONLY
# when the Agent explicitly makes a claim about the resolved doctor, never
# added to validated_fields (and so never counted toward fields_checked,
# never able to lower the overall outcome) when absent. Kept as a single
# named tuple so the "which fields are optional" list and the skipped-
# fields diagnostic below can never silently drift apart.
_OPTIONAL_CLAIM_FIELDS = ("degree", "doctor_notes", "qualifications", "examination_age", "walkin_fee")


# ── Structured failure_details (Part: doctor-validation results/logging/
# persistence/UI presentation) ───────────────────────────────────────────
# Human-readable label/reason text for every field that can produce a
# validated_fields[...]["outcome"] == "FAIL" entry — used only to build
# the UI/persistence-facing failure_details list below, never consulted
# by the PASS/FAIL decision itself (that decision is already final by the
# time _build_field_failure_detail runs).
_FIELD_FAILURE_LABELS: dict[str, str] = {
    "degree": "Degree/title mismatch",
    "specialty": "Specialty mismatch",
    "subspecialty": "Subspecialty mismatch",
    "business_unit": "Business unit/branch mismatch",
    "doctor_notes": "Doctor notes mismatch",
    "qualifications": "Qualifications mismatch",
    "examination_age": "Examination age mismatch",
    "walkin_fee": "Consultation fee mismatch",
    "scope_of_service": "Scope of service mismatch",
}
_FIELD_FAILURE_REASONS: dict[str, str] = {
    "degree": "The professional degree stated by the agent does not match the authoritative CRM record.",
    "specialty": "The specialty stated by the agent does not match the authoritative CRM record.",
    "subspecialty": "The subspecialty stated by the agent does not match the authoritative CRM record.",
    "business_unit": "The business unit/branch stated by the agent does not match the authoritative CRM record.",
    "doctor_notes": "The doctor notes stated by the agent do not match the authoritative CRM record.",
    "qualifications": "The qualifications stated by the agent do not match the authoritative CRM record.",
    "examination_age": "The examination age stated by the agent does not match the authoritative CRM record.",
    "walkin_fee": "The consultation fee stated by the agent does not match the authoritative CRM record.",
    "scope_of_service": "The scope of service stated by the agent does not match the authoritative CRM record.",
}


def _field_failure_predicate(field: str) -> Any:
    """A text->bool check, reusing the SAME extractor/trigger each field's
    own validation block above already used, applied to ONE turn's text
    in isolation — used only to locate a local transcript_excerpt for a
    FAILED field (see _field_transcript_excerpt), never to re-decide the
    field's own PASS/FAIL outcome (already decided, against this doctor's
    full scoped text, by the time this runs)."""
    if field == "degree":
        return lambda t: _extract_degree_claim(t) is not None
    if field == "examination_age":
        return lambda t: _extract_examination_age_claim(t) is not None
    if field == "walkin_fee":
        return lambda t: _extract_fee_claim(t) is not None
    if field == "business_unit":
        return lambda t: resolve_business_unit(t) is not None
    if field in ("specialty", "subspecialty"):
        return lambda t: _resolve_specialty_category(t) is not None
    if field == "doctor_notes":
        return lambda t: bool(_NOTES_CLAIM_TRIGGER_RE.search(normalize_arabic_text(t)))
    if field == "scope_of_service":
        return lambda t: (
            bool(_SCOPE_CLAIM_TRIGGER_RE.search(normalize_arabic_text(t)))
            or _resolve_specialty_category(t) is not None
        )
    if field == "qualifications":
        return lambda t: bool(_QUALIFICATION_CLAIM_TRIGGER_RE.search(normalize_arabic_text(t)))
    return lambda t: False


def _raw_claim_for_field(field: str, entry: dict[str, Any], doctor_scope_text: str) -> Any:
    """Best-effort ORIGINAL agent wording for one validated_fields entry —
    see failure_details' schema: "chat_value" must preserve the original
    agent wording, never a canonicalised bucket (e.g. degree's "claimed"
    is the bucket "specialist", not the Agent's actual "اخصائيه"). Prefers
    a "raw_claim" evidence key when the field's own validation block
    already computed one (degree/examination_age/walkin_fee — see their
    _xxx_claim_raw_match siblings); every other field falls back to this
    doctor's own scoped agent text (the real original wording near the
    claim — never a resolved category/BU-code/canonical value), and only
    as a last resort to entry["claimed"] itself."""
    raw = entry.get("raw_claim")
    if raw not in (None, ""):
        return raw
    if doctor_scope_text:
        return doctor_scope_text
    return entry.get("claimed")


def _build_field_failure_detail(
    field: str, entry: dict[str, Any], *,
    call: CallTranscript | None, resolved_name: str | None,
    doctor_scope_text: str,
) -> dict[str, Any]:
    """Build one structured failure_details entry (Part: doctor-validation
    results/logging/persistence/UI presentation schema) for a field whose
    validated_fields[field]["outcome"] == "FAIL". chat_value preserves the
    Agent's ORIGINAL wording; crm_value is always a genuine CRM display
    value the field was actually checked against, never a bare None;
    transcript_excerpt is local to THIS doctor and THIS field only (see
    _field_transcript_excerpt) — never a different recommended doctor's
    evidence in a multi-doctor recommendation set."""
    crm_value = entry.get("reference")
    if crm_value in (None, ""):
        crm_value = "Not on file"
    return {
        "field": field,
        "label": _FIELD_FAILURE_LABELS.get(field, field.replace("_", " ").title() + " mismatch"),
        "chat_value": _raw_claim_for_field(field, entry, doctor_scope_text),
        "crm_value": crm_value,
        "chat_canonical": entry.get("claimed"),
        "crm_canonical": crm_value.lower() if isinstance(crm_value, str) else crm_value,
        "reason": _FIELD_FAILURE_REASONS.get(
            field, "The agent-stated value for this field does not match the authoritative CRM record.",
        ),
        "transcript_excerpt": _field_transcript_excerpt(call, resolved_name, _field_failure_predicate(field)),
    }


def _log_skipped_optional_fields(validated: dict[str, Any]) -> None:
    """Concise diagnostic listing which optional (claim-driven) fields were
    never mentioned at all for this doctor — printed alongside, never
    instead of, the per-field validation logs above. Purely observational:
    skipped fields are never added to validated_fields and never affect
    fields_checked or the aggregated outcome."""
    skipped = [f for f in _OPTIONAL_CLAIM_FIELDS if f not in validated]
    if skipped:
        print(f"[doctor] optional_fields_not_mentioned={skipped}", flush=True)


def _result(
    outcome: str, reason: str, *, applicable: bool = False,
    doctor: dict[str, Any] | None = None, validated_fields: dict[str, Any] | None = None,
    candidate_count: int = 0, call: CallTranscript | None = None,
    context_specialty: str | None = None,
    input_name: str | None = None, resolution_source: str | None = None,
    failure_details: list[dict[str, Any]] | None = None,
    warning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "applicable": applicable,
        "outcome": outcome,
        # Which requested name this result is for, and which resolver tier
        # found it — additive fields, mainly useful for multi-doctor
        # recommendation-set logging/aggregation (see
        # validate_doctor_information's multi-doctor path); always None/
        # unset for the single-doctor path, harmless either way.
        "input_name": input_name,
        "resolution_source": resolution_source,
        "doctor_resolved": doctor is not None,
        "doctor_key": doctor.get("cr301_doctorkey") if doctor else None,
        "doctor_name_ar": doctor.get("cr301_doctornamear") if doctor else None,
        "doctor_name_en": doctor.get("servhub_doctornameen") if doctor else None,
        "business_unit": doctor.get("cr18c_buname") if doctor else None,
        "status": doctor.get("statuscodename") if doctor else None,
        "opd_flag": doctor.get("cr301_opdflag") if doctor else None,
        "candidate_count": candidate_count,
        "validated_fields": validated_fields or {},
        # Additive context for the (separate, LLM-based) scope/recommendation
        # validation — never sent as a raw CRM dump, just the fields it
        # actually needs (see doctor_scope_validation.py).
        "scope_reference": _summarize_doctor(doctor) if doctor else None,
        # Supporting evidence ONLY — the specialty/field the recommendation
        # was framed around in the conversation (e.g. "طبيب مخ واعصاب" said
        # just before naming the doctor). NEVER the doctor's authoritative
        # CRM specialty (see scope_reference["specialty"] for that) and
        # NEVER the appointment extractor's requested-service specialty
        # (app.agent.nodes.extract_appointment_details's specialty_name) —
        # the two must never be conflated or overwrite one another. None
        # when no such contextual specialty phrase was found. Callers
        # should pass the ALREADY-SCOPED value from classify_specific_
        # doctor_intent's own "doctor_context_specialty" (scoped to the
        # accepted doctor's own turn — see extract_doctor_context_
        # specialty's docstring) via context_specialty rather than letting
        # this recompute an unscoped one from just `call`.
        "doctor_context_specialty": (
            context_specialty if context_specialty is not None
            else (extract_doctor_context_specialty(call) if call is not None else None)
        ),
        "is_violation": outcome in DOCTOR_FAILURES,
        "reason": reason,
        # Structured, UI/persistence-facing detail lists (Part: doctor-
        # validation results/logging/persistence/UI presentation) — kept
        # STRICTLY separate: failure_details are genuine violations
        # (already reflected in outcome/is_violation above);
        # warning_details are non-punitive quality notices (e.g.
        # name_completeness) that NEVER affect outcome/is_violation,
        # scoring, needs_review, or escalation — see _name_completeness_
        # warning's docstring. Always present (empty list, never omitted)
        # so a consumer never has to guess whether an absent key means "no
        # failures" or "not computed".
        "failure_details": failure_details or [],
        "warning_details": warning_details or [],
    }


def _resolve_and_validate_one_doctor(
    query_candidates: list[str],
    *,
    call: CallTranscript,
    call_id: str,
    agent_text: str,
    patient_text: str,
    full_pool: list[dict[str, Any]],
    authoritative_pool: list[dict[str, Any]],
    bu_scoped_pool: list[dict[str, Any]],
    call_bu: str | None,
    intent_ctx: dict[str, Any],
    patient_candidates: list[str],
    agent_candidates: list[str],
    allow_single_token: bool = False,
    allow_contextual_single_name: bool = False,
) -> dict[str, Any]:
    """Resolve ONE doctor from *query_candidates* (tried in order, first
    match wins — see resolve_doctor_candidates) against the ALREADY-BUILT
    pools, then validate the Agent's factual claims about it. This is the
    single-doctor resolution+validation body validate_doctor_information
    has always used, now a reusable unit so a multi-doctor recommendation
    set (see validate_doctor_information's docstring) can call it once per
    recommended name against the exact same pools — the CRM dataset itself
    is still fetched/deduplicated/pool-built exactly once per call, in the
    caller, never once per doctor.

    *allow_single_token* (default False, preserving existing behaviour for
    every pre-existing caller) is forwarded to resolve_doctor_candidates —
    see that function's docstring. It is only ever set True by
    validate_doctor_information for a query that already came from the
    semantic (LLM-vetted) doctor-name extraction, never for a raw,
    unvetted candidate. Because bu_scoped_pool is tried BEFORE
    authoritative_pool, this naturally implements the required priority
    order end to end: (1) exact name in the same BU, (2) safe partial name
    in the same BU, (3) exact name across the full authoritative pool only
    once the same-BU attempt found nothing, (4) safe partial across the
    full authoritative pool likewise — never picking a same-name doctor
    from a DIFFERENT business unit while a same-BU candidate pool exists.

    *allow_contextual_single_name* (default False) is ONLY ever set True
    by validate_doctor_information's MULTI-doctor recommendation-set
    branch — never the single-doctor path — and only adds ONE further
    fallback tier, tried after the ordinary authoritative-pool tier above
    and before the outside-authoritative-scope fallback below: a bare,
    single-token query may still resolve via _resolve_single_token_
    candidate_with_context's safe, non-fuzzy CONTEXTUAL narrowing (Active +
    supported BU + call BU + specialty context) when every earlier tier
    found nothing. See that function's own docstring for the full
    business rationale; this flag never changes behaviour for a
    multi-token query or for the single-doctor path.
    """
    candidates: list[dict[str, Any]] = []
    resolution_source = "none"
    resolved_query: str | None = None  # the actual query string that produced `candidates` — needed for diagnostics
    if bu_scoped_pool:
        for query in query_candidates:
            found = resolve_doctor_candidates(query, bu_scoped_pool, allow_fuzzy=True, allow_single_token=allow_single_token)
            if found:
                candidates, resolution_source, resolved_query = found, "authoritative_pool_bu_scoped", query
                break

    if not candidates:
        for query in query_candidates:
            found = resolve_doctor_candidates(query, authoritative_pool, allow_fuzzy=True, allow_single_token=allow_single_token)
            if found:
                candidates, resolution_source, resolved_query = found, "authoritative_pool", query
                break

    if not candidates and allow_contextual_single_name:
        _specialty_context = _recommendation_set_specialty_context(call) if call is not None else None
        for query in query_candidates:
            found = _resolve_single_token_candidate_with_context(
                query, authoritative_pool, call_bu, _specialty_context,
            )
            if found:
                candidates, resolution_source, resolved_query = found, "contextual_single_name", query
                break
        if candidates:
            _doc_print_section(
                "contextual single-name resolution", query=resolved_query,
                specialty_context=_specialty_context, candidate_count=len(candidates),
            )

    if not candidates:
        # Defensive fallback: does this doctor exist at all, just outside
        # authoritative scope (inactive / unsupported BU)? That's
        # a FAIL, not a bare "no such doctor" — never resolved arbitrarily
        # against noise (non-human rows) since name matching stays
        # conservative regardless of which pool it runs against.
        for query in query_candidates:
            found = resolve_doctor_candidates(query, full_pool, allow_fuzzy=True, allow_single_token=allow_single_token)
            if found:
                candidates, resolution_source, resolved_query = found, "outside_authoritative_scope", query
                break

    _input_name = query_candidates[0] if query_candidates else None

    if not candidates:
        logger.info("doctor resolution | call_id=%s source=none resolved=None candidate_count=0", call_id)
        _doc_print_section("resolution", source="none", doctor_key=None, candidate_count=0)
        _requested = query_candidates[0] if query_candidates else ""
        if _requested:
            _log_near_miss_candidates(call_id, _requested, full_pool)
        _doc_print()
        _doc_print("outcome: DOCTOR_UNRESOLVED")
        return _result(
            "DOCTOR_UNRESOLVED",
            "No authoritative CRM doctor record could be resolved from the conversation.",
            applicable=True, call=call, context_specialty=intent_ctx.get("doctor_context_specialty"),
            input_name=_input_name, resolution_source="none",
        )

    if len(candidates) > 1:
        keys = {c.get("cr301_doctorkey") for c in candidates}
        if len(keys) > 1:
            # Real CRM data quirk (distinct from same-key duplicate rows,
            # already merged upstream): the same physical doctor sometimes
            # exists under two SEPARATE doctor keys (e.g. a re-onboarded
            # contract) with identical name + profile. If every tied
            # candidate shares the exact same name AND the same validatable
            # profile fields, the identity itself isn't ambiguous — proceed
            # safely with one of them (mirrors location_validation.py's
            # "shared CRM address -> safe to proceed" tie-breaker).
            profile_fp = {
                (
                    normalize_arabic_text(str(c.get("cr301_doctornamear") or "")),
                    normalize_arabic_text(str(c.get("servhub_doctornameen") or "")),
                    c.get("cr301_degreename"), c.get("cr301_specialtyname"),
                    c.get("cr301_subspecialtyname"), c.get("cr18c_buname"),
                )
                for c in candidates
            }
            if len(profile_fp) == 1:
                logger.info(
                    "doctor multiple records matched | call_id=%s candidate_count=%d "
                    "ambiguity=safe_identical_profile_across_keys",
                    call_id, len(candidates),
                )
                candidates = candidates[:1]
            else:
                # Last-resort BU tie-break (belt-and-suspenders alongside
                # the BU-scoped resolution attempt above, which already
                # handles the common case): if the call's own detected BU
                # is known and matches EXACTLY ONE tied candidate on EITHER
                # of that candidate's two CRM BU fields (see
                # _match_doctor_business_unit), prefer it over reporting
                # AMBIGUOUS_DOCTOR — never applied when the BU is unknown or
                # matches more than one candidate.
                bu_matches = (
                    [c for c in candidates if _match_doctor_business_unit(call_bu, c) is not None]
                    if call_bu else []
                )
                if len(bu_matches) == 1:
                    logger.info(
                        "doctor multiple records matched | call_id=%s candidate_count=%d "
                        "ambiguity=resolved_by_call_business_unit=%s",
                        call_id, len(candidates), call_bu,
                    )
                    candidates = bu_matches
                    resolution_source = "authoritative_pool_bu_tiebreak"
                else:
                    _doc_print_section("resolution", source=resolution_source, candidate_count=len(candidates))
                    _log_ambiguous_resolution(call_id, resolved_query or "", call_bu, candidates)
                    _doc_print()
                    _doc_print("outcome: AMBIGUOUS_DOCTOR")
                    return _result(
                        "AMBIGUOUS_DOCTOR",
                        f"The mentioned doctor name matches {len(keys)} equally plausible CRM records — cannot disambiguate without more context.",
                        applicable=True, candidate_count=len(keys), call=call,
                        context_specialty=intent_ctx.get("doctor_context_specialty"),
                        input_name=_input_name, resolution_source=resolution_source,
                    )
        else:
            candidates = candidates[:1]  # same doctor key, duplicate rows already merged upstream

    doctor = candidates[0]

    # Non-punitive name_completeness WARNING (Part: doctor-validation
    # results/logging/persistence/UI presentation) — computed as soon as
    # identity is resolved, reused by every return path below that has a
    # resolved doctor. NEVER a failure: see _name_completeness_warning's
    # docstring for why it never affects outcome/is_violation.
    _name_warning = _name_completeness_warning(
        _input_name, doctor, call, resolved_query or _input_name,
    )
    _warning_details = [_name_warning] if _name_warning else []
    if _name_warning:
        _doc_print_section(
            "name completeness", outcome="WARNING",
            chat_value=_name_warning["chat_value"], crm_value=_name_warning["crm_value"],
        )

    logger.info(
        "doctor resolution | call_id=%s source=%s doctor_key=%s doctor_name_ar=%r business_unit=%s "
        "opd_flag=%s candidate_count=%d",
        call_id, resolution_source, doctor.get("cr301_doctorkey"), doctor.get("cr301_doctornamear"),
        doctor.get("cr18c_buname"), doctor.get("cr301_opdflag"), len(candidates),
    )
    _doc_print_section(
        "resolution", source=resolution_source, doctor_key=doctor.get("cr301_doctorkey"),
        doctor_name_ar=doctor.get("cr301_doctornamear"), doctor_name_en=doctor.get("servhub_doctornameen"),
        business_unit=doctor.get("cr18c_buname"),
        # opd_flag is informational only — carried and shown here, but it
        # never determined whether this doctor was searchable/resolvable.
        opd_flag=doctor.get("cr301_opdflag"), candidate_count=len(candidates),
    )

    # Mandatory doctor evidence, surfaced explicitly for every resolved
    # doctor (not just when the Agent happened to make a validatable claim
    # about it below) — specialty/subspecialty/scope-of-service are core
    # doctor evidence, not an optional afterthought; when CRM has no
    # detailed scope text this says so explicitly rather than staying
    # silent (see has_detailed_scope_evidence()).
    _scope_ref_for_log = _summarize_doctor(doctor)
    _doc_print_section(
        "CRM evidence",
        specialty=doctor.get("cr301_specialtyname") or doctor.get("cr18c_manualspecialtyname"),
        subspecialty=doctor.get("cr301_subspecialtyname") or doctor.get("cr18c_manualsubspecialtyname"),
        scope_of_service_available=has_detailed_scope_evidence(_scope_ref_for_log),
        scope_source=(
            "scope_of_service" if has_detailed_scope_evidence(_scope_ref_for_log) else "missing"
        ),
        doctor_notes_available=bool((doctor.get("cr301_drnotes") or "").strip()),
    )

    # Observability: make the doctor-role/scope-applicability decision
    # visible right where the doctor was resolved, not just buried in the
    # separate [doctor_scope] routing log — see classify_doctor_context()'s
    # docstring for what each field means. Reuses the SAME intent_ctx the
    # caller already computed once — never re-derives a second detector.
    _doc_print_section(
        "context classification",
        business_unit=doctor.get("cr18c_buname"),
        doctor_intent=intent_ctx["doctor_intent"].upper(),
        doctor_role=intent_ctx["doctor_role"],
        booking_target=intent_ctx["booking_target"],
        inquiry_target=intent_ctx["inquiry_target"],
        patient_named_doctor=bool(patient_candidates),
        agent_named_doctor=bool(agent_candidates),
        clinical_need_detected=patient_describes_medical_complaint(patient_text),
        scope_applicable=intent_ctx["scope_applicable"],
    )

    if resolution_source == "outside_authoritative_scope":
        reasons = []
        if not _is_active(doctor):
            reasons.append(f"status={doctor.get('statuscodename')!r} (not Active)")
        if not is_supported_doctor_bu(doctor.get("cr18c_buname")):
            reasons.append(f"business_unit={doctor.get('cr18c_buname')!r} (not in supported scope)")
        # opd_flag is NEVER a reason here — it is informational only (see
        # _is_active's docstring); this branch is only reached at all when
        # the doctor is inactive or outside the supported-BU scope.
        _doc_print()
        _doc_print(f"opd_flag={doctor.get('cr301_opdflag')!r} (informational only, not a searchability gate)")
        if _mentions_home_care(doctor):
            _doc_print(
                "note: this doctor's own scope/notes text mentions home-care/home-visit service.",
            )
        _doc_print(f"outcome: FAIL ({'; '.join(reasons)})")
        return _result(
            "FAIL",
            f"The recommended doctor is not a valid authoritative record: {'; '.join(reasons)}.",
            applicable=True, doctor=doctor, candidate_count=1, call=call,
            context_specialty=intent_ctx.get("doctor_context_specialty"),
            input_name=_input_name, resolution_source=resolution_source,
            warning_details=_warning_details,
        )

    # ── Field-by-field validation of ONLY the Agent's actual claims ────────
    validated: dict[str, Any] = {}

    # Every optional (claim-driven) field below — degree, business_unit,
    # doctor_notes, scope_of_service, qualifications, examination_age,
    # walkin_fee — is scoped to THIS resolved doctor's own agent turn(s),
    # computed ONCE and shared, never a blind scan of the whole call's
    # agent_text: in a multi-doctor conversation an explicit claim about a
    # DIFFERENT recommended doctor must never be attributed to this one
    # (Section 17). Falls back to the whole agent_text only when no anchor
    # name is available at all (e.g. a pure Trigger-B/no-candidates path).
    _doctor_scope_anchor_name = resolved_query or (query_candidates[0] if query_candidates else None)
    _doctor_scope_text = (
        _agent_turns_text_for_doctor(call, _doctor_scope_anchor_name)
        if _doctor_scope_anchor_name else agent_text
    )

    # Degree/rank claim. A bare دكتور/دكتورة/د/Dr title is never itself a degree
    # claim (see _DEGREE_CLAIM_PATTERNS — it only recognises explicit
    # professional ranks like استشاري/اخصائي/استاذ/GP/نائب/مقيم).
    degree_claim = _extract_degree_claim(_doctor_scope_text)
    if degree_claim is not None:
        crm_degree = _canon_degree_value(doctor.get("cr301_degreename"))
        ok = bool(crm_degree) and degree_claim == crm_degree
        degree_outcome = "PASS" if ok else "FAIL"
        validated["degree"] = _field(
            degree_claim, degree_outcome, doctor.get("cr301_degreename"),
            raw_claim=_degree_claim_raw_match(_doctor_scope_text),
        )
        _log_field_validation(
            "degree",
            raw_claim=_degree_claim_raw_match(_doctor_scope_text),
            canonical_claim=_DEGREE_CANON_DISPLAY.get(degree_claim, degree_claim),
            crm_reference=doctor.get("cr301_degreename"),
            canonical_crm=crm_degree,
            outcome=degree_outcome,
        )

    # Specialty claim: scoped to THIS resolved doctor's own turn(s) — never
    # a blind scan of the whole call's agent_text, which could pick up an
    # unrelated specialty mentioned elsewhere for a different reason (e.g.
    # "بعد الكشف تروحي قسم الاشعة" said after an Oncology doctor was
    # already booked must never become that doctor's Radiology claim).
    # intent_ctx["doctor_context_specialty"] is reused directly when it is
    # already scoped to this exact doctor (this also covers the LLM-vetted
    # semantic_specialty_context override — see validate_doctor_
    # information's docstring); otherwise it is recomputed fresh, scoped
    # to THIS doctor's own name, so a multi-doctor recommendation set
    # never has one doctor's specialty context bleed into another's.
    _specialty_anchor_name = resolved_query or (query_candidates[0] if query_candidates else None)
    _anchor_matches_classifier_name = bool(
        _specialty_anchor_name and intent_ctx.get("named_doctor")
        and _is_plausible_same_person_name(
            _name_tokens(_specialty_anchor_name), _name_tokens(intent_ctx["named_doctor"]),
        )
    )
    if _anchor_matches_classifier_name:
        # Safe (same-person, not necessarily identical-string — the
        # anchor may be a cleaner CRM-resolved/semantic name than the
        # classifier's own noisier regex candidate, e.g. "وصال" vs "وصال
        # بعياده") match against the classifier's own anchor name — reuse
        # its doctor_context_specialty directly, since that already
        # covers the LLM-vetted semantic_specialty_context override (see
        # validate_doctor_information's docstring).
        _scoped_specialty_text = intent_ctx.get("doctor_context_specialty")
    elif _specialty_anchor_name:
        # Multi-doctor recommendation set (or any name the classifier's
        # own single anchor doesn't correspond to) — recompute fresh,
        # scoped to THIS doctor's own name only, so specialty context
        # never bleeds from one recommended doctor to another.
        _scoped_specialty_text = _specialty_context_for_resolved_name(call, _specialty_anchor_name)
    else:
        _scoped_specialty_text = None

    # Required lookup order (see module docstring / README_doctor.md):
    #   1. CRM specialty fields    (cr301_specialtyname, cr18c_manualspecialtyname)
    #   2. CRM subspecialty fields (cr301_subspecialtyname, cr18c_manualsubspecialtyname)
    #   3. CRM scope of service    (cr301_scopeofservicear, cr301_scopeofservice)
    #      — ONLY tried once neither 1 nor 2 produced a match, using the
    #      RAW claim text (_match_specialty_against_scope) so a genuine
    #      service phrase with no top-level category of its own (e.g.
    #      "الحشوات") can still be confirmed via a doctor's detailed scope.
    # Entered whenever there is ANY specialty-shaped claim at all — either
    # a coarse category was resolved (see _resolve_specialty_category) OR
    # the raw scoped text is itself a recognised specialty/service phrase
    # even without one (see _GENERIC_SPECIALTY_WORDS). _scoped_specialty_
    # text is None/empty only when nothing specialty-shaped was said about
    # this doctor at all, which must never itself cause a FAIL.
    if _scoped_specialty_text:
        specialty_claim = _resolve_specialty_category(_scoped_specialty_text)

        # Compound-claim decomposition ("عظام اطفال" = base specialty
        # "عظام"/Orthopedics + a pediatric QUALIFIER "اطفال" — see
        # _split_compound_specialty_claim's own docstring for the full
        # root-cause explanation). specialty_claim above is resolved from
        # the FULL, un-decomposed claim text via a longest-alias-SUBSTRING
        # search that has no notion of "compound phrase": a co-occurring
        # pediatric qualifier word can accidentally out-rank a genuinely
        # SHORTER base-specialty word purely on character length (e.g.
        # "اطفال", 5 chars, beating "عظام", 4 chars) and hijack the whole
        # claim's resolved category. Re-resolving the QUALIFIER-STRIPPED
        # remainder on its own has no such competing token, so it reliably
        # recovers the real base specialty; when that succeeds, it REPLACES
        # specialty_claim for every comparison below. specialty_ok/
        # subspecialty_ok just below still compare against the FULL
        # original claim text (never the stripped remainder), so an
        # already-exact formal pediatric specialty/subspecialty on file
        # (e.g. CRM "Pediatric Orthopedics") keeps matching directly there,
        # exactly as it already did before this fix for claims where the
        # naive whole-phrase resolution happened to already be correct
        # (e.g. "غدد صماء أطفال" -> "Endocrinology").
        _pediatric_base_text, _pediatric_qualifiers = _split_compound_specialty_claim(_scoped_specialty_text)
        _base_specialty_claim = (
            _resolve_specialty_category(_pediatric_base_text)
            if _pediatric_qualifiers and _pediatric_base_text else None
        )
        if _base_specialty_claim:
            specialty_claim = _base_specialty_claim

        # specialty and subspecialty may BOTH legitimately be valid for the
        # same claim wording (Part: specialty=Pediatrics, subspecialty=
        # Pediatric Endocrinology are both correct depending on what the
        # Agent said) — so check specialty FIRST, and only fall back to
        # checking subspecialty when the claim doesn't describe the general
        # specialty. A claim that's really a subspecialty description (e.g.
        # "غدد صماء" for a Pediatrics doctor whose subspecialty is Pediatric
        # Endocrinology) must not be reported as a FAILED specialty claim —
        # it was never an assertion about the general specialty at all.
        # Stay None (never computed) when no coarse category resolved at
        # all — there is nothing to compare against cr301_specialtyname/
        # cr301_subspecialtyname in that case (the raw-phrase-only path,
        # e.g. "الحشوات", goes straight to the scope fallback below).
        specialty_ok = subspecialty_ok = None
        if specialty_claim is not None:
            specialty_ok = _specialty_claim_result(
                specialty_claim, _scoped_specialty_text,
                doctor.get("cr301_specialtyname"), doctor.get("cr18c_manualspecialtyname"),
            )
            subspecialty_ok = _specialty_claim_result(
                specialty_claim, _scoped_specialty_text,
                doctor.get("cr301_subspecialtyname"), doctor.get("cr18c_manualsubspecialtyname"),
            )

        # Compound base+qualifier path: only engaged when the claim
        # genuinely decomposed into base+qualifier AND neither check above
        # already PASSed outright (i.e. CRM has no formal pediatric-
        # flavoured specialty/subspecialty on file matching this claim
        # directly). The BASE specialty is matched deliberately IGNORING
        # the pediatric guard — comparing against the QUALIFIER-STRIPPED
        # text (which mentions no pediatric marker at all) rather than the
        # full claim — so a doctor whose CRM specialty is plain
        # "Orthopedics" is a legitimate base match for "عظام اطفال" even
        # though bare "Orthopedics" is not itself pediatric-flagged; the
        # pediatric QUALIFIER is then confirmed SEPARATELY, only after the
        # base already matched, against the doctor's own scope-of-service
        # text (see _match_pediatric_qualifier_against_scope) — never
        # merely because the scope happens to contain "أطفال" somewhere
        # unrelated to the base specialty.
        _base_specialty_match: bool | None = None
        _qualifier_evidence: dict[str, Any] | None = None
        if specialty_ok is not True and subspecialty_ok is not True and _base_specialty_claim and _pediatric_qualifiers:
            _base_specialty_match = _specialty_claim_result(
                _base_specialty_claim, _pediatric_base_text,
                doctor.get("cr301_specialtyname"), doctor.get("cr18c_manualspecialtyname"),
                doctor.get("cr301_subspecialtyname"), doctor.get("cr18c_manualsubspecialtyname"),
            )
            if _base_specialty_match is True:
                _qualifier_evidence = _match_pediatric_qualifier_against_scope(doctor)

        _scope_fallback: dict[str, Any] | None = None
        if specialty_ok is True:
            validated["specialty"] = _field(specialty_claim, "PASS", doctor.get("cr301_specialtyname") or doctor.get("cr18c_manualspecialtyname"))
        elif subspecialty_ok is True:
            validated["subspecialty"] = _field(
                specialty_claim, "PASS", doctor.get("cr301_subspecialtyname") or doctor.get("cr18c_manualsubspecialtyname"),
            )
        elif _base_specialty_match is True:
            _pediatric_reference = (
                doctor.get("cr301_specialtyname") or doctor.get("cr18c_manualspecialtyname")
                or doctor.get("cr301_subspecialtyname") or doctor.get("cr18c_manualsubspecialtyname")
            )
            if _qualifier_evidence and _qualifier_evidence["matched"]:
                validated["specialty"] = _field(
                    specialty_claim, "PASS", _pediatric_reference,
                    base_specialty=_base_specialty_claim, qualifiers=_pediatric_qualifiers,
                    base_specialty_match=True, qualifier_match=True,
                    match_source=_qualifier_evidence["source"], matched_phrase=_qualifier_evidence["phrase"],
                )
            else:
                # Base specialty confirmed, but nothing in the doctor's own
                # scope text evidences the pediatric qualifier —
                # NEEDS_REVIEW (never a silent FAIL just because CRM only
                # stores the general specialty), UNLESS the doctor's own
                # CRM data explicitly contradicts a pediatric claim (e.g. a
                # structured adult-only examination-age range).
                _outcome = "FAIL" if _pediatric_scope_contradicted(doctor) else "NEEDS_REVIEW"
                validated["specialty"] = _field(
                    specialty_claim, _outcome, _pediatric_reference,
                    base_specialty=_base_specialty_claim, qualifiers=_pediatric_qualifiers,
                    base_specialty_match=True, qualifier_match=False,
                )
        else:
            # Step 3: CRM scope of service — tried because neither
            # specialty nor subspecialty produced a PASS above (whether
            # they contradicted the claim, or there was no coarse category
            # to compare against them in the first place), and the compound
            # base+qualifier path above either didn't apply or the base
            # specialty itself didn't match anything on file either (a
            # genuinely unrelated specialty, e.g. "عظام اطفال" against a
            # Cardiology-only doctor, still correctly FAILs/NEEDS_REVIEWs
            # below exactly as any other unmatched claim would).
            _scope_fallback = _match_specialty_against_scope(_scoped_specialty_text, specialty_claim, doctor)
            if _scope_fallback["matched"]:
                validated["scope_of_service"] = _field(
                    _scoped_specialty_text, "PASS", _scope_fallback["reference"],
                    match_source=_scope_fallback["source"], matched_phrase=_scope_fallback["matched_phrase"],
                    scope_fallback_used=True,
                )
            else:
                # No match anywhere in the required lookup order. FAIL when
                # this doctor actually had SOME populated evidence (in
                # specialty, subspecialty, OR scope) that simply didn't
                # match; NEEDS_REVIEW only when every one of the 6 fields
                # is empty — never a silent FAIL for missing CRM data.
                has_any_evidence = any(
                    doctor.get(f) for f in (
                        "cr301_specialtyname", "cr18c_manualspecialtyname",
                        "cr301_subspecialtyname", "cr18c_manualsubspecialtyname",
                        "cr301_scopeofservicear", "cr301_scopeofservice",
                    )
                )
                reference = (
                    doctor.get("cr301_specialtyname") or doctor.get("cr18c_manualspecialtyname")
                    or doctor.get("cr301_subspecialtyname") or doctor.get("cr18c_manualsubspecialtyname")
                )
                outcome = "FAIL" if has_any_evidence else "NEEDS_REVIEW"
                validated["specialty"] = _field(specialty_claim or _scoped_specialty_text, outcome, reference)

        # Concise, always-visible (not DEBUG-gated) diagnostic for exactly
        # this one checked claim — never a dump of every normalisation
        # attempt — so a production PASS/FAIL/NEEDS_REVIEW is never an
        # unexplained verdict (see this module's specialty-canonicalisation
        # design).
        _field_result = validated.get("specialty") or validated.get("subspecialty") or validated.get("scope_of_service")
        _matched_field_name = (
            "subspecialty" if "subspecialty" in validated
            else "scope_of_service" if "scope_of_service" in validated
            else "specialty"
        )
        print(
            f"[doctor] field validation:\n"
            f"[doctor]     field={_matched_field_name!r}\n"
            f"[doctor]     claimed_raw={_scoped_specialty_text!r}\n"
            f"[doctor]     resolved_base_specialty={specialty_claim!r}\n"
            f"[doctor]     detected_qualifiers={_pediatric_qualifiers!r}\n"
            f"[doctor]     claimed_canonical={(_specialty_core(specialty_claim) if specialty_claim else None)!r}\n"
            f"[doctor]     crm_specialty={doctor.get('cr301_specialtyname')!r}\n"
            f"[doctor]     crm_manual_specialty={doctor.get('cr18c_manualspecialtyname')!r}\n"
            f"[doctor]     crm_subspecialty={doctor.get('cr301_subspecialtyname')!r}\n"
            f"[doctor]     crm_manual_subspecialty={doctor.get('cr18c_manualsubspecialtyname')!r}\n"
            f"[doctor]     base_specialty_match={_base_specialty_match!r}\n"
            f"[doctor]     qualifier_match={bool(_qualifier_evidence and _qualifier_evidence['matched'])!r}\n"
            f"[doctor]     qualifier_match_source={(_qualifier_evidence['source'] if _qualifier_evidence else None)!r}\n"
            f"[doctor]     matched_pediatric_phrase={(_qualifier_evidence['phrase'] if _qualifier_evidence else None)!r}\n"
            f"[doctor]     scope_fallback_used={bool(_scope_fallback and _scope_fallback['matched'])}\n"
            f"[doctor]     match_source={(_scope_fallback['source'] if _scope_fallback else None)!r}\n"
            f"[doctor]     matched_phrase={(_scope_fallback['matched_phrase'] if _scope_fallback else None)!r}\n"
            f"[doctor]     result={_field_result['outcome'] if _field_result else None}",
            flush=True,
        )

    # Examination age: optional, but HIGH PRIORITY when mentioned — checked
    # (and logged, with full diagnostics) right after specialty, before the
    # lower-priority free-text optional fields below (doctor_notes/
    # scope_of_service/qualifications) and before walkin_fee.
    age_claim = _extract_examination_age_claim(_doctor_scope_text)
    if age_claim is not None:
        parsed = parse_examination_age(doctor.get("servhub_examinationage"))
        if parsed is None:
            # Missing/unparseable CRM age data — never silently accepted,
            # never a hard FAIL either: no authoritative range to confirm
            # OR contradict the claim against.
            age_outcome = "NEEDS_REVIEW"
        else:
            min_age, max_age = parsed
            if age_claim == "children":
                ok = min_age <= 12  # accepts pediatric patients somewhere in range
            elif age_claim == "adults_only":
                ok = min_age >= 14
            elif age_claim.startswith("from:"):
                claimed_from = float(age_claim.split(":", 1)[1])
                ok = abs(claimed_from - min_age) <= 1  # tolerate off-by-one boundary phrasing
            elif age_claim.startswith("range:"):
                claimed_lo_s, claimed_hi_s = age_claim.split(":", 1)[1].split("-")
                claimed_lo, claimed_hi = float(claimed_lo_s), float(claimed_hi_s)
                ok = claimed_lo >= min_age - 1 and (max_age == float("inf") or claimed_hi <= max_age + 1)
            else:
                # Defensive only: every tag _extract_examination_age_claim
                # can actually return is handled above. An unrecognised tag
                # must never silently PASS.
                ok = None
            age_outcome = "NEEDS_REVIEW" if ok is None else ("PASS" if ok else "FAIL")
        validated["examination_age"] = _field(
            age_claim, age_outcome, doctor.get("servhub_examinationage"),
            raw_claim=_examination_age_claim_raw_match(_doctor_scope_text),
        )
        _log_field_validation(
            "examination_age",
            raw_claim=_examination_age_claim_raw_match(_doctor_scope_text),
            canonical_claim=age_claim,
            crm_reference=doctor.get("servhub_examinationage"),
            canonical_crm=(f"{parsed[0]}-{parsed[1]}" if parsed else None),
            outcome=age_outcome,
        )

    # Business-unit/branch claim: resolved from free text via the project's
    # one canonical BU-alias table (app.models.input.BUSINESS_UNIT_KEYWORD_MAP,
    # via resolve_business_unit), never just a literal search for the
    # doctor's OWN BU code in the chat — that could only ever confirm a
    # match and could never catch a genuinely WRONG stated branch.
    #
    # PASSes when the claim matches EITHER of the doctor's two CRM BU
    # fields (cr301_businessunitname OR cr18c_buname) — see
    # _match_doctor_business_unit. Real CRM data has doctors where these
    # two fields disagree; a claim matching just one of them is still a
    # genuine, correct claim, and the disagreement itself must never turn
    # that into a FAIL. This is claim validation only — it never touches
    # record eligibility/searchability, which stays cr18c_buname-only (see
    # authoritative_doctor_pool).
    _claimed_bu_app_code = resolve_business_unit(_doctor_scope_text)
    if _claimed_bu_app_code is not None:
        claimed_bu_crm = canonical_doctor_bu(_claimed_bu_app_code)
        _crm_bu_cr301 = doctor.get("cr301_businessunitname")
        _crm_bu_cr18c = doctor.get("cr18c_buname")
        matched_crm_field = _match_doctor_business_unit(claimed_bu_crm, doctor)
        bu_outcome = "PASS" if matched_crm_field else "FAIL"
        # Reference value: the doctor's own value on whichever CRM field
        # matched the claim; falls back to cr18c_buname (the eligibility-
        # authoritative field elsewhere in this module) when nothing
        # matched, so a FAIL still shows a concrete CRM value to compare
        # the claim against.
        crm_reference = doctor.get(matched_crm_field) if matched_crm_field else _crm_bu_cr18c
        validated["business_unit"] = _field(
            claimed_bu_crm, bu_outcome, crm_reference,
            matched_crm_field=matched_crm_field,
            crm_business_units={
                "cr301_businessunitname": _crm_bu_cr301,
                "cr18c_buname": _crm_bu_cr18c,
            },
        )
        print(
            f"[doctor] field validation:\n"
            f"[doctor]     field={'business_unit'!r}\n"
            f"[doctor]     claimed_raw={_claimed_bu_app_code!r}\n"
            f"[doctor]     claimed_canonical={claimed_bu_crm!r}\n"
            f"[doctor]     crm_cr301_businessunitname={_crm_bu_cr301!r}\n"
            f"[doctor]     crm_cr18c_buname={_crm_bu_cr18c!r}\n"
            f"[doctor]     matched_crm_field={matched_crm_field!r}\n"
            f"[doctor]     crm_reference={crm_reference!r}\n"
            f"[doctor]     result={bu_outcome}",
            flush=True,
        )

    notes_claim = _claim_text_if_triggered(_doctor_scope_text, _NOTES_CLAIM_TRIGGER_RE)
    if notes_claim is not None:
        crm_notes = doctor.get("cr301_drnotes")
        if not crm_notes:
            notes_outcome = "NEEDS_REVIEW"
        else:
            notes_outcome = "PASS" if _overlap_supports_claim(notes_claim, crm_notes) else "FAIL"
        validated["doctor_notes"] = _field(notes_claim, notes_outcome, crm_notes)
        _log_field_validation(
            "doctor_notes", raw_claim=notes_claim, canonical_claim=notes_claim,
            crm_reference=crm_notes, canonical_crm=crm_notes, outcome=notes_outcome,
        )

    # Skipped when the specialty/subspecialty scope FALLBACK above already
    # populated "scope_of_service" (e.g. the "الحشوات" case) — that is a
    # more specifically-targeted match (a claimed specialty/service phrase
    # confirmed against this exact doctor's scope text) than this open-
    # ended "what does the doctor treat" trigger-phrase check below, so it
    # must never be silently overwritten by this one re-evaluating the
    # same or a broader span of text.
    scope_claim = _claim_text_if_triggered(_doctor_scope_text, _SCOPE_CLAIM_TRIGGER_RE)
    if scope_claim is not None and "scope_of_service" not in validated:
        crm_scope = doctor.get("cr301_scopeofservicear") or doctor.get("cr301_scopeofservice")
        if not crm_scope:
            scope_outcome = "NEEDS_REVIEW"
        else:
            ok = _overlap_supports_claim(scope_claim, doctor.get("cr301_scopeofservicear"), doctor.get("cr301_scopeofservice"))
            scope_outcome = "PASS" if ok else "FAIL"
        validated["scope_of_service"] = _field(scope_claim, scope_outcome, crm_scope)
        _log_field_validation(
            "scope_of_service", raw_claim=scope_claim, canonical_claim=scope_claim,
            crm_reference=crm_scope, canonical_crm=crm_scope, outcome=scope_outcome,
        )

    qual_claim = _claim_text_if_triggered(_doctor_scope_text, _QUALIFICATION_CLAIM_TRIGGER_RE)
    if qual_claim is not None:
        crm_qual = doctor.get("cr301_qualificationsandexperiencear") or doctor.get("cr301_qualificationsandexperience")
        if not crm_qual:
            qual_outcome = "NEEDS_REVIEW"
        else:
            ok = _overlap_supports_claim(qual_claim, doctor.get("cr301_qualificationsandexperiencear"), doctor.get("cr301_qualificationsandexperience"))
            qual_outcome = "PASS" if ok else "FAIL"
        validated["qualifications"] = _field(qual_claim, qual_outcome, crm_qual)
        _log_field_validation(
            "qualifications", raw_claim=qual_claim, canonical_claim=qual_claim,
            crm_reference=crm_qual, canonical_crm=crm_qual, outcome=qual_outcome,
        )

    fee_claim = _extract_fee_claim(_doctor_scope_text)
    if fee_claim is not None:
        crm_fee = doctor.get("cr301_walkinconsultationfees")
        if crm_fee in (None, ""):
            fee_outcome = "NEEDS_REVIEW"
        else:
            try:
                ok = abs(float(fee_claim) - float(crm_fee)) < 1
            except (TypeError, ValueError):
                ok = False
            fee_outcome = "PASS" if ok else "FAIL"
        validated["walkin_fee"] = _field(
            fee_claim, fee_outcome, crm_fee,
            raw_claim=_fee_claim_raw_match(_doctor_scope_text),
        )
        _log_field_validation(
            "walkin_fee",
            raw_claim=_fee_claim_raw_match(_doctor_scope_text), canonical_claim=fee_claim,
            crm_reference=crm_fee, canonical_crm=(float(crm_fee) if crm_fee not in (None, "") else crm_fee),
            outcome=fee_outcome,
        )

    if validated:
        _doc_print_section("field validation", **{k: v["outcome"] for k, v in validated.items()})
    _log_skipped_optional_fields(validated)

    outcomes = [v["outcome"] for v in validated.values()]
    if "FAIL" in outcomes:
        outcome, reason = "FAIL", "One or more Agent-stated doctor details do not match the authoritative CRM record."
    else:
        outcome, reason = "PASS", "All Agent-stated doctor details that could be checked match the authoritative CRM record."

    # Structured, per-field failure evidence (Part: doctor-validation
    # results/logging/persistence/UI presentation) — built ONLY from
    # fields that actually FAILed above, local to THIS resolved doctor's
    # own scoped text (_doctor_scope_text) only — never another
    # recommended doctor's evidence in a multi-doctor set.
    _resolved_name_for_excerpts = resolved_query or _input_name
    failure_details = [
        _build_field_failure_detail(
            field, entry, call=call, resolved_name=_resolved_name_for_excerpts,
            doctor_scope_text=_doctor_scope_text,
        )
        for field, entry in validated.items() if entry["outcome"] == "FAIL"
    ]

    logger.info(
        "doctor validation | call_id=%s outcome=%s doctor_key=%s fields_checked=%d",
        call_id, outcome, doctor.get("cr301_doctorkey"), len(validated),
    )
    _doc_print()
    _doc_print(f"outcome: {outcome}")

    return _result(
        outcome, reason, applicable=True, doctor=doctor, validated_fields=validated,
        candidate_count=1, call=call, context_specialty=intent_ctx.get("doctor_context_specialty"),
        input_name=_input_name, resolution_source=resolution_source,
        failure_details=failure_details, warning_details=_warning_details,
    )


def _aggregate_doctor_recommendation_outcomes(per_doctor: list[dict[str, Any]]) -> tuple[str, str, bool]:
    """Combine N independent per-doctor validate_doctor_information results
    (a recommendation SET — see validate_doctor_information's multi-doctor
    path) into one overall (outcome, reason, is_violation), reusing the
    existing PASS/FAIL/AMBIGUOUS_DOCTOR/DOCTOR_UNRESOLVED semantics rather
    than inventing a new scoring scheme. The FIRST doctor's own outcome
    must NEVER stand in for the whole set:
      - any doctor FAIL or AMBIGUOUS_DOCTOR -> overall FAIL (a genuine
        violation exists somewhere in the recommended set)
      - none FAIL/AMBIGUOUS but one or more UNRESOLVED -> overall
        DOCTOR_UNRESOLVED (never silently dropped — the caller is always
        told which recommended name(s) never resolved)
      - every doctor resolved and PASS -> overall PASS
    """
    outcomes = [d.get("outcome") for d in per_doctor]
    if any(o in ("FAIL", "AMBIGUOUS_DOCTOR") for o in outcomes):
        bad = [d for d in per_doctor if d.get("outcome") in ("FAIL", "AMBIGUOUS_DOCTOR")]
        names = ", ".join(str(d.get("input_name")) for d in bad)
        return (
            "FAIL",
            f"{len(bad)} of {len(per_doctor)} recommended doctor(s) failed validation: {names}.",
            True,
        )
    if any(o == "DOCTOR_UNRESOLVED" for o in outcomes):
        unresolved = [d for d in per_doctor if d.get("outcome") == "DOCTOR_UNRESOLVED"]
        names = ", ".join(str(d.get("input_name")) for d in unresolved)
        if len(unresolved) == len(per_doctor):
            return (
                "DOCTOR_UNRESOLVED",
                f"None of the {len(per_doctor)} recommended doctors could be resolved against the authoritative CRM record.",
                False,
            )
        return (
            "DOCTOR_UNRESOLVED",
            f"{len(unresolved)} of {len(per_doctor)} recommended doctor(s) could not be resolved: {names}.",
            False,
        )
    return (
        "PASS",
        f"All {len(per_doctor)} recommended doctors resolved and passed validation.",
        False,
    )


def _active_doctor_in_recommendation_set(
    call: CallTranscript, per_doctor: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Which per-doctor entry (see validate_doctor_information's multi-
    doctor "doctors" list) the conversation most recently, UNAMBIGUOUSLY
    narrowed the discussion to — purely OBSERVATIONAL, used only to
    populate the active_doctor_input_name/active_doctor_key metadata
    fields (see validate_doctor_information's own docstring on them);
    never consulted by resolution/validation itself, and never a second,
    competing name-matching mechanism (it only ever compares an already-
    extracted turn candidate against the ALREADY-COMPUTED per_doctor
    entries' own input_name, via the SAME _is_plausible_same_person_name
    check every other same-person comparison in this module already
    uses).

    Scans every AGENT turn in transcript order (never reversed — the
    LAST such turn wins, since that is the most recent, most relevant
    narrowing); a turn is only ever informative when its own name
    candidate(s) are compatible with EXACTLY ONE per_doctor entry — a
    turn that names two doctors from the set at once (an "either of
    them" offer, not a narrowing) never changes which one is "active".
    Returns None when no Agent turn ever narrows unambiguously to a
    single entry at all."""
    if not per_doctor:
        return None
    active: dict[str, Any] | None = None
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "agent" or is_agent_self_introduction(text):
            continue
        # Every DISTINCT per_doctor entry any candidate in THIS turn
        # matches — deliberately de-duplicated by identity (id()) before
        # counting, so the SAME doctor named twice in one turn (e.g. via
        # both a bare first name and a fuller name candidate) still counts
        # as ONE match, while genuinely offering "بدريه والطبيبه اميره"
        # together (two DIFFERENT doctors in the same turn) correctly
        # counts as two and is therefore NOT a narrowing at all.
        turn_matches: dict[int, dict[str, Any]] = {}
        for candidate in _doctor_name_candidates_in_text(text):
            candidate_tokens = _name_tokens(candidate)
            if not candidate_tokens:
                continue
            for d in per_doctor:
                input_tokens = _name_tokens(str(d.get("input_name") or ""))
                if input_tokens and _is_plausible_same_person_name(candidate_tokens, input_tokens):
                    turn_matches[id(d)] = d
        if len(turn_matches) == 1:
            active = next(iter(turn_matches.values()))
    return active


def validate_doctor_information(
    call: CallTranscript,
    doctor_records: list[dict[str, Any]],
    signals: tuple[bool, bool, str, str] | None = None,
    *,
    semantic_doctor_name: str | None = None,
    semantic_specialty_context: str | None = None,
) -> dict[str, Any]:
    """Top-level entry: resolve the named doctor(s) -> validate only the
    factual claims the Agent actually made against the authoritative CRM
    record(s). Mirrors validate_ksa_bank_information()/
    validate_location_request()'s structure and terminal-logging style.

    semantic_doctor_name / semantic_specialty_context (both default None,
    preserving byte-for-byte the original behaviour for every existing
    caller/test that doesn't pass them): an already-vetted human doctor
    name and specialty/service context, produced by the semantic
    conversation-extraction step in app.agent.nodes (see
    extract_doctor_semantic_context there) — the SAME quality bar the
    appointment-extraction node already achieves, applied here without
    reusing or changing that flow itself. When semantic_doctor_name is
    non-empty AND this call resolves to the SINGLE-doctor path (see below),
    it REPLACES the raw regex-derived candidate list as the query used for
    CRM resolution (with single-token partial matching allowed — see
    resolve_doctor_candidates), and semantic_specialty_context (when given)
    replaces the regex-derived doctor_context_specialty. This never
    happens for a genuine multi-doctor recommendation SET (2+ candidates)
    — that path is left entirely to the existing deterministic mechanism,
    unchanged. When no semantic name is available (e.g. the LLM call
    failed, or returned nothing usable), this function behaves exactly as
    it always has — the raw candidate list still gets a chance to resolve,
    never silently downgraded to NOT_APPLICABLE just because the semantic
    layer had nothing to add.

    Supports TWO shapes, decided by classify_specific_doctor_intent's
    "named_doctor_candidates":
      - Single doctor (0 or 1 candidate) — the original, unchanged
        behaviour: one _result()-shaped dict, doctor_resolved/doctor_key/
        scope_reference/etc. all describing that one doctor.
      - A genuine recommendation SET (2+ candidates — see classify_
        specific_doctor_intent's docstring on multi-doctor recommendation
        lists) — every named doctor is resolved and validated
        INDEPENDENTLY against the SAME already-fetched/deduplicated/BU-
        filtered pool (the CRM dataset is fetched once per call regardless
        of how many doctors are being evaluated — see
        _resolve_and_validate_one_doctor). The returned dict adds a
        "doctors" list (one _result()-shaped entry per recommended name,
        in the SAME order — an unresolved name stays explicitly
        unresolved, never dropped) plus an aggregated top-level outcome/
        reason/is_violation (see _aggregate_doctor_recommendation_
        outcomes) — the first doctor's outcome never silently stands in
        for the whole set. Top-level scalar fields (doctor_key,
        doctor_name_ar, scope_reference, ...) mirror the FIRST doctor for
        backward compatibility with single-doctor consumers;
        multi-doctor-aware code should read "doctors" instead.
    """
    call_id = call.call_id
    patient_mentions, agent_mentions, patient_text, agent_text = (
        signals if signals is not None else detect_doctor_signals(call)
    )
    _doc_print(f"Doctor validation started | call_id={call_id}")
    _doc_print(f"intent: patient_mentions_doctor={patient_mentions} agent_mentions_doctor={agent_mentions}")
    _intent_ctx = classify_specific_doctor_intent(call)
    if _intent_ctx["doctor_intent"] == "not_applicable":
        _doc_print_section(
            "intent classification", doctor_intent="NOT_APPLICABLE",
            doctor_role=_intent_ctx["doctor_role"], booking_target=_intent_ctx["booking_target"],
            inquiry_target=_intent_ctx["inquiry_target"], named_doctor=_intent_ctx["named_doctor"],
            reason=_intent_ctx["reason"],
        )
        _doc_print()
        _doc_print("outcome: NOT_APPLICABLE")
        return _result(
            "NOT_APPLICABLE", "No named-doctor mention or recommendation was detected.",
            call=call, context_specialty=_intent_ctx.get("doctor_context_specialty"),
        )

    if not doctor_records:
        _doc_print()
        _doc_print("outcome: INSUFFICIENT_REFERENCE_DATA")
        return _result(
            "INSUFFICIENT_REFERENCE_DATA",
            "Authoritative CRM doctor data is unavailable; no agent violation was inferred.",
            applicable=True, call=call, context_specialty=_intent_ctx.get("doctor_context_specialty"),
        )

    # ── Shared setup: fetched/deduplicated/pool-built exactly ONCE per
    # call, regardless of whether one doctor or a whole recommendation set
    # gets resolved below (never re-fetch/re-filter CRM per doctor). ──
    full_pool = dedupe_doctors(doctor_records)
    authoritative_pool = [r for r in full_pool if _is_active(r) and is_supported_doctor_bu(r.get("cr18c_buname"))]
    logger.info(
        "doctor CRM records | call_id=%s fetched=%d deduplicated=%d authoritative=%d opd_filtering=disabled",
        call_id, len(doctor_records), len(full_pool), len(authoritative_pool),
    )
    _doc_print()
    _doc_print(f"CRM records fetched: {len(doctor_records)}")
    _doc_print(f"records after deduplication: {len(full_pool)}")
    _doc_print(f"OPD filtering: disabled (opd_flag is informational only, not a searchability gate)")
    _doc_print(f"active supported-BU records considered: {len(authoritative_pool)}")

    # Message-by-message extraction (never the whole concatenated
    # transcript) — see extract_doctor_turn_candidates()'s docstring. Agent
    # candidates are tried FIRST (they are the source of the claims being
    # validated); Patient candidates are context used only when the Agent
    # itself named no one. Agent self-introductions ("مع حضرتك د محمد من
    # قسم ...") never contribute a candidate at all.
    patient_candidates, agent_candidates, ignored_self_intros = extract_doctor_turn_candidates(call)

    # Cross-turn linkage: classify_specific_doctor_intent() can resolve a
    # doctor named only in reply to an Agent's "which doctor?" clarifying
    # question (a bare name with no دكتور/طبيب title of its own, on a
    # DIFFERENT turn from the booking verb) — extract_doctor_turn_candidates
    # above only ever looks at title-bearing turns, so that cross-turn name
    # would otherwise never reach CRM resolution even though the gate
    # already confirmed doctor_intent is applicable. Reusing the SAME
    # classifier result the gate itself computed (_intent_ctx) rather than
    # re-deriving it keeps this a pure addition, never a second detector.
    _cross_turn_name = _intent_ctx.get("named_doctor")
    if _cross_turn_name and _cross_turn_name not in patient_candidates and _cross_turn_name not in agent_candidates:
        patient_candidates = [*patient_candidates, _cross_turn_name]

    if patient_candidates or agent_candidates or ignored_self_intros:
        _doc_print_section(
            "extracted doctor evidence",
            patient_candidates=", ".join(patient_candidates) or None,
            agent_candidates=", ".join(agent_candidates) or None,
            ignored_agent_self_introductions=", ".join(ignored_self_intros) or None,
        )

    # Business-Unit-scoped resolution FIRST: the call's already-detected BU
    # (CallTranscript.business_unit — the same authoritative signal
    # verify_appointment_in_db uses) is a reliable, independent hint of
    # which physical location this conversation belongs to. When it matches
    # a candidate, prefer that candidate over the general multi-BU pool —
    # this is what correctly picks the AHJ record for "محمد الألفي" when the
    # call is known to be AHJ, and is also what prevents the "same physical
    # doctor exists under two different doctor keys at different BUs"
    # real-data quirk (see README_doctor.md) from ever reaching
    # AMBIGUOUS_DOCTOR when the call's own BU already disambiguates it.
    # Never narrows resolution when the call's BU is unknown, or when the
    # BU-scoped search finds nothing — it always falls through to the
    # existing full-pool search below, so this can only ever RESOLVE a case
    # the old logic left ambiguous, never break a case the old logic
    # already resolved correctly.
    #
    # canonical_doctor_bu() runs FIRST: the conversation's own business_unit
    # may carry an application-level alias (e.g. "LIVE") that never
    # literally appears on a CRM doctor row (cr18c_buname="AHJ" there) —
    # comparing the raw alias against CRM would silently find zero BU-
    # scoped candidates and fall through to the unscoped authoritative
    # pool every time. All BU comparisons below (bu_scoped_pool, the
    # tie-break in _resolve_and_validate_one_doctor) use ONLY the
    # canonical, CRM-comparable value from here on.
    #
    # Matching is against EITHER of a candidate's two CRM BU fields
    # (cr301_businessunitname OR cr18c_buname — see
    # _match_doctor_business_unit), not cr18c_buname alone: a doctor row
    # can carry a BU on cr301_businessunitname that the call's own BU
    # signal matches even when it isn't one of the 7 codes
    # is_supported_doctor_bu recognises for record-eligibility purposes —
    # that eligibility gate already ran (authoritative_pool is built from
    # it) and is untouched here; this is purely which ALREADY-eligible
    # candidate the call's BU picks out.
    call_bu = getattr(call, "business_unit", None)
    canonical_bu = canonical_doctor_bu(call_bu)
    bu_scoped_pool: list[dict[str, Any]] = []
    if canonical_bu:
        bu_scoped_pool = [
            r for r in authoritative_pool
            if _match_doctor_business_unit(canonical_bu, r) is not None
        ]

    _resolve_kwargs: dict[str, Any] = dict(
        call=call, call_id=call_id, agent_text=agent_text, patient_text=patient_text,
        full_pool=full_pool, authoritative_pool=authoritative_pool, bu_scoped_pool=bu_scoped_pool,
        call_bu=canonical_bu, intent_ctx=_intent_ctx,
        patient_candidates=patient_candidates, agent_candidates=agent_candidates,
    )

    named_doctor_candidates = _intent_ctx.get("named_doctor_candidates") or []

    # Cross-turn identity enrichment (see _enrich_candidates_with_fuller_
    # agent_names' own module-level regression note): a candidate the
    # classifier resolved from an EARLIER turn (e.g. bare "اميره") is
    # superseded by a more complete, name-compatible mention found
    # anywhere else the Agent spoke in this same call (e.g. a LATER
    # "اميره بركات", given in reply to the Patient's own follow-up
    # question) — the Patient need never repeat the name themselves.
    # Sources: every Agent-turn candidate across the whole call
    # (deterministic, always available), PLUS the semantic LLM name ONLY
    # when it is independently GROUNDED in one of those same Agent
    # candidates (_semantic_name_is_grounded) — an ungrounded LLM name is
    # never added as an enrichment source at all, so it can never replace
    # real transcript evidence. This only ever replaces an EXISTING
    # candidate's string with a strictly fuller, compatible one; it never
    # changes how many candidates there are, so it is always safe to run
    # regardless of whether the single- or multi-doctor path is taken
    # below.
    _enrichment_sources = list(agent_candidates)
    _semantic_name_stripped = (semantic_doctor_name or "").strip()
    if _semantic_name_stripped and _semantic_name_is_grounded(_semantic_name_stripped, agent_candidates):
        _enrichment_sources.append(_semantic_name_stripped)
    named_doctor_candidates = _enrich_candidates_with_fuller_agent_names(
        named_doctor_candidates, _enrichment_sources,
    )

    if len(named_doctor_candidates) <= 1:
        # Single-doctor path — resolves against the FULL flat candidate
        # list exactly as before (never just the classifier's one scalar),
        # so no existing single-doctor resolution fallback behaviour
        # changes: this is byte-for-byte the same query list the old,
        # unrefactored function always tried.
        query_candidates = agent_candidates + patient_candidates
        allow_single_token = False
        _semantic_name = (semantic_doctor_name or "").strip()
        if _semantic_name:
            # The semantic layer already established this is a plausible
            # human doctor name — use it as the ONLY query (never appended
            # after the noisy regex candidates, which can still contain an
            # unrelated fragment ahead of it in list order) and allow a
            # single first-name token to reach CRM partial matching.
            query_candidates = [_semantic_name]
            allow_single_token = True
            _resolve_kwargs["intent_ctx"] = {
                **_intent_ctx,
                "doctor_context_specialty": (
                    semantic_specialty_context or _intent_ctx.get("doctor_context_specialty")
                ),
            }
        _single_result = _resolve_and_validate_one_doctor(
            query_candidates, allow_single_token=allow_single_token, **_resolve_kwargs
        )
        # Observability-only shape markers (see the multi-doctor branch's
        # identical fields below for the full rationale) — never consulted
        # by resolution/validation itself, purely additive so a caller can
        # tell "this result already IS one doctor" apart from "this is the
        # top-level mirror of a multi-doctor set" without inspecting
        # "doctors" first.
        _single_result["result_shape"] = "single_doctor"
        _single_result["doctor_count"] = 1 if _single_result.get("doctor_resolved") else 0
        _single_result["resolved_doctor_count"] = _single_result["doctor_count"]
        return _single_result

    # ── Multi-doctor recommendation set: resolve/validate EVERY recommended
    # name independently against the SAME pools built above — never stop
    # after the first match, and never let the first doctor's outcome
    # stand in for the whole set (see _aggregate_doctor_recommendation_
    # outcomes). "I will choose later"/no patient selection yet is exactly
    # the case that reaches here: named_doctor_candidates holds the full
    # recommendation set, not a single guessed target.
    _doc_print()
    _doc_print("multi-doctor resolution:")
    _doc_print(f"  requested={len(named_doctor_candidates)}")
    per_doctor: list[dict[str, Any]] = []
    for name in named_doctor_candidates:
        # allow_contextual_single_name=True ONLY here (the genuine multi-
        # doctor recommendation-SET path) — never for the single-doctor
        # path above, so a bare first name mentioned outside a real
        # recommendation set is completely unaffected (see _resolve_
        # single_token_candidate_with_context's own module-level comment).
        per_doctor.append(
            _resolve_and_validate_one_doctor([name], allow_contextual_single_name=True, **_resolve_kwargs),
        )

    resolved_count = sum(1 for d in per_doctor if d["doctor_resolved"])
    _doc_print(f"  resolved={resolved_count}")
    _doc_print(f"  unresolved={len(per_doctor) - resolved_count}")
    for name, d in zip(named_doctor_candidates, per_doctor):
        _doc_print()
        _doc_print("recommendation candidate:")
        _doc_print(f"  input_name={name}")
        _doc_print(f"  doctor_key={d.get('doctor_key')}")
        _doc_print(f"  business_unit={d.get('business_unit')}")
        _doc_print(f"  resolution_source={d.get('resolution_source')}")
        _doc_print(f"  validation_outcome={d.get('outcome')}")

    outcome, reason, is_violation = _aggregate_doctor_recommendation_outcomes(per_doctor)
    logger.info(
        "doctor recommendation set | call_id=%s requested=%d resolved=%d outcome=%s",
        call_id, len(named_doctor_candidates), resolved_count, outcome,
    )
    _doc_print()
    _doc_print(f"outcome: {outcome}")

    # Top-level scalar fields mirror the FIRST doctor for backward
    # compatibility (single-doctor consumers reading doctor_key/
    # scope_reference/etc. directly still get a sensible value) — EXCEPT
    # when the first candidate is unresolved while exactly ONE other
    # candidate in the set DID resolve: mirroring an unresolved "None"
    # doctor_key/business_unit/resolved_name at the top level while a
    # real, resolved doctor sits later in "doctors" is exactly the
    # misleading summary this generalises away (real regression: Amira
    # resolved cleanly, but every top-level log/routing signal still read
    # doctor_key=None/resolved_name=None because Badria — unresolved —
    # was per_doctor[0]). Falls back to per_doctor[0] whenever which
    # doctor is "the" active one is genuinely ambiguous (zero or 2+
    # resolved) — this never GUESSES; "doctors" remains the untouched,
    # authoritative, full per-candidate evidence either way (every
    # candidate, resolved or not, always stays in it — this only changes
    # which ONE entry's fields are additionally mirrored at the top level,
    # and is purely additive/observational, never a second resolution
    # decision).
    _resolved_in_set = [d for d in per_doctor if d.get("doctor_resolved")]
    if len(_resolved_in_set) == 1:
        _mirrored_doctor = _resolved_in_set[0]
        _doc_print(
            f"top-level summary mirrors the SOLE resolved doctor "
            f"({_mirrored_doctor.get('doctor_name_ar') or _mirrored_doctor.get('doctor_name_en')!r}), "
            f"not necessarily per_doctor[0] — see 'doctors' for the full, untouched per-candidate set.",
        )
    elif per_doctor:
        _mirrored_doctor = per_doctor[0]
        if len(_resolved_in_set) != 1:
            _doc_print(
                f"top-level summary mirrors per_doctor[0] ({len(_resolved_in_set)} of "
                f"{len(per_doctor)} resolved) — read 'doctors' for the authoritative per-candidate breakdown.",
            )
    else:
        _mirrored_doctor = {}
    base: dict[str, Any] = dict(_mirrored_doctor)
    base["outcome"] = outcome
    base["reason"] = reason
    base["is_violation"] = is_violation
    base["candidate_count"] = len(named_doctor_candidates)
    base["recommended_doctor_count"] = len(named_doctor_candidates)
    base["doctors"] = per_doctor
    # Observability-only shape markers — NEVER authoritative on their own;
    # "doctors" above remains the one authoritative per-candidate result
    # for logging/scope-routing/aggregation/persistence/serialization (see
    # this function's own docstring). These purely let a caller confirm
    # "this IS a genuine multi-doctor set" and get an accurate resolved
    # count without re-deriving it from "doctors" itself every time.
    base["result_shape"] = "multi_doctor"
    base["doctor_count"] = len(per_doctor)
    base["resolved_doctor_count"] = resolved_count
    # Failed/warning doctor counts (Part: doctor-validation results/
    # logging/persistence/UI presentation) — derived directly from each
    # per-doctor entry's own failure_details/warning_details, kept
    # strictly separate: warning_count NEVER contributes to failed_count
    # or to outcome/is_violation above (a doctor can be counted here in
    # warning_count while its own validation_outcome stays PASS).
    base["failed_count"] = sum(1 for d in per_doctor if d.get("failure_details"))
    base["warning_count"] = sum(1 for d in per_doctor if d.get("warning_details"))
    # Explicit ACTIVE-doctor identity (see _active_doctor_in_recommendation_
    # set's own docstring) — set ONLY when the conversation itself
    # unambiguously narrowed to one doctor AND that doctor actually
    # resolved; left unset (base.get(...) is None) otherwise, so a
    # consumer never has to guess whether an unset value means "no active
    # doctor" or "the active doctor's key happens to be None". Deliberately
    # separate fields from the generic top-level doctor_key/resolved_name
    # mirror above — those intentionally stay ambiguous/best-effort for
    # backward compatibility; these two are the unambiguous signal.
    _active_doctor = _active_doctor_in_recommendation_set(call, per_doctor)
    if _active_doctor and _active_doctor.get("doctor_resolved"):
        base["active_doctor_input_name"] = _active_doctor.get("input_name")
        base["active_doctor_key"] = _active_doctor.get("doctor_key")
    return base


# ── Applicability gate for the SEPARATE, LLM-based scope/recommendation
# validation (doctor_scope_validation is intentionally not merged into this
# deterministic module's output — see app.prompts.qa_prompt.
# build_doctor_scope_prompt() and app.agent.nodes.infer_doctor_scope_validation) ──
# Cheap, vocabulary-based: does the Patient describe a medical issue/
# symptom/desired treatment at all? This is NOT a diagnosis — it only
# decides whether the semantic step is worth running.
#
# IMPORTANT: bare ownership/existence phrases (عندي/عنده/عندها/بعاني/اعاني,
# English "I have"/"I feel") were REMOVED as standalone triggers — a real
# regression: "عندي موعد", "عندي ملف عندكم", "عندي جواز سفر", "عندي تأمين"
# all matched purely because of "عندي", producing a clinical-need false
# positive for calls that never described an actual medical problem. "عندي"
# alone is an ownership/existence expression, not medical evidence; it is
# only ever a legitimate signal when it co-occurs with one of the actual
# symptom/disease/procedure-with-pathology words below (e.g. "عندي وجع"
# still matches, via وجع — not via عندي).
#
# IMPORTANT (second): bare "مريض" ("patient"/"sick person") is NOT itself
# unconditional evidence — same regression class as "عندي" above, but
# handled differently: "مريض"/"المريض" is routinely used as a plain ROLE
# LABEL identifying WHO is being discussed, with no medical content of its
# own at all ("المريض كاش" = "the patient [will pay] cash", "المريض
# تأمين" = "...has insurance", "المريض جديد" = "...is a new patient",
# "المريض عنده ملف" = "...has a file with us") — a patient role label is
# not a diagnosis. Real diseases/conditions named right after "مريض" are
# open-ended and impossible to enumerate exhaustively (e.g. "مريض فشل
# كلوي" — kidney failure — a genuine, pre-existing regression this must
# keep recognising), so rather than requiring an explicit disease word
# after it (which would silently miss any condition not already listed
# below), only the SPECIFIC administrative/payment continuations reported
# as false positives are excluded via a negative lookahead immediately
# after "مريض" — every other continuation (a real disease name, or
# anything else) still counts, exactly as before. سكر/ربو are additionally
# listed below as their own standalone disease words (redundant with, but
# independent of, the "مريض" trigger) so "المريض عنده سكر"/"مريض ربو"
# keep matching even via a DIFFERENT sentence structure than "مريض
# <disease>" directly. Plain "مرض" ("disease/illness" as a bare noun, e.g.
# "عندي مرض") is UNCHANGED — that word names an illness on its own,
# unlike the "مريض" role/identity word.
_MEDICAL_COMPLAINT_RE = re.compile(
    r"(?<![\w])وجع(?![\w])|(?<![\w])الم(?![\w])|اصابه|إصابة|قطع\s*في|تمزق|كسر|حصو[ةه]|سرطان|"
    r"مريض(?!\s*(?:كاش|تأمين|تامين|جديد|عنده\s*ملف))|(?<![\w])مرض(?![\w])|حساسيه|حساسية|التهاب|صداع|دوخه|دوخة|سكري|سكر|ربو|"
    r"تأخر\s*(?:في\s*)?(?:ال)?نمو|تاخر\s*(?:في\s*)?(?:ال)?نمو|"
    r"تأخر\s*(?:في\s*)?(?:ال)?حمل|تاخر\s*(?:في\s*)?(?:ال)?حمل|"
    r"صرع|تشنج|"
    r"مشكل[ةه]\s*في|اعراض|أعراض|تشخيص|عايز\s*اعالج|محتاج\s*علاج|كشف\s*عاجل|"
    # Minimal English-language complaint indicators — the LLM's own
    # semantic comparison is expected to work across Arabic/English (see
    # qa_prompt.build_doctor_scope_prompt), so the applicability gate must
    # not silently block an English-language complaint from ever reaching
    # it. Deliberately generous, same philosophy as the Arabic patterns
    # above — a false positive here just means the LLM correctly returns
    # NOT_APPLICABLE/UNCLEAR. Bare "I have"/"I feel" are deliberately
    # excluded too, for the exact same reason عندي alone is — "I have an
    # appointment" is not a symptom.
    r"\bpain\b|\bache\b|\bsymptom|\binjury\b|\btorn\b|\bfracture\b|\bepilepsy\b|\bseizure",
    re.I,
)


def patient_describes_medical_complaint(patient_text: str) -> bool:
    """Cheap gate: does the Patient's text describe a medical issue/symptom/
    desired treatment (as opposed to e.g. a purely administrative question)?
    Deliberately generous — a false positive here just means the semantic
    LLM step runs and correctly returns NOT_APPLICABLE/UNCLEAR; a false
    negative would silently skip a check that should have run."""
    return bool(_MEDICAL_COMPLAINT_RE.search(normalize_arabic_text(patient_text or "")))


def extract_patient_clinical_need(call: CallTranscript) -> str:
    """Speaker- AND turn-aware extraction of the Patient's actual clinical
    need (symptom/disease/complaint/requested procedure) for the doctor
    scope-suitability prompt — never the whole concatenated Patient
    transcript, which routinely also contains unrelated booking/price/
    location turns that would dilute the LLM's semantic comparison with
    noise (and could never be confused with Agent explanations in the first
    place, since only Patient turns are considered here). Reuses the exact
    same per-turn vocabulary gate patient_describes_medical_complaint()
    already uses to decide whether scope validation applies at all, so a
    call whose gate passed is guaranteed to have at least one matching turn
    to extract here."""
    matches = [
        text for speaker, text in split_transcript_turns(call.transcript)
        if speaker == "patient" and patient_describes_medical_complaint(text)
    ]
    return "\n".join(matches)


# Conservative, explicit-only patient-age extraction for
# check_age_eligibility() below — requires an عمر/age keyword directly
# attached to a number; a bare "طفلي"/"ابني" never implies an age on its
# own, so ambiguous mentions correctly yield None rather than a guess.
_PATIENT_AGE_MONTHS_RE = re.compile(
    r"عمر(?:ه|ها|ي)?\s*(\d{1,3})\s*(?:شهر|شهور|أشهر|اشهر)", re.I,
)
_PATIENT_AGE_YEARS_RE = re.compile(
    r"عمر(?:ه|ها|ي)?\s*(\d{1,3})\s*(?:سن[ةه]|سنين|عام|أعوام|اعوام)?"
    r"|(\d{1,3})\s*(?:سن[ةه]|سنين|عام|أعوام|اعوام)"
    r"|age\D{0,5}(\d{1,3})",
    re.I,
)


def extract_patient_stated_age(patient_text: str) -> float | None:
    """Best-effort, deliberately conservative extraction of an explicitly
    patient-stated age (the patient's own age, or a family member's — e.g.
    a child being booked for) in YEARS. Used only by check_age_eligibility()
    to compare against a doctor's servhub_examinationage range — never to
    drive any other decision. Returns None (never a guess) whenever no
    number is directly attached to an عمر/age keyword."""
    if not patient_text:
        return None
    norm = normalize_arabic_text(patient_text)
    m = _PATIENT_AGE_MONTHS_RE.search(norm)
    if m:
        try:
            return round(float(m.group(1).translate(_ARABIC_DIGITS)) / 12, 2)
        except ValueError:
            pass
    m = _PATIENT_AGE_YEARS_RE.search(norm)
    if m:
        num = m.group(1) or m.group(2) or m.group(3)
        if num:
            try:
                return float(num.translate(_ARABIC_DIGITS))
            except ValueError:
                return None
    return None


def check_age_eligibility(patient_age: float | None, examination_age_text: str | None) -> str | None:
    """Deterministically compare an explicitly-stated patient age against a
    doctor's servhub_examinationage eligibility range (parse_examination_age).
    Returns 'within_range' / 'outside_range', or None when either input is
    unavailable/unparseable — this must never guess: a missing patient age
    should never count against the doctor (see the conservative
    age-validation requirement in infer_doctor_scope_validation)."""
    if patient_age is None:
        return None
    parsed = parse_examination_age(examination_age_text)
    if parsed is None:
        return None
    lo, hi = parsed
    return "within_range" if lo <= patient_age <= hi else "outside_range"


# ── Doctor-to-clinical-need RELATIONSHIP (Condition #4 of the scope gate) ──
# A resolved doctor and a genuine patient complaint can both be present in
# a call without the doctor ever being recommended/selected/booked FOR that
# complaint. Two distinct real regressions drove this:
#   1. "الدكتور احمد افندي طلب اشعه مقطعيه" + an unrelated, later "انا
#      مريض فشل كلوي" — he is the ORDERING/referring doctor for a CT scan;
#      a referring doctor's suitability for an unrelated condition was
#      never in question.
#   2. "ابغى احجز دكتور احمد" + a separate, later "انا مريض سكر" — the
#      PATIENT chose this specific doctor themselves; a coincidental,
#      unlinked complaint elsewhere in the call is not a request to judge
#      that doctor's fit for it. Scope suitability only becomes meaningful
#      here if the patient EXPLICITLY links the two (e.g. "هل دكتور احمد
#      مناسب لحالتي؟") — see classify_doctor_context()'s docstring.
#
# ── Specific-doctor-intent classification ───────────────────────────────────
# The applicability question is NOT "does a named doctor appear anywhere in
# this conversation" — it is "is the ACTIVE booking/inquiry TARGET the named
# doctor itself". A valid, resolvable doctor name is routinely present in a
# conversation that is really about something else entirely: the doctor who
# ORDERED a scan, an EXISTING follow-up appointment mentioned only as
# context while the fresh booking targets an unrelated service, or a
# question about the imaging/lab REQUEST rather than the doctor. Real
# regressions: "الدكتور احمد افندي طلب اشعه مقطعيه ، هل تطلع في التطبيق؟"
# (he is only the ordering physician for a CT request; the active target is
# the request/application, never his clinical suitability) and "عندنا
# مراجعة مع دكتور امير ... عايزة اعمل حجز اشعة رنين" (the booking targets
# an MRI; the doctor is only an existing/referring relationship).
#
# ORDERING-VERB markers — the doctor issued/wrote/referred something.
_ORDERING_VERB_MARKERS = {
    "طلب", "يطلب", "بيطلب", "طلبها", "طالب",
    "كتب", "حولني", "حولتني", "تحويل",
    "ordered", "requested", "referred",
}

# EXISTING-RELATIONSHIP markers — a past/ongoing appointment or follow-up
# WITH the doctor, stated possessively/statively ("عندنا حجز مع الدكتور",
# "كان عندي موعد مع الدكتور") rather than as an active request. This is
# what lets a LATER, bare rescheduling verb with no explicit target of its
# own ("وابغى اجل الموعد") correctly inherit the doctor as its target (see
# classify_specific_doctor_intent's clause loop) — but, critically, an
# ORDERING mention never seeds that inheritance, so an ordering-only doctor
# reference can never accidentally "leak" into a later, unrelated booking.
_EXISTING_RELATIONSHIP_RE = re.compile(
    r"عندنا\s*(?:حجز|موعد|مراجع[ةه])|عندي\s*(?:حجز|موعد|مراجع[ةه])|"
    r"كان\s*عندي\s*موعد|كان\s*عندنا\s*موعد|"
    r"i\s+have\s+an?\s+(?:follow-?up|appointment)\s+with",
    re.I,
)

# ACTIVE booking/rescheduling verbs — a genuine request/desire ("ابغى
# احجز", "اجل", "غير الموعد", "الغي"), never a bare noun ("حجز"/"موعد" on
# their own) or a completed/passive statement ("تم تأكيد حجزك"). This
# distinction is what keeps "عندنا حجز مع الدكتور" (existing, stative) from
# being misread as an active booking attempt.
#
# Every bare Arabic word-form alternative is wrapped in (?<![\w])...(?![\w])
# (a Unicode-safe word-boundary equivalent to \b for Arabic) — real
# regression: "من الممكن نقوم بزيارة طبيب..." matched bare "ممكن" as a
# SUBSTRING of the unrelated word "الممكن" ("the possible/available [thing]"
# — a subordinate-clause qualifier, not the patient/agent actively
# requesting anything), which then wrongly fed a doctor-booking-attachment
# match nearby. The same substring risk applies to "اجل"/"أجل" (inside
# "لأجل" = "for the sake of") and every other bare short word here, so all
# of them are boundaried the same way — never just the one reported word.
_ACTIVE_BOOKING_VERB_RE = re.compile(
    r"(?<![\w])(?:ابغى|أبغى|عايز[ةه]?|عاوز[ةه]?|محتاج[ةه]?|ممكن|"
    r"احجز|أحجز|احجزلي|أحجزلي|اجل|أجل)(?![\w])|"
    r"غير\s*(?:ال)?موعد|(?<![\w])الغ[يى](?![\w])|"
    r"\bi\s+want\b|\bi\s+need\b|\bi'd\s+like\b|(?<![\w])book(?![\w])|"
    r"\bschedule\b|\breschedule\b|\bcancel\b|\bmove\s+it\b|\bchange\s+(?:it|my)\b|"
    # "get an appointment" is an ACTIVE request for a NEW booking ("Can I
    # get an appointment with Dr X?"); a bare, unqualified "appointment"
    # is deliberately excluded — it also shows up in stative/passive
    # statements about an appointment that already exists ("Your
    # appointment with Dr X is confirmed"), which is not itself a booking
    # attempt.
    r"\bget\s+an?\s*appointment\b",
    re.I,
)

# The active booking verb's target is DIRECTLY the doctor: the verb and a
# دكتور/طبيب/د. title appear close together in the SAME clause (whether
# via an explicit "مع/عند" preposition — "احجز مع دكتور X" — or a bare
# direct object — "احجز دكتور X" — in either order), or "موعد الدكتور"
# names the doctor's OWN appointment as the grammatical object of a
# reschedule verb. A short proximity window (not the whole clause) keeps
# this from crossing into an unrelated part of a longer sentence.
# Same boundary treatment as _ACTIVE_BOOKING_VERB_RE above (and for the
# same reason — "ممكن" must never match as a substring of "الممكن"): every
# bare Arabic word-form alternative is wrapped in (?<![\w])...(?![\w]).
_BOOKING_VERB_FRAGMENT = (
    r"(?:(?<![\w])(?:ابغى|أبغى|عايز[ةه]?|عاوز[ةه]?|محتاج[ةه]?|ممكن|"
    r"احجز|أحجز|احجزلي|أحجزلي|اجل|أجل)(?![\w])|"
    r"غير\s*(?:ال)?موعد|(?<![\w])الغ[يى](?![\w])|want|need|book|schedule|reschedule|cancel|"
    r"get\s+an?\s*appointment)"
)
_DOCTOR_TITLE_FRAGMENT = r"(?:دكتور[ةه]?|طبيب[ةه]?|(?<![\w])د(?:[./\\-])?(?![\w])|dr\.?|doctor)"
_DOCTOR_BOOKING_ATTACH_RE = re.compile(
    rf"{_BOOKING_VERB_FRAGMENT}.{{0,20}}{_DOCTOR_TITLE_FRAGMENT}"
    rf"|{_DOCTOR_TITLE_FRAGMENT}.{{0,20}}{_BOOKING_VERB_FRAGMENT}"
    rf"|موعد\s*(?:ال)?دكتور[ةه]?|appointment\s+with\s+(?:dr\.?|doctor)",
    re.I,
)


def _title_is_role_reference(clause: str) -> bool:
    """True when a دكتور/طبيب title in *clause* is immediately followed by
    a specialty/role/generic word (e.g. "دكتور عظام", "دكتور مناسب") rather
    than a plausible person name or nothing at all — such a clause names a
    SPECIALTY or a generic request, not a person, and must never count as
    a doctor-attached booking/inquiry target even when a booking verb or
    attribute marker happens to appear nearby (real regression: "محتاج
    دكتور عظام" matched the verb-title proximity pattern purely because
    "محتاج" and "دكتور" are close together, with no person ever named). A
    BARE title with nothing following (e.g. "...موعد الدكتور" referring
    back to an already-established doctor) is NOT a role reference — only
    an actual blocked word right after the title is.

    English grammar fronts the specialty adjective BEFORE the noun
    ("orthopedic doctor", "female doctor") — the opposite order from
    Arabic's construct state ("دكتور عظام"). Real regression: "I need an
    orthopedic doctor." has nothing at all after "doctor" (just a period),
    so it was never caught as a role reference, and "need...doctor"
    matched the booking-verb proximity pattern regardless. The word
    immediately BEFORE the title is checked too, for exactly this
    English-order case."""
    m = _DOCTOR_TITLE_RE.search(clause)
    if not m:
        return False
    tail = clause[m.end():]
    stop = _NAME_STOP_RE.search(tail)
    raw = tail[:stop.start()] if stop else tail
    words = normalize_arabic_text(raw).split()
    if words and _is_non_name_word(words[0]):
        return True
    # Only a SPECIALTY/ROLE word immediately before the title counts here
    # (e.g. "orthopedic"/"female") — deliberately narrower than
    # _is_non_name_word's full blocklist, which also includes prepositions
    # like "مع"/"عند" that legitimately and correctly precede the title in
    # a normal "احجز مع دكتور محمد" attachment (that pattern must NOT be
    # misread as a role reference).
    preceding = normalize_arabic_text(clause[:m.start()]).split()
    if not preceding:
        return False
    last = preceding[-1]
    return bool({last, _strip_leading_al(last)} & _GENERIC_SPECIALTY_WORDS)

# Doctor-attribute markers — split by whether the attribute is a CLINICAL
# suitability/capability question (scope validation IS meaningful), a
# purely factual credential (specialty/degree — doctor-information
# validation, not scope), or an administrative fact (availability/branch/
# fee/schedule — identity/factual validation, never scope).
_DOCTOR_SUITABILITY_MARKERS = {
    "مناسب", "مناسبه", "يناسب", "تناسب",
    "يعالج", "تعالج", "بيعالج", "بتعالج", "يستقبل",
    "suitable", "appropriate", "recommend", "treat", "treats",
}
_DOCTOR_FACTUAL_MARKERS = {
    "تخصص", "تخصصه", "تخصصها", "استشاري", "اخصائي", "أخصائي", "خبره", "خبرة",
    "specialty", "consultant", "specialist",
}
_DOCTOR_ADMIN_MARKERS = {
    "موجود", "فرع", "مواعيد", "سعر", "كشفيه", "كشفية", "شغال",
    "متاح", "متاحه", "متاحة",
    "available", "branch", "fee", "price", "cost",
}

# The subset of _DOCTOR_SUITABILITY_MARKERS that is an explicit
# RECOMMENDATION verb rather than a bare suitability/treatment-capability
# adjective. The distinction matters only by WHO says it: a Patient asking
# "do you recommend Dr X?" is genuinely an inquiry (doctor_role=
# "doctor_inquiry" below); an Agent asserting "I recommend Dr X" /
# "أنصحك بدكتور X" is the Agent actively proposing a specific doctor to the
# patient — the same "agent_recommended" relationship the Agent-only-naming
# fallback further below already assigns, and must be classified the same
# way here too rather than falling through to "doctor_inquiry" just because
# "recommend" also happens to be a suitability-marker word.
_EXPLICIT_RECOMMENDATION_MARKERS = {"recommend", "suggest", "انصح", "انصحك", "ننصح", "نرشح", "رشح"}

# These same attribute words must ALSO never become (part of) a doctor
# NAME — real regression: "هل الدكتور أحمد يعالج الرباط الصليبي؟" was
# extracted as candidate name "احمد يعالج الرباط الصليبي" because
# _doctor_name_candidate's continuation-stop list didn't know about them.
# Retroactively folding them into _NON_NAME_WORDS (defined far earlier in
# this file) keeps _is_non_name_word() the single source of truth for
# "is this token part of a person's name" — both the first-word gate and
# the continuation cutoff pick up this update automatically, since Python
# looks up the module-level set at call time, not at the time
# _is_non_name_word/_doctor_name_candidate were defined.
_NON_NAME_WORDS.update(_DOCTOR_SUITABILITY_MARKERS | _DOCTOR_FACTUAL_MARKERS | _DOCTOR_ADMIN_MARKERS)

# Booking-CONFIRMATION markers — an already-completed/passive transaction
# ("تم تأكيد حجزك مع الدكتور..."), as opposed to a genuine recommendation
# proposed in response to a clinical need. Both can only ever be reached
# via the agent-recommendation FALLBACK path (the Patient never named a
# doctor, only the Agent did) — this distinction is what keeps a bare
# booking confirmation from being mislabelled "agent_recommended" and
# therefore incorrectly eligible for scope validation.
_BOOKING_CONFIRMATION_MARKERS = {"تاكيد", "أكد", "اكد", "confirmed"}

# "Which doctor?" clarifying question — an Agent turn asking the Patient to
# name a doctor after an earlier, still-unattached active booking verb
# ("ابغى احجز موعد" ... "مع دكتور مين؟"). See classify_specific_doctor_intent's
# cross-turn handling: the Patient's very next turn is then read as the
# answer even when it is a bare name with no دكتور/طبيب title at all.
#
# MUST explicitly mention دكتور/طبيب — a bare "مع مين؟"/"حاجز مع مين؟" is
# NOT doctor-specific at all (it could just as easily be "who is this
# appointment/call for" — a family member, a patient-identity check, etc.)
# and a real regression showed exactly that: a bare "مع مين؟" armed the
# cross-turn mechanism and turned the next Patient turn's name ("يوسف")
# into a fabricated doctor candidate. Requiring the title explicitly is
# what keeps this mechanism scoped to genuine "which DOCTOR" questions.
_WHICH_DOCTOR_QUESTION_RE = re.compile(
    r"(?:دكتور|طبيب)\s*(?:مين|أي|انهي|إيه)|مع\s*أي\s*(?:دكتور|طبيب)",
    re.I,
)

# Service/procedure nouns that make a booking/inquiry clause's target
# explicitly something OTHER than the doctor, even when a doctor happens to
# be named nearby — reuses the same imaging/procedure/lab vocabulary as
# _GENERIC_SPECIALTY_WORDS (see that set's own docstring for provenance).
_SERVICE_TARGET_WORDS = _GENERIC_SPECIALTY_WORDS

# Finer-grained service categories, purely for observability
# (booking_target/inquiry_target in logs) — never used to change the
# underlying True/False routing decision, only to describe it. Checked in
# this order (most specific first) since e.g. "منظار" could plausibly
# overlap with more than one bucket.
_RADIOLOGY_WORDS = {
    "اشعه", "أشعة", "رنين", "مقطعي", "مقطعيه", "مقطعية", "سونار", "ايكو", "إيكو", "تصوير",
    "ct", "mri", "scan", "ultrasound", "x-ray", "xray",
}
_LAB_WORDS = {"تحليل", "تحاليل", "مختبر", "lab", "labs"}
_HOME_CARE_WORDS = {"زياره", "زيارة", "منزليه", "منزلية"}
_PHYSIOTHERAPY_WORDS = {"طبيعي", "تاهيل", "تأهيل", "فيزيوثيرابي", "physiotherapy", "therapy"}
_PROCEDURE_WORDS = {"عمليه", "عمليات", "منظار", "قسطره", "قسطرة", "تنظير"}
_SERVICE_CATEGORY_BUCKETS = (
    ("radiology", _RADIOLOGY_WORDS),
    ("laboratory", _LAB_WORDS),
    ("home_care", _HOME_CARE_WORDS),
    ("physiotherapy", _PHYSIOTHERAPY_WORDS),
    ("procedure", _PROCEDURE_WORDS),
)


# The union of every service-shaped vocabulary this module recognises —
# _clause_has_service_noun (the boolean gate deciding whether a clause
# targets "a different service" at all) and _service_category (the finer
# label for that same clause) must draw from the exact same word set, or a
# clause could pass the gate but land in neither an explicit doctor target
# nor a resolvable category. Real regression: "أبغى أحجز زيارة منزلية" and
# "احجز موعد للتحليل" were falling through with booking_target=None because
# _clause_has_service_noun only checked _GENERIC_SPECIALTY_WORDS, never the
# newer home_care/laboratory/... buckets.
_ALL_SERVICE_WORDS = _SERVICE_TARGET_WORDS | _RADIOLOGY_WORDS | _LAB_WORDS | _HOME_CARE_WORDS | _PHYSIOTHERAPY_WORDS | _PROCEDURE_WORDS


def _turn_has_marker(text: str, markers: set[str]) -> bool:
    return bool(set(normalize_arabic_text(text or "").split()) & markers)


def _strip_common_prefixes(word: str) -> str:
    """Beyond _strip_leading_al's bare 'ال', Arabic also produces
    contracted prefixes like 'لل' (ل + ال, "for the") or 'بال'/'كال' (ب/ك +
    ال) on a service noun — real regression: "للتحليل" ("for the lab
    test") never matched "تحليل" because only a literal leading 'ال' was
    ever stripped. Reuses _strip_contracted_prefix (defined earlier
    alongside _is_non_name_word, which needs the exact same contraction
    handling for person-name blocklist checks) so the two never drift into
    separate prefix lists. Only used for service-vocabulary lookups here,
    never for person names."""
    contracted = _strip_contracted_prefix(word)
    return contracted if contracted != word else _strip_leading_al(word)


def _clause_has_service_noun(clause: str) -> bool:
    tokens = set(normalize_arabic_text(clause).split())
    if tokens & _ALL_SERVICE_WORDS:
        return True
    return any(_strip_common_prefixes(t) in _ALL_SERVICE_WORDS for t in tokens)


def _service_category(clause: str) -> str:
    """Best-effort, observability-only categorisation of a clause's service
    target (see _SERVICE_CATEGORY_BUCKETS) — falls back to the generic
    "service" label when a _GENERIC_SPECIALTY_WORDS term matched but none
    of the finer buckets did."""
    tokens = set(normalize_arabic_text(clause).split())
    tokens |= {_strip_common_prefixes(t) for t in tokens}
    for label, words in _SERVICE_CATEGORY_BUCKETS:
        if tokens & words:
            return label
    return "service"


# Arabic clause separators/conjunctions — splitting on these prevents an
# active booking verb in one clause from being read as "attached" to a
# doctor named only in a different, unrelated clause of the same turn (real
# regression: "عايزة اعمل حجز اشعة رنين ... لان عندنا مراجعة مع دكتور
# امير..." — the MRI booking verb must never attach to "دكتور امير", named
# only in the reason clause introduced by "لان").
#
# A line break is a clause boundary too — a real Agent turn sometimes lists
# more than one named doctor as separate lines within a single turn (e.g. a
# short set of options offered to the patient). Without this, "دكتور أحمد
# \nدكتور سالم" would be read as ONE clause and only the first title's
# attachment/role would ever be evaluated, silently dropping the second
# doctor from routing/candidate consideration entirely. Splitting on '\n'
# lets each line stand on its own for booking/inquiry/addressee/role
# attachment, exactly like a comma-separated clause already does.
_CLAUSE_SPLIT_RE = re.compile(r"[،,؛\n]|\bلان\b|\bلأن\b|\bعشان\b|\bعلشان\b|\bولان\b", re.I)


def _split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text or "") if c.strip()]


def _bare_reply_name_candidate(text: str) -> str | None:
    """Conservative name extraction for a Patient turn that is ITSELF the
    direct answer to an Agent's "which doctor?" clarifying question (see
    _WHICH_DOCTOR_QUESTION_RE) — such a reply may be a bare name with no
    دكتور/طبيب title at all (e.g. just "محمد الألفي"). Only ever consulted
    in that narrow, explicitly-anchored context inside
    classify_specific_doctor_intent — never as a general whole-transcript
    name-extraction rule, since a bare untitled turn is otherwise far too
    ambiguous to treat as naming a doctor."""
    words = normalize_arabic_text(text or "").split()[:_NAME_MAX_TOKENS]
    if not words or _is_non_name_word(words[0]) or _contains_digit(words[0]):
        return None
    for i, word in enumerate(words[1:], start=1):
        if _is_non_name_word(word) or word in _NAME_CONTINUATION_EXTRA_STOP_WORDS or _contains_digit(word):
            words = words[:i]
            break
    candidate = " ".join(words)
    return candidate if is_plausible_person_name(candidate) else None


def _classify_specific_doctor_intent_impl(call: CallTranscript) -> dict[str, Any]:
    """The SINGLE deterministic classifier for whether the ACTIVE
    conversational intent is specifically about a named doctor — reused by
    BOTH the graph router (_doctor_intent_router) and validate_doctor_node/
    validate_doctor_information's own defensive checks, so there is never a
    looser detector in one place and a stricter one in another.

    Not called directly outside this module — see
    classify_specific_doctor_intent(), the public, memoised entry point
    every real caller uses (this function is a pure function of
    call.transcript alone, so memoising on that text is safe and avoids
    re-running/re-logging the full extraction+classification pass every
    time the graph router, the doctor-scope router, and
    validate_doctor_information each ask for it on the same call).

    Returns doctor_intent as one of:
      "specific_doctor_booking" — the patient is booking/rescheduling/
          checking availability specifically WITH a named doctor (either
          because the patient explicitly attached the action to the
          doctor — "احجز مع دكتور X" / "غير موعد الدكتور" — or because the
          agent named a doctor after the patient asked for "a suitable
          doctor" with no name of their own).
      "specific_doctor_inquiry" — the patient is asking something
          specifically ABOUT a named doctor (specialty/degree, suitability/
          treatment capability, or availability/branch/fee).
      "not_applicable" — a named doctor may still appear (as the ordering/
          referring physician for a different service, or as an existing/
          follow-up relationship mentioned only as context), but the
          active booking/inquiry target is something else. No CRM fetch,
          no validate_doctor, no scope validation should ever happen here.

    Per-clause analysis (NOT whole-transcript keyword scanning): each turn
    is split into clauses (see _split_clauses) so an active booking verb is
    only ever read as attached to a doctor named in the SAME clause. A
    bare active-verb clause with no explicit target of its own (no doctor,
    no service noun) inherits the most recently ESTABLISHED target — but
    only an EXISTING-RELATIONSHIP clause (_EXISTING_RELATIONSHIP_RE, e.g.
    "كان عندي موعد مع الدكتور") ever establishes "doctor" as that target;
    an ORDERING-verb mention never does, so an ordering-only doctor
    reference can never leak into a later, unrelated active booking.

    doctor_role is one of: "patient_selected", "agent_recommended",
    "doctor_booking_confirmed", "doctor_inquiry", "ordering_or_referring",
    "existing_doctor_reference", "administrative_reference", or None.
    "doctor_booking_confirmed" is distinguished from "agent_recommended"
    because both are only ever reached via the same fallback (the Patient
    never named a doctor, only the Agent did) — but a bare transactional
    confirmation ("تم تأكيد حجزك مع الدكتور...") is not a clinical
    recommendation, so it must not become scope-eligible the way a genuine
    "مين دكتور مناسب؟" → Agent names one exchange is.

    scope_applicable is True only for "agent_recommended" or a
    "doctor_inquiry" driven by an explicit suitability/treatment-capability
    marker (_DOCTOR_SUITABILITY_MARKERS) — never for a bare factual/
    administrative inquiry, never for "doctor_booking_confirmed", and never
    for "patient_selected" alone (a coincidental, unlinked complaint
    elsewhere in the call is not a request to judge that doctor's fit for
    it).

    Cross-turn linkage: a booking verb and the doctor's name need not share
    a clause, or even a turn. Two mechanisms handle this without resorting
    to whole-transcript keyword scanning:
      1. last_established_target — a bare, targetless active-verb clause
         inherits "doctor" only from an EXISTING-RELATIONSHIP clause
         earlier in the same call (see the main loop below).
      2. The "which doctor?" exchange — an ambiguous active-verb clause
         with no target of its own (_PENDING_UNATTACHED_BOOKING), followed
         by an Agent clarifying question (_WHICH_DOCTOR_QUESTION_RE, e.g.
         "مع دكتور مين؟"), makes the very next Patient turn count as
         naming the doctor even when that reply is a bare name with no
         دكتور/طبيب title at all (see _bare_reply_name_candidate).
    """
    patient_candidates, agent_candidates, ignored_self_intros = extract_doctor_turn_candidates(call)
    turns = split_transcript_turns(call.transcript)

    doctor_intent = "not_applicable"
    doctor_role: str | None = None
    booking_target: str | None = None
    inquiry_target: str | None = None
    named_doctor: str | None = None
    has_suitability_signal = False
    saw_doctor_mention = False
    saw_ordering_marker = False
    saw_existing_marker = False
    saw_isolated_addressee = False  # title+name clause with nothing else, split off from a longer turn by punctuation
    last_established_target: str | None = None  # "doctor" | "service" | None
    pending_unattached_booking = False  # active verb seen with no doctor/service target yet
    awaiting_doctor_name_reply = False  # Agent just asked "which doctor?"
    # Multi-doctor recommendation support: an Agent turn that lists SEVERAL
    # bare "<title> <name>" clauses (no active verb, no service noun in any
    # of them — the exact same structural shape a single such clause
    # already gets read as an addressee/option, just repeated ≥2 times in
    # one turn) is a genuine recommendation SET, not one doctor repeated —
    # see the end of the per-turn loop below. recommended_doctors holds
    # ALL of them; patient_selected_doctor is set only if the Patient later
    # explicitly books/selects ONE specific name from that set (or makes
    # their own fresh single-doctor booking with no prior list at all).
    recommended_doctors: list[str] = []
    patient_selected_doctor: str | None = None

    for speaker, text in turns:
        if speaker == "agent" and is_agent_self_introduction(text):
            continue

        if speaker == "patient" and awaiting_doctor_name_reply and doctor_intent == "not_applicable":
            bare_cand = _bare_reply_name_candidate(text)
            if bare_cand:
                doctor_intent = "specific_doctor_booking"
                booking_target = "doctor"
                named_doctor = bare_cand
                doctor_role = "patient_selected"
                last_established_target = "doctor"
        awaiting_doctor_name_reply = False

        clauses = _split_clauses(text)
        turn_bare_name_candidates: list[str] = []  # bare "<title> <name>"-only clauses seen in THIS turn
        for clause in clauses:
            title_match = _DOCTOR_TITLE_RE.search(clause)
            has_title = bool(title_match)
            cand = _doctor_name_candidate(clause) if has_title else None
            is_existing = bool(_EXISTING_RELATIONSHIP_RE.search(clause))
            is_ordering = _turn_has_marker(clause, _ORDERING_VERB_MARKERS)
            has_active_verb = bool(_ACTIVE_BOOKING_VERB_RE.search(clause))
            has_service_noun = _clause_has_service_noun(clause)

            if has_title:
                saw_doctor_mention = True
                if is_existing:
                    saw_existing_marker = True
                    last_established_target = "doctor"
                elif is_ordering:
                    saw_ordering_marker = True
                elif (
                    not has_active_verb and not has_service_noun
                    and len(_doctor_name_candidates_in_text(clause)) >= 2
                ):
                    # TWO OR MORE doctors named via their own fused
                    # "<title> <name>" pair WITHIN THIS ONE CLAUSE — e.g.
                    # "الدكتورة بدريه والطبيبه اميره بركات" with no comma
                    # or other punctuation between the two names at all (an
                    # ordinary Arabic conjunction "و" fused directly onto
                    # the second title, "والطبيبه" — see _is_non_name_word/
                    # _DEGREE_TITLE_WORDS' own regression note on why that
                    # fused title must independently stop the FIRST name's
                    # extraction from running into the second). This is the
                    # SAME recommendation-SET structural signature as the
                    # multi-CLAUSE branch just below — a clause that
                    # independently names 2+ people via title-anchored
                    # candidates, with no active-verb/service content of
                    # its own, needs no further "nothing else register"
                    # check: naming multiple people this way IS the
                    # register. _doctor_name_candidates_in_text (not the
                    # single-result `cand`) is what finds every one of
                    # them independently — real regression: previously only
                    # `cand` (the FIRST occurrence) was ever captured here,
                    # and this whole branch never even ran at all for a
                    # single, comma-less clause (len(clauses) > 1 required
                    # below), so the SECOND doctor was silently dropped
                    # from named_doctor_candidates entirely rather than
                    # merely mis-extracted.
                    saw_isolated_addressee = True
                    for _clause_cand in _doctor_name_candidates_in_text(clause):
                        if _clause_cand not in turn_bare_name_candidates:
                            turn_bare_name_candidates.append(_clause_cand)
                elif (
                    cand is not None and len(clauses) > 1
                    and not has_active_verb and not has_service_noun
                ):
                    # This clause is essentially JUST "<title> <name>" —
                    # nothing else was said in it — and the turn as a whole
                    # had OTHER clauses too (split off by punctuation/
                    # conjunction). That is the structural signature of a
                    # direct address/vocative ("Dr Ahmed, is MRI available
                    # tomorrow?" → "Dr Ahmed" and "is MRI available
                    # tomorrow?" land in separate clauses): the named
                    # person is being ADDRESSED, and whatever is actually
                    # being asked/requested lives in a different clause
                    # entirely — never itself about this named person,
                    # since it never even mentions a title. Any clause that
                    # DOES independently establish a genuine doctor-
                    # targeted signal (booking/inquiry attachment, checked
                    # below) still takes priority over this default.
                    remaining = normalize_arabic_text(clause[title_match.end():]).split()
                    if len(remaining) <= len(cand.split()) + 1:
                        saw_isolated_addressee = True
                        if cand not in turn_bare_name_candidates:
                            turn_bare_name_candidates.append(cand)

            # ── Booking-target resolution (only while not yet decided) ──
            # A Patient turn making a genuine, resolved single-name booking
            # attachment is allowed to OVERRIDE a prior Agent recommendation
            # SET — "I will choose later" leaves nothing to override, but
            # "Book Dr B" after Dr A/B/C were recommended must narrow the
            # active target down to just Dr B, not stay stuck on the whole
            # list (see patient_selected_doctor below).
            allow_patient_selection_override = (
                speaker == "patient" and doctor_role == "agent_recommended" and cand is not None
            )
            if (doctor_intent != "specific_doctor_booking" or allow_patient_selection_override) and has_active_verb:
                doctor_attached = (
                    has_title and not _title_is_role_reference(clause)
                    and _DOCTOR_BOOKING_ATTACH_RE.search(clause) and not is_existing
                )
                if doctor_attached:
                    doctor_intent = "specific_doctor_booking"
                    booking_target = "doctor"
                    named_doctor = cand or named_doctor
                    if speaker == "patient":
                        doctor_role = "patient_selected"
                        patient_selected_doctor = cand or named_doctor
                    else:
                        doctor_role = "agent_recommended"
                    last_established_target = "doctor"
                    pending_unattached_booking = False
                elif has_service_noun:
                    booking_target = booking_target or _service_category(clause)
                    last_established_target = "service"
                    pending_unattached_booking = False
                elif not has_title and last_established_target == "doctor":
                    # Bare rescheduling verb with no target of its own —
                    # inherits the doctor established via an existing-
                    # relationship clause earlier in the SAME call (e.g.
                    # "كان عندي موعد مع الدكتور..." ... later "وابغى اجل
                    # الموعد لبكره"). Never inherits from an ordering-only
                    # mention (last_established_target is only ever set to
                    # "doctor" by an existing-relationship clause above).
                    doctor_intent = "specific_doctor_booking"
                    booking_target = "doctor"
                    doctor_role = "patient_selected" if speaker == "patient" else "agent_recommended"
                    pending_unattached_booking = False
                elif not has_title:
                    # A genuinely ambiguous active-verb clause with no
                    # target at all — may be resolved a turn or two later
                    # by an Agent's "which doctor?" clarifying question
                    # (see the awaiting_doctor_name_reply handling above).
                    pending_unattached_booking = True

            # ── Inquiry-target resolution (only a real name, never a bare
            # "دكتور مناسب؟" with nobody named — that falls through to the
            # agent-recommendation path below instead) ──
            if doctor_intent == "not_applicable" and has_title and cand is not None:
                clause_tokens = set(normalize_arabic_text(clause).split())
                if speaker == "agent" and clause_tokens & _EXPLICIT_RECOMMENDATION_MARKERS:
                    # The Agent is actively proposing this named doctor to
                    # the patient ("I recommend Dr X" / "أنصحك بدكتور X") —
                    # an assertion, not a question — so this is the same
                    # "agent_recommended" relationship the Agent-only-naming
                    # fallback further below assigns, not a bare inquiry.
                    doctor_intent = "specific_doctor_booking"
                    booking_target = "doctor"
                    named_doctor = cand
                    doctor_role = "agent_recommended"
                    has_suitability_signal = True
                    last_established_target = "doctor"
                elif clause_tokens & _DOCTOR_SUITABILITY_MARKERS:
                    doctor_intent = "specific_doctor_inquiry"
                    inquiry_target = "doctor"
                    named_doctor = cand
                    doctor_role = "doctor_inquiry"
                    has_suitability_signal = True
                elif clause_tokens & _DOCTOR_FACTUAL_MARKERS:
                    doctor_intent = "specific_doctor_inquiry"
                    inquiry_target = "doctor"
                    named_doctor = cand
                    doctor_role = "doctor_inquiry"
                elif clause_tokens & _DOCTOR_ADMIN_MARKERS:
                    doctor_intent = "specific_doctor_inquiry"
                    inquiry_target = "doctor"
                    named_doctor = cand
                    doctor_role = "administrative_reference"

        # ── Multi-doctor recommendation list (whole-turn, after all its
        # clauses have been scanned) — an Agent turn containing TWO OR MORE
        # bare "<title> <name>"-only clauses (each individually the same
        # structural shape a single isolated addressee/option clause
        # already has — see turn_bare_name_candidates above) is a genuine
        # recommendation SET, e.g.:
        #   "DR \ Mohamed Adel Belkadhi
        #    DR \ Ameer Elsayed
        #    DR \ ELSAYED SHAHEEN"
        # (one name per line — each line is its own clause, see
        # _split_clauses' newline handling) or a numbered/bulleted list, in
        # either Arabic or English. Never fires for a Patient turn (a
        # patient reciting several names is far rarer and structurally
        # ambiguous — this mechanism is deliberately scoped to genuine
        # Agent recommendations) and never overrides an already-decided
        # non-recommendation intent (ordering/existing/inquiry). A LATER
        # Patient turn can still narrow this down to one selected doctor
        # via the patient-override mechanism above.
        if (
            speaker == "agent" and doctor_intent != "specific_doctor_booking"
            and len(turn_bare_name_candidates) >= 2
        ):
            doctor_intent = "specific_doctor_booking"
            booking_target = "doctor"
            recommended_doctors = list(turn_bare_name_candidates)
            named_doctor = recommended_doctors[0]
            doctor_role = "agent_recommended"
            last_established_target = "doctor"
            pending_unattached_booking = False

        if (
            speaker == "agent" and pending_unattached_booking
            and doctor_intent == "not_applicable"
            and _WHICH_DOCTOR_QUESTION_RE.search(text)
        ):
            awaiting_doctor_name_reply = True

    # Agent-only-naming fallback: the Patient never named ANY doctor (only
    # asked for "a suitable doctor" generically, if anything), but the
    # Agent named a specific, resolvable one. Two sub-cases share this
    # fallback and must NOT be conflated:
    #   - "agent_recommended" — a genuine proposal (e.g. "مين دكتور
    #     مناسب؟" → "ممكن نحجز مع دكتورة خيرية محمد"). scope_applicable
    #     eligible.
    #   - "doctor_booking_confirmed" — the Agent is merely confirming an
    #     already-arranged booking ("تم تأكيد حجزك مع الدكتور...") — a
    #     transactional statement, not a clinical recommendation, so it
    #     must never become scope-eligible just because it happens to be
    #     the only doctor-naming turn in the call.
    if doctor_intent == "not_applicable" and not patient_candidates and agent_candidates:
        agent_naming_turns = [
            text for speaker, text in turns
            if speaker == "agent" and _doctor_name_candidate(text) is not None
            and not is_agent_self_introduction(text)
        ]
        agent_named_with_ordering = any(_turn_has_marker(t, _ORDERING_VERB_MARKERS) for t in agent_naming_turns)
        agent_named_with_confirmation = any(_turn_has_marker(t, _BOOKING_CONFIRMATION_MARKERS) for t in agent_naming_turns)
        if not agent_named_with_ordering:
            doctor_intent = "specific_doctor_booking"
            booking_target = "doctor"
            doctor_role = "doctor_booking_confirmed" if agent_named_with_confirmation else "agent_recommended"
            named_doctor = agent_candidates[0]

    if doctor_intent == "not_applicable" and saw_doctor_mention and doctor_role is None:
        if saw_ordering_marker:
            doctor_role = "ordering_or_referring"
        elif saw_existing_marker:
            doctor_role = "existing_doctor_reference"
        elif saw_isolated_addressee:
            doctor_role = "conversation_addressee"

    # No genuine doctor-targeted intent was ever found, and the ONLY
    # doctor-shaped name(s) anywhere in the call came from an Agent
    # introducing THEMSELVES ("Hello, this is Dr. Mohamed Ahmed from
    # Andalusia.") — label this distinctly for observability, per the
    # explicit requirement that agent self-introductions must always be
    # visibly excluded, not silently indistinguishable from "no doctor
    # mentioned at all". Never overrides a role already established by a
    # genuine, non-self-intro doctor reference elsewhere in the same call.
    if doctor_intent == "not_applicable" and doctor_role is None and ignored_self_intros:
        doctor_role = "agent_self_introduction"

    scope_applicable = doctor_role == "agent_recommended" or has_suitability_signal

    if doctor_intent == "not_applicable":
        reason = f"named doctor is only {doctor_role or 'mentioned in passing'}; active booking/inquiry target is a different service" if saw_doctor_mention or doctor_role == "agent_self_introduction" else "no named doctor detected"
    elif doctor_intent == "specific_doctor_booking":
        reason = "a booking/rescheduling action is directly attached to the named doctor"
    else:
        reason = "the question concerns the named doctor's specialty/suitability/availability/fee/branch"

    # The authoritative set of names CRM resolution should actually attempt
    # — never assumes only one doctor can be semantically relevant to the
    # turn. Priority: an explicit Patient selection narrows things down to
    # exactly that one doctor (even after a prior multi-doctor
    # recommendation); otherwise a genuine recommendation SET (2+ names in
    # one Agent turn) is resolved in full; otherwise the single scalar
    # named_doctor this classifier already tracked (unchanged behaviour for
    # every existing single-doctor case).
    if patient_selected_doctor:
        named_doctor_candidates = [patient_selected_doctor]
    elif recommended_doctors:
        named_doctor_candidates = list(recommended_doctors)
    elif named_doctor:
        named_doctor_candidates = [named_doctor]
    else:
        named_doctor_candidates = []

    return {
        "doctor_intent": doctor_intent,
        "doctor_role": doctor_role,
        "booking_target": booking_target,
        "inquiry_target": inquiry_target,
        "named_doctor": named_doctor,
        # Multi-doctor recommendation support (see this function's
        # docstring section on recommendation sets):
        #   named_doctor_candidates — every name CRM resolution should
        #     actually attempt for this turn (see priority above).
        #   recommended_doctors — every name the Agent proposed in a
        #     genuine recommendation-set turn, kept for history/
        #     observability even after a later Patient selection narrows
        #     the active target down to one.
        #   patient_selected_doctor — set only when the Patient explicitly
        #     picked/booked ONE specific name (from a recommendation set or
        #     a fresh single-doctor booking); "I will choose later" leaves
        #     this None, correctly signalling no selection was made yet.
        "named_doctor_candidates": named_doctor_candidates,
        "recommended_doctors": recommended_doctors,
        "patient_selected_doctor": patient_selected_doctor,
        "patient_candidates": patient_candidates,
        "agent_candidates": agent_candidates,
        "scope_applicable": scope_applicable,
        # Supporting evidence only — the specialty/field CONTEXT a doctor
        # recommendation was framed around in the conversation (e.g. "طبيب
        # مخ واعصاب" said just before naming the doctor), never the
        # doctor's authoritative CRM specialty and never the appointment
        # extractor's requested-service specialty — see
        # extract_doctor_context_specialty()'s docstring.
        "doctor_context_specialty": extract_doctor_context_specialty(call, named_doctor),
        "reason": reason,
    }


@functools.lru_cache(maxsize=512)
def _classify_specific_doctor_intent_cached(transcript: str) -> dict[str, Any]:
    return _classify_specific_doctor_intent_impl(_TranscriptOnly(transcript))


def classify_specific_doctor_intent(call: CallTranscript) -> dict[str, Any]:
    """Public entry point — memoised on the transcript TEXT (see
    _classify_specific_doctor_intent_impl for the full field set/semantics
    and the caching rationale). Returns a shallow copy (with its own
    mutable patient_candidates/agent_candidates list copies) on every call
    so a caller can never accidentally mutate the shared cached result."""
    cached = _classify_specific_doctor_intent_cached(call.transcript or "")
    result = dict(cached)
    result["patient_candidates"] = list(cached["patient_candidates"])
    result["agent_candidates"] = list(cached["agent_candidates"])
    return result


def classify_doctor_context(call: CallTranscript) -> dict[str, Any]:
    """Backward-compatible alias for classify_specific_doctor_intent() —
    kept so existing callers/tests that only need doctor_role/
    scope_applicable (the scope-gate-specific fields) don't need to change.
    See classify_specific_doctor_intent()'s docstring for the full field
    set and semantics."""
    return classify_specific_doctor_intent(call)


def doctor_is_selected_for_clinical_need(call: CallTranscript) -> bool:
    """Boolean entry point over classify_doctor_context() — Condition #4 of
    the scope-validation gate (see doctor_scope_skip_reason)."""
    return classify_doctor_context(call)["scope_applicable"]


def doctor_scope_skip_reason(
    doctor_result: dict[str, Any] | None,
    patient_text: str,
    call: CallTranscript | None = None,
) -> str | None:
    """Diagnostic companion to doctor_scope_validation_needed() — same four
    conditions, but returns WHY the gate fails ('no_resolved_doctor' /
    'no_scope_evidence' / 'no_patient_clinical_need' /
    'doctor_reference_is_ordering_or_referring_context'), or None when it
    passes. This is the SINGLE SOURCE OF TRUTH for the gate — both
    doctor_scope_validation_needed() below and app.agent.graph's
    _doctor_scope_intent_router (the graph-level conditional edge that
    decides whether infer_doctor_scope_validation even executes) call this
    same function, so the routing decision and the node's own defensive
    fallback can never drift apart:
      1. A doctor was successfully resolved deterministically (doctor_result
         is a PASS/FAIL from validate_doctor_information — never
         DOCTOR_UNRESOLVED/AMBIGUOUS_DOCTOR/NOT_APPLICABLE; the LLM must
         never independently guess which doctor was meant).
      2. Authoritative scope-of-service reference text is actually
         available for that doctor (scope, specialty, or subspecialty —
         at least one rung of the evidence hierarchy).
      3. The Patient described a genuine medical complaint/symptom/desired
         treatment at all — NOT merely a booking/administrative mention of
         "the doctor" (see _MEDICAL_COMPLAINT_RE's docstring: bare "عندي"
         is never enough on its own).
      4. The resolved doctor is contextually connected to that clinical
         need as a recommended/selected/newly-booked/proposed doctor, or a
         doctor whose suitability is being asked about — not merely an
         ordering/referring/existing doctor mentioned elsewhere in the same
         call as an unrelated complaint (see doctor_is_selected_for_
         clinical_need). `call` is optional ONLY for backward compatibility
         with callers that don't have a CallTranscript on hand (e.g. tests
         exercising conditions 1-3 in isolation) — both real call sites
         (app.agent.graph._doctor_scope_intent_router and
         app.agent.nodes.infer_doctor_scope_validation) always pass it.
    """
    if not doctor_result:
        return "no_resolved_doctor"

    # Multi-doctor recommendation set (see validate_doctor_information's
    # "doctors" list, 2+ entries): conditions 1/2 pass as long as AT LEAST
    # ONE recommended doctor independently resolved with usable scope
    # evidence — an unresolved name in the set must never block scope-
    # checking the doctors that DID resolve (infer_doctor_scope_validation
    # evaluates each resolved doctor independently either way).
    recommended = doctor_result.get("doctors") or []
    if len(recommended) > 1:
        any_resolved = any(d.get("doctor_resolved") for d in recommended)
        if not any_resolved:
            return "no_resolved_doctor"
        any_resolved_with_evidence = any(
            d.get("doctor_resolved") and d.get("outcome") in ("PASS", "FAIL")
            and any(
                (d.get("scope_reference") or {}).get(k)
                for k in ("scope_of_service", "scope_of_service_ar", "subspecialty", "specialty")
            )
            for d in recommended
        )
        if not any_resolved_with_evidence:
            return "no_scope_evidence"
    else:
        if not doctor_result.get("doctor_resolved"):
            return "no_resolved_doctor"
        if doctor_result.get("outcome") not in ("PASS", "FAIL"):
            return "no_resolved_doctor"
        scope_ref = doctor_result.get("scope_reference") or {}
        has_scope_evidence = any(
            scope_ref.get(k) for k in (
                "scope_of_service", "scope_of_service_ar", "subspecialty", "specialty",
            )
        )
        if not has_scope_evidence:
            return "no_scope_evidence"
    if not patient_describes_medical_complaint(patient_text):
        return "no_patient_clinical_need"
    if call is not None and not doctor_is_selected_for_clinical_need(call):
        return "doctor_reference_is_ordering_or_referring_context"
    return None


def doctor_scope_validation_needed(
    doctor_result: dict[str, Any] | None,
    patient_text: str,
    call: CallTranscript | None = None,
) -> bool:
    """The gate BOTH app.agent.graph's _doctor_scope_intent_router (graph-
    level conditional edge — decides whether infer_doctor_scope_validation
    executes AT ALL) and the node's own internal defensive fallback use
    before ever calling the LLM. See doctor_scope_skip_reason() for the
    four underlying conditions this reduces to a bool."""
    return doctor_scope_skip_reason(doctor_result, patient_text, call) is None