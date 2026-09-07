"""Deterministic, Arabic-first trigger detection, complaint-mapping, and
authoritative primary-doctor matching for COE (Center of Excellence)
validation.

COE validation as its own feature — see bank_validation.py / location_
validation.py / doctor_validation.py (same app/service_hub/ package) for the
sibling deterministic validators. Shares only the transcript-turn split and
Arabic text normalisation (app.services.text_helpers), not any bank/
location/doctor/COE-specific logic.

Flow (mirrors the project's "keep trigger detection separate from actual
validation" philosophy — see app.agent.nodes.infer_coe_validation):
    first, check for STRONG campaign/post-origin evidence (a structured
    campaign identifier containing "COE", e.g. "BU-AHJ-COE-...") anywhere
    in the Patient's own turns
        -> found -> campaign origin (Path C), TRIGGERED regardless of
           whether the Agent ever repeats COE language (see
           campaign_origin_evidence) — a bare "مركز تميز" mention alone is
           NEVER enough for this path; it requires the stronger, structured
           campaign-identifier evidence
    otherwise, scan the transcript, turn by turn, for a COE / specialized-
    center mention
        -> was the FIRST such mention made (or immediately preceded) by the
           Agent -> proactive recommendation (Path A)
        -> was it first raised by the Patient and then answered by the
           Agent -> customer inquiry (Path B)
        -> otherwise (only the Patient ever raised it, or nobody did) ->
           NOT triggered
        -> only when triggered: classify the patient's primary complaint,
           map it to the expected COE, extract the COE the Agent actually
           recommended/confirmed, and validate the initial doctor against
           the authoritative primary-doctor list for that COE. A campaign-
           origin message is marketing/system context: it is NEVER treated
           as something the human Agent said, wrote, or delivered (see
           campaign_origin_evidence and app.prompts.qa_prompt.
           build_coe_prompt's TRIGGER CONTEXT section).

A qualifying medical complaint alone (see COMPLAINT_KEYWORDS) is
deliberately NEVER sufficient to trigger this validator — only genuine
COE/specialized-center discussion is (see classify_coe_trigger).
"""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz as _rfuzz

from app.models.input import CallTranscript
from app.services.text_helpers import (
    normalize_arabic_text,
    split_transcript_turns,
    strip_html_tags,
)
# Reused for per-turn doctor-name candidate extraction ONLY (the exact same
# regex-based name-extraction engine app.service_hub.doctor_validation's own
# deterministic factual-information validator uses) — see
# extract_doctor_context_associations. Deliberately NOT re-implemented here:
# COE validation adds turn-index/COE-association bookkeeping on top of it,
# never a second copy of the name-extraction logic itself.
from app.service_hub.doctor_validation import _doctor_name_candidates_in_text

# ── Supported COEs ───────────────────────────────────────────────────────────
# Only these four COEs are ever validated — an unsupported COE/complaint must
# never be force-classified into one of these (see resolve_primary_complaint).
COE_KEYS: tuple[str, ...] = ("IBD", "Headache", "Asthma", "Diabetes")

FIRST_CLINIC: dict[str, str] = {
    "IBD": "Gastroenterology",
    "Headache": "Neurology",
    "Asthma": "Pulmonology",
    "Diabetes": "Diabetes/Endocrinology",
}

# Authoritative primary doctors for STARTING a new COE booking — explicitly
# hardcoded per business rules. Deliberately NEVER derived from CRM
# cr301_coemembers/cr301_clinicalleader/cr301_clinicalcoordinator: every
# doctor in Member, the clinic coordinator, every clinical leader, and
# doctors from supporting/secondary specialties are NOT automatically
# approved primary doctors (see resolve_primary_doctor_identity).
#
# Each approved doctor is represented as ONE canonical identity (the
# English spelling business rules already use — see AUTHORITATIVE_PRIMARY_
# DOCTORS below, derived from this) plus an explicit list of English and
# Arabic aliases that all identify that SAME doctor. This is deliberately
# an EXPLICIT alias table, not a fuzzy cross-script comparison: an Arabic
# transcript extraction (e.g. "اسامه عبد السلام") must never be compared
# directly against an English canonical name ("Osama Abdel Salam") via
# ordinary string similarity — see resolve_primary_doctor_identity's
# docstring for why, and normalize_doctor_name_for_match for the shared
# normalisation (titles, diacritics, tatweel, punctuation, case, alef/yeh
# variants, compound-name spacing) applied to every alias and every
# extracted candidate before comparison.
PRIMARY_DOCTOR_ALIASES: dict[str, dict[str, list[str]]] = {
    "IBD": {
        "Dalinda": [
            "Dalinda",
            "Dr. Dalinda",
            "د داليندا",
            "د. داليندا",
            "دكتورة داليندا",
            "دكتور داليندا",
            "داليندة",
            "د. داليندة",
            "دكتورة داليندة",
            # "عرفاوي" is a surname that appears attached to this SAME
            # configured Dalinda identity in some transcripts — an alias
            # of the existing canonical identity, never a second doctor
            # (see the module docstring's Dalinda-identity note).
            "داليندا عرفاوي",
            "دكتورة داليندا عرفاوي",
            "دكتور داليندا عرفاوي",
            "د. داليندا عرفاوي",
        ],
    },
    "Headache": {
        "Osama Abdel Salam": [
            "Osama Abdel Salam",
            "Osama Abdelsalam",
            "Osama Abdul Salam",
            "Dr. Osama Abdel Salam",
            "اسامة عبد السلام",
            "أسامة عبد السلام",
            "اسامه عبد السلام",
            "أسامه عبد السلام",
            "اسامة عبدالسلام",
            "أسامة عبدالسلام",
            "دكتور اسامة عبد السلام",
            "دكتور أسامة عبد السلام",
            "دكتور اسامه عبد السلام",
            "دكتور أسامه عبد السلام",
            "د اسامة عبدالسلام",
            "د أسامة عبدالسلام",
            "د اسامه عبدالسلام",
            "د أسامه عبدالسلام",
        ],
        "Abdelrhman Alshehri": [
            "Abdelrhman Alshehri",
            "Abdelrahman Alshehri",
            "Abdulrahman Alshehri",
            "Dr. Abdelrhman Alshehri",
            "عبدالرحمن الشهري",
            "عبد الرحمن الشهري",
            "دكتور عبدالرحمن الشهري",
            "دكتور عبد الرحمن الشهري",
            "د. عبدالرحمن الشهري",
            "د. عبد الرحمن الشهري",
            "د. عبد الرحمن الشهرى",
            "عبدالرحمن الشهرى",
            "عبد الرحمن الشهرى",
            "دكتور عبدالرحمن الشهرى",
        ],
        "Omar Ayoub": [
            "Omar Ayoub",
            "Omar Ayyoub",
            "Dr. Omar Ayoub",
            "عمر ايوب",
            "عمر أيوب",
            "دكتور عمر ايوب",
            "دكتور عمر أيوب",
            "د. عمر ايوب",
            "د. عمر أيوب",
            " عمر أيوب",
        ],
        "Abdulrahman Bogus": [
            "Abdulrahman Bogus",
            "Abdelrahman Bogus",
            "Abdulrahman Bogas",
            "Dr. Abdulrahman Bogus",
            "عبدالرحمن بوقس",
            "عبد الرحمن بوقس",
            "دكتور عبدالرحمن بوقس",
            "دكتور عبد الرحمن بوقس",
            "د. عبدالرحمن بوقس",
            "د. عبد الرحمن بوقس",
        ],
    },
    "Asthma": {
        "Nagwa Elhalawani": [
            "Nagwa Elhalawani",
            "Nagwa El Halawani",
            "Nagwa Alhalawani",
            "Dr. Nagwa Elhalawani",
            "نجوى الحلواني",
            "نجوي الحلواني",
            "دكتورة نجوى الحلواني",
            "دكتور نجوى الحلواني",
            "د. نجوى الحلواني",
            "د. نجوي الحلوانى",
            " نجوي الحلوانى",
            "نجوى الحلوانى",
            "دكتورة نجوى الحلوانى",
            "دكتور نجوى الحلوانى",
        ],
        "Eid Elajmi": [
            "Eid Elajmi",
            "Eid El Ajmi",
            "Eid Alajmi",
            "Dr. Eid Elajmi",
            "عيد العجمي",
            "دكتور عيد العجمي",
            "د. عيد العجمي",
        ],
    },
    "Diabetes": {
        "Badri Bairuti": [
            "Badri Bairuti",
            "Badri Beiruti",
            "Badri Bairouty",
            "Dr. Badri Bairuti",
            "بدري بيروتي",
            "بدري البيروتي",
            "دكتور بدري بيروتي",
            "دكتور بدري البيروتي",
            "د. بدري بيروتي",
            "د. بدرى البيروتى",
            "بدري البيروتى",
            "بدرى بيروتى",

        ],
    },
}

# Backward/forward-compatible flat view — {coe: [canonical_name, ...]} —
# used for the `approved_primary_doctors` output field and applicability
# checks (e.g. "is this a supported COE key at all?"). Derived from
# PRIMARY_DOCTOR_ALIASES so the canonical name is defined in exactly ONE
# place; never maintained as a second, independent list.
AUTHORITATIVE_PRIMARY_DOCTORS: dict[str, list[str]] = {
    coe: list(doctors.keys()) for coe, doctors in PRIMARY_DOCTOR_ALIASES.items()
}

# Fallback canonical Arabic scripts — used only when the CRM COE reference
# (cr301_arabicscript, via crm_coe.fetch_coe_reference) is unavailable, so
# script-adherence detection still functions during a CRM outage. The CRM
# value remains the primary/authoritative source when present (see
# build_coe_reference).
DEFAULT_SCRIPTS_AR: dict[str, str] = {
    "IBD": (
        "لضمان تحقيق أقصى استفادة والوصول إلى تشخيص دقيق لحالتك من جميع الجوانب الطبية، "
        "سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الجهاز الهضمي، والذي يضم "
        "نخبة من أفضل الاستشاريين والأخصائيين في هذا المجال."
    ),
    "Headache": (
        "لضمان تحقيق أقصى استفادة والوصول إلى تشخيص دقيق لحالتك من جميع الجوانب الطبية، "
        "سيتم حجز موعد لحضرتك بمركز التميز المتخصص في تشخيص وعلاج الصداع، والذي يضم نخبة "
        "من أفضل الاستشاريين والأخصائيين في هذا المجال."
    ),
    "Asthma": (
        "لضمان تحقيق أقصى استفادة والوصول إلى تشخيص دقيق لحالتك من جميع الجوانب الطبية، "
        "سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض الصدر والجهاز التنفسي، "
        "والذي يضم نخبة من أفضل الاستشاريين والأخصائيين."
    ),
    "Diabetes": (
        "لضمان تحقيق أقصى استفادة والوصول إلى تشخيص دقيق لحالتك من جميع الجوانب الطبية، "
        "سيتم حجز موعد لحضرتك بمركز التميز المتخصص في علاج أمراض السكر والغدد الصماء، "
        "والذي يضم نخبة من أفضل الاستشاريين والأخصائيين."
    ),
}

# ── Complaint -> expected-COE keyword mapping (non-exhaustive, extensible —
# same open-vocabulary philosophy as doctor_validation.py's blocklists) ─────
COMPLAINT_KEYWORDS: dict[str, set[str]] = {
    "IBD": {
        "الجهاز الهضمي", "جهاز هضمي", "القولون", "قولون", "الامعاء", "امعاء",
        "كرون", "التهاب القولون", "قرحة", "قرحه", "اسهال مزمن", "اسهال",
        "إسهال", "امساك مزمن", "امساك", "إمساك", "نزيف معوي", "الجهاز الهضمى",
        "كبد", "الكبد", "أمراض الكبد", "امراض الكبد",
        "gastroenterology", "gastrointestinal", "ibd", "crohn", "colitis",
        "digestive", "bowel", "stomach ulcer", "liver",
    },
    "Headache": {
        "صداع", "الصداع", "صداع نصفي", "صداع نصفى", "شقيقة", "شقيقه",
        "headache", "migraine",
    },
    "Asthma": {
        "ربو", "الربو", "امراض الصدر", "أمراض الصدر", "الجهاز التنفسي",
        "جهاز تنفسي", "ضيق تنفس", "ضيق في التنفس", "صدرية", "صدريه",
        "asthma", "chest disease", "respiratory", "pulmonary", "shortness of breath",
    },
    "Diabetes": {
        "السكر", "سكري", "السكري", "مرض السكر", "الغدد الصماء", "غدد صماء","غدة","الغدة",
        "الغدد", "diabetes", "diabetic", "endocrine", "endocrinology",
    },
}

# ── COE-name recognition markers (used to tell WHICH COE the Agent actually
# recommended/confirmed — deliberately more specific than the bare trigger
# phrases below, so a generic "مركز التميز" mention alone still falls back
# to script-similarity matching rather than a wrong specialty guess) ────────
COE_NAME_MARKERS: dict[str, set[str]] = {
    "IBD": {"الجهاز الهضمي", "امراض الجهاز الهضمي", "gastroenterology", "ibd"},
    "Headache": {"تشخيص وعلاج الصداع", "علاج الصداع", "الصداع", "headache"},
    "Asthma": {"امراض الصدر والجهاز التنفسي", "امراض الصدر", "الصدر والجهاز التنفسي", "asthma"},
    "Diabetes": {"امراض السكر والغدد الصماء", "السكر والغدد الصماء", "diabetes"},
}

# ═════════════════════════════════════════════════════════════════════════
# Centralized specialty taxonomy — the SINGLE source of truth for every
# specialty-aware decision in this module: specialty normalisation,
# COE-context detection, complaint/service classification, doctor-to-
# context association, prompt construction, deterministic safeguards, and
# tests. This reflects the confirmed ORGANIZATIONAL COE taxonomy, which is
# deliberately broader than narrow clinical definitions (e.g. Cardiology
# and Dental count as Headache-supporting specialties here because that is
# how this business groups referral/supporting specialties under each
# COE — see the "CONTEXT ASSOCIATION IS NOT PRIMARY-DOCTOR APPROVAL"
# section below: belonging to a COE's specialty list only means a
# specialty CAN support that COE's context, never that every doctor in it
# is an approved COE primary doctor).
# ═════════════════════════════════════════════════════════════════════════

# COE -> the canonical specialties organizationally grouped under it.
# "ENT" deliberately appears under BOTH Headache and Asthma — a genuinely
# SHARED specialty that must never be arbitrarily resolved to one COE on
# its own (see resolve_specialty_coes / the disambiguation priority order
# documented on detect_specialty_mentions).
COE_SPECIALTIES: dict[str, list[str]] = {
    "IBD": ["GIT", "Nutrition", "General Surgery"],
    "Headache": ["Neurology", "Ophthalmology", "ENT", "Cardiology", "Psychiatry", "Dental"],
    "Diabetes": ["Diabetes", "Diabetic Educator", "Orthopedics"],
    "Asthma": ["Pulmonology", "ENT", "Allergy & Immunology"],
}

# Canonical specialty -> English/Arabic aliases (common spelling/spacing/
# transliteration variants). Matched phrase-aware (see _contains_phrase),
# never via naive substring containment — several of these aliases are
# short enough (e.g. "قلب", "كبد", "سكر") that blind substring matching
# could false-positive inside an unrelated longer word (e.g. "قلب" inside
# "انقلاب").
SPECIALTY_ALIASES: dict[str, list[str]] = {
    "GIT": [
        "GIT", "Gastroenterology", "Gastrointestinal", "Digestive system", "Digestive",
        "الجهاز الهضمي", "جهاز هضمي", "أمراض الجهاز الهضمي", "امراض الجهاز الهضمي",
        "كبد", "الكبد", "أمراض الكبد", "امراض الكبد", "قولون", "القولون",
    ],
    "Nutrition": ["Nutrition", "Clinical Nutrition", "تغذية", "التغذية", "تغذية علاجية"],
    "General Surgery": ["General Surgery", "جراحة عامة", "الجراحة العامة"],
    "Neurology": [
        "Neurology", "Neurological", "مخ وأعصاب", "مخ واعصاب", "المخ والأعصاب", "المخ والاعصاب",
        "مخ و اعصاب", "مخ و أعصاب", "اعصاب", "أعصاب", "الأعصاب", "الاعصاب",
    ],
    "Ophthalmology": ["Ophthalmology", "طب العيون", "عيون"],
    "ENT": [
        "ENT", "Ear, Nose and Throat", "Otolaryngology", "أنف وأذن وحنجرة", "انف واذن وحنجرة",
    ],
    "Cardiology": ["Cardiology", "قلب", "القلب", "طب القلب"],
    "Psychiatry": ["Psychiatry", "طب نفسي", "نفسي", "الصحة النفسية"],
    "Dental": ["Dental", "Dentistry", "أسنان", "اسنان", "طب الأسنان"],
    "Diabetes": ["Diabetes", "Diabetology", "سكري", "السكري", "سكر", "مرض السكر"],
    "Diabetic Educator": [
        "Diabetic Educator", "Diabetes Educator", "مثقف سكري", "مثقفة سكري",
        "تثقيف سكري", "التثقيف السكري",
    ],
    "Orthopedics": ["Orthopedics", "Orthopedic", "عظام", "العظام", "جراحة العظام"],
    "Pulmonology": [
        "Pulmonology", "Pulmonary", "Respiratory", "Chest",
        "صدر", "صدرية", "أمراض الصدر", "امراض الصدر", "الجهاز التنفسي", "جهاز تنفسي",
    ],
    "Allergy & Immunology": [
        "Allergy & Immunology", "Allergy and Immunology", "Allergy", "Immunology",
        "حساسية ومناعة", "الحساسية والمناعة", "حساسية", "مناعة",
    ],
}

# WEAK specialty evidence — a term that is only meaningfully connected to
# a specialty when corroborated (either an existing context for that
# specialty's COE is already active, or a STRONG alias for the same
# specialty co-occurs) — never sufficient, by itself, to create a context
# on its own (see detect_specialty_mentions / build_coe_contexts).
# "مناظير"/"منظار" (endoscopy/scopes) is ambiguous on its own — it is only
# GIT/IBD evidence when connected to actual gastroenterology or liver
# context.
WEAK_SPECIALTY_ALIASES: dict[str, list[str]] = {
    "GIT": ["مناظير", "منظار"],
}


def _build_specialty_to_coes() -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for coe, specialties in COE_SPECIALTIES.items():
        for specialty in specialties:
            mapping.setdefault(specialty, [])
            if coe not in mapping[specialty]:
                mapping[specialty].append(coe)
    return mapping


# Canonical specialty -> every COE it organizationally supports. Length 1
# for an unambiguous specialty (e.g. "Neurology" -> ["Headache"]), length
# 2+ for a genuinely SHARED specialty (e.g. "ENT" -> ["Headache",
# "Asthma"]) that must be disambiguated, never guessed (see
# detect_specialty_mentions).
SPECIALTY_TO_COES: dict[str, list[str]] = _build_specialty_to_coes()


def _tokens(text: str | None) -> list[str]:
    return _norm(text).split()


def _contains_phrase(haystack_tokens: list[str], needle: str) -> bool:
    """True when *needle* (normalised and tokenised) appears as a
    CONTIGUOUS run of whole tokens inside haystack_tokens — phrase-aware
    matching that never lets a short alias (e.g. "قلب") false-positive
    merely because it is a SUBSTRING of an unrelated longer word (e.g.
    "انقلاب") the way naive `alias in text` containment would."""
    needle_tokens = _tokens(needle)
    if not needle_tokens:
        return False
    n = len(needle_tokens)
    return any(
        haystack_tokens[i:i + n] == needle_tokens
        for i in range(len(haystack_tokens) - n + 1)
    )


def resolve_canonical_specialty(text: str) -> str | None:
    """Deterministic, phrase-aware, longest-alias-wins resolution of the
    canonical specialty named in *text* (e.g. "المخ والاعصاب" -> "Neurology",
    "الكبد" -> "GIT"), or None when no specialty alias is present. See
    detect_specialty_mentions for the full set of specialties mentioned
    (this returns only the single best match)."""
    tokens = _tokens(text)
    if not tokens:
        return None
    best, best_len = None, 0
    for canonical, aliases in SPECIALTY_ALIASES.items():
        for alias in aliases:
            alias_tokens = _tokens(alias)
            if alias_tokens and _contains_phrase(tokens, alias) and len(alias_tokens) > best_len:
                best, best_len = canonical, len(alias_tokens)
    return best


def detect_specialty_mentions(text: str) -> list[tuple[str, list[str]]]:
    """Every canonical specialty with STRONG evidence in *text*, phrase-
    aware, as (canonical_specialty, candidate_coes) pairs — candidate_coes
    has length 1 for an unambiguous specialty, length 2+ for a SHARED one
    (e.g. "ENT" -> ["Headache", "Asthma"]).

    Deliberately excludes WEAK_SPECIALTY_ALIASES (e.g. "مناظير") — a weak
    term is never, by itself, sufficient evidence that a specialty was
    discussed; see detect_weak_specialty_mentions for the separate,
    corroboration-gated path build_coe_contexts uses for those.
    """
    tokens = _tokens(text)
    if not tokens:
        return []
    found: list[tuple[str, list[str]]] = []
    for canonical, aliases in SPECIALTY_ALIASES.items():
        if any(_contains_phrase(tokens, alias) for alias in aliases):
            found.append((canonical, SPECIALTY_TO_COES.get(canonical, [])))
    return found


def detect_weak_specialty_mentions(text: str) -> list[tuple[str, list[str]]]:
    """Every canonical specialty named ONLY via WEAK_SPECIALTY_ALIASES in
    *text* (e.g. "مناظير"/"منظار" -> GIT) — returned separately from
    detect_specialty_mentions because a weak term must NEVER create a
    context by itself; the caller (build_coe_contexts) only accepts it as
    corroborating evidence for a specialty/COE that is already otherwise
    established (a strong alias for the SAME specialty elsewhere in the
    conversation, or an already-active context for one of its candidate
    COEs) — never standalone."""
    tokens = _tokens(text)
    if not tokens:
        return []
    return [
        (canonical, SPECIALTY_TO_COES.get(canonical, []))
        for canonical, aliases in WEAK_SPECIALTY_ALIASES.items()
        if any(_contains_phrase(tokens, alias) for alias in aliases)
    ]


# Clinic/specialty synonyms that identify each COE's FIRST CLINIC
# contextually (e.g. "عيادة المخ والاعصاب" for Headache's Neurology
# clinic) — distinct from a disease-symptom complaint (COMPLAINT_KEYWORDS)
# and from an explicit COE-name recommendation (COE_NAME_MARKERS).
# Derived from the centralized SPECIALTY_ALIASES/COE_SPECIALTIES taxonomy
# above (union of every specialty's aliases organizationally grouped under
# each COE) — kept as its own name for the sibling functions below that
# still reason per-COE rather than per-specialty (CAMPAIGN_COE_CONTEXT_
# MARKERS, resolve_campaign_coe, ground_llm_coe_value). A SHARED specialty
# like ENT contributes to BOTH COEs' sets here (this is a permissive
# "is there SOME evidence" check, not an exclusive-attribution decision —
# see build_coe_contexts / extract_doctor_context_associations for the
# disambiguation-aware per-specialty logic that IS exclusive).
SPECIALTY_MARKERS: dict[str, set[str]] = {
    coe: {alias for specialty in specialties for alias in SPECIALTY_ALIASES.get(specialty, [])}
    for coe, specialties in COE_SPECIALTIES.items()
}

# Broader per-COE evidence vocabulary — union of the complaint keywords
# (what the PATIENT would say), the narrow COE-name markers (what an AGENT
# explicitly recommending the COE would say), and the clinic/specialty
# synonyms above. Used for:
#   1. resolve_campaign_coe — identifying which COE a campaign/post message
#      establishes from its own text.
#   2. ground_llm_coe_value / _agent_turn_supports_category — verifying
#      that an LLM-claimed recommended_coe has REAL supporting transcript
#      evidence in an Agent turn, never accepting a category merely because
#      it's one of the four the LLM was told about in reference data.
#   3. The single-scalar legacy _coe_matches_in_text path (kept for
#      backward compatibility only — see its docstring).
# Never used to deterministically SET recommended_coe on its own (see
# resolve_recommended_coe, which stays on the narrower COE_NAME_MARKERS),
# and never used by the multi-context builder for EXCLUSIVE attribution of
# a shared specialty (see detect_specialty_mentions / build_coe_contexts).
CAMPAIGN_COE_CONTEXT_MARKERS: dict[str, set[str]] = {
    key: set(COMPLAINT_KEYWORDS[key]) | set(COE_NAME_MARKERS[key]) | set(SPECIALTY_MARKERS[key])
    for key in COE_KEYS
}

# ── COE / specialized-center trigger phrases ────────────────────────────────
# Deliberately compound phrases only — a bare "مركز" (center/branch) must
# never trigger this validator on its own (see the module docstring / "do
# not trigger" rules). Matched against normalize_arabic_text()'s OUTPUT
# (diacritics stripped, أ/إ/آ->ا, ة->ه, lowercased, punctuation removed —
# see app.services.text_helpers), so patterns below use the post-
# normalisation spelling (e.g. "ه" not "ة").
_COE_MENTION_RE = re.compile(
    r"مركز\s*(?:ال)?تميز|"
    r"مركز\s*متخصص[ةه]?|"
    r"عياد[ةه]\s*متخصص[ةه]|"
    r"مركز\s*تخصصي|"
    r"مركز\s*لعلاج\s*(?:ال)?صداع|"
    r"مركز\s*لعلاج\s*(?:ال)?سكر[ي]?|"
    r"مركز\s*لامراض\s*(?:ال)?جهاز\s*(?:ال)?هضمي|"
    r"مركز\s*لامراض\s*(?:ال)?صدر|"
    r"مركز\s*متعدد\s*التخصصات|"
    r"center\s*of\s*excellence|specialized\s*cent(?:er|re)|specialised\s*cent(?:er|re)|"
    r"multidisciplinary\s*cent(?:er|re)",
    re.I,
)

# ── COE marketing-campaign/post origin marker ───────────────────────────────
# A conversation may begin with an automatically populated Patient message
# from clicking a COE ad/post (e.g. "BU-AHJ-COE- أضغطي علي أرسال..."). This
# is STRONGER, more specific evidence than a bare "مركز تميز" mention (see
# _COE_MENTION_RE above) — it is a structured campaign/ad identifier that
# contains "COE" as its own hyphen-delimited segment, never a bare "COE"
# floating in ordinary prose. Matched against the RAW (non-normalised) turn
# text, since normalize_arabic_text() would replace the identifier's
# hyphens with spaces and destroy its structure — a plain "مركز تميز"
# mention must NEVER satisfy this pattern (see campaign_origin_evidence's
# docstring and the module docstring's Path C).
_CAMPAIGN_ORIGIN_RE = re.compile(
    r"\b[A-Za-z0-9]{2,10}-[A-Za-z0-9]{2,10}-COE\b|\b[A-Za-z0-9]{2,10}-COE\b",
    re.I,
)

# ── Existing-patient / established-treating-doctor exception ───────────────
# Deliberately narrow: only phrases that name an ONGOING relationship with a
# SPECIFIC treating doctor count as evidence — a vague self-label like
# "مريض قديم" ("I'm an old/existing customer") is NOT included, since a
# patient can say that while still starting a brand-new COE journey (see
# module docstring's "new COE journey" carve-out) — see
# existing_patient_exception_evidence's docstring.
_EXISTING_PATIENT_RE = re.compile(
    r"بتابع\s*مع\s*(?:ال)?(?:دكتور|طبيب)|"
    r"متابع[ةه]\s*مع\s*(?:ال)?(?:دكتور|طبيب)|"
    r"من\s*زمان\s*بتابع\s*مع\s*(?:ال)?(?:دكتور|طبيب)|"
    r"(?:دكتور[يى]|طبيب[يى])\s*(?:ال)?معالج|"
    r"(?:ال)?طبيب\s*(?:ال)?معالج\s*(?:بتاعي|الخاص\s*بي|الخاص)|"
    r"existing\s*patient\s*(?:of|with)\s*dr|"
    r"follow[\s\-]?up\s*with\s*(?:my\s*)?(?:doctor|dr\.?)",
    re.I,
)

SCRIPT_MATCH_THRESHOLD = 65


def _norm(text: str | None) -> str:
    return normalize_arabic_text(text)


# ── COE reference building (CRM + safe fallback) ────────────────────────────

def build_coe_reference(coe_rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    """Build a {coe_key: {clinic, bu, specialty, leader, coordinator,
    members, script}} mapping from raw CRM rows (see coe_query.sql),
    handling missing rows, duplicate rows, null fields, and HTML content in
    Script_AR (the Asthma script may contain raw HTML — see
    app.services.text_helpers.strip_html_tags) defensively. Falls back to
    DEFAULT_SCRIPTS_AR for any COE whose CRM row is missing/malformed so
    script-adherence detection keeps working during a CRM outage or
    reference-data gap. Never raises.
    """
    reference: dict[str, dict[str, Any]] = {
        key: {
            "clinic_name": key,
            "first_clinic": FIRST_CLINIC[key],
            "business_unit": None,
            "specialty": None,
            "clinic_leader": None,
            "clinic_coordinator": None,
            "members": None,
            "script_ar": DEFAULT_SCRIPTS_AR[key],
        }
        for key in COE_KEYS
    }

    for row in coe_rows or []:
        if not isinstance(row, dict):
            continue
        clinic_name = str(row.get("Clinic_Name") or "").strip()
        if clinic_name not in reference:
            continue  # unsupported/malformed COE row — ignore, never crash
        entry = reference[clinic_name]
        entry["business_unit"] = row.get("BU") or entry["business_unit"]
        entry["specialty"] = row.get("Specialty") or entry["specialty"]
        entry["clinic_leader"] = row.get("Clinic_Leader") or entry["clinic_leader"]
        entry["clinic_coordinator"] = row.get("Clinic_Coordinator") or entry["clinic_coordinator"]
        entry["members"] = row.get("Member") or entry["members"]
        script_raw = row.get("Script_AR")
        if script_raw:
            # Strip HTML + decode entities defensively (Asthma's Script_AR is
            # known to sometimes carry raw HTML) before it's ever used for
            # semantic comparison or included in an LLM prompt.
            stripped = strip_html_tags(str(script_raw))
            if stripped:
                entry["script_ar"] = stripped
    return reference


def scripts_from_reference(reference: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {key: entry.get("script_ar") or DEFAULT_SCRIPTS_AR[key] for key, entry in reference.items()}


# ── Trigger detection (Path A / Path B) ─────────────────────────────────────

def detect_coe_mention(text: str) -> bool:
    """Cheap vocabulary gate: does this single turn mention a COE or a
    specialized/multidisciplinary center — never a bare generic "مركز"."""
    return bool(_COE_MENTION_RE.search(_norm(text)))


def campaign_origin_evidence(call: CallTranscript) -> str | None:
    """Deterministic evidence that this conversation began from a COE
    marketing post/campaign click — a STRUCTURED campaign/ad identifier
    containing "COE" (e.g. "BU-AHJ-COE-..."), never a bare "مركز تميز"
    mention (see _CAMPAIGN_ORIGIN_RE's docstring — that phrase alone is
    handled by the existing proactive_recommendation/customer_inquiry
    paths, not this one).

    Scanned on Patient turns only, on each turn's RAW (non-normalised)
    text — an auto-populated campaign message is persisted as the
    customer's own first message, but it is marketing/system content, not
    something the customer (or the human Agent) personally wrote; it is
    used only to ESTABLISH that the conversation is already in a COE
    context (see classify_coe_trigger's Path C and app.prompts.qa_prompt.
    build_coe_prompt's TRIGGER CONTEXT section, which is responsible for
    never attributing this wording to the Agent).
    """
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "patient":
            continue
        if _CAMPAIGN_ORIGIN_RE.search(text):
            return text.strip()[:300]
    return None


def resolve_campaign_coe(call: CallTranscript) -> str | None:
    """Identify which supported COE a campaign/post-origin message
    EXPLICITLY establishes, from that message's own text only — e.g.
    "مركز تميز الصداع" -> Headache, or neurology-clinic language such as
    "مخ واعصاب" -> Headache (see CAMPAIGN_COE_CONTEXT_MARKERS).

    Scanned ONLY on the Patient turn(s) that actually carry the structured
    campaign identifier (see campaign_origin_evidence) — never the whole
    transcript, and never an Agent turn: a campaign message is marketing/
    system content, not something the human agent wrote (see module
    docstring). This is the AUTHORITATIVE source for campaign_coe — an
    LLM's own opinion must never override it (see
    app.agent.nodes.infer_coe_validation).

    Returns None when no campaign marker is present, or when the campaign
    text itself contains no recognisable COE-identifying language — never
    guessed.
    """
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "patient":
            continue
        if not _CAMPAIGN_ORIGIN_RE.search(text):
            continue
        norm = _norm(text)
        for key, markers in CAMPAIGN_COE_CONTEXT_MARKERS.items():
            if any(_norm(m) in norm for m in markers):
                return key
    return None


def classify_coe_trigger(call: CallTranscript) -> dict[str, Any]:
    """Determine whether COE validation should run at all, and via which
    path — turn-order-aware and speaker-attributed, so a Patient statement
    is never misattributed as an Agent recommendation (see module
    docstring).

    Checks, in order:

      0. Campaign/post origin (Path C, "campaign_origin") — a STRUCTURED
         campaign identifier (see campaign_origin_evidence) found anywhere
         in the Patient's own turns. TRIGGERED immediately, regardless of
         whether the Agent ever mentions a COE at all — the campaign
         message already establishes the COE context on its own (see
         module docstring). Checked first because it does not depend on
         turn order the way Paths A/B do.

    Otherwise walks the transcript turn by turn (in original order) for an
    ordinary "مركز تميز"/specialized-center mention:

      - The FIRST turn (Patient or Agent) that mentions a COE/specialized
        center is found.
      - If an EARLIER Patient turn already raised it and a LATER Agent turn
        responds to/confirms it -> Path B ("customer_inquiry"), TRIGGERED.
      - If the FIRST such mention is made by the Agent (no prior Patient
        mention at all) -> Path A ("proactive_recommendation"), TRIGGERED.
      - If only the Patient ever mentions it (the Agent never responds/
        confirms) -> NOT triggered — a customer mention alone is never
        enough (see module docstring's "do not trigger" rules).
      - If nobody mentions it at all -> NOT triggered.
    """
    campaign_evidence = campaign_origin_evidence(call)
    if campaign_evidence:
        return {
            "triggered": True,
            "trigger_path": "campaign_origin",
            "trigger_reason": (
                "The customer's message contains a COE marketing-campaign/post identifier, "
                "establishing that this conversation began from a Center of Excellence "
                "campaign. This is a marketing/system message, not something the human agent "
                "wrote — it is never treated as an agent recommendation or script delivery."
            ),
            "evidence": campaign_evidence,
            "patient_evidence": campaign_evidence,
        }

    turns = split_transcript_turns(call.transcript)
    patient_raised = False
    patient_evidence: str | None = None

    for speaker, text in turns:
        if not detect_coe_mention(text):
            continue
        if speaker == "patient":
            if not patient_raised:
                patient_raised = True
                patient_evidence = text.strip()[:300]
            continue
        # speaker == "agent" and this turn mentions a COE/specialized center
        if patient_raised:
            return {
                "triggered": True,
                "trigger_path": "customer_inquiry",
                "trigger_reason": (
                    "The customer asked about a Center of Excellence / specialized center "
                    "and the agent responded to or confirmed that inquiry."
                ),
                "evidence": text.strip()[:300],
                "patient_evidence": patient_evidence,
            }
        return {
            "triggered": True,
            "trigger_path": "proactive_recommendation",
            "trigger_reason": (
                "The agent proactively recommended, introduced, or explained a Center of "
                "Excellence without the customer asking first."
            ),
            "evidence": text.strip()[:300],
            "patient_evidence": None,
        }

    if patient_raised:
        return {
            "triggered": False,
            "trigger_path": None,
            "trigger_reason": (
                "The customer mentioned a specialized center, but the agent never "
                "responded to or confirmed it."
            ),
            "evidence": None,
            "patient_evidence": patient_evidence,
        }
    return {
        "triggered": False,
        "trigger_path": None,
        "trigger_reason": "No Center of Excellence or specialized-center discussion was found.",
        "evidence": None,
        "patient_evidence": None,
    }


def coe_validation_needed(call: CallTranscript, trigger_ctx: dict[str, Any] | None = None) -> bool:
    """Graph-router-friendly boolean wrapper around classify_coe_trigger."""
    ctx = trigger_ctx if trigger_ctx is not None else classify_coe_trigger(call)
    return bool(ctx.get("triggered"))


# ── Primary-complaint -> expected-COE mapping ───────────────────────────────

def detect_complaint_categories(text: str) -> set[str]:
    """All COE complaint categories whose keywords appear in *text* — a
    single turn/utterance may legitimately match more than one."""
    norm = _norm(text)
    found: set[str] = set()
    for key, keywords in COMPLAINT_KEYWORDS.items():
        for kw in keywords:
            if _norm(kw) and _norm(kw) in norm:
                found.add(key)
                break
    return found


def resolve_primary_complaint(call: CallTranscript) -> tuple[str | None, list[str]]:
    """Identify the patient's PRIMARY complaint category (never guessed when
    ambiguous — see module docstring's "uncertain" rule).

    Returns (primary_category_or_None, all_categories_mentioned). When
    exactly one category is mentioned anywhere, it IS the primary complaint.
    When several distinct categories are mentioned, the last Patient turn
    that names exactly ONE category on its own is treated as the primary
    complaint (closest to the point of discussion/booking — same
    "emphasized/discussed most clearly" heuristic the business rules call
    for); if no such turn exists, the primary complaint is genuinely
    ambiguous and None is returned so the caller reports 'uncertain' rather
    than fabricating a choice.
    """
    turns = split_transcript_turns(call.transcript)
    patient_turns = [text for speaker, text in turns if speaker == "patient"]

    all_found: list[str] = []
    per_turn: list[set[str]] = []
    for text in patient_turns:
        cats = detect_complaint_categories(text)
        per_turn.append(cats)
        for c in cats:
            if c not in all_found:
                all_found.append(c)

    if not all_found:
        return None, []
    if len(all_found) == 1:
        return all_found[0], all_found

    for cats in reversed(per_turn):
        if len(cats) == 1:
            return next(iter(cats)), all_found
    return None, all_found


# ── Recommended-COE extraction (Agent turns only) ───────────────────────────

def script_similarity(agent_text: str, script_ar: str) -> float:
    """rapidfuzz token_set_ratio between normalised Agent text and a
    normalised approved script — tolerant of paraphrase/word-order/ASR
    differences while still requiring substantial shared vocabulary."""
    a, s = _norm(agent_text), _norm(script_ar)
    if not a or not s:
        return 0.0
    return _rfuzz.token_set_ratio(s, a)


def resolve_recommended_coe(call: CallTranscript, scripts: dict[str, str] | None = None) -> str | None:
    """Identify which supported COE the AGENT actually recommended or
    confirmed — Patient turns are never consulted here (see module
    docstring's speaker-attribution rule). Checks every Agent turn (not
    just the first) for an explicit COE-name marker; falls back to
    approved-script similarity (paraphrase-tolerant) across all Agent turns
    when no explicit marker is found.
    """
    scripts = scripts or DEFAULT_SCRIPTS_AR
    turns = split_transcript_turns(call.transcript)
    best_key: str | None = None
    best_score = 0.0

    for speaker, text in turns:
        if speaker != "agent":
            continue
        norm = _norm(text)
        for key, markers in COE_NAME_MARKERS.items():
            if any(_norm(m) in norm for m in markers):
                return key
        for key, script in scripts.items():
            score = script_similarity(text, script)
            if score > best_score:
                best_key, best_score = key, score

    return best_key if best_score >= SCRIPT_MATCH_THRESHOLD else None


def _agent_turn_supports_category(call: CallTranscript, category: str | None) -> bool:
    """True when SOME actual Agent turn contains real transcript evidence
    connecting to *category* (see CAMPAIGN_COE_CONTEXT_MARKERS) — used to
    ground an LLM-produced COE value (see ground_llm_coe_value), never to
    deterministically set recommended_coe itself (that stays on the
    narrower resolve_recommended_coe/COE_NAME_MARKERS)."""
    if not category or category not in CAMPAIGN_COE_CONTEXT_MARKERS:
        return False
    markers = CAMPAIGN_COE_CONTEXT_MARKERS[category]
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "agent":
            continue
        norm = _norm(text)
        if any(_norm(m) in norm for m in markers):
            return True
    return False


def ground_llm_coe_value(call: CallTranscript, value: str | None) -> str | None:
    """Reject an LLM-produced COE category value (e.g. recommended_coe)
    unless it is one of the four supported keys AND an actual Agent turn
    contains real transcript evidence for it (see
    _agent_turn_supports_category). An LLM must never be trusted to select
    a category merely because all four COE records/scripts were shown to
    it as reference data — that data describes what's POSSIBLE, not what
    happened in this call (see app.prompts.qa_prompt.build_coe_prompt's
    grounding rules). Returns None (discarded, never a fabricated
    mismatch) when unsupported.
    """
    if not value or value not in COE_KEYS:
        return None
    return value if _agent_turn_supports_category(call, value) else None


# ── Doctor-name extraction boundary cleanup ─────────────────────────────────
# app.service_hub.doctor_validation.extract_doctor_turn_candidates's stop-
# marker regex requires WHITESPACE immediately before "عياده"/"عيادة" (etc.)
# to recognise it as a boundary — it does not catch these words FUSED with
# a leading "ب" (e.g. "بعياده", one token, no space), so a candidate like
# "محمود الحوراني بعياده المخ" can slip through with the clinic reference
# still attached. Rather than touching that shared, heavily-tested
# extraction engine (used by several unrelated validators), this narrow,
# COE-local post-processing step trims the same class of clinic/
# department/specialty connector — and everything after it — from an
# already-extracted candidate. Only this small, closed set of
# administrative connector words (never a genuine person-name token) is
# ever treated as a stop point, so a real multi-token compound name is
# never truncated.
_DOCTOR_NAME_CLINIC_STOP_WORDS: set[str] = {
    "عياده", "عيادة", "قسم", "تخصص",
    "بعياده", "بعيادة", "بقسم", "بتخصص",
    "في",  # covers "في عياده" / "في قسم" / "في تخصص" — "في" alone is
           # never part of a person's name (mirrors doctor_validation.py's
           # own _NAME_STOP_RE, which treats a bare "في" the same way).
}


def clean_extracted_doctor_name(name: str | None) -> str | None:
    """Trim a trailing clinic/department/specialty connector — and
    everything after it — from an already-extracted doctor-name candidate.

    Example: "محمود الحوراني بعياده المخ والاعصاب" -> "محمود الحوراني".

    Idempotent (safe to call on an already-clean name) and never truncates
    a genuine multi-token personal name: only the small, closed connector
    set in _DOCTOR_NAME_CLINIC_STOP_WORDS is ever treated as a stop point.
    Returns the input unchanged (including falsy values) if no stop word
    is found.
    """
    if not name:
        return name
    words = name.split()
    for i, word in enumerate(words):
        if normalize_arabic_text(word) in _DOCTOR_NAME_CLINIC_STOP_WORDS:
            cleaned = " ".join(words[:i]).strip()
            return cleaned or name
    return name


# ── Existing-patient exception ──────────────────────────────────────────────

def existing_patient_exception_evidence(call: CallTranscript) -> str | None:
    """Deterministic evidence that the customer is an EXISTING patient with
    an established treating doctor — applied narrowly (explicit follow-up/
    existing-doctor language only), never inferred from a vague doctor
    mention (see module docstring)."""
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "patient":
            continue
        if _EXISTING_PATIENT_RE.search(_norm(text)):
            return text.strip()[:300]
    return None


# ── Authoritative primary-doctor matching ───────────────────────────────────
#
# Cross-script identity resolution (Arabic transcript extraction <-> English
# canonical business-rule name) is handled ENTIRELY through the explicit
# PRIMARY_DOCTOR_ALIASES table above — never through fuzzy string
# similarity between an Arabic string and a Latin one, which is meaningless
# (they share no characters to compare). Fuzzy matching, when used at all,
# is restricted to comparing a candidate against aliases written in the
# SAME script, for minor spelling/ASR variants only — see
# resolve_primary_doctor_identity.

_EN_DOCTOR_TITLE_RE = re.compile(r"^(?:dr\.?|doctor)\.?\s+", re.I)
_AR_DOCTOR_TITLE_RE = re.compile(r"^(?:دكتور[ةه]?|د\s*[./\\\-]?)\s+")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]")
_ARABIC_TATWEEL_RE = re.compile("ـ+")
_ARABIC_CHAR_RANGE_RE = re.compile(r"[؀-ۿ]")

# Minimum normalised-name length and same-script fuzzy-matching thresholds.
# Deliberately conservative — see resolve_primary_doctor_identity's
# docstring for the rationale behind each guard.
_MIN_MATCHABLE_NAME_LENGTH = 4
_FUZZY_SCORE_THRESHOLD = 92
_FUZZY_LENGTH_RATIO_THRESHOLD = 0.8  # rejects a partial name (e.g. a bare
                                     # first name, or a first+middle name
                                     # missing its surname) from ever
                                     # fuzzy-matching a full approved name.


def _strip_doctor_title(name: str) -> str:
    """Strip a single leading Dr/Dr./Doctor/دكتور/دكتورة/د./د title —
    English checked first, then Arabic, so either language's title is
    removed regardless of which one is present."""
    s = name.strip()
    s = _EN_DOCTOR_TITLE_RE.sub("", s)
    s = _AR_DOCTOR_TITLE_RE.sub("", s)
    return s.strip()


def normalize_doctor_name_for_match(name: str | None) -> str:
    """Normalise a doctor-name string for comparison — never used to alter
    the ORIGINAL extracted transcript evidence, only for matching.

    Applies, in order:
      1. Arabic diacritics (tashkeel) removal.
      2. Arabic tatweel (ـ) removal.
      3. A single leading English or Arabic doctor title strip
         (Dr/Dr./Doctor/دكتور/دكتورة/د.).
      4. The project's shared Arabic/ASCII text normaliser
         (app.services.text_helpers.normalize_arabic_text) — case-folding,
         punctuation/whitespace collapse, and Arabic character-variant
         collapsing (أ/إ/آ->ا, ة->ه, ى->ي, ؤ->و).

    Compound-name spacing (e.g. "عبد السلام" vs "عبدالسلام") is handled
    separately by comparing the fully whitespace-collapsed form of the
    result — see resolve_primary_doctor_identity — rather than folded into
    this function, so callers that need the spaced form (e.g. for a
    same-script token-overlap check) can still get it.
    """
    if not name:
        return ""
    s = str(name).strip()
    s = _ARABIC_DIACRITICS_RE.sub("", s)
    s = _ARABIC_TATWEEL_RE.sub("", s)
    s = _strip_doctor_title(s)
    return normalize_arabic_text(s)


def _is_arabic_text(normalized: str) -> bool:
    """True when *normalized* contains Arabic-script characters — used to
    keep fuzzy matching strictly WITHIN one writing system (see
    resolve_primary_doctor_identity's docstring)."""
    return bool(_ARABIC_CHAR_RANGE_RE.search(normalized))


def _build_alias_lookup() -> dict[str, dict[str, str]]:
    """{coe: {normalized_alias_or_compact_form: canonical_name}} built once
    from PRIMARY_DOCTOR_ALIASES. Both the normally-spaced normalised form
    AND a fully whitespace-collapsed ("compact") form are indexed for every
    alias (and the canonical name itself), so a compound-name spacing
    difference not literally present in the alias list (e.g. "عبدالسلام"
    when only "عبد السلام" is listed) still resolves via an exact
    deterministic comparison — never via fuzzy matching."""
    lookup: dict[str, dict[str, str]] = {}
    for coe, doctors in PRIMARY_DOCTOR_ALIASES.items():
        table: dict[str, str] = {}
        for canonical, aliases in doctors.items():
            for alias in (*aliases, canonical):
                norm = normalize_doctor_name_for_match(alias)
                if not norm:
                    continue
                table.setdefault(norm, canonical)
                table.setdefault(norm.replace(" ", ""), canonical)
        lookup[coe] = table
    return lookup


_ALIAS_LOOKUP: dict[str, dict[str, str]] = _build_alias_lookup()


def resolve_primary_doctor_identity(extracted_name: str | None, coe: str | None) -> str | None:
    """Resolve an extracted Arabic or English doctor name to its canonical
    approved-doctor identity for *coe* — the SINGLE authority for primary-
    doctor approval decisions (see match_primary_doctor, the boolean
    wrapper other callers use).

    Matching order (never skipped or reordered):
      1. EXACT normalized-alias match (including the compact,
         whitespace-collapsed form) against PRIMARY_DOCTOR_ALIASES for
         *coe* only — this is how Arabic/English cross-script identity is
         resolved. An Arabic extraction is NEVER compared to an English
         canonical name via string similarity; it is only ever looked up
         against its own explicit Arabic aliases.
      2. Conservative SAME-SCRIPT fuzzy matching for minor spelling/ASR
         variants not already covered by an alias — Arabic candidates are
         only ever fuzzy-compared against Arabic aliases, English
         candidates only against English aliases. Guarded by:
           - a minimum candidate length (rejects a bare 2-3 letter
             fragment),
           - a length-ratio pre-filter that rejects a PARTIAL name (e.g. a
             bare "عبد الرحمن" — the shared prefix of two different
             approved doctors' names) from ever being compared as if it
             were a full name,
           - a high similarity threshold, and
           - outright rejection when the best-scoring match ties with a
             DIFFERENT canonical doctor — this function never guesses just
             because one option happens to be the closest.
      3. Otherwise: no match (None) — never guessed.

    Restricted to *coe*'s own roster: a name that matches a doctor approved
    for a DIFFERENT COE never resolves here, regardless of how confident
    the match would be for that other COE.
    """
    if not coe or coe not in PRIMARY_DOCTOR_ALIASES:
        return None
    norm = normalize_doctor_name_for_match(extracted_name)
    if not norm or len(norm) < _MIN_MATCHABLE_NAME_LENGTH:
        return None

    # Tier 1 — exact normalized alias match (spaced or compact form).
    table = _ALIAS_LOOKUP.get(coe, {})
    if norm in table:
        return table[norm]
    compact = norm.replace(" ", "")
    if compact in table:
        return table[compact]

    # Tier 2 — conservative same-script fuzzy matching only.
    candidate_is_arabic = _is_arabic_text(norm)
    scored: list[tuple[str, str, float]] = []
    for canonical, aliases in PRIMARY_DOCTOR_ALIASES[coe].items():
        for alias in (*aliases, canonical):
            alias_norm = normalize_doctor_name_for_match(alias)
            if not alias_norm or _is_arabic_text(alias_norm) != candidate_is_arabic:
                continue  # never fuzzy-compare across writing systems
            alias_compact = alias_norm.replace(" ", "")
            shorter, longer = sorted((len(compact), len(alias_compact)))
            if longer == 0 or shorter / longer < _FUZZY_LENGTH_RATIO_THRESHOLD:
                continue  # rejects a partial name (missing a whole token)
            score = float(_rfuzz.token_sort_ratio(norm, alias_norm))
            if score >= _FUZZY_SCORE_THRESHOLD:
                scored.append((canonical, alias_norm, score))

    if not scored:
        return None
    scored.sort(key=lambda t: -t[2])
    best_canonical, best_alias, best_score = scored[0]
    ties_with_other_doctor = [s for s in scored if s[2] == best_score and s[0] != best_canonical]
    if ties_with_other_doctor:
        return None  # ambiguous — never select solely because it's closest
    return best_canonical


def match_primary_doctor(candidate_name: str | None, coe_key: str | None) -> bool:
    """Boolean convenience wrapper around resolve_primary_doctor_identity —
    True only when *candidate_name* confidently identifies one of
    *coe_key*'s authoritative primary doctors."""
    return resolve_primary_doctor_identity(candidate_name, coe_key) is not None


def ground_doctor_names(names: list[str], transcript_candidates: list[str]) -> list[str]:
    """Filter *names* (as produced by semantic/LLM extraction) down to only
    those that are actually GROUNDED in the deterministically-extracted
    Agent doctor candidates from the transcript
    (app.service_hub.doctor_validation.extract_doctor_turn_candidates) —
    never trusting a semantically-extracted name that was not, in fact,
    said by the agent (see module docstring's "never invent a doctor"
    rule). Returns the ORIGINAL transcript-extracted string for every
    grounded match (never the possibly-reworded input name), de-duplicated,
    order preserved.
    """
    grounded: list[str] = []
    normalized_candidates = [(orig, normalize_doctor_name_for_match(orig)) for orig in transcript_candidates]
    for name in names or []:
        if not isinstance(name, str) or not name.strip():
            continue
        norm_name = normalize_doctor_name_for_match(name)
        if not norm_name:
            continue
        for orig, norm_candidate in normalized_candidates:
            if not norm_candidate:
                continue
            if norm_name == norm_candidate or norm_name in norm_candidate or norm_candidate in norm_name:
                grounded.append(orig)
                break
            if _rfuzz.token_sort_ratio(norm_name, norm_candidate) >= 80:
                grounded.append(orig)
                break
    seen: set[str] = set()
    out: list[str] = []
    for g in grounded:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

# ═════════════════════════════════════════════════════════════════════════
# Multi-context COE evaluation
#
# A single conversation may legitimately discuss MORE than one COE (e.g. a
# Headache campaign click followed by an unrelated Agent recommendation of
# the IBD COE) — collapsing everything into one scalar expected_coe/
# recommended_coe/validation_coe silently conflates them and lets one
# approved doctor "cover for" a different, unapproved doctor offered under
# a different COE. This section builds ONE INDEPENDENT evaluation context
# per grounded COE (see build_coe_contexts) and associates every extracted
# doctor with the correct context using TURN-LEVEL transcript evidence
# (see extract_doctor_context_associations), never by cross-matching every
# doctor against every COE.
#
# This is purely ADDITIVE: the pre-existing single-scalar fields
# (expected_coe/campaign_coe/recommended_coe/validation_coe/
# coe_match_status/primary_doctor_status/matched_primary_doctors/
# recommended_or_selected_doctors) are computed exactly as before and kept
# unchanged for single-context calls (see app.agent.nodes.
# infer_coe_validation) — this section only adds the richer
# coe_evaluations/overall_coe_status representation alongside them, per
# the project's "keep existing scalar fields when exactly one context
# exists, never silently pick one when several do" backward-compatibility
# rule.
# ═════════════════════════════════════════════════════════════════════════

# Referral/later-involvement language — a doctor mentioned alongside this
# (in the same or an adjacent turn) is a possible LATER referral, never the
# INITIAL COE appointment doctor (mirrors the LLM-facing instruction in
# app.prompts.qa_prompt.build_coe_prompt, but applied deterministically
# here so multi-context role assignment never depends on an LLM call).
_REFERRAL_LANGUAGE_RE = re.compile(
    r"بعد\s*(?:ال)?تقييم|لاحقا|لو\s*احتجت|قد\s*يتابع|ربما\s*تحتاج|"
    r"في\s*مرحل[ةه]\s*لاحق[ةه]|لاحق[ةه]\s*لو|"
    r"might\s*(?:also\s*)?(?:get\s*)?involve|later\s*referral|after\s*the\s*initial",
    re.I,
)

# ── Doctor-name-candidate rejection filter ──────────────────────────────────
# Additive, COE-local defense against referral/service/administrative
# phrases that can slip past the shared extraction engine's blocklists in
# some phrasings (e.g. "تحويل طبي" / "وبيتم التحويل بعد ذلك" — a passive
# future-tense construction not covered by doctor_validation.py's own,
# independently-tested blocklist). Deliberately kept HERE rather than
# added to that shared, heavily-tested engine (used by unrelated
# validators) — see clean_extracted_doctor_name's docstring for the same
# rationale.
_NON_DOCTOR_CANDIDATE_PHRASES: set[str] = {
    "تحويل طبي", "استشارة طبيب", "موعد مع الدكتور", "تحويل", "استشارة", "موعد",
}
_NON_DOCTOR_CANDIDATE_FIRST_WORDS: set[str] = {
    "تحويل", "التحويل", "لتحويل", "استشارة", "الاستشارة", "موعد", "الموعد",
    "وبيتم", "بيتم", "يتم", "سيتم", "هيتم", "تم", "وتم",
}


def _specialty_alias_token_tuples() -> set[tuple[str, ...]]:
    """Every specialty/clinic alias (STRONG and WEAK) from the centralized
    taxonomy, as a tuple of normalised tokens — reused as a deterministic
    safeguard so a candidate that IS ENTIRELY a specialty/clinic phrase
    (e.g. "جهاز هضمي" extracted from "لدكتور جهاز هضمي") is rejected as a
    doctor name, never accepted as one (see
    is_plausible_coe_doctor_candidate). This is exactly the centralized
    mapping's "deterministic safeguards" consumer the module docstring
    calls for."""
    tuples: set[tuple[str, ...]] = set()
    for aliases in SPECIALTY_ALIASES.values():
        for alias in aliases:
            tuples.add(tuple(_tokens(alias)))
    for aliases in WEAK_SPECIALTY_ALIASES.values():
        for alias in aliases:
            tuples.add(tuple(_tokens(alias)))
    return tuples


_SPECIALTY_ALIAS_TOKEN_TUPLES: set[tuple[str, ...]] = _specialty_alias_token_tuples()


def is_plausible_coe_doctor_candidate(name: str | None) -> bool:
    """Reject a referral/service/administrative phrase, or a bare
    specialty/clinic phrase, from ever being reported as a doctor name
    (see _NON_DOCTOR_CANDIDATE_PHRASES / _NON_DOCTOR_CANDIDATE_FIRST_WORDS
    / _SPECIALTY_ALIAS_TOKEN_TUPLES) — a deliberately narrow, closed
    rejection list, same non-exhaustive philosophy as doctor_validation.
    py's own blocklists: only well-defined negative cases are excluded,
    never a positive name-vocabulary guess."""
    if not name:
        return False
    norm = normalize_arabic_text(name)
    if not norm:
        return False
    if norm in {normalize_arabic_text(p) for p in _NON_DOCTOR_CANDIDATE_PHRASES}:
        return False
    words = norm.split()
    if not words or words[0] in _NON_DOCTOR_CANDIDATE_FIRST_WORDS:
        return False
    if tuple(words) in _SPECIALTY_ALIAS_TOKEN_TUPLES:
        return False
    return True


_ALIAS_TITLE_PREFIX_RE = re.compile(r"^(?:dr\.?|doctor|د(?:[./\\-])?|دكتور[ةه]?)\s+", re.I)


def _known_doctor_alias_candidates(text: str) -> list[str]:
    """Doctor names/aliases from the authoritative PRIMARY_DOCTOR_ALIASES
    table found as an exact multi-word phrase anywhere in *text* — a
    SUPPLEMENTARY extraction path alongside the shared title-anchored
    engine (_doctor_name_candidates_in_text), needed for phrasing where a
    degree/title word FOLLOWS the name rather than preceding it (e.g.
    "داليندا عرفاوي استشاري امراض الجهاز الهضمي" — the shared engine only
    anchors on a title BEFORE the name, so it finds nothing here at all).

    Only considers aliases with NO leading Dr/Dr./Doctor/دكتور/د title —
    _tokens() (via normalize_arabic_text) silently STRIPS a leading title
    from its own normalised output, so a title-prefixed alias like
    "دكتورة داليندا عرفاوي" would otherwise phrase-match on its
    title-STRIPPED form alone, yet still be returned with the raw,
    untouched title still attached — every doctor already has a bare
    (title-free) alias covering the same name, so title-prefixed aliases
    are redundant here, not a coverage gap.

    Restricted to multi-word aliases (>= 2 tokens) to avoid a bare
    single-token alias over-matching without any title/context anchor —
    single-token mentions stay the shared engine's job."""
    tokens = _tokens(text)
    if not tokens:
        return []
    found: list[str] = []
    for doctors in PRIMARY_DOCTOR_ALIASES.values():
        for aliases in doctors.values():
            for alias in aliases:
                if _ALIAS_TITLE_PREFIX_RE.match(alias.strip()):
                    continue
                if len(_tokens(alias)) >= 2 and alias not in found and _contains_phrase(tokens, alias):
                    found.append(alias)
    return found


def _coe_matches_in_text(text: str) -> list[str]:
    """Every supported COE whose CAMPAIGN_COE_CONTEXT_MARKERS vocabulary
    appears in *text* (a single turn) — order follows COE_KEYS. Kept for
    the pre-existing single-scalar backward-compatibility call sites
    (resolve_campaign_coe's per-turn scan) — this is a permissive "is
    there SOME evidence for COE X" check and, unlike the multi-context
    builder below, does not attempt to EXCLUSIVELY resolve a SHARED
    specialty (e.g. ENT) to one specific COE."""
    norm = _norm(text)
    return [key for key in COE_KEYS if any(_norm(m) in norm for m in CAMPAIGN_COE_CONTEXT_MARKERS[key])]


def _unambiguous_coe_matches(text: str) -> list[str]:
    """COE matches that never require disambiguation: explicit COE-name
    markers, complaint keywords, and specialties that map to exactly ONE
    COE (see detect_specialty_mentions)."""
    norm = _norm(text)
    matched: list[str] = [key for key in COE_KEYS if any(_norm(m) in norm for m in COE_NAME_MARKERS[key])]
    for key in COE_KEYS:
        if key not in matched and any(_norm(kw) in norm for kw in COMPLAINT_KEYWORDS[key]):
            matched.append(key)
    for _specialty, candidate_coes in detect_specialty_mentions(text):
        if len(candidate_coes) == 1 and candidate_coes[0] not in matched:
            matched.append(candidate_coes[0])
    return matched


def resolve_specialty_coes(text: str, active_coes: Any = ()) -> list[str]:
    """Every COE resolvable from *text*'s specialty mentions — an
    unambiguous specialty (maps to exactly one COE) always resolves; a
    SHARED specialty (e.g. "ENT" -> Headache or Asthma) resolves ONLY when
    EXACTLY ONE of its candidate COEs is already active (this text's own
    unambiguous evidence, plus whatever the caller passes as
    *active_coes* — see the module docstring's disambiguation priority
    order: explicit campaign/COE name > patient complaint > specialty/
    doctor proximity > active booking context > turn order). A shared
    specialty that cannot be disambiguated this way contributes NOTHING
    here — never guessed (see ambiguous_specialty_mentions for surfacing
    it instead)."""
    matched = _unambiguous_coe_matches(text)
    local_active = set(active_coes) | set(matched)
    for _specialty, candidate_coes in detect_specialty_mentions(text):
        if len(candidate_coes) <= 1:
            continue
        overlap = [c for c in candidate_coes if c in local_active]
        if len(overlap) == 1 and overlap[0] not in matched:
            matched.append(overlap[0])
    for _specialty, candidate_coes in detect_weak_specialty_mentions(text):
        for coe in candidate_coes:
            if coe in local_active and coe not in matched:
                matched.append(coe)
    return matched


def ambiguous_specialty_mentions(text: str, active_coes: Any = ()) -> list[tuple[str, list[str]]]:
    """Every SHARED specialty mentioned in *text* whose ambiguity could
    NOT be resolved by *active_coes* (see resolve_specialty_coes) —
    surfaced so an unresolved shared-specialty mention (e.g. a bare "ENT"
    referral with no established Headache/Asthma context yet) reports as
    uncertain rather than being silently dropped or guessed into one."""
    matched = _unambiguous_coe_matches(text)
    local_active = set(active_coes) | set(matched)
    unresolved: list[tuple[str, list[str]]] = []
    for specialty, candidate_coes in detect_specialty_mentions(text):
        if len(candidate_coes) <= 1:
            continue
        overlap = [c for c in candidate_coes if c in local_active]
        if len(overlap) != 1:
            unresolved.append((specialty, candidate_coes))
    return unresolved


def build_coe_contexts(
    call: CallTranscript, scripts: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Build one independent evaluation context per COE with REAL grounded
    evidence anywhere in the transcript — never for a COE that merely
    appears in CRM reference data or an LLM prompt (see module docstring).

    A context is created for a COE the moment ANY of the following is
    found (turn-by-turn, in transcript order):
      - "campaign"            — a Patient turn carries the structured
                                 campaign identifier (see
                                 campaign_origin_evidence) AND its own text
                                 identifies this COE.
      - "agent_recommendation" — an Agent turn explicitly names this COE
                                 (COE_NAME_MARKERS) or closely paraphrases
                                 its approved script (script_similarity).
      - "patient_complaint"   — a Patient turn describes a symptom mapping
                                 to this COE (COMMPLAINT_KEYWORDS).
      - "agent_specialty" / "patient_specialty" — a turn mentions a
                                 canonical specialty organizationally
                                 grouped under this COE (see
                                 COE_SPECIALTIES/SPECIALTY_ALIASES),
                                 tagged by speaker. A SHARED specialty
                                 (e.g. "ENT" -> Headache or Asthma) is
                                 resolved via detect_specialty_mentions's
                                 disambiguation — it is NEVER credited to
                                 every COE it could organizationally
                                 belong to, and never guessed when
                                 unresolved.

    Each "specialties" entry is a structured
    {"canonical_specialty", "original_text", "speaker", "evidence"} dict —
    both the canonical specialty AND the original transcript wording are
    preserved (see the module docstring's specialty-normalisation
    requirement).

    Returns {coe_key: {"coe", "context_sources", "complaints",
    "specialties", "campaign_evidence", "agent_coe_evidence"}} — a COE with
    zero evidence never appears as a key at all.
    """
    scripts = scripts or DEFAULT_SCRIPTS_AR
    turns = split_transcript_turns(call.transcript)
    contexts: dict[str, dict[str, Any]] = {}

    def _ctx(coe: str) -> dict[str, Any]:
        return contexts.setdefault(coe, {
            "coe": coe,
            "context_sources": [],
            "complaints": [],
            "specialties": [],
            "campaign_evidence": None,
            "agent_coe_evidence": None,
        })

    def _add_source(coe: str, source: str) -> None:
        c = _ctx(coe)
        if source not in c["context_sources"]:
            c["context_sources"].append(source)

    running_active: list[str] = []

    def _add_specialty(coe: str, canonical_specialty: str, original_text: str, speaker: str, excerpt: str) -> None:
        c = _ctx(coe)
        entry = {
            "canonical_specialty": canonical_specialty,
            "original_text": original_text,
            "speaker": speaker,
            "evidence": excerpt,
        }
        if entry not in c["specialties"]:
            c["specialties"].append(entry)

    for speaker, text in turns:
        norm = _norm(text)
        excerpt = text.strip()[:300]
        this_turn_coes: list[str] = []

        if speaker == "patient" and _CAMPAIGN_ORIGIN_RE.search(text):
            for coe in _unambiguous_coe_matches(text):
                _add_source(coe, "campaign")
                this_turn_coes.append(coe)
                c = _ctx(coe)
                if c["campaign_evidence"] is None:
                    c["campaign_evidence"] = excerpt

        if speaker == "agent":
            for coe, markers in COE_NAME_MARKERS.items():
                if any(_norm(m) in norm for m in markers):
                    _add_source(coe, "agent_recommendation")
                    this_turn_coes.append(coe)
                    c = _ctx(coe)
                    if c["agent_coe_evidence"] is None:
                        c["agent_coe_evidence"] = excerpt
            # Script similarity is a fuzzy, "which ONE approved script does
            # this turn most resemble" signal — the four approved scripts
            # share substantial boilerplate wording ("لضمان تحقيق أقصى
            # استفادة...سيتم حجز موعد لحضرتك بمركز التميز المتخصص في..."),
            # so checking each COE's script INDEPENDENTLY against the same
            # threshold would credit ALL four from one turn's shared
            # boilerplate alone. Only the single BEST-scoring COE for THIS
            # turn is ever credited (mirrors resolve_recommended_coe's own
            # "best match wins" semantics) — never more than one per turn,
            # though separate turns may still independently recommend
            # separate COEs.
            best_script_key, best_script_score = None, 0.0
            for coe, script in scripts.items():
                score = script_similarity(text, script)
                if score > best_script_score:
                    best_script_key, best_script_score = coe, score
            if best_script_key and best_script_score >= SCRIPT_MATCH_THRESHOLD:
                _add_source(best_script_key, "agent_recommendation")
                this_turn_coes.append(best_script_key)
                c = _ctx(best_script_key)
                if c["agent_coe_evidence"] is None:
                    c["agent_coe_evidence"] = excerpt

        if speaker == "patient":
            for coe, kws in COMPLAINT_KEYWORDS.items():
                if any(_norm(kw) in norm for kw in kws):
                    _add_source(coe, "patient_complaint")
                    this_turn_coes.append(coe)
                    c = _ctx(coe)
                    if excerpt[:200] not in c["complaints"]:
                        c["complaints"].append(excerpt[:200])

        # Specialty evidence — resolved with disambiguation, never
        # cross-attributing a SHARED specialty (e.g. ENT) to every COE it
        # could organizationally belong to. active_coes here is every COE
        # established so far (prior turns) PLUS this turn's own
        # unambiguous evidence (so e.g. a Headache complaint and an ENT
        # referral in the SAME turn/conversation still correctly resolve
        # ENT -> Headache — see the module's disambiguation priority
        # order: explicit COE/campaign and patient complaint both outrank
        # bare specialty/turn proximity).
        active_for_specialty = set(running_active) | set(this_turn_coes)
        for specialty, candidate_coes in detect_specialty_mentions(text):
            resolved_coes = (
                candidate_coes if len(candidate_coes) == 1
                else [c for c in candidate_coes if c in active_for_specialty]
            )
            if len(resolved_coes) != 1:
                continue  # ambiguous/unresolved — never guessed into a context
            coe = resolved_coes[0]
            _add_source(coe, "agent_specialty" if speaker == "agent" else "patient_specialty")
            this_turn_coes.append(coe)
            _add_specialty(coe, specialty, specialty, speaker, excerpt[:200])

        for specialty, candidate_coes in detect_weak_specialty_mentions(text):
            for coe in candidate_coes:
                if coe not in active_for_specialty:
                    continue  # weak evidence alone never creates/extends a context
                _add_source(coe, "agent_specialty" if speaker == "agent" else "patient_specialty")
                _add_specialty(coe, specialty, specialty, speaker, excerpt[:200])

        for coe in this_turn_coes:
            if coe not in running_active:
                running_active.append(coe)

    return contexts


# Deliberately does NOT include "." — an English title abbreviation
# ("Dr.") relies on that exact period immediately before the name, and
# splitting there would sever the title from the name it introduces
# (see _doctor_name_candidates_in_text, which needs both together).
# Arabic commas/semicolons and newlines are the primary clause boundary
# in these transcripts regardless.
_CLAUSE_SPLIT_RE = re.compile(r"[،؛\n]+")


def _split_clauses(text: str) -> list[str]:
    """Split one turn's text into rough clauses on sentence-level
    punctuation — used ONLY so a doctor's role and COE association are
    read from the specific clause naming them, not bled in from an
    unrelated LATER clause in the same Agent turn (e.g. an initial-doctor
    offer immediately followed, in the same turn, by a sentence about a
    possible later referral — see extract_doctor_context_associations)."""
    return [p.strip() for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]


def _global_unambiguous_coes(turns: list[tuple[str, str]]) -> list[str]:
    """Every COE unambiguously grounded ANYWHERE in the call — order-
    independent (explicit campaign/COE-name/complaint evidence, priority
    levels 1-3 of the module's disambiguation order, are position-
    independent global signals) — used as the baseline pool for resolving
    a SHARED specialty (e.g. ENT) mentioned anywhere in the same call."""
    found: list[str] = []
    for _speaker, text in turns:
        for coe in _unambiguous_coe_matches(text):
            if coe not in found:
                found.append(coe)
    return found


def extract_doctor_context_associations(call: CallTranscript) -> list[dict[str, Any]]:
    """Extract every AGENT-turn doctor-name mention together with the
    COE(s) it is grounded to via turn/clause-level evidence — never a flat
    list cross-matched against every COE (see module docstring).

    Association order (first match wins, never guessed further):
      1. The SAME CLAUSE the doctor was named in (tightest scope) — COE
         evidence in that exact clause, e.g. "دكتور محمود الحوراني بعيادة
         المخ والاعصاب" -> Headache. A SHARED specialty in this clause is
         resolved against the call's global unambiguous COEs (see
         resolve_specialty_coes / _global_unambiguous_coes) — priority
         levels 1-3 (explicit campaign/COE name, patient complaint) always
         outrank bare turn proximity.
      2. Elsewhere in the SAME Agent turn (other clauses of it).
      3. An IMMEDIATELY SURROUNDING turn (one turn before or after).
      4. The single COE established so far elsewhere in the call, ONLY
         when EXACTLY ONE is active (unambiguous) — never guessed when
         zero or several are active (see evaluate_context_doctors's
         "uncertain" handling for that case).

    Every extracted candidate is passed through
    is_plausible_coe_doctor_candidate — a referral/service/administrative
    phrase (e.g. "تحويل طبي", "وبيتم التحويل بعد ذلك") that slips past the
    shared extraction engine in some phrasings is rejected here rather
    than ever being reported as a doctor.

    Role ("initial_primary" / "referral_only" / "existing_treating_doctor")
    is likewise read from the doctor's OWN clause first (falling back to
    the immediately preceding Patient turn only for the existing-treating-
    doctor check) — so a later-referral sentence appended after the
    initial doctor's offer, in the SAME turn, never demotes that initial
    doctor to referral_only.

    Each entry's "associated_coes" is the ordered list of COEs the doctor
    resolved to at that step (possibly more than one) — empty when
    genuinely unresolvable.
    """
    turns = split_transcript_turns(call.transcript)
    global_active = _global_unambiguous_coes(turns)
    per_turn_coes = [resolve_specialty_coes(text, active_coes=global_active) for _speaker, text in turns]
    associations: list[dict[str, Any]] = []
    running_active: list[str] = []

    for idx, (speaker, text) in enumerate(turns):
        for coe in per_turn_coes[idx]:
            if coe not in running_active:
                running_active.append(coe)

        if speaker != "agent":
            continue
        clauses = _split_clauses(text) or [text]

        nearby_coes: list[str] = []
        for neighbour in (idx - 1, idx + 1):
            if 0 <= neighbour < len(turns):
                for coe in per_turn_coes[neighbour]:
                    if coe not in nearby_coes:
                        nearby_coes.append(coe)

        preceding_patient_text = ""
        if idx > 0 and turns[idx - 1][0] == "patient":
            preceding_patient_text = turns[idx - 1][1]
        preceding_patient_is_existing = bool(
            preceding_patient_text and _EXISTING_PATIENT_RE.search(_norm(preceding_patient_text))
        )

        clause_coes = [resolve_specialty_coes(clause, active_coes=global_active) for clause in clauses]

        for clause_idx, clause in enumerate(clauses):
            candidates = [
                cleaned for cleaned in (
                    clean_extracted_doctor_name(raw) for raw in _doctor_name_candidates_in_text(clause)
                )
                if is_plausible_coe_doctor_candidate(cleaned)
            ]
            for alias_name in _known_doctor_alias_candidates(clause):
                if alias_name not in candidates:
                    candidates.append(alias_name)
            if not candidates:
                continue

            norm_clause = _norm(clause)
            same_clause_coes = clause_coes[clause_idx]
            other_clause_coes: list[str] = []
            for other_idx, other_coes in enumerate(clause_coes):
                if other_idx == clause_idx:
                    continue
                for coe in other_coes:
                    if coe not in other_clause_coes:
                        other_clause_coes.append(coe)

            if same_clause_coes:
                associated_coes, certainty = same_clause_coes, "same_clause"
            elif other_clause_coes:
                associated_coes, certainty = other_clause_coes, "same_turn"
            elif nearby_coes:
                associated_coes, certainty = nearby_coes, "nearby_turn"
            elif len(running_active) == 1:
                associated_coes, certainty = list(running_active), "most_recent_active"
            else:
                associated_coes, certainty = [], "unclear"

            role = "referral_only" if _REFERRAL_LANGUAGE_RE.search(norm_clause) else "initial_primary"
            if _EXISTING_PATIENT_RE.search(norm_clause) or preceding_patient_is_existing:
                role = "existing_treating_doctor"

            association_evidence = clause.strip()[:300]
            for cleaned in candidates:
                associations.append({
                    "extracted_name": cleaned,
                    "turn_index": idx,
                    "associated_coes": list(associated_coes),
                    "association_certainty": certainty,
                    "role": role,
                    "association_evidence": association_evidence,
                })

    return associations


def _dedupe_doctor_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated mentions of the SAME doctor within one context
    (e.g. the full name offered by the agent, then a later shortened
    confirmation like "اسامه" for "اسامه عبدالسلام") into ONE entry —
    never reporting the same doctor twice within a context. The FULLEST
    extracted name is kept as the primary evidence (see the module
    docstring's deduplication requirement); role/evidence are taken from
    whichever mention is kept.

    Two mentions are treated as the same doctor when their normalised
    forms are equal, or one is a normalised prefix/substring of the other
    (a later shortened reference is, by definition, shorter) — this is a
    looser check than resolve_primary_doctor_identity's own ambiguous-
    partial-name guard, because here the question is only "was this
    person already mentioned", not "does this confidently identify an
    APPROVED doctor".
    """
    kept: list[dict[str, Any]] = []
    for entry in entries:
        norm_name = normalize_doctor_name_for_match(entry.get("extracted_name"))
        merged = False
        if norm_name:
            for existing in kept:
                existing_norm = normalize_doctor_name_for_match(existing.get("extracted_name"))
                if not existing_norm:
                    continue
                if norm_name == existing_norm or norm_name in existing_norm or existing_norm in norm_name:
                    if len(entry.get("extracted_name") or "") > len(existing.get("extracted_name") or ""):
                        existing["extracted_name"] = entry["extracted_name"]
                        existing["association_evidence"] = entry.get("association_evidence")
                    merged = True
                    break
        if not merged:
            kept.append(dict(entry))
    return kept


def evaluate_context_doctors(
    coe_key: str, doctor_entries: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Validate every doctor already associated with *coe_key* INDEPENDENTLY
    against that COE's own authoritative primary-doctor list — never
    letting one approved doctor stand in for another unapproved one (see
    module docstring's core requirement). Repeated mentions of the SAME
    doctor within this context are first collapsed into one entry (see
    _dedupe_doctor_entries) — a shortened later confirmation must never
    create a second doctor candidate.

    Only entries with role == "initial_primary" are checked for primary-
    doctor approval and count toward the aggregate status — a
    referral_only, initial_supporting, or existing_treating_doctor entry
    is reported (role preserved) but never penalises or passes the
    context on its own (see the aggregation rules below, matching
    app.agent.nodes.infer_coe_validation's single-context precedent).

    Aggregate status:
      - "fail"           at least one initial_primary doctor is unapproved.
      - "pass"           at least one initial_primary doctor exists and
                          every initial_primary doctor is approved.
      - "uncertain"       an initial_primary doctor's identity could not be
                          resolved either way (name too ambiguous/partial —
                          see resolve_primary_doctor_identity).
      - "not_applicable"  no initial_primary doctor was discussed for this
                          COE (only referral/follow-up/supporting doctors,
                          or none).
    """
    per_doctor: list[dict[str, Any]] = []
    for entry in _dedupe_doctor_entries(doctor_entries):
        name = entry.get("extracted_name")
        role = entry.get("role", "initial_primary")
        base = {
            "extracted_name": name,
            "role": role,
            "association_evidence": entry.get("association_evidence"),
        }
        if role != "initial_primary":
            per_doctor.append({
                **base,
                "canonical_name": None,
                "primary_doctor_status": "not_applicable",
                "reason": f"Mentioned only as a {role.replace('_', ' ')}, not the initial COE appointment doctor.",
            })
            continue

        canonical = resolve_primary_doctor_identity(name, coe_key)
        if canonical:
            per_doctor.append({
                **base,
                "canonical_name": canonical,
                "primary_doctor_status": "pass",
                "reason": f"{canonical} ({name}) is an approved primary doctor for the {coe_key} COE." if canonical != name else f"{canonical} is an approved primary doctor for the {coe_key} COE.",
            })
        elif name:
            per_doctor.append({
                **base,
                "canonical_name": None,
                "primary_doctor_status": "fail",
                "reason": f"{name} is not on the approved primary-doctor list for the {coe_key} COE.",
            })
        else:
            per_doctor.append({
                **base,
                "canonical_name": None,
                "primary_doctor_status": "uncertain",
                "reason": "The initial doctor's identity could not be resolved.",
            })

    initial_statuses = [d["primary_doctor_status"] for d in per_doctor if d["role"] == "initial_primary"]
    if not initial_statuses:
        status = "not_applicable"
    elif "fail" in initial_statuses:
        status = "fail"
    elif "uncertain" in initial_statuses:
        status = "uncertain"
    else:
        status = "pass"
    return per_doctor, status


def build_coe_evaluations(
    call: CallTranscript, scripts: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Top-level orchestrator: one independent, fully-evaluated context per
    grounded COE (see build_coe_contexts / extract_doctor_context_
    associations / evaluate_context_doctors above) — the authoritative
    multi-context representation app.agent.nodes.infer_coe_validation
    exposes as coe_validation["coe_evaluations"], alongside (never instead
    of) the pre-existing single-scalar fields.

    A doctor whose association is genuinely "unclear" (see
    extract_doctor_context_associations) is never attached to any context
    — attaching it to a guessed COE would be exactly the kind of
    cross-matching this design forbids. It is still recoverable in the
    returned entries' own list for callers that want to surface an
    "uncertain association" note (see infer_coe_validation).
    """
    contexts = build_coe_contexts(call, scripts)
    associations = extract_doctor_context_associations(call)

    evaluations: list[dict[str, Any]] = []
    for coe_key in COE_KEYS:
        ctx = contexts.get(coe_key)
        if ctx is None:
            continue
        doctors_here = [a for a in associations if coe_key in a["associated_coes"]]
        per_doctor, primary_status = evaluate_context_doctors(coe_key, doctors_here)

        has_recommendation_side = any(
            s in ctx["context_sources"] for s in ("campaign", "agent_recommendation", "agent_specialty")
        )
        coe_match_status = "pass" if has_recommendation_side else "uncertain"

        evaluations.append({
            "coe": coe_key,
            "context_sources": ctx["context_sources"],
            "complaints": ctx["complaints"],
            "specialties": ctx["specialties"],
            "campaign_evidence": ctx["campaign_evidence"],
            "agent_coe_evidence": ctx["agent_coe_evidence"],
            "doctors": per_doctor,
            "coe_match_status": coe_match_status,
            "primary_doctor_status": primary_status,
            "is_violation": primary_status == "fail",
        })

    return evaluations


def unassociated_initial_doctors(call: CallTranscript) -> list[dict[str, Any]]:
    """Doctor mentions whose COE association is genuinely ambiguous (see
    extract_doctor_context_associations's "unclear" certainty) — never
    attached to a guessed context. Surfaced separately so an ambiguous
    association reports as uncertain rather than silently disappearing or
    being force-matched (see module docstring's core requirement)."""
    return [
        a for a in extract_doctor_context_associations(call)
        if a["role"] == "initial_primary" and not a["associated_coes"]
    ]
