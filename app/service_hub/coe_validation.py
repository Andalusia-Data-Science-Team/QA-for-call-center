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
            # "اسمة عبدالسلام" — a shortened/typo'd spelling (missing the
            # middle alef of "اسامة") observed in real transcripts; same
            # person, not a new identity.
            "اسمة عبدالسلام",
            "اسمه عبدالسلام",
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

# ═════════════════════════════════════════════════════════════════════════
# Restricted, business-approved complaint/diagnosis categories — the
# AUTHORITATIVE, SOLE source for the "patient_approved_complaint"
# recommendation-eligibility route (see classify_coe_trigger's Path E and
# build_coe_contexts' "patient_complaint" source below).
#
# Deliberately SEPARATE from COMPLAINT_KEYWORDS above: COMPLAINT_KEYWORDS
# is the older, intentionally BROADER symptom vocabulary that still
# powers the legacy single-scalar expected_coe/coe_match_status path
# (resolve_primary_complaint/detect_complaint_categories),
# _unambiguous_coe_matches' campaign disambiguation, and
# CAMPAIGN_COE_CONTEXT_MARKERS' LLM-grounding vocabulary — none of those
# pre-existing, independently-tested behaviors are in scope here. This
# registry answers a NARROWER, DIFFERENT question — "does an EXPLICITLY
# APPROVED diagnosis/complaint category, on its OWN (with no specialty
# request at all), obligate the human agent to have recommended a COE" —
# and is therefore held to a stricter, business-confirmed list: a vague
# symptom (dizziness, nausea, cough, general foot pain, ...) must NEVER
# by itself create this obligation, even though it may still remain in
# COMPLAINT_KEYWORDS for the older, broader classification purposes above.
#
# Structure: {coe: {category_key: [aliases...]}}. Every alias is matched
# phrase/token-aware (see _contains_phrase) — never naive substring.
COE_COMPLAINTS: dict[str, dict[str, list[str]]] = {
    "Headache": {
        "migraine": [
            "الصداع النصفي", "صداع نصفي", "الشقيقة", "الشقيقه", "Migraine",
        ],
        "tension_headache": [
            "الصداع التوتري", "صداع توتري", "Tension headache", "Tension-type headache",
        ],
        "chronic_headache": [
            "الصداع المزمن", "صداع مزمن", "Chronic headache", "Chronic headaches",
        ],
        "sinus_headache": [
            "صداع الجيوب الأنفية", "صداع الجيوب الانفية", "الصداع الناتج عن الجيوب الأنفية",
            "Sinus headache",
        ],
    },
    "IBD": {
        "digestive_disease": [
            "أمراض الجهاز الهضمي", "امراض الجهاز الهضمي", "مرض في الجهاز الهضمي",
            "مشاكل الجهاز الهضمي", "أمراض المعدة والأمعاء", "امراض المعدة والامعاء",
            "Gastrointestinal disease", "Digestive disease", "Bowel disease",
        ],
        "colon_disease": [
            "قولون", "القولون", "أمراض القولون", "التهاب القولون", "كرون", "مرض كرون",
            "Colon disease", "Colitis", "Crohn's disease", "Crohn disease",
        ],
        "liver_disease": [
            "أمراض الكبد", "امراض الكبد", "مرض في الكبد", "كبد", "الكبد",
            "Liver disease", "Hepatic disease",
        ],
        "pancreatic_disease": [
            "أمراض البنكرياس", "امراض البنكرياس", "مرض في البنكرياس", "التهاب البنكرياس",
            "بنكرياس", "البنكرياس", "Pancreatic disease", "Pancreatitis",
        ],
    },
    "Asthma": {
        "asthma": ["الربو", "ربو", "Asthma"],
        "bronchial_asthma": ["الربو الشعبي", "ربو شعبي", "Bronchial asthma"],
        "chest_allergy": [
            "الحساسية الصدرية", "حساسية صدرية", "Chest allergy", "Respiratory allergy",
        ],
        "shortness_of_breath": [
            "ضيق التنفس", "ضيق في التنفس", "صعوبة التنفس",
            "Shortness of breath", "Difficulty breathing", "Breathlessness",
        ],
    },
    "Diabetes": {
        "diabetes": [
            "السكر", "مرض السكر", "السكري", "مرض السكري", "مريض سكر",
            "Diabetes", "Diabetes mellitus", "Diabetic",
        ],
        "endocrine_disease": [
            "الغدد الصماء", "غدد صماء", "أمراض الغدد الصماء", "امراض الغدد الصماء",
            "Endocrine disease", "Endocrinology",
        ],
        "diabetic_foot": [
            "قدم سكري", "القدم السكري", "القدم السكرية", "مشاكل القدم السكري",
            "Diabetic foot", "Diabetic foot complications",
        ],
    },
}

# The AUTHORITATIVE registry for deciding whether a human-agent COE
# recommendation is REQUIRED — literally the SAME data as COE_COMPLAINTS
# above (never a second, independently-authored copy), exposed under this
# name because "does this patient need require a COE recommendation" is a
# conceptually distinct question from "what counts as an approved
# diagnosis/complaint category", even though today they share one list.
# recommendation_required / missed_recommendation / the "patient's own
# active need" trigger path (see classify_coe_trigger's Path E) are
# decided from THIS registry only — never from the broad COE_SPECIALTIES/
# SPECIALTY_REGISTRY taxonomy, which remains available ONLY for
# supporting-specialty/referral/doctor-association classification (see
# the module's "SEPARATE TWO DIFFERENT CONCEPTS" requirement). A
# specialty's presence in COE_SPECIALTIES (e.g. Dental under Headache)
# never implies it belongs here — Dental, Ophthalmology, ENT, Cardiology,
# Psychiatry, Neurology (on its own), Nutrition, General Surgery,
# Orthopedics, Diabetic Educator, and generic Allergy/Pulmonology
# mentions are all DELIBERATELY absent from this registry: they may still
# appear as supporting specialties inside an ALREADY-established context,
# but they must never independently create the recommendation
# requirement (see build_coe_evaluations' patient_eligible computation).
COE_RECOMMENDATION_TRIGGERS: dict[str, dict[str, list[str]]] = COE_COMPLAINTS


def resolve_approved_complaint(text: str) -> tuple[str, str, str] | None:
    """Deterministic, phrase-aware, longest-alias-wins resolution of the
    approved complaint CATEGORY named in *text*, as (coe, category,
    matched_alias) — or None when no approved-complaint alias is present.
    Never expanded beyond COE_COMPLAINTS' literal alias lists — see the
    module's "DO NOT EXPAND THE LIST SEMANTICALLY" requirement; a vague,
    unlisted symptom always returns None here, regardless of general
    medical plausibility."""
    tokens = _tokens(text)
    if not tokens:
        return None
    best: tuple[str, str, str, int] | None = None
    for coe, categories in COE_COMPLAINTS.items():
        for category, aliases in categories.items():
            for alias in aliases:
                alias_tokens = _tokens(alias)
                if alias_tokens and _contains_phrase(tokens, alias):
                    if best is None or len(alias_tokens) > best[3]:
                        best = (coe, category, alias, len(alias_tokens))
    return (best[0], best[1], best[2]) if best else None


def detect_approved_complaints(text: str) -> list[tuple[str, str, str]]:
    """Every (coe, category, matched_alias) approved-complaint mention in
    *text* — unlike resolve_approved_complaint (single best match across
    ALL coes), this returns one entry per COE that has ANY match, e.g. a
    turn describing both a chronic headache AND a separate liver
    complaint. WITHIN one COE, only the single LONGEST/most specific
    matching category wins (mirrors resolve_canonical_specialty's
    longest-alias-wins design) — e.g. "الربو الشعبي" (2 tokens,
    bronchial_asthma) must win over the bare "الربو" (1 token, asthma)
    it also happens to contain, never reporting the less specific
    category merely because of dict iteration order."""
    tokens = _tokens(text)
    if not tokens:
        return []
    best_per_coe: dict[str, tuple[str, str, int]] = {}
    for coe, categories in COE_COMPLAINTS.items():
        for category, aliases in categories.items():
            for alias in aliases:
                alias_tokens = _tokens(alias)
                if alias_tokens and _contains_phrase(tokens, alias):
                    if coe not in best_per_coe or len(alias_tokens) > best_per_coe[coe][2]:
                        best_per_coe[coe] = (category, alias, len(alias_tokens))
    return [(coe, category, alias) for coe, (category, alias, _length) in best_per_coe.items()]

# A complaint keyword found in a clause naming a THIRD PARTY (a family
# member) or under NEGATION is never the PATIENT's own active complaint —
# see ELIGIBILITY DETECTION's "do not trigger from ... a negated complaint
# / a family member's condition when the patient is booking for something
# else". Deliberately clause-scoped (via _split_clauses below), never
# whole-turn, so an unrelated family-member mention earlier in the SAME
# turn never suppresses the patient's own, separately-stated complaint.
_THIRD_PARTY_SUBJECT_MARKERS: set[str] = {
    "امي", "أمي", "ابويا", "أبويا", "والدي", "والدتي", "اخويا", "أخويا",
    "اخي", "أخي", "اختي", "أختي", "زوجي", "جوزي", "زوجتي", "مراتي",
    "ابني", "ابنتي", "بنتي", "جدي", "جدتي", "عمي", "عمتي", "خالي", "خالتي",
    "my mother", "my father", "my son", "my daughter", "my husband", "my wife",
}
_NEGATION_MARKERS: set[str] = {
    "مش", "مافيش", "مفيش", "ماعندي", "ما عندي", "معنديش", "بدون", "لا يوجد",
    "no longer", "not anymore",
}

# A complaint/specialty mention framed as PAST/RESOLVED ("كان عندي ... لكنه
# انتهى" — "I HAD ... but it's over") or as a HYPOTHETICAL ("لو كان عندي"
# — "if I had") is never the patient's CURRENT, active reason for seeking
# service — see ACTIVE PATIENT NEED's "a historical resolved condition" /
# "a hypothetical question" exclusions.
_HISTORICAL_RESOLVED_MARKERS: set[str] = {
    "كان عندي", "كان عندها", "كان عندك", "انتهى", "انتهت", "خلص", "خلصت",
    "اتعالجت", "اتعالج", "ماعادش", "ما عادش", "قبل كده", "من زمان وخلص",
    "used to have", "no longer have", "resolved now", "it's over now",
}
_HYPOTHETICAL_MARKERS: set[str] = {
    "لو كان", "لو كنت", "لو عندي", "افرض", "بفرض", "لو حصل", "ايه لو",
    "if i had", "what if", "hypothetically",
}

# ── Completed diagnostic results inquiry ─────────────────────────────────
# A patient asking to retrieve/view/download an ALREADY-COMPLETED lab or
# radiology result is never a COE-eligible need on its own — a test NAME
# ("سكر تراكمي"/"كوليسترول"/HbA1c/CBC/...) is an investigation, never a
# diagnosis, complaint, or specialty request by itself (see "TEST NAMES
# ARE NOT DIAGNOSES OR SPECIALTIES"). These are deliberately CONTEXTUAL
# PHRASES (an actual completed-result marker), never a bare "تحليل"/
# "أشعة" — those alone are ambiguous (could equally be a NEW test
# request, itself still not COE-eligible on its own, see the module's
# "DISTINGUISH RESULTS FROM NEW CARE" rules, but distinct from ASKING FOR
# an existing one).
_COMPLETED_RESULTS_MARKERS: set[str] = {
    "اخر تحليل", "آخر تحليل", "نتيجة التحليل", "نتيجة تحليل", "نتائج التحاليل",
    "نتيجة الأشعة", "نتيجة الاشعة", "تقرير الأشعة", "تقرير الاشعة",
    "آخر أشعة", "اخر اشعة", "التحاليل السابقة", "الأشعة السابقة", "الاشعة القديمة",
    "التحليل اللي عملته", "الأشعة اللي عملتها", "الاشعة اللي سويتها",
    "طلعت النتيجة", "ظهرت النتيجة", "أريد نسخة من التحليل", "اريد نسخة من الاشعة",
    "نسخة من التحاليل", "تحميل التحاليل", "تنزيل التقرير", "أرسل لي النتيجة",
    "أشوف النتيجة", "وين النتيجة", "فين النتيجة", "فين نتائج التحاليل",
    # Generalized "نتيجة/تقرير + [organ/test noun]" forms — "نتيجة" ("the
    # result of") on its OWN is unambiguous enough combined with any of
    # these common follow-on nouns (unlike bare "تحليل"/"أشعة" alone,
    # which stay excluded per "Do not classify based on تحليل or أشعة
    # alone").
    "نتيجة وظائف", "نتيجة فحص", "نتيجة فحوصات", "نتيجة الفحص", "نتيجة فحوصاتي",
    "تقرير وظائف", "تقرير الفحص",
    "latest result", "previous result", "test result", "lab result",
    "radiology result", "scan result", "download my report", "view my report",
    "send me my result", "previous analysis", "completed test",
}


def _clause_is_completed_results_inquiry(clause: str) -> bool:
    """True when *clause* is asking to retrieve/view/download an already-
    completed lab or radiology result — a CONTEXTUAL phrase match only
    (see _COMPLETED_RESULTS_MARKERS' docstring), never triggered by a bare
    "تحليل"/"أشعة" alone."""
    tokens = _tokens(clause)
    return any(_contains_phrase(tokens, m) for m in _COMPLETED_RESULTS_MARKERS)


def _turn_is_bare_continuation(text: str) -> bool:
    """True when *text* is a SHORT continuation with no desire-verb/
    booking-intent language of its own (e.g. "وكوليسترول" following "اريد
    اخر تحليل سكر تراكمي") — used ONLY to extend an ALREADY-established
    completed-results-inquiry topic from the immediately preceding patient
    turn to a bare follow-up naming another test, never to exclude a
    genuinely new, independently-stated need (see
    _completed_results_turn_flags)."""
    tokens = _tokens(text)
    if _patient_turn_has_booking_intent(text):
        return False
    # Strip a single leading conjunction token ("و"/"وبعدها"-style is
    # already a separate word after tokenising) before counting length.
    if tokens and tokens[0] in {"و", "وكمان", "كمان"}:
        tokens = tokens[1:]
    return 0 < len(tokens) <= 3


def _completed_results_turn_flags(turns: list[tuple[str, str]]) -> list[bool]:
    """One flag per turn (False for non-Patient turns): True when that
    Patient turn is part of a completed-diagnostic-results inquiry —
    either directly (a completed-results marker) or as a bare, verb-less
    continuation of the IMMEDIATELY PRECEDING Patient turn's own
    results-inquiry (e.g. "وكوليسترول" after "اريد اخر تحليل سكر
    تراكمي") — never propagated across a turn that introduces its own
    genuine booking-intent/complaint (a real topic change, see "Mixed
    intent" in ACTIVE PATIENT NEED)."""
    flags: list[bool] = [False] * len(turns)
    last_patient_was_results = False
    for i, (speaker, text) in enumerate(turns):
        if speaker != "patient":
            continue
        if _clause_is_completed_results_inquiry(text):
            flags[i] = True
        elif last_patient_was_results and _turn_is_bare_continuation(text):
            flags[i] = True
        last_patient_was_results = flags[i]
    return flags


def _clause_is_third_party_or_negated(clause: str) -> bool:
    """True when *clause* names a THIRD PARTY (a family member), is under
    NEGATION, describes a PAST/RESOLVED condition, or poses a
    HYPOTHETICAL — none of these are the patient's own CURRENT, active
    reason for seeking service (see ACTIVE PATIENT NEED). Phrase/token-
    aware (via _contains_phrase, defined below) — never a naive substring
    check, which would false-positive on e.g. "مشكلة" ("problem") merely
    because it happens to START WITH the negation marker "مش" ("not")."""
    tokens = _tokens(clause)
    return (
        any(_contains_phrase(tokens, m) for m in _THIRD_PARTY_SUBJECT_MARKERS)
        or any(_contains_phrase(tokens, m) for m in _NEGATION_MARKERS)
        or any(_contains_phrase(tokens, m) for m in _HISTORICAL_RESOLVED_MARKERS)
        or any(_contains_phrase(tokens, m) for m in _HYPOTHETICAL_MARKERS)
    )

# ── COE-name recognition markers (used to tell WHICH COE the Agent actually
# recommended/confirmed — deliberately more specific than the bare trigger
# phrases below, so a generic "مركز التميز" mention alone still falls back
# to script-similarity matching rather than a wrong specialty guess) ────────
#
# IBD deliberately does NOT list the bare specialty/complaint phrases
# "الجهاز الهضمي"/"امراض الجهاز الهضمي" here (unlike the other three COEs,
# whose markers are specific enough not to double as generic language) —
# those exact words are also a doctor's ordinary specialty/title
# description (e.g. "استشاري امراض الجهاز الهضمي والكبد والمناظير" naming a
# GIT consultant while confirming a booking) and are already fully covered
# as evidence via COMPLAINT_KEYWORDS/SPECIALTY_ALIASES["GIT"]. Keeping them
# here would count a doctor's specialty title as if the agent had
# explicitly announced "the IBD Center of Excellence", which is exactly
# the false cross-context "explicit recommendation" that previously
# corrupted multi-context aggregation (see build_coe_evaluations'
# explicit_agent_recommended field and its module-level regression note).
COE_NAME_MARKERS: dict[str, set[str]] = {
    "IBD": {"gastroenterology", "ibd"},
    # Bare "صداع" (no definite article) deliberately mirrors COMPLAINT_
    # KEYWORDS["Headache"]'s own bare form, so a prefixed/attached spelling
    # like "للصداع" ("for the headache", in e.g. "مركز التميز للصداع") still
    # matches — "الصداع" alone does NOT match "للصداع" as a substring
    # (Arabic's ل+ال contraction drops the alef), which previously made an
    # agent's own campaign-branded COE announcement invisible to this
    # marker even though the exact same word already counts as complaint
    # evidence when the PATIENT says it.
    "Headache": {"تشخيص وعلاج الصداع", "علاج الصداع", "صداع", "headache"},
    # Bare "ربو"/"سكر" (no definite article) mirror the SAME "صداع" fix
    # above for the SAME reason — "مركز التميز للربو"/"...للسكر" use the
    # ل+ال contraction ("للربو"/"للسكر"), which "الربو"/"السكري" alone
    # would never match as a substring. Both words are already accepted,
    # by the same precedent as "صداع", as sufficiently COE-specific
    # (never a generic complaint-only word) to serve as an explicit-
    # recommendation marker.
    "Asthma": {
        "امراض الصدر والجهاز التنفسي", "امراض الصدر", "الصدر والجهاز التنفسي", "asthma", "ربو",
    },
    "Diabetes": {"امراض السكر والغدد الصماء", "السكر والغدد الصماء", "diabetes", "سكر"},
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

# COE -> the canonical specialties organizationally grouped under it. This
# is the ROOT of the whole specialty taxonomy — the confirmed business
# taxonomy, authored ONCE, here. "ENT" deliberately appears under BOTH
# Headache and Asthma — a genuinely SHARED specialty that must never be
# arbitrarily resolved to one COE on its own (see resolve_specialty_coes /
# the disambiguation priority order documented on detect_specialty_
# mentions). Every other structure below (SPECIALTY_REGISTRY,
# SPECIALTY_TO_COES, SPECIALTY_ALIASES, WEAK_SPECIALTY_ALIASES,
# SPECIALTY_MARKERS, CAMPAIGN_COE_CONTEXT_MARKERS) is DERIVED from this
# dict plus the alias data below — never a second, independently-authored
# copy of the COE/specialty membership.
COE_SPECIALTIES: dict[str, list[str]] = {
    "IBD": ["GIT", "Nutrition", "General Surgery"],
    "Headache": ["Neurology", "Ophthalmology", "ENT", "Cardiology", "Psychiatry", "Dental"],
    "Diabetes": ["Diabetes", "Diabetic Educator", "Orthopedics"],
    "Asthma": ["Pulmonology", "ENT", "Allergy & Immunology"],
}

# Each COE's normally-PRIMARY specialty (its "first clinic") — used ONLY
# for the advisory is_primary_specialty_for_coe/is_supporting_specialty_
# for_coe classification below, NEVER for doctor approval (see
# "PRIMARY VS SUPPORTING SPECIALTIES" in the module docstring): specialty
# membership determines COE eligibility, not primary-doctor status. Every
# specialty is canonical-key-matched here (unlike the human-readable
# FIRST_CLINIC display strings above, e.g. "Gastroenterology"/"Diabetes/
# Endocrinology", which are for display/reference text only).
_FIRST_CLINIC_SPECIALTY: dict[str, str] = {
    "IBD": "GIT",
    "Headache": "Neurology",
    "Asthma": "Pulmonology",
    "Diabetes": "Diabetes",
}

# Canonical specialty -> English/Arabic aliases (common spelling/spacing/
# transliteration variants) — the SOLE place alias lists are hand-authored.
# Matched phrase-aware (see _contains_phrase), never via naive substring
# containment — several of these aliases are short enough (e.g. "قلب",
# "كبد", "سكر", "عظام", "عيون", "صدر", "ENT", "GIT") that blind substring
# matching could false-positive inside an unrelated longer word (e.g.
# "قلب" inside "انقلاب").
_SPECIALTY_ALIAS_DATA: dict[str, list[str]] = {
    "GIT": [
        "GIT", "Gastroenterology", "Gastrointestinal", "Digestive system", "Digestive diseases",
        "Digestive", "Hepatology", "Liver", "Colon", "Colorectal",
        "الجهاز الهضمي", "جهاز هضمي", "أمراض الجهاز الهضمي", "امراض الجهاز الهضمي", "هضمي",
        "كبد", "الكبد", "أمراض الكبد", "امراض الكبد", "قولون", "القولون",
    ],
    "Nutrition": [
        "Nutrition", "Clinical Nutrition", "Dietitian", "Dietetics",
        "تغذية", "التغذية", "تغذية علاجية", "التغذية العلاجية", "أخصائي تغذية", "اخصائي تغذية",
    ],
    "General Surgery": [
        "General Surgery", "General Surgeon", "جراحة عامة", "الجراحة العامة", "جراح عام",
    ],
    "Neurology": [
        "Neurology", "Neurologist", "Neurological", "مخ وأعصاب", "مخ واعصاب",
        "المخ والأعصاب", "المخ والاعصاب", "مخ و اعصاب", "مخ و أعصاب",
        "أعصاب", "اعصاب", "الأعصاب", "الاعصاب",
    ],
    "Ophthalmology": [
        "Ophthalmology", "Ophthalmologist", "Eye Clinic", "Eye Doctor",
        "طب العيون", "عيون", "طبيب عيون", "دكتور عيون",
    ],
    "ENT": [
        "ENT", "Ear, Nose and Throat", "Otolaryngology",
        "أنف وأذن وحنجرة", "انف واذن وحنجرة", "أنف اذن حنجرة", "انف اذن حنجرة",
        "أنف وأذن", "انف واذن",
    ],
    "Cardiology": [
        "Cardiology", "Cardiologist", "Heart Clinic",
        "قلب", "القلب", "طب القلب", "أمراض القلب", "امراض القلب", "دكتور قلب",
    ],
    "Psychiatry": [
        "Psychiatry", "Psychiatrist", "Mental Health",
        "طب نفسي", "طبيب نفسي", "نفسي", "الصحة النفسية", "صحة نفسية",
    ],
    "Dental": [
        "Dental", "Dentistry", "Dentist",
        "أسنان", "اسنان", "طب الأسنان", "طب الاسنان", "طبيب أسنان", "دكتور أسنان",
    ],
    "Diabetes": [
        "Diabetes", "Diabetology", "Diabetologist",
        "سكري", "السكري", "سكر", "مرض السكر", "عيادة السكر", "طبيب سكر",
    ],
    "Diabetic Educator": [
        "Diabetic Educator", "Diabetes Educator", "Diabetes Education",
        "مثقف سكري", "مثقفة سكري", "مثقف السكر", "مثقفة السكر",
        "تثقيف سكري", "التثقيف السكري", "تثقيف مرضى السكر",
    ],
    "Orthopedics": [
        "Orthopedics", "Orthopedic", "Orthopaedics", "Orthopaedic",
        "عظام", "العظام", "جراحة العظام", "طبيب عظام", "دكتور عظام",
    ],
    "Pulmonology": [
        "Pulmonology", "Pulmonologist", "Pulmonary", "Respiratory", "Respiratory Medicine",
        "Chest", "Chest Clinic",
        "صدر", "صدرية", "أمراض الصدر", "امراض الصدر", "طب الصدر",
        "الجهاز التنفسي", "جهاز تنفسي", "أمراض الجهاز التنفسي", "امراض الجهاز التنفسي",
    ],
    "Allergy & Immunology": [
        "Allergy & Immunology", "Allergy and Immunology", "Allergist", "Immunology", "Allergy",
        "حساسية ومناعة", "الحساسية والمناعة", "حساسية", "الحساسية", "مناعة", "المناعة",
        "طبيب حساسية", "عيادة الحساسية",
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
_WEAK_SPECIALTY_ALIAS_DATA: dict[str, list[str]] = {
    "GIT": ["مناظير", "منظار"],
}


def _build_specialty_registry() -> dict[str, dict[str, Any]]:
    """The single authoritative, specialty-keyed registry — built from
    COE_SPECIALTIES (membership) plus _SPECIALTY_ALIAS_DATA/
    _WEAK_SPECIALTY_ALIAS_DATA (aliases) above. Every other specialty-
    aware structure in this module (SPECIALTY_ALIASES, WEAK_SPECIALTY_
    ALIASES, SPECIALTY_TO_COES, SPECIALTY_MARKERS, CAMPAIGN_COE_CONTEXT_
    MARKERS) is a thin DERIVED view of this registry, kept only for
    backward-compatible naming — none of them is ever hand-authored a
    second time. A specialty is "shared" precisely when len(coes) > 1
    (e.g. ENT -> ["Headache", "Asthma"]) — never a separately-tracked
    boolean that could drift out of sync with its own coes list."""
    registry: dict[str, dict[str, Any]] = {}
    for coe, specialties in COE_SPECIALTIES.items():
        for specialty in specialties:
            entry = registry.setdefault(specialty, {"coes": [], "aliases": [], "weak_aliases": []})
            if coe not in entry["coes"]:
                entry["coes"].append(coe)
    for specialty, entry in registry.items():
        entry["aliases"] = list(_SPECIALTY_ALIAS_DATA.get(specialty, []))
        entry["weak_aliases"] = list(_WEAK_SPECIALTY_ALIAS_DATA.get(specialty, []))
    return registry


# ── THE authoritative specialty registry ────────────────────────────────────
# {canonical_specialty: {"coes": [...], "aliases": [...], "weak_aliases":
# [...]}} — the single source every specialty-aware layer of the pipeline
# (routing, patient-need detection, context creation, shared-specialty
# disambiguation, doctor-to-context association, prompt reference data,
# LLM-output grounding, service-alignment/recommendation validation,
# logging) reads from, directly or via one of the thin derived views below.
SPECIALTY_REGISTRY: dict[str, dict[str, Any]] = _build_specialty_registry()

# Thin derived views — kept under their existing names for every caller
# and test already using them; each is computed ONCE from SPECIALTY_
# REGISTRY, never independently authored.
SPECIALTY_ALIASES: dict[str, list[str]] = {sp: meta["aliases"] for sp, meta in SPECIALTY_REGISTRY.items()}
WEAK_SPECIALTY_ALIASES: dict[str, list[str]] = {
    sp: meta["weak_aliases"] for sp, meta in SPECIALTY_REGISTRY.items() if meta["weak_aliases"]
}

# Canonical specialty -> every COE it organizationally supports. Length 1
# for an unambiguous specialty (e.g. "Neurology" -> ["Headache"]), length
# 2+ for a genuinely SHARED specialty (e.g. "ENT" -> ["Headache",
# "Asthma"]) that must be disambiguated, never guessed (see
# detect_specialty_mentions).
SPECIALTY_TO_COES: dict[str, list[str]] = {sp: list(meta["coes"]) for sp, meta in SPECIALTY_REGISTRY.items()}


def is_shared_specialty(specialty: str) -> bool:
    """True when *specialty* organizationally supports more than one COE
    (e.g. ENT) — derived from SPECIALTY_TO_COES, never a separately
    tracked flag."""
    return len(SPECIALTY_TO_COES.get(specialty, [])) > 1


def is_primary_specialty_for_coe(specialty: str, coe: str) -> bool:
    """Advisory classification ONLY (see "PRIMARY VS SUPPORTING
    SPECIALTIES" in the module docstring) — whether *specialty* is *coe*'s
    normally-primary ("first clinic") specialty. Specialty membership
    determines COE eligibility and the recommendation requirement; it
    NEVER determines primary-doctor approval on its own, and this
    function is never consulted by evaluate_context_doctors or any
    pass/fail decision."""
    return _FIRST_CLINIC_SPECIALTY.get(coe) == specialty


def is_supporting_specialty_for_coe(specialty: str, coe: str) -> bool:
    """The complement of is_primary_specialty_for_coe — *specialty*
    belongs to *coe* but is not its normally-primary specialty. False for
    a specialty that doesn't belong to *coe* at all (see SPECIALTY_TO_
    COES). Advisory/reporting only — see is_primary_specialty_for_coe."""
    return coe in SPECIALTY_TO_COES.get(specialty, []) and not is_primary_specialty_for_coe(specialty, coe)


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


# Common Arabic function-word prefixes (preposition/conjunction + optional
# definite article, including the ل+ال -> لل contraction) — tried LONGEST
# cluster first so e.g. "بال" is stripped as one unit rather than leaving
# a stray "ال". Used ONLY by _contains_phrase_arabic_prefix_tolerant below,
# for a small set of deliberately bare, single-word EXPLICIT markers (see
# COE_NAME_MARKERS) where an attached preposition ("للصداع"/"للربو"/
# "للسكر") must still match the bare word ("صداع"/"ربو"/"سكر") — never
# used for SPECIALTY_ALIASES matching, which already lists every needed
# "ال"-prefixed form explicitly.
_ARABIC_PREFIX_STRIP_RE = re.compile(r"^(?:وال|فال|بال|كال|لل|ال|و|ف|ب|ل|ك)")


def _contains_phrase_arabic_prefix_tolerant(haystack_tokens: list[str], needle: str) -> bool:
    """Like _contains_phrase, but for a single-token *needle* also matches
    a haystack token that equals *needle* once ONE leading Arabic
    preposition/article prefix cluster is stripped from it (see
    _ARABIC_PREFIX_STRIP_RE) — e.g. needle "صداع" matches the token
    "للصداع" ("لل" stripped) but never merely a SUBSTRING match: "السكري"
    stripped of "ال" is "سكري", which still does not equal "سكر", so it
    correctly does NOT match (the residual must be an EXACT token match,
    not a further substring check) — this is what keeps a short marker
    safe against a longer, morphologically DIFFERENT word (see the
    _contains_phrase docstring's "قلب" vs "انقلاب" example, which applies
    here identically)."""
    if _contains_phrase(haystack_tokens, needle):
        return True
    needle_tokens = _tokens(needle)
    if len(needle_tokens) != 1:
        return False  # only single bare-word markers need this tolerance
    needle_token = needle_tokens[0]
    return any(_ARABIC_PREFIX_STRIP_RE.sub("", t, count=1) == needle_token for t in haystack_tokens)


def _contains_multiword_phrase_prefix_tolerant(haystack_tokens: list[str], needle: str) -> bool:
    """Like _contains_phrase_arabic_prefix_tolerant, but ALSO tolerant for
    a MULTI-token *needle* whose FIRST word carries an attached Arabic
    preposition/article prefix in the haystack — e.g. needle "مركز
    التميز" matches the haystack token sequence ["بمركز", "التميز"] once
    the leading "ب" is stripped from "بمركز" ("بمركز التميز" — "by/in the
    center of excellence" — is exactly how this phrase is normally
    attached in a sentence, e.g. "سيتم حجز موعد لحضرتك بمركز التميز..."). Only
    the phrase's OWN first token is ever prefix-stripped, and the
    remaining tokens must still match EXACTLY — never a substring/naive
    match (see _contains_phrase_arabic_prefix_tolerant's docstring for the
    same "قلب" vs "انقلاب" safety rationale)."""
    if _contains_phrase(haystack_tokens, needle):
        return True
    needle_tokens = _tokens(needle)
    if not needle_tokens:
        return False
    n = len(needle_tokens)
    for i in range(len(haystack_tokens) - n + 1):
        window = haystack_tokens[i:i + n]
        stripped_first = _ARABIC_PREFIX_STRIP_RE.sub("", window[0], count=1)
        if [stripped_first, *window[1:]] == needle_tokens:
            return True
    return False


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


def _matched_alias_for_specialty(text: str, canonical_specialty: str) -> tuple[str, str] | None:
    """The single LONGEST alias of *canonical_specialty* (from either
    SPECIALTY_ALIASES or WEAK_SPECIALTY_ALIASES) that appears as a phrase
    in *text*, as (matched_alias, raw_span):
      - matched_alias — the alias-table entry itself (a stable, canonical
        spelling, for grounding/reporting).
      - raw_span — the ACTUAL substring from *text* at that position (the
        literal wording as said/typed, diacritics/spelling-variant
        preserved) when a positional token-count mapping is safe, else
        the same value as matched_alias.
    None when no alias of this specialty is present."""
    tokens = _tokens(text)
    if not tokens:
        return None
    raw_tokens = text.split()
    positional_mapping_safe = len(raw_tokens) == len(tokens)
    candidates = list(SPECIALTY_ALIASES.get(canonical_specialty, [])) + list(
        WEAK_SPECIALTY_ALIASES.get(canonical_specialty, [])
    )
    best, best_len, best_start = None, 0, None
    for alias in candidates:
        alias_tokens = _tokens(alias)
        n = len(alias_tokens)
        if not alias_tokens or n <= best_len:
            continue
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == alias_tokens:
                best, best_len, best_start = alias, n, i
                break
    if best is None:
        return None
    if positional_mapping_safe and best_start is not None:
        return best, " ".join(raw_tokens[best_start:best_start + best_len])
    return best, best


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


# ═════════════════════════════════════════════════════════════════════════
# CAMPAIGN DETECTION IS NOT CAMPAIGN ENGAGEMENT — a structured campaign
# identifier (see campaign_origin_evidence) proves only WHERE the
# conversation originated; it never proves that the patient's substantive
# inquiry is actually about that campaign's COE. classify_campaign_relevance
# separates the two: campaign_detected/campaign_candidate_coe describe the
# campaign message itself, while campaign_relevance/active_campaign_coe
# describe whether the REST of the call actually continues that topic.
# Only an "engaged" campaign may ever create patient eligibility or a
# campaign COE evaluation context (see build_coe_evaluations' patient_
# eligible computation and build_coe_contexts' context-admission filter).
# ═════════════════════════════════════════════════════════════════════════

# Greetings, acknowledgements, and other non-substantive turns — never
# counted as the patient's substantive inquiry when determining campaign
# engagement (see "DETERMINE THE SUBSTANTIVE PATIENT INTENT": ignore bot
# language selection, menus, handoff messages, greetings, names,
# insurance, IDs, and other administrative turns).
_ADMINISTRATIVE_TURN_MARKERS: set[str] = {
    "السلام عليكم", "وعليكم السلام", "مرحبا", "اهلا", "أهلا", "هاي", "شكرا", "شكراً",
    "تمام", "طيب", "اوك", "ok", "okay", "نعم", "ايوه", "أيوه", "لا", "yes", "no",
    "hello", "hi", "thanks", "thank you",
}


def _turn_is_administrative(text: str) -> bool:
    """True for a greeting, plain acknowledgement, bot menu selection, or
    other non-substantive turn with no clinical/booking content of its
    own — never treated as the patient's substantive inquiry for campaign-
    relevance purposes. A short reply (<=2 tokens) is administrative
    unless it independently carries booking-intent, approved-complaint, or
    specialty content (e.g. "ابغى دكتور" is short but substantive)."""
    tokens = _tokens(text)
    if not tokens:
        return True
    if _norm(text) in {_norm(m) for m in _ADMINISTRATIVE_TURN_MARKERS}:
        return True
    if len(tokens) <= 2 and not (
        _patient_turn_has_booking_intent(text)
        or detect_approved_complaints(text)
        or detect_specialty_mentions(text)
    ):
        return True
    return False


def _turn_shows_campaign_engagement(text: str, candidate_coe: str) -> bool:
    """True when *text* (a Patient or Agent turn found AFTER the campaign
    click) shows the campaign's candidate COE remains the ACTIVE topic —
    an approved complaint category mapped to it (see COE_RECOMMENDATION_
    TRIGGERS), or explicit COE-name/package language naming it (see
    COE_NAME_MARKERS, which already covers phrasing like "باقة الصداع"/
    "تفاصيل باقة الصداع" via its bare category-word markers). Deliberately
    narrower than a bare mapped-specialty mention (e.g. Dental under
    Headache) — per the confirmed business rule, a shared/supporting
    specialty alone is never evidence that the campaign's OWN topic is
    still active (see COE_RECOMMENDATION_TRIGGERS' docstring)."""
    for coe, _category, _alias in detect_approved_complaints(text):
        if coe == candidate_coe:
            return True
    tokens = _tokens(text)
    return any(
        _contains_phrase_arabic_prefix_tolerant(tokens, m)
        for m in COE_NAME_MARKERS.get(candidate_coe, ())
    )


def classify_campaign_relevance(call: CallTranscript) -> dict[str, Any]:
    """Determine whether a detected COE marketing campaign is actually the
    ACTIVE topic of this call, never merely assumed from its presence.

    Returns a dict with:
      - campaign_detected: whether a structured campaign identifier (see
        campaign_origin_evidence) was found at all.
      - campaign_candidate_coe: the COE that campaign message's OWN text
        identifies (see resolve_campaign_coe) — set even when the
        campaign turns out to be diverted; None when the campaign text
        itself names no single unambiguous COE.
      - campaign_relevance: one of "engaged" (the later conversation
        substantively continues the campaign's COE topic — see
        _turn_shows_campaign_engagement), "diverted" (the patient makes a
        clear, unrelated substantive request and never returns to the
        campaign topic), "pending" (only the campaign click and/or
        administrative turns exist — no substantive patient inquiry at
        all), or "uncertain" (the campaign text itself names no single
        COE, so relevance cannot be safely classified); None when no
        campaign was detected at all.
      - active_campaign_coe: campaign_candidate_coe when campaign_relevance
        is "engaged", else None — this is the ONLY campaign value ever
        allowed to create patient eligibility or an evaluation context.
      - campaign_relevance_evidence: the verbatim turn excerpt that
        justifies the relevance verdict (the engaging turn, or the first
        diverting turn) — None for "pending" (nothing substantive exists)
        and "uncertain".

    A later return to the campaign topic (after an earlier diverting turn)
    still activates it — every turn after the campaign click is scanned in
    order, and engagement evidence found ANYWHERE wins over any earlier
    diverting turns (see "If the patient later returns to the Headache
    campaign topic, activate it from that later evidence").
    """
    turns = split_transcript_turns(call.transcript)
    campaign_turn_idx: int | None = None
    campaign_candidate_coe: str | None = None
    campaign_evidence: str | None = None
    for idx, (speaker, text) in enumerate(turns):
        if speaker == "patient" and _CAMPAIGN_ORIGIN_RE.search(text):
            campaign_turn_idx = idx
            campaign_evidence = text.strip()[:300]
            matches = _unambiguous_coe_matches(text)
            campaign_candidate_coe = matches[0] if len(matches) == 1 else None
            break

    if campaign_turn_idx is None:
        return {
            "campaign_detected": False,
            "campaign_candidate_coe": None,
            "campaign_relevance": None,
            "active_campaign_coe": None,
            "campaign_relevance_evidence": None,
        }

    if campaign_candidate_coe is None:
        # The campaign's OWN text names no single unambiguous COE (rare —
        # e.g. an ambiguous/multi-COE campaign identifier) — evidence is
        # insufficient to classify relevance safely either way.
        return {
            "campaign_detected": True,
            "campaign_candidate_coe": None,
            "campaign_relevance": "uncertain",
            "active_campaign_coe": None,
            "campaign_relevance_evidence": campaign_evidence,
        }

    substantive_found = False
    diverted_evidence: str | None = None
    engaged_evidence: str | None = None
    for idx in range(campaign_turn_idx + 1, len(turns)):
        speaker, text = turns[idx]
        if _turn_is_administrative(text):
            continue
        if speaker == "patient" and _clause_is_third_party_or_negated(text):
            continue
        if _turn_shows_campaign_engagement(text, campaign_candidate_coe):
            engaged_evidence = text.strip()[:300]
            break
        if speaker == "patient":
            substantive_found = True
            if diverted_evidence is None:
                diverted_evidence = text.strip()[:300]

    if engaged_evidence:
        relevance = "engaged"
    elif substantive_found:
        relevance = "diverted"
    else:
        relevance = "pending"

    return {
        "campaign_detected": True,
        "campaign_candidate_coe": campaign_candidate_coe,
        "campaign_relevance": relevance,
        "active_campaign_coe": campaign_candidate_coe if relevance == "engaged" else None,
        "campaign_relevance_evidence": engaged_evidence if relevance == "engaged" else diverted_evidence,
    }


# ── Patient booking-intent detection ──────────────────────────────────────
# Used by _turn_is_bare_continuation (to tell a genuinely new, actively-
# stated need apart from a bare follow-up naming another completed-results
# test — see _completed_results_turn_flags) and, historically, by the now-
# REMOVED specialty-only trigger path ("Path D" in earlier revisions of
# this module — a bare specialty/doctor request no longer independently
# triggers COE validation on its own; see COE_RECOMMENDATION_TRIGGERS'
# docstring). A standalone "I want/need" desire-verb TOKEN, anywhere in
# the turn, is enough on its own to count as booking intent — the patient
# may follow it with a booking noun ("ابغى موعد"), a doctor/availability
# question ("ابغى دكتور"), or a specialty name directly. Still gated by
# _clause_is_third_party_or_negated, so "مش عايز موعد"/"امي عايزة ..." are
# correctly excluded despite containing a desire verb.
_PATIENT_DESIRE_VERB_TOKENS: set[str] = {
    "ابغى", "ابي", "أبغى", "عايز", "عاوز", "عايزة", "عاوزة",
    "محتاج", "محتاجة", "احتاج", "أحتاج", "بدي", "اريد", "أريد",
}

# Phrases with no standalone desire-verb token of their own (asking who/
# what is already available is a direct continuation of an active booking
# request, never an incidental/historical mention on its own) plus a few
# English equivalents.
_PATIENT_BOOKING_INTENT_MARKERS: set[str] = {
    "مين الدكتور", "الدكتور الموجود", "متاح دكتور", "في دكتور متاح",
    "want an appointment", "need an appointment", "book an appointment",
    "i want a doctor", "i need a doctor",
}


_NORMALIZED_DESIRE_VERB_TOKENS: set[str] = {_norm(v) for v in _PATIENT_DESIRE_VERB_TOKENS}


def _patient_turn_has_booking_intent(text: str) -> bool:
    """True when *text* (one Patient turn) actively asks for a doctor,
    appointment, availability, or specialty — never a passive/incidental/
    historical mention on its own (see _PATIENT_DESIRE_VERB_TOKENS /
    _PATIENT_BOOKING_INTENT_MARKERS)."""
    tokens = _tokens(text)
    if any(t in _NORMALIZED_DESIRE_VERB_TOKENS for t in tokens):
        return True
    return any(_contains_phrase(tokens, m) for m in _PATIENT_BOOKING_INTENT_MARKERS)


# ── New diagnostic test request (without a doctor/specialty/COE need) ──────
# "اريد اعمل تحليل سكر"/"عايز أحجز أشعة" ask to HAVE A NEW TEST DONE — a
# diagnostic test alone (new OR completed-result retrieval) is never a
# specialty or COE booking on its own (see "DISTINGUISH RESULTS FROM NEW
# CARE"). Distinguished from a genuine doctor/clinic/specialty booking by
# the presence of an ACTUAL clinic/doctor/appointment word: "اريد اعمل
# تحليل سكر" (bare test only) is excluded, but "أنا مريض سكر وأريد أحجز
# عيادة السكر" (names a CLINIC) or "عندي قدم سكري وأحتاج دكتور" (names a
# DOCTOR) are not — the test-object word alone never overrides an
# actually-present clinic/doctor/appointment request in the SAME clause.
_TEST_REQUEST_OBJECT_MARKERS: set[str] = {
    "تحليل", "تحاليل", "أشعة", "اشعة", "فحص", "فحوصات", "سونار", "منظار", "رنين",
    "test", "analysis", "scan", "x-ray", "lab test",
}
_CLINIC_DOCTOR_APPOINTMENT_MARKERS: set[str] = {
    "دكتور", "طبيب", "عيادة", "استشاري", "اخصائي", "أخصائي", "موعد",
    "doctor", "clinic", "appointment", "specialist",
}


def _clause_is_bare_new_test_request(text: str) -> bool:
    """True when *text* asks to HAVE a NEW diagnostic test/scan performed,
    with no accompanying doctor/clinic/appointment word — the object of
    the patient's request is the TEST itself, not a specialty booking
    (see _TEST_REQUEST_OBJECT_MARKERS' docstring). Never true when a
    genuine clinic/doctor/appointment word is also present in the same
    text — that always signals an actual booking request instead."""
    tokens = _tokens(text)
    has_test_object = any(_contains_phrase(tokens, m) for m in _TEST_REQUEST_OBJECT_MARKERS)
    has_clinic_or_doctor = any(_contains_phrase(tokens, m) for m in _CLINIC_DOCTOR_APPOINTMENT_MARKERS)
    return has_test_object and not has_clinic_or_doctor


def _first_approved_complaint_match(
    turns: list[tuple[str, str]], results_flags: list[bool],
) -> tuple[str, str, str] | None:
    """Path E's own scan, factored out so it can be reused both for an
    ordinary (no-campaign) call and for a campaign call whose candidate
    was rejected as diverted/pending/uncertain (see CAMPAIGN DETECTION IS
    NOT CAMPAIGN ENGAGEMENT — "continue checking for another
    independently eligible COE need using the strict complaint/diagnosis
    rules"). Returns the first (coe, category, clause) match, or None —
    never a broader inferred category (see resolve_approved_complaint)."""
    for idx, (speaker, text) in enumerate(turns):
        if speaker != "patient":
            continue
        if results_flags[idx]:
            continue  # retrieving an already-completed lab/radiology result, not a complaint
        for clause in (_split_clauses(text) or [text]):
            if _clause_is_third_party_or_negated(clause):
                continue
            if detect_coe_mention(clause):
                # An explicit "مركز تميز"/COE-name mention (e.g. reciting
                # the official program description back), not ordinary
                # clinical complaint language — governed by Paths A-C's
                # own rules, never by Path E.
                continue
            match = resolve_approved_complaint(clause)
            if match:
                coe, category, _alias = match
                return coe, category, clause
    return None


def classify_coe_trigger(call: CallTranscript) -> dict[str, Any]:
    """Determine whether COE validation should run at all, and via which
    path — turn-order-aware and speaker-attributed, so a Patient statement
    is never misattributed as an Agent recommendation (see module
    docstring).

    Checks, in order:

      0. Campaign/post origin (Path C, "campaign_origin") — a STRUCTURED
         campaign identifier (see campaign_origin_evidence) found anywhere
         in the Patient's own turns. This identifies only a CANDIDATE COE
         (see classify_campaign_relevance) — it never, by itself,
         establishes that this conversation IS a COE conversation. Path C
         TRIGGERS only when campaign_relevance is "engaged" (the later
         conversation actually continues the campaign's own topic). When
         the campaign is "diverted", "pending", or "uncertain", the
         candidate alone is REJECTED — this function then falls through to
         Path E only (see below) to check for another, independently
         eligible COE need using the strict complaint/diagnosis rules;
         Path A/B's bare "مركز تميز" mention scan is skipped in this case,
         since the campaign turn's own text would otherwise always satisfy
         it regardless of relevance. Checked first because it does not
         depend on turn order the way Paths A/B do.

    Otherwise (no campaign detected at all) walks the transcript turn by
    turn (in original order) for an
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
      - If nobody mentions it at all -> checked against Path E
        (patient_approved_complaint) next, rather than immediately
        returning NOT triggered.

      Path E ("patient_approved_complaint") — checked only when NEITHER
      Path A/B/C above already triggered: a Patient turn actively
      describing one of the restricted, business-approved diagnosis/
      complaint categories (see COE_RECOMMENDATION_TRIGGERS) is enough on
      its own to require the human agent to have recommended the mapped
      COE — this validator exists specifically to catch a missed COE
      recommendation, so it must never require the COE to already have
      been named before checking whether it SHOULD have been. A bare
      specialty/doctor request with no approved complaint category present
      (e.g. Dental, or Neurology on its own) is deliberately NOT enough —
      see COE_RECOMMENDATION_TRIGGERS' docstring for the confirmed
      business rule superseding the earlier, broader specialty-based
      trigger.
    """
    # CAMPAIGN DETECTION IS NOT CAMPAIGN ENGAGEMENT — computed ONCE here,
    # the single authoritative source every downstream caller (nodes.py's
    # infer_coe_validation/skip_coe_validation, app.agent.graph's
    # _coe_intent_router) reuses via this function's own "campaign_info"
    # key, rather than each independently recomputing campaign relevance
    # (which could otherwise drift out of sync — see classify_coe_routing).
    campaign_info = classify_campaign_relevance(call)
    if campaign_info["campaign_detected"]:
        if campaign_info["campaign_relevance"] == "engaged":
            _campaign_click_evidence = campaign_origin_evidence(call)
            return {
                "triggered": True,
                "trigger_path": "campaign_origin",
                "trigger_reason": (
                    "The customer's message contains a COE marketing-campaign/post identifier, "
                    "and the patient's later inquiry confirms the advertised Center of Excellence "
                    "remains the active topic. This campaign message is marketing/system content, "
                    "not something the human agent wrote — it is never treated as an agent "
                    "recommendation or script delivery."
                ),
                "evidence": _campaign_click_evidence,
                "patient_evidence": _campaign_click_evidence,
                "campaign_info": campaign_info,
            }
        # Campaign detected but NOT engaged — the candidate alone is
        # REJECTED as a trigger (see CAMPAIGN CONTEXT ADMISSION). Continue
        # checking ONLY the strict complaint/diagnosis rules (Path E) for
        # another, independently eligible COE need — never Path A/B's bare
        # mention scan, which the campaign turn's own text would otherwise
        # always satisfy regardless of relevance.
        _turns_for_fallback = split_transcript_turns(call.transcript)
        _results_flags_for_fallback = _completed_results_turn_flags(_turns_for_fallback)
        fallback_match = _first_approved_complaint_match(_turns_for_fallback, _results_flags_for_fallback)
        if fallback_match:
            coe, category, clause = fallback_match
            return {
                "triggered": True,
                "trigger_path": "patient_approved_complaint",
                "trigger_reason": (
                    f"The campaign candidate ({campaign_info['campaign_candidate_coe']}) was "
                    f"{campaign_info['campaign_relevance']}, but the customer separately described "
                    f"an approved {category.replace('_', ' ')} complaint mapped to the {coe} Center "
                    "of Excellence — the human agent was still required to recommend/explain that "
                    "COE service."
                ),
                "evidence": clause.strip()[:300],
                "patient_evidence": clause.strip()[:300],
                "campaign_info": campaign_info,
            }
        return {
            "triggered": False,
            "trigger_path": None,
            # A short reason CODE (not a sentence) — mirrors the existing
            # "completed_diagnostic_results_inquiry" precedent — so a log
            # line never needs to embed the patient's actual complaint
            # text (see "avoid logging raw...complaints" requirement).
            "trigger_reason": f"campaign_{campaign_info['campaign_relevance']}_no_eligible_coe_need",
            "evidence": None,
            "patient_evidence": None,
            "campaign_info": campaign_info,
        }

    turns = split_transcript_turns(call.transcript)
    # Which Patient turns are part of a completed-diagnostic-results
    # inquiry (see _completed_results_turn_flags) — computed once, used to
    # exclude those turns from Paths D/E below, so a test-NAME word (e.g.
    # bare "سكر" inside "اخر تحليل سكر تراكمي") is never read as an
    # active specialty/complaint need on its own.
    results_flags = _completed_results_turn_flags(turns)
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
                "campaign_info": campaign_info,
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
            "campaign_info": campaign_info,
        }

    # Path D ("patient_specialty_booking_intent" — a request for ANY
    # specialty in the broad COE_SPECIALTIES/SPECIALTY_REGISTRY taxonomy
    # was, by itself, enough to trigger) is REMOVED — SUPERSEDED by the
    # confirmed business rule that the broad specialty taxonomy may be
    # used for supporting-specialty/referral/doctor-association
    # classification ONLY, never to decide recommendation_required/
    # missed_recommendation/the trigger itself (see "SEPARATE TWO
    # DIFFERENT CONCEPTS" and COE_RECOMMENDATION_TRIGGERS' docstring). A
    # bare specialty request — Dental, Ophthalmology, ENT, Cardiology,
    # Psychiatry, Neurology on its own, Nutrition, General Surgery,
    # Orthopedics, Diabetic Educator, generic Allergy/Pulmonology — no
    # longer independently triggers this validator; it may still
    # contribute supporting-specialty/doctor evidence WITHIN a context
    # already established by Path E below.

    # Path E — patient_approved_complaint. The SOLE remaining "patient's
    # own active need" trigger — a patient stating one of the restricted,
    # business-approved diagnosis/complaint categories (see
    # COE_RECOMMENDATION_TRIGGERS) is, on its own, enough to require the
    # human agent to have recommended the mapped COE, even with no COE
    # terminology at all (e.g. "عندي صداع نصفي وأريد أحجز له"). A vague,
    # unapproved symptom — OR a bare specialty/doctor request with no
    # approved category — never reaches this far (see
    # resolve_approved_complaint, which only ever recognises the literal
    # approved alias lists, never a broader inferred category).
    no_campaign_match = _first_approved_complaint_match(turns, results_flags)
    if no_campaign_match:
        coe, category, clause = no_campaign_match
        return {
            "triggered": True,
            "trigger_path": "patient_approved_complaint",
            "trigger_reason": (
                f"The customer actively described an approved {category.replace('_', ' ')} "
                f"complaint mapped to the {coe} Center of Excellence, without ever using COE "
                "terminology or requesting a specific specialty — the human agent was still "
                "required to recommend/explain that COE service."
            ),
            "evidence": clause.strip()[:300],
            "patient_evidence": clause.strip()[:300],
            "campaign_info": campaign_info,
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
            "campaign_info": campaign_info,
        }
    if any(results_flags):
        # A completed-diagnostic-results inquiry is not, on its own,
        # evidence of nothing — it's a specific, deliberate exclusion
        # (see COMPLETED RESULTS INQUIRY) called out separately here so
        # both the logs and the trigger reason are honest about WHY this
        # call was skipped, rather than reading "no discussion found" for
        # a call that plainly discussed test results.
        return {
            "triggered": False,
            "trigger_path": None,
            "trigger_reason": (
                "completed_diagnostic_results_inquiry: the customer is requesting an already-"
                "completed laboratory/radiology result, not booking care for a COE-related "
                "complaint or specialty."
            ),
            "evidence": None,
            "patient_evidence": None,
            "campaign_info": campaign_info,
        }
    return {
        "triggered": False,
        "trigger_path": None,
        "trigger_reason": "No Center of Excellence or specialized-center discussion was found.",
        "evidence": None,
        "patient_evidence": None,
        "campaign_info": campaign_info,
    }


def coe_validation_needed(call: CallTranscript, trigger_ctx: dict[str, Any] | None = None) -> bool:
    """Graph-router-friendly boolean wrapper around classify_coe_trigger."""
    ctx = trigger_ctx if trigger_ctx is not None else classify_coe_trigger(call)
    return bool(ctx.get("triggered"))


def classify_coe_routing(call: CallTranscript, trigger_ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    """The single, flat deterministic preflight result every caller that
    needs BOTH the campaign-relevance fields AND the trigger decision
    should use — wraps classify_coe_trigger (never a second,
    independently-computed campaign-relevance check, so the two can never
    diverge — see classify_coe_trigger's own "campaign_info" key, which
    this simply re-shapes). Pass an already-computed trigger_ctx (from a
    prior classify_coe_trigger call on this exact same call) to avoid
    recomputing it a second time.

    Returns:
      - campaign_detected, campaign_candidate, campaign_relevance,
        active_campaign_coe: see classify_campaign_relevance
        (campaign_candidate_coe renamed to campaign_candidate here).
      - eligible_patient_coes: every COE this preflight itself found an
        independently grounded active patient need for — the engaged
        campaign's own COE, plus the COE a Path E complaint match
        resolved, when applicable. Diagnostic/best-effort: the
        authoritative, complete multi-COE picture is still
        build_coe_evaluations' own coe_evaluations list.
      - triggered, trigger_path, reason: the routing decision itself
        (reason mirrors trigger_reason exactly — a short code for a
        campaign-rejected/completed-results case, a descriptive sentence
        otherwise).
    """
    ctx = trigger_ctx if trigger_ctx is not None else classify_coe_trigger(call)
    campaign_info = ctx.get("campaign_info") or classify_campaign_relevance(call)

    eligible_patient_coes: list[str] = []
    if campaign_info.get("active_campaign_coe"):
        eligible_patient_coes.append(campaign_info["active_campaign_coe"])
    if ctx["triggered"] and ctx["trigger_path"] == "patient_approved_complaint":
        match = resolve_approved_complaint(ctx.get("evidence") or "")
        if match and match[0] not in eligible_patient_coes:
            eligible_patient_coes.append(match[0])

    return {
        "campaign_detected": campaign_info["campaign_detected"],
        "campaign_candidate": campaign_info["campaign_candidate_coe"],
        "campaign_relevance": campaign_info["campaign_relevance"],
        "active_campaign_coe": campaign_info["active_campaign_coe"],
        "eligible_patient_coes": eligible_patient_coes,
        "triggered": ctx["triggered"],
        "trigger_path": ctx["trigger_path"],
        "reason": ctx["trigger_reason"],
    }


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


# ═════════════════════════════════════════════════════════════════════════
# SCRIPT SIMILARITY GATING — fuzzy script similarity must NEVER
# independently create or choose a COE category by itself. The four
# approved scripts share substantial boilerplate ("لضمان تحقيق أقصى
# استفادة...سيتم حجز موعد لحضرتك بمركز التميز المتخصص في...والذي يضم نخبة
# من أفضل الاستشاريين والأخصائيين"), so an agent's generic closing/wrap-up
# wording built ENTIRELY from that shared boilerplate scores extremely
# high (often 90-99) against EVERY script at once, with no category-
# specific content at all — comparing each COE's script independently
# against the same threshold would then credit whichever COE happens to
# win an arbitrary tie (see the reported false-IBD-match regression).
# Fuzzy similarity may therefore only ever CONFIRM a paraphrase of a
# category that is ALREADY independently, deterministically evidenced in
# the SAME turn: an explicit COE/service-program marker (e.g. "مركز
# التميز"/"برنامج مركز التميز"/"باقة مركز التميز"/"Center of Excellence")
# AND category-specific evidence for that specific candidate COE (a
# specialty or approved-complaint alias belonging to it). It can never
# invent a category from generic shared wording alone.
# ═════════════════════════════════════════════════════════════════════════

_EXPLICIT_COE_PROGRAM_MARKERS: tuple[str, ...] = (
    "مركز التميز", "مركز تميز", "برنامج مركز التميز", "باقة مركز التميز",
    "برنامج التميز", "باقة التميز", "center of excellence", "coe program", "coe package",
)


def _turn_has_explicit_coe_program_marker(text: str) -> bool:
    """True when *text* contains an explicit COE/service-program marker
    (see _EXPLICIT_COE_PROGRAM_MARKERS) — the deterministic PREREQUISITE
    that must hold before script similarity is ever consulted at all (see
    SCRIPT SIMILARITY GATING). Prefix-tolerant on the marker's own first
    word (see _contains_multiword_phrase_prefix_tolerant) so the normal
    attached preposition in "...لحضرتك بمركز التميز المتخصص..." still
    matches the bare marker "مركز التميز"."""
    tokens = _tokens(text)
    return any(_contains_multiword_phrase_prefix_tolerant(tokens, m) for m in _EXPLICIT_COE_PROGRAM_MARKERS)


def _turn_has_category_specific_evidence(text: str, coe: str) -> bool:
    """True when *text* independently names a specialty or approved-
    complaint category belonging to *coe* — the second half of the SCRIPT
    SIMILARITY GATING prerequisite. An ordinary phrase like "عيادة الجهاز
    الهضمي" alone (with no explicit COE/program marker present) never
    reaches this check at all; this only disambiguates WHICH COE a
    genuinely COE/program-flavoured turn is actually paraphrasing."""
    for _canonical, candidate_coes in detect_specialty_mentions(text):
        if coe in candidate_coes:
            return True
    for matched_coe, _category, _alias in detect_approved_complaints(text):
        if matched_coe == coe:
            return True
    return False


def _explicit_name_matches(text: str) -> list[str]:
    """Every COE this Agent turn explicitly names via COE_NAME_MARKERS, in
    COE_KEYS order."""
    tokens = _tokens(text)
    return [
        coe for coe in COE_KEYS
        if any(_contains_phrase_arabic_prefix_tolerant(tokens, m) for m in COE_NAME_MARKERS[coe])
    ]


def _agent_turn_grounded_coe_recommendations(
    text: str, scripts: dict[str, str],
) -> list[tuple[str, str]]:
    """Every (coe, grounding_method) this ONE Agent turn safely, explicitly
    recommends — grounding_method is one of "explicit_name" (a direct
    COE_NAME_MARKERS match) or "explicit_coe_gated_script_paraphrase" (a
    close approved-script paraphrase, considered ONLY when this exact turn
    also independently satisfies BOTH SCRIPT SIMILARITY GATING
    prerequisites — see _turn_has_explicit_coe_program_marker /
    _turn_has_category_specific_evidence). Never "ungated_fuzzy_
    similarity", "reference_only", "llm_only", "specialty_only", or
    "generic_agent_text" — those grounding methods are explicitly
    disallowed and must never set explicit_agent_recommended=True.

    An explicit-name match always wins outright for this turn (no fuzzy
    paraphrase is layered on top of it) — mirrors the pre-existing
    "explicit name short-circuits" precedent."""
    explicit = _explicit_name_matches(text)
    if explicit:
        return [(coe, "explicit_name") for coe in explicit]
    if not _turn_has_explicit_coe_program_marker(text):
        return []
    best_key: str | None = None
    best_score = 0.0
    for coe, script in scripts.items():
        if not _turn_has_category_specific_evidence(text, coe):
            continue
        score = script_similarity(text, script)
        if score > best_score:
            best_key, best_score = coe, score
    if best_key and best_score >= SCRIPT_MATCH_THRESHOLD:
        return [(best_key, "explicit_coe_gated_script_paraphrase")]
    return []


def resolve_recommended_coe(call: CallTranscript, scripts: dict[str, str] | None = None) -> str | None:
    """Identify which supported COE the AGENT actually, safely recommended
    or confirmed — Patient turns are never consulted here (see module
    docstring's speaker-attribution rule). Checks every Agent turn (not
    just the first) for an explicit COE-name marker; falls back to a
    GATED approved-script paraphrase (see SCRIPT SIMILARITY GATING) only
    when no explicit marker is found anywhere. The first Agent turn with
    ANY grounded match wins — ungated fuzzy similarity against generic,
    boilerplate-only wording can never set this value (see the reported
    false-IBD-match regression, where shared script boilerplate alone
    used to score >=90 against every COE at once).
    """
    scripts = scripts or DEFAULT_SCRIPTS_AR
    for speaker, text in split_transcript_turns(call.transcript):
        if speaker != "agent":
            continue
        matches = _agent_turn_grounded_coe_recommendations(text, scripts)
        if matches:
            return matches[0][0]
    return None


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
    # A second title word embedded MID-CANDIDATE (e.g. "اسمة عبدالسلام
    # ودكتور محمود" from "متاح دكتور اسمة عبدالسلام ودكتور محمود ...")
    # means the shared extraction engine's name-capture ran on past the
    # first doctor's name into a SECOND "دكتور"-anchored offer — a title
    # word is never legitimately part of a person's own name, so this is
    # always an over-capture boundary, never a real name token.
    "دكتور", "دكتوره", "ودكتور", "ودكتوره", "والدكتور", "والدكتوره",
    "الدكتور", "الدكتوره",
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
    tokens = _tokens(text)
    matched: list[str] = [
        key for key in COE_KEYS if any(_contains_phrase_arabic_prefix_tolerant(tokens, m) for m in COE_NAME_MARKERS[key])
    ]
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


def specialty_evidence_is_grounded(canonical_specialty: str, evidence: str | None, transcript: str) -> bool:
    """Defense-in-depth safeguard: a specialty is only ever retained on a
    context when its *evidence* excerpt (a) is an actual verbatim
    substring of *transcript* and (b) the deterministic specialty
    detectors, run over that SAME evidence, independently confirm
    *canonical_specialty* is genuinely among the specialties that excerpt
    supports — never accepted merely because the taxonomy
    (COE_SPECIALTIES/SPECIALTY_ALIASES), CRM reference data, or a prompt's
    reference block happens to mention it.

    Checks membership rather than "sole winner" deliberately — a single
    excerpt can legitimately support more than one specialty for the SAME
    COE (e.g. both GIT and Nutrition mentioned in one turn), so this must
    not reject a genuine second specialty merely because
    resolve_canonical_specialty's own longest-alias-wins tie-break would
    have picked a different one first.

    build_coe_contexts's own specialty detection (detect_specialty_mentions
    /detect_weak_specialty_mentions) already only ever fires on real
    transcript text, so this check is always true for evidence it
    produces — it exists so this remains true by CONSTRUCTION (never by
    convention) even if a future caller starts assembling specialty
    entries from another source (e.g. an LLM-touched path).
    """
    if not evidence or not canonical_specialty:
        return False
    if _norm(evidence) not in _norm(transcript):
        return False
    strong = {specialty for specialty, _coes in detect_specialty_mentions(evidence)}
    weak = {specialty for specialty, _coes in detect_weak_specialty_mentions(evidence)}
    return canonical_specialty in strong or canonical_specialty in weak


# ── Actionable-service evidence (confirmed business rule) ───────────────────
# A COE can be confirmed through a substantive agent response involving one
# of its mapped specialties — the agent never has to repeat "COE"/"Center
# of Excellence"/"مركز التميز" (see build_coe_evaluations' coe_match_status
# and explicit_coe_recommendation_status, kept deliberately separate).
# "Substantive" is deliberately narrower than merely NAMING the specialty
# (a bare/incidental mention like "we also have a GIT department" must
# never count) — it requires the SAME turn to also show a concrete booking
# action (offering availability, confirming/creating an appointment,
# routing the patient) or to name an actual doctor.
_ACTIONABLE_SERVICE_ACTION_MARKERS: set[str] = {
    "تم تأكيد", "تأكيد الحجز", "تأكيد الموعد", "تم الحجز", "سيتم حجز",
    "حجز موعد", "هحجزلك", "هوصلك", "هحولك", "هحجز", "متواجد", "متاح",
    "confirmed", "available", "booked", "booking",
}


def _turn_suggests_actionable_service(text: str) -> bool:
    """True when *text* (a single Agent turn already known to mention a
    mapped specialty) shows a concrete, substantive action — a booking/
    routing/confirmation marker, or an actual doctor name — rather than
    just naming the specialty in passing."""
    norm = _norm(text)
    if any(_norm(m) in norm for m in _ACTIONABLE_SERVICE_ACTION_MARKERS):
        return True
    for raw in _doctor_name_candidates_in_text(text):
        if is_plausible_coe_doctor_candidate(clean_extracted_doctor_name(raw)):
            return True
    return bool(_known_doctor_alias_candidates(text))


def build_coe_contexts(
    call: CallTranscript,
    scripts: dict[str, str] | None = None,
    *,
    return_rejected: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
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
    # Which Patient turns are a completed-diagnostic-results inquiry (see
    # _completed_results_turn_flags) — a test NAME mentioned only while
    # retrieving an already-completed result (e.g. bare "سكر" inside "اخر
    # تحليل سكر تراكمي") must never independently ground a
    # patient_complaint/patient_specialty source.
    results_flags = _completed_results_turn_flags(turns)
    contexts: dict[str, dict[str, Any]] = {}
    # Rejected fuzzy-similarity near-misses — never create a context, kept
    # only so the caller (build_coe_evaluations/nodes.py logging) can
    # report WHY a decent-looking script match was discarded (see SCRIPT
    # SIMILARITY GATING). Populated only when return_rejected=True.
    rejected_recommendations: list[dict[str, Any]] = []

    def _ctx(coe: str) -> dict[str, Any]:
        return contexts.setdefault(coe, {
            "coe": coe,
            "context_sources": [],
            "complaints": [],
            "complaint_details": [],
            "specialties": [],
            "campaign_evidence": None,
            "agent_coe_evidence": None,
            # One entry per grounded explicit agent recommendation for this
            # COE — {"coe", "agent_evidence", "turn_index",
            # "explicit_coe_indicator", "category_specific_evidence",
            # "grounding_method"} (see GROUND EXPLICIT AGENT
            # RECOMMENDATIONS). Never populated by an ungated fuzzy match.
            "recommendation_grounding": [],
            # Index (in split_transcript_turns's order) of the FIRST turn
            # that grounded this COE at all — every context-creating branch
            # below calls _add_source before any other _ctx() access, so
            # this is always set on a context's very first touch. Used to
            # order build_coe_evaluations' output chronologically (by
            # first grounded evidence) rather than by COE_KEYS' fixed
            # declaration order.
            "first_turn_index": None,
        })

    def _add_source(coe: str, source: str, turn_idx: int) -> None:
        c = _ctx(coe)
        if c["first_turn_index"] is None:
            c["first_turn_index"] = turn_idx
        if source not in c["context_sources"]:
            c["context_sources"].append(source)

    running_active: list[str] = []

    def _add_specialty(
        coe: str, canonical_specialty: str, speaker: str, excerpt: str, candidate_coes: list[str],
    ) -> None:
        # Defense-in-depth grounding check (see specialty_evidence_is_
        # grounded's docstring) — always true for evidence this function's
        # own callers produce (they already scanned real transcript text),
        # but guarantees a hallucinated/ungrounded specialty can never
        # silently slip onto a context even if a future change starts
        # feeding this from a less trustworthy source.
        if not specialty_evidence_is_grounded(canonical_specialty, excerpt, call.transcript):
            return
        _match = _matched_alias_for_specialty(excerpt, canonical_specialty)
        matched_alias, original_text = _match if _match else (canonical_specialty, canonical_specialty)
        c = _ctx(coe)
        entry = {
            "canonical_specialty": canonical_specialty,
            "original_text": original_text,
            "matched_alias": matched_alias,
            "speaker": speaker,
            "evidence": excerpt,
            "verbatim_evidence": excerpt,
            "candidate_coes": list(candidate_coes),
            "resolved_coe": coe,
        }
        if entry not in c["specialties"]:
            c["specialties"].append(entry)

    for turn_idx, (speaker, text) in enumerate(turns):
        norm = _norm(text)
        excerpt = text.strip()[:300]
        this_turn_coes: list[str] = []

        if speaker == "patient" and _CAMPAIGN_ORIGIN_RE.search(text):
            for coe in _unambiguous_coe_matches(text):
                _add_source(coe, "campaign", turn_idx)
                this_turn_coes.append(coe)
                c = _ctx(coe)
                if c["campaign_evidence"] is None:
                    c["campaign_evidence"] = excerpt

        if speaker == "agent":
            # Every grounded (never ungated-fuzzy) explicit recommendation
            # this turn makes — see SCRIPT SIMILARITY GATING /
            # _agent_turn_grounded_coe_recommendations. Fuzzy script
            # similarity can never, by itself, create a NEW COE category
            # from generic shared boilerplate; it may only confirm a
            # paraphrase of a category the SAME turn already, independently
            # evidences via an explicit COE/program marker plus category-
            # specific content.
            grounded = _agent_turn_grounded_coe_recommendations(text, scripts)
            for coe, method in grounded:
                _add_source(coe, "agent_recommendation", turn_idx)
                this_turn_coes.append(coe)
                c = _ctx(coe)
                if c["agent_coe_evidence"] is None:
                    c["agent_coe_evidence"] = excerpt
                c["recommendation_grounding"].append({
                    "coe": coe,
                    "agent_evidence": excerpt,
                    "turn_index": turn_idx,
                    "explicit_coe_indicator": True,
                    "category_specific_evidence": (
                        method == "explicit_name" or _turn_has_category_specific_evidence(text, coe)
                    ),
                    "grounding_method": method,
                })
            if return_rejected and not grounded:
                # Transparency only — a decent-looking raw fuzzy score that
                # never satisfied the gating prerequisites (see
                # SCRIPT SIMILARITY GATING) is logged as a rejected
                # recommendation, never turned into a context.
                raw_best_key, raw_best_score = None, 0.0
                for coe, script in scripts.items():
                    score = script_similarity(text, script)
                    if score > raw_best_score:
                        raw_best_key, raw_best_score = coe, score
                if raw_best_key and raw_best_score >= SCRIPT_MATCH_THRESHOLD:
                    rejected_recommendations.append({
                        "turn_index": turn_idx,
                        "candidate": raw_best_key,
                        "score": raw_best_score,
                        "method": "fuzzy_similarity",
                        "reason": "no_explicit_coe_or_category_evidence",
                        "evidence": excerpt,
                    })

        if speaker == "patient":
            # Clause-scoped (never whole-turn) so an approved complaint
            # category appearing in a clause naming a THIRD PARTY ("امي
            # عندها سكري"), under negation, describing a PAST/RESOLVED
            # condition, or posing a HYPOTHETICAL is never credited as the
            # patient's OWN CURRENT active complaint — see ACTIVE PATIENT
            # NEED. A genuinely separate, patient-own complaint stated
            # elsewhere in the SAME turn (a different clause) is still
            # credited. Uses ONLY the restricted, business-approved
            # COE_COMPLAINTS registry (the "patient_approved_complaint"
            # eligibility route) — deliberately narrower than the legacy
            # COMPLAINT_KEYWORDS vocabulary still used elsewhere in this
            # module (resolve_primary_complaint / campaign disambiguation
            # / LLM grounding), which serves different, pre-existing
            # purposes this recommendation-eligibility route must not
            # inherit (a vague symptom like dizziness or nausea must never
            # by itself create a recommendation obligation).
            for clause in ([] if results_flags[turn_idx] else (_split_clauses(text) or [text])):
                if _clause_is_third_party_or_negated(clause):
                    continue
                for coe, category, matched_alias in detect_approved_complaints(clause):
                    _add_source(coe, "patient_complaint", turn_idx)
                    this_turn_coes.append(coe)
                    c = _ctx(coe)
                    if excerpt[:200] not in c["complaints"]:
                        c["complaints"].append(excerpt[:200])
                    detail = {
                        "category": category,
                        "matched_alias": matched_alias,
                        "evidence": excerpt[:200],
                    }
                    if detail not in c["complaint_details"]:
                        c["complaint_details"].append(detail)

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
        # A specialty named only in a clause about a THIRD PARTY or under
        # negation is never the PATIENT's own eligibility evidence (same
        # rule as the patient_complaint loop above) — scan a version of
        # the turn with those clauses removed, never the raw excerpt
        # actually stored as evidence.
        if speaker == "patient":
            # A whole turn classified as a completed-diagnostic-results
            # inquiry never contributes specialty evidence either — a
            # test NAME (e.g. bare "سكر" in "اخر تحليل سكر تراكمي") is
            # an investigation, never a specialty request on its own (see
            # "TEST NAMES ARE NOT DIAGNOSES OR SPECIALTIES").
            qualifying_clauses = [] if results_flags[turn_idx] else [
                cl for cl in (_split_clauses(text) or [text])
                if not _clause_is_third_party_or_negated(cl) and not _clause_is_bare_new_test_request(cl)
            ]
            specialty_scan_text = "، ".join(qualifying_clauses)
        else:
            specialty_scan_text = text
        for specialty, candidate_coes in detect_specialty_mentions(specialty_scan_text):
            resolved_coes = (
                candidate_coes if len(candidate_coes) == 1
                else [c for c in candidate_coes if c in active_for_specialty]
            )
            if len(resolved_coes) != 1:
                continue  # ambiguous/unresolved — never guessed into a context
            coe = resolved_coes[0]
            _add_source(coe, "agent_specialty" if speaker == "agent" else "patient_specialty", turn_idx)
            this_turn_coes.append(coe)
            _add_specialty(coe, specialty, speaker, excerpt[:200], candidate_coes)
            if speaker == "agent" and _turn_suggests_actionable_service(text):
                _add_source(coe, "agent_actionable_service", turn_idx)

        for specialty, candidate_coes in detect_weak_specialty_mentions(specialty_scan_text):
            for coe in candidate_coes:
                if coe not in active_for_specialty:
                    continue  # weak evidence alone never creates/extends a context
                _add_source(coe, "agent_specialty" if speaker == "agent" else "patient_specialty", turn_idx)
                _add_specialty(coe, specialty, speaker, excerpt[:200], candidate_coes)
                if speaker == "agent" and _turn_suggests_actionable_service(text):
                    _add_source(coe, "agent_actionable_service", turn_idx)

        for coe in this_turn_coes:
            if coe not in running_active:
                running_active.append(coe)

    if return_rejected:
        return contexts, rejected_recommendations
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

    Each kept entry also tracks "_last_turn_index" — the LATEST turn_index
    across every mention merged into it — used by evaluate_context_doctors
    to tell an OFFERED-but-not-revisited doctor apart from the one a later
    turn actually confirmed/selected (see its docstring).
    """
    kept: list[dict[str, Any]] = []
    for entry in entries:
        norm_name = normalize_doctor_name_for_match(entry.get("extracted_name"))
        entry_turn_index = entry.get("turn_index", 0) or 0
        merged = False
        if norm_name:
            for existing in kept:
                existing_norm = normalize_doctor_name_for_match(existing.get("extracted_name"))
                if not existing_norm:
                    continue
                same_person = (
                    norm_name == existing_norm
                    or norm_name in existing_norm
                    or existing_norm in norm_name
                )
                if not same_person and len(norm_name) >= _MIN_MATCHABLE_NAME_LENGTH:
                    # Conservative same-script fuzzy fallback (reuses the
                    # exact thresholds resolve_primary_doctor_identity uses
                    # against the approved-alias table) — catches minor
                    # transcription/ASR spelling variants of an UNAPPROVED
                    # doctor's name too (e.g. "الحوارني" vs "الحوراني", a
                    # letter transposition), which the approved-alias
                    # table naturally never covers since this person isn't
                    # on it. Never applied across writing systems.
                    compact, existing_compact = norm_name.replace(" ", ""), existing_norm.replace(" ", "")
                    shorter, longer = sorted((len(compact), len(existing_compact)))
                    if (
                        longer > 0
                        and shorter / longer >= _FUZZY_LENGTH_RATIO_THRESHOLD
                        and _is_arabic_text(norm_name) == _is_arabic_text(existing_norm)
                        and _rfuzz.token_sort_ratio(norm_name, existing_norm) >= _FUZZY_SCORE_THRESHOLD
                    ):
                        same_person = True
                if same_person:
                    if len(entry.get("extracted_name") or "") > len(existing.get("extracted_name") or ""):
                        existing["extracted_name"] = entry["extracted_name"]
                        existing["association_evidence"] = entry.get("association_evidence")
                    existing["_last_turn_index"] = max(existing.get("_last_turn_index", 0), entry_turn_index)
                    merged = True
                    break
        if not merged:
            new_entry = dict(entry)
            new_entry["_last_turn_index"] = entry_turn_index
            kept.append(new_entry)
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

    Offered alternatives vs. the selected/confirmed doctor: when a call
    offers several DISTINCT initial_primary doctors (e.g. "متاح دكتور X
    ودكتور Y") and a STRICTLY LATER turn goes on to name only ONE of them
    again (e.g. the patient picks one and the agent confirms the booking),
    that later, re-confirmed doctor is the one actually booked — the
    earlier, never-revisited alternative(s) are downgraded to
    "initial_supporting" so they are reported (never silently dropped)
    but no longer affect the pass/fail aggregate. This is deliberately
    NOT applied when every distinct doctor's LATEST mention shares the
    same turn (e.g. "دكتور X أو دكتور Y" with no follow-up at all) — with
    no turn ever narrowing it down, both remain genuinely open initial
    options and BOTH still count (an approved alternative must never let
    an unapproved one pass — see the module docstring's core requirement).
    """
    deduped = _dedupe_doctor_entries(doctor_entries)
    initial_entries = [e for e in deduped if e.get("role", "initial_primary") == "initial_primary"]
    if len(initial_entries) > 1:
        latest_turn = max(e.get("_last_turn_index", 0) for e in initial_entries)
        if any(e.get("_last_turn_index", 0) < latest_turn for e in initial_entries):
            for e in initial_entries:
                if e.get("_last_turn_index", 0) < latest_turn:
                    e["role"] = "initial_supporting"

    per_doctor: list[dict[str, Any]] = []
    for entry in deduped:
        name = entry.get("extracted_name")
        role = entry.get("role", "initial_primary")
        base = {
            "extracted_name": name,
            "role": role,
            "association_evidence": entry.get("association_evidence"),
        }
        if role == "initial_supporting":
            per_doctor.append({
                **base,
                "canonical_name": None,
                "primary_doctor_status": "not_applicable",
                "reason": (
                    f"{name} was offered as an initial alternative but a later turn confirmed a "
                    "different doctor for this booking — not counted in the primary-doctor decision."
                ),
            })
            continue
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


# Human-readable label for each canonical specialty, used only to phrase
# the deterministic patient_need/QA-note text below — never used for any
# matching/grounding decision (those stay driven entirely by
# COE_SPECIALTIES/SPECIALTY_ALIASES/resolve_canonical_specialty).
_SPECIALTY_NEED_LABEL: dict[str, str] = {
    "GIT": "a gastroenterology appointment",
    "Nutrition": "a nutrition appointment",
    "General Surgery": "a general surgery appointment",
    "Neurology": "a neurology appointment",
    "Ophthalmology": "an ophthalmology appointment",
    "ENT": "an ENT appointment",
    "Cardiology": "a cardiology appointment",
    "Psychiatry": "a psychiatry appointment",
    "Dental": "a dental appointment",
    "Diabetes": "a diabetes appointment",
    "Diabetic Educator": "a diabetes education appointment",
    "Orthopedics": "an orthopedics appointment",
    "Pulmonology": "a pulmonology appointment",
    "Allergy & Immunology": "an allergy/immunology appointment",
}


def _describe_patient_need(coe_key: str, ctx: dict[str, Any]) -> tuple[str, str | None]:
    """Deterministic (coe_key, ctx) -> (patient_need, patient_need_evidence)
    — a short, grounded, human-readable description of what the PATIENT
    actually asked for, plus the verbatim excerpt supporting it. Prefers
    an actual patient complaint excerpt, then a patient-authored specialty
    mention, then the campaign entry point — never an agent-side excerpt,
    since patient_need describes the PATIENT's own need (see ELIGIBILITY
    DETECTION)."""
    if ctx["complaints"]:
        return (
            f"The patient described a complaint/request connected to the {coe_key} COE.",
            ctx["complaints"][0],
        )
    patient_specialty_entries = [s for s in ctx["specialties"] if s.get("speaker") == "patient"]
    if patient_specialty_entries:
        specialty = patient_specialty_entries[0]["canonical_specialty"]
        label = _SPECIALTY_NEED_LABEL.get(specialty, f"a {specialty} appointment")
        return (f"The patient requested {label}.", patient_specialty_entries[0]["evidence"])
    if ctx["campaign_evidence"]:
        return (
            f"The patient entered this conversation via a {coe_key} COE campaign/post.",
            ctx["campaign_evidence"],
        )
    return (f"The patient's stated need connects to the {coe_key} COE.", None)


_COMPLAINT_CATEGORY_LABEL: dict[str, str] = {
    "migraine": "migraine headache", "tension_headache": "tension headache",
    "chronic_headache": "chronic headache", "sinus_headache": "sinus headache",
    "digestive_disease": "digestive disease", "colon_disease": "colon disease",
    "liver_disease": "liver disease", "pancreatic_disease": "pancreatic disease",
    "asthma": "asthma", "bronchial_asthma": "bronchial asthma",
    "chest_allergy": "chest allergy", "shortness_of_breath": "shortness of breath",
    "diabetes": "diabetes", "endocrine_disease": "endocrine disease", "diabetic_foot": "diabetic foot",
}


def _build_missed_recommendation_note(
    coe_key: str,
    patient_need: str,
    patient_need_evidence: str | None,
    mapped_specialties: list[dict[str, Any]],
    agent_response_evidence: str | None,
    wrong_coe: bool,
    complaint_category: str | None = None,
) -> str:
    """Deterministic, grounded QA note for a missed/wrong_coe finding —
    names the patient need, the matched specialty OR approved complaint
    category (whichever actually grounded eligibility — see OUTPUT AND
    EVIDENCE), the expected COE, why the agent's response did not qualify
    as a COE recommendation, and the supporting evidence (see QA NOTE
    requirements)."""
    if complaint_category:
        mapped_via = (
            f"{_COMPLAINT_CATEGORY_LABEL.get(complaint_category, complaint_category)}, which is an "
            f"approved {coe_key} COE complaint category"
        )
    else:
        specialty_names = ", ".join(dict.fromkeys(s["canonical_specialty"] for s in mapped_specialties))
        mapped_via = specialty_names or "a mapped specialty"
    evidence_bits = [f'patient need: "{patient_need_evidence}"' if patient_need_evidence else None]
    if wrong_coe:
        reason = (
            f"the agent explicitly recommended a DIFFERENT Center of Excellence that has no "
            f"patient-side basis of its own in this call"
        )
    elif agent_response_evidence:
        evidence_bits.append(f'agent response: "{agent_response_evidence}"')
        reason = (
            "the agent's response only handled the specialty/service directly (a doctor offer, "
            "availability, or an ordinary appointment) without explaining or recommending the "
            f"{coe_key} Center of Excellence itself"
        )
    else:
        reason = f"no agent turn recommended or explained the {coe_key} Center of Excellence"
    evidence_text = "; ".join(b for b in evidence_bits if b)
    return (
        f"The {coe_key} Center of Excellence should have been recommended. {patient_need} "
        f"This maps to the {coe_key} COE via {mapped_via}, but {reason}."
        + (f" Evidence — {evidence_text}." if evidence_text else "")
    )


def build_coe_evaluations(
    call: CallTranscript,
    scripts: dict[str, str] | None = None,
    *,
    return_rejected: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
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

    When return_rejected=True, returns (evaluations, rejected) instead,
    where rejected == {"recommendations": [...], "contexts": [...]} — see
    CAMPAIGN CONTEXT ADMISSION / AGENT RECOMMENDATION IS NOT AUTOMATICALLY
    A PATIENT CONTEXT. Never affects the default (plain-list) return shape
    every pre-existing caller relies on.
    """
    contexts, rejected_recommendations = build_coe_contexts(call, scripts, return_rejected=True)
    associations = extract_doctor_context_associations(call)
    # A structured campaign identifier proves only where the conversation
    # ORIGINATED — never that the patient's substantive inquiry is
    # actually about that campaign's COE (see CAMPAIGN DETECTION IS NOT
    # CAMPAIGN ENGAGEMENT). Only an "engaged" campaign may ever set
    # patient_eligible for its candidate COE.
    campaign_info = classify_campaign_relevance(call)
    active_campaign_coe = campaign_info["active_campaign_coe"]

    # Chronological order — by each context's OWN first grounded transcript
    # evidence (see build_coe_contexts' first_turn_index), never COE_KEYS'
    # fixed declaration order. A call where a Headache campaign/booking is
    # discussed first and an unrelated IBD request comes up later (e.g.
    # after "وبعده") must report Headache before IBD, matching the order
    # events actually happened in the call.
    ordered_coe_keys = sorted(
        (key for key in COE_KEYS if key in contexts),
        key=lambda key: (
            contexts[key]["first_turn_index"]
            if contexts[key]["first_turn_index"] is not None
            else float("inf")
        ),
    )

    # ── Pass 1: per-context facts that do NOT depend on any OTHER context
    # (doctors, service alignment, explicit-recommendation attribution,
    # patient-side eligibility). ─────────────────────────────────────────
    _pending: list[dict[str, Any]] = []
    for coe_key in ordered_coe_keys:
        ctx = contexts[coe_key]
        doctors_here = [a for a in associations if coe_key in a["associated_coes"]]
        per_doctor, primary_status = evaluate_context_doctors(coe_key, doctors_here)

        campaign_coe = "campaign" in ctx["context_sources"]
        # "agent_recommendation" is only ever added from an explicit
        # COE_NAME_MARKERS match or a close full-script paraphrase (see
        # build_coe_contexts) — never from a bare specialty/complaint
        # mention (those are tagged "agent_specialty"/"patient_specialty"/
        # "patient_complaint" instead), so this never mistakes a doctor's
        # specialty/title description for the agent announcing a COE.
        explicit_agent_recommended = "agent_recommendation" in ctx["context_sources"]
        explicit_coe_recommendation_status = "pass" if explicit_agent_recommended else "not_applicable"

        # "agent_actionable_service" is only ever added when an ACTUAL
        # Agent turn both uses a mapped-specialty alias word AND shows a
        # concrete booking/routing/confirmation action or names an actual
        # doctor (see build_coe_contexts/_turn_suggests_actionable_
        # service) — i.e. the agent substantively offered, selected, or
        # booked something connected to this COE's specialty. This is
        # SERVICE alignment only — per the confirmed business rule
        # superseding the earlier one, it is NOT by itself a COE
        # recommendation (see recommendation_status below, computed
        # separately in pass 2).
        has_actionable_mapped_service = (
            explicit_agent_recommended or "agent_actionable_service" in ctx["context_sources"]
        )
        # Patient-originated eligibility evidence — an approved diagnosis/
        # active complaint (see COE_RECOMMENDATION_TRIGGERS) or an ENGAGED
        # campaign for THIS SPECIFIC coe (the campaign establishes the
        # patient's COE context ONLY when the rest of the call actually
        # continues that topic — see CAMPAIGN DETECTION IS NOT CAMPAIGN
        # ENGAGEMENT / classify_campaign_relevance; a diverted, pending, or
        # uncertain campaign can never satisfy this, even though the
        # campaign message itself is still recorded in context_sources for
        # metadata/reporting purposes). Never derived from an agent-only
        # mention, and — per the confirmed business rule superseding the
        # earlier one — never derived from a bare "patient_specialty"
        # mention either: a patient requesting Dental, Ophthalmology, ENT,
        # Cardiology, Neurology, etc. on its own is NOT, by itself,
        # evidence that a COE recommendation was owed. A specialty mention
        # may still be reported for supporting/referral context (see
        # canonical_specialty/specialty_evidence below), but it never sets
        # patient_eligible.
        patient_eligible = (
            "patient_complaint" in ctx["context_sources"]
            or (campaign_coe and coe_key == active_campaign_coe)
        )
        if has_actionable_mapped_service:
            service_alignment_status = "pass"
        elif patient_eligible:
            # The patient raised/requested this COE's territory, but no
            # actual Agent turn ever substantively engaged with it (no
            # offer, booking, or mapped-specialty response) — genuinely
            # unresolved, never guessed into a "pass".
            service_alignment_status = "uncertain"
        else:
            # A context cannot normally exist with NEITHER agent nor
            # patient evidence (build_coe_contexts requires at least one
            # source to create it), but this is a safe, conservative
            # fallback should that ever change.
            service_alignment_status = "not_applicable"

        patient_need, patient_need_evidence = _describe_patient_need(coe_key, ctx)

        # Eligibility source/evidence — see OUTPUT AND EVIDENCE: which of
        # the two independent eligibility routes established this context.
        # Priority when more than one applies (e.g. a campaign-established
        # context the patient ALSO later describes a complaint for):
        # a direct specialty/service request is the most specific signal,
        # then an approved complaint category, then the campaign entry
        # point on its own. Never a diagnosis mislabeled as a specialty
        # request, or vice versa — the two are reported from entirely
        # separate underlying evidence (ctx["complaint_details"] vs.
        # ctx["specialties"]).
        patient_specialty_entry = next(
            (s for s in ctx["specialties"] if s.get("speaker") == "patient"), None
        )
        complaint_detail = ctx["complaint_details"][0] if ctx["complaint_details"] else None
        # Only two routes can ever establish eligibility now — an approved
        # complaint/diagnosis category (COE_RECOMMENDATION_TRIGGERS) or the
        # campaign entry point itself. A bare "patient_specialty" mention
        # (Dental, Ophthalmology, ENT, Cardiology, Neurology alone, etc.)
        # is REMOVED as an eligibility route per the confirmed business
        # rule — it is reported below only as supporting/referral
        # reference data (canonical_specialty/specialty_evidence), never
        # as the reason a recommendation was required.
        has_patient_complaint = "patient_complaint" in ctx["context_sources"] and complaint_detail
        if has_patient_complaint:
            eligibility_source = "patient_approved_complaint"
        elif campaign_coe:
            eligibility_source = "campaign_origin"
        else:
            eligibility_source = None
        canonical_specialty = (
            patient_specialty_entry["canonical_specialty"] if patient_specialty_entry
            else (ctx["specialties"][0]["canonical_specialty"] if ctx["specialties"] else None)
        )
        specialty_evidence = (
            patient_specialty_entry["verbatim_evidence"] if patient_specialty_entry
            else (ctx["specialties"][0]["verbatim_evidence"] if ctx["specialties"] else None)
        )

        _pending.append({
            "coe": coe_key,
            "resolved_coe": coe_key,
            "context_sources": ctx["context_sources"],
            "complaints": ctx["complaints"],
            "complaint_details": ctx["complaint_details"],
            "specialties": ctx["specialties"],
            "mapped_specialties": ctx["specialties"],
            "campaign_evidence": ctx["campaign_evidence"],
            "agent_coe_evidence": ctx["agent_coe_evidence"],
            "patient_need": patient_need,
            "patient_need_evidence": patient_need_evidence,
            "eligibility_source": eligibility_source,
            "complaint_category": complaint_detail["category"] if complaint_detail else None,
            "complaint_evidence": complaint_detail["evidence"] if complaint_detail else None,
            "canonical_specialty": canonical_specialty,
            "specialty_evidence": specialty_evidence,
            # Every eligibility decision in this module is 100%
            # deterministic (no LLM ever touches specialty/complaint
            # identification) — always True/1.0 here; kept as explicit
            # fields per the OUTPUT AND EVIDENCE schema rather than a
            # hidden assumption.
            "eligibility_deterministic": True,
            "eligibility_confidence": 1.0 if eligibility_source else 0.0,
            "doctors": per_doctor,
            "campaign_coe": campaign_coe,
            "explicit_agent_recommended": explicit_agent_recommended,
            "explicit_coe_recommendation_status": explicit_coe_recommendation_status,
            "service_alignment_status": service_alignment_status,
            "validated_coe": coe_key,
            "primary_doctor_status": primary_status,
            "_patient_eligible": patient_eligible,
            "recommendation_grounding": ctx["recommendation_grounding"],
        })

    # ── Pass 2: recommendation_status — the NEW authoritative "was the
    # relevant COE actually recommended by the human agent" check (see the
    # module's missed-COE-recommendation business rule, which supersedes
    # the earlier "an actionable mapped service alone is a recommendation"
    # rule). Needs visibility into every OTHER context to distinguish:
    #   - "missed": the patient's own, independently-eligible need (e.g.
    #     a Headache package AND a separate later GIT request) simply
    #     never got its OWN COE recommended — the OTHER explicit
    #     recommendation in the call (if any) is itself justified by ITS
    #     OWN patient-side evidence, so this is a second, independent,
    #     unaddressed need, not a substitution.
    #   - "wrong_coe": the sole patient need went unaddressed while the
    #     agent explicitly recommended a DIFFERENT COE that has NO
    #     patient-side eligibility of its own anywhere in the call (e.g.
    #     patient only ever mentions Diabetes, agent proactively
    #     recommends Headache instead) — an unjustified substitution,
    #     not a second legitimate topic. ───────────────────────────────────
    evaluations: list[dict[str, Any]] = []
    for entry in _pending:
        primary_status = entry["primary_doctor_status"]
        if entry["explicit_agent_recommended"]:
            # An explicit recommendation is unconditionally a pass — even
            # a purely proactive one with no prior patient prompt is a
            # GOOD outcome, never something to obscure as not_applicable.
            recommendation_required: Any = True
            recommendation_status = "pass"
            human_agent_recommended_coe = True
            recommendation_evidence = entry["agent_coe_evidence"]
        elif not entry["_patient_eligible"]:
            # No patient-originated need was ever established for this
            # COE (e.g. a purely agent-proactive context) — the missed-
            # recommendation requirement, which exists to catch an
            # UNADDRESSED patient need, simply does not apply.
            recommendation_required = False
            recommendation_status = "not_applicable"
            human_agent_recommended_coe = False
            recommendation_evidence = None
        else:
            conflicting = [
                other for other in _pending
                if other is not entry and other["explicit_agent_recommended"] and not other["_patient_eligible"]
            ]
            # "wrong_coe" (an unjustified substitution) only when THIS
            # context's own need received NO agent engagement of any kind
            # (service_alignment_status != "pass") — if the agent DID
            # substantively handle this need too (e.g. an ordinary
            # specialty booking), the correct finding is "missed" (the
            # COE recommendation itself was skipped), regardless of some
            # OTHER, unrelated explicit-but-unjustified recommendation
            # elsewhere in the same call.
            recommendation_required = True
            recommendation_status = (
                "wrong_coe" if conflicting and entry["service_alignment_status"] != "pass" else "missed"
            )
            human_agent_recommended_coe = False
            recommendation_evidence = None

        missed_recommendation = recommendation_status in ("missed", "wrong_coe")
        is_violation = primary_status == "fail" or missed_recommendation

        note = None
        if missed_recommendation:
            agent_response_evidence = entry["agent_coe_evidence"] or next(
                (s["evidence"] for s in entry["mapped_specialties"] if s.get("speaker") == "agent"),
                None,
            )
            note = _build_missed_recommendation_note(
                coe_key=entry["coe"],
                patient_need=entry["patient_need"],
                patient_need_evidence=entry["patient_need_evidence"],
                mapped_specialties=entry["mapped_specialties"],
                agent_response_evidence=agent_response_evidence,
                wrong_coe=recommendation_status == "wrong_coe",
                complaint_category=entry["complaint_category"] if entry["eligibility_source"] == "patient_approved_complaint" else None,
            )

        # coe_match_status is kept, mirroring recommendation_status
        # exactly, ONLY for backward compatibility with existing callers
        # that read this field name — recommendation_status is the new
        # authoritative field; this is never a second, independently-
        # computed value.
        coe_match_status = recommendation_status

        _applicable_statuses = [
            s for s in (recommendation_status, entry["service_alignment_status"], primary_status)
            if s and s != "not_applicable"
        ]
        context_status = (
            "fail" if is_violation
            else "uncertain" if any(s == "uncertain" for s in _applicable_statuses)
            else "pass"
        )

        evaluations.append({
            "coe": entry["coe"],
            "resolved_coe": entry["resolved_coe"],
            "context_sources": entry["context_sources"],
            "complaints": entry["complaints"],
            "specialties": entry["specialties"],
            "mapped_specialties": entry["mapped_specialties"],
            "campaign_evidence": entry["campaign_evidence"],
            "agent_coe_evidence": entry["agent_coe_evidence"],
            "patient_need": entry["patient_need"],
            "patient_need_evidence": entry["patient_need_evidence"],
            "eligibility_source": entry["eligibility_source"],
            "complaint_category": entry["complaint_category"],
            "complaint_evidence": entry["complaint_evidence"],
            "canonical_specialty": entry["canonical_specialty"],
            "specialty_evidence": entry["specialty_evidence"],
            "eligibility_deterministic": entry["eligibility_deterministic"],
            "eligibility_confidence": entry["eligibility_confidence"],
            "doctors": entry["doctors"],
            # offered_doctors: every distinct doctor named for this context
            # (any role), in first-mention order — the full set the agent
            # put in front of the patient. selected_initial_doctors: the
            # subset actually validated as the initial booking (role ==
            # "initial_primary" after evaluate_context_doctors' offered-
            # vs-selected narrowing) — an approved doctor merely OFFERED
            # but not selected/confirmed must never hide an unapproved
            # SELECTED one (see evaluate_context_doctors' docstring).
            "offered_doctors": [d["extracted_name"] for d in entry["doctors"]],
            "selected_initial_doctors": [
                d["extracted_name"] for d in entry["doctors"] if d["role"] == "initial_primary"
            ],
            "campaign_coe": entry["campaign_coe"],
            "explicit_agent_recommended": entry["explicit_agent_recommended"],
            "explicit_coe_recommendation_status": entry["explicit_coe_recommendation_status"],
            "human_agent_recommended_coe": human_agent_recommended_coe,
            "recommendation_required": recommendation_required,
            "recommendation_status": recommendation_status,
            "recommendation_evidence": recommendation_evidence,
            "missed_recommendation": missed_recommendation,
            "note": note,
            "service_alignment_status": entry["service_alignment_status"],
            "validated_coe": entry["validated_coe"],
            "coe_match_status": coe_match_status,
            "primary_doctor_status": primary_status,
            "context_status": context_status,
            "is_violation": is_violation,
            # Whether this context has its OWN independent patient-side
            # basis (an approved complaint, or an ENGAGED campaign for this
            # exact COE) — see PATIENT ELIGIBILITY / CAMPAIGN ATTRIBUTION
            # and MULTI-CONTEXT ADMISSION. False for a purely agent-side
            # (proactive) recommendation or a diverted/pending/uncertain
            # campaign candidate.
            "patient_eligible": entry["_patient_eligible"],
            # The first grounded explicit-recommendation record for this
            # COE (see GROUND EXPLICIT AGENT RECOMMENDATIONS) — coe,
            # agent_evidence, turn_index, explicit_coe_indicator,
            # category_specific_evidence, grounding_method. None when this
            # context was never explicitly recommended by the agent at all.
            "explicit_recommendation_grounding": (
                entry["recommendation_grounding"][0] if entry["recommendation_grounding"] else None
            ),
        })

    # ── Context admission — a context whose ONLY basis is unsupported
    # (never a real patient/agent journey of its own) must never survive
    # as a false "context", even though pass 1/2 above still computed its
    # facts (needed to justify ANOTHER context's wrong_coe verdict). See
    # CAMPAIGN CONTEXT ADMISSION and AGENT RECOMMENDATION IS NOT
    # AUTOMATICALLY A PATIENT CONTEXT. ─────────────────────────────────────
    rejected_contexts: list[dict[str, Any]] = []
    admitted_evaluations: list[dict[str, Any]] = []
    for pending_entry, evaluation in zip(_pending, evaluations):
        sources = set(pending_entry["context_sources"])
        is_pure_campaign_candidate = sources == {"campaign"}
        is_unsupported_recommendation = (
            pending_entry["explicit_agent_recommended"] and not pending_entry["_patient_eligible"]
        )
        # A context built from NOTHING but a campaign mention is only ever
        # admitted when that campaign is genuinely ENGAGED for this exact
        # COE (see CAMPAIGN CONTEXT ADMISSION) — a diverted/pending/
        # uncertain campaign candidate must not appear in coe_evaluations
        # at all, and must never generate a recommendation_required/
        # missed_recommendation finding.
        if is_pure_campaign_candidate and pending_entry["coe"] == campaign_info.get("campaign_candidate_coe") and not pending_entry["_patient_eligible"]:
            rejected_contexts.append({
                "coe": pending_entry["coe"],
                "reason": f"campaign_{campaign_info.get('campaign_relevance') or 'diverted'}",
            })
            continue
        # An agent recommendation with NO patient-side basis of its own is
        # recorded as an agent action and used (mirrors pass 2's own
        # "conflicting" check) to justify ANOTHER active context's
        # wrong_coe — but it never becomes a separate, falsely-passing
        # patient context in its own right. Dropped ONLY when it is
        # actually consumed as a wrong_coe substitution for some OTHER
        # eligible context whose OWN need received NO agent engagement at
        # all (service_alignment_status != "pass") — never merely because
        # some OTHER unrelated eligible context also happens to exist: an
        # agent who both handles the patient's own need AND additionally,
        # genuinely recommends/books a completely separate COE service
        # (see test_headache_and_ibd_both_discussed_one_doctor_each) has
        # created a real second journey, not an unsupported substitution.
        is_consumed_as_wrong_coe_substitution = any(
            other is not pending_entry
            and other["_patient_eligible"]
            and other["service_alignment_status"] != "pass"
            for other in _pending
        )
        if is_unsupported_recommendation and is_consumed_as_wrong_coe_substitution:
            rejected_contexts.append({"coe": pending_entry["coe"], "reason": "ungrounded_agent_recommendation"})
            continue
        admitted_evaluations.append(evaluation)

    if return_rejected:
        return admitted_evaluations, {
            "recommendations": rejected_recommendations,
            "contexts": rejected_contexts,
        }
    return admitted_evaluations


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
