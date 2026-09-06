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
    scan the transcript, turn by turn, for a COE / specialized-center
    mention
        -> was the FIRST such mention made (or immediately preceded) by the
           Agent -> proactive recommendation (Path A)
        -> was it first raised by the Patient and then answered by the
           Agent -> customer inquiry (Path B)
        -> otherwise (only the Patient ever raised it, or nobody did) ->
           NOT triggered
        -> only when triggered: classify the patient's primary complaint,
           map it to the expected COE, extract the COE the Agent actually
           recommended/confirmed, and validate the initial doctor against
           the authoritative primary-doctor list for that COE.

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
        "gastroenterology", "gastrointestinal", "ibd", "crohn", "colitis",
        "digestive", "bowel", "stomach ulcer",
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


def classify_coe_trigger(call: CallTranscript) -> dict[str, Any]:
    """Determine whether COE validation should run at all, and via which
    path — turn-order-aware and speaker-attributed, so a Patient statement
    is never misattributed as an Agent recommendation (see module
    docstring). Walks the transcript turn by turn (in original order):

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