"""
CRM Offers Data Layer — services/crm_offers.py

Fetches live promotional offers from Dynamics 365 (new_offer_equest table),
filtered to BU=AHJ, with a 30-minute in-memory TTL cache.

Public API:
    fetch_offers(force_refresh=False)       → list[dict]  all AHJ offers
    get_offers_for_specialty(specialty_en)  → list[dict]  filtered + ranked

Status constants match CRM values in new_offerstatusnewname:
    STATUS_ACTIVE   = "Active"
    STATUS_INACTIVE = "Inactive"
    STATUS_DRAFT    = "Draft"
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from rapidfuzz import fuzz as _rfuzz

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_ACTIVE   = "Active"
STATUS_INACTIVE = "Inactive"
STATUS_DRAFT    = "Draft"

# Cache TTL — offers change more often than doctor prices, so 30 min
_CACHE_TTL_SECONDS = 30 * 60

_cache: dict = {"offers": [], "loaded_at": 0.0, "failed": False, "auth_error": False}
_lock  = threading.Lock()

# ── SQL Query ─────────────────────────────────────────────────────────────────
# Loaded from app/SQL/offer_query.sql at import time so the query can be
# edited without touching Python code.
from pathlib import Path as _Path

_SQL_FILE = _Path(__file__).resolve().parent.parent / "SQL" / "offer_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

# ── Specialty mapping table ───────────────────────────────────────────────────
# Maps hospital DB specialty names (state["speciality"]) → CRM offer specialty
# names (new_offer_equest.Specialty). Keys are lower-cased for matching.
# Add entries here as new specialties appear in the CRM.
_SPECIALTY_MAP: dict[str, list[str]] = {
    # Hospital DB name            CRM name(s)
    "dermatology":                     ["Dermatology"],
    "dermatology and cosmatology":     ["Dermatology"],
    "dermatology and cosmetology":     ["Dermatology"],
    "dental":                    ["Dental Services", "Dental"],
    "dental services":           ["Dental Services", "Dental"],
    "pedodontic":                ["Dental Services", "Dental"],   # pediatric dentistry → same CRM bucket
    "pedodontics":               ["Dental Services", "Dental"],
    "paedodontics":              ["Dental Services", "Dental"],
    "obstetrics":                ["OBE & GYN", "Obstetrics & Gynecology"],
    "gynecology":                ["OBE & GYN", "Obstetrics & Gynecology"],
    "obs & gyn":                 ["OBE & GYN"],
    "obe & gyn":                 ["OBE & GYN"],
    "obstetrics & gynecology":   ["OBE & GYN"],
    # IVF — fertility/pregnancy offers are filed under OBE & GYN in CRM,
    # so we must search both. The ICSI lab packages under "Screening"
    # and male-fertility offers under "Andrology" are added dynamically by
    # the hint-based expansion in get_offers_for_specialty() when the
    # service hint contains IVF-related keywords (حقن مجهري, عقم, etc.).
    "ivf":                       ["IVF", "OBE & GYN"],
    "internal medicine":         ["Internal Medicine"],
    "pediatrics":                ["Pediatrics"],
    "general pediatrics":        ["Pediatrics"],
    "paediatrics":               ["Pediatrics"],
    "nutrition":                 ["Nutrition"],
    "ent":                       ["E.N.T", "ENT"],
    "e.n.t.":                    ["E.N.T"],
    "e.n.t":                     ["E.N.T"],
    "ent head & neck surgery":   ["E.N.T"],
    "ear nose throat":           ["E.N.T"],
    "physiotherapy":             ["Physiotherapy"],
    "physical therapy":          ["Physiotherapy"],
    "andrology":                 ["Andrology"],
    "general surgery":           ["General Surgery"],
    "orthopedics":               ["Orthopedics"],
    "orthopaedics":              ["Orthopedics"],
    "orthopedic surgery":        ["Orthopedics"],
    "arthroplasty":              ["Orthopedics"],
    "spine":                     ["Orthopedics"],
    "psychology":                ["Psychology"],
    "psychiatry":                ["Psychiatry & Mental Health", "Psychology"],
    "speech therapy":            ["Speech and Phonetics"],
    "speech and phonetics":      ["Speech and Phonetics"],
    "family medicine":           ["Family Medicine"],
    "urology":                   ["Urology"],
    "endocrinology":             ["Endocrinology"],
    "chest":                     ["Chest"],
    "pulmonology":               ["Chest"],
    "ophthalmology":             ["Ophthalmology"],
    "cardiology":                ["Cardiology"],
    "cardiac surgery":           ["Cardiology"],
    "neurology":                 ["Neurology"],
    "rheumatology":              ["Internal Medicine", "Rheumatology"],
    "nephrology":                ["Internal Medicine", "Urology"],
    "gastroenterology":          ["GIT"],
    "git":                       ["GIT"],
    "bariatric surgery":         ["Bariatric Surgery"],
    "emergency":                 ["Emergency Medicine"],
    "occupational therapy":      ["Occupational therapy (OT)"],
    "laboratory":                ["Laboratory"],
    "radiology":                 ["Radiology"],
    "screening":                 ["Screening"],
}


def _crm_specialty_names(specialty_en: str) -> list[str]:
    """
    Given a hospital-DB specialty name, return the list of CRM specialty
    names to match against. Falls back to the raw name if no mapping found.
    """
    key = (specialty_en or "").strip().lower()
    mapped = _SPECIALTY_MAP.get(key)
    if mapped:
        return mapped
    # Partial match — catch cases like "Dermatology - Laser" → "Dermatology"
    for map_key, crm_names in _SPECIALTY_MAP.items():
        if map_key in key or key in map_key:
            return crm_names
    # No mapping — try the raw name as-is (title-cased)
    return [specialty_en.strip()]


# ── Reverse CRM → bookable-DB specialty mapping ──────────────────────────────
# CRM specialty categories that do NOT exist in the hospital scheduling DB.
# When an offer is accepted, we must NOT set state["speciality"] to these —
# doctor fetching would return 0 results. Map them to the nearest bookable
# hospital specialty instead.
_CRM_TO_BOOKABLE: dict[str, str] = {
    "screening":              "OBE & GYN",       # IVF lab packages → book under OBE & GYN
    "andrology":              "Urology",          # male fertility → book under Urology
    "git":                    "Internal Medicine", # gastro → Internal Medicine
    "psychiatry & mental health": "Psychiatry",
    "e.n.t":                  "E.N.T.",
    "occupational therapy (ot)": "Occupational Therapy",
}


def crm_to_bookable_specialty(crm_specialty: str) -> str:
    """Convert a CRM offer specialty name to a bookable hospital-DB specialty.

    If the CRM specialty is already a bookable name (e.g. "Dental Services"),
    returns it unchanged.  If it's a CRM-only category (e.g. "Screening"),
    maps it to the nearest bookable department.
    """
    key = (crm_specialty or "").strip().lower()
    return _CRM_TO_BOOKABLE.get(key, crm_specialty)


# ── Gender-aware offer filtering ──────────────────────────────────────────────

# Offer name keywords that indicate the procedure is for MALES ONLY.
# Any offer whose Arabic or English name contains one of these (word/substring)
# should be hidden when the patient is female.
import re as _re


# ── Session count + duration extraction ──────────────────────────────────────
_SESSION_COUNT_RE = _re.compile(
    r'(\d+)\s*(?:جلس[ةه]|جلسات|session)',
    _re.IGNORECASE,
)
_DURATION_HALF_RE = _re.compile(
    r'نص[ف]?\s*ساع[ةه]|half\s*hour',
    _re.IGNORECASE,
)
_DURATION_FULL_RE = _re.compile(
    r'(?<!\u0646\u0635\u0641\s)(?<!\u0646\u0635\s)\u0633\u0627\u0639[\u0629\u0647]|(?<!half\s)\bhour\b',
    _re.IGNORECASE,
)


def _extract_session_info(text: str) -> tuple[int | None, str | None]:
    """Extract session count and duration from text.

    Returns (count, duration) where duration is 'half'|'full'|None.
    Examples:
        '12 جلسه ساعه'       → (12, 'full')
        '8 جلسات نص ساعه'    → (8, 'half')
        'باقة 3 جلسات'       → (3, None)
        'علاج طبيعي'         → (None, None)
    """
    count = None
    m = _SESSION_COUNT_RE.search(text or "")
    if m:
        count = int(m.group(1))

    duration = None
    if _DURATION_HALF_RE.search(text or ""):
        duration = "half"
    elif _DURATION_FULL_RE.search(text or ""):
        duration = "full"
    # Also detect from offer names: "(نصف ساعة للجلسة)" / "(ساعة للجلسة)"
    if duration is None and text:
        if "نصف ساعة" in text or "نص ساعه" in text:
            duration = "half"
        elif "ساعة للجلسة" in text or "ساعه للجلسه" in text:
            duration = "full"
    return count, duration


# ── Similarity-based service hint validation ─────────────────────────────────
# Build vocabulary from CRM offer names at query time. Uses rapidfuzz for
# flexible matching so typos and colloquial variants pass validation.
_HINT_VOCAB_CACHE: dict[str, set[str]] = {"words": set(), "ts": 0.0}


def _build_offer_vocab() -> set[str]:
    """Extract unique words from all cached CRM offer names/descriptions."""
    now = time.time()
    if _HINT_VOCAB_CACHE["words"] and (now - _HINT_VOCAB_CACHE["ts"]) < 1800:
        return _HINT_VOCAB_CACHE["words"]
    words: set[str] = set()
    for offer in _cache.get("offers") or []:
        for field in ("Offer_Name_AR", "Offer_Name_EN", "Offer_Description_AR"):
            val = (offer.get(field) or "").strip()
            if val:
                for w in val.split():
                    w = w.strip("()–—-.,،؟?!")
                    if len(w) > 2:
                        words.add(w.lower())
    # Add core medical/service terms that may not appear in current offers
    _CORE_MEDICAL = {
        "جلسات", "جلسة", "جلسه", "عملية", "عمليات", "علاج", "فحص", "تحليل",
        "أشعة", "اشعه", "سونار", "تنظيف", "تبييض", "ليزر", "بلازما", "حقن",
        "بوتكس", "فيلر", "تقويم", "زراعة", "خلع", "حشو", "تركيب", "كشف",
        "استشارة", "متابعة", "تطعيم", "تلقيح", "غسيل", "مسح", "تصوير",
        "منظار", "قسطرة", "عظام", "قلب", "عيون", "اسنان", "جلدية", "باطنة",
        "نساء", "اطفال", "مخ", "اعصاب", "صدر", "كلى", "مسالك", "غدد",
        "تجميل", "تأهيل", "طبيعي", "نفسية", "تغذية",
    }
    words.update(_CORE_MEDICAL)
    _HINT_VOCAB_CACHE["words"] = words
    _HINT_VOCAB_CACHE["ts"] = now
    return words


def is_valid_service_hint(hint: str) -> bool:
    """Check if a service hint contains medically-relevant content.

    Uses fuzzy similarity (rapidfuzz token_set_ratio) against the CRM offer
    vocabulary so typos and colloquial variants are accepted.
    Returns False for conversational filler like 'رقمى الحالى'.
    """
    if not hint or not hint.strip():
        return False
    vocab = _build_offer_vocab()
    if not vocab:
        return True  # can't validate without vocab — assume valid
    words = [w for w in hint.split() if len(w) > 2]
    if not words:
        return False
    # Check each hint word against vocab with fuzzy matching
    for w in words:
        w_lower = w.lower()
        # Exact match
        if w_lower in vocab:
            return True
        # Strip ال prefix and check
        bare = w_lower[2:] if w_lower.startswith("ال") and len(w_lower) > 3 else w_lower
        if bare in vocab:
            return True
        # Fuzzy match — threshold 75 is lenient enough for typos
        for v in vocab:
            if len(v) >= 3 and _rfuzz.ratio(bare, v) >= 75:
                return True
    return False


_MALE_ONLY_KEYWORDS = _re.compile(
    r"(خصية|الخصية|خصيتين| القيلة المائية|القيلة|القيله|بروستاتا|البروستاتا|بروستات|القضيب|قضيب"
    r"|استئصال الخصية|orchi|prostat|testicular|testis|testes|vasectomy"
    r"|andrology|الأندروجين|عقم الذكور|fertility.*male|male.*fertility)",
    _re.IGNORECASE,
)




# Offer name keywords that indicate the procedure is for FEMALES ONLY.
_FEMALE_ONLY_KEYWORDS = _re.compile(
    r"(رحم|الرحم|مبيض|المبيض|ولادة|الولادة|قيصري|قيصرية|حمل|الحمل"
    r"|نزول الرحم|استئصال الرحم|عملية المبيض|تكيسات|حقن مجهري"
    r"|ivf|uterus|ovarian|ovary|hysterectomy|caesarean|cesarean"
    r"|obstetric|gynecolog|antenatal|prenatal|maternity|mammograph)",
    _re.IGNORECASE,
)

# ── "Booking for another person" detection ────────────────────────────────────
# When a male patient is in an OBE & GYN context, check if they mentioned
# booking for wife/daughter/mother. If yes → skip gender filter.
_BOOKING_FOR_OTHER_RE = _re.compile(
    r"(زوجتي|زوجت[ىي]|مرات[يى]|مراتي|حرم[تة]ي|حرمتي"
    r"|بنت[يى]|ابنت[يى]|بنتي|ابنتي"
    r"|أم[يى]|امي|والدت[يى]|والدتي"
    r"|أخت[يى]|اختي"
    r"|لزوجتي|لمراتي|لبنتي|لابنتي"
    r"|wife|daughter|mother|sister"
    r"|حامل|زوجتي حامل|مراتي حامل)",
    _re.IGNORECASE,
)

def infer_patient_gender(name: str | None) -> str | None:
    """Return 'male', 'female', or None (unknown) from patient first name.

    This is a lightweight FALLBACK used only when the conversation LLM has not
    yet emitted a patient_gender value into state (e.g. very first turn before
    the name is known).  The primary source of truth is state['patient_gender']
    which the LLM sets after seeing the patient's name.

    Uses Arabic linguistic suffix heuristics:
      • names ending in 'اء' / 'ية' / 'ة' (ta marbuta) / 'ى' → female
      • all other cases → None (unknown; don't guess, don't filter)
    """
    if not name:
        return None
    first = (name or "").strip().split()[0].strip()
    if first.endswith(("اء", "ية", "ّة", "ة", "ى", "يه")):
        return "female"
    return None


def get_patient_gender(state: dict) -> str | None:
    """Primary gender accessor used by the offer filter.

    Priority:
      1. state['patient_gender']  — set by the conversation LLM from the name.
      2. infer_patient_gender(state['patient_name'])  — suffix-based fallback.
      3. None — don't filter when uncertain.
    """
    gender = (state.get("patient_gender") or "").strip().lower()
    if gender in ("male", "female"):
        return gender
    # Fallback: Arabic suffix heuristic from the stored name
    return infer_patient_gender(state.get("patient_name"))


def _offer_gender_compatible(
    offer: dict, gender: str | None, *, skip_female_filter: bool = False,
) -> bool:
    """Return False if this offer is anatomically incompatible with the patient's gender.

    When gender is unknown (None), all offers pass through.
    When skip_female_filter is True, female-only keyword matching is bypassed
    (used when a male patient is booking for wife/daughter in OBE & GYN).
    """
    if not gender:
        return True  # can't determine — don't hide anything
    name_ar = (offer.get("Offer_Name_AR") or "").strip()
    name_en = (offer.get("Offer_Name_EN") or "").strip()
    combined = name_ar + " " + name_en
    if gender == "female" and _MALE_ONLY_KEYWORDS.search(combined):
        return False
    if gender == "male" and not skip_female_filter and _FEMALE_ONLY_KEYWORDS.search(combined):
        return False
    return True


def fetch_offers(force_refresh: bool = False) -> list[dict]:
    """
    Return all AHJ offers from CRM. Cached for _CACHE_TTL_SECONDS.
    Never raises — returns [] on failure so booking flow is unaffected.
    Thread-safe.
    """
    from app.services.crm_connector import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    with _lock:
        now  = time.time()
        age  = now - _cache["loaded_at"]
        data = _cache["offers"]

        if not force_refresh and data and age < _CACHE_TTL_SECONDS:
            return data

        # Back off 5 min after a failure to avoid hammering CRM —
        # BUT skip back-off for token-expiry errors: after _run_query_with_retry
        # clears the stale token, the very next call should go straight to CRM
        # with a fresh token instead of waiting 5 minutes.
        if _cache["failed"] and not _cache["auth_error"] and age < 300:
            return data

        print("[offers] Fetching offers from CRM…", flush=True)
        t0 = time.time()
        try:
            rows = _run_query_with_retry(_QUERY, max_attempts=3)
            print(f"[offers] Fetched {len(rows)} offers in {time.time()-t0:.1f}s", flush=True)
            _cache["offers"]     = rows
            _cache["loaded_at"]  = now
            _cache["failed"]     = False
            _cache["auth_error"] = False
            return rows
        except Exception as e:
            err_str = str(e).lower()
            is_auth = any(k in err_str for k in ("login failed", "connection expired", "28000", "crm auth failed"))
            print(f"[offers] fetch failed after {time.time()-t0:.1f}s: {e}", flush=True)
            _cache["failed"]     = True
            _cache["auth_error"] = is_auth   # don't back-off on token errors
            _cache["loaded_at"]  = now
            return data  # return stale data if any


def get_offers_for_specialty(
    specialty_en: str,
    service_hint: str = "",
    specialty_only: bool = False,
    patient_gender: str | None = None,
    patient_age: int | None = None,
    _booking_for_other_flag: bool = False,
) -> list[dict]:
    """
    Return offers matching `specialty_en`, sorted newest-first.
    Only returns the single most recent ACTIVE offer, plus any
    Inactive offers (for status messaging). Draft offers are excluded.

    If `service_hint` is provided (e.g. "بلازما", "ليزر"), the function
    first looks for active offers whose name (AR or EN) contains the hint
    word(s). If a service-specific match is found it is returned instead
    of the generic top-1 active offer, giving patients the most relevant
    result. Falls back to the most recent active offer when no name match
    is found.

    If `specialty_only=True`, "All Specialties" offers are always excluded
    regardless of whether a service_hint is provided. Use this for proactive
    mid-booking offer checks where only specialty-specific offers are relevant.

    If `patient_name` is provided, offers that are anatomically incompatible
    gender-incompatible with the patient are silently filtered out (e.g. male-only
    procedures hidden for a female patient).

    Returns:
        list[dict] — active offers (primary first, alternatives tagged
                     _is_alternative=True) + up to 2 inactive offers.
    """
    all_offers = fetch_offers()
    if not all_offers:
        print(
            f"[offers] get_offers_for_specialty({specialty_en!r}): "
            f"fetch_offers() returned empty — CRM unreachable or in back-off window",
            flush=True,
        )
        return []

    # ── Direct offer-name lookup (fast path) ──────────────────────────────────
    # When the patient (or transcript) mentions a specific offer name like
    # "باقة الصحة المتكاملة", search by name first before falling back to the
    # full specialty + scoring pipeline.  Uses rapidfuzz partial_ratio so
    # minor word-order / diacritic differences still match.
    if service_hint and len(service_hint.strip()) >= 4:
        _hint_norm = service_hint.strip().lower()
        _name_candidates: list[tuple[float, dict]] = []
        for _o in all_offers:
            if (_o.get("Offer_Status") or "") == STATUS_DRAFT:
                continue
            _on_ar = (_o.get("Offer_Name_AR") or "").strip().lower()
            _on_en = (_o.get("Offer_Name_EN") or "").strip().lower()
            _score = max(
                _rfuzz.partial_ratio(_hint_norm, _on_ar),
                _rfuzz.partial_ratio(_hint_norm, _on_en),
            )
            if _score >= 85:  # high-confidence name match
                _name_candidates.append((_score, _o))
        if _name_candidates:
            _name_candidates.sort(key=lambda x: -x[0])
            _best_score, _best_offer = _name_candidates[0]
            print(
                f"[offers] direct name match: hint={service_hint!r} → "
                f"{(_best_offer.get('Offer_Name_AR') or _best_offer.get('Offer_Name_EN'))!r} "
                f"(score={_best_score})",
                flush=True,
            )
            _best = dict(_best_offer)
            _best["_service_matched"] = True
            _best["_is_alternative"] = False
            _gender_ok = _offer_gender_compatible(_best, patient_gender)
            if not _gender_ok:
                print(
                    f"[offers] direct name match gender-filtered — continuing to specialty search",
                    flush=True,
                )
            else:
                print(
                    f"[offers] direct name match RETURNED offer:\n"
                    f"  Offer_ID      : {_best.get('Offer_ID')}\n"
                    f"  Offer_Name_AR : {_best.get('Offer_Name_AR')}\n"
                    f"  Offer_Name_EN : {_best.get('Offer_Name_EN')}\n"
                    f"  Specialty     : {_best.get('Specialty')}\n"
                    f"  Offer_Status  : {_best.get('Offer_Status')}\n"
                    f"  Price_Before  : {_best.get('Price_Before_Discount')}\n"
                    f"  Price_After   : {_best.get('Price_After_Discount')}\n"
                    f"  End_Date      : {_best.get('End_Date')}\n"
                    f"  match_score   : {_best_score}",
                    flush=True,
                )
                return [_best]

    crm_names = _crm_specialty_names(specialty_en)

    # Gender-aware base filtering: IVF maps to ['IVF', 'OBE & GYN'] but
    # for male patients (not booking for wife/daughter), remove OBE & GYN
    # from the base mapping — male infertility is under Andrology, not OBE.
    _gender_for_filter = (patient_gender or "").lower() or None
    if _gender_for_filter == "male" and not _booking_for_other_flag:
        _had_obe = "OBE & GYN" in crm_names
        crm_names = [n for n in crm_names if n != "OBE & GYN"]
        if _had_obe:
            print(
                f"[offers] base mapping: removed 'OBE & GYN' (male patient, no booking-for-other)",
                flush=True,
            )

    # ── Hint-based specialty expansion ────────────────────────────────────────
    # Some procedures (IVF/ICSI, fertility tests) have offers filed under a
    # DIFFERENT CRM specialty than the routing specialty. When the patient's
    # service hint contains keywords for such procedures, merge the relevant
    # CRM specialty names so the search isn't artificially narrow.
    _HINT_SPECIALTY_EXPANSIONS: list[tuple[_re.Pattern, list[str]]] = [
        # IVF / ICSI / fertility
        # The gender-appropriate specialties are filtered below.
        (_re.compile(
            r"(حقن\s*(ال)?مجهر|مجهر[يى]|المجهر[يى]|أطفال\s*(ال)?أنابيب|اطفال\s*(ال)?انابيب|تلقيح|ivf|icsi|iui"
            r"|تأخر\s*إنجاب|تاخر\s*انجاب|عقم|fertility|infertil"
            r"|خصوب[ةه]|الخصوب[ةه]|فحوصات\s*الع[قك]م)",
            _re.IGNORECASE,
        ), ["OBE & GYN", "IVF", "Screening", "Andrology"]),
        # Pregnancy / maternity → always include OBE & GYN
        (_re.compile(
            r"(حمل|الحمل|حامل|متابع[ةه]\s*(ال)?حمل|ولاد[ةه]|قيصري[ةه]?"
            r"|pregnancy|maternity|prenatal|antenatal|trimester"
            r"|ثلث|الثلث|ثلاث[ةه]?\s*شهور)",
            _re.IGNORECASE,
        ), ["OBE & GYN"]),
    ]
    if service_hint:
        for pattern, extra_crm_names in _HINT_SPECIALTY_EXPANSIONS:
            if pattern.search(service_hint):
                # Gender-aware filtering: don't add OBE & GYN for males
                # (unless booking for wife/daughter) and don't add
                # Andrology for females.
                _filtered = list(extra_crm_names)
                if _gender_for_filter == "male" and not _booking_for_other_flag:
                    _filtered = [n for n in _filtered if n != "OBE & GYN"]
                elif _gender_for_filter == "female":
                    _filtered = [n for n in _filtered if n != "Andrology"]
                added = [n for n in _filtered if n not in crm_names]
                if added:
                    crm_names = list(crm_names) + added
                    print(
                        f"[offers] hint-based specialty expansion: "
                        f"added {added} (gender={_gender_for_filter})",
                        flush=True,
                    )
                # Don't break — allow multiple patterns to match

    crm_names_lower = [n.lower() for n in crm_names]
    print(
        f"[offers] get_offers_for_specialty({specialty_en!r}): "
        f"cache has {len(all_offers)} offers | mapping to CRM names {crm_names}",
        flush=True,
    )

    # "All Specialties" offers are always excluded — they are generic
    # promotions (e.g. hair-loss packages) that don't relate to the patient's
    # actual specialty and confuse the booking flow.
    include_all_specs = False

    matched: list[dict] = []
    for offer in all_offers:
        offer_spec = (offer.get("Specialty") or "").strip().lower()
        status = (offer.get("Offer_Status") or "").strip()
        if status == STATUS_DRAFT:
            continue  # Draft: hide entirely
        is_all_specs = offer_spec == "all specialties"
        if is_all_specs:
            continue  # always skip "All Specialties" offers
        elif offer_spec in crm_names_lower:
            matched.append(offer)

    # Active offers: take only the most recent (index 0, already sorted DESC)
    active   = [o for o in matched if o.get("Offer_Status") == STATUS_ACTIVE]
    inactive = [o for o in matched if o.get("Offer_Status") == STATUS_INACTIVE]

    # Gender filter — remove anatomically incompatible offers
    # SPECIAL CASE: For OBE & GYN with male patients, check if they're booking
    # for another person (wife/daughter). If so, skip the female-only filter.
    gender = patient_gender or None
    _is_obe = any(n.lower().startswith("obe") for n in crm_names)
    _skip_female_filter = False
    if gender == "male" and _is_obe:
        # Check service_hint for spouse/daughter mentions
        if service_hint and _BOOKING_FOR_OTHER_RE.search(service_hint):
            _skip_female_filter = True
            print(
                f"[offers] gender filter: male patient booking for other person "
                f"(detected in hint) → skipping female-only filter for OBE & GYN",
                flush=True,
            )
        elif _booking_for_other_flag:
            _skip_female_filter = True
            print(
                f"[offers] gender filter: _booking_for_other flag set "
                f"→ skipping female-only filter for OBE & GYN",
                flush=True,
            )
    if gender:
        before = len(active)
        active   = [o for o in active   if _offer_gender_compatible(o, gender, skip_female_filter=_skip_female_filter)]
        inactive = [o for o in inactive if _offer_gender_compatible(o, gender, skip_female_filter=_skip_female_filter)]
        hidden = before - len(active)
        if hidden:
            print(
                f"[offers] gender filter ({gender}): removed {hidden} incompatible offer(s)",
                flush=True,
            )

    print(
        f"[offers] specialty match: {len(matched)} total ({len(active)} active, "
        f"{len(inactive)} inactive) for CRM names {crm_names}",
        flush=True,
    )
    if not matched:
        # Sample distinct specialty values to diagnose mismatches
        sample = list({
            (o.get("Specialty") or "").strip()
            for o in all_offers[:200]
        })
        print(f"[offers] sample CRM specialty values (first 200 rows): {sorted(sample)[:15]}", flush=True)

    # ── Procedure-type anti-match groups ──────────────────────────────────────
    # Words that are mutually exclusive: if patient's hint contains a word from
    # group A (e.g. "عمليات") and the offer name contains a word from group B
    # (e.g. "تحاليل"), that offer is excluded regardless of other score.
    _PROC_TYPE_GROUPS: list[set[str]] = [
        # Surgeries / procedures vs. lab tests
        {"عمليات", "عملية", "جراحة", "operation", "surgery", "surgical", "procedure"},
        {"تحاليل", "تحليل", "فحوصات", "فحص", "analysis", "lab", "labs", "test", "tests", "checkup", "screening"},
        # Sessions (laser/plasma etc.) vs. surgery
        {"جلسات", "جلسة", "session", "sessions"},
        # Radiology / imaging vs. labs
        {"اشعة", "أشعة", "تصوير", "radiology", "imaging", "xray", "x-ray", "scan", "mri"},
    ]

    def _get_proc_types(text: str) -> set[int]:
        """Return set of group-indices present in text."""
        t = text.lower()
        return {i for i, grp in enumerate(_PROC_TYPE_GROUPS) if any(w in t for w in grp)}

    # ── Tag active offers so the caller knows if the hint matched specifically ──
    _service_matched_flag = False

    # ── Semantic pre-match: get per-offer similarity scores ───────────────────
    # Run semantic matching for the hint query up-front so we can use the
    # scores as bonuses inside the per-offer loop below.
    _semantic_scores: dict[str, float] = {}   # offer_id → cosine similarity
    _semantic_top_name: str = ""
    if service_hint:
        try:
            from services.offer_semantic import semantic_match_offers as _sem_match
            # Map specialty_en to CRM specialty name for filter
            _crm_specs = _SPECIALTY_MAP.get((specialty_en or "").lower(), [])
            _sem_spec = _crm_specs[0] if _crm_specs else None
            _sem_results = _sem_match(
                service_hint,
                specialty_filter=_sem_spec,
                active_only=False,
                top_k=10,
                threshold=0.75,
            )
            if _sem_results:
                _semantic_top_name = _sem_results[0]["name_ar"] or _sem_results[0]["name_en"]
                for _sr in _sem_results:
                    _semantic_scores[_sr["name_ar"]] = _sr["score"]
                    _semantic_scores[_sr["name_en"]] = _sr["score"]
                print(
                    f"[offers] semantic pre-match: top={_semantic_top_name!r} "
                    f"(score={_sem_results[0]['score']}) | {len(_sem_results)} candidates",
                    flush=True,
                )
                # Log all semantic candidates for debugging
                for _si, _sr in enumerate(_sem_results[:5], 1):
                    print(
                        f"[offers]   sem #{_si}: score={_sr['score']:.4f} | "
                        f"{_sr['name_ar']!r} | status={_sr.get('status', '?')}",
                        flush=True,
                    )
        except Exception as _sem_err:
            print(f"[offers] semantic pre-match skipped: {_sem_err}", flush=True)

    if service_hint and active:
        raw_words = [w for w in service_hint.split() if len(w) > 2]
        # Expand: strip Arabic definite article so "الليزر" matches "ليزر", etc.
        hint_words: list[str] = []
        for w in raw_words:
            hint_words.append(w)
            if w.startswith("ال") and len(w) > 3:
                hint_words.append(w[2:])   # bare root without ال

        # ── Arabic text normalizer ────────────────────────────────────────────
        # Unifies hamza variants (أ/إ/آ → ا), removes tashkeel, strips ال,
        # and collapses doubled consonants so "تبيض" matches "تبييض", etc.
        import unicodedata as _ud
        def _norm_ar(text: str) -> str:
            t = text.lower()
            # Remove Arabic tashkeel (diacritics)
            t = _re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', t)
            # Normalize hamza variants
            t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
            t = t.replace("ؤ", "و").replace("ئ", "ي")
            t = t.replace("ة", "ه")  # ta marbuta → ha
            t = t.replace("ى", "ي")  # alef maqsura → ya
            # Collapse doubled Arabic letters: "تبييض" → "تبيض", "لييزر" → "ليزر"
            t = _re.sub(r'(.)\1+', r'\1', t)
            return t

        # ── Common spelling variant synonyms ──────────────────────────────────
        # Patient may use short/long Arabic spelling forms; CRM may have either.
        # Expand hint words to cover both variants so matching is robust.
        _SYNONYMS = {
            "تبيض": ["تبييض", "تبيض"],
            "تبييض": ["تبييض", "تبيض"],
            "تنضيف": ["تنظيف", "تنضيف"],
            "فلوريد": ["فلورايد", "فلوريد"],
            "لزر": ["ليزر", "لزر"],
            "بوتكس": ["بوتوكس", "بوتكس"],
            "فيلر": ["فيلر", "فللر"],
            # Arabic number words → digit + ordinal forms
            "تلات": ["3", "ثلاث", "ثالث", "الثالث", "تلات", "ثلاثه", "ثلاثة"],
            "ثلاث": ["3", "ثلاث", "ثالث", "الثالث", "تلات", "ثلاثه", "ثلاثة"],
            "اتنين": ["2", "اثنين", "ثاني", "الثاني", "اتنين"],
            "اربع": ["4", "أربع", "رابع", "الرابع", "اربعه", "أربعة"],
            "خمس": ["5", "خمسة", "خامس", "الخامس", "خمسه"],
            "ست": ["6", "سته", "ستة", "سادس", "السادس"],
            "سبع": ["7", "سبعة", "سبعه", "سابع", "السابع"],
            "تمان": ["8", "ثمانية", "ثمانيه", "ثامن", "الثامن", "تمانيه"],
            "تسع": ["9", "تسعة", "تسعه", "تاسع", "التاسع"],
            "عشر": ["10", "عشره", "عشرة", "عاشر", "العاشر"],
            "اتناشر": ["12", "اثنا عشر", "اثنى عشر"],
            # Trimester / period words
            "اخر": ["اخير", "الاخير", "الأخير", "ثالث", "الثالث", "آخر"],
            "شهور": ["شهر", "أشهر", "اشهر", "ثلث", "الثلث"],
            # ── Medical concept synonyms ──────────────────────────────────
            # Fertility / IVF — user says "خصوبة" but CRM uses "حقن مجهري"
            "خصوبة": ["حقن", "مجهري", "المجهري", "تلقيح", "أنابيب", "الأنابيب", "إخصاب", "خصوبه"],
            "خصوبه": ["حقن", "مجهري", "المجهري", "تلقيح", "أنابيب", "الأنابيب", "إخصاب", "خصوبة"],
            "عقم": ["حقن", "مجهري", "تلقيح", "خصوبة", "إخصاب"],
            "إخصاب": ["حقن", "مجهري", "تلقيح", "خصوبة", "أنابيب"],
            # Gender qualifiers — user says "المرأة" but CRM uses "للنساء"
            "المرأة": ["نساء", "للنساء", "النساء"],
            "مرأة": ["نساء", "للنساء", "النساء"],
            "الرجل": ["رجال", "للرجال", "الرجال", "ذكور"],
            "رجال": ["للرجال", "الرجال", "ذكور"],
        }

        # ── Trimester phrase detection ─────────────────────────────────────────
        # When the user says "اخر تلات شهور" (last 3 months), this is medically
        # equivalent to "الثلث الثالث" (third trimester). Inject the CRM term
        # into hint_words so the fuzzy scorer finds the exact offer name match.
        _TRIMESTER_PHRASES: list[tuple[_re.Pattern, list[str]]] = [
            # First trimester
            (_re.compile(r'(اول|أول|الاول|الأول)\s*(ثل[ثت]|تلات|ثلاث)', _re.IGNORECASE),
             ["الثلث", "الاول", "متابعه"]),
            # Second trimester
            (_re.compile(r'(ثاني|تاني|الثاني|التاني)', _re.IGNORECASE),
             ["الثلث", "الثاني", "متابعه"]),
            # Third trimester — "اخر تلات شهور" / "الثلث الثالث" / "آخر 3 شهور"
            (_re.compile(
                r'(اخر|آخر|الاخير|الأخير)\s*(تلات|ثلاث|3|ثلاثه|ثلاثة)?\s*(شهور|شهر|أشهر|اشهر)?'
                r'|(ثالث|تالت|الثالث)\s*(ثل[ثت])?',
                _re.IGNORECASE,
            ), ["الثلث", "الثالث", "متابعه"]),
        ]
        _hint_text = " ".join(hint_words)
        for _tri_pat, _tri_words in _TRIMESTER_PHRASES:
            if _tri_pat.search(_hint_text) or _tri_pat.search(service_hint):
                for tw in _tri_words:
                    if tw not in hint_words:
                        hint_words.append(tw)
                print(
                    f"[offers] trimester phrase detected → injected {_tri_words}",
                    flush=True,
                )
                break  # only one trimester match

        # Expand hint_words with synonyms
        expanded_hint_words = list(hint_words)
        for w in list(hint_words):
            normed_w = _norm_ar(w)
            for canon, variants in _SYNONYMS.items():
                if normed_w == _norm_ar(canon):
                    for v in variants:
                        if v not in expanded_hint_words:
                            expanded_hint_words.append(v)
                    break
        hint_words = expanded_hint_words

        # Normalize hint words for matching
        norm_hint_words = [_norm_ar(w) for w in hint_words]
        norm_raw_words  = [_norm_ar(w) for w in raw_words]

        # ── Specialty-generic word detection ──────────────────────────────────
        # A hint word is GENERIC if it appears in EVERY (or nearly every) offer
        # in the matched set — it has no discriminating value.
        # Two-pass approach:
        #  Pass 1: specialty-name words (e.g. "اسنان" for "Dental Services")
        #  Pass 2: ubiquity check — if a word appears in ≥80% of matched offers
        #          it is de-facto generic regardless of specialty-name mapping.
        #          This catches cases like Pedodontic (no AR mapping) where
        #          "اسنان" still appears in every dental offer.
        from config.constants import SPECIALTY_EN_TO_AR
        _spec_ar = SPECIALTY_EN_TO_AR.get(specialty_en, "")
        _spec_ar_words = set()
        if _spec_ar:
            for sw in _spec_ar.split():
                _spec_ar_words.add(_norm_ar(sw))
                if sw.startswith("ال") and len(sw) > 3:
                    _spec_ar_words.add(_norm_ar(sw[2:]))
        # Pass 1: specialty-name based flag
        _is_generic = [nw in _spec_ar_words for nw in norm_hint_words]
        # (Pass 2 ubiquity check applied after service_matched is built — see below)

        # Determine which procedure-type group(s) the patient's hint belongs to
        hint_proc_types = _get_proc_types(service_hint)

        if hint_words:
            service_matched = []
            for offer in active:
                name_ar  = (offer.get("Offer_Name_AR") or "").lower()
                name_en  = (offer.get("Offer_Name_EN") or "").lower()
                desc_ar  = (offer.get("Offer_Description_AR") or "").lower()
                desc_en  = (offer.get("Offer_Description_EN") or "").lower()
                tag      = (offer.get("Offer_Tag") or "").lower()
                offer_text = f"{name_ar} {name_en} {desc_ar} {desc_en} {tag}"

                # Normalized versions for Arabic matching
                norm_name_ar = _norm_ar(name_ar)
                norm_desc_ar = _norm_ar(desc_ar)

                # ── Anti-match exclusion ───────────────────────────────────────
                if hint_proc_types:
                    offer_proc_types = _get_proc_types(offer_text)
                    if offer_proc_types and offer_proc_types.isdisjoint(hint_proc_types):
                        print(
                            f"[offers] anti-match: hint_types={hint_proc_types} "
                            f"contradicts offer_types={offer_proc_types} for "
                            f"{(offer.get('Offer_Name_AR') or offer.get('Offer_Name_EN'))!r} → excluded",
                            flush=True,
                        )
                        continue

                # Name matches score double so name-specific offers rank higher
                # Use BOTH raw and normalized matching to catch hamza/tashkeel diffs
                # Skip generic specialty words — they match every offer and don't discriminate
                name_score = sum(
                    2 for w, nw, gen in zip(hint_words, norm_hint_words, _is_generic)
                    if not gen and (w in name_ar or w in name_en or nw in norm_name_ar)
                )
                desc_score = sum(
                    1 for w, nw, gen in zip(hint_words, norm_hint_words, _is_generic)
                    if not gen and (w in desc_ar or w in desc_en or w in tag or nw in norm_desc_ar)
                )
                score = name_score + desc_score

                # Bonus: if ALL original hint words (2+) match in name → strong relevance
                # Single-word hints match too broadly — no bonus for trivial matches
                if len(norm_raw_words) >= 2 and all(
                    nw in norm_name_ar or w in name_en
                    for w, nw in zip(raw_words, norm_raw_words)
                ):
                    score += 5  # strong bonus for full-phrase match

                # ── Session count + duration matching ─────────────────────────
                hint_count, hint_dur = _extract_session_info(service_hint)
                if hint_count is not None:
                    offer_count, offer_dur = _extract_session_info(
                        (offer.get("Offer_Name_AR") or "") + " " +
                        (offer.get("Offer_Name_EN") or "")
                    )
                    if offer_count is not None:
                        if offer_count == hint_count:
                            score += 15  # strong bonus for exact session count match
                        else:
                            score -= 8   # penalty for wrong count
                    if hint_dur and offer_dur:
                        if hint_dur == offer_dur:
                            score += 10  # bonus for matching duration
                        else:
                            score -= 5   # penalty for wrong duration

                # Age filter: skip child-specific offers for adult patients
                _CHILD_OFFER_KW = ("طفل", "الطفل", "أطفال", "اطفال", "الأطفال", "child", "pediatric", "paediatric")
                _combined_name = name_ar + " " + name_en
                _is_child_offer = any(kw in _combined_name for kw in _CHILD_OFFER_KW)

                if _is_child_offer:
                    if patient_age and patient_age >= 14:
                        print(
                            f"[offers] age filter: skipping child offer "
                            f"{(offer.get('Offer_Name_AR') or offer.get('Offer_Name_EN'))!r} "
                            f"for patient age={patient_age}",
                            flush=True,
                        )
                        continue
                    elif patient_age is None:
                        # Age unknown — deprioritize child offers so adult ones rank first
                        score -= 10

                # ── Semantic bonus ─────────────────────────────────────────
                if _semantic_scores:
                    _sem_score = max(
                        _semantic_scores.get(name_ar, 0.0),
                        _semantic_scores.get(name_en, 0.0),
                    )
                    if _sem_score >= 0.82:
                        score += int((_sem_score - 0.75) * 100)  # e.g. 0.90 → +15 bonus

                if score > 0:
                    service_matched.append((score, offer))
            if service_matched:
                # Save the full active list BEFORE narrowing it to keyword matches.
                # The semantic override scan needs ALL active offers (including ones
                # with score=0) to correctly identify active semantic candidates.
                _all_active_pre_kw = list(active)
                # Sort by score DESC (most hint-words matched first)
                service_matched.sort(key=lambda x: -x[0])
                # Within same score, prefer cheapest price
                top_score = service_matched[0][0]
                top_group = [o for s, o in service_matched if s == top_score]
                rest      = [o for s, o in service_matched if s < top_score]
                top_group.sort(key=lambda o: float(o.get("Price_After_Discount") or 9_999_999))
                active = top_group + rest
                _service_matched_flag = True
                print(
                    f"[offers] service hint {hint_words!r} matched "
                    f"{len(service_matched)} offer(s) | "
                    f"top={active[0].get('Offer_Name_AR')!r} (score={service_matched[0][0]})",
                    flush=True,
                )
                # ── Candidate detail logging ──────────────────────────────────
                for _rank, (_sc, _o) in enumerate(service_matched[:5], 1):
                    print(
                        f"[offers]   #{_rank} score={_sc} | "
                        f"{(_o.get('Offer_Name_AR') or _o.get('Offer_Name_EN'))!r}",
                        flush=True,
                    )

                # ── Semantic override guard ────────────────────────────────────
                # If ALL hint words that contributed to the keyword winner's score
                # are generic (appear in every/most offers), and the semantic model
                # found a high-confidence specific match (score ≥ 0.82), prefer the
                # semantic result.  Active matches are always preferred over inactive.
                if _semantic_top_name and _sem_results:
                    _sem_top_score = _sem_results[0]["score"]
                    # Bug fix: use the actual keyword winner object (service_matched[0][1])
                    # NOT active[0] which is just the first offer in the active list
                    # and may be a completely different offer from the keyword winner.
                    _kw_winner_obj     = service_matched[0][1]
                    _kw_winner_name_ar = (_kw_winner_obj.get("Offer_Name_AR") or "").strip()
                    _kw_winner_name_en = (_kw_winner_obj.get("Offer_Name_EN") or "").strip()
                    _sem_is_diff = (
                        _semantic_top_name != _kw_winner_name_ar
                        and _semantic_top_name != _kw_winner_name_en
                    )
                    _kw_top_score_val = service_matched[0][0]

                    # Pass 2: ubiquity check — mark a word generic if it appears
                    # in ≥80% of matched offers (catches unmapped specialties like
                    # Pedodontic where _spec_ar_words is empty).
                    _all_matched_names = [
                        _norm_ar((s_o.get("Offer_Name_AR") or "").lower())
                        for _, s_o in service_matched
                    ]
                    _n_matched = len(_all_matched_names) or 1
                    # Ubiquity threshold is only reliable with ≥ 3 matched offers.
                    # With 1-2 matches, 1/1=100% makes EVERY word look generic,
                    # triggering the semantic override even for specific terms like تبيض.
                    _ubiquity_reliable = _n_matched >= 3
                    _is_generic_ub = [
                        _is_generic[i] or (
                            _ubiquity_reliable and
                            sum(1 for mn in _all_matched_names if nw in mn) / _n_matched >= 0.80
                        )
                        for i, nw in enumerate(norm_hint_words)
                    ]

                    # Check if keyword winner scored only from generic words
                    _kw_norm_name = _norm_ar(_kw_winner_name_ar.lower())
                    _non_generic_score = sum(
                        2 for w, nw, gen in zip(hint_words, norm_hint_words, _is_generic_ub)
                        if not gen and (w in _kw_winner_name_ar.lower() or nw in _kw_norm_name)
                    )
                    _only_generic_hit = _non_generic_score == 0 and _kw_top_score_val > 0
                    if _sem_is_diff and _sem_top_score >= 0.82 and _only_generic_hit:
                        # Build lookup from ALL active offers (pre-keyword-filter).
                        # BUG FIX: `active` at this point is only the keyword-matched subset.
                        # Non-keyword-matched active offers (e.g. باقة تنظيف الأسنان which
                        # scored 0 keywords but IS semantically correct and IS Active) were
                        # missing from the lookup → wrongly treated as Inactive.
                        _lookup_base = _all_active_pre_kw if "_all_active_pre_kw" in locals() else active
                        _active_names_ar = {
                            (o.get("Offer_Name_AR") or "").strip() for o in _lookup_base
                        }
                        _active_names_en = {
                            (o.get("Offer_Name_EN") or "").strip() for o in _lookup_base
                        }
                        _active_by_name = {
                            (o.get("Offer_Name_AR") or "").strip(): o for o in _lookup_base
                        }
                        _active_by_name.update({
                            (o.get("Offer_Name_EN") or "").strip(): o for o in _lookup_base
                            if (o.get("Offer_Name_EN") or "").strip()
                        })

                        # Bug fix: scan ALL semantic candidates in rank order for an
                        # Active offer first.  The old code only checked sem#1 — if it
                        # was Inactive it returned "expired" without checking sem#2+.
                        # e.g. sem#1=تبييض الاسنان (Inactive, 0.8965)
                        #       sem#2=عرض تبييض الأسنان مع تنظيف الأسنان (Active, 0.8894)
                        # → should return sem#2, not the expired sem#1.
                        _sem_active_match   = None
                        _sem_active_score   = 0.0
                        _sem_inactive_match = None
                        _sem_inactive_score = 0.0

                        for _sr in _sem_results:
                            # _sem_results items use "name_ar"/"name_en" keys (not "name")
                            _sr_name_ar = (_sr.get("name_ar") or "").strip()
                            _sr_name_en = (_sr.get("name_en") or "").strip()
                            _sr_score   = _sr.get("score", 0.0)
                            if _sr_score < 0.82:
                                break  # below confidence threshold — stop scanning
                            # Check active list by both AR and EN name
                            _sr_active_obj = (
                                _active_by_name.get(_sr_name_ar)
                                or _active_by_name.get(_sr_name_en)
                            )
                            if _sr_active_obj is not None:
                                if _sem_active_match is None:
                                    _sem_active_match = _sr_active_obj
                                    _sem_active_score = _sr_score
                            else:
                                if _sem_inactive_match is None:
                                    _sem_inactive_match = next(
                                        (
                                            o for o in inactive
                                            if (o.get("Offer_Name_AR") or "").strip() == _sr_name_ar
                                            or (o.get("Offer_Name_EN") or "").strip() == _sr_name_en
                                            or (o.get("Offer_Name_AR") or "").strip() == _sr_name_en
                                            or (o.get("Offer_Name_EN") or "").strip() == _sr_name_ar
                                        ),
                                        None,
                                    )
                                    if _sem_inactive_match:
                                        _sem_inactive_score = _sr_score

                        # Prefer Active semantic match over any Inactive one
                        if _sem_active_match:
                            print(
                                f"[offers] semantic override (active): keyword winner "
                                f"{_kw_winner_name_ar!r} (generic-only score={_kw_top_score_val}) "
                                f"→ prefer semantic {(_sem_active_match.get('Offer_Name_AR') or '')!r} "
                                f"(sem_score={_sem_active_score:.3f})",
                                flush=True,
                            )
                            active = (
                                [_sem_active_match]
                                + [o for o in active if o is not _sem_active_match]
                            )
                        elif _sem_inactive_match:
                            # No Active semantic candidate found — show the closest
                            # expired offer so the patient knows it has ended.
                            print(
                                f"[offers] semantic override (inactive): keyword winner "
                                f"{_kw_winner_name_ar!r} (generic-only score={_kw_top_score_val}) "
                                f"→ all semantic candidates inactive; best="
                                f"{(_sem_inactive_match.get('Offer_Name_AR') or '')!r} "
                                f"(sem_score={_sem_inactive_score:.3f}) — returning as expired",
                                flush=True,
                            )
                            _sem_match_inactive = dict(_sem_inactive_match)
                            _sem_match_inactive["_best_match_is_inactive"] = True
                            _sem_match_inactive["_service_matched"] = True
                            return [_sem_match_inactive]
            else:
                # No fuzzy match — try semantic-only ranking
                if _semantic_scores and active:
                    _sem_ranked = []
                    for _ao in active:
                        _an_ar = (_ao.get("Offer_Name_AR") or "").strip()
                        _an_en = (_ao.get("Offer_Name_EN") or "").strip()
                        # Age filter: skip child-specific offers for adult patients
                        _combined_sem = _an_ar.lower() + " " + _an_en.lower()
                        _is_child_sem = any(kw in _combined_sem for kw in _CHILD_OFFER_KW)
                        if _is_child_sem and patient_age and patient_age >= 14:
                            print(
                                f"[offers] semantic-fb age filter: skipping child offer "
                                f"{_an_ar!r} for patient age={patient_age}",
                                flush=True,
                            )
                            continue
                        _sc = max(
                            _semantic_scores.get(_an_ar, 0.0),
                            _semantic_scores.get(_an_en, 0.0),
                        )
                        # Deprioritize child offers when age is unknown
                        if _is_child_sem and patient_age is None:
                            _sc -= 0.10
                        if _sc >= 0.75:
                            _sem_ranked.append((_sc, _ao))
                    if _sem_ranked:
                        # ── Tie-breaker: boost offers sharing hint keywords ───
                        # Semantic scores alone can misrank similar offers.
                        # Add a small bonus for offers whose name contains
                        # non-generic hint words, so the semantically-close
                        # AND keyword-relevant offer wins.
                        for _i, (_sc_val, _ao) in enumerate(_sem_ranked):
                            _n = _norm_ar((_ao.get("Offer_Name_AR") or "").lower())
                            _overlap = sum(
                                1 for nw, gen in zip(norm_hint_words, _is_generic)
                                if not gen and nw in _n
                            )
                            if _overlap:
                                _sem_ranked[_i] = (_sc_val + _overlap * 0.01, _ao)
                        _sem_ranked.sort(key=lambda x: -x[0])
                        active = [o for _, o in _sem_ranked]
                        _service_matched_flag = True
                        print(
                            f"[offers] semantic-only fallback: "
                            f"top={active[0].get('Offer_Name_AR')!r} "
                            f"(sem_score={_sem_ranked[0][0]:.3f})",
                            flush=True,
                        )
                        # Log all semantic-only candidates
                        for _rank, (_sc_val, _o) in enumerate(_sem_ranked[:5], 1):
                            print(
                                f"[offers]   sem-fb #{_rank} score={_sc_val:.4f} | "
                                f"{(_o.get('Offer_Name_AR') or _o.get('Offer_Name_EN'))!r}",
                                flush=True,
                            )
                    else:
                        # ── Bug #11 fix: check inactive offers for best semantic match ──
                        # If the top semantic hit is an INACTIVE offer that scores better
                        # than any active offer, tell the patient THAT offer expired
                        # rather than showing an unrelated active offer.
                        _inactive_sem: list[tuple[float, dict]] = []
                        for _io in inactive:
                            _in_ar = (_io.get("Offer_Name_AR") or "").strip()
                            _in_en = (_io.get("Offer_Name_EN") or "").strip()
                            _isc = max(
                                _semantic_scores.get(_in_ar, 0.0),
                                _semantic_scores.get(_in_en, 0.0),
                            )
                            if _isc >= 0.80:  # strong match
                                _inactive_sem.append((_isc, _io))
                        if _inactive_sem:
                            _inactive_sem.sort(key=lambda x: -x[0])
                            _best_inactive_score, _best_inactive = _inactive_sem[0]
                            print(
                                f"[offers] Bug#11: top semantic match is INACTIVE offer "
                                f"{_best_inactive.get('Offer_Name_AR')!r} "
                                f"(score={_best_inactive_score:.3f}) — tagging as expired",
                                flush=True,
                            )
                            _best_inactive["_best_match_is_inactive"] = True
                            _best_inactive["_service_matched"] = True
                            # Return this inactive offer — caller will show expiry message
                            return [_best_inactive]
                        print(
                            f"[offers] service hint {hint_words!r} → no fuzzy or semantic match "
                            f"→ falling back to cheapest active offer for specialty",
                            flush=True,
                        )
                else:
                    print(
                        f"[offers] service hint {hint_words!r} → no name/desc match "
                        f"→ falling back to cheapest active offer for specialty",
                        flush=True,
                    )
                # active stays as-is (sorted by price below)
    elif active:
        # No hint given — still sort by cheapest among same-service duplicates
        # Group by name; if >1 active for same specialty, prefer cheapest
        active.sort(key=lambda o: float(o.get("Price_After_Discount") or 9_999_999))

    result: list[dict] = []
    if active:
        # First active offer is the primary (best match / cheapest among ties)
        top = dict(active[0])
        top["_service_matched"] = _service_matched_flag
        top["_is_alternative"] = False
        result.append(top)
        # Remaining active offers are alternatives
        for alt in active[1:]:
            a = dict(alt)
            a["_service_matched"] = _service_matched_flag
            a["_is_alternative"] = True
            result.append(a)
    result.extend(inactive[:2])
    return result



def _fmt_num(value) -> str:
    """Format a numeric value cleanly — strips redundant .0 from whole numbers.

    Examples:  500.0 → "500"  |  249.5 → "249.5"  |  20.0 → "20"
    """
    try:
        f = float(value)
        if f == int(f):
            return str(int(f))
        return f"{f:.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _discount_pct(price_bef, price_aft) -> str:
    """Compute discount percentage from before/after prices.

    Returns a clean percentage string like \"36%\" or empty string if the
    data is missing or would produce a nonsensical result.
    The CRM field `new_offerdiscountf` stores the absolute SAR discount
    amount (e.g. 1333 SAR), NOT a percentage — so we compute it ourselves.
    """
    try:
        bef = float(price_bef or 0)
        aft = float(price_aft or 0)
        if bef > 0 and aft >= 0 and bef > aft:
            pct = round((bef - aft) / bef * 100)
            return f"{pct}%"
    except (TypeError, ValueError):
        pass
    return ""


def format_offer_card(offer: dict, lang: str) -> str:
    """
    Format one offer as a patient-facing card string.
    """
    name_en   = (offer.get("Offer_Name_EN") or "").strip()
    name_ar   = (offer.get("Offer_Name_AR") or "").strip()
    price_bef = offer.get("Price_Before_Discount")
    price_aft = offer.get("Price_After_Discount")
    discount  = offer.get("Discount")
    end_date  = offer.get("End_Date")
    specialty = (offer.get("Specialty") or "").strip()

    # Format end date nicely
    end_str = ""
    if end_date:
        try:
            from datetime import datetime
            if hasattr(end_date, "strftime"):
                end_str = end_date.strftime("%d %b %Y")
            else:
                dt = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")
                end_str = dt.strftime("%d %b %Y")
        except Exception:
            end_str = str(end_date)[:10]

    if lang == "ar":
        name = name_ar or name_en
        lines = []
        if name:
            lines.append(f"📋 اسم العرض: *{name}*")
        if specialty:
            lines.append(f"🏥 التخصص: {specialty}")
        if price_aft and float(price_aft or 0) > 0:
            lines.append(f"💰 السعر بعد الخصم: {_fmt_num(price_aft)} ريال")

        if end_str:
            lines.append(f"📅 ينتهي: {end_str}")
    else:
        name = name_en or name_ar
        lines = []
        if name:
            lines.append(f"📋 Offer: *{name}*")
        if specialty:
            lines.append(f"🏥 Specialty: {specialty}")
        if price_aft and float(price_aft or 0) > 0:
            lines.append(f"💰 Price after discount: {_fmt_num(price_aft)} SAR")

        if end_str:
            lines.append(f"📅 Valid until: {end_str}")

    return "\n".join(lines)
