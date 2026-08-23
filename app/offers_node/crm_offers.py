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

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_ACTIVE   = "Active"
STATUS_INACTIVE = "Inactive"
STATUS_DRAFT    = "Draft"

# Cache TTL — offers change more often than doctor prices, so 30 min
_CACHE_TTL_SECONDS = 30 * 60

_cache: dict = {"offers": [], "loaded_at": 0.0, "failed": False, "auth_error": False}
_lock  = threading.Lock()

# ── SQL Query ─────────────────────────────────────────────────────────────────
# Filtered to BU=AHJ only. Ordered newest first so top-1 selection is
# always the most recently created active offer.
_QUERY = """
SELECT
     [new_offerid]                   AS Offer_ID
    ,cr4be_dotcarecodenname          AS Offer_Name_EN
    ,[new_offernamear]               AS Offer_Name_AR
    ,[new_offerdescriptionen]        AS Offer_Description_EN
    ,[new_offerdescriptionar]        AS Offer_Description_AR
    ,[new_buname]                    AS BU
    ,[new_specialtyname]             AS Specialty
    ,[new_offerstatusnewname]        AS Offer_Status
    ,[new_publicationstatusname]     AS Publication_Status
    ,[new_totalofferbeforediscf]     AS Price_Before_Discount
    ,[new_totalofferafterdiscf]      AS Price_After_Discount
    ,[new_offerdiscountf]            AS Discount
    ,[new_startdatef]                AS Start_Date
    ,[new_offerenddatef]             AS End_Date
    ,[new_offertypef]                AS Offer_Type
    ,[new_offertagname]              AS Offer_Tag
    ,[new_servicecodedataset]        AS DotCare_Code
    ,[createdon]                     AS Created_On
FROM new_offer_equest
ORDER BY [createdon] DESC
"""

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
    # IVF — base map is IVF-only. The ICSI lab packages under "Screening"
    # and male-fertility offers under "Andrology" are added dynamically by
    # the hint-based expansion in get_offers_for_specialty() when the
    # service hint contains IVF-related keywords (حقن مجهري, عقم, etc.).
    "ivf":                       ["IVF"],
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


def _offer_gender_compatible(offer: dict, gender: str | None) -> bool:
    """Return False if this offer is anatomically incompatible with the patient's gender.

    When gender is unknown (None), all offers pass through.
    """
    if not gender:
        return True  # can't determine — don't hide anything
    name_ar = (offer.get("Offer_Name_AR") or "").strip()
    name_en = (offer.get("Offer_Name_EN") or "").strip()
    combined = name_ar + " " + name_en
    if gender == "female" and _MALE_ONLY_KEYWORDS.search(combined):
        return False
    if gender == "male" and _FEMALE_ONLY_KEYWORDS.search(combined):
        return False
    return True


def fetch_offers(force_refresh: bool = False) -> list[dict]:
    """
    Return all AHJ offers from CRM. Cached for _CACHE_TTL_SECONDS.
    Never raises — returns [] on failure so booking flow is unaffected.
    Thread-safe.
    """
    from app.offers_node.crm_database import _run_query_with_retry, _is_configured

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

        print("[offers] Fetching AHJ offers from CRM…", flush=True)
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

    crm_names = _crm_specialty_names(specialty_en)

    # ── Hint-based specialty expansion ────────────────────────────────────────
    # Some procedures (IVF/ICSI, fertility tests) have offers filed under a
    # DIFFERENT CRM specialty than the routing specialty. When the patient's
    # service hint contains keywords for such procedures, merge the relevant
    # CRM specialty names so the search isn't artificially narrow.
    _HINT_SPECIALTY_EXPANSIONS: list[tuple[_re.Pattern, list[str]]] = [
        # IVF / ICSI / fertility → also search IVF, Screening, Andrology
        (_re.compile(
            r"(حقن\s*(ال)?مجهر|مجهر[يى]|المجهر[يى]|أطفال\s*(ال)?أنابيب|اطفال\s*(ال)?انابيب|تلقيح|ivf|icsi|iui"
            r"|تأخر\s*إنجاب|تاخر\s*انجاب|عقم|fertility|infertil)",
            _re.IGNORECASE,
        ), ["IVF", "Screening"]),#, "Andrology"
    ]
    if service_hint:
        for pattern, extra_crm_names in _HINT_SPECIALTY_EXPANSIONS:
            if pattern.search(service_hint):
                added = [n for n in extra_crm_names if n not in crm_names]
                if added:
                    crm_names = list(crm_names) + added
                    print(
                        f"[offers] hint-based specialty expansion: "
                        f"added {added} for hint containing IVF/fertility keywords",
                        flush=True,
                    )
                break

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
    gender = patient_gender or None
    if gender:
        before = len(active)
        active   = [o for o in active   if _offer_gender_compatible(o, gender)]
        inactive = [o for o in inactive if _offer_gender_compatible(o, gender)]
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
        }

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
        # Words that are just the specialty's Arabic name (e.g. "اسنان" for
        # "Dental Services" = "خدمات الأسنان") appear in EVERY offer within
        # that specialty. They add no discriminating value and inflate scores
        # for unrelated offers. Mark them so scoring uses only service-specific
        # words (e.g. "فلورايد", "تنظيف") for ranking.
        from config.constants import SPECIALTY_EN_TO_AR
        _spec_ar = SPECIALTY_EN_TO_AR.get(specialty_en, "")
        _spec_ar_words = set()
        if _spec_ar:
            for sw in _spec_ar.split():
                _spec_ar_words.add(_norm_ar(sw))
                # Also add without ال prefix
                if sw.startswith("ال") and len(sw) > 3:
                    _spec_ar_words.add(_norm_ar(sw[2:]))
        # Mark which hint words are generic (specialty name only)
        _is_generic = [nw in _spec_ar_words for nw in norm_hint_words]

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

                if score > 0:
                    service_matched.append((score, offer))
            if service_matched:
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
            else:
                # No match after anti-exclusion — clear active so caller can say
                # "no specific offer found for this service type"
                print(
                    f"[offers] service hint {hint_words!r} → all offers excluded by "
                    f"anti-match or no name/desc match → no active offer returned",
                    flush=True,
                )
                active = []   # caller will handle "no offer" message
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
        if price_bef and float(price_bef or 0) > 0:
            lines.append(f"🔖 السعر الأصلي: {_fmt_num(price_bef)} ريال")
        pct = _discount_pct(price_bef, price_aft)
        if pct:
            lines.append(f"✂️ الخصم: {pct}")
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
        if price_bef and float(price_bef or 0) > 0:
            lines.append(f"🔖 Original price: {_fmt_num(price_bef)} SAR")
        pct = _discount_pct(price_bef, price_aft)
        if pct:
            lines.append(f"✂️ Discount: {pct}")
        if end_str:
            lines.append(f"📅 Valid until: {end_str}")

    return "\n".join(lines)
