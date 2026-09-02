"""
offer_search.py — Offer Lookup by Specialty + Service Hint
============================================================
Standalone extraction of the offer-search helpers from
nodes/offers_check.py in the main contact-center-chatbot project.

No LangGraph / state-machine dependencies — safe to import anywhere.

This module handles:
  1. Specialty resolution   — free Arabic/English text → EN specialty name
  2. Service hint extraction — strip boilerplate, keep the medical keyword
  3. Ambiguous service detection — "ليزر" could be dermatology OR ophthalmology
  4. CRM offer lookup        — calls get_offers_for_specialty() with the
                               resolved specialty + clean hint

Public API
----------
    resolve_specialty(text, known_specialties_en=None) -> str
        Map a free-text patient message → EN specialty name (or "" if unknown).

    extract_service_hint(text) -> str
        Strip offer-query boilerplate; return the core service keyword.

    detect_ambiguous_service(text) -> list[tuple[str, str]]
        Returns [(en_specialty, ar_label), ...] if the service is ambiguous.

    search_offers(specialty_en, service_hint, patient_gender, patient_age)
        -> list[dict]
        Call the CRM layer and return ranked offers for the specialty + hint.

    format_offer_response(offer, lang) -> str
        Format an active offer as a patient-facing card string.
"""
from __future__ import annotations
import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Arabic alias table: colloquial terms → EN specialty name
# ─────────────────────────────────────────────────────────────────────────────
_AR_ALIAS: dict[str, str] = {
    # ── Orthopedics ──────────────────────────────────────────────────────────
    "عظام":              "Orthopedics",
    "عظم":               "Orthopedics",
    "العظام":            "Orthopedics",
    "العظم":             "Orthopedics",
    "عظمية":             "Orthopedics",
    "العظميه":           "Orthopedics",
    "جراحة عظام":        "Orthopedics",
    "العظام والمفاصل":   "Orthopedics",
    # ── Cardiology ───────────────────────────────────────────────────────────
    "قلب":               "Cardiology",
    "القلب":             "Cardiology",
    "قلبية":             "Cardiology",
    "القلبيه":           "Cardiology",
    # ── Internal Medicine ────────────────────────────────────────────────────
    "باطنه":             "Internal Medicine",
    "باطنيه":            "Internal Medicine",
    "طب باطني":          "Internal Medicine",
    "الباطنيه":          "Internal Medicine",
    "الباطنه":           "Internal Medicine",
    # ── Dermatology ──────────────────────────────────────────────────────────
    "جلدية":             "Dermatology and Cosmatology",
    "جلديه":             "Dermatology and Cosmatology",
    "الجلديه":           "Dermatology and Cosmatology",
    "الجلدية":           "Dermatology and Cosmatology",
    "طب جلدي":           "Dermatology and Cosmatology",
    # ── ENT ──────────────────────────────────────────────────────────────────
    "انف اذن":           "E.N.T.",
    "انف واذن":          "E.N.T.",
    "الانف والاذن":      "E.N.T.",
    "حنجره":             "E.N.T.",
    "انف":               "E.N.T.",
    "اذن":               "E.N.T.",
    # ── Ophthalmology ────────────────────────────────────────────────────────
    "عيون":              "Ophthalmology",
    "العيون":            "Ophthalmology",
    "طب عيون":           "Ophthalmology",
    # ── IVF / Fertility — MUST come before Pediatrics ────────────────────────
    "حقن مجهري":         "IVF",
    "الحقن المجهري":     "IVF",
    "الحقن مجهري":       "IVF",
    "مجهري":             "IVF",
    "انابيب":            "IVF",
    "الانابيب":          "IVF",
    "اطفال انابيب":      "IVF",
    "أطفال أنابيب":      "IVF",
    "أطفال الأنابيب":    "IVF",
    "ivf":               "IVF",
    "icsi":              "IVF",
    "تلقيح صناعي":       "IVF",
    "عقم":               "IVF",
    "خصوبة":             "IVF",
    "ضعف خصوبة":         "IVF",
    "حمل اطفال انابيب":  "IVF",
    # ── Pediatrics ───────────────────────────────────────────────────────────
    "اطفال":             "General Pediatrics",
    "الاطفال":           "General Pediatrics",
    "طب اطفال":          "General Pediatrics",
    "أطفال":             "General Pediatrics",
    # ── Gynecology ───────────────────────────────────────────────────────────
    "نساء":              "OBE & GYN",
    "نسائيه":            "OBE & GYN",
    "النسائيه":          "OBE & GYN",
    "توليد":             "OBE & GYN",
    "نسائية":            "OBE & GYN",
    "نساء وتوليد":       "OBE & GYN",
    "النساء والتوليد":   "OBE & GYN",
    # ── Neurology ────────────────────────────────────────────────────────────
    "مخ":                "Neurology",
    "اعصاب":             "Neurology",
    "المخ والاعصاب":     "Neurology",
    "الاعصاب":           "Neurology",
    # ── Urology ──────────────────────────────────────────────────────────────
    "بوليه":             "Urology",
    "البوليه":           "Urology",
    "المسالك البولية":   "Urology",
    "مسالك":             "Urology",
    # ── General Surgery ──────────────────────────────────────────────────────
    "جراحة عامه":        "General Surgery",
    "جراحه":             "General Surgery",
    "جراحة عامة":        "General Surgery",
    # ── Oncology — verified against real CRM doctor data (cr301_specialtyname/
    # cr301_subspecialtyname = "Oncology" for a real, active doctor record;
    # see app.service_hub.doctor_validation's specialty-canonicalisation
    # comparison for how a CRM variant like "Medical Oncology" still safely
    # matches this same category) ─────────────────────────────────────────
    "اورام":             "Oncology",
    "أورام":             "Oncology",
    "الأورام":           "Oncology",
    "الاورام":           "Oncology",
    "طب اورام":          "Oncology",
    "طب أورام":          "Oncology",
    "عيادة الاورام":     "Oncology",
    "عيادة الأورام":     "Oncology",
    # ── Endocrinology ────────────────────────────────────────────────────────
    "غدد":               "Endocrinology",
    "الغدد":             "Endocrinology",
    "سكر":               "Endocrinology",
    "الغدد الصماء":      "Endocrinology",
    "سكري":              "Endocrinology",
    # ── Psychiatry ───────────────────────────────────────────────────────────
    "نفسيه":             "Psychiatry",
    "النفسيه":           "Psychiatry",
    "طب نفسي":           "Psychiatry",
    "نفسية":             "Psychiatry",
    # ── Rheumatology ─────────────────────────────────────────────────────────
    "روماتيزم":          "Rheumatology",
    "روماتيزميه":        "Rheumatology",
    "روماتيزمية":        "Rheumatology",
    # ── Dental ───────────────────────────────────────────────────────────────
    "أسنان":             "Dental Services",
    "اسنان":             "Dental Services",
    "الأسنان":           "Dental Services",
    "الاسنان":           "Dental Services",
    "طب أسنان":          "Dental Services",
    "طب الأسنان":        "Dental Services",
    "طب الاسنان":        "Dental Services",
    "ضرس":               "Dental Services",
    "ضروس":              "Dental Services",
    "تسوس":              "Dental Services",
    "تقويم":             "Dental Services",
    "تقويم أسنان":       "Dental Services",
    "زراعة أسنان":       "Dental Services",
    "تنظيف أسنان":       "Dental Services",
    "طب الفم":           "Dental Services",
    # ── GIT / Gastroenterology ───────────────────────────────────────────────
    "معدة":              "GIT",
    "المعدة":            "GIT",
    "هضمي":              "GIT",
    "الجهاز الهضمي":     "GIT",
    "كبد":               "GIT",
    "الكبد":             "GIT",
    "قولون":             "GIT",
    "القولون":           "GIT",
    "حموضة":             "GIT",
    "باطنية هضمية":      "GIT",
    # ── Chest / Pulmonology ──────────────────────────────────────────────────
    "صدر":               "Chest",
    "الصدر":             "Chest",
    "رئة":               "Chest",
    "الرئة":             "Chest",
    "تنفس":              "Chest",
    "الجهاز التنفسي":    "Chest",
    "ربو":               "Chest",
    # ── Nephrology ───────────────────────────────────────────────────────────
    "كلى":               "Nephrology",
    "الكلية":            "Nephrology",
    "غسيل كلى":          "Nephrology",
    # ── Bariatric Surgery ────────────────────────────────────────────────────
    "سمنة":              "Bariatric Surgery",
    "السمنة":            "Bariatric Surgery",
    "تكميم":             "Bariatric Surgery",
    "تكميم معدة":        "Bariatric Surgery",
    "بالون معدة":        "Bariatric Surgery",
    "تحويل مسار":        "Bariatric Surgery",
    # ── Laboratory ───────────────────────────────────────────────────────────
    "تحاليل":            "Laboratory",
    "مختبر":             "Laboratory",
    "المختبر":           "Laboratory",
    # ── Radiology ────────────────────────────────────────────────────────────
    "أشعة":              "Radiology",
    "الأشعة":            "Radiology",
    "اشعة":              "Radiology",
    "سونار":             "Radiology",
    "رنين":              "Radiology",
    # ── Cosmetic — unambiguous (always Dermatology) ──────────────────────────
    "كوزميتولوجي":       "Dermatology",
    "ليزر جلدي":         "Dermatology",
    "ازاله شعر":         "Dermatology",
    "جنتل ليزر":        "Dermatology",
    "الجنتل ليزر":      "Dermatology",
    "ليزر للجسم":       "Dermatology",
    "ليزر الجسم":       "Dermatology",
    "ليزر البشرة":      "Dermatology",
    "ليزر الشعر":       "Dermatology",
    # NOTE: bare "ليزر" / "تجميل" are intentionally NOT here —
    # they are ambiguous and handled by _AMBIGUOUS_SERVICES below.
    # ── Physiotherapy ────────────────────────────────────────────────────────
    "علاج طبيعي":        "Physiotherapy",
    "العلاج الطبيعي":    "Physiotherapy",
    "طبيعي":             "Physiotherapy",
    "فيزيوثيرابي":       "Physiotherapy",
    "تأهيل":             "Physiotherapy",
    "اعادة تاهيل":       "Physiotherapy",
    # ── Plastic Surgery ──────────────────────────────────────────────────────
    "جراحة تجميل":       "Plastic Surgery",
    "جراحة تجميليه":     "Plastic Surgery",
    "جراحة تجميلية":     "Plastic Surgery",
    "تجميل جراحي":       "Plastic Surgery",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ambiguous service table
# ─────────────────────────────────────────────────────────────────────────────
_AMBIGUOUS_SERVICES: dict[str, list[tuple[str, str]]] = {
    # ليزر could mean skin laser (dermatology) or LASIK (ophthalmology)
    "ليزر":   [("Dermatology", "ليزر الجلد (جلدية)"),      ("Ophthalmology", "ليزر العيون / ليزك (عيون)")],
    "الليزر": [("Dermatology", "ليزر الجلد (جلدية)"),      ("Ophthalmology", "ليزر العيون / ليزك (عيون)")],
    "laser":  [("Dermatology", "Skin Laser (Dermatology)"), ("Ophthalmology", "Eye Laser LASIK (Ophthalmology)")],
    # تجميل could mean cosmetic dermatology or plastic surgery
    "تجميل":  [("Dermatology", "تجميل الجلد (جلدية)"),     ("General Surgery", "جراحة تجميلية (جراحة)")],
    "التجميل":[("Dermatology", "تجميل الجلد (جلدية)"),     ("General Surgery", "جراحة تجميلية (جراحة)")],
    "تجميليه":[("Dermatology", "تجميل الجلد (جلدية)"),     ("General Surgery", "جراحة تجميلية (جراحة)")],
    # غسيل could mean dialysis (nephrology) or teeth cleaning (dental)
    "غسيل":   [("Nephrology", "غسيل الكلى (كلى)"),         ("Dental Services", "تنظيف الأسنان (أسنان)")],
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Service-hint stop-word list (stripped before passing hint to CRM search)
# ─────────────────────────────────────────────────────────────────────────────
_OFFER_STOP_RE = re.compile(
    r"\b(?:"
    r"هل|في|فى|يوجد|عندكم|عندك|لديكم|عندنا|"
    r"عروض|عرض|خصم|خصومات|تخفيض|تنزيل|بروموشن|باقة|"
    r"بخصوص|على|عن|بشأن|تخص|هناك|توجد|"
    r"عملية|عمليه|خدمة|خدمه|تخصص|"
    r"تمام|اوكيه|صح|واضح|نفسها|نفسه|نفس|ذاتها|ذاته|النفس|تبغي|تبغى|"
    r"موافق|موافقه|موافقة|نعم|ايوه|ايه|اه|اها|ابشر|طيب|ماشي|اكيد|زين|"
    r"حابب|حابه|حابة|بقولك|ابغى|ابي|عايز|عايزه|اريد|محتاج|محتاجه|"
    r"مافيش|مافيهش|مافي|يعنى|يعني|ليه|ليها|ياعنى|ولا|ولالا|إذا|اذا|"
    r"تفاصيل|تفاضيل|معلومات|تفصيل|وصف|وصفي|اشرح|شرح|وضح|فسر|اخبرني|خبرني|"
    r"رقم|رقمي|رقمى|الرقم|نمره|نمرة|تليفون|موبايل|جوال|هاتف|فون|"
    r"الحالي|الحالى|حالي|حالى|كده|كذا|بتاعي|بتاعى|ده|دي|دى|هو|هي|هى|"
    r"offer[s]?|discount|deal|package|for|about|on|any|the|a|an|is|are|there|details|info"
    r")\b"
    r"|[؟?!،,.\u060c]",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Arabic text normalizer (standalone — no external deps)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_ar(text: str) -> str:
    """Normalize Arabic: strip tashkeel, unify hamza, strip ال prefix."""
    t = text.lower()
    # Remove tashkeel diacritics
    t = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', t)
    # Normalize hamza variants
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("ة", "ه")   # ta marbuta → ha
    t = t.replace("ى", "ي")   # alef maqsura → ya
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_ambiguous_service(text: str) -> list[tuple[str, str]]:
    """Return disambiguation options if text contains an ambiguous service keyword.

    Returns a list of (EN_specialty, AR_label) pairs, or [] if unambiguous.
    The caller should ask the patient to choose before proceeding.

    Example:
        detect_ambiguous_service("في عروض على الليزر؟")
        → [("Dermatology", "ليزر الجلد (جلدية)"),
           ("Ophthalmology", "ليزر العيون / ليزك (عيون)")]
    """
    norm = _normalize_ar((text or "").strip())
    for keyword, options in _AMBIGUOUS_SERVICES.items():
        kw_norm = _normalize_ar(keyword)
        if kw_norm and kw_norm in norm:
            return options
    return []


def resolve_specialty(text: str, known_specialties_en: list[str] | None = None) -> str:
    """Map a free-text patient message to an EN specialty name.

    Strategy (priority order):
      1. Direct EN specialty name substring match
      2. Colloquial Arabic alias table (_AR_ALIAS)

    Parameters
    ----------
    text : str
        Raw patient message or extracted specialty hint.
    known_specialties_en : list[str] | None
        Optional list of valid EN specialty names from your system.
        If not provided, falls back to alias table only.

    Returns
    -------
    str
        EN specialty name (e.g. "Dermatology") or "" if not recognized.

    Examples
    --------
        resolve_specialty("عندي مشكله بالقلب")     → "Cardiology"
        resolve_specialty("مشكله في الجلد")         → "Dermatology and Cosmatology"
        resolve_specialty("ophthalmology offers")   → "Ophthalmology"
    """
    if not text:
        return ""

    normalized = _normalize_ar(text.strip())

    # 1. Direct EN match
    if known_specialties_en:
        for spec_en in known_specialties_en:
            if spec_en.lower() in normalized:
                return spec_en

    # 2. Colloquial Arabic alias (longest match wins)
    best_match = ""
    best_len = 0
    for alias, en_name in _AR_ALIAS.items():
        alias_norm = _normalize_ar(alias)
        if alias_norm and alias_norm in normalized:
            if len(alias_norm) > best_len:
                best_match = en_name
                best_len = len(alias_norm)
    if best_match:
        return best_match

    return ""


def extract_service_hint(text: str) -> str:
    """Strip offer-query boilerplate; return the core medical service keyword.

    The returned string is passed as `service_hint` to `search_offers()` so
    the CRM lookup can find an offer specific to that service (e.g. "بلازما",
    "ليزر", "تنظيف أسنان") rather than just the specialty's top active offer.

    Examples
    --------
        "هل يوجد عروض على جلسة بلازما"  →  "بلازما"
        "هل في عروض بخصوص الليزر"       →  "الليزر"
        "عروض تخصص العظام"               →  "العظام"
        "هل عندكم عروض؟"                 →  ""
    """
    cleaned = _OFFER_STOP_RE.sub(" ", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    hint = cleaned.lower()
    if not hint:
        return ""
    # Optionally validate hint using CRM offer vocabulary (requires crm_offers)
    try:
        from crm_offers import is_valid_service_hint  # type: ignore
        if not is_valid_service_hint(hint):
            return ""
    except ImportError:
        pass  # crm_offers not available — return hint as-is
    return hint


def search_offers(
    specialty_en: str,
    service_hint: str = "",
    patient_gender: str | None = None,
    patient_age: int | None = None,
) -> list[dict]:
    """Look up CRM offers for a specialty + optional service hint.

    Parameters
    ----------
    specialty_en : str
        Hospital-DB specialty name, e.g. "Dermatology and Cosmatology".
    service_hint : str
        Core service keyword from the patient's message (output of
        ``extract_service_hint()``). Pass "" for generic specialty lookup.
    patient_gender : str | None
        "male" or "female" — used to filter gender-incompatible offers.
    patient_age : int | None
        Patient age in years — used to deprioritise child-specific offers
        for adult patients.

    Returns
    -------
    list[dict]
        Ranked list of offer dicts from the CRM.  Each dict has:
            Offer_ID, Offer_Name_AR, Offer_Name_EN,
            Offer_Description_AR, Offer_Description_EN,
            Specialty, Offer_Status, Price_Before_Discount,
            Price_After_Discount, Discount, Start_Date, End_Date,
            Offer_Type, Offer_Tag, DotCare_Code, Created_On,
            _service_matched (bool), _is_alternative (bool)

    Notes
    -----
    Returns [] when:
    - CRM is unreachable or in back-off window
    - specialty_en is empty
    - No offers match the specialty in the CRM cache
    """
    if not specialty_en:
        return []
    # crm_offers.py must be in the same directory or importable
    from crm_offers import get_offers_for_specialty  # type: ignore
    return get_offers_for_specialty(
        specialty_en=specialty_en,
        service_hint=service_hint,
        patient_gender=patient_gender,
        patient_age=patient_age,
    )


def format_offer_response(offer: dict, lang: str = "ar") -> str:
    """Format a single CRM offer dict as a patient-facing card string.

    Requires ``crm_offers.format_offer_card`` to be importable.
    """
    from crm_offers import format_offer_card  # type: ignore
    card = format_offer_card(offer, lang)
    if lang == "ar":
        return (
            f"عندنا عرض خاص يناسب تخصصك! 🌟\n\n"
            f"{card}\n\n"
            "هل تريد الحجز بهذا العرض؟"
        )
    return (
        f"We have a special offer for your specialty! 🌟\n\n"
        f"{card}\n\n"
        "Would you like to book with this offer?"
    )
