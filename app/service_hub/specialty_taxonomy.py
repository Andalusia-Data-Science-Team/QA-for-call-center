"""specialty_taxonomy.py — Shared English/Arabic medical-specialty taxonomy
============================================================================
A single, authoritative EN<->AR specialty-name table, kept deliberately
SEPARATE from app.service_hub.offer_search._AR_ALIAS (the project's
existing colloquial-phrase-to-specialty alias table used for patient-facing
OFFER search). The two serve different jobs and are NOT duplicates of one
another:

  - offer_search._AR_ALIAS maps loose, colloquial CONVERSATIONAL phrases
    ("قلب", "عظام", "اطفال انابيب") to a coarse EN specialty bucket, for
    finding an offer to show a patient. It has no notion of exact specialty
    *names* as CRM stores them, and no pediatric/adult distinction at all.
  - This module maps the actual, FORMAL specialty/subspecialty NAMES (as
    they appear verbatim in CRM data or as an Agent might state them
    explicitly, in either language) to one canonical EN specialty per
    concept — including the pediatric subspecialties CRM data carries that
    offer_search has no reason to know about (a patient asking for an
    "offer" is never asking specifically for Pediatric Cardiology).

app.service_hub.doctor_validation imports this module for CRM specialty/
subspecialty claim validation; it also keeps using offer_search._AR_ALIAS
as its primary colloquial-phrase resolver (see doctor_validation's own
_resolve_specialty_category) — the two tables are merged there, not
duplicated here.

Public API
----------
    PEDIATRIC_SPECIALTIES_EN / PEDIATRIC_SPECIALTIES_AR
        The specialties that must always stay distinct from their adult
        equivalent (see canonicalize_specialty_name / is_pediatric_specialty).

    SPECIALTY_EN_TO_AR
        EN specialty display name -> its Arabic equivalent. Several EN
        names may share the same Arabic concept (e.g. "General Pediatrics"
        / "Pediatrics" / "Pediatric medicine" all mean "طب الأطفال") — see
        canonicalize_specialty_name for how those collapse onto one
        canonical EN bucket.

    SPECIALTY_TAXONOMY_ALIASES
        Every EN name and AR name in SPECIALTY_EN_TO_AR, each mapped to its
        canonical EN bucket — a ready-to-merge alias table for a caller
        (doctor_validation._resolve_specialty_category) that already does
        its own longest-alias-wins substring search over a different table.

    canonicalize_specialty_name(value) -> canonical EN name, or None
        EXACT (whole normalized string) lookup only — by design: this
        taxonomy exists specifically to keep e.g. "Pediatric Cardiology"
        and "Cardiology" as two distinct, unambiguous entries, which only
        holds if the lookup is exact rather than substring/fuzzy.

    is_pediatric_specialty(canonical_name) -> bool
        Whether a canonical EN name (as returned by
        canonicalize_specialty_name) is one of the pediatric-only
        specialties that must never collapse onto its adult equivalent.
"""
from __future__ import annotations

from app.services.text_helpers import normalize_arabic_text

# ─────────────────────────────────────────────────────────────────────────
# Pediatric specialties that must remain distinct from their adult
# equivalent (e.g. "Pediatric Cardiology" must never collapse onto
# "Cardiology") — see canonicalize_specialty_name / is_pediatric_specialty.
# ─────────────────────────────────────────────────────────────────────────
PEDIATRIC_SPECIALTIES_EN = [
    "General Pediatrics",
    "Pediatrics",
    "Pediatric medicine",
    "Pediatric Medicine",
    "Pediatric Cardiology",
    "Pediatric Endocrinology",
    "Pediatric Neurology",
    "Pediatrics Neurology",
    "Pediatric Orthopedics",
    "Pediatric surgery",
    "Pediatric Surgery",
    "Pedodontic",
    "Pediatric Dentistry",
    "Pediatric Allergy & Immunology",
    "Pediatric Nephrology",
    "Pediatric hematology",
    "Pediatric Hematology",
    "Pediatric Gastroenterology",
    "Pediatric Rheumatology",
    "Pediatric Pulmonology",
    "Pediatric Nutrition",
    "Pediatric neuropsychiatry",
    "NICU",
    "PICU",
    "Immunology",
]

PEDIATRIC_SPECIALTIES_AR = [
    "طب الأطفال",
    "طب الأطفال العام",
    "القلب والأوعية الدموية للأطفال",
    "أمراض الغدد الصماء للأطفال",
    "طب أعصاب الأطفال",
    "جراحة العظام للأطفال",
    "جراحة الأطفال",
    "طب أسنان الأطفال",
    "وحدة العناية المركزة لحديثي الولادة",
    "وحدة العناية المركزة للأطفال",
    "الحساسية والمناعة للأطفال",
    "المناعة",
    "أمراض الكلى للأطفال",
    "أمراض الدم للأطفال",
    "أمراض الجهاز الهضمي للأطفال",
    "أمراض الروماتيزم للأطفال",
    "أمراض الصدر والرئة للأطفال",
    "تغذية الأطفال",
    "الطب النفسي العصبي للأطفال",
]

# EN specialty display name -> Arabic equivalent. Multiple EN names may
# describe the SAME Arabic concept (e.g. General Pediatrics / Pediatrics /
# Pediatric medicine); see canonicalize_specialty_name for how those
# collapse onto one canonical bucket while every real CRM/claim spelling
# still resolves correctly.
SPECIALTY_EN_TO_AR: dict[str, str] = {
    "Allergy & Immunology": "الحساسية والمناعة",
    "Andrology": "أمراض الذكورة",
    "Anesthesiology": "التخدير",
    "Audiological medicine": "السمعيات",
    "Bariatric Surgery": "جراحات السمنة",
    "Breast Surgery": "جراحة الثدي",
    "Cardiac surgery": "جراحة القلب",
    "Cardiology": "القلب والأوعية الدموية",
    "Cardiothoracic Surgery": "جراحة القلب والصدر",
    "Chest": "الصدر",
    "Cosmetology": "التجميل",
    "Dental Services": "خدمات الأسنان",
    "Dermatology and Cosmatology": "الأمراض الجلدية",
    "E.N.T.": "أنف وأذن وحنجرة",
    "ENT Head & Neck Surgery": "جراحة الرأس والرقبة",
    "Emergency": "طوارئ",
    "Endocrinology": "الغدد الصماء",
    "Endodontic": "علاج جذور الأسنان",
    "Family Medicine": "طب الأسرة",
    "General Pediatrics": "طب الأطفال",
    "General Surgery": "الجراحة العامة",
    "General Surgery and Bariatric": "الجراحة العامة وجراحات السمنة",
    "Hematology": "أمراض الدم",
    "Hernia Surgery": "جراحة الفتق",
    "IVF": "علاج العقم وتأخر الحمل",
    "Internal Medicine": "الطب الباطني",
    "Interventional Cardiology": "أمراض القلب التدخلية",
    "Interventional Neurology": "علاج الأعصاب التداخلي",
    "Maxillofacial Surgery": "جراحة الوجه والفكين",
    "Medical Oncology": "الأورام الطبية",
    "Nephrology": "الكلى",
    "Neuro Surgery": "جراحة المخ والأعصاب",
    "Neurology": "المخ والأعصاب",
    "Neuropsychiatry": "طب النفس والأعصاب",
    "Nutrition": "تغذية",
    "OBE & GYN": "أمراض النساء والتوليد",
    "Oncology": "علم الأورام",
    "Ophthalmology": "طب العيون",
    "Oral Surgery": "جراحة الفم",
    "Orthodontic": "تقويم الأسنان",
    "Orthopedics": "جراحة العظام",
    "Otorhinolaryngology": "أنف وأذن وحنجرة",
    "Pediatric Allergy & Immunology": "الحساسية والمناعة للأطفال",
    "Pediatric Cardiology": "القلب والأوعية الدموية للأطفال",
    "Pediatric Endocrinology": "أمراض الغدد الصماء للأطفال",
    "Pediatric Gastroenterology": "أمراض الجهاز الهضمي للأطفال",
    "Pediatric hematology": "أمراض الدم للأطفال",
    "Pediatric medicine": "طب الأطفال",
    "Pediatric Nephrology": "أمراض الكلى للأطفال",
    "Pediatric Neurology": "طب أعصاب الأطفال",
    "Pediatric Orthopedics": "جراحة العظام للأطفال",
    "Pediatric Rheumatology": "أمراض الروماتيزم للأطفال",
    "Pediatric surgery": "جراحة الأطفال",
    "Pediatrics": "طب الأطفال",
    "Pedodontic": "طب أسنان الأطفال",
    "NICU": "وحدة العناية المركزة لحديثي الولادة",
    "Physiotherapy": "العلاج الطبيعي",
    "Plastic Surgery": "جراحة التجميل",
    "Prosthesis": "تركيبات الأسنان",
    "Psychiatry": "الطب النفسي",
    "Rheumatology": "أمراض الروماتيزم",
    "Spine Surgery": "جراحة العمود الفقري",
    "Surgical Oncology": "الأورام الجراحية",
    "Urology": "جراحة المسالك البولية",
    "Urology and Andrology": "المسالك البولية والذكورة",
    "Vascular Surgery": "جراحة الأوعية الدموية",
}

# Additional real CRM specialty/subspecialty display-name values observed in
# production (verified against a live CRM export this session) — extends
# rather than replaces the table above. Deliberately KEEPS every differently
# -cased variant of an already-known name as its own separate key when both
# spellings genuinely occur in source data (e.g. "Pediatric surgery" from
# the base table above vs "Pediatric Surgery" here) — collapsing them onto
# one key would silently lose which exact CRM spelling was actually seen,
# and this taxonomy's exact-match lookup (canonicalize_specialty_name) is
# case-insensitive after normalisation anyway, so keeping both costs
# nothing. Known CRM misspellings ("Anastasia" for Anesthesia,
# "Neurospsychiatry"/"Neurospsychiry" for Neuropsychiatry, "Pain Managment"
# for Pain Management, "Summar" for Summer) are intentionally preserved
# verbatim rather than corrected — they are the literal values CRM stores,
# and normalising them away here would make evidence/logs built from the
# raw CRM value unrecognisable against the source record.
SPECIALTY_EN_TO_AR.update({
    # Allergy / Anesthesia
    "Allergy and Immunology": "الحساسية والمناعة",
    "Anastasia": "التخدير",
    "Anesthesia": "التخدير",

    # Programs
    "Art Drawing Skills Program": "برنامج مهارات الرسم الفني",
    "Behavior Modification (ABA)": "تعديل السلوك (ABA)",

    # Cardiology
    "Cardiac procedure": "إجراءات القلب",
    "Cardiology Screening": "فحص القلب",
    "Cardiothoracic": "القلب والصدر",
    "Cardiothoracic surgery": "جراحة القلب والصدر",
    "CTO Advanced coronary intervention":
        "التدخل المتقدم للشرايين التاجية للانسداد الكلي المزمن (CTO)",
    "interventional cardiology": "أمراض القلب التدخلية",

    # Oncology / Pathology
    "Clinical oncology": "الأورام الإكلينيكية",
    "Clinical Pathology": "الباثولوجيا الإكلينيكية",

    # Conservative
    "Conservative": "العلاج التحفظي",

    # Dental
    "Dental Screening": "فحص الأسنان",
    "Dentistry": "طب الأسنان",
    "Endodontic and cosmetic dentistry":
        "علاج جذور الأسنان وتجميل الأسنان",
    "Endodontics": "علاج جذور الأسنان",
    "Pediatric Dentistry": "طب أسنان الأطفال",
    "Periodontics": "أمراض وعلاج اللثة",
    "Prosthodontics": "تركيبات الأسنان",

    # Dermatology
    "cosmatology": "التجميل",
    "Derma-hair removal": "إزالة الشعر - جلدية",
    "Dermatology": "الأمراض الجلدية",
    "Dermatology, Cosmatology and Andrology":
        "الأمراض الجلدية والتجميل والذكورة",

    # ENT
    "E.N.T": "أنف وأذن وحنجرة",
    "E.N.T surgery": "جراحة الأنف والأذن والحنجرة",
    "Facioplastic and Reconstructive Surgery":
        "جراحة تجميل وترميم الوجه",
    "Phonatics": "الصوتيات والتخاطب",

    # Early intervention / Emergency
    "Early Intervention": "التدخل المبكر",
    "Emergency Medicine": "طب الطوارئ",
    "Emotional Intelligence Program": "برنامج الذكاء العاطفي",

    # Gastro / Liver
    "Gastroenterology": "أمراض الجهاز الهضمي",
    "Hepatology": "أمراض الكبد",

    # Surgery
    "Colorectal Surgery": "جراحة القولون والمستقيم",
    "head and neck surgery": "جراحة الرأس والرقبة",
    "Hepatobiliary Surgery": "جراحة الكبد والقنوات المرارية",
    "Pediatric Surgery": "جراحة الأطفال",
    "Surgical specialty": "تخصص جراحي",

    # ICU
    "I.C.U": "العناية المركزة",
    "ICU": "العناية المركزة",

    # Internal Medicine
    "Clinical hematology": "أمراض الدم الإكلينيكية",
    "Diabetic medicine": "طب السكري",
    "General Checkup": "فحص عام",
    "Geriatric medicine": "طب كبار السن",
    "Home Care": "الرعاية المنزلية",
    "Infectious Disease": "الأمراض المعدية",
    "Laboratory": "المختبر",
    "Sleep Medicine": "طب النوم",

    # Radiology
    "Intervention Radiology": "الأشعة التداخلية",
    "Interventional radiology": "الأشعة التداخلية",
    "Interventional Radiology": "الأشعة التداخلية",

    # Kids programs
    "Kids PHYSICAL STRENGTH - YOGA":
        "القوة البدنية للأطفال - يوجا",
    "Kids PHYSICAL STRENGTH - Zumba":
        "القوة البدنية للأطفال - زومبا",

    # Learning
    "Learning Difficulties": "صعوبات التعلم",

    # Maxillofacial
    "Maxillofacial surgery": "جراحة الوجه والفكين",

    # Neurology
    "E.E.G": "رسم المخ",
    "N.C.S": "دراسة توصيل الأعصاب",
    "Neurospsychiatry": "طب النفس والأعصاب",
    "Neurospsychiry": "طب النفس والأعصاب",
    "Neurosurgery": "جراحة المخ والأعصاب",

    # Nutrition
    "Pediatric Nutrition": "تغذية الأطفال",

    # Obstetrics / Gynecology
    "Gynecological Oncology": "أورام النساء",
    "Maternal Fetal Medicine": "طب الأم والجنين",
    "Obestetrics and Gynecology": "أمراض النساء والتوليد",
    "Obstetrics and gynecology": "أمراض النساء والتوليد",

    # Occupational Therapy
    "Occupational therapy (OT)": "العلاج الوظيفي",

    # Oncology
    "Radiation Oncology": "العلاج الإشعاعي للأورام",

    # Orthopedics
    "Arthroplasty": "جراحة استبدال المفاصل",
    "Arthroplasty & orthopedic Oncology":
        "استبدال المفاصل وأورام العظام",
    "Foot & Ankle": "القدم والكاحل",
    "Hand": "جراحة اليد",
    "Knee": "الركبة",
    "Shoulder": "الكتف",
    "Spine": "العمود الفقري",
    "Trauma & orthopedics": "إصابات وجراحة العظام",
    "Trauma & Orthopedics-Screening":
        "فحص إصابات وجراحة العظام",

    # Pain
    "Pain Management": "علاج الألم",
    "Pain management (specialty)": "تخصص علاج الألم",
    "Pain Managment": "علاج الألم",

    # Pediatrics
    "Pediatric Hematology": "أمراض الدم للأطفال",
    "Pediatric Medicine": "طب الأطفال",
    "Pediatric neuropsychiatry": "الطب النفسي العصبي للأطفال",
    "Pediatrics Neurology": "طب أعصاب الأطفال",
    "Pediatric Pulmonology": "أمراض الصدر والرئة للأطفال",
    "Immunology": "المناعة",
    "PICU": "وحدة العناية المركزة للأطفال",

    # Therapy
    "Play Therapy": "العلاج باللعب",

    # Preventive Medicine
    "Preventive medicine": "الطب الوقائي",

    # Procedure
    "Procedure": "إجراءات طبية",
    "Procedure Room": "غرفة الإجراءات",

    # Psychology
    "Psychology": "علم النفس",

    # Pulmonary
    "Pulmonary medicine": "طب الأمراض الصدرية والرئوية",

    # Radiology
    "Breast Imaging": "تصوير الثدي",
    "Radiology": "الأشعة",

    # Screening
    "Screening": "الفحص",

    # Speech
    "Speech and Phonetics": "التخاطب والصوتيات",
    "ABA Therapist": "أخصائي تعديل السلوك (ABA)",

    # Summer Club
    "Summar Club 1 Month": "النادي الصيفي - شهر",
    "Summar Club 1 Visit": "النادي الصيفي - زيارة واحدة",
    "Summar Club 1 week": "النادي الصيفي - أسبوع",
    "Summar Club 6 week": "النادي الصيفي - 6 أسابيع",
})


def _build_canonical_groups(en_to_ar: dict[str, str]) -> dict[str, str]:
    """EN specialty name -> canonical EN name: the FIRST EN name
    encountered (in en_to_ar's own insertion/declaration order) for a given
    Arabic concept. Lets multiple EN synonyms for the same Arabic concept
    (General Pediatrics / Pediatric medicine / Pediatrics, all "طب
    الأطفال") collapse onto one shared bucket, while every caller — CRM
    field canonicalisation included — still gets back a real, displayable
    specialty name (never an opaque id). Deterministic and declaration-
    order-dependent BY DESIGN: SPECIALTY_EN_TO_AR lists "General
    Pediatrics" before its synonyms specifically so that (pre-existing)
    canonical name stays the one every synonym collapses onto — the same
    name app.service_hub.offer_search._AR_ALIAS already resolves "اطفال"
    to, so the two tables agree."""
    ar_to_canonical_en: dict[str, str] = {}
    en_to_canonical: dict[str, str] = {}
    for en, ar in en_to_ar.items():
        canonical_en = ar_to_canonical_en.setdefault(ar, en)
        en_to_canonical[en] = canonical_en
    return en_to_canonical


_EN_TO_CANONICAL = _build_canonical_groups(SPECIALTY_EN_TO_AR)

# normalized EN name -> canonical EN name / normalized AR name -> canonical
# EN name. EXACT-lookup structures only (see canonicalize_specialty_name) —
# built once at import time, never duplicated per-call.
NORMALIZED_EN_TO_CANONICAL: dict[str, str] = {
    normalize_arabic_text(en): canonical for en, canonical in _EN_TO_CANONICAL.items()
}
NORMALIZED_AR_TO_CANONICAL: dict[str, str] = {
    normalize_arabic_text(ar): _EN_TO_CANONICAL[en] for en, ar in SPECIALTY_EN_TO_AR.items()
}

# Ready-to-merge alias table (display name, EN or AR -> canonical EN name)
# for a caller that does its own longest-alias-wins substring search over a
# DIFFERENT table (doctor_validation._resolve_specialty_category merges
# this with offer_search._AR_ALIAS) — never iterate SPECIALTY_EN_TO_AR
# directly for that purpose, since its values are Arabic TRANSLATIONS, not
# canonical buckets (see _build_canonical_groups above for why those
# differ for a synonym group).
SPECIALTY_TAXONOMY_ALIASES: dict[str, str] = {
    **{en: canonical for en, canonical in _EN_TO_CANONICAL.items()},
    **{ar: _EN_TO_CANONICAL[en] for en, ar in SPECIALTY_EN_TO_AR.items()},
}

# Canonical EN names that must never collapse onto a non-pediatric
# equivalent — every PEDIATRIC_SPECIALTIES_EN entry is itself a
# SPECIALTY_EN_TO_AR key, so this is always non-empty for every entry.
PEDIATRIC_CANONICAL_NAMES: frozenset[str] = frozenset(
    _EN_TO_CANONICAL[name] for name in PEDIATRIC_SPECIALTIES_EN if name in _EN_TO_CANONICAL
)


def canonicalize_specialty_name(value: str | None) -> str | None:
    """EXACT (whole normalized string) specialty-name lookup -> canonical
    EN specialty name, or None when *value* is not, in its entirety, one
    of this taxonomy's known EN or AR names (in either language, case-
    insensitive, Arabic-normalised). Deliberately EXACT rather than
    substring/fuzzy: this taxonomy's whole purpose is to keep closely
    related specialties (e.g. "Pediatric Cardiology" vs "Cardiology",
    "Oncology" vs "Medical Oncology") as distinct, unambiguous entries —
    which only holds if a shorter name is never treated as "contained in"
    a longer, different one here. Substring/phrase-search resolution (safe
    for free CONVERSATIONAL text, where a longer/more-specific alias is
    allowed to out-rank a shorter one) is the caller's own job — see
    doctor_validation._resolve_specialty_category, which merges
    SPECIALTY_TAXONOMY_ALIASES into its OWN longest-alias-wins search
    rather than duplicating that algorithm here."""
    if not value:
        return None
    norm = normalize_arabic_text(value)
    if not norm:
        return None
    return NORMALIZED_EN_TO_CANONICAL.get(norm) or NORMALIZED_AR_TO_CANONICAL.get(norm)


def is_pediatric_specialty(canonical_name: str | None) -> bool:
    """Whether *canonical_name* (already resolved via
    canonicalize_specialty_name) is one of the pediatric-only specialties
    that must never be treated as equivalent to its adult counterpart."""
    return bool(canonical_name) and canonical_name in PEDIATRIC_CANONICAL_NAMES
