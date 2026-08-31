"""
CRM Services Data Layer — service_hub/crm_services.py

Fetches active service records from [dbo].[cr301_ksaservicedataset]
(filtered to active BUs) via SQL Server (Passcode.json connection).

Public API:
    fetch_services(bu="LIVE", force_refresh=False) → list[dict]
    get_services_for_specialty(specialty_en, service_hint, ...)  → list[dict]
    format_service_card(service, lang)             → str

Column aliases (from service_query.sql):
    createdon, modifiedon, cr301_title, cr301_code,
    cr301_nphiescode, cr301_nphiesdescription, servhub_priced,
    cr301_productcategory, cr301_service, cr301_servicear,
    cr301_servicekey, cr301_specialty, cr301_status,
    cr301_businessunitname, cr301_servicecategoryname,
    cr301_specialtyname, servhub_priced, cr18c_buname
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz as _rfuzz

# ── SQL query (loaded from file) ──────────────────────────────────────────────
_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "service_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

# ── Cache: keyed by BU ────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 30 * 60   # 30 min — services change slowly

# bu → {"rows": [], "loaded_at": 0.0, "failed": False}
_cache: dict[str, dict] = {}
_lock = threading.Lock()

# BU values that map to a single cr18c_buname code
_BU_MAP: dict[str, str] = {
    "AHJ": "AHJ", "MKR": "MKR", "LCH": "LCH",
    "LIVE": "AHJ",  # LIVE is the main AHJ branch
    "SNB": "SNB", "ALW": "ALW", "AKW": "AKW",
    "HJH": "HJH", "AFW": "AFW", "ADC": "ADC",
}


def _resolve_bu(bu: str | None) -> str:
    """Normalise a business_unit value to the cr18c_buname code used in the DB."""
    return _BU_MAP.get((bu or "").strip().upper(), "AHJ")

# ── Arabic medical terms → English equivalents (for service name matching) ───
_ARABIC_MEDICAL_TERMS: dict[str, list[str]] = {
    # Ultrasound variations
    "موجات فوق الصوتية": ["ultrasound", "u/s", "us", "echo", "sono"],
    "موجات صوتية": ["ultrasound", "u/s", "us"],
    "سونار": ["ultrasound", "u/s", "us", "sono"],
    "إيكو": ["echo", "echocardiography", "u/s"],
    "ايكو": ["echo", "echocardiography", "u/s"],
    "دوبلر": ["doppler"],
    
    # Dimensions
    "ثنائي الأبعاد": ["2d"],
    "ثلاثي الأبعاد": ["3d"],
    "رباعي الأبعاد": ["4d"],
    "رباعية الأبعاد": ["4d"],
    
    # Radiology
    "أشعة مقطعية": ["ct", "ct scan", "computed tomography"],
    "اشعة مقطعية": ["ct", "ct scan"],
    "أشعة": ["x-ray", "xray", "radiograph"],
    "اشعة": ["x-ray", "xray"],
    "رنين مغناطيسي": ["mri", "magnetic resonance"],
    "رنين": ["mri"],
    
    # Lab tests
    "تحليل دم": ["blood test", "cbc", "complete blood count"],
    "صورة دم": ["cbc", "complete blood count", "blood count"],
    "سكر": ["glucose", "sugar", "hba1c", "diabetes"],
    "سكر تراكمي": ["hba1c", "a1c", "glycated hemoglobin"],
    "كوليسترول": ["cholesterol", "lipid"],
    "دهون": ["lipid", "cholesterol", "triglyceride"],
    "وظائف كلى": ["kidney function", "renal", "creatinine", "urea"],
    "وظائف كبد": ["liver function", "lft", "alt", "ast"],
    "غدة درقية": ["thyroid", "tsh", "t3", "t4"],
    "فيتامين د": ["vitamin d", "vit d", "25-oh"],
    
    # Scans
    "منظار": ["endoscopy", "scope"],
    "قسطرة": ["catheter", "catheterization"],
}

# ── Specialty mapping (hospital DB name → cr301_specialtyname in CRM) ─────────
_SPECIALTY_MAP: dict[str, list[str]] = {
    # English names
    "dermatology":                 ["Dermatology", "Dermatology and Cosmatology"],
    "dermatology and cosmatology": ["Dermatology", "Dermatology and Cosmatology"],
    "dermatology and cosmetology": ["Dermatology", "Dermatology and Cosmatology"],
    "dental":                      ["Dental Services", "Dental"],
    "dental services":             ["Dental Services", "Dental"],
    "pedodontic":                  ["Dental Services", "Dental"],
    "pedodontics":                 ["Dental Services", "Dental"],
    "obstetrics":                  ["OBE & GYN", "Obstetrics & Gynecology"],
    "gynecology":                  ["OBE & GYN", "Obstetrics & Gynecology"],
    "obe & gyn":                   ["OBE & GYN"],
    "obstetrics & gynecology":     ["OBE & GYN"],
    "ivf":                         ["IVF", "OBE & GYN"],
    "internal medicine":           ["Internal Medicine"],
    "pediatrics":                  ["Pediatrics", "General Pediatrics"],
    "general pediatrics":          ["Pediatrics", "General Pediatrics"],
    "nutrition":                   ["Nutrition"],
    "ent":                         ["E.N.T", "ENT"],
    "e.n.t.":                      ["E.N.T"],
    "e.n.t":                       ["E.N.T"],
    "physiotherapy":               ["Physiotherapy"],
    "physical therapy":            ["Physiotherapy"],
    "andrology":                   ["Andrology"],
    "general surgery":             ["General Surgery"],
    "orthopedics":                 ["Orthopedics"],
    "orthopaedics":                ["Orthopedics"],
    "psychology":                  ["Psychology"],
    "psychiatry":                  ["Psychiatry", "Psychiatry & Mental Health"],
    "speech therapy":              ["Speech and Phonetics"],
    "family medicine":             ["Family Medicine"],
    "urology":                     ["Urology"],
    "endocrinology":               ["Endocrinology"],
    "chest":                       ["Chest"],
    "pulmonology":                 ["Chest"],
    "ophthalmology":               ["Ophthalmology"],
    "cardiology":                  ["Cardiology"],
    "neurology":                   ["Neurology"],
    "rheumatology":                ["Rheumatology", "Internal Medicine"],
    "nephrology":                  ["Nephrology", "Urology"],
    "gastroenterology":            ["GIT", "Gastroenterology"],
    "git":                         ["GIT"],
    "bariatric surgery":           ["Bariatric Surgery"],
    "laboratory":                  ["Laboratory"],
    "radiology":                   ["Radiology"],
    
    # Arabic specialty names → English CRM equivalents
    "جلدية":                      ["Dermatology", "Dermatology and Cosmatology"],
    "أسنان":                      ["Dental Services", "Dental"],
    "اسنان":                      ["Dental Services", "Dental"],
    "نساء":                       ["OBE & GYN", "Obstetrics & Gynecology"],
    "نسا":                        ["OBE & GYN", "Obstetrics & Gynecology"],
    "عيادة النسا":                ["OBE & GYN", "Obstetrics & Gynecology"],
    "عيادة النساء":               ["OBE & GYN", "Obstetrics & Gynecology"],
    "نساء وتوليد":                ["OBE & GYN", "Obstetrics & Gynecology"],
    "ولادة":                      ["OBE & GYN", "Obstetrics & Gynecology"],
    "أطفال":                      ["Pediatrics", "General Pediatrics"],
    "اطفال":                      ["Pediatrics", "General Pediatrics"],
    "باطنة":                      ["Internal Medicine"],
    "باطني":                      ["Internal Medicine"],
    "تغذية":                      ["Nutrition"],
    "أنف وأذن وحنجرة":            ["E.N.T", "ENT"],
    "انف واذن وحنجرة":            ["E.N.T", "ENT"],
    "أذن":                        ["E.N.T", "ENT"],
    "علاج طبيعي":                 ["Physiotherapy"],
    "طبيعي":                      ["Physiotherapy"],
    "عظام":                       ["Orthopedics"],
    "جراحة":                      ["General Surgery"],
    "جراحة عامة":                 ["General Surgery"],
    "قلب":                        ["Cardiology"],
    "قلبية":                      ["Cardiology"],
    "أعصاب":                      ["Neurology"],
    "اعصاب":                      ["Neurology"],
    "عيون":                       ["Ophthalmology"],
    "رمد":                        ["Ophthalmology"],
    "مسالك":                      ["Urology"],
    "مسالك بولية":                ["Urology"],
    "صدر":                        ["Chest"],
    "صدرية":                      ["Chest"],
    "تحاليل":                     ["Laboratory"],
    "معمل":                       ["Laboratory"],
    "مختبر":                      ["Laboratory"],
    "أشعة":                       ["Radiology"],
    "اشعة":                       ["Radiology"],
}


def _crm_specialty_names(specialty_en: str) -> list[str]:
    """Convert specialty name (Arabic or English) to CRM specialty name(s).
    
    Handles:
    - Direct lookup in _SPECIALTY_MAP
    - Partial matching (substring search)
    - Arabic specialty names → English CRM names
    - Laboratory variations (تحاليل، معمل، مختبر) → "Laboratory"
    """
    key = (specialty_en or "").strip().lower()
    
    # Direct lookup
    mapped = _SPECIALTY_MAP.get(key)
    if mapped:
        print(f"[services] _crm_specialty_names: {specialty_en!r} → {mapped} (direct)", flush=True)
        return mapped
    
    # Partial matching (substring search)
    for map_key, names in _SPECIALTY_MAP.items():
        if map_key in key or key in map_key:
            print(f"[services] _crm_specialty_names: {specialty_en!r} → {names} (partial match via {map_key!r})", flush=True)
            return names
    
    # Fallback: return the original name
    print(f"[services] _crm_specialty_names: {specialty_en!r} → [{specialty_en.strip()}] (no match, using original)", flush=True)
    return [specialty_en.strip()]


# ── Fetch all services ────────────────────────────────────────────────────────

def fetch_services(bu: str | None = None, force_refresh: bool = False) -> list[dict]:
    """
    Return all active service records for the given BU from the CRM.
    Uses the same crm_database connection as crm_offers (Azure AD / MSAL).
    Cached per BU for _CACHE_TTL_SECONDS. Never raises — returns [] on failure.
    Thread-safe.
    """
    from app.service_hub.crm_database import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    bu_code = _resolve_bu(bu)

    with _lock:
        entry = _cache.setdefault(bu_code, {"rows": [], "loaded_at": 0.0, "failed": False})
        now = time.time()
        age = now - entry["loaded_at"]

        if not force_refresh and entry["rows"] and age < _CACHE_TTL_SECONDS:
            return entry["rows"]

        if entry["failed"] and age < 300:
            print(f"[services] {bu_code}: skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return entry["rows"]

        print(f"[services] {bu_code}: fetching from CRM…", flush=True)
        t0 = time.time()
        try:
            # Fetch all services for BU - filtering by service name happens in Python for better fuzzy matching
            # Pass bu_code as named parameter (:bu_name) in SQL query
            rows = _run_query_with_retry(_QUERY, max_attempts=3, params={"bu_name": bu_code})
            print(f"[services] {bu_code}: fetched {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)
            entry["rows"] = rows
            entry["loaded_at"] = now
            entry["failed"] = False
            return rows
        except Exception as e:
            print(f"[services] {bu_code}: fetch failed after {time.time()-t0:.1f}s: {e}", flush=True)
            entry["failed"] = True
            entry["loaded_at"] = now
            return entry["rows"]


# ── Core search ───────────────────────────────────────────────────────────────

def get_services_for_specialty(
    specialty_en: str,
    service_hint: str = "",
    bu: str | None = None,
    patient_gender: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Return ranked service records for specialty_en + optional hint.

    Strategy (mirrors crm_offers.get_offers_for_specialty):
      1. Direct name match  — if hint ≥ 4 chars, fuzzy-match cr301_service /
                              cr301_servicear against hint (partial_ratio ≥ 85).
                              Returns immediately on a confident match.
      2. Specialty filter   — keep rows whose cr301_specialtyname matches the
                              hospital-DB specialty (via _SPECIALTY_MAP).
      3. Hint scoring       — score each row by how many hint words appear in
                              cr301_service, cr301_servicear, cr301_title.
      4. Fallback           — return cheapest active rows for the specialty.

    Returns list[dict] — each dict is a raw DB row, augmented with:
        _service_matched  (bool)
        _is_alternative   (bool)
    """
    bu_code = _resolve_bu(bu)
    all_rows = fetch_services(bu=bu)
    if not all_rows:
        print(f"[services] get_services_for_specialty({specialty_en!r}) bu={bu_code}: cache empty", flush=True)
        return []

    # ── 1. Direct name fast-path (enhanced normalization) ─────────────────
    if service_hint and len(service_hint.strip()) >= 4:
        _hint_norm = service_hint.strip().lower()
        
        # ── Enhanced normalization for extracted service/package names ────────
        def _normalize_service_hint(hint: str) -> str:
            """Normalize extracted service/package names for better matching."""
            import re as _re
            # First, remove newlines and normalize whitespace
            text = _re.sub(r'[\r\n]+', ' ', hint)
            text = _re.sub(r'\s+', ' ', text).strip()
            
            # Remove trailing prices: numbers with SAR/ريال/SR or standalone numbers at end
            text = _re.sub(r'\s+\d+[\s\.]*(ريال|SAR|SR|جنيه|دينار|درهم)\s*$', '', text, flags=_re.IGNORECASE)
            text = _re.sub(r'\s+\d{2,}[\s\.]*$', '', text)  # "... 750" → "..."
            
            # Remove branch name patterns at the beginning
            text = _re.sub(r'^فرع\s+\S+\s+', '', text)  # "فرع السنابل ..." → "..."
            text = _re.sub(r'^(في|فى)\s+فرع\s+\S+\s+', '', text)  # "في فرع X ..." → "..."
            
            return text.strip().lower()
        
        def _extract_service_keywords(hint: str) -> tuple[list[str], bool]:
            """Extract English words and context around Arabic service keywords from hint.
            
            Also translates Arabic medical terms to English equivalents for better matching.
            
            Returns:
                (keywords_list, has_service_intent)
                - keywords_list: English words + translated medical terms + context
                - has_service_intent: True if Arabic service trigger words found
            """
            import re as _re
            keywords = []
            has_service_intent = False
            
            # Extract English words (2+ consecutive Latin chars) from Arabic context
            # This catches service names like "CBC", "CT Scan", "MRI", "X-Ray"
            en_words = _re.findall(r'[a-zA-Z][a-zA-Z0-9\-\.]*[a-zA-Z]|[a-zA-Z]{2,}', hint)
            keywords.extend(w.lower() for w in en_words if len(w) >= 2)
            
            # Translate Arabic medical terms to English equivalents
            hint_lower = hint.lower()
            for arabic_term, english_terms in _ARABIC_MEDICAL_TERMS.items():
                if arabic_term in hint_lower:
                    keywords.extend(english_terms)
                    has_service_intent = True
            
            # Arabic service indicator keywords (INTENT TRIGGERS ONLY)
            ar_triggers = [
                'فحص', 'فحوص', 'فحوصات',  # tests/examinations
                'تحليل', 'تحاليل',         # analysis/lab tests  
                'أشعة', 'اشعة', 'الأشعة', 'الاشعة',  # radiology/imaging
                'تصوير',                   # imaging
                'صورة', 'صور',             # image/scan
                'سونار',                   # ultrasound
                'رنين',                    # MRI (resonance)
                'مقطعية',                  # CT (tomography)
                'إيكو', 'ايكو',            # echo
                'دوبلر',                   # doppler
            ]
            
            # Check if hint contains any Arabic service triggers
            for trigger in ar_triggers:
                if trigger in hint_lower:
                    has_service_intent = True
                    # Extract 2-4 words AFTER the trigger (the actual service name)
                    # Pattern: trigger + next 2-4 Arabic/English words
                    pattern = trigger + r'\s+((?:[\w\-\.]+\s*){1,4})'
                    matches = _re.findall(pattern, hint_lower)
                    for match in matches:
                        # Clean and split the captured context
                        context_words = match.strip().split()
                        keywords.extend(w for w in context_words if len(w) > 2)
            
            return keywords, has_service_intent
        
        _hint_clean = _normalize_service_hint(_hint_norm)
        _extracted_keywords, _has_service_intent = _extract_service_keywords(service_hint)
        
        # If we have service intent triggers but no concrete keywords extracted,
        # use the full cleaned hint as fallback
        if _has_service_intent and not _extracted_keywords:
            _extracted_keywords = [w for w in _hint_clean.split() if len(w) > 2]
        
        # Apply Arabic normalization (hamza, tashkeel, etc.)
        def _norm_ar_fast(text: str) -> str:
            """Quick Arabic normalization for matching."""
            import re as _re
            t = text.lower()
            # Remove tashkeel
            t = _re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]', '', t)
            # Normalize hamza
            t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
            t = t.replace("ؤ", "و").replace("ئ", "ي")
            t = t.replace("ة", "ه").replace("ى", "ي")
            return t
        
        _hint_normalized = _norm_ar_fast(_hint_clean)
        
        # Detect if hint is primarily Arabic or English
        import re as _re
        arabic_chars = len(_re.findall(r'[\u0600-\u06FF]', service_hint))
        english_chars = len(_re.findall(r'[a-zA-Z]', service_hint))
        is_arabic_input = arabic_chars > english_chars
        
        _candidates: list[tuple[float, dict]] = []
        for row in all_rows:
            svc_en = (row.get("cr301_service") or "").strip().lower()
            svc_ar = (row.get("cr301_servicear") or "").strip().lower()
            title  = (row.get("cr301_title") or "").strip().lower()
            
            # Normalize database names too
            svc_ar_norm = _norm_ar_fast(svc_ar)
            title_norm = _norm_ar_fast(title)
            
            # Check for exact/near-exact match in cr301_title (for codes like "U/S 4D")
            _exact_title_boost = 0
            
            # Exact match: title equals hint (ignoring case, spaces, separators)
            _title_normalized = title.replace('/', '').replace(' ', '').replace('-', '').replace('_', '').lower()
            _hint_normalized_clean = _hint_clean.replace('/', '').replace(' ', '').replace('-', '').replace('_', '')
            
            if _title_normalized == _hint_normalized_clean:
                # Perfect exact match: "anomaly scan" == "ANOMALY SCAN"
                _exact_title_boost = 1000  # Massive boost for perfect match
            elif _hint_clean in title or title in _hint_clean:
                # Substring match: hint is part of title or vice versa
                _exact_title_boost = 50  # Strong boost for exact substring match
            elif _title_normalized in _hint_normalized_clean or _hint_normalized_clean in _title_normalized:
                # Normalized substring match without separators
                _exact_title_boost = 40  # Boost for match without separators
            
            # Penalize titles with extra qualifiers (–, Twins, Triplets, etc.)
            # when hint doesn't contain those qualifiers
            if any(q in title.lower() for q in ['–', 'twins', 'triplets', 'quadruplets']) and \
               not any(q in _hint_clean for q in ['–', 'twins', 'triplets', 'quadruplets']):
                _exact_title_boost = max(0, _exact_title_boost - 500)  # Heavy penalty for unwanted qualifiers
            
            # Strategy: prioritize the right field based on input language
            if is_arabic_input:
                # Arabic input → prioritize cr301_servicear field
                _score1 = _rfuzz.partial_ratio(_hint_normalized, svc_ar_norm)  # Primary: Arabic field
                _score2 = _rfuzz.token_set_ratio(_hint_normalized, svc_ar_norm)
                _score3 = _rfuzz.partial_ratio(_hint_clean, svc_ar)
                # Secondary: title (may contain transliterated terms)
                _score4 = _rfuzz.partial_ratio(_hint_normalized, title_norm)
                _score5 = _rfuzz.partial_ratio(_hint_clean, title)
                # Tertiary: English field (fallback)
                _score6 = _rfuzz.partial_ratio(_hint_clean, svc_en)
                _score7 = 0  # unused
            else:
                # English/Lab/Abbreviation input → prioritize cr301_title field
                _score1 = _rfuzz.partial_ratio(_hint_clean, title)  # Primary: title (abbreviations)
                _score2 = _rfuzz.token_set_ratio(_hint_clean, title)
                _score3 = _rfuzz.ratio(_hint_clean, title)  # Exact ratio for codes
                # Secondary: English service name
                _score4 = _rfuzz.partial_ratio(_hint_clean, svc_en)
                _score5 = _rfuzz.token_set_ratio(_hint_clean, svc_en)
                # Tertiary: Arabic field (fallback)
                _score6 = _rfuzz.partial_ratio(_hint_normalized, svc_ar_norm)
                _score7 = 0  # unused
            
            # 4. Keyword matching boost (prioritize by input language)
            _keyword_score = 0
            if _extracted_keywords:
                for kw in _extracted_keywords:
                    if is_arabic_input:
                        # Arabic input: check Arabic field first, then title, then English
                        if kw in svc_ar or kw in svc_ar_norm:
                            _keyword_score += 40  # Strong boost for Arabic match
                        elif kw in title or kw in title_norm:
                            _keyword_score += 30  # Medium boost for title match
                        elif kw in svc_en:
                            _keyword_score += 20  # Weak boost for English match
                    else:
                        # English/Lab input: check title first, then English, then Arabic
                        if kw in title or kw in title_norm:
                            _keyword_score += 40  # Strong boost for title match (abbreviations)
                        elif kw in svc_en:
                            _keyword_score += 30  # Medium boost for English match
                        elif kw in svc_ar or kw in svc_ar_norm:
                            _keyword_score += 20  # Weak boost for Arabic match
            
            # Take the best score from all strategies and add boosts
            score = max(_score1, _score2, _score3, _score4, _score5, _score6, _score7) + _keyword_score + _exact_title_boost
            
            # Lower threshold for normalized matching (70 for better coverage)
            # Exact title matches will score very high due to boost
            if score >= 70:
                _candidates.append((score, row))
        
        if _candidates:
            _candidates.sort(key=lambda x: -x[0])
            best_score, best_row = _candidates[0]
            _keywords_str = f" | keywords={_extracted_keywords}" if _extracted_keywords else ""
            print(
                f"[services] direct name match: hint={service_hint!r} → cleaned={_hint_clean!r}{_keywords_str} → "
                f"{(best_row.get('cr301_service') or best_row.get('cr301_servicear'))!r} "
                f"(score={best_score})",
                flush=True,
            )
            
            # Log all high-scoring candidates to show related services
            if len(_candidates) > 1:
                print(f"[services] Found {len(_candidates)} related services:", flush=True)
                for idx, (_sc, _r) in enumerate(_candidates[:5], 1):
                    print(
                        f"[services]   #{idx}: score={_sc} | "
                        f"title={_r.get('cr301_title')!r} | "
                        f"service={(_r.get('cr301_servicear') or _r.get('cr301_service'))!r} | "
                        f"specialty={_r.get('cr301_specialtyname')!r} | "
                        f"Price={_r.get('servhub_priced')}",
                        flush=True,
                    )
            
            result = dict(best_row)
            result["_service_matched"] = True
            result["_is_alternative"] = False
            print(
                f"[services] direct name match RETURNED:\n"
                f"  cr301_title      : {result.get('cr301_title')}\n"
                f"  cr301_service    : {result.get('cr301_service')}\n"
                f"  cr301_servicear  : {result.get('cr301_servicear')}\n"
                f"  cr301_specialtyname: {result.get('cr301_specialtyname')}\n"
                f"  cr301_price      : {result.get('servhub_priced')}\n"
                f"  cr18c_buname     : {result.get('cr18c_buname')}\n"
                f"  match_score      : {best_score}",
                flush=True,
            )
            return [result]

    # ── 2. Specialty filter ───────────────────────────────────────────────
    crm_names = _crm_specialty_names(specialty_en)
    crm_names_lower = [n.lower() for n in crm_names]
    
    # Special handling for Laboratory services OR generic/unknown specialties
    # If specialty is "Laboratory" or similar, search across ALL services
    # because lab tests span multiple specialties.
    # If specialty is "General" or empty (from service-only queries), also
    # search across all services and rely on service_hint matching.
    is_laboratory = any("laboratory" in n.lower() or "lab" in n.lower() for n in crm_names)
    is_generic = specialty_en.lower() in ("general", "", "unknown")
    
    if is_laboratory or is_generic:
        # For laboratory or generic specialty, search in service names (cr301_service, cr301_servicear)
        # instead of specialty field, as lab tests can be under any specialty and
        # generic queries rely entirely on service_hint matching
        matched = all_rows  # Start with all services
        mode = "LABORATORY" if is_laboratory else "GENERIC"
        print(
            f"[services] bu={bu_code} specialty={specialty_en!r} → {mode} mode "
            f"→ searching across all {len(matched)} services",
            flush=True,
        )
    else:
        matched = [
            row for row in all_rows
            if (row.get("cr301_specialtyname") or "").strip().lower() in crm_names_lower
        ]
        print(
            f"[services] bu={bu_code} specialty={specialty_en!r} → CRM names={crm_names} "
            f"→ {len(matched)} matched rows",
            flush=True,
        )
    
    if not matched:
        sample = sorted({(r.get("cr301_specialtyname") or "").strip() for r in all_rows[:200]})[:15]
        print(f"[services] sample specialtyname values: {sample}", flush=True)
        return []

    # ── 3. Hint scoring ───────────────────────────────────────────────────
    if service_hint:
        raw_words = [w for w in service_hint.split() if len(w) > 2]
        hint_words = list(raw_words)
        for w in raw_words:
            if w.startswith("ال") and len(w) > 3:
                hint_words.append(w[2:])

        scored: list[tuple[int, dict]] = []
        for row in matched:
            svc_en = (row.get("cr301_service") or "").lower()
            svc_ar = (row.get("cr301_servicear") or "").lower()
            title  = (row.get("cr301_title") or "").lower()
            name_score = sum(2 for w in hint_words if w in svc_en or w in svc_ar)
            title_score = sum(1 for w in hint_words if w in title)
            total = name_score + title_score
            if total > 0:
                scored.append((total, row))

        if scored:
            scored.sort(key=lambda x: -x[0])
            print(
                f"[services] hint {hint_words!r} matched {len(scored)} row(s) | "
                f"top={(scored[0][1].get('cr301_service') or scored[0][1].get('cr301_servicear'))!r} "
                f"(score={scored[0][0]})",
                flush=True,
            )
            for rank, (sc, r) in enumerate(scored[:5], 1):
                print(
                    f"[services]   #{rank} score={sc} | "
                    f"{(r.get('cr301_service') or r.get('cr301_servicear'))!r}",
                    flush=True,
                )
            matched = [r for _, r in scored]
        else:
            print(f"[services] hint {hint_words!r} → no name match → returning cheapest", flush=True)
            matched.sort(key=lambda r: float(r.get("servhub_priced") or 9_999_999))

    # ── 4. Build result list ──────────────────────────────────────────────
    results: list[dict] = []
    for i, row in enumerate(matched[:top_n]):
        r = dict(row)
        r["_service_matched"] = bool(service_hint)
        r["_is_alternative"] = i > 0
        results.append(r)
    return results


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_service_card(service: dict, lang: str = "ar") -> str:
    """Format a service row as a patient-facing card string."""
    title   = (service.get("cr301_title") or "").strip()
    svc_en  = (service.get("cr301_service") or "").strip()
    svc_ar  = (service.get("cr301_servicear") or "").strip()
    # Check cr301_price first (primary field), then servhub_priced as fallback
    price   = service.get("cr301_price") or service.get("servhub_priced")
    spec    = (service.get("cr301_specialtyname") or "").strip()
    code    = (service.get("cr301_code") or "").strip()

    price_str = ""
    if price:
        try:
            f = float(price)
            price_str = str(int(f)) if f == int(f) else f"{f:.2f}"
        except (TypeError, ValueError):
            price_str = str(price)

    if lang == "ar":
        name = svc_ar or svc_en or title
        lines = []
        if name:
            lines.append(f"🔬 الخدمة: *{name}*")
        # Show English title separately if available and different from name
        if title and title.lower() != name.lower():
            lines.append(f"   (English: {title})")
        if spec:
            lines.append(f"🏥 التخصص: {spec}")
        if price_str:
            lines.append(f"💰 السعر: {price_str} ريال")
        if code:
            lines.append(f"🔑 الكود: {code}")
    else:
        name = svc_en or svc_ar or title
        lines = []
        if name:
            lines.append(f"🔬 Service: *{name}*")
        if spec:
            lines.append(f"🏥 Specialty: {spec}")
        if price_str:
            lines.append(f"💰 Price: {price_str} SAR")
        if code:
            lines.append(f"🔑 Code: {code}")

    return "\n".join(lines)
