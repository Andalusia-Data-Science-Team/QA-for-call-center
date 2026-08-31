"""
CRM Packages Data Layer — offers_node/crm_packages.py

Fetches package records from [dbo].[cr301_ksaservicedataset]
(where cr301_servicecategoryname = 'Package', filtered to active BUs)
via SQL Server (Passcode.json connection).

Public API:
    fetch_packages(bu="LIVE", force_refresh=False) → list[dict]
    get_packages_for_specialty(specialty_en, service_hint, ...)  → list[dict]
    format_package_card(package, lang)             → str

Column aliases mirror service_query.sql (same table, different WHERE clause).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz as _rfuzz

# ── SQL query (loaded from file) ──────────────────────────────────────────────
_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "packages_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

# ── Cache: keyed by BU ────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 30 * 60   # 30 min

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

# ── Specialty mapping (reuse same map as services) ────────────────────────────
_SPECIALTY_MAP: dict[str, list[str]] = {
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
}


def _crm_specialty_names(specialty_en: str) -> list[str]:
    key = (specialty_en or "").strip().lower()
    mapped = _SPECIALTY_MAP.get(key)
    if mapped:
        return mapped
    for map_key, names in _SPECIALTY_MAP.items():
        if map_key in key or key in map_key:
            return names
    return [specialty_en.strip()]


# ── Fetch all packages ────────────────────────────────────────────────────────

def fetch_packages(bu: str | None = None, force_refresh: bool = False) -> list[dict]:
    """
    Return all package records for the given BU from the CRM.
    Uses the same crm_database connection as crm_offers (Azure AD / MSAL).
    Cached per BU for _CACHE_TTL_SECONDS. Never raises — returns [] on failure.
    Thread-safe.
    """
    from app.offers_node.crm_database import _run_query_with_retry, _is_configured

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
            print(f"[packages] {bu_code}: skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return entry["rows"]

        print(f"[packages] {bu_code}: fetching from CRM…", flush=True)
        t0 = time.time()
        try:
            rows = _run_query_with_retry(_QUERY, max_attempts=3, params={"bu_name": bu_code})
            print(f"[packages] {bu_code}: fetched {len(rows)} rows in {time.time()-t0:.1f}s", flush=True)
            entry["rows"] = rows
            entry["loaded_at"] = now
            entry["failed"] = False
            return rows
        except Exception as e:
            print(f"[packages] {bu_code}: fetch failed after {time.time()-t0:.1f}s: {e}", flush=True)
            entry["failed"] = True
            entry["loaded_at"] = now
            return entry["rows"]


# ── Core search ───────────────────────────────────────────────────────────────

def get_packages_for_specialty(
    specialty_en: str,
    service_hint: str = "",
    bu: str | None = None,
    patient_gender: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Return ranked package records for specialty_en + optional hint.

    Strategy (mirrors crm_offers.get_offers_for_specialty):
      1. Direct name match  — if hint ≥ 4 chars, fuzzy-match cr301_service /
                              cr301_servicear / cr301_title against hint
                              (partial_ratio ≥ 85). Returns immediately.
      2. Specialty filter   — keep rows whose cr301_specialtyname matches the
                              hospital-DB specialty (via _SPECIALTY_MAP).
      3. Hint scoring       — score each row by hint-word overlap.
      4. Fallback           — return cheapest rows for the specialty.

    Returns list[dict] — each dict is a raw DB row, augmented with:
        _service_matched  (bool)
        _is_alternative   (bool)
    """
    bu_code = _resolve_bu(bu)
    all_rows = fetch_packages(bu=bu)
    if not all_rows:
        print(f"[packages] get_packages_for_specialty({specialty_en!r}) bu={bu_code}: cache empty", flush=True)
        return []

    # ── 1. Direct name fast-path (enhanced normalization) ─────────────────
    if service_hint and len(service_hint.strip()) >= 4:
        _hint_norm = service_hint.strip().lower()
        
        # ── Enhanced normalization for extracted service/package names ────────
        def _normalize_package_hint(hint: str) -> str:
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
        
        _hint_clean = _normalize_package_hint(_hint_norm)
        
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
        
        _candidates: list[tuple[float, dict]] = []
        for row in all_rows:
            svc_en = (row.get("cr301_service") or "").strip().lower()
            svc_ar = (row.get("cr301_servicear") or "").strip().lower()
            title  = (row.get("cr301_title") or "").strip().lower()
            
            # Normalize database names too
            svc_ar_norm = _norm_ar_fast(svc_ar)
            title_norm = _norm_ar_fast(title)
            
            # Try multiple matching strategies
            # 1. Partial ratio with cleaned hint
            _score1 = _rfuzz.partial_ratio(_hint_clean, svc_en)
            _score2 = _rfuzz.partial_ratio(_hint_clean, svc_ar)
            _score3 = _rfuzz.partial_ratio(_hint_clean, title)
            
            # 2. Partial ratio with fully normalized Arabic
            _score4 = _rfuzz.partial_ratio(_hint_normalized, svc_ar_norm)
            _score5 = _rfuzz.partial_ratio(_hint_normalized, title_norm)
            
            # 3. Token set ratio (word order independent)
            _score6 = _rfuzz.token_set_ratio(_hint_normalized, svc_ar_norm)
            _score7 = _rfuzz.token_set_ratio(_hint_clean, svc_en)
            
            # Take the best score from all strategies
            score = max(_score1, _score2, _score3, _score4, _score5, _score6, _score7)
            
            # Lower threshold for normalized matching (75 instead of 85)
            if score >= 75:
                _candidates.append((score, row))
        
        if _candidates:
            _candidates.sort(key=lambda x: -x[0])
            best_score, best_row = _candidates[0]
            print(
                f"[packages] direct name match: hint={service_hint!r} → cleaned={_hint_clean!r} → "
                f"{(best_row.get('cr301_service') or best_row.get('cr301_servicear'))!r} "
                f"(score={best_score})",
                flush=True,
            )
            
            # Log all high-scoring candidates to show related packages
            if len(_candidates) > 1:
                print(f"[packages] Found {len(_candidates)} related packages:", flush=True)
                for idx, (_sc, _r) in enumerate(_candidates[:5], 1):
                    print(
                        f"[packages]   #{idx}: score={_sc} | "
                        f"{(_r.get('cr301_servicear') or _r.get('cr301_service'))!r} | "
                        f"Price={_r.get('cr301_price')}",
                        flush=True,
                    )
            
            result = dict(best_row)
            result["_service_matched"] = True
            result["_is_alternative"] = False
            print(
                f"[packages] direct name match RETURNED:\n"
                f"  cr301_title      : {result.get('cr301_title')}\n"
                f"  cr301_service    : {result.get('cr301_service')}\n"
                f"  cr301_servicear  : {result.get('cr301_servicear')}\n"
                f"  cr301_specialtyname: {result.get('cr301_specialtyname')}\n"
                f"  cr301_price      : {result.get('cr301_price')}\n"
                f"  cr18c_buname     : {result.get('cr18c_buname')}\n"
                f"  match_score      : {best_score}",
                flush=True,
            )
            return [result]

    # ── 2. Specialty filter ───────────────────────────────────────────────
    crm_names = _crm_specialty_names(specialty_en)
    crm_names_lower = [n.lower() for n in crm_names]
    matched = [
        row for row in all_rows
        if (row.get("cr301_specialtyname") or "").strip().lower() in crm_names_lower
    ]
    print(
        f"[packages] bu={bu_code} specialty={specialty_en!r} → CRM names={crm_names} "
        f"→ {len(matched)} matched rows",
        flush=True,
    )
    if not matched:
        sample = sorted({(r.get("cr301_specialtyname") or "").strip() for r in all_rows[:200]})[:15]
        print(f"[packages] sample specialtyname values: {sample}", flush=True)
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
            name_score  = sum(2 for w in hint_words if w in svc_en or w in svc_ar)
            title_score = sum(1 for w in hint_words if w in title)
            total = name_score + title_score
            if total > 0:
                scored.append((total, row))

        if scored:
            scored.sort(key=lambda x: -x[0])
            print(
                f"[packages] hint {hint_words!r} matched {len(scored)} row(s) | "
                f"top={(scored[0][1].get('cr301_service') or scored[0][1].get('cr301_servicear'))!r} "
                f"(score={scored[0][0]})",
                flush=True,
            )
            for rank, (sc, r) in enumerate(scored[:5], 1):
                print(
                    f"[packages]   #{rank} score={sc} | "
                    f"{(r.get('cr301_service') or r.get('cr301_servicear'))!r}",
                    flush=True,
                )
            matched = [r for _, r in scored]
        else:
            print(f"[packages] hint {hint_words!r} → no name match → returning cheapest", flush=True)
            matched.sort(key=lambda r: float(r.get("cr301_price") or 9_999_999))

    # ── 4. Build result list ──────────────────────────────────────────────
    results: list[dict] = []
    for i, row in enumerate(matched[:top_n]):
        r = dict(row)
        r["_service_matched"] = bool(service_hint)
        r["_is_alternative"] = i > 0
        results.append(r)
    return results


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_package_card(package: dict, lang: str = "ar") -> str:
    """Format a package row as a patient-facing card string."""
    title   = (package.get("cr301_title") or "").strip()
    svc_en  = (package.get("cr301_service") or "").strip()
    svc_ar  = (package.get("cr301_servicear") or "").strip()
    price   = package.get("cr301_price") or package.get("servhub_priced")
    spec    = (package.get("cr301_specialtyname") or "").strip()
    code    = (package.get("cr301_code") or "").strip()
    desc    = (package.get("cr301_nphiesdescription") or "").strip()

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
            lines.append(f"📦 الباقة: *{name}*")
        if spec:
            lines.append(f"🏥 التخصص: {spec}")
        if desc:
            lines.append(f"📝 الوصف: {desc}")
        if price_str:
            lines.append(f"💰 السعر: {price_str} ريال")
        if code:
            lines.append(f"🔑 الكود: {code}")
    else:
        name = svc_en or svc_ar or title
        lines = []
        if name:
            lines.append(f"📦 Package: *{name}*")
        if spec:
            lines.append(f"🏥 Specialty: {spec}")
        if desc:
            lines.append(f"📝 Description: {desc}")
        if price_str:
            lines.append(f"💰 Price: {price_str} SAR")
        if code:
            lines.append(f"🔑 Code: {code}")

    return "\n".join(lines)

if __name__ == "__main__":
    # Quick test
    print(_QUERY)
    fetched = fetch_packages(bu="LIVE", force_refresh=True)
    print(f"Fetched {len(fetched)} packages for LIVE")