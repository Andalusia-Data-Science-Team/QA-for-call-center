"""
CRM doctor reference data — walk-in/cash price, specialty, etc. — pulled
from Dynamics 365 via the shared connector in app/services/crm_connector.py.

Location/bank reference data used to be fetched from this module too; that
now lives in its own independent nodes — app/bank_node/crm_bank.py and
app/location_node/crm_location.py — and the generic auth/connection/retry
primitives all three depend on live in app/services/crm_connector.py
(no domain package owns them).

Cache: in-memory dict with a TTL (default 24h) — CRM data changes slowly
and we never want to block the booking flow on CRM latency.
"""
import threading
import time

import pyodbc

from app.config import settings
from app.services.crm_connector import _run_query_with_retry, _is_configured

CRM_DOCTOR_TABLE = settings.CRM_DOCTOR_TABLE
CRM_FEE_TABLE = settings.CRM_FEE_TABLE
CRM_PRICE_CACHE_TTL_SECONDS = settings.CRM_PRICE_CACHE_TTL_SECONDS

_doctor_cache: dict = {"doctors": [], "loaded_at": 0.0, "failed": False}
_doctor_lock = threading.Lock()


# Walk-in fees live on cr301_table1 (one row per doctor, joined on cr301_doctorkey).
# Try the richest projection first; fall back to the minimum if the tenant's
# view is missing any of the denormalized lookup-name columns.
_QUERY_VARIANTS = [
    # Full: everything useful for debugging + display
    """
    SELECT
        D.[servhub_doctornameen]          AS DoctorEn,
        D.[cr301_doctornamear]            AS DoctorAr,
        D.[cr301_specialtyname]           AS Specialty,
        D.[cr301_subspecialtyname]        AS SubSpecialty,
        D.[cr301_businessunitname]        AS BusinessUnit,
        D.[cr301_stardoctorname]          AS IsStar,
        D.[cr18c_firstpriorityname]       AS IsPriority,
        D.[cr301_degreename]             AS Degree,
        D.[cr301_scopeofservice]          AS ScopeEN,
        D.[cr301_scopeofservicear]        AS ScopeAR,
        D.[cr301_opdflag]                 AS OPDFlag,
        D.[servhub_examinationage]        AS ExaminationAge,
        F.[cr301_walkinconsultationfees]  AS WalkInPrice
    FROM {doctor_table} D
    LEFT JOIN {fee_table} F
        ON D.[cr301_doctorkey] = F.[cr301_doctorkey]
    WHERE D.[statuscodename] = 'Active'
      AND D.[servhub_doctornameen] IS NOT NULL
    """,
    # Minimal: just what we actually need to show a price
    """
    SELECT
        D.[servhub_doctornameen]          AS DoctorEn,
        D.[cr301_doctornamear]            AS DoctorAr,
        D.[cr301_opdflag]                 AS OPDFlag,
        F.[cr301_walkinconsultationfees]  AS WalkInPrice
    FROM {doctor_table} D
    LEFT JOIN {fee_table} F
        ON D.[cr301_doctorkey] = F.[cr301_doctorkey]
    WHERE D.[servhub_doctornameen] IS NOT NULL
    """,
]


def get_crm_doctors_by_specialty(
    specialty_en: str,
    child_age: int = None,
) -> list[dict]:
    """Return all active CRM doctors for the given specialty (English name).

    When child_age is provided, only doctors whose ExaminationAge covers that
    age are returned (same logic as _filter_dental_for_child in routing.py).

    Each returned dict has at minimum:
        DoctorEn, DoctorAr, Specialty, SubSpecialty, ExaminationAge,
        WalkInPrice, IsStar, IsPriority, Degree, ScopeEN, ScopeAR
    """
    from services.doctor_price import _specialty_matches  # avoid circular at top-level
    all_doctors = fetch_all_doctor_prices()
    if not all_doctors:
        return []

    matched = [
        r for r in all_doctors
        if _specialty_matches(specialty_en, r.get("Specialty") or "")
        and r.get("DoctorEn")
    ]

    if child_age is None:
        return matched

    # Filter by ExaminationAge when child age is known
    from nodes.routing import _parse_examination_age_range  # lazy import
    eligible = []
    for r in matched:
        exam_age_str = r.get("ExaminationAge") or ""
        if not exam_age_str:
            continue  # SKIP-STRICT: no evidence of child acceptance
        lo, hi = _parse_examination_age_range(exam_age_str)
        # Guard against None/invalid returns from the parser
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue  # unparseable → skip
        if lo == 0 and hi == 0:
            continue  # explicit "no range" sentinel → skip
        if lo <= child_age <= hi:
            eligible.append(r)
    return eligible


def fetch_all_doctor_prices(force_refresh: bool = False) -> list[dict]:
    """
    Return list of dicts: {DoctorEn, DoctorAr, Specialty, WalkInPrice, BusinessUnit}.
    Cached for CRM_PRICE_CACHE_TTL_SECONDS. Never raises — returns [] on failure
    so the booking flow can still function without price data.
    Thread-safe: the full fetch runs under _doctor_lock so concurrent callers
    wait for a single result instead of all launching their own CRM query.
    """
    if not _is_configured():
        return []

    with _doctor_lock:
        now = time.time()
        cached = _doctor_cache["doctors"]
        age = now - _doctor_cache["loaded_at"]
        if not force_refresh and cached and age < CRM_PRICE_CACHE_TTL_SECONDS:
            return cached
        # If the last attempt failed, back off for 5 minutes before retrying
        if _doctor_cache["failed"] and age < 300:
            print("[CRM] skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return cached

        print("[CRM] Fetching all doctor prices from Dynamics 365...", flush=True)
        t0 = time.time()
        rows: list[dict] = []
        try:
            for i, template in enumerate(_QUERY_VARIANTS, 1):
                try:
                    rows = _run_query_with_retry(
                        template.format(doctor_table=CRM_DOCTOR_TABLE, fee_table=CRM_FEE_TABLE)
                    )
                    break
                except pyodbc.Error as e:
                    print(f"[CRM] query variant {i} failed: {e}")
                    if i == len(_QUERY_VARIANTS):
                        raise
        except Exception as e:
            print(f"[CRM] fetch_all_doctor_prices failed after {time.time()-t0:.1f}s: {e}", flush=True)
            _doctor_cache["failed"] = True
            _doctor_cache["loaded_at"] = now
            return _doctor_cache["doctors"] or []

        print(f"[CRM] Fetched {len(rows)} doctor price records in {time.time()-t0:.1f}s", flush=True)
        _doctor_cache["doctors"] = rows
        _doctor_cache["loaded_at"] = now
        _doctor_cache["failed"] = False
        return rows
