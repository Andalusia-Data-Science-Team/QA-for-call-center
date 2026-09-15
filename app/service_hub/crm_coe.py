"""
CRM COE (Center of Excellence) Reference Data — app/service_hub/crm_coe.py

Fetches + caches the authoritative COE reference rows (cr301_coelist) from
Dynamics 365, independently of doctor/bank/location/offer data — see
crm_doctors.py / crm_bank.py / crm_location.py / crm_offers.py (same
app/service_hub/ package) for the sibling equivalents. Shares only the
generic, domain-agnostic connector in app/services/crm_connector.py.

This module supplies COE-level reference/context data only (clinic name,
business unit, specialty, clinic leader/coordinator, members, and the
approved Arabic script) for app.service_hub.coe_validation. It is
deliberately NOT the source of the approved *primary* doctor for a new COE
booking — that authoritative list is hardcoded in coe_validation.py per
business rules (see AUTHORITATIVE_PRIMARY_DOCTORS there); cr301_coemembers
and cr301_clinicalleader must never be used to infer it.

Public API:
    fetch_coe_reference(force_refresh=False) -> list[dict]
        Never raises — returns cached (possibly empty) data on failure so a
        CRM outage degrades COE validation to a non-punitive "uncertain"/
        "not_applicable" outcome rather than crashing the QA pipeline.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from app.config import settings

# Reference data changes slowly — same TTL policy as crm_doctors.py/crm_bank.py.
_CACHE_TTL_SECONDS = settings.CRM_PRICE_CACHE_TTL_SECONDS

_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "coe_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

_cache: dict = {"coes": [], "loaded_at": 0.0, "failed": False}
_lock = threading.Lock()


def fetch_coe_reference(force_refresh: bool = False) -> list[dict]:
    """
    Return the four supported COE reference rows used by
    app.service_hub.coe_validation for COE-routing/script-adherence checks.

    Cached for _CACHE_TTL_SECONDS. Thread-safe single-flight fetch. Never
    raises — degrades to whatever is cached (or []) on failure, backing off
    for 5 minutes before retrying, matching crm_doctors.py/crm_bank.py's
    contract exactly. Missing rows, duplicate rows, and malformed/HTML
    content are all handled defensively by the caller
    (coe_validation.build_coe_reference), never here.
    """
    from app.services.crm_connector import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    with _lock:
        now = time.time()
        age = now - _cache["loaded_at"]
        if not force_refresh and _cache["loaded_at"] and age < _CACHE_TTL_SECONDS:
            return _cache["coes"]
        if _cache["failed"] and age < 300:
            print("[crm_coe] skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return _cache["coes"]

        print("[crm_coe] Fetching COE reference data from Dynamics 365...", flush=True)
        t0 = time.time()
        try:
            coes = _run_query_with_retry(_QUERY)
        except Exception as exc:
            print(f"[crm_coe] fetch failed after {time.time()-t0:.1f}s: {exc}", flush=True)
            _cache["failed"] = True
            _cache["loaded_at"] = now
            return _cache["coes"]

        print(f"[crm_coe] Fetched {len(coes)} COE record(s) in {time.time()-t0:.1f}s", flush=True)
        _cache["coes"] = coes
        _cache["loaded_at"] = now
        _cache["failed"] = False
        return coes
