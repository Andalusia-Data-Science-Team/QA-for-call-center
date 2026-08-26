"""
CRM Bank-Account Reference Data — app/service_hub/crm_bank.py

Fetches + caches the authoritative KSA bank-account reference rows
(cr301_bankaccounts) from Dynamics 365, independently of location data —
see app/service_hub/crm_location.py for the location-only equivalent.
They used to be fetched together by one function in a shared
"locbank_node" package; splitting them lets bank and location be entirely
separate graph nodes/features (each with its own cache, its own failure
mode) while still sharing only the generic, domain-agnostic connector in
app/services/crm_connector.py — the same one app/service_hub/crm_offers.py
uses. (bank_node/location_node were later merged with offers_node into
this single app/service_hub/ package; the modules themselves stayed
independent.)

Public API:
    fetch_bank_accounts(force_refresh=False) -> list[dict]
        Never raises — returns cached (possibly empty) data on failure so a
        CRM outage degrades bank validation to BUSINESS_UNIT_UNRESOLVED
        rather than crashing the QA pipeline.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from app.config import settings

# Reference data changes slowly — reuse the same TTL as the doctor-price
# cache in app/service_hub/crm_database.py rather than inventing a new policy.
_CACHE_TTL_SECONDS = settings.CRM_PRICE_CACHE_TTL_SECONDS

_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "bank_accounts_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

_cache: dict = {"banks": [], "loaded_at": 0.0, "failed": False}
_lock = threading.Lock()


def fetch_bank_accounts(force_refresh: bool = False) -> list[dict]:
    """
    Return the active KSA bank-account reference rows used by
    app.service_hub.bank_validation for deterministic QA checks.

    Cached for _CACHE_TTL_SECONDS. Thread-safe single-flight fetch. Never
    raises — degrades to whatever is cached (or []) on failure, backing off
    for 5 minutes before retrying, matching the doctor-price cache's contract.
    """
    from app.services.crm_connector import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    with _lock:
        now = time.time()
        age = now - _cache["loaded_at"]
        if not force_refresh and _cache["loaded_at"] and age < _CACHE_TTL_SECONDS:
            return _cache["banks"]
        if _cache["failed"] and age < 300:
            print("[crm_bank] skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return _cache["banks"]

        print("[crm_bank] Fetching bank-account reference data from Dynamics 365...", flush=True)
        t0 = time.time()
        try:
            banks = _run_query_with_retry(_QUERY)
        except Exception as exc:
            print(f"[crm_bank] fetch failed after {time.time()-t0:.1f}s: {exc}", flush=True)
            _cache["failed"] = True
            _cache["loaded_at"] = now
            return _cache["banks"]

        print(f"[crm_bank] Fetched {len(banks)} bank account(s) in {time.time()-t0:.1f}s", flush=True)
        _cache["banks"] = banks
        _cache["loaded_at"] = now
        _cache["failed"] = False
        return banks
