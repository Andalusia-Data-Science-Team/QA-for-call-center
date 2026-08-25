"""
CRM Branch/Location Reference Data — location_node/crm_location.py

Fetches + caches the authoritative KSA branch/location reference rows
(cr301_andalusialocations) from Dynamics 365, independently of bank data —
see app/bank_node/crm_bank.py for the bank-only equivalent. Split out of what
used to be one shared "locbank_node" package so bank and location can be
entirely separate graph nodes/features (each with its own cache, its own
failure mode) while still sharing only the generic, domain-agnostic
connector in app/services/crm_connector.py.

Public API:
    fetch_ksa_locations(force_refresh=False) -> list[dict]
        Never raises — returns cached (possibly empty) data on failure so a
        CRM outage degrades location validation to a non-punitive
        "unresolved" outcome rather than crashing the QA pipeline.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from app.config import settings

_CACHE_TTL_SECONDS = settings.CRM_PRICE_CACHE_TTL_SECONDS

_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "ksa_locations_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

_cache: dict = {"locations": [], "loaded_at": 0.0, "failed": False}
_lock = threading.Lock()


def fetch_ksa_locations(force_refresh: bool = False) -> list[dict]:
    """
    Return the active KSA branch/location reference rows used by
    app.location_node.location_validation for deterministic QA checks.

    Cached for _CACHE_TTL_SECONDS. Thread-safe single-flight fetch. Never
    raises — degrades to whatever is cached (or []) on failure, backing off
    for 5 minutes before retrying.
    """
    from app.services.crm_connector import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    with _lock:
        now = time.time()
        age = now - _cache["loaded_at"]
        if not force_refresh and _cache["loaded_at"] and age < _CACHE_TTL_SECONDS:
            return _cache["locations"]
        if _cache["failed"] and age < 300:
            print("[crm_location] skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return _cache["locations"]

        print("[crm_location] Fetching location reference data from Dynamics 365...", flush=True)
        t0 = time.time()
        try:
            locations = _run_query_with_retry(_QUERY)
        except Exception as exc:
            print(f"[crm_location] fetch failed after {time.time()-t0:.1f}s: {exc}", flush=True)
            _cache["failed"] = True
            _cache["loaded_at"] = now
            return _cache["locations"]

        print(f"[crm_location] Fetched {len(locations)} location(s) in {time.time()-t0:.1f}s", flush=True)
        _cache["locations"] = locations
        _cache["loaded_at"] = now
        _cache["failed"] = False
        return locations
