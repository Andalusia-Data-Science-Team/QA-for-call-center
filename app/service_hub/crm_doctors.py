"""
CRM Doctor Reference Data — app/service_hub/crm_doctors.py

Fetches + caches the authoritative doctor dataset (cr301_newdoctordataset,
joined with cr301_table1 for consultation fees) from Dynamics 365,
independently of bank/location data — see crm_bank.py / crm_location.py
(same app/service_hub/ package) for the sibling equivalents. Shares only the
generic, domain-agnostic connector in app/services/crm_connector.py.

The SQL query fetches ALL doctor records — it is NOT filtered by
cr301_opdflag, since a doctor who isn't flagged OPD (e.g. a home-care or
other non-OPD service context) can still be a genuine, resolvable doctor;
cr301_opdflag is carried through as informational metadata only. Active
status and the supported-BU allowlist are applied in Python
(doctor_validation.py, see _is_active()) since they depend on this
project's own BU vocabulary, not a raw CRM value.

Public API:
    fetch_doctors(force_refresh=False) -> list[dict]
        Never raises — returns cached (possibly empty) data on failure so a
        CRM outage degrades doctor validation to a non-punitive
        "insufficient reference data" outcome rather than crashing the QA
        pipeline.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from app.config import settings

# Reference data changes slowly — same TTL policy as crm_bank.py/crm_location.py.
_CACHE_TTL_SECONDS = settings.CRM_PRICE_CACHE_TTL_SECONDS

_SQL_FILE = Path(__file__).resolve().parent.parent / "SQL" / "doctors_query.sql"
_QUERY = _SQL_FILE.read_text(encoding="utf-8")

_cache: dict = {"doctors": [], "loaded_at": 0.0, "failed": False}
_lock = threading.Lock()


def fetch_doctors(force_refresh: bool = False) -> list[dict]:
    """
    Return the full doctor reference rows (OPD and non-OPD alike) used by
    app.service_hub.doctor_validation for deterministic + semantic QA
    checks.

    Cached for _CACHE_TTL_SECONDS. Thread-safe single-flight fetch. Never
    raises — degrades to whatever is cached (or []) on failure, backing off
    for 5 minutes before retrying, matching crm_bank.py/crm_location.py's
    contract exactly.
    """
    from app.services.crm_connector import _run_query_with_retry, _is_configured

    if not _is_configured():
        return []

    with _lock:
        now = time.time()
        age = now - _cache["loaded_at"]
        if not force_refresh and _cache["loaded_at"] and age < _CACHE_TTL_SECONDS:
            return _cache["doctors"]
        if _cache["failed"] and age < 300:
            print("[crm_doctors] skipping fetch — last attempt failed, in 5-min backoff", flush=True)
            return _cache["doctors"]

        print("[crm_doctors] Fetching doctor reference data from Dynamics 365...", flush=True)
        t0 = time.time()
        try:
            doctors = _run_query_with_retry(_QUERY)
        except Exception as exc:
            print(f"[crm_doctors] fetch failed after {time.time()-t0:.1f}s: {exc}", flush=True)
            _cache["failed"] = True
            _cache["loaded_at"] = now
            return _cache["doctors"]

        print(f"[crm_doctors] Fetched {len(doctors)} doctor record(s) in {time.time()-t0:.1f}s", flush=True)
        _cache["doctors"] = doctors
        _cache["loaded_at"] = now
        _cache["failed"] = False
        return doctors
