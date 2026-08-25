"""
Dynamics 365 CRM connector — generic auth/connection/query-execution layer
for the CRM SQL/TDS endpoint using Azure AD auth.

This module is intentionally domain-agnostic: it knows nothing about offers,
doctors, locations, or bank accounts — it just gets a token and runs a query.
app/offers_node/crm_offers.py (promotional offers), app/bank_node/crm_bank.py
(KSA bank-account reference data), app/location_node/crm_location.py (KSA
branch/location reference data), and app/offers_node/crm_database.py (doctor
walk-in prices) all depend on it. Keeping it neutral and shared here avoids
any of those domain packages importing from one another just to reach
connection primitives — this used to live inside offers_node/crm_database.py,
which made every other CRM-backed domain reach into an "offers" module for
something that has nothing to do with offers.

Auth strategy (MSAL, public client):
  1. Silent — read a cached refresh token from disk (works after first login).
  2. Username+password — headless fallback when CRM_PASSWORD is set and MFA
     is disabled on the account.
  3. Interactive — opens a browser so the user can complete MFA. Only useful
     on a dev machine; on the deployed server the token cache must already
     exist (run scripts/check_crm_prices.py locally, then copy the cache).

Connection: pyodbc with the access token via SQL_COPT_SS_ACCESS_TOKEN.
"""
import atexit
import os
import struct
import threading
import time
from typing import Optional

import pyodbc

from app.config import settings

CRM_SERVER = settings.CRM_SERVER
CRM_CLIENT_ID = settings.CRM_CLIENT_ID
CRM_TENANT = settings.CRM_TENANT
CRM_USERNAME = settings.CRM_USERNAME
CRM_PASSWORD = settings.CRM_PASSWORD
DB_DRIVER = settings.DB_DRIVER

# Where MSAL persists the refresh token between runs. Override with
# MSAL_TOKEN_CACHE_PATH if you want a different location (e.g. shared volume).
_MSAL_CACHE_PATH = os.environ.get(
    "MSAL_TOKEN_CACHE_PATH",
    os.path.join(os.path.expanduser("~"), ".andalusia_crm_cache.bin"),
)
# Enable the interactive (browser) flow on first run. Disable on the server.
_ALLOW_INTERACTIVE = os.environ.get("CRM_ALLOW_INTERACTIVE", "1") != "0"

SQL_COPT_SS_ACCESS_TOKEN = 1256

_token_cache: dict = {"token": None, "expires_at": 0.0}
_token_lock = threading.Lock()
# Set to True when a 28000/connection-expired error is detected so _get_token()
# forces MSAL to do a real network refresh instead of returning the stale
# cached access token that Dynamics rejected.
_force_token_refresh = False

# Set to True after the first call to `_load_msal_cache` registers its persist
# callback with atexit. Without this guard, every token refresh would queue
# another copy of the same callback, and atexit would fire N copies at
# shutdown — each writing the same cache file.
_msal_atexit_registered = False


def _crm_host() -> str:
    # "org2f45e702.crm4.dynamics.com,5558" → "org2f45e702.crm4.dynamics.com"
    return (CRM_SERVER or "").split(",")[0].strip()


def _is_configured() -> bool:
    # Server is the only hard requirement — password is optional when a cached
    # refresh token or interactive login is used.
    return bool(CRM_SERVER and _crm_host())


def _load_msal_cache(msal_mod):
    global _msal_atexit_registered
    cache = msal_mod.SerializableTokenCache()
    try:
        if os.path.exists(_MSAL_CACHE_PATH):
            with open(_MSAL_CACHE_PATH, "r", encoding="utf-8") as f:
                cache.deserialize(f.read())
    except Exception as e:
        print(f"[CRM] token cache unreadable ({e}); starting fresh")

    def _persist():
        if cache.has_state_changed:
            try:
                os.makedirs(os.path.dirname(_MSAL_CACHE_PATH) or ".", exist_ok=True)
                with open(_MSAL_CACHE_PATH, "w", encoding="utf-8") as f:
                    f.write(cache.serialize())
            except Exception as e:
                print(f"[CRM] could not persist token cache: {e}")

    # Register at most once per process. `_load_msal_cache` is called from
    # `_get_token` on every token refresh; without the guard each call queues
    # another copy of `_persist` into atexit, leaking unboundedly.
    if not _msal_atexit_registered:
        atexit.register(_persist)
        _msal_atexit_registered = True
    return cache, _persist


def _get_token(force_refresh: bool = False) -> Optional[str]:
    """Acquire a bearer token for the CRM TDS endpoint. Caches in memory until near expiry.

    Args:
        force_refresh: When True, bypasses both the in-memory token cache AND
            MSAL's internal SerializableTokenCache, forcing a real network
            round-trip to Azure AD. Use this after a Dynamics 365 28000
            'Connection expired' error to guarantee a brand-new access token.
    """
    global _force_token_refresh
    if not _is_configured():
        return None

    now = time.time()
    # Consume the module-level force flag (set by _run_query_with_retry on
    # auth errors) in addition to any caller-supplied force_refresh argument.
    effective_force = force_refresh or _force_token_refresh

    with _token_lock:
        _force_token_refresh = False  # consumed — reset immediately
        if not effective_force and _token_cache["token"] and _token_cache["expires_at"] > now + 60:
            return _token_cache["token"]

        import msal

        authority = f"https://login.microsoftonline.com/{CRM_TENANT}"
        scope = [f"https://{_crm_host()}/.default"]

        cache, persist = _load_msal_cache(msal)
        app = msal.PublicClientApplication(
            CRM_CLIENT_ID, authority=authority, token_cache=cache,
        )

        result = None

        # 1. Silent — refresh token from disk cache.
        #    Pass force_refresh=True when we detected a 28000 error so MSAL
        #    makes a real HTTP refresh request instead of returning the stale
        #    cached access token that Dynamics already rejected.
        accounts = app.get_accounts(username=CRM_USERNAME or None)
        if accounts:
            result = app.acquire_token_silent(
                scope, account=accounts[0], force_refresh=effective_force
            )
            if effective_force:
                print(
                    f"[CRM] forced MSAL refresh → "
                    f"{'ok' if result and 'access_token' in result else 'failed'}",
                    flush=True,
                )

        # 2. Username + password — only if password was provided
        if (not result or "access_token" not in result) and CRM_USERNAME and CRM_PASSWORD:
            print("[CRM] Authenticating with username+password…", flush=True)
            result = app.acquire_token_by_username_password(
                username=CRM_USERNAME, password=CRM_PASSWORD, scopes=scope,
            )

        # 3. Interactive — opens a browser (useful for MFA-enabled accounts)
        if (not result or "access_token" not in result) and _ALLOW_INTERACTIVE:
            print("[CRM] opening browser for interactive login…")
            result = app.acquire_token_interactive(
                scopes=scope, login_hint=CRM_USERNAME or None,
            )

        if not result or "access_token" not in result:
            err = result.get('error_description') if result else 'no result'
            raise RuntimeError(f"CRM auth failed: {err}")

        persist()
        _token_cache["token"] = result["access_token"]
        _token_cache["expires_at"] = now + int(result.get("expires_in", 3599))
        return _token_cache["token"]


def _get_connection() -> pyodbc.Connection:
    token = _get_token()
    if not token:
        raise RuntimeError("CRM not configured")

    token_bytes = token.encode("UTF-16-LE")
    # Little-endian unsigned int (<I) is what SQL Server actually requires.
    # The previous =i (native signed) worked on x86 but could silently break
    # on big-endian hosts. Matches the proven working db.py pattern.
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

    # Dynamics TDS endpoint requires TLS and a Database= value equal to the
    # org name (first DNS label of the server, e.g. "org2f45e702"). Without
    # these the driver opens the socket but the server drops it on first
    # execute, surfacing as "Communication link failure" on SELECT 1.
    # Connection Timeout=120 must also appear inside the connection string —
    # the pyodbc timeout= kwarg is a connect-phase hint only on some drivers.
    org = _crm_host().split(".")[0]
    conn_str = (
        f"DRIVER={{{DB_DRIVER}}};"
        f"SERVER={CRM_SERVER};"
        f"DATABASE={org};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=120;"
    )
    conn = pyodbc.connect(
        conn_str,
        attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct},
        timeout=120,   # matches db.py working pattern
    )
    conn.timeout = 300  # 5-min query timeout for large CRM result sets
    return conn


def _run_query_with_retry(query: str, max_attempts: int = 3) -> list[dict]:
    """Open a fresh connection and execute the query, retrying on transient errors."""
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            with _get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                cols = [c[0] for c in cursor.description]
                return [dict(zip(cols, r)) for r in cursor.fetchall()]
        except pyodbc.Error as e:
            last_err = e
            msg = str(e).lower()
            # Retry on connection drops / timeouts / expired tokens.
            # Dynamics 365 TDS returns SQLSTATE 28000 / error 18456 with the
            # message "Connection expired" when the OAuth access token has
            # expired. Treat it as transient: clear the stale token and let
            # _get_token() acquire a fresh one on the next attempt.
            transient = any(k in msg for k in (
                "communication link failure", "tcp provider", "timeout expired",
                "connection is closed", "connection reset",
                "connection expired",   # Dynamics 365 TDS: stale OAuth token
                "login failed",          # 28000 / 18456 — always retry once
            ))
            # On auth-related errors: clear our in-memory cache AND set the
            # module-level force flag so _get_token() bypasses MSAL's internal
            # SerializableTokenCache on the next attempt, forcing a real HTTP
            # refresh round-trip to Azure AD.
            is_auth_err = any(k in msg for k in ("login failed", "connection expired", "28000"))
            if is_auth_err:
                global _force_token_refresh
                _force_token_refresh = True        # tell _get_token() to force-refresh
                with _token_lock:
                    _token_cache["token"] = None
                    _token_cache["expires_at"] = 0.0
            if not transient or attempt == max_attempts:
                raise
            backoff = 2 ** (attempt - 1)
            print(f"[CRM] query attempt {attempt} failed ({e}); retrying in {backoff}s")
            time.sleep(backoff)
            # For non-auth transient errors, still clear the token as belt-and-braces.
            if not is_auth_err:
                with _token_lock:
                    _token_cache["token"] = None
                    _token_cache["expires_at"] = 0.0
    if last_err:
        raise last_err
    return []
