"""CRM lead lookup and validation-prompt helpers for the QA graph."""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyodbc
import httpx

from app.config import settings
from app.models.input import CallTranscript
from app.services.text_helpers import normalize_arabic_text


logger = logging.getLogger(__name__)
_IDENTIFIER_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Business unit abbreviations. Different abbreviations for the same branch
# are used interchangeably in chat transcripts and CRM records; each maps to
# the one code that branch is evaluated as.
LEAD_BU_CODE: dict[str, str] = {
    "AHJ": "HJH", "HJH": "HJH",
    "SNB": "AFW", "AFW": "AFW",
    "CHT": "ALW", "LCH": "ALW", "ALW": "ALW",
    "MKR": "ADC", "JDC": "ADC", "ADC": "ADC",
    "AKW": "AKW",
}


def lead_clinic_bu(record: dict[str, Any]) -> tuple[Any, str]:
    """The lead's branch to evaluate, and the field it came from: new_clinicbu,
    or new_sourcebusinessunitname only when new_clinicbu is empty."""
    clinic = record.get("new_clinicbu")
    if str(clinic or "").strip():
        return clinic, "new_clinicbu"
    source = record.get("new_sourcebusinessunitname")
    if str(source or "").strip():
        return source, "new_sourcebusinessunitname"
    return clinic, "new_clinicbu"


def lead_bu_code(bu: str | None) -> str | None:
    """The branch code *bu* is evaluated as ("BU-MKR" → "ADC"), or None when
    it is empty or not a known code (e.g. a full branch name)."""
    if not bu:
        return None
    code = str(bu).strip().upper().removeprefix("BU-").strip()
    return LEAD_BU_CODE.get(code)


def normalize_business_unit(bu: str | None) -> str:
    """Normalize a business unit abbreviation to a canonical form for comparison.

    Returns the branch code from LEAD_BU_CODE, or the uppercased input if no
    mapping exists.
    """
    if not bu:
        return ""
    return lead_bu_code(bu) or str(bu).strip().upper()


def _quoted_table_name(raw_table: str) -> str:
    """Validate and quote a one- or two-part SQL identifier."""
    parts = [part.strip().strip("[]") for part in (raw_table or "").split(".")]
    if not parts or len(parts) > 2 or any(not _IDENTIFIER_PART.fullmatch(p) for p in parts):
        raise ValueError("CRM_LEADS_TABLE must be a valid table or schema.table identifier")
    # Dataverse entities are exposed without a physical dbo schema. Some
    # endpoints drop schema-qualified entity queries during SQLExecute.
    if len(parts) == 2 and parts[0].lower() == "dbo":
        parts = parts[1:]
    return ".".join(f"[{part}]" for part in parts)


_SAUDI_NATIONAL_RE = re.compile(r"^5\d{8}$")
# CRM saves every non-Saudi patient with mobilephone 0555555550 and types the
# real number into new_notes.
_PLACEHOLDER_NATIONAL = "555555550"


def _is_saudi_national(national: str) -> bool:
    """Return True only when *national* looks like a valid Saudi mobile number."""
    return bool(_SAUDI_NATIONAL_RE.fullmatch(national))


def _phone_values(phone: str) -> tuple[dict[str, str], bool]:
    """Build the common Saudi CRM representations of a retrieved chat phone.

    Returns a tuple of (phone_dict, is_non_saudi).

    If the supplied phone cannot be normalised to a 9-digit Saudi national
    number (e.g. a foreign number like +20…), it is parsed into the CRM
    placeholder 0555555550 and is_non_saudi is set to True; the caller then
    narrows the placeholder leads by the real number in new_notes.
    """
    digits = re.sub(r"\D", "", str(phone or ""))
    if digits.startswith("966"):
        national = digits[3:]
    elif digits.startswith("0"):
        national = digits[1:]
    else:
        national = digits

    is_non_saudi = not _is_saudi_national(national)
    if is_non_saudi:
        logger.warning(
            "Phone '%s' is not a recognised Saudi number; "
            "looking up the placeholder mobilephone 0%s, filtered by new_notes.",
            phone,
            _PLACEHOLDER_NATIONAL,
        )
        national = _PLACEHOLDER_NATIONAL

    return (
        {
            "mobile_number": national,
            "local_number": f"0{national}" if national else "",
            "intl_number": f"966{national}" if national else "",
            "plus_intl_number": f"+966{national}" if national else "",
        },
        is_non_saudi,
    )


def notes_phone_pattern(phone: str) -> str:
    """The patient's number as it is typed into new_notes: "+" and its digits
    ("201001234567", "+201001234567" and "00201001234567" → "+201001234567")."""
    digits = re.sub(r"\D", "", str(phone or ""))
    return "+" + (digits[2:] if digits.startswith("00") else digits)


def notes_contain_phone(notes: Any, phone: str) -> bool:
    """True when free-text *notes* contain the patient's "+" number — the same
    test as the query's new_notes LIKE '%+<digits>%'."""
    return notes_phone_pattern(phone) in str(notes or "")


def _load_query(non_saudi: bool = False) -> str:
    sql_path = Path(__file__).resolve().parent.parent / "SQL" / "CRM_leads.sql"
    query = sql_path.read_text(encoding="utf-8")
    table = _quoted_table_name(settings.CRM_LEADS_TABLE)
    query = query.replace("[dbo].[lead]", table)
    # Accept the phone formats used by both chat retrieval and Dynamics.
    query = query.replace(
        "mobilephone = :mobile_number",
        "mobilephone IN (:mobile_number, :local_number, :intl_number, :plus_intl_number)"
        # Non-Saudi patients share the placeholder 0555555550; their real
        # number, written "+<digits>", is in new_notes.
        + (" AND new_notes LIKE :notes_phone" if non_saudi else ""),
    )
    return query


# CRM Leads uses a confidential Azure application (client credentials).
# CRM_Secret_ID identifies the secret record in Azure; OAuth requires its value,
# CLIENT_SECRET, together with CRM_LEAD_CLIENT_ID and CRM_Tenant_Id.
_SQL_COPT_SS_ACCESS_TOKEN = 1256
_leads_token_cache: dict[str, Any] = {
    "key": None,
    "token": None,
    "expires_at": 0.0,
}
_leads_token_lock = threading.Lock()


def _crm_leads_server() -> str:
    """Return the Dataverse TDS endpoint with its required default port."""
    server = (settings.CRM_LEADS_SERVER or "").strip()
    if server and "," not in server:
        return f"{server},5558"
    return server


def _crm_leads_host() -> str:
    return _crm_leads_server().split(",")[0].strip()


def _crm_leads_is_configured() -> bool:
    return all(
        (
            _crm_leads_host(),
            settings.CRM_TENANT_ID,
            settings.CRM_LEAD_CLIENT_ID,
            settings.CLIENT_SECRET,
        )
    )


def _clear_crm_leads_token() -> None:
    with _leads_token_lock:
        _leads_token_cache.update({"key": None, "token": None, "expires_at": 0.0})


def _get_crm_leads_access_token(force_refresh: bool = False) -> str:
    """Acquire an app-only Dynamics token with OAuth client credentials."""
    if not _crm_leads_is_configured():
        raise RuntimeError(
            "CRM Leads client-credentials configuration is incomplete; "
            "CRM_LEADS_SERVER, CRM_Tenant_Id, CRM_LEAD_CLIENT_ID, and "
            "CLIENT_SECRET are required."
        )

    cache_key = (
        _crm_leads_host(),
        settings.CRM_TENANT_ID,
        settings.CRM_LEAD_CLIENT_ID,
    )
    now = time.time()
    with _leads_token_lock:
        if (
            not force_refresh
            and _leads_token_cache["key"] == cache_key
            and _leads_token_cache["token"]
            and _leads_token_cache["expires_at"] > now + 60
        ):
            return str(_leads_token_cache["token"])

        import msal

        application = msal.ConfidentialClientApplication(
            client_id=settings.CRM_LEAD_CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{settings.CRM_TENANT_ID}",
            client_credential=settings.CLIENT_SECRET,
        )
        result = application.acquire_token_for_client(
            scopes=[f"https://{_crm_leads_host()}/.default"]
        )
        token = result.get("access_token")
        if not token:
            error = result.get("error_description") or result.get("error") or "unknown error"
            raise RuntimeError(f"CRM Leads OAuth client-credentials authentication failed: {error}")

        _leads_token_cache.update(
            {
                "key": cache_key,
                "token": token,
                "expires_at": now + int(result.get("expires_in", 3599)),
            }
        )
        return str(token)


def _get_crm_leads_access_token_connection(
    force_token_refresh: bool = False,
) -> pyodbc.Connection:
    """Legacy manual-token TDS connection retained for possible reversion."""
    token = _get_crm_leads_access_token(force_refresh=force_token_refresh)
    token_bytes = token.encode("UTF-16-LE")
    token_struct = struct.pack(
        f"<I{len(token_bytes)}s",
        len(token_bytes),
        token_bytes,
    )
    server = _crm_leads_server()
    org_name = _crm_leads_host().split(".")[0]
    connection_string = (
        f"DRIVER={{{settings.DB_DRIVER}}};"
        f"SERVER={server};"
        f"DATABASE={org_name};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=120;"
    )
    connection = pyodbc.connect(
        connection_string,
        attrs_before={_SQL_COPT_SS_ACCESS_TOKEN: token_struct},
        timeout=120,
    )
    connection.timeout = 300
    return connection


def _get_crm_leads_connection(force_token_refresh: bool = False) -> pyodbc.Connection:
    """Open a fresh TDS connection using ODBC service-principal auth."""
    del force_token_refresh  # Every call creates a connection and authenticates again.
    if not _crm_leads_is_configured():
        raise RuntimeError(
            "CRM Leads client-credentials configuration is incomplete; "
            "CRM_LEADS_SERVER, CRM_Tenant_Id, CRM_LEAD_CLIENT_ID, and "
            "CLIENT_SECRET are required."
        )

    server = _crm_leads_server()
    org_name = _crm_leads_host().split(".")[0]
    connection_string = (
        f"DRIVER={{{settings.DB_DRIVER}}};"
        f"SERVER={server};"
        f"DATABASE={org_name};"
        f"UID={settings.CRM_LEAD_CLIENT_ID};"
        f"PWD={settings.CLIENT_SECRET};"
        "Authentication=ActiveDirectoryServicePrincipal;"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=120;"
    )
    connection = pyodbc.connect(connection_string, timeout=120)
    connection.timeout = 300
    return connection


def _run_crm_leads_query_once(
    query: str,
    params: dict[str, Any],
    force_token_refresh: bool = False,
) -> list[dict]:
    """Open a fresh confidential-client connection and execute one query."""
    keys = re.findall(r":([A-Za-z_][A-Za-z0-9_]*)", query)
    values = tuple(params[key] for key in keys)
    positional_query = re.sub(r":[A-Za-z_][A-Za-z0-9_]*", "?", query)

    with _get_crm_leads_connection(force_token_refresh=force_token_refresh) as connection:
        cursor = connection.cursor()
        cursor.execute(positional_query, values) if values else cursor.execute(positional_query)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _is_retryable_crm_leads_error(exc: Exception) -> bool:
    message = str(exc).lower()
    # Entra ID AADSTS responses and HTTP 401/403 are deterministic identity,
    # tenant, secret, consent, or permission errors. Reconnecting cannot fix
    # them and only delays the pipeline.
    if (
        "aadsts" in message
        or "http 401" in message
        or "http 403" in message
    ):
        return False

    return any(
        marker in message
        for marker in (
            "08s01",
            "08001",
            "28000",
            "communication link failure",
            "connection",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "connecterror",
            "tcp provider",
            "timeout",
            "temporarily unavailable",
            "login failed",
            "authentication failed",
        )
    )


def _run_crm_leads_query_with_retry(
    query: str,
    params: dict[str, Any],
) -> list[dict]:
    """Reconnect with exponential backoff after CRM Leads connection failures."""
    attempts = settings.CRM_LEADS_MAX_RETRIES
    base_delay = settings.CRM_LEADS_RETRY_DELAY_SECONDS
    searched_phone = (
        params.get("local_number")
        or params.get("mobile_number")
        or "N/A"
    )
    report_date = params.get("Report_Date") or "N/A"

    for attempt in range(1, attempts + 1):
        logger.info(
            "CRM Leads connection attempt %d/%d starting | phone=%s report_date=%s server=%s",
            attempt,
            attempts,
            searched_phone,
            report_date,
            _crm_leads_server(),
        )
        try:
            return _run_crm_leads_query_once(
                query,
                params,
                force_token_refresh=attempt > 1,
            )
        except Exception as exc:
            if not _is_retryable_crm_leads_error(exc) or attempt == attempts:
                logger.error(
                    "CRM Leads connection attempt %d/%d failed; no more retries | phone=%s report_date=%s error=%s",
                    attempt,
                    attempts,
                    searched_phone,
                    report_date,
                    exc,
                )
                raise
            _clear_crm_leads_token()
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "CRM Leads connection attempt %d/%d failed; retrying in %gs | phone=%s report_date=%s error=%s",
                attempt,
                attempts,
                delay,
                searched_phone,
                report_date,
                exc,
            )
            if delay:
                time.sleep(delay)

    return []


_FORMATTED_VALUE_SUFFIX = "@OData.Community.Display.V1.FormattedValue"


def _dataverse_leads_entity_set() -> str:
    logical_name = (settings.CRM_LEADS_TABLE or "lead").split(".")[-1].strip("[]")
    return "leads" if logical_name.casefold() == "lead" else logical_name


def _odata_escape(value: str) -> str:
    return value.replace("'", "''")


def _display_value(record: dict[str, Any], logical_name: str) -> Any:
    formatted_key = f"{logical_name}{_FORMATTED_VALUE_SUFFIX}"
    lookup_formatted_key = f"_{logical_name}_value{_FORMATTED_VALUE_SUFFIX}"
    if record.get(formatted_key) not in (None, ""):
        return record[formatted_key]
    if record.get(lookup_formatted_key) not in (None, ""):
        return record[lookup_formatted_key]
    return record.get(logical_name)


def _normalize_web_api_lead(record: dict[str, Any]) -> dict[str, Any]:
    """Map Web API lookup/choice annotations to the SQL-style validation names."""
    normalized = dict(record)
    normalized["modifiedbyname"] = _display_value(record, "modifiedby")
    normalized["leadsourcecodename"] = _display_value(record, "leadsourcecode")
    normalized["statuscodename"] = _display_value(record, "statuscode")
    normalized["new_clinicbu"] = (
        record.get("new_clinicbu")
        or _display_value(record, "new_clinicbu")
    )
    normalized["new_doctor"] = _display_value(record, "new_doctor")
    normalized["new_lastcallresult"] = _display_value(record, "new_lastcallresult")
    normalized["new_lastcallcreatedbyname"] = (
        record.get("new_lastcallcreatedbyname")
        or _display_value(record, "new_lastcallcreatedby")
    )
    return normalized


def _saudi_day_utc_filter(report_date: str) -> str:
    """OData createdon bounds (UTC) for the Saudi (UTC+3) calendar day
    *report_date*: 21:00Z the day before up to 21:00Z on the day."""
    day = datetime.strptime(report_date, "%Y-%m-%d")
    start = day - timedelta(hours=3)
    end = start + timedelta(days=1)
    return (
        f"createdon ge {start:%Y-%m-%dT%H:%M:%SZ} and "
        f"createdon lt {end:%Y-%m-%dT%H:%M:%SZ}"
    )


def _fetch_crm_leads_web_api_once(
    phone: str,
    report_date: str,
    force_token_refresh: bool = False,
) -> list[dict]:
    """Fetch matching leads through Dataverse Web API using the app token."""
    token = _get_crm_leads_access_token(force_refresh=force_token_refresh)
    phone_dict, is_non_saudi = _phone_values(phone)
    phone_clause = " or ".join(
        f"mobilephone eq '{_odata_escape(value)}'"
        for value in dict.fromkeys(phone_dict.values())
        if value
    )
    if not phone_clause:
        return []
    if is_non_saudi:
        # Same as the SQL path's new_notes LIKE '%+<digits>%'.
        phone_clause = (
            f"({phone_clause}) and "
            f"contains(new_notes,'{_odata_escape(notes_phone_pattern(phone))}')"
        )

    url = (
        f"https://{_crm_leads_host()}/api/data/v9.2/"
        f"{_dataverse_leads_entity_set()}"
    )
    response = httpx.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "OData-MaxVersion": "4.0",
            "OData-Version": "4.0",
            "Prefer": 'odata.include-annotations="OData.Community.Display.V1.FormattedValue"',
        },
        params={
            "$" + "filter": (
                f"({phone_clause}) and {_saudi_day_utc_filter(report_date)}"
            ),
            "$" + "orderby": "createdon desc",
            "$" + "top": "100",
        },
        timeout=120.0,
    )
    if response.status_code in {401, 403}:
        raise RuntimeError(
            f"CRM Leads Web API authorization failed with HTTP {response.status_code}"
        )
    if response.status_code >= 400:
        raise RuntimeError(f"CRM Leads Web API request failed with HTTP {response.status_code}")

    records = [
        _normalize_web_api_lead(item)
        for item in response.json().get("value", [])
    ]
    eligible: list[dict] = []
    for record in records:
        source = str(record.get("leadsourcecodename") or "").casefold()
        status = str(record.get("statuscodename") or "").casefold()
        created_by = str(record.get("new_lastcallcreatedbyname") or "").casefold()
        if source and source != "whatsapp":
            continue
        if status == "untouched":
            continue
        if created_by == "andalusia sharepoint":
            continue
        eligible.append(record)
    return eligible


def _fetch_crm_leads_web_api_with_retry(
    phone: str,
    report_date: str,
) -> list[dict]:
    """Reconnect to Web API with a fresh token after transient failures."""
    attempts = settings.CRM_LEADS_MAX_RETRIES
    base_delay = settings.CRM_LEADS_RETRY_DELAY_SECONDS

    for attempt in range(1, attempts + 1):
        try:
            return _fetch_crm_leads_web_api_once(
                phone,
                report_date,
                force_token_refresh=attempt > 1,
            )
        except Exception as exc:
            if not _is_retryable_crm_leads_error(exc) or attempt == attempts:
                raise
            _clear_crm_leads_token()
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"[CRM LEADS] Web API attempt {attempt}/{attempts} failed; "
                f"reconnecting in {delay:g}s"
            )
            if delay:
                time.sleep(delay)
    return []


# Lead fields that name the agent who handled it, in priority order.
_LEAD_AGENT_FIELDS = ("modifiedbyname", "new_lastcallcreatedbyname")


def same_agent_name(a: Any, b: Any) -> bool:
    """True when two agent names are the same person, ignoring case, spacing
    and Arabic letter variants. A shorter multi-word name matches a longer
    one that contains all its words ("Maha Gamal" ~ "Maha Gamal Ahmed")."""
    left, right = normalize_arabic_text(str(a or "")).split(), normalize_arabic_text(str(b or "")).split()
    if not left or not right:
        return False
    if left == right:
        return True
    short, long_ = sorted((left, right), key=len)
    return len(short) >= 2 and set(short) <= set(long_)


def select_lead_for_agents(
    rows: list[dict[str, Any]],
    agent_names: list[str],
) -> tuple[dict[str, Any], str | None]:
    """The first lead (newest first, as every lookup sorts) whose agent fields
    name one of the chat's agents, with that agent's name; otherwise the
    newest lead and None.
    A transferred chat has several agents, and the lead names only the one
    who recorded it — that is the agent the lead is evaluated against."""
    for record in rows:
        for field in _LEAD_AGENT_FIELDS:
            for agent in agent_names:
                if same_agent_name(record.get(field), agent):
                    return record, agent
    return rows[0], None


def fetch_crm_lead(
    phone: str,
    report_date: str,
    agent_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch the WhatsApp lead for the chat phone on/after report_date.

    Among the qualifying leads, the one that names a chat agent (any of
    *agent_names*) is selected and that agent is returned as ``agent_name``.
    """
    agent_names = [n for n in (agent_names or []) if n]
    logger.info(
        "CRM Leads lookup starting | phone=%s report_date=%s",
        phone,
        report_date,
    )
    if not _crm_leads_is_configured():
        return {
            "status": "unavailable",
            "record": None,
            "message": "CRM leads lookup is disabled because client-credentials configuration is incomplete.",
        }

    phone_dict, is_non_saudi = _phone_values(phone)
    params = {**phone_dict, "Report_Date": report_date}
    if is_non_saudi:
        params["notes_phone"] = f"%{notes_phone_pattern(phone)}%"
    try:
        rows = _run_crm_leads_query_with_retry(
            _load_query(non_saudi=is_non_saudi), params=params,
        )
    except Exception as tds_exc:
        logger.warning(
            "CRM Leads TDS query failed; falling back to Web API | error=%s",
            tds_exc,
        )
        try:
            rows = _fetch_crm_leads_web_api_with_retry(phone, report_date)
        except Exception as web_api_exc:
            # CRM availability must not become an agent violation.
            return {
                "status": "unavailable",
                "record": None,
                "message": (
                    f"CRM leads TDS query failed: {tds_exc}; "
                    f"Web API fallback failed: {web_api_exc}"
                ),
            }

    # Non-Saudi: the query already kept only placeholder leads whose
    # new_notes holds "+<digits>"; re-check in case a path returned others.
    notes_phone_match: bool | None = None
    if is_non_saudi:
        rows = [r for r in rows if notes_contain_phone(r.get("new_notes"), phone)]
        notes_phone_match = bool(rows)
        logger.info(
            "CRM Leads non-Saudi lookup | phone=%s new_notes LIKE '%%%s%%' leads=%d",
            phone, notes_phone_pattern(phone), len(rows),
        )

    if not rows:
        logger.info(
            "CRM Leads lookup result | phone=%s report_date=%s status=not_found",
            phone,
            report_date,
        )
        return {
            "status": "not_found",
            "record": None,
            "message": (
                "No qualifying WhatsApp CRM lead has the patient's non-Saudi number in new_notes."
                if is_non_saudi
                else "No qualifying WhatsApp CRM lead matched the chat phone and report date."
            ),
        }
    record, matched_agent = select_lead_for_agents(rows, agent_names)
    logger.info(
        "CRM Leads lookup result | phone=%s report_date=%s status=found "
        "total_qualifying=%d chat_agents=%s matched_agent=%s record=%s",
        phone,
        report_date,
        len(rows),
        agent_names,
        matched_agent,
        json.dumps(record, ensure_ascii=False, default=str),
    )
    selection = (
        f"lead naming chat agent {matched_agent} selected"
        if matched_agent
        else "no lead names a chat agent; newest selected"
    )
    return {
        "status": "found",
        "record": record,
        "agent_name": matched_agent,
        "chat_agents": agent_names,
        "non_saudi": is_non_saudi,
        "notes_phone_match": notes_phone_match,
        "message": f"CRM lead found ({len(rows)} qualifying record(s); {selection}).",
    }



def matched_crm_lead_attributes(
    evaluation: dict[str, Any],
    crm_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return CRM values for fields the evaluation matched to call insights."""
    matched: list[dict[str, Any]] = []
    for check in evaluation.get("field_checks") or []:
        if not isinstance(check, dict) or check.get("matches") is not True:
            continue
        field = str(check.get("field") or "").strip()
        if not field:
            continue
        value = lead_clinic_bu(crm_record)[0] if field == "new_clinicbu" else crm_record.get(field)
        matched.append(
            {
                "field": field,
                "crm_value": value if value not in (None, "") else check.get("actual"),
                "call_insight": check.get("expected"),
            }
        )
    return matched


def normalize_crm_lead_evaluation(data: dict[str, Any]) -> dict[str, Any]:
    """Enforce the public flag contract regardless of model formatting drift."""
    normalized_flags: list[dict[str, str]] = []
    for raw_flag in data.get("crm_leads_flags") or []:
        if not isinstance(raw_flag, dict):
            continue
        description = str(
            raw_flag.get("description") or "CRM lead data does not match the chat."
        ).strip()
        excerpt = str(raw_flag.get("transcript_excerpt") or "N/A").strip()
        normalized_flags.append(
            {
                "type": "C2B",
                "severity": "critical",
                "description": description,
                "transcript_excerpt": excerpt,
            }
        )
        if len(normalized_flags) == 4:
            break
    # A model can correctly mark a field_check as false yet accidentally omit
    # the parallel flag. Materialize it here so every detected mismatch is C2B.
    flagged_fields = " ".join(flag["description"].lower() for flag in normalized_flags)
    for check in data.get("field_checks") or []:
        if len(normalized_flags) == 4:
            break
        if not isinstance(check, dict) or check.get("matches") is not False:
            continue
        field = str(check.get("field") or "unknown field").strip()
        if field.lower() in flagged_fields:
            continue
        reason = str(check.get("reason") or "does not match the chat facts").strip()
        normalized_flags.append(
            {
                "type": "C2B",
                "severity": "critical",
                "description": f"CRM lead mismatch in {field}: {reason}",
                "transcript_excerpt": "N/A",
            }
        )

    if data.get("crm_lead_status") == "violation" and not normalized_flags:
        normalized_flags.append(
            {
                "type": "C2B",
                "severity": "critical",
                "description": "CRM lead data does not match the authoritative chat facts.",
                "transcript_excerpt": "N/A",
            }
        )
    data["crm_leads_flags"] = normalized_flags
    data["crm_lead_status"] = "violation" if normalized_flags else "match"
    return data


def apply_notes_phone_check(
    evaluation: dict[str, Any],
    lookup: dict[str, Any],
    phone: str,
) -> dict[str, Any]:
    """For a non-Saudi number, record whether the lead's new_notes holds the
    patient's mobile number; a missing number is a C2B violation."""
    if not lookup.get("non_saudi"):
        return evaluation
    notes = (lookup.get("record") or {}).get("new_notes")
    matches = notes_contain_phone(notes, phone)
    evaluation.setdefault("field_checks", []).append({
        "field": "new_notes",
        "expected": phone,
        "actual": notes,
        "matches": matches,
        "reason": (
            "new_notes contains the patient's non-Saudi mobile number."
            if matches
            else "new_notes does not contain the patient's non-Saudi mobile number."
        ),
    })
    if not matches:
        flags = evaluation.get("crm_leads_flags") or []
        flags.insert(0, {
            "type": "C2B",
            "severity": "critical",
            "description": (
                f"CRM lead mismatch in new_notes: the patient's non-Saudi mobile "
                f"number {phone} is not recorded in new_notes (mobilephone holds "
                f"the placeholder 0{_PLACEHOLDER_NATIONAL})."
            ),
            "transcript_excerpt": "N/A",
        })
        evaluation["crm_leads_flags"] = flags[:4]
        evaluation["crm_lead_status"] = "violation"
    return evaluation


def missing_lead_evaluation(message: str) -> dict[str, Any]:
    """Create the deterministic C2B result for a successful lookup with no lead."""
    return {
        "crm_lead_status": "violation",
        "objective": None,
        "final_outcome": None,
        "summary": message,
        "field_checks": [],
        "crm_leads_flags": [
            {
                "type": "C2B",
                "severity": "critical",
                "description": "CRM lead missing for the chat phone and selected report date.",
                "transcript_excerpt": "N/A",
            }
        ],
    }
