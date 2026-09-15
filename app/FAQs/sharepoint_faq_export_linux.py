#!/usr/bin/env python3
"""
Linux-native SharePoint FAQ exporter.

What it does
------------
1. Authenticates to Microsoft Entra using device-code flow.
2. Requires the signed-in SharePoint user to be:
      rafik.atallah@andalusiagroup.net
3. Retrieves the latest 1,500 items from:
      https://andalusiagroupegypt.sharepoint.com/sites/AHJ/Medical
      List: FAQ
4. Writes atomically to:
      /home/ai/Workspace/Rafik/QA_System-main/app/FAQs/FAQ_latest.csv

Requirements
------------
    pip install msal requests

Environment
-----------
Required:
    FAQ_ENTRA_CLIENT_ID=<Application/Client ID of your Entra public-client app>

Optional:
    FAQ_ENTRA_TENANT=andalusiagroupegypt.onmicrosoft.com
    FAQ_EXPECTED_EMAIL=rafik.atallah@andalusiagroup.net
    FAQ_OUTPUT_TIMEZONE=Africa/Cairo

The Entra application must support public-client/device-code authentication and
have delegated SharePoint read access approved by your organization.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import msal
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


# =============================================================================
# CONFIGURATION
# =============================================================================

SITE_URL = "https://andalusiagroupegypt.sharepoint.com/sites/AHJ/Medical"
SHAREPOINT_ORIGIN = "https://andalusiagroupegypt.sharepoint.com"
LIST_NAME = "FAQ"

EXPECTED_EMAIL = os.getenv(
    "FAQ_EXPECTED_EMAIL",
    "rafik.atallah@andalusiagroup.net",
).strip()

TENANT = os.getenv(
    "FAQ_ENTRA_TENANT",
    "andalusiagroupegypt.onmicrosoft.com",
).strip()

CLIENT_ID = "31359c7f-bd7e-475c-86db-fdb8c937548e"

MAX_RECORDS = 1500
PAGE_SIZE = 500

TARGET_DIR = Path("/home/ai/Workspace/Rafik/QA_System-main/app/FAQs")
FINAL_FILE = TARGET_DIR / "FAQ_latest.csv"

TOKEN_CACHE_FILE = (
    Path.home() / ".cache" / "sharepoint_faq" / "msal_token_cache.json"
)

OUTPUT_TIMEZONE_NAME = os.getenv("FAQ_OUTPUT_TIMEZONE", "Africa/Cairo")

# Confirmed internal names from the existing working SharePoint export.
FIELD_MAP = {
    "Date-Time": "Date",
    "Customer Name": "Title",
    "BU": "BU",
    "Mobile NO": "Mobile_x002d_NO",
    "inquiry": "group1",
    "Specialty": "field2",
    "CST Inquiry / Request Details": "Service_x002d_Availability",
    "BU Response": "Response",
    "End Call Result": "field4",
    "Created By": "Author",
    "Created": "Created",
    "Response Time": "field5",
    "Modified": "Modified",
    "Modified By": "Editor",
    "Flag": "field8",
    "Inject to CRM": "Inject_x0020_to_x0020_CRM",
    "Item Type": "FSObjType",
    "Path": "FileDirRef",
}

OUTPUT_COLUMNS = list(FIELD_MAP.keys())


# =============================================================================
# HELPERS
# =============================================================================

def fail(message: str, exit_code: int = 1) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def create_http_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json;odata=nometadata",
            "User-Agent": "Andalusia-FAQ-Exporter/1.0",
        }
    )
    return session


def load_token_cache() -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()

    if TOKEN_CACHE_FILE.exists():
        try:
            cache.deserialize(TOKEN_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: Could not read token cache: {exc}")

    return cache


def save_token_cache(cache: msal.SerializableTokenCache) -> None:
    if not cache.has_state_changed:
        return

    TOKEN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    temp_cache = TOKEN_CACHE_FILE.with_suffix(".tmp")
    temp_cache.write_text(cache.serialize(), encoding="utf-8")
    os.chmod(temp_cache, 0o600)
    os.replace(temp_cache, TOKEN_CACHE_FILE)


def acquire_access_token() -> str:
    if not CLIENT_ID:
        fail(
            "FAQ_ENTRA_CLIENT_ID is not set.\n\n"
            "Example:\n"
            "export FAQ_ENTRA_CLIENT_ID='xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'\n\n"
            "Use the Application (client) ID of an Entra public-client app "
            "with delegated SharePoint read access."
        )

    cache = load_token_cache()

    app = msal.PublicClientApplication(
        client_id=CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT}",
        token_cache=cache,
    )

    scopes = [f"{SHAREPOINT_ORIGIN}/.default"]
    result: dict[str, Any] | None = None

    # Reuse only a cached identity matching the required account.
    for account in app.get_accounts():
        username = (account.get("username") or "").strip().lower()

        if username == EXPECTED_EMAIL.lower():
            result = app.acquire_token_silent(
                scopes=scopes,
                account=account,
            )

            if result and "access_token" in result:
                print(f"Using cached Microsoft session for {EXPECTED_EMAIL}.")
                break

    if not result or "access_token" not in result:
        print("\nMicrosoft authentication is required.")
        print(f"Sign in specifically as: {EXPECTED_EMAIL}\n")

        flow = app.initiate_device_flow(scopes=scopes)

        if "user_code" not in flow:
            fail(
                "Could not start device-code authentication:\n"
                + json.dumps(flow, indent=2)
            )

        print(flow["message"])
        sys.stdout.flush()

        result = app.acquire_token_by_device_flow(flow)

    if "access_token" not in result:
        fail(
            "Microsoft authentication failed:\n"
            + result.get("error_description", json.dumps(result, indent=2))
        )

    save_token_cache(cache)

    return result["access_token"]


def sharepoint_get(
    session: requests.Session,
    token: str,
    url: str,
    *,
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = session.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=(20, 120),
    )

    if not response.ok:
        fail(
            f"SharePoint request failed: HTTP {response.status_code}\n"
            f"URL: {response.url}\n"
            f"{response.text[:3000]}"
        )

    try:
        return response.json()
    except ValueError:
        fail(
            "SharePoint returned a non-JSON response.\n"
            f"URL: {response.url}\n"
            f"{response.text[:2000]}"
        )


def extract_email_from_login(login_name: str) -> str:
    if not login_name:
        return ""

    # Example:
    # i:0#.f|membership|rafik.atallah@andalusiagroup.net
    tail = login_name.split("|")[-1].strip()

    if "@" in tail:
        return tail

    match = re.search(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        login_name,
    )

    return match.group(0) if match else ""


def validate_authenticated_user(
    session: requests.Session,
    token: str,
) -> None:
    data = sharepoint_get(
        session,
        token,
        f"{SITE_URL}/_api/web/currentuser",
        params={"$select": "Title,Email,LoginName"},
    )

    actual_email = (
        (data.get("Email") or "").strip()
        or extract_email_from_login(data.get("LoginName") or "")
    )

    print("\nAuthenticated SharePoint user:")
    print(f"  Name : {data.get('Title', '')}")
    print(f"  Email: {actual_email or '(not returned)'}")

    if not actual_email:
        fail("SharePoint did not return an email/login that can be validated.")

    if actual_email.lower() != EXPECTED_EMAIL.lower():
        fail(
            "Wrong Microsoft account authenticated.\n"
            f"Expected:      {EXPECTED_EMAIL}\n"
            f"Authenticated: {actual_email}"
        )

    print("Email validation: SUCCESS\n")


def get_field_schema(
    session: requests.Session,
    token: str,
) -> dict[str, dict[str, Any]]:
    list_name = escape_odata_string(LIST_NAME)

    data = sharepoint_get(
        session,
        token,
        f"{SITE_URL}/_api/web/lists/GetByTitle('{list_name}')/fields",
        params={
            "$select": "Title,InternalName,TypeAsString,LookupField,Hidden"
        },
    )

    fields = data.get("value", [])

    schema = {
        field["InternalName"]: field
        for field in fields
        if field.get("InternalName")
    }

    missing = [
        internal_name
        for internal_name in FIELD_MAP.values()
        if internal_name not in schema
    ]

    if missing:
        fail(
            "These expected SharePoint internal fields were not found:\n  - "
            + "\n  - ".join(missing)
        )

    return schema


def build_query_definition(
    schema: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, dict[str, Any]]]:
    select_fields: list[str] = ["Id"]
    expand_fields: list[str] = []
    extractors: dict[str, dict[str, Any]] = {}

    for output_name, internal_name in FIELD_MAP.items():
        info = schema[internal_name]
        field_type = (info.get("TypeAsString") or "").strip()
        lookup_field = (info.get("LookupField") or "Title").strip() or "Title"

        if field_type in {"Lookup", "LookupMulti", "User", "UserMulti"}:
            select_fields.append(f"{internal_name}/{lookup_field}")
            expand_fields.append(internal_name)

            extractors[output_name] = {
                "kind": "lookup",
                "internal": internal_name,
                "lookup_field": lookup_field,
                "type": field_type,
            }
        else:
            select_fields.append(internal_name)

            extractors[output_name] = {
                "kind": "raw",
                "internal": internal_name,
                "type": field_type,
            }

    # Remove duplicates without changing order.
    select_fields = list(dict.fromkeys(select_fields))
    expand_fields = list(dict.fromkeys(expand_fields))

    return select_fields, expand_fields, extractors


def format_datetime(value: str) -> str:
    if not value:
        return ""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if (
            parsed.tzinfo is not None
            and ZoneInfo is not None
            and OUTPUT_TIMEZONE_NAME
        ):
            parsed = parsed.astimezone(
                ZoneInfo(OUTPUT_TIMEZONE_NAME)
            )

        return parsed.strftime("%Y-%m-%d %H:%M:%S")

    except Exception:
        return value


def lookup_to_text(value: Any, lookup_field: str) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return "; ".join(
            text
            for text in (
                lookup_to_text(v, lookup_field)
                for v in value
            )
            if text
        )

    if isinstance(value, dict):
        if "results" in value:
            return lookup_to_text(value["results"], lookup_field)

        result = value.get(lookup_field)

        if result is None:
            for key in ("Title", "Email", "Name", "LookupValue"):
                if value.get(key) is not None:
                    result = value[key]
                    break

        if isinstance(result, (list, dict)):
            return lookup_to_text(result, lookup_field)

        return "" if result is None else str(result)

    return str(value)


def raw_to_text(value: Any, field_type: str) -> str:
    if value is None:
        return ""

    if field_type == "DateTime" and isinstance(value, str):
        return format_datetime(value)

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, list):
        return "; ".join(
            raw_to_text(v, "")
            for v in value
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return str(value)


def convert_item(
    item: dict[str, Any],
    extractors: dict[str, dict[str, Any]],
) -> dict[str, str]:
    row: dict[str, str] = {}

    for output_name in OUTPUT_COLUMNS:
        spec = extractors[output_name]
        internal_name = spec["internal"]

        if spec["kind"] == "lookup":
            value = lookup_to_text(
                item.get(internal_name),
                spec["lookup_field"],
            )

        else:
            raw = item.get(internal_name)

            if output_name == "Item Type" and internal_name == "FSObjType":
                if raw in (0, "0"):
                    value = "Item"
                elif raw in (1, "1"):
                    value = "Folder"
                else:
                    value = raw_to_text(raw, spec["type"])
            else:
                value = raw_to_text(raw, spec["type"])

        row[output_name] = value

    return row


def normalize_items_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None]:
    # Standard nometadata response.
    if "value" in payload:
        return (
            payload.get("value") or [],
            payload.get("odata.nextLink")
            or payload.get("@odata.nextLink"),
        )

    # Defensive support for verbose OData.
    if "d" in payload and isinstance(payload["d"], dict):
        d = payload["d"]

        return (
            d.get("results") or [],
            d.get("__next"),
        )

    fail("Unexpected SharePoint list-items response format.")


def retrieve_latest_items(
    session: requests.Session,
    token: str,
    schema: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    select_fields, expand_fields, extractors = build_query_definition(schema)

    list_name = escape_odata_string(LIST_NAME)

    url = (
        f"{SITE_URL}/_api/web/lists/"
        f"GetByTitle('{list_name}')/items"
    )

    params: dict[str, str] | None = {
        "$select": ",".join(select_fields),
        "$orderby": "Id desc",
        "$top": str(PAGE_SIZE),
    }

    if expand_fields:
        params["$expand"] = ",".join(expand_fields)

    rows: list[dict[str, str]] = []
    page_number = 0

    print("Retrieving latest SharePoint FAQ records...")

    while url and len(rows) < MAX_RECORDS:
        page_number += 1

        payload = sharepoint_get(
            session,
            token,
            url,
            params=params,
        )

        items, next_url = normalize_items_payload(payload)

        remaining = MAX_RECORDS - len(rows)

        for item in items[:remaining]:
            rows.append(
                convert_item(
                    item,
                    extractors,
                )
            )

        print(
            f"  Page {page_number}: "
            f"{len(items)} received | "
            f"{len(rows)}/{MAX_RECORDS} collected"
        )

        if len(rows) >= MAX_RECORDS:
            break

        if not next_url:
            break

        url = urljoin(SITE_URL, next_url)

        # nextLink already contains the paging and query parameters.
        params = None

        # Gentle pacing.
        time.sleep(1)

    if len(rows) != MAX_RECORDS:
        fail(
            f"Expected {MAX_RECORDS} rows but SharePoint returned "
            f"{len(rows)}."
        )

    return rows


def atomic_write_csv(rows: list[dict[str, str]]) -> None:
    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TARGET_DIR.is_dir():
        fail(f"Target is not a directory: {TARGET_DIR}")

    if not os.access(TARGET_DIR, os.W_OK):
        fail(
            "The current Linux user cannot write to:\n"
            f"{TARGET_DIR}\n\n"
            "Fix the directory ownership/permissions first."
        )

    temp_file = TARGET_DIR / (
        f".FAQ_latest.{os.getpid()}.{int(time.time())}.uploading.csv"
    )

    try:
        with temp_file.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=OUTPUT_COLUMNS,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

            csv_file.flush()
            os.fsync(csv_file.fileno())

        if temp_file.stat().st_size <= 0:
            fail("Temporary CSV was created but is empty.")

        # Atomic replacement because the temp and final files
        # are in the same Linux directory/filesystem.
        os.replace(
            temp_file,
            FINAL_FILE,
        )

    finally:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)

    if not FINAL_FILE.exists():
        fail(f"Final CSV was not created: {FINAL_FILE}")

    print("\n==========================================")
    print("EXPORT COMPLETED SUCCESSFULLY")
    print("==========================================")
    print(f"Authenticated user : {EXPECTED_EMAIL}")
    print(f"Rows exported      : {len(rows)}")
    print(f"Final file         : {FINAL_FILE}")
    print(f"File size          : {FINAL_FILE.stat().st_size:,} bytes")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("==========================================")
    print("SHAREPOINT FAQ -> LOCAL LINUX CSV")
    print("==========================================")
    print(f"Site        : {SITE_URL}")
    print(f"List        : {LIST_NAME}")
    print(f"Latest rows : {MAX_RECORDS}")
    print(f"Output      : {FINAL_FILE}")
    print(f"User check  : {EXPECTED_EMAIL}")

    token = acquire_access_token()
    session = create_http_session()

    try:
        validate_authenticated_user(
            session,
            token,
        )

        print("Validating SharePoint field schema...")

        schema = get_field_schema(
            session,
            token,
        )

        print("Field schema validation: SUCCESS\n")

        rows = retrieve_latest_items(
            session,
            token,
            schema,
        )

        atomic_write_csv(rows)

    finally:
        session.close()


if __name__ == "__main__":
    main()
