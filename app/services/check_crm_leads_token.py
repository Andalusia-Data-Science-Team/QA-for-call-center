"""Verify the CRM Leads OAuth client-credentials token using the .env values.

Runs the exact request the CRM leads validation service makes
(app/service_hub/crm_leads_validation.py):

    POST https://login.microsoftonline.com/{CRM_Tenant_Id}/oauth2/v2.0/token
         grant_type=client_credentials
         client_id={CRM_LEAD_CLIENT_ID}
         client_secret={CLIENT_SECRET}
         scope=https://{CRM_LEADS_SERVER host}/.default

Usage:
    python app/services/check_crm_leads_token.py

Exit codes: 0 = token OK, 1 = configuration incomplete, 2 = token rejected.
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

TOKEN_PATH = "/oauth2/v2.0/token"

# AADSTS error → remediation hint, so the output is actionable.
AADSTS_HINTS = {
    "AADSTS7000222": (
        "The client secret for this app registration is EXPIRED.\n"
        "     Fix: an admin must sign in to the Azure portal → Microsoft Entra ID →\n"
        "     App registrations → this app → 'Certificates & secrets' →\n"
        "     'New client secret', then paste the new secret VALUE into\n"
        "     CLIENT_SECRET in the project .env and re-run this script.\n"
        "     (CRM_Secret_ID in .env is only the Azure secret record ID —\n"
        "     OAuth needs the secret VALUE, not the record ID.)"
    ),
    "AADSTS700016": (
        "The client_id was not found in this tenant — check CRM_LEAD_CLIENT_ID\n"
        "     and CRM_Tenant_Id in .env point at the same Entra tenant."
    ),
    "AADSTS7000212": (
        "The client secret VALUE does not match any active secret on the app\n"
        "     registration — a new secret was likely created; copy its VALUE\n"
        "     into CLIENT_SECRET in .env."
    ),
    "AADSTS90002": "The tenant was not found — check CRM_Tenant_Id in .env.",
}


def decode_jwt_expiry(token: str) -> datetime | None:
    """Extract the 'exp' claim from a JWT without verifying the signature."""
    try:
        payload_part = token.split(".")[1]
        payload_part += "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part))
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp, tz=timezone.utc)
    except Exception:
        pass
    return None


def main() -> None:
    load_dotenv(ENV_FILE)

    tenant_id = (os.getenv("CRM_Tenant_Id") or "").strip()
    client_id = (os.getenv("CRM_LEAD_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("CLIENT_SECRET") or "").strip()
    leads_server = (os.getenv("CRM_LEADS_SERVER") or "").strip()

    missing = [
        name
        for name, value in (
            ("CRM_Tenant_Id", tenant_id),
            ("CRM_LEAD_CLIENT_ID", client_id),
            ("CLIENT_SECRET", client_secret),
            ("CRM_LEADS_SERVER", leads_server),
        )
        if not value
    ]
    if missing:
        print(f"[FAIL] Configuration incomplete — missing in .env: {missing}")
        sys.exit(1)

    scope_host = leads_server.split(",")[0].strip()
    token_url = f"https://login.microsoftonline.com/{tenant_id}{TOKEN_PATH}"

    print("CRM Leads OAuth client-credentials check")
    print(f"  token url : {token_url}")
    print(f"  client_id : {client_id}")
    print(f"  scope     : https://{scope_host}/.default")

    response = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": f"https://{scope_host}/.default",
        },
        timeout=30,
    )
    payload = response.json()

    if "access_token" not in payload:
        description = payload.get("error_description", payload.get("error", "unknown error"))
        code = next(
            (hint for hint in AADSTS_HINTS if hint in description),
            None,
        )
        print(f"\n[FAIL] Token request rejected (HTTP {response.status_code}):")
        print(f"  {description}")
        if code:
            print(f"\n  {AADSTS_HINTS[code]}")
        sys.exit(2)

    expiry = decode_jwt_expiry(payload["access_token"])
    remaining = expiry - datetime.now(timezone.utc) if expiry else None
    print("\n[PASS] Access token acquired.")
    if expiry:
        print(f"  expires at: {expiry.isoformat()} (in {remaining})")
    print(f"  expires_in: {payload.get('expires_in')}s")
    print(
        "\nNext step: the Dataverse TDS query and the Web API fallback in\n"
        "app/service_hub/crm_leads_validation.py will now authenticate."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()