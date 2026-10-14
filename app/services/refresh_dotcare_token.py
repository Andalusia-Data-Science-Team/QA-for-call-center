"""Refresh the MPI authentication token and persist it to the project .env file.

Calls the SecurityService login endpoint with NTLM authentication, extracts
the "token" field from the returned JSON payload, and updates MPI_AUTH_TOKEN
in .env so that get_patient.py (and other MPI consumers) pick up the new token.

Usage:
    python3 refresh_dotcare_token.py            # use .env credentials
    python3 refresh_dotcare_token.py --check    # only show current token expiry
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv, set_key
from requests_ntlm import HttpNtlmAuth


DEFAULT_LOGIN_URL = "http://ahj-hmis:7000/SercurityService/api/security/login"
DEFAULT_SERVICE_USERNAME = "AiModule"
DEFAULT_SERVICE_PASSWORD = "123456"

# Project root (two levels up: app/services -> app -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def decode_jwt_expiry(token: str) -> datetime | None:
    """Extract the 'exp' claim from a JWT without verifying the signature."""
    try:
        payload_part = token.split(".")[1]
        # Fix base64 padding
        payload_part += "=" * (-len(payload_part) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_part))
        exp = payload.get("exp")
        if exp:
            return datetime.fromtimestamp(exp, tz=timezone.utc)
    except Exception:
        pass
    return None


def refresh_token() -> str:
    """Call the login endpoint and return the new token."""
    load_dotenv(ENV_FILE)

    login_url = os.getenv("MPI_LOGIN_URL", DEFAULT_LOGIN_URL)
    service_username = os.getenv("MPI_SERVICE_USERNAME", DEFAULT_SERVICE_USERNAME)
    service_password = os.getenv("MPI_SERVICE_PASSWORD", DEFAULT_SERVICE_PASSWORD)
    ntlm_user = os.getenv("MPI_NTLM_USERNAME")
    ntlm_password = os.getenv("MPI_NTLM_PASSWORD")

    if not ntlm_user or not ntlm_password:
        sys.exit(
            "Set MPI_NTLM_USERNAME and MPI_NTLM_PASSWORD in the project .env file "
            "before running this script (format: andalusia\\username)."
        )

    params = {"username": service_username, "password": service_password}
    auth = HttpNtlmAuth(ntlm_user, ntlm_password)

    try:
        response = requests.get(login_url, params=params, auth=auth, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        body = getattr(error.response, "text", "").strip()
        sys.exit(f"Login request failed: {error}\nServer response: {body}")

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        sys.exit(f"Login endpoint returned non-JSON response:\n{response.text[:500]}")

    token = payload.get("token") if isinstance(payload, dict) else None
    if isinstance(token, dict):
        # Login response nests the JWT under token.Value
        token = token.get("Value")
    if not token or not isinstance(token, str):
        sys.exit(
            "Login response did not contain a usable 'token' field.\n"
            f"Response keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        )
    return token


def update_env_token(token: str) -> None:
    """Write the new token to the .env file (create it if missing)."""
    ENV_FILE.touch(exist_ok=True)
    set_key(ENV_FILE, "MPI_AUTH_TOKEN", token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only decode and show the current MPI_AUTH_TOKEN expiry, no refresh.",
    )
    args = parser.parse_args()

    load_dotenv(ENV_FILE)
    current_token = os.getenv("MPI_AUTH_TOKEN", "")

    if args.check:
        if not current_token:
            print("No MPI_AUTH_TOKEN is currently set in .env")
            return
        expiry = decode_jwt_expiry(current_token)
        if expiry:
            remaining = expiry - datetime.now(timezone.utc)
            print(f"Current token expires at: {expiry.isoformat()}")
            print(f"Time remaining: {remaining}")
            if remaining.total_seconds() <= 0:
                print("STATUS: EXPIRED — run refresh_mpi_token.py to renew it.")
            else:
                print("STATUS: valid")
        else:
            print("Could not decode token expiry (not a standard JWT).")
        return

    new_token = refresh_token()
    update_env_token(new_token)

    expiry = decode_jwt_expiry(new_token)
    print("MPI_AUTH_TOKEN updated successfully in .env")
    if expiry:
        print(f"New token expires at: {expiry.isoformat()}")


if __name__ == "__main__":
    main()
