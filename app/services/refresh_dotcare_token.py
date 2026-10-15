"""Refresh the MPI authentication token and persist it to the project .env file.

Calls the SecurityService login endpoint with NTLM authentication, extracts
the "token" field from the returned JSON payload, and updates MPI_AUTH_TOKEN
or environment-specific MPI_AUTH_TOKEN_<ENV> in .env so that get_patient.py
(and other MPI consumers) pick up the new token.

Usage:
    python3 refresh_dotcare_token.py                # refresh generic MPI_AUTH_TOKEN
    python3 refresh_dotcare_token.py --env AHJ      # refresh MPI_AUTH_TOKEN_AHJ
    python3 refresh_dotcare_token.py --all          # refresh tokens for all environments
    python3 refresh_dotcare_token.py --check        # show current token expiry
    python3 refresh_dotcare_token.py --check --env SNB  # check SNB token expiry
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

# Environment definitions (same as get_patient.py)
ENVIRONMENTS = {
    "AHJ": "ahj-hmis",
    "SNB": "snb-hmisapp-01",
    "AKW": "akw-hmis",
    "ALW": "alw-hmis",
    "ERS": "ers-fwzhmisapp",
    "JDC": "jdc-hmis",
}

SECURITY_SERVICE_PORT = 7000
LOGIN_PATH = "/SercurityService/api/security/login"


def build_login_url(env: str | None = None) -> str:
    """Build the login URL for an environment, or use the default."""
    if env:
        override = os.getenv(f"MPI_LOGIN_URL_{env}")
        if override:
            return override
        return f"http://{ENVIRONMENTS[env]}:{SECURITY_SERVICE_PORT}{LOGIN_PATH}"
    return os.getenv("MPI_LOGIN_URL", DEFAULT_LOGIN_URL)


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


def refresh_token(env: str | None = None) -> str:
    """Call the login endpoint and return the new token.
    
    Args:
        env: Optional environment name (AHJ, SNB, etc.). If None, uses default.
    """
    load_dotenv(ENV_FILE)

    login_url = build_login_url(env)
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

    env_label = f" for {env}" if env else ""
    print(f"Requesting token{env_label} from {login_url}...")

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


def update_env_token(token: str, env: str | None = None) -> None:
    """Write the new token to the .env file (create it if missing).
    
    Args:
        token: The JWT token to save
        env: Optional environment name. If provided, saves as MPI_AUTH_TOKEN_<ENV>
    """
    ENV_FILE.touch(exist_ok=True)
    env_var = f"MPI_AUTH_TOKEN_{env}" if env else "MPI_AUTH_TOKEN"
    set_key(ENV_FILE, env_var, token)
    print(f"Updated {env_var} in .env")


def check_token_expiry(env: str | None = None) -> None:
    """Check and display token expiry information."""
    load_dotenv(ENV_FILE)
    env_var = f"MPI_AUTH_TOKEN_{env}" if env else "MPI_AUTH_TOKEN"
    current_token = os.getenv(env_var, "")
    
    env_label = f" ({env})" if env else ""
    
    if not current_token:
        print(f"No {env_var} is currently set in .env")
        return
    
    expiry = decode_jwt_expiry(current_token)
    if expiry:
        remaining = expiry - datetime.now(timezone.utc)
        print(f"Token{env_label}: {env_var}")
        print(f"Expires at: {expiry.isoformat()}")
        print(f"Time remaining: {remaining}")
        if remaining.total_seconds() <= 0:
            print("STATUS: EXPIRED — run refresh_dotcare_token.py to renew it.")
        else:
            print("STATUS: valid")
    else:
        print(f"Could not decode token{env_label} expiry (not a standard JWT).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only decode and show the current token expiry, no refresh.",
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENTS),
        help="Specific environment to refresh/check (AHJ, SNB, AKW, ALW, ERS, JDC)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Refresh tokens for all environments",
    )
    args = parser.parse_args()

    if args.check:
        if args.all:
            print("Checking all environment tokens:")
            print("=" * 78)
            for env in sorted(ENVIRONMENTS):
                check_token_expiry(env)
                print()
        elif args.env:
            check_token_expiry(args.env)
        else:
            check_token_expiry(None)
        return

    # Refresh mode
    if args.all:
        print("Refreshing tokens for all environments:")
        print("=" * 78)
        success_count = 0
        fail_count = 0
        
        for env in sorted(ENVIRONMENTS):
            try:
                new_token = refresh_token(env)
                update_env_token(new_token, env)
                expiry = decode_jwt_expiry(new_token)
                if expiry:
                    print(f"  → Expires at: {expiry.isoformat()}")
                success_count += 1
            except SystemExit as e:
                print(f"  ✗ Failed to refresh {env}: {e}")
                fail_count += 1
            print()
        
        print("=" * 78)
        print(f"Summary: {success_count} succeeded, {fail_count} failed")
    else:
        new_token = refresh_token(args.env)
        update_env_token(new_token, args.env)
        expiry = decode_jwt_expiry(new_token)
        if expiry:
            print(f"New token expires at: {expiry.isoformat()}")


if __name__ == "__main__":
    main()
