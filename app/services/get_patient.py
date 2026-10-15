"""MPI patient lookup client for the AIAgentIntegration.Service endpoints.

Every endpoint, environment and validation rule below is taken from
"AIAgentIntegration.Service Endpoint Documentation.md".

Endpoints
---------
Security Service (port 7000) — Section 3:
    GET /SercurityService/api/security/login?username=<user>&password=<pass>

AIAgentIntegration.Service (port 8090) — Section 4:
    GET /AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue
        ?identificationValue=<national id>

Environments (Section 2): AHJ, SNB, AKW, ALW, ERS.

Validation rules (Section 4)
----------------------------
    exactly 14 digits   Egyptian National ID
    exactly 10 digits   KSA National ID / Iqama
    empty / whitespace  200 OK with a null body
    anything else       400 Bad Request

Usage
-----
    python get_patient.py               # documentation test suite against AHJ
    python get_patient.py --env SNB     # ... against a single environment
    python get_patient.py --all         # ... against all five environments
    python get_patient.py --id 29901011200345
    python get_patient.py --no-login    # reuse MPI_AUTH_TOKEN instead of logging in

Configuration (.env): MPI_ENV, MPI_IDENTIFICATION_VALUE, MPI_AUTH_TOKEN,
MPI_SERVICE_USERNAME, MPI_SERVICE_PASSWORD, MPI_NTLM_USERNAME,
MPI_NTLM_PASSWORD. URLs are derived from the Section 2 environment table;
optional MPI_API_URL_<ENV> / MPI_LOGIN_URL_<ENV> entries in .env override
them per environment (all ten documented endpoints are listed there).
"""

import argparse
import json
import os
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Endpoints (AIAgentIntegration.Service Endpoint Documentation.md) ──────────

SECURITY_SERVICE_PORT = 7000
AI_SERVICE_PORT = 8090

LOGIN_PATH = "/SercurityService/api/security/login"
PATIENT_LOOKUP_PATH = (
    "/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue"
)

# Section 2 — one HMIS host per deployed environment.
ENVIRONMENTS = {
    "AHJ": "ahj-hmis",
    "SNB": "snb-hmisapp-01",
    "AKW": "akw-hmis",
    "ALW": "alw-hmis",
    #"ERS": "ers-fwzhmisapp",
    "JDC": "jdc-hmis",
}

# Section 4 — identificationValue validation rules.
EGYPTIAN_ID_DIGITS = 14  # Egyptian National ID
KSA_ID_DIGITS = 10  # KSA National ID / Iqama
_DIGITS_RE = re.compile(r"\A\d+\Z")

# Section 4 example request: ?identificationValue=9876541231
DOC_EXAMPLE_IDENTIFICATION_VALUE = "9876541231"

DEFAULT_ENV = "AHJ"
DEFAULT_SERVICE_USERNAME = "AiModule"
DEFAULT_SERVICE_PASSWORD = "123456"
DEFAULT_TIMEOUT_SECONDS = 30

# QA outcome states: PASS matches the documentation, MISMATCH means the
# server answered but differently than documented (a QA finding to report),
# and FAIL means the request could not be completed at all.
PASS = "PASS"
MISMATCH = "MISMATCH"
FAIL = "FAIL"

# Project root (app/services -> app -> project root); .env lives there.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def get_token_for_env(env: str) -> str | None:
    """Get the saved authentication token for a specific environment from .env.
    
    Returns the token from MPI_AUTH_TOKEN_<ENV> (e.g., MPI_AUTH_TOKEN_AHJ).
    Falls back to the generic MPI_AUTH_TOKEN if environment-specific token is not set.
    """
    env_specific_token = os.getenv(f"MPI_AUTH_TOKEN_{env}")
    if env_specific_token:
        return env_specific_token
    # Fallback to generic token for backward compatibility
    return os.getenv("MPI_AUTH_TOKEN")


def save_token_to_env(env: str, token: str) -> None:
    """Save an authentication token for a specific environment to the .env file.
    
    Creates or updates the MPI_AUTH_TOKEN_<ENV> variable in .env.
    This allows each business unit to maintain its own refresh token.
    """
    if not ENV_FILE.exists():
        print(f"Warning: .env file not found at {ENV_FILE}")
        return
    
    # Read the current .env content
    env_content = ENV_FILE.read_text()
    env_var_name = f"MPI_AUTH_TOKEN_{env}"
    token_line = f"{env_var_name}='{token}'\n"
    
    # Check if the environment-specific token already exists
    pattern = re.compile(rf"^{env_var_name}=.*$", re.MULTILINE)
    
    if pattern.search(env_content):
        # Replace existing token
        new_content = pattern.sub(token_line.rstrip(), env_content)
    else:
        # Add new token after the MPI section
        # Find a good insertion point (after MPI_AUTH_TOKEN or at the end of MPI section)
        mpi_token_pattern = re.compile(r"(^MPI_AUTH_TOKEN=.*$)", re.MULTILINE)
        match = mpi_token_pattern.search(env_content)
        
        if match:
            # Insert after the generic MPI_AUTH_TOKEN
            insert_pos = match.end()
            new_content = (
                env_content[:insert_pos] + 
                "\n" + token_line.rstrip() +
                env_content[insert_pos:]
            )
        else:
            # Append at the end
            new_content = env_content.rstrip() + "\n\n" + token_line
    
    # Write back to .env
    ENV_FILE.write_text(new_content)
    print(f"  → Token saved to .env as {env_var_name}")


def build_login_url(env: str) -> str:
    """Section 3 — Security Service login URL for an environment.

    Honors an optional MPI_LOGIN_URL_<ENV> override from .env (useful for jump
    hosts or alternate DNS names); otherwise the URL is derived from the
    Section 2 host table.
    """
    override = os.getenv(f"MPI_LOGIN_URL_{env}")
    if override:
        return override
    return f"http://{ENVIRONMENTS[env]}:{SECURITY_SERVICE_PORT}{LOGIN_PATH}"


def build_patient_url(env: str) -> str:
    """Section 4 — GetPatientByIdentificationValue URL for an environment.

    Honors an optional MPI_API_URL_<ENV> override from .env; otherwise the URL
    is derived from the Section 2 host table.
    """
    override = os.getenv(f"MPI_API_URL_{env}")
    if override:
        return override
    return f"http://{ENVIRONMENTS[env]}:{AI_SERVICE_PORT}{PATIENT_LOOKUP_PATH}"


def validate_identification_value(value: str | None) -> str | None:
    """Mirror IdentificationValueValidator.cs (Section 4).

    Returns the trimmed value, or None when the input is empty — the server
    answers 200 OK with a null body in that case. Raises ValueError for any
    other format — the server answers 400 Bad Request.
    """
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if _DIGITS_RE.match(trimmed) and len(trimmed) in (EGYPTIAN_ID_DIGITS, KSA_ID_DIGITS):
        return trimmed
    raise ValueError(
        f"identificationValue must be exactly {EGYPTIAN_ID_DIGITS} digits "
        f"(Egyptian National ID) or {KSA_ID_DIGITS} digits (KSA National ID / "
        f"Iqama); got {trimmed!r}"
    )


def build_headers(token: str | None) -> dict[str, str]:
    """Browser-like headers the HMIS stack expects, plus the auth token."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    headers = {
        "Referer": (
            "http://dev-testing.andalusiagroup.net:5004/"
            "finance/receivable/opd-invoice-list"
        ),
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36 Edg/153.0.0.0"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json; charset=utf-8",
        "x-request-timestamp": timestamp,
    }
    if token:
        headers["Authentication"] = token
    return headers


def login(
    env: str,
    username: str,
    password: str,
    ntlm_username: str | None = None,
    ntlm_password: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Section 3 — authenticate against the Security Service, return the token."""
    url = build_login_url(env)
    params = {"username": username, "password": password}

    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code == 401 and ntlm_username and ntlm_password:
        # Corporate endpoints can insist on NTLM; retry with domain credentials.
        from requests_ntlm import HttpNtlmAuth

        response = requests.get(
            url,
            params=params,
            auth=HttpNtlmAuth(ntlm_username, ntlm_password),
            timeout=timeout,
        )
    response.raise_for_status()

    payload = response.json()
    token = payload.get("token") if isinstance(payload, dict) else None
    if isinstance(token, dict):
        # Some deployments nest the JWT under token.Value.
        token = token.get("Value")
    if not isinstance(token, str) or not token:
        raise ValueError(f"login response has no usable 'token' field: {str(payload)[:200]}")
    return token


def get_patient(
    env: str,
    identification_value: str | None,
    token: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> requests.Response:
    """Section 4 — call GetPatientByIdentificationValue, return the raw response."""
    return requests.get(
        build_patient_url(env),
        params={"identificationValue": identification_value},
        headers=build_headers(token),
        timeout=timeout,
    )


def run_environment_suite(env: str, args: argparse.Namespace) -> list[str]:
    """Run every documented example request against one environment.

    Returns per-step outcomes: PASS (matches the documentation), MISMATCH
    (server answered, but differently than documented — a QA finding), or
    FAIL (the request could not be completed at all).
    """
    host = ENVIRONMENTS[env]
    print()
    print("═" * 78)
    print(f" Environment {env} — {host}")
    print(f"   login  → {build_login_url(env)}")
    print(f"   lookup → {build_patient_url(env)}")
    print("═" * 78)

    outcomes: list[str] = []

    def check(step: str, status: str, detail: str = "") -> None:
        outcomes.append(status)
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {step}{suffix}")

    # [1] The host must resolve before any request can be sent.
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        check(f"DNS resolution of '{host}'", FAIL, f"{error} — connect to the {env} network/VPN")
        return outcomes
    check(f"DNS resolution of '{host}'", PASS)

    # [2] Section 3 — SecurityService login (port 7000).
    token: str | None = None
    if args.no_login:
        token = get_token_for_env(env)
        check(
            "SecurityService login (skipped: --no-login)",
            PASS if token else FAIL,
            f"reusing MPI_AUTH_TOKEN_{env}" if token else f"MPI_AUTH_TOKEN_{env} is not set",
        )
    else:
        try:
            token = login(
                env,
                args.username,
                args.password,
                ntlm_username=os.getenv("MPI_NTLM_USERNAME"),
                ntlm_password=os.getenv("MPI_NTLM_PASSWORD"),
            )
            check("SecurityService login (port 7000)", PASS, f"token {token[:24]}…")
            # Save the token to .env for future use
            save_token_to_env(env, token)
        except (requests.RequestException, ValueError) as error:
            token = get_token_for_env(env)
            check(
                "SecurityService login (port 7000)",
                FAIL,
                str(error) + (f" — falling back to MPI_AUTH_TOKEN_{env}" if token else ""),
            )

    # [3] Section 4 — GetPatientByIdentificationValue (port 8090).
    #     Each case: (label, identificationValue, documented status codes).
    #     Valid IDs: 200 with a PatientDto JSON body or a null body (Section 4).
    cases: list[tuple[str, str, tuple[int, ...]]] = [
        (
            f"primary lookup (?identificationValue={args.identification_value})",
            args.identification_value,
            (200,),
        ),
        # Known patient from app/services/patient_Data.json — exercises the
        # full PatientDto contract (Section 5) on AHJ.
        ("known KSA ID 1120579634 → 200 PatientDto", "1120579634", (200,)),
        ("empty value → 200 with null body", "", (200,)),
        ("letters → 400", "abc12345", (400,)),
        ("wrong length → 400", "12345", (400,)),
        ("14-digit Egyptian ID → 200", "29901011200345", (200,)),
    ]

    primary_patient: dict | None = None
    for label, value, expected in cases:
        try:
            response = get_patient(env, value, token=token)
        except requests.RequestException as error:
            check(f"lookup: {label}", FAIL, str(error))
            continue

        body = response.text.strip()
        status = response.status_code
        if status in expected:
            result = PASS
        else:
            # The server answered definitively, just not as documented —
            # e.g. 204 No Content where the doc promises 200 OK + null.
            result = MISMATCH

        detail = f"HTTP {status}"
        if status == 400:
            detail += f": {body[:120]}"
        elif status in (200, 204):
            if not body or body.lower() == "null":
                detail += " — null/empty body (no matching patient)"
            else:
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    detail += (
                        f" — patient {payload.get('Code')} "
                        f"{payload.get('EnFirstName', '')} {payload.get('EnLastName', '')}"
                        f" (ID {payload.get('ID')})"
                    )
                    if label.startswith("primary"):
                        primary_patient = payload
                else:
                    detail += f" — non-JSON body: {body[:120]}"
        if result == MISMATCH:
            detail += f" [doc says HTTP {expected[0]}]"
        check(f"lookup: {label}", result, detail)

    # [4] Section 5 — PatientDto response contract on the primary lookup.
    if primary_patient is not None:
        required = (
            "ID",
            "Code",
            "EnFirstName",
            "EnLastName",
            "IdentificationValue",
            "EntityID",
            "HealthId",
        )
        missing = [field for field in required if field not in primary_patient]
        check(
            "PatientDto contract (Section 5)",
            PASS if not missing else FAIL,
            "all core fields present" if not missing else f"missing {missing}",
        )
        print(json.dumps(primary_patient, indent=2, ensure_ascii=False))

    return outcomes


def initialize_all_tokens(
    username: str,
    password: str,
    ntlm_username: str | None = None,
    ntlm_password: str | None = None,
) -> None:
    """Initialize authentication tokens for all environments.
    
    Logs in to each environment's SecurityService and saves the token to .env.
    This is useful for first-time setup or when all tokens need refreshing.
    """
    print()
    print("═" * 78)
    print(" Initializing tokens for all environments")
    print("═" * 78)
    
    success_count = 0
    fail_count = 0
    
    for env in sorted(ENVIRONMENTS):
        host = ENVIRONMENTS[env]
        try:
            # Check DNS resolution first
            socket.getaddrinfo(host, None)
        except socket.gaierror as error:
            print(f"  [{env}] SKIP - DNS resolution failed: {error}")
            print(f"         Connect to the {env} network/VPN to initialize this token")
            fail_count += 1
            continue
        
        try:
            token = login(env, username, password, ntlm_username, ntlm_password)
            save_token_to_env(env, token)
            print(f"  [{env}] SUCCESS - Token saved as MPI_AUTH_TOKEN_{env}")
            print(f"         Token preview: {token[:24]}...")
            success_count += 1
        except (requests.RequestException, ValueError) as error:
            print(f"  [{env}] FAIL - {error}")
            fail_count += 1
    
    print()
    print("═" * 78)
    print(f" Token initialization complete: {success_count} succeeded, {fail_count} failed")
    print("═" * 78)
    
    if fail_count > 0:
        print()
        print("Note: Failed environments may require VPN access or correct credentials.")
        print("      You can retry initialization later by running with --init-tokens again.")


def main() -> None:
    load_dotenv(ENV_FILE)

    parser = argparse.ArgumentParser(
        description=(
            "Test the MPI endpoints documented in "
            "'AIAgentIntegration.Service Endpoint Documentation.md': the "
            "SecurityService login (port 7000) and GetPatientByIdentificationValue "
            "(port 8090)."
        )
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENVIRONMENTS),
        default=os.getenv("MPI_ENV", DEFAULT_ENV),
        help="environment to test (default: %(default)s)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="test every environment from Section 2 of the documentation",
    )
    parser.add_argument(
        "--id",
        "--identification-value",
        dest="identification_value",
        default=os.getenv("MPI_IDENTIFICATION_VALUE", DOC_EXAMPLE_IDENTIFICATION_VALUE),
        help="identificationValue for the primary Section 4 lookup (default: %(default)s)",
    )
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="skip the Section 3 login and reuse MPI_AUTH_TOKEN from .env",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("MPI_SERVICE_USERNAME", DEFAULT_SERVICE_USERNAME),
        help="SecurityService login username (default: %(default)s)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("MPI_SERVICE_PASSWORD", DEFAULT_SERVICE_PASSWORD),
        help="SecurityService login password",
    )
    parser.add_argument(
        "--init-tokens",
        action="store_true",
        help="initialize authentication tokens for all environments and save to .env",
    )
    args = parser.parse_args()

    # Handle token initialization mode
    if args.init_tokens:
        initialize_all_tokens(
            args.username,
            args.password,
            ntlm_username=os.getenv("MPI_NTLM_USERNAME"),
            ntlm_password=os.getenv("MPI_NTLM_PASSWORD"),
        )
        return

    environments = sorted(ENVIRONMENTS) if args.all else [args.env]

    print(
        "MPI endpoint test suite — "
        "AIAgentIntegration.Service Endpoint Documentation.md"
    )
    outcomes: list[str] = []
    for env in environments:
        outcomes.extend(run_environment_suite(env, args))

    passed = outcomes.count(PASS)
    mismatches = outcomes.count(MISMATCH)
    failed = outcomes.count(FAIL)
    print()
    print("═" * 78)
    print(
        f" SUMMARY: {passed} passed, {mismatches} mismatched, "
        f"{failed} failed, {len(outcomes)} total"
    )
    if mismatches:
        print(
            " MISMATCH = server answered but not as documented — "
            "report these findings to the backend team."
        )
    print("═" * 78)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
