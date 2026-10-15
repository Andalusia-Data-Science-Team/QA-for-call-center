"""MPI Client - High-level interface for patient lookup across business units.

This module provides a simple API for getting patient information from any
MPI environment (business unit) with automatic token management and refresh.

Usage:
    from app.services.mpi_client import get_patient_by_id, MPIEnvironment
    
    # Get patient from AHJ environment
    patient = get_patient_by_id("1120579634", MPIEnvironment.AHJ)
    
    # Get patient from SNB environment
    patient = get_patient_by_id("29901011200345", MPIEnvironment.SNB)
    
    # Auto-detect environment and search
    patient = get_patient_by_id("1120579634")  # searches all environments

The module handles:
- Per-environment authentication tokens (MPI_AUTH_TOKEN_<ENV>)
- Automatic token refresh when needed
- Token persistence to .env
- Validation of identification values
- Graceful fallback across environments
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# Import functions from get_patient.py
try:
    from .get_patient import (
        ENVIRONMENTS,
        build_patient_url,
        build_headers,
        get_token_for_env,
        login,
        save_token_to_env,
        validate_identification_value,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from get_patient import (
        ENVIRONMENTS,
        build_patient_url,
        build_headers,
        get_token_for_env,
        login,
        save_token_to_env,
        validate_identification_value,
    )


class MPIEnvironment(str, Enum):
    """MPI Environment / Business Unit identifiers."""
    AHJ = "AHJ"
    SNB = "SNB"
    AKW = "AKW"
    ALW = "ALW"
    ERS = "ERS"
    JDC = "JDC"


class MPIError(Exception):
    """Base exception for MPI-related errors."""
    pass


class MPIAuthenticationError(MPIError):
    """Authentication with MPI service failed."""
    pass


class MPIValidationError(MPIError):
    """Invalid identification value provided."""
    pass


class MPINotFoundError(MPIError):
    """Patient not found in the MPI system."""
    pass


# Project root (app/services -> app -> project root); .env lives there.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables once
load_dotenv(ENV_FILE)


def _get_or_refresh_token(env: str, force_refresh: bool = False) -> str:
    """Get a valid token for an environment, refreshing if necessary.
    
    Args:
        env: Environment name (AHJ, SNB, etc.)
        force_refresh: If True, always fetch a new token
        
    Returns:
        Valid authentication token
        
    Raises:
        MPIAuthenticationError: If login fails
    """
    if not force_refresh:
        token = get_token_for_env(env)
        if token:
            return token
    
    # Need to refresh - get credentials
    username = os.getenv("MPI_SERVICE_USERNAME", "AiModule")
    password = os.getenv("MPI_SERVICE_PASSWORD", "123456")
    ntlm_username = os.getenv("MPI_NTLM_USERNAME")
    ntlm_password = os.getenv("MPI_NTLM_PASSWORD")
    
    try:
        token = login(env, username, password, ntlm_username, ntlm_password)
        save_token_to_env(env, token)
        return token
    except Exception as error:
        raise MPIAuthenticationError(
            f"Failed to authenticate with {env} environment: {error}"
        ) from error


def get_patient_by_id(
    identification_value: str,
    env: MPIEnvironment | str | None = None,
    auto_search_all: bool = False,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Get patient information by identification value (National ID / Iqama).
    
    Args:
        identification_value: Patient's national ID or Iqama number
            - 14 digits: Egyptian National ID
            - 10 digits: KSA National ID / Iqama
        env: Specific environment to query. If None and auto_search_all is False,
            uses the default environment (AHJ). If None and auto_search_all is True,
            searches all environments until a match is found.
        auto_search_all: If True and env is None, automatically search all
            environments until patient is found.
        timeout: Request timeout in seconds
        
    Returns:
        Patient data dictionary if found, None if not found
        
    Raises:
        MPIValidationError: Invalid identification value format
        MPIAuthenticationError: Authentication failed
        MPIError: Other MPI service errors
    """
    # Validate identification value
    try:
        validated_id = validate_identification_value(identification_value)
        if validated_id is None:
            raise MPIValidationError("Identification value cannot be empty")
    except ValueError as error:
        raise MPIValidationError(str(error)) from error
    
    # Determine which environment(s) to query
    if env is not None:
        env_str = env.value if isinstance(env, MPIEnvironment) else env
        environments = [env_str]
    elif auto_search_all:
        environments = list(ENVIRONMENTS.keys())
    else:
        environments = [os.getenv("MPI_ENV", "AHJ")]
    
    last_error = None
    
    for env_name in environments:
        try:
            # Get or refresh token
            token = _get_or_refresh_token(env_name)
            
            # Make the patient lookup request
            url = build_patient_url(env_name)
            response = requests.get(
                url,
                params={"identificationValue": validated_id},
                headers=build_headers(token),
                timeout=timeout,
            )
            
            # Handle 401 by refreshing token once
            if response.status_code == 401:
                token = _get_or_refresh_token(env_name, force_refresh=True)
                response = requests.get(
                    url,
                    params={"identificationValue": validated_id},
                    headers=build_headers(token),
                    timeout=timeout,
                )
            
            # Check response
            if response.status_code == 200:
                body = response.text.strip()
                if body and body.lower() != "null":
                    try:
                        patient_data = response.json()
                        if isinstance(patient_data, dict):
                            # Add metadata about which environment found the patient
                            patient_data["_mpi_environment"] = env_name
                            return patient_data
                    except ValueError:
                        pass
                # 200 with null/empty body means not found in this environment
                continue
            elif response.status_code == 400:
                # Shouldn't happen due to validation, but handle anyway
                raise MPIValidationError(
                    f"Server rejected identification value: {response.text[:200]}"
                )
            else:
                response.raise_for_status()
                
        except MPIValidationError:
            # Re-raise validation errors immediately
            raise
        except MPIAuthenticationError as error:
            # Save auth error but continue to next environment
            last_error = error
            continue
        except requests.RequestException as error:
            # Save network error but continue to next environment
            last_error = MPIError(f"Request to {env_name} failed: {error}")
            continue
    
    # If we got here, patient was not found in any queried environment
    if last_error:
        raise last_error
    
    return None


def get_patient_by_id_safe(
    identification_value: str,
    env: MPIEnvironment | str | None = None,
    auto_search_all: bool = False,
) -> dict[str, Any] | None:
    """Safe version of get_patient_by_id that never raises exceptions.
    
    Returns None for any error condition and logs the error message.
    Useful for non-critical patient lookups where you want graceful degradation.
    """
    try:
        return get_patient_by_id(identification_value, env, auto_search_all)
    except Exception as error:
        print(f"MPI lookup failed for {identification_value}: {error}")
        return None


def validate_patient_id(identification_value: str) -> bool:
    """Check if an identification value is valid format without querying MPI.
    
    Args:
        identification_value: Value to validate
        
    Returns:
        True if valid format, False otherwise
    """
    try:
        validated = validate_identification_value(identification_value)
        return validated is not None
    except ValueError:
        return False


# Convenience functions for specific environments
def get_patient_ahj(identification_value: str) -> dict[str, Any] | None:
    """Get patient from AHJ environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.AHJ)


def get_patient_snb(identification_value: str) -> dict[str, Any] | None:
    """Get patient from SNB environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.SNB)


def get_patient_akw(identification_value: str) -> dict[str, Any] | None:
    """Get patient from AKW environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.AKW)


def get_patient_alw(identification_value: str) -> dict[str, Any] | None:
    """Get patient from ALW environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.ALW)


def get_patient_ers(identification_value: str) -> dict[str, Any] | None:
    """Get patient from ERS environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.ERS)


def get_patient_jdc(identification_value: str) -> dict[str, Any] | None:
    """Get patient from JDC environment."""
    return get_patient_by_id(identification_value, MPIEnvironment.JDC)


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mpi_client.py <identification_value> [environment]")
        print("Example: python mpi_client.py 1120579634 AHJ")
        sys.exit(1)
    
    patient_id = sys.argv[1]
    env_arg = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        if env_arg:
            print(f"Searching for patient {patient_id} in {env_arg}...")
            patient = get_patient_by_id(patient_id, env_arg)
        else:
            print(f"Searching for patient {patient_id} across all environments...")
            patient = get_patient_by_id(patient_id, auto_search_all=True)
        
        if patient:
            import json
            print("\nPatient found:")
            print(json.dumps(patient, indent=2, ensure_ascii=False))
        else:
            print("\nPatient not found in any environment")
    except MPIError as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)
