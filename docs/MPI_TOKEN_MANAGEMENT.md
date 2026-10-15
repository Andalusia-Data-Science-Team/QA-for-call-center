# MPI Per-Environment Token Management

This document explains the per-business-unit (per-environment) authentication token system for the MPI integration.

## Overview

Each business unit (AHJ, SNB, AKW, ALW, ERS, JDC) maintains its own authentication token for accessing the GetPatientByIdentificationValue endpoint. Tokens are automatically refreshed on first use and saved to the `.env` file for reuse.

## Environment Variables

### Token Storage

Each environment has its own token variable in `.env`:

```bash
# Environment-specific tokens (automatically managed)
MPI_AUTH_TOKEN_AHJ='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
MPI_AUTH_TOKEN_SNB='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
MPI_AUTH_TOKEN_AKW='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
MPI_AUTH_TOKEN_ALW='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
MPI_AUTH_TOKEN_ERS='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
MPI_AUTH_TOKEN_JDC='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'

# Generic fallback token (backward compatibility)
MPI_AUTH_TOKEN='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
```

### Configuration Variables

```bash
# Login credentials
MPI_SERVICE_USERNAME=AiModule
MPI_SERVICE_PASSWORD=123456
MPI_NTLM_USERNAME=andalusia\your.username
MPI_NTLM_PASSWORD=your_password

# Default environment
MPI_ENV=AHJ
```

## Usage

### 1. Initialize All Tokens (First Run)

Initialize authentication tokens for all environments at once:

```bash
cd app/services
python get_patient.py --init-tokens
```

This will:
- Connect to each environment's SecurityService
- Authenticate and retrieve a token
- Save each token to `.env` as `MPI_AUTH_TOKEN_<ENV>`
- Display the expiry time for each token

**Note**: You must be connected to the appropriate network/VPN for each environment you want to initialize.

### 2. Use the MPI Client API

The recommended way to get patient data is through the `mpi_client.py` module:

```python
from app.services.mpi_client import get_patient_by_id, MPIEnvironment

# Get patient from a specific environment
patient = get_patient_by_id("1120579634", MPIEnvironment.AHJ)

# Search all environments until found
patient = get_patient_by_id("1120579634", auto_search_all=True)

# Convenience functions
from app.services.mpi_client import get_patient_ahj, get_patient_snb

patient_ahj = get_patient_ahj("1120579634")
patient_snb = get_patient_snb("29901011200345")
```

The client automatically:
- Uses the appropriate environment-specific token
- Refreshes expired tokens
- Persists new tokens to `.env`
- Handles authentication errors gracefully

### 3. Manual Token Refresh

Refresh tokens manually when needed:

```bash
# Refresh a specific environment token
python refresh_dotcare_token.py --env AHJ

# Refresh all environment tokens
python refresh_dotcare_token.py --all

# Check token expiry without refreshing
python refresh_dotcare_token.py --check --env SNB

# Check all tokens
python refresh_dotcare_token.py --check --all
```

### 4. Run Tests

Test patient lookup for one or all environments:

```bash
# Test a specific environment
python get_patient.py --env AHJ --id 1120579634

# Test all environments
python get_patient.py --all

# Test using cached tokens (no login)
python get_patient.py --env SNB --no-login
```

## Token Lifecycle

### Automatic Token Management

1. **First Request**: When you query an environment for the first time:
   - If no token exists for that environment, the system automatically logs in
   - The new token is saved to `.env` as `MPI_AUTH_TOKEN_<ENV>`
   - The token is used for the request

2. **Subsequent Requests**:
   - The system reads the saved token from `.env`
   - If the request returns 401 Unauthorized, the token is automatically refreshed
   - The new token replaces the old one in `.env`

3. **Expiry Handling**:
   - Tokens are typically valid for extended periods (days to months)
   - Check expiry with `refresh_dotcare_token.py --check --env <ENV>`
   - Refresh before expiry with `refresh_dotcare_token.py --env <ENV>`

### Token Fallback

The system provides backward compatibility:

1. First tries environment-specific token: `MPI_AUTH_TOKEN_<ENV>`
2. Falls back to generic token: `MPI_AUTH_TOKEN`
3. If neither exists, automatically logs in and saves the new token

## API Reference

### mpi_client.py

#### Functions

**`get_patient_by_id(identification_value, env=None, auto_search_all=False, timeout=30)`**
- Get patient by ID from one or all environments
- `identification_value`: 10-digit (KSA) or 14-digit (Egyptian) ID
- `env`: MPIEnvironment enum or string ('AHJ', 'SNB', etc.)
- `auto_search_all`: If True, searches all environments until found
- Returns: Patient dict or None

**`get_patient_by_id_safe(identification_value, env=None, auto_search_all=False)`**
- Safe version that never raises exceptions
- Returns None on any error

**`validate_patient_id(identification_value)`**
- Validates ID format without querying MPI
- Returns: True if valid, False otherwise

**Environment-specific shortcuts:**
- `get_patient_ahj(id)`
- `get_patient_snb(id)`
- `get_patient_akw(id)`
- `get_patient_alw(id)`
- `get_patient_ers(id)`
- `get_patient_jdc(id)`

#### Exceptions

- `MPIError`: Base exception
- `MPIAuthenticationError`: Login failed
- `MPIValidationError`: Invalid ID format
- `MPINotFoundError`: Patient not found

### get_patient.py

#### Command-line Options

```
--env <ENV>          Test specific environment (AHJ, SNB, AKW, ALW, ERS, JDC)
--all                Test all environments
--id <ID>            Identification value to lookup
--init-tokens        Initialize tokens for all environments
--no-login           Use cached tokens, don't login
--username <USER>    Service username (default: AiModule)
--password <PASS>    Service password (default: 123456)
```

### refresh_dotcare_token.py

#### Command-line Options

```
--env <ENV>          Refresh specific environment token
--all                Refresh all environment tokens
--check              Check token expiry without refreshing
```

## Best Practices

1. **Initial Setup**: Run `--init-tokens` once during deployment to populate all tokens

2. **Development**: Use `mpi_client.py` API rather than direct HTTP requests

3. **Production**: 
   - Set up token refresh as a scheduled task (e.g., weekly)
   - Monitor token expiry with `--check` flags
   - Use environment-specific functions when you know the target environment

4. **Error Handling**: Use `get_patient_by_id_safe()` for non-critical lookups

5. **Multi-Environment Search**: Use `auto_search_all=True` when patient location is unknown

## Troubleshooting

### "DNS resolution failed"
- **Cause**: Not connected to the environment's network/VPN
- **Solution**: Connect to the appropriate VPN before initializing tokens

### "Login request failed: 401"
- **Cause**: Invalid credentials or NTLM authentication required
- **Solution**: Check `MPI_NTLM_USERNAME` and `MPI_NTLM_PASSWORD` in `.env`

### "Token expired"
- **Cause**: Saved token has expired
- **Solution**: Run `refresh_dotcare_token.py --env <ENV>` or let the system auto-refresh

### "Patient not found"
- **Cause**: Patient doesn't exist in the queried environment
- **Solution**: Try `auto_search_all=True` to search all environments

## Migration from Single Token

If you're upgrading from the single `MPI_AUTH_TOKEN` system:

1. Keep your existing `MPI_AUTH_TOKEN` in `.env` (backward compatibility)
2. Run `python get_patient.py --init-tokens` to populate per-environment tokens
3. Update your code to use `mpi_client.py` API
4. Existing code continues to work using the fallback token

## Example Integration

```python
# In your application code
from app.services.mpi_client import (
    get_patient_by_id,
    MPIEnvironment,
    MPIError,
    validate_patient_id,
)

def lookup_patient_safely(national_id: str, business_unit: str = None):
    """Lookup patient with proper validation and error handling."""
    
    # Validate format first
    if not validate_patient_id(national_id):
        return {
            "error": "Invalid ID format",
            "message": "ID must be 10 or 14 digits"
        }
    
    try:
        # Try specific environment if known
        if business_unit:
            env = MPIEnvironment[business_unit.upper()]
            patient = get_patient_by_id(national_id, env)
        else:
            # Search all environments
            patient = get_patient_by_id(national_id, auto_search_all=True)
        
        if patient:
            return {
                "success": True,
                "patient": patient,
                "environment": patient.get("_mpi_environment")
            }
        else:
            return {
                "error": "Patient not found",
                "searched": "all environments" if not business_unit else business_unit
            }
    
    except MPIError as e:
        return {
            "error": "MPI service error",
            "message": str(e)
        }
```

## Security Notes

1. **Token Storage**: Tokens are stored in `.env` which should be in `.gitignore`
2. **Token Rotation**: Tokens have long expiry but should be refreshed periodically
3. **Access Control**: Each environment token has its own permissions
4. **Credentials**: Never commit `MPI_NTLM_USERNAME` and `MPI_NTLM_PASSWORD` to version control

## Support

For issues or questions:
1. Check token expiry: `python refresh_dotcare_token.py --check --all`
2. Review logs for authentication errors
3. Verify network connectivity to the environment's host
4. Confirm credentials in `.env` are correct
