# Per-Environment Token Management Implementation Summary

## What Was Changed

This implementation adds per-business-unit (per-environment) authentication token management for the MPI integration system.

### Files Modified

1. **`get_patient.py`** - Core patient lookup script
   - Added `get_token_for_env(env)` - Retrieves environment-specific token from .env
   - Added `save_token_to_env(env, token)` - Saves token to .env as `MPI_AUTH_TOKEN_<ENV>`
   - Added `initialize_all_tokens()` - Batch token initialization for all environments
   - Updated `run_environment_suite()` - Uses per-environment tokens automatically
   - Added `--init-tokens` CLI flag - Initialize all tokens at once

2. **`refresh_dotcare_token.py`** - Token refresh utility
   - Added `build_login_url(env)` - Per-environment login URL construction
   - Updated `refresh_token(env)` - Supports environment-specific refresh
   - Updated `update_env_token(token, env)` - Saves per-environment tokens
   - Added `check_token_expiry(env)` - Check expiry for specific environments
   - Added `--env` flag - Target specific environment
   - Added `--all` flag - Operate on all environments

3. **`.env`** - Environment configuration
   - Added `MPI_AUTH_TOKEN_AHJ` - AHJ environment token
   - Added `MPI_AUTH_TOKEN_SNB` - SNB environment token
   - Added `MPI_AUTH_TOKEN_AKW` - AKW environment token
   - Added `MPI_AUTH_TOKEN_ALW` - ALW environment token
   - Added `MPI_AUTH_TOKEN_ERS` - ERS environment token
   - Added `MPI_AUTH_TOKEN_JDC` - JDC environment token
   - Kept `MPI_AUTH_TOKEN` - Generic fallback for backward compatibility

### Files Created

1. **`mpi_client.py`** - High-level Python API
   - `get_patient_by_id()` - Main patient lookup function with auto token management
   - `get_patient_by_id_safe()` - Exception-safe version
   - `validate_patient_id()` - ID format validation
   - Environment-specific shortcuts: `get_patient_ahj()`, `get_patient_snb()`, etc.
   - `MPIEnvironment` enum - Type-safe environment identifiers
   - Custom exceptions: `MPIError`, `MPIAuthenticationError`, `MPIValidationError`

2. **`MPI_TOKEN_MANAGEMENT.md`** - Comprehensive documentation
   - System overview and architecture
   - Usage examples and best practices
   - API reference for all functions
   - Troubleshooting guide
   - Migration guide from single-token system

3. **`IMPLEMENTATION_SUMMARY.md`** - This file

## Key Features

### 1. Automatic Token Management
- First request to an environment automatically logs in and saves the token
- Tokens are reused from .env on subsequent requests
- Expired tokens are automatically refreshed on 401 responses
- New tokens are immediately persisted to .env

### 2. Per-Environment Isolation
- Each business unit (AHJ, SNB, AKW, ALW, ERS, JDC) has its own token
- Tokens are independent and don't interfere with each other
- Environment-specific tokens follow naming convention: `MPI_AUTH_TOKEN_<ENV>`

### 3. Backward Compatibility
- Existing `MPI_AUTH_TOKEN` still works as a fallback
- Old code continues to function without changes
- Graceful migration path for existing deployments

### 4. Batch Operations
- Initialize all environment tokens with one command
- Refresh all tokens at once
- Check expiry status for all environments

### 5. Developer-Friendly API
- Simple Python API: `get_patient_by_id(id, env)`
- Auto-search across all environments: `get_patient_by_id(id, auto_search_all=True)`
- Type-safe enums for environment selection
- Exception-safe variant for graceful degradation

## Usage Examples

### Initial Setup

```bash
# Initialize tokens for all environments (first run)
cd app/services
python3 get_patient.py --init-tokens
```

### Python API Usage

```python
from app.services.mpi_client import get_patient_by_id, MPIEnvironment

# Query specific environment
patient = get_patient_by_id("1120579634", MPIEnvironment.AHJ)

# Search all environments
patient = get_patient_by_id("1120579634", auto_search_all=True)

# Safe version (never raises exceptions)
from app.services.mpi_client import get_patient_by_id_safe
patient = get_patient_by_id_safe("1120579634")
```

### Token Management

```bash
# Check specific environment token expiry
python3 refresh_dotcare_token.py --check --env AHJ

# Check all tokens
python3 refresh_dotcare_token.py --check --all

# Refresh specific environment
python3 refresh_dotcare_token.py --env SNB

# Refresh all environments
python3 refresh_dotcare_token.py --all
```

### Testing

```bash
# Test specific environment
python3 get_patient.py --env AHJ --id 1120579634

# Test all environments
python3 get_patient.py --all

# Test with cached tokens (no login)
python3 get_patient.py --env SNB --no-login --id 1120579634
```

## Token Lifecycle

1. **First Request**
   - System checks for `MPI_AUTH_TOKEN_<ENV>` in .env
   - If not found, automatically logs in to SecurityService
   - Saves new token to .env
   - Uses token for the request

2. **Subsequent Requests**
   - Reads token from .env
   - Uses cached token
   - If 401 response, refreshes token automatically

3. **Manual Refresh**
   - Run `refresh_dotcare_token.py --env <ENV>`
   - Or use `--all` to refresh all environments

4. **Expiry Monitoring**
   - Check with `refresh_dotcare_token.py --check --env <ENV>`
   - Tokens typically valid for days/months

## Integration Points

### Where Tokens Are Used

1. **GetPatientByIdentificationValue Endpoint**
   - Port 8090
   - Requires authentication token in `Authentication` header
   - Each environment has its own endpoint URL

2. **SecurityService Login Endpoint**
   - Port 7000
   - Returns JWT token for subsequent API calls
   - Requires NTLM authentication

### Environment Definitions

```python
ENVIRONMENTS = {
    "AHJ": "ahj-hmis",
    "SNB": "snb-hmisapp-01",
    "AKW": "akw-hmis",
    "ALW": "alw-hmis",
    "ERS": "ers-fwzhmisapp",
    "JDC": "jdc-hmis",
}
```

## Benefits

1. **Isolation**: Each environment's authentication is independent
2. **Reliability**: Token failures in one environment don't affect others
3. **Performance**: Cached tokens eliminate repeated logins
4. **Maintainability**: Clear separation of concerns
5. **Scalability**: Easy to add new environments
6. **Developer Experience**: Simple, intuitive API

## Next Steps

### For First-Time Setup

1. Ensure `.env` has correct credentials:
   ```bash
   MPI_SERVICE_USERNAME=AiModule
   MPI_SERVICE_PASSWORD=123456
   MPI_NTLM_USERNAME=andalusia\your.username
   MPI_NTLM_PASSWORD=your_password
   ```

2. Connect to appropriate VPN/network for each environment

3. Initialize all tokens:
   ```bash
   python3 get_patient.py --init-tokens
   ```

4. Verify tokens were saved:
   ```bash
   python3 refresh_dotcare_token.py --check --all
   ```

### For Existing Deployments

1. Keep existing `MPI_AUTH_TOKEN` (backward compatibility)
2. Run token initialization when ready
3. Update application code to use `mpi_client.py` API
4. Set up periodic token refresh (e.g., weekly cron job)

### For Monitoring

1. Create scheduled task to check token expiry
2. Alert when tokens are close to expiry
3. Auto-refresh tokens before expiry
4. Log all authentication attempts for audit trail

## Security Considerations

1. **Token Storage**: `.env` file should be in `.gitignore`
2. **Access Control**: File permissions on `.env` should be restricted
3. **Token Rotation**: Consider periodic manual refresh
4. **Credential Protection**: Never commit NTLM credentials to version control
5. **Network Security**: Use VPN for production environments

## Troubleshooting

### Common Issues

1. **"DNS resolution failed"**
   - Not connected to environment's network/VPN
   - Solution: Connect to VPN and retry

2. **"Login request failed: 401"**
   - Invalid credentials
   - Solution: Check `MPI_NTLM_USERNAME` and `MPI_NTLM_PASSWORD`

3. **"Token expired"**
   - Cached token is no longer valid
   - Solution: Will auto-refresh on next request

4. **"Patient not found"**
   - Patient doesn't exist in queried environment
   - Solution: Try `auto_search_all=True`

## Documentation References

- **`MPI_TOKEN_MANAGEMENT.md`** - Full user documentation
- **`get_patient.py`** - Implementation details and docstrings
- **`mpi_client.py`** - API reference and usage examples
- **`refresh_dotcare_token.py`** - Token refresh utility

## Testing

Run the test suite to verify functionality:

```bash
# Test AHJ environment
python3 get_patient.py --env AHJ --id 1120579634

# Test all environments with comprehensive test cases
python3 get_patient.py --all

# Test token initialization (requires VPN access)
python3 get_patient.py --init-tokens
```

## Version History

- **v2.0** - Per-environment token management (current)
  - Added environment-specific tokens
  - Created Python API client
  - Added batch token operations
  - Maintained backward compatibility

- **v1.0** - Single token system (legacy)
  - Used single `MPI_AUTH_TOKEN`
  - Manual token refresh only
  - No environment isolation
