# Quick Start Guide - Per-Environment MPI Tokens

## 5-Minute Setup

### Step 1: Configure Credentials (1 minute)

Edit `.env` in the project root and ensure these are set:

```bash
# Required credentials
MPI_SERVICE_USERNAME=AiModule
MPI_SERVICE_PASSWORD=123456
MPI_NTLM_USERNAME=andalusia\your.username
MPI_NTLM_PASSWORD=your_password
```

### Step 2: Initialize All Tokens (3 minutes)

Connect to VPN, then run:

```bash
cd /home/ai/Workspace/Rafik/QA_System-main/app/services
python3 get_patient.py --init-tokens
```

You should see:

```
═══════════════════════════════════════════════════════════════════════
 Initializing tokens for all environments
═══════════════════════════════════════════════════════════════════════
  [AHJ] SUCCESS - Token saved as MPI_AUTH_TOKEN_AHJ
         Token preview: eyJ0eXAiOiJKV1QiLCJhbGci...
  [SNB] SUCCESS - Token saved as MPI_AUTH_TOKEN_SNB
         Token preview: eyJ0eXAiOiJKV1QiLCJhbGci...
  ...
═══════════════════════════════════════════════════════════════════════
 Token initialization complete: 6 succeeded, 0 failed
═══════════════════════════════════════════════════════════════════════
```

### Step 3: Verify Setup (1 minute)

Check that tokens were saved:

```bash
python3 refresh_dotcare_token.py --check --all
```

You should see expiry information for each environment.

## Usage Examples

### In Python Code

```python
# Import the client
from app.services.mpi_client import get_patient_by_id, MPIEnvironment

# Get patient from specific environment
patient = get_patient_by_id("1120579634", MPIEnvironment.AHJ)

if patient:
    print(f"Found: {patient['EnFirstName']} {patient['EnLastName']}")
    print(f"Patient ID: {patient['ID']}")
    print(f"Code: {patient['Code']}")
```

### Search All Environments

```python
from app.services.mpi_client import get_patient_by_id

# Auto-search all environments until found
patient = get_patient_by_id("1120579634", auto_search_all=True)

if patient:
    env = patient.get("_mpi_environment")
    print(f"Found in {env} environment")
```

### Safe Lookup (Never Raises Exceptions)

```python
from app.services.mpi_client import get_patient_by_id_safe

# Returns None on any error
patient = get_patient_by_id_safe("1120579634")
```

### From Command Line

```bash
# Test specific environment
python3 get_patient.py --env AHJ --id 1120579634

# Test all environments
python3 get_patient.py --all --id 1120579634

# Use the client directly
python3 mpi_client.py 1120579634 AHJ
```

## Common Tasks

### Check Token Status

```bash
# Check specific environment
python3 refresh_dotcare_token.py --check --env AHJ

# Check all environments
python3 refresh_dotcare_token.py --check --all
```

### Refresh Expired Token

```bash
# Refresh specific environment
python3 refresh_dotcare_token.py --env SNB

# Refresh all environments
python3 refresh_dotcare_token.py --all
```

### Re-initialize After Network Issues

```bash
# Re-run initialization for all environments
python3 get_patient.py --init-tokens
```

## Integration Example

### Web API Endpoint

```python
from flask import Flask, jsonify, request
from app.services.mpi_client import (
    get_patient_by_id,
    MPIEnvironment,
    MPIError,
    validate_patient_id
)

app = Flask(__name__)

@app.route('/api/patient/<national_id>')
def get_patient_endpoint(national_id):
    """Get patient by national ID, searching all environments."""
    
    # Validate format
    if not validate_patient_id(national_id):
        return jsonify({
            "error": "Invalid ID format",
            "message": "ID must be 10 or 14 digits"
        }), 400
    
    try:
        # Search all environments
        patient = get_patient_by_id(national_id, auto_search_all=True)
        
        if patient:
            return jsonify({
                "success": True,
                "patient": patient,
                "environment": patient.get("_mpi_environment")
            })
        else:
            return jsonify({
                "success": False,
                "message": "Patient not found in any environment"
            }), 404
    
    except MPIError as e:
        return jsonify({
            "error": "MPI service error",
            "message": str(e)
        }), 500


@app.route('/api/patient/<national_id>/<env>')
def get_patient_from_env(national_id, env):
    """Get patient from specific environment."""
    
    if not validate_patient_id(national_id):
        return jsonify({
            "error": "Invalid ID format"
        }), 400
    
    try:
        env_enum = MPIEnvironment[env.upper()]
        patient = get_patient_by_id(national_id, env_enum)
        
        if patient:
            return jsonify({"success": True, "patient": patient})
        else:
            return jsonify({
                "success": False,
                "message": f"Patient not found in {env}"
            }), 404
    
    except KeyError:
        return jsonify({
            "error": "Invalid environment",
            "valid_environments": ["AHJ", "SNB", "AKW", "ALW", "ERS", "JDC"]
        }), 400
    except MPIError as e:
        return jsonify({"error": str(e)}), 500
```

### Background Job

```python
from app.services.mpi_client import get_patient_by_id_safe, MPIEnvironment

def sync_patient_data(patient_ids: list[str], environment: str):
    """Background job to sync patient data from MPI."""
    
    env = MPIEnvironment[environment.upper()]
    
    for patient_id in patient_ids:
        # Safe version never raises exceptions
        patient = get_patient_by_id_safe(patient_id, env)
        
        if patient:
            # Update local database
            update_local_patient(patient)
            print(f"✓ Synced {patient_id}")
        else:
            print(f"✗ Not found: {patient_id}")
```

## Troubleshooting

### Problem: "DNS resolution failed"

**Cause**: Not connected to the environment's network

**Solution**:
```bash
# Connect to VPN first, then retry
python3 get_patient.py --init-tokens
```

### Problem: "Login request failed: 401"

**Cause**: Invalid NTLM credentials

**Solution**: Check `.env` file:
```bash
# Ensure these are correct
MPI_NTLM_USERNAME=andalusia\your.username
MPI_NTLM_PASSWORD=your_password
```

### Problem: "Token expired"

**Cause**: Cached token is no longer valid

**Solution**: Token will auto-refresh, or manually refresh:
```bash
python3 refresh_dotcare_token.py --env AHJ
```

### Problem: "Patient not found"

**Cause**: Patient doesn't exist in queried environment

**Solution**: Try searching all environments:
```python
patient = get_patient_by_id(national_id, auto_search_all=True)
```

## Environment Reference

| Code | Name | Host | Notes |
|------|------|------|-------|
| AHJ | Al-Hayat (Jeddah) | ahj-hmis | Default environment |
| SNB | Saudi New Berlin | snb-hmisapp-01 | |
| AKW | Al-Khobar West | akw-hmis | |
| ALW | Al-Waha (Dammam) | alw-hmis | |
| ERS | Eradah (Riyadh) | ers-fwzhmisapp | |
| JDC | Jeddah Clinic | jdc-hmis | |

## ID Format Validation

- **10 digits**: KSA National ID or Iqama
  - Example: `1120579634`
- **14 digits**: Egyptian National ID
  - Example: `29901011200345`

Any other format is rejected before making an API call.

## Best Practices

1. **Initialize Once**: Run `--init-tokens` during deployment
2. **Use the API**: Import `mpi_client.py` instead of direct HTTP calls
3. **Handle Errors**: Use try/except or the `_safe` variant
4. **Monitor Tokens**: Check expiry weekly
5. **Log Lookups**: Track which environments are used most
6. **Cache Results**: Store patient data locally when appropriate

## File Reference

| File | Purpose |
|------|---------|
| `mpi_client.py` | High-level Python API (use this) |
| `get_patient.py` | Core services and CLI tool |
| `refresh_dotcare_token.py` | Token refresh utility |
| `.env` | Configuration and tokens |
| `MPI_TOKEN_MANAGEMENT.md` | Full documentation |
| `ARCHITECTURE.md` | System architecture |
| `QUICKSTART.md` | This file |

## Next Steps

1. ✅ Configure credentials in `.env`
2. ✅ Initialize tokens with `--init-tokens`
3. ✅ Verify with `--check --all`
4. 📝 Update your application code to use `mpi_client.py`
5. 📝 Set up periodic token refresh (optional)
6. 📝 Add monitoring/alerting (optional)

## Support

- **Documentation**: See `MPI_TOKEN_MANAGEMENT.md` for complete guide
- **Architecture**: See `ARCHITECTURE.md` for system design
- **Implementation**: See source code docstrings for details

## Quick Command Reference

```bash
# Setup
python3 get_patient.py --init-tokens

# Check status
python3 refresh_dotcare_token.py --check --all

# Refresh tokens
python3 refresh_dotcare_token.py --all

# Test lookup
python3 get_patient.py --env AHJ --id 1120579634

# Direct client usage
python3 mpi_client.py 1120579634 AHJ

# Search all environments
python3 mpi_client.py 1120579634
```

---

**Ready to use!** 🚀

The system is now configured for per-environment token management. Each business unit maintains its own authentication token, and tokens are automatically refreshed when needed.
