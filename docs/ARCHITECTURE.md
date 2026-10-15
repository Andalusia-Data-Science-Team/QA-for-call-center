# MPI Token Management Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                            │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│  │  Web API       │  │  Service Hub   │  │  Other Apps    │       │
│  │  Endpoints     │  │  Integration   │  │                │       │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘       │
│           └──────────────┬─────────────────────────┘               │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MPI Client API (mpi_client.py)                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • get_patient_by_id(id, env, auto_search_all)                │ │
│  │  • get_patient_by_id_safe(id, env)                            │ │
│  │  • get_patient_ahj(id), get_patient_snb(id), etc.             │ │
│  │  • validate_patient_id(id)                                     │ │
│  │  • Automatic token management                                  │ │
│  │  • Exception handling & retries                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                Core Services (get_patient.py)                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • get_token_for_env(env) → Read from .env                    │ │
│  │  • save_token_to_env(env, token) → Write to .env              │ │
│  │  • login(env, ...) → Authenticate with SecurityService        │ │
│  │  • get_patient(env, id, token) → Query MPI API                │ │
│  │  • validate_identification_value(id) → Format validation      │ │
│  │  • initialize_all_tokens() → Batch token setup                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└───────────┬──────────────────────────────────┬──────────────────────┘
            │                                  │
            ▼                                  ▼
┌───────────────────────────┐    ┌───────────────────────────────────┐
│   Token Refresh Utility   │    │      Environment Storage          │
│  (refresh_dotcare_token)  │    │          (.env file)              │
│  ┌─────────────────────┐  │    │  ┌──────────────────────────────┐│
│  │ • refresh_token()   │  │    │  │ MPI_AUTH_TOKEN_AHJ          ││
│  │ • check_expiry()    │  │◄───┼──┤ MPI_AUTH_TOKEN_SNB          ││
│  │ • batch refresh     │  │    │  │ MPI_AUTH_TOKEN_AKW          ││
│  │ • CLI interface     │  │    │  │ MPI_AUTH_TOKEN_ALW          ││
│  └─────────────────────┘  │    │  │ MPI_AUTH_TOKEN_ERS          ││
└───────────┬───────────────┘    │  │ MPI_AUTH_TOKEN_JDC          ││
            │                    │  │ MPI_AUTH_TOKEN (fallback)   ││
            │                    │  │ MPI_SERVICE_USERNAME        ││
            │                    │  │ MPI_SERVICE_PASSWORD        ││
            │                    │  │ MPI_NTLM_USERNAME           ││
            │                    │  │ MPI_NTLM_PASSWORD           ││
            │                    │  └──────────────────────────────┘│
            │                    └───────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    External MPI Services                            │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  AHJ (ahj-   │  │  SNB (snb-   │  │  AKW (akw-   │            │
│  │   hmis)      │  │  hmisapp-01) │  │   hmis)      │            │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │            │
│  │  │Port    │  │  │  │Port    │  │  │  │Port    │  │            │
│  │  │7000    │  │  │  │7000    │  │  │  │7000    │  │            │
│  │  │Security│  │  │  │Security│  │  │  │Security│  │  +3 more   │
│  │  │Service │  │  │  │Service │  │  │  │Service │  │  (ALW,     │
│  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │   ERS,     │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │   JDC)     │
│  │  │Port    │  │  │  │Port    │  │  │  │Port    │  │            │
│  │  │8090    │  │  │  │8090    │  │  │  │8090    │  │            │
│  │  │MPI API │  │  │  │MPI API │  │  │  │MPI API │  │            │
│  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

## Token Lifecycle

```
┌──────────────────────────────────────────────────────────────────┐
│                     Initial Setup Phase                          │
└──────────────────────────────────────────────────────────────────┘

    1. Run: python3 get_patient.py --init-tokens
       │
       ├─► Connect to AHJ SecurityService (port 7000)
       │   └─► Login → Get Token → Save to .env as MPI_AUTH_TOKEN_AHJ
       │
       ├─► Connect to SNB SecurityService (port 7000)
       │   └─► Login → Get Token → Save to .env as MPI_AUTH_TOKEN_SNB
       │
       └─► ... repeat for AKW, ALW, ERS, JDC


┌──────────────────────────────────────────────────────────────────┐
│                   Runtime Request Flow                           │
└──────────────────────────────────────────────────────────────────┘

    Application calls: get_patient_by_id("1120579634", "AHJ")
       │
       ▼
    ┌────────────────────────────────────────┐
    │ Check for MPI_AUTH_TOKEN_AHJ in .env   │
    └─────────────┬──────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
    Found               Not Found
        │                   │
        │                   ▼
        │           ┌──────────────────┐
        │           │ Auto-login to    │
        │           │ AHJ Security     │
        │           │ Service          │
        │           └────────┬─────────┘
        │                    │
        │                    ▼
        │           ┌──────────────────┐
        │           │ Save new token   │
        │           │ to .env          │
        │           └────────┬─────────┘
        │                    │
        └────────────────────┘
                  │
                  ▼
    ┌────────────────────────────────────────┐
    │ Call MPI API (port 8090) with token    │
    └─────────────┬──────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
    Success             401 Unauthorized
        │                   │
        │                   ▼
        │           ┌──────────────────┐
        │           │ Refresh token    │
        │           │ from Security    │
        │           │ Service          │
        │           └────────┬─────────┘
        │                    │
        │                    ▼
        │           ┌──────────────────┐
        │           │ Update .env      │
        │           │ with new token   │
        │           └────────┬─────────┘
        │                    │
        │                    ▼
        │           ┌──────────────────┐
        │           │ Retry request    │
        │           └────────┬─────────┘
        │                    │
        └────────────────────┘
                  │
                  ▼
    ┌────────────────────────────────────────┐
    │ Return patient data to application     │
    └────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────┐
│                   Multi-Environment Search                       │
└──────────────────────────────────────────────────────────────────┘

    Application calls: get_patient_by_id("1120579634", auto_search_all=True)
       │
       ▼
    ┌────────────────────────────────────────┐
    │ Try AHJ environment                    │
    └─────────────┬──────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
    Found               Not Found
        │                   │
        └──► Return         ▼
                  ┌────────────────────────────────────────┐
                  │ Try SNB environment                    │
                  └─────────────┬──────────────────────────┘
                                │
                      ┌─────────┴─────────┐
                      │                   │
                      ▼                   ▼
                  Found               Not Found
                      │                   │
                      └──► Return         ▼
                                ┌────────────────────────────────────┐
                                │ Try AKW environment                │
                                └─────────────┬──────────────────────┘
                                              │
                                    ... continue until found or
                                        all environments exhausted
```

## Token Storage Structure

```
.env File Layout
─────────────────────────────────────────────────────────────

# Authentication Credentials
MPI_SERVICE_USERNAME=AiModule
MPI_SERVICE_PASSWORD=123456
MPI_NTLM_USERNAME=andalusia\username
MPI_NTLM_PASSWORD=password

# Generic Token (Fallback)
MPI_AUTH_TOKEN='eyJ0eXAiOiJKV1Qi...[long JWT]...'

# Environment-Specific Tokens (Primary)
MPI_AUTH_TOKEN_AHJ='eyJ0eXAiOiJKV1Qi...[JWT for AHJ]...'
MPI_AUTH_TOKEN_SNB='eyJ0eXAiOiJKV1Qi...[JWT for SNB]...'
MPI_AUTH_TOKEN_AKW='eyJ0eXAiOiJKV1Qi...[JWT for AKW]...'
MPI_AUTH_TOKEN_ALW='eyJ0eXAiOiJKV1Qi...[JWT for ALW]...'
MPI_AUTH_TOKEN_ERS='eyJ0eXAiOiJKV1Qi...[JWT for ERS]...'
MPI_AUTH_TOKEN_JDC='eyJ0eXAiOiJKV1Qi...[JWT for JDC]...'
```

## Component Responsibilities

### mpi_client.py (High-Level API)
- **Purpose**: Developer-friendly interface
- **Responsibilities**:
  - Input validation
  - Environment selection
  - Multi-environment search
  - Exception handling
  - Retry logic
- **Used by**: Application code, web services, background jobs

### get_patient.py (Core Services)
- **Purpose**: Core MPI operations
- **Responsibilities**:
  - Token reading/writing
  - SecurityService authentication
  - MPI API communication
  - Token lifecycle management
- **Used by**: mpi_client.py, CLI tools

### refresh_dotcare_token.py (Maintenance)
- **Purpose**: Manual token management
- **Responsibilities**:
  - Manual token refresh
  - Expiry checking
  - Batch operations
  - Monitoring/alerting support
- **Used by**: Administrators, scheduled tasks

### .env (Configuration)
- **Purpose**: Secure credential/token storage
- **Responsibilities**:
  - Store per-environment tokens
  - Store authentication credentials
  - Store endpoint URLs
- **Accessed by**: All components

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                          │
└─────────────────────────────────────────────────────────────┘

Layer 1: File System Security
├─ .env file permissions (600 or restrictive)
├─ .gitignore prevents version control commit
└─ Process-level access control

Layer 2: Authentication
├─ NTLM authentication for SecurityService
├─ Service account credentials (MPI_SERVICE_USERNAME/PASSWORD)
└─ Domain credentials (MPI_NTLM_USERNAME/PASSWORD)

Layer 3: Authorization
├─ JWT tokens with role-based claims
├─ Per-environment token isolation
└─ Token expiry enforcement

Layer 4: Transport Security
├─ Internal network (VPN-protected)
├─ HMIS stack network isolation
└─ HTTP (internal network, controlled environment)

Layer 5: Application Security
├─ Input validation (ID format)
├─ Exception handling (no credential leakage)
├─ Audit logging (authentication attempts)
└─ Secure token storage (memory + .env only)
```

## Deployment Scenarios

### Scenario 1: Single Environment (Production)

```
┌────────────────────────────────────────┐
│  Production Application (AHJ only)     │
└────────────────────────────────────────┘
         │
         ▼
   Uses only MPI_AUTH_TOKEN_AHJ
         │
         ▼
   ┌──────────────┐
   │  AHJ HMIS    │
   │  (ahj-hmis)  │
   └──────────────┘
```

### Scenario 2: Multi-Environment (Aggregation Service)

```
┌────────────────────────────────────────┐
│  Patient Search Service                │
│  (Searches all environments)           │
└────────────────────────────────────────┘
         │
         ├──► MPI_AUTH_TOKEN_AHJ ──► AHJ HMIS
         ├──► MPI_AUTH_TOKEN_SNB ──► SNB HMIS
         ├──► MPI_AUTH_TOKEN_AKW ──► AKW HMIS
         ├──► MPI_AUTH_TOKEN_ALW ──► ALW HMIS
         ├──► MPI_AUTH_TOKEN_ERS ──► ERS HMIS
         └──► MPI_AUTH_TOKEN_JDC ──► JDC HMIS
```

### Scenario 3: Development/Testing

```
┌────────────────────────────────────────┐
│  Development Environment               │
│  (VPN access to test environments)     │
└────────────────────────────────────────┘
         │
         ├──► Test subset (e.g., AHJ + SNB)
         │    with environment-specific tokens
         │
         └──► Mock/stub other environments
```

## Maintenance Windows

### Daily Operations
- No maintenance required
- Tokens auto-refresh on demand

### Weekly Tasks (Optional)
```bash
# Check all token expiry status
python3 refresh_dotcare_token.py --check --all

# Proactive refresh if needed
python3 refresh_dotcare_token.py --all
```

### Monthly Tasks
```bash
# Audit token usage
# Review authentication logs
# Update credentials if rotated
```

## Monitoring & Alerting

### Health Checks
1. Token expiry monitoring
2. Authentication success rate
3. API response times
4. Environment availability

### Alert Triggers
- Token expiry within 7 days
- Authentication failures > threshold
- API 401 rate > threshold
- Environment unreachable

### Metrics to Track
- Requests per environment
- Token refresh frequency
- Average response time per environment
- Error rate per environment
