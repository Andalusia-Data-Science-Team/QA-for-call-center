# Patient Profile Validation Feature

## Overview
This feature adds patient profile validation to the QA system, ensuring agents collect complete patient information and acknowledge profile updates during booking calls.

## Implementation Summary

### 1. New Files Created
- **`app/agent/patient_profile_nodes.py`**: Contains two new nodes:
  - `fetch_patient_profile()`: Fetches patient profile from MPI service
  - `validate_patient_profile()`: Validates agent collected required info

### 2. Modified Files

#### `app/agent/state.py`
- Added new state fields:
  - `patient_profile_result`: Stores fetched MPI profile
  - `patient_profile_eval`: Stores validation results

#### `app/agent/nodes.py`
- Added import for get_patient service functions
- Updated `aggregate_results()` to include patient_profile_eval flags
- Updated `infer_overall_scoring()` to pass patient_profile_summary to scoring prompt

#### `app/prompts/qa_prompt.py`
- Added `build_patient_profile_validation_prompt()`: LLM prompt for validating patient info
- Updated `build_scoring_prompt()` to accept and display patient_profile_summary

#### `app/agent/graph.py`
- Imported patient profile nodes
- Added nodes to graph builder:
  - `fetch_patient_profile`
  - `validate_patient_profile`
- Updated graph flow to route through patient profile validation after eligibility check

## Graph Flow Changes

### Original Flow
```
check_patient_eligibility
  ├─ (eligible) → booking_merge
  └─ (not_eligible) → handle_ineligible_patient
```

### New Flow
```
check_patient_eligibility
  ├─ (eligible) → fetch_patient_profile
  │                     ↓
  │               validate_patient_profile
  │                     ↓
  │               booking_merge
  └─ (not_eligible) → handle_ineligible_patient
```

## Validation Logic

### Required Patient Fields (from actual MPI API response)

#### Critical Fields
1. **EnFirstName/ArFirstName** - Patient name (English/Arabic)
2. **MainPhone/Phones** - Primary mobile phone contact
3. **DateOfBirth** - Date of birth (ISO format)
4. **Gender** - Gender code (m/f)

#### Important Fields
5. **MaritalStatus** - Marital status (Single/Married/etc.)
6. **Nationality** - Nationality code (e.g., 113 for Saudi Arabia)
7. **IdentificationType** - ID type (Identity Card/Iqama/Passport)

**Note**: Email is optional in the MPI system (often null for active patients)

### Compliance Mapping
The validation uses **existing** compliance rules:

| Scenario | Rule | Description |
|----------|------|-------------|
| Missing required field not asked by agent | **C2C_021** | "Described Inaccurate, Incomplete Information-None Medical" |
| Wrong information about profile update | **C2C_023** | "Described wrong Information" |
| Profile update not acknowledged | **C2C_028** | "Didn't confirm the booking or the solution before end of the call" |

### Profile Update Detection
Monitors patient mentions of:
- Name corrections (EnFirstName/ArFirstName/EnLastName/ArLastName)
- Phone number changes (MainPhone/Phones)
- Marital status changes (MaritalStatus)
- Nationality updates (Nationality)
- Gender corrections (Gender)

Validates agent acknowledgment phrases like:
- "سأقوم بتحديث" (I will update)
- "سيتم التعديل" (It will be modified)
- "تم التسجيل" (It has been recorded)
- "هنحدث" (We will update)

## MPI Integration

### Environment Mapping
Business units are mapped to MPI environments:
```python
{
    "BU-AHJ": "AHJ",
    "LIVE": "AHJ",
    "BU-SNB": "SNB",
    "BU-AKW": "AKW",
    "BU-ALW": "ALW",
    "BU-ERS": "ERS",
    "BU-JDC": "JDC",
}
```

### Authentication
- Uses per-environment tokens from `.env` (MPI_AUTH_TOKEN_<ENV>)
- Falls back to login with MPI_SERVICE_USERNAME/PASSWORD if token unavailable
- Tokens are cached after successful login

### API Call
```python
GET /AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue
    ?identificationValue=<iqama_number>
```

## Execution Conditions

The patient profile validation **only runs when**:
1. `is_booking_intent` = True (booking keywords detected)
2. `iqama_number` is available (extracted from transcript or provided)
3. Patient eligibility check passed

**Skips when**:
- No booking intent
- No iqama number
- Eligibility check failed
- MPI profile not found

## Output Schema

### Patient Profile Result
```json
{
  "found": true/false,
  "profile": {...},  // Full MPI PatientDto
  "error": null,
  "env": "AHJ"
}
```

### Patient Profile Evaluation
```json
{
  "applicable": true,
  "validation_status": "complete|incomplete|profile_updates_not_acknowledged|pass",
  "missing_fields": [
    {
      "field_name": "MainPhone",
      "field_value": null,
      "agent_asked": false,
      "criticality": "critical",
      "reason": "MainPhone is missing and agent never asked for mobile number"
    }
  ],
  "profile_updates": [
    {
      "update_type": "phone_change",
      "patient_statement": "رقمي الجديد 0501234567",
      "agent_acknowledged": false,
      "agent_response": null
    }
  ],
  "compliance_flags": [
    {
      "type": "C2C",
      "severity": "critical",
      "description": "C2C_021: Agent failed to collect required MainPhone (mobile number)",
      "transcript_excerpt": "..."
    }
  ]
}
```

### MPI PatientDto Sample (Actual API Response)
```json
{
    "ID": 1162238,
    "Code": "1008934",
    "ArFirstName": "عبدالمجيد",
    "EnFirstName": "ABDULMAJEED",
    "DateOfBirth": "2001-08-16T00:00:00+03:00",
    "MainPhone": "0508382553",
    "Phones": "0508382553,",
    "Email": null,
    "Gender": {"Code": "m", "NameAr": "ذكر", "NameEn": "Male"},
    "MaritalStatus": {"Code": "s", "NameAr": "أعزب", "NameEn": "Single"},
    "Nationality": {"Code": "113", "NameAr": "العربية السعودية"},
    "IdentificationType": {"Code": "1", "NameAr": "رقم هوية"}
}
```

## Integration with Existing System

### Scoring Impact
- Patient profile flags are merged with all other compliance flags in `aggregate_results()`
- Included in overall scoring via `infer_overall_scoring()`
- Affects Agent Classification (A/B/C/D) based on violation counts

### Reporting
- Flags appear in final `QAAnalysisResult.compliance_flags`
- Included in database persistence
- Visible in QA dashboards alongside other compliance issues

## Testing Recommendations

1. **Test with complete profile**: Verify no false positives when all fields present
2. **Test missing fields**: Ensure C2C_021 flags when agent doesn't ask
3. **Test profile updates**: Verify C2C_028 when updates not acknowledged
4. **Test MPI unavailable**: Ensure graceful degradation
5. **Test non-booking calls**: Verify proper skipping
6. **Test different business units**: Verify correct environment routing

## Dependencies

- `app/services/get_patient.py`: MPI integration service
- `.env`: MPI authentication tokens per environment
- LangGraph: For graph execution flow
- Existing compliance regulations YAML

## Configuration

Add to `.env` if not present:
```bash
# MPI Service Credentials
MPI_SERVICE_USERNAME=AiModule
MPI_SERVICE_PASSWORD=123456

# Per-environment tokens (generated by get_patient.py --init-tokens)
MPI_AUTH_TOKEN_AHJ='<token>'
MPI_AUTH_TOKEN_SNB='<token>'
MPI_AUTH_TOKEN_AKW='<token>'
MPI_AUTH_TOKEN_ALW='<token>'
MPI_AUTH_TOKEN_ERS='<token>'
MPI_AUTH_TOKEN_JDC='<token>'
```

## Future Enhancements

1. **Field Priority**: Weight critical fields (phone, name) higher than optional ones
2. **Partial Updates**: Track which specific field was updated
3. **CRM Sync**: Validate if updates were actually persisted to CRM
4. **Retry Logic**: Implement retry for transient MPI failures
5. **Cache Profiles**: Cache profiles for same iqama within session
6. **Audit Trail**: Log all profile access for compliance
