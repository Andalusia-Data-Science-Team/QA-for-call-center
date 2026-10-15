# Quick Start: Patient Profile Validation

## What Was Added?

A new validation step that checks if agents:
1. ✅ Collect all **required patient information** during booking calls
2. ✅ Acknowledge when patients mention **profile updates**

## When Does It Run?

**Only on booking calls** where:
- Agent asks for iqama number ✓
- Patient is eligible (insurance check passed) ✓
- Patient profile exists in MPI ✓

## What Does It Check?

### Required Patient Fields (Actual MPI Fields)
| MPI Field | Example | Why Critical |
|-----------|---------|--------------|
| **EnFirstName/ArFirstName** | ABDULMAJEED / عبدالمجيد | Identity verification |
| **MainPhone/Phones** | 0508382553 | Appointment reminders & contact |
| **DateOfBirth** | 2001-08-16 | Age verification & eligibility |
| **Gender** | {"Code": "m"} | Service appropriateness |
| **MaritalStatus** | {"Code": "s"} | Demographic completeness |
| **Nationality** | {"Code": "113"} | Eligibility & billing |
| **IdentificationType** | {"Code": "1"} | Verification method |

**Note**: Email is **optional** in MPI (often null for active patients)

### Profile Updates
When patient says:
- "اسمي الصحيح..." (My correct name is...)
- "رقمي الجديد..." (My new number is...)
- "تزوجت مؤخراً" (I got married recently)
- "جنسيتي..." (My nationality is...)

Agent MUST acknowledge:
- "سأقوم بتحديث" (I will update)
- "تم التسجيل" (Recorded)
- "هنحدث البيانات" (We'll update the data)

## What Flags Get Generated?

### C2C_021: Missing Required Info
**When**: Agent never asked for a **critical** required field that's missing in MPI

**Example**:
```
MPI Shows: MainPhone = null
Transcript: Agent never asks for mobile number
→ Flag: C2C_021 "Agent failed to collect required MainPhone (mobile number)"
```

**Non-Example** (Email is optional):
```
MPI Shows: Email = null
Transcript: Agent never asks for email
→ No Flag: Email is optional in MPI system
```

### C2C_028: Profile Update Not Acknowledged
**When**: Patient mentions profile change but agent doesn't confirm

**Example**:
```
Patient: "رقمي الجديد 0501234567"
Agent: "ممتاز، تم الحجز" (proceeds without acknowledgment)
→ Flag: C2C_028 "Agent didn't confirm profile update"
```

## Configuration Required

### 1. Environment Variables (.env)
```bash
# MPI Service Authentication
MPI_SERVICE_USERNAME=AiModule
MPI_SERVICE_PASSWORD=123456

# Per-Business-Unit Tokens (auto-generated)
MPI_AUTH_TOKEN_AHJ='your-token-here'
MPI_AUTH_TOKEN_SNB='your-token-here'
MPI_AUTH_TOKEN_AKW='your-token-here'
# ... etc
```

### 2. Generate Tokens (One Time)
```bash
cd app/services
python3 get_patient.py --init-tokens
```

This will:
- Login to each MPI environment
- Save tokens to `.env`
- Show success/failure per business unit

## How to Test

### Test Case 1: Complete Profile
```python
# Given: All fields present in MPI
# When: Agent makes booking
# Then: No flags generated ✓
```

### Test Case 2: Missing MainPhone, Not Asked
```python
# Given: MainPhone = null in MPI
# When: Agent never asks for mobile number
# Then: C2C_021 flag generated ✗
```

### Test Case 3: Missing MainPhone, Asked
```python
# Given: MainPhone = null in MPI
# When: Agent asks "ممكن رقم الجوال؟"
# And: Patient refuses or provides it
# Then: No flag (agent tried) ✓
```

### Test Case 4: Profile Update Mentioned
```python
# Given: Patient says "غيرت رقمي"
# When: Agent says "سأقوم بالتحديث"
# Then: No flag (acknowledged) ✓
```

### Test Case 5: Profile Update Ignored
```python
# Given: Patient says "غيرت رقمي"  
# When: Agent ignores and continues
# Then: C2C_028 flag generated ✗
```

## Troubleshooting

### Issue: "MPI returned status 401"
**Solution**: Regenerate tokens
```bash
python3 get_patient.py --init-tokens
```

### Issue: "Patient not found in MPI"
**Cause**: Invalid iqama or patient doesn't exist
**Action**: Validation skips (no flags generated)

### Issue: "No iqama number available"
**Cause**: Agent never asked or transcript unclear
**Action**: Validation skips (no flags generated)

### Issue: MPI Service Unavailable
**Action**: Validation gracefully skips, logs warning

## Impact on Agent Scoring

### Agent Classification
```
Before: A = No violations
After:  A = No violations (including patient profile)

Before: C = 1 critical violation
After:  C = 1 critical (patient profile flags count)
```

### Overall Assessment
- **Pass**: All checks pass including patient profile
- **Needs Review**: Any C2C violations (including C2C_021, C2C_028)
- **Escalate**: Critical violations remain critical

## Quick Commands

```bash
# Check syntax
python3 -m py_compile app/agent/patient_profile_nodes.py

# Test MPI connection for specific BU
python3 app/services/get_patient.py --env AHJ --id 1234567890

# Test all environments
python3 app/services/get_patient.py --all

# Initialize all tokens
python3 app/services/get_patient.py --init-tokens

# Check graph wiring
python3 -m py_compile app/agent/graph.py
```

## Integration Checklist

- [x] `app/agent/state.py` - Added state fields
- [x] `app/agent/patient_profile_nodes.py` - New validation nodes
- [x] `app/prompts/qa_prompt.py` - LLM validation prompt
- [x] `app/agent/nodes.py` - Import patient profile service
- [x] `app/agent/graph.py` - Wired into graph flow
- [x] `.env` - MPI credentials configured
- [ ] Run token initialization
- [ ] Test with sample calls
- [ ] Monitor production logs

## Expected Behavior

### Before This Feature
```
Booking Call → Extract Details → Verify DB → Score
(No check if agent collected complete patient info)
```

### After This Feature
```
Booking Call 
  → Check Eligibility 
  → Fetch Patient Profile from MPI
  → Validate Agent Collected All Info
  → Flag if Missing/Not Acknowledged
  → Extract Details → Verify DB → Score
```

## Monitoring

### Key Metrics to Track
1. **Profile Fetch Success Rate**: % of successful MPI fetches
2. **C2C_021 Frequency**: How often agents miss required fields
3. **C2C_028 Frequency**: How often profile updates not acknowledged
4. **Skip Rate**: % of calls where validation skipped (expected on non-booking)

### Log Messages to Watch
```
fetch_patient_profile | call_id=X result=found
fetch_patient_profile | call_id=X result=not_found
validate_patient_profile | call_id=X status=incomplete missing=2
validate_patient_profile | call_id=X status=pass
```

## Support

**For MPI Issues**: Check `app/services/get_patient.py` documentation
**For Validation Logic**: See `app/prompts/qa_prompt.py` - build_patient_profile_validation_prompt
**For Graph Flow**: See `docs/patient_profile_flow.md`

## Rollback Plan

If issues arise, disable by commenting out in `app/agent/graph.py`:

```python
# builder.add_edge("check_patient_eligibility", "fetch_patient_profile")
# Replace with:
builder.add_edge("check_patient_eligibility", "booking_merge")
```

This bypasses patient profile validation entirely.
