# MPI Field Mapping & Validation Rules

## Actual MPI PatientDto Structure

Based on the actual API response from `GetPatientByIdentificationValue`:

```json
{
    "ID": 1162238,
    "Code": "1008934",
    "ArFirstName": "عبدالمجيد",
    "ArSecondName": "مطير",
    "ArThirdName": "صليح",
    "ArLastName": "الزهراني",
    "EnFirstName": "ABDULMAJEED",
    "EnSecondName": "MUTAIR",
    "EnThridName": "SULAYYIH",
    "EnLastName": "ALZAHRANI",
    "DateOfBirth": "2001-08-16T00:00:00+03:00",
    "HijriDateOfBirth": "1422-05-26 00:00:00",
    "IdentificationValue": "1120579634",
    "Email": null,
    "MainPhone": "0508382553",
    "Phones": "0508382553,",
    "Gender": {
        "Code": "m",
        "NameAr": "ذكر",
        "NameEn": "Male"
    },
    "MaritalStatus": {
        "Code": "s",
        "NameAr": "أعزب",
        "NameEn": "Single"
    },
    "Nationality": {
        "Code": "113",
        "NameAr": "العربية السعودية",
        "NameEn": "Saudi Arabia"
    },
    "IdentificationType": {
        "Code": "1",
        "NameAr": "رقم هوية",
        "NameEn": "Identity Card"
    }
}
```

## Field Priority Classification

### 🔴 Critical Fields (MUST be collected)
These fields are essential for patient identification and service delivery. Missing values trigger **C2C_021** flags if agent didn't ask.

| MPI Field | Type | Example Value | Arabic Name | Why Critical |
|-----------|------|---------------|-------------|--------------|
| `EnFirstName` | string | "ABDULMAJEED" | الاسم الأول بالإنجليزية | Identity verification (bilingual) |
| `ArFirstName` | string | "عبدالمجيد" | الاسم الأول بالعربية | Identity verification (primary) |
| `MainPhone` | string | "0508382553" | الجوال الرئيسي | Appointment reminders, contact |
| `DateOfBirth` | datetime | "2001-08-16T00:00:00+03:00" | تاريخ الميلاد | Age verification, eligibility |
| `Gender` | object | {"Code": "m"} | الجنس | Service appropriateness (gender-specific clinics) |

**Validation Logic**:
- Name: Consider complete if **EITHER** `EnFirstName` **OR** `ArFirstName` is present
- Phone: Check `MainPhone` first, fallback to `Phones` field
- Gender: Must have valid code ("m" or "f")
- DateOfBirth: Must be valid ISO datetime

---

### 🟡 Important Fields (Should be collected)
Demographic completeness fields. Lower priority than critical fields but still important.

| MPI Field | Type | Example Value | Arabic Name | Why Important |
|-----------|------|---------------|-------------|---------------|
| `MaritalStatus` | object | {"Code": "s"} | الحالة الاجتماعية | Demographic records, statistics |
| `Nationality` | object | {"Code": "113"} | الجنسية | Billing, eligibility verification |
| `IdentificationType` | object | {"Code": "1"} | نوع الهوية | Verification method (ID/Iqama/Passport) |

**Validation Logic**:
- Flag with **C2C_021** but lower severity mention
- Code-based objects: Check for valid `Code` field

---

### ⚪ Optional Fields (Not validated)
These fields are optional in the MPI system. Missing values do **NOT** trigger flags.

| MPI Field | Type | Note |
|-----------|------|------|
| `Email` | string | Often null even for active patients |
| `ArSecondName` | string | Optional middle name |
| `ArThirdName` | string | Optional third name |
| `EnThridName` | string | Optional (note typo in API) |
| `HijriDateOfBirth` | string | Secondary birth date format |

---

## Field Access Patterns

### Nested Objects
Several fields are nested objects with structure:
```json
{
    "Code": "value",
    "NameAr": "Arabic label",
    "NameEn": "English label"
}
```

**Access pattern**:
```python
# Check if Gender is present
if profile.get("Gender") and profile["Gender"].get("Code"):
    gender_code = profile["Gender"]["Code"]
else:
    gender_code = None  # Missing
```

### Name Handling
```python
# Check if name is complete (either language)
has_english_name = profile.get("EnFirstName")
has_arabic_name = profile.get("ArFirstName")

if not (has_english_name or has_arabic_name):
    # Flag: Patient name is missing
```

### Phone Handling
```python
# Check phone availability
main_phone = profile.get("MainPhone")
phones_field = profile.get("Phones")

if not main_phone and not phones_field:
    # Flag: No phone number available
elif not main_phone but phones_field:
    # MainPhone is null but Phones has value
    # Only flag if agent never asked at all
```

---

## Compliance Flag Mapping

### C2C_021: "Described Inaccurate, Incomplete Information-None Medical"

**Trigger Conditions**:
1. **Critical field** is `null` in MPI **AND**
2. Agent never asked for it in the transcript **AND**
3. Patient didn't explicitly refuse to provide it

**Examples**:
```
MPI: MainPhone = null
Transcript: Agent never mentions "رقم الجوال" or "mobile"
→ Flag: C2C_021 "Agent failed to collect required MainPhone (mobile number)"

MPI: Gender.Code = null
Transcript: Agent never asks about gender
→ Flag: C2C_021 "Agent failed to collect required Gender"

MPI: Email = null
Transcript: Agent never asks for email
→ NO FLAG: Email is optional
```

---

### C2C_028: "Didn't confirm the booking or the solution before end of the call"

**Trigger Conditions**:
1. Patient mentions a profile update **AND**
2. Agent doesn't acknowledge the update

**Profile Update Types**:
- Name correction: "اسمي الصحيح..." / "الاسم خطأ..."
- Phone change: "رقمي الجديد..." / "غيرت رقمي..."
- Marital status change: "تزوجت" / "انفصلت" / "أرمل الآن"
- Nationality update: "جنسيتي..." / "حصلت على جنسية..."
- Gender correction: (rare) "الجنس خطأ في النظام"

**Acknowledgment Phrases** (agent must say one of these):
- "سأقوم بتحديث" (I will update)
- "سيتم التعديل" (It will be modified)
- "تم التسجيل" (It has been recorded)
- "هنحدث البيانات" (We will update the data)
- "هحدثها في النظام" (I'll update it in the system)

**Examples**:
```
Patient: "رقمي الجديد 0501234567"
Agent: "تمام، هنحدث الرقم في النظام"
→ NO FLAG: Update acknowledged

Patient: "رقمي الجديد 0501234567"
Agent: "ممتاز، تم الحجز"
→ Flag: C2C_028 "Agent didn't acknowledge phone number update"
```

---

## Special Cases

### Case 1: Phones vs MainPhone
```json
{
    "MainPhone": null,
    "Phones": "0508382553,"
}
```
**Validation**: Only flag if agent never asked for **any** phone number. The presence of `Phones` value suggests phone was collected at some point.

### Case 2: Bilingual Names
```json
{
    "EnFirstName": "ABDULMAJEED",
    "ArFirstName": null
}
```
**Validation**: Name is considered **complete** - at least one language present.

### Case 3: Nested Object Null Check
```json
{
    "Gender": null
}
```
vs
```json
{
    "Gender": {
        "Code": null,
        "NameAr": null,
        "NameEn": null
    }
}
```
**Validation**: Both are considered missing. Check for `Gender.Code` specifically.

### Case 4: Patient Deceased
```json
{
    "PatientClassification": {
        "Code": "Dead",
        "NameAr": "متوفى"
    },
    "DateOfDeath": "2026-09-09T06:17:26.967+03:00"
}
```
**Validation**: Skip all profile validation - deceased patients don't need data updates.

---

## LLM Prompt Field References

The validation prompt uses these exact field names:

```
REQUIRED PATIENT PROFILE FIELDS (from MPI GetPatientByIdentificationValue API)
These fields MUST be collected by the agent for any booking:
- **EnFirstName / ArFirstName** (English/Arabic first name)
- **MainPhone / Phones** (Primary mobile phone)
- **DateOfBirth** (Date of birth in ISO format)
- **Gender** (Gender code: m/f)
- **MaritalStatus** (Marital status code)
- **Nationality** (Nationality code)
- **IdentificationType** (ID type: Identity Card/Iqama/Passport)

Note: Email is optional in this system (often null even for active patients)
```

---

## Testing Checklist

- [ ] Test with complete profile (all fields present) → No flags
- [ ] Test with null MainPhone, agent didn't ask → C2C_021
- [ ] Test with null MainPhone, agent asked → No flag
- [ ] Test with null Gender, agent didn't ask → C2C_021
- [ ] Test with null Email, agent didn't ask → No flag (optional)
- [ ] Test patient mentions phone update, agent acknowledges → No flag
- [ ] Test patient mentions phone update, agent ignores → C2C_028
- [ ] Test with Phones populated but MainPhone null → No flag if agent collected
- [ ] Test with only EnFirstName (no ArFirstName) → No flag (name complete)
- [ ] Test with deceased patient classification → Skip validation

---

## Database Implications

The MPI profile fetch does **NOT** modify any database records. It is read-only:
- ✅ Reads patient profile from MPI API
- ✅ Validates agent's collection behavior
- ❌ Does NOT update patient records
- ❌ Does NOT persist profile changes

**Actual profile updates** must be done through proper MPI update endpoints (not in scope of QA validation).
