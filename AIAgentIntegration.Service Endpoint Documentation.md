# Change Summary — New Endpoint: GetPatientByIdentificationValue

| | |
|---|---|
| **Branch** | `Create-new-endpoint-for-PatientByIdentificationValue` |
| **Commit** | `2be9d7f` — Create new endpoint for PatientByIdentificationValue |
| **Author** | Mahmoud Ahmed AbdElFattah |
| **Date** | 2026-10-14 |
| **Scope** | 9 files changed, 227 insertions, 0 deletions |

---

## 1. Overview

A new API endpoint has been added that allows searching for a patient by their **national identification value** (Egyptian National ID or KSA National ID / Iqama) and returns the **full patient record** from the MPI (Master Patient Index).

The endpoint is exposed through the **AIAgentIntegration.Service** (port `8090`) in each environment. Authentication is handled by the **Security Service** (port `7000`) in the same environment.

---

## 2. Environments & Backend URLs

The service is deployed to 5 environments. Each environment exposes:

- **AIAgentIntegration.Service** → port **8090**
- **Security Service** (`SercurityService`) → port **7000**

### 2.1 AIAgentIntegration.Service Base URLs (port 8090)

| Environment | Base URL |
|---|---|
| AHJ | `http://ahj-hmis:8090/AIAgentIntegration.Service` |
| SNB | `http://snb-hmisapp-01:8090/AIAgentIntegration.Service` |
| AKW | `http://akw-hmis:8090/AIAgentIntegration.Service` |
| ALW | `http://alw-hmis:8090/AIAgentIntegration.Service` |
| ERS | `http://ers-fwzhmisapp:8090/AIAgentIntegration.Service` |

### 2.2 Security Service Base URLs (port 7000)

| Environment | Base URL |
|---|---|
| AHJ | `http://ahj-hmis:7000/SercurityService` |
| SNB | `http://snb-hmisapp-01:7000/SercurityService` |
| AKW | `http://akw-hmis:7000/SercurityService` |
| ALW | `http://alw-hmis:7000/SercurityService` |
| ERS | `http://ers-fwzhmisapp:7000/SercurityService` |

---

## 3. Security Service — Login Endpoint

### `GET /SercurityService/api/security/login`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `username` | string | Yes | Login username |
| `password` | string | Yes | Login password |

#### Example Request

```
GET http://ahj-hmis:7000/SercurityService/api/security/login?username=emp&password=123456
```

#### Full Login URLs — All Environments

| Environment | URL |
|---|---|
| AHJ | `http://ahj-hmis:7000/SercurityService/api/security/login?username=emp&password=123456` |
| SNB | `http://snb-hmisapp-01:7000/SercurityService/api/security/login?username=emp&password=123456` |
| AKW | `http://akw-hmis:7000/SercurityService/api/security/login?username=emp&password=123456` |
| ALW | `http://alw-hmis:7000/SercurityService/api/security/login?username=emp&password=123456` |
| ERS | `http://ers-fwzhmisapp:7000/SercurityService/api/security/login?username=emp&password=123456` |

---

## 4. New Endpoint (For Frontend Team)

### `GET /AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue`

| Parameter | Type | Required | Description |
|---|---|---|---|
| `identificationValue` | string | Yes | Patient national ID number |

#### Validation Rules
- Must be **exactly 14 digits** → Egyptian National ID
- Must be **exactly 10 digits** → KSA National ID / Iqama
- Empty / null / whitespace → returns `200 OK` with `null` body
- Any other format (letters, wrong length) → returns `400 Bad Request`

#### Possible Responses

| Status | Body | Case |
|---|---|---|
| `200 OK` | `PatientDto` (JSON) | Patient found |
| `200 OK` | `null` | No matching patient, or empty input |
| `400 Bad Request` | Error message string | Invalid identification value format |
| `500 Internal Server Error` | — | MPI backend failure |

#### Example Request

```
GET http://ahj-hmis:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231
```

#### Full URLs — All Environments

| Environment | URL |
|---|---|
| AHJ | `http://ahj-hmis:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231` |
| SNB | `http://snb-hmisapp-01:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231` |
| AKW | `http://akw-hmis:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231` |
| ALW | `http://alw-hmis:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231` |
| ERS | `http://ers-fwzhmisapp:8090/AIAgentIntegration.Service/api/MpiOperations/GetPatientByIdentificationValue?identificationValue=9876541231` |

---

## 5. Response Contract — `PatientDto`

The search is scoped to the caller's business unit (EntityID) and returns the full patient profile:

| Field | Type | Description |
|---|---|---|
| `ID` | long | Patient ID |
| `Code` | string | Patient code |
| `ArFirstName` / `ArSecondName` / `ArThirdName` / `ArLastName` | string | Arabic name parts |
| `EnFirstName` / `EnSecondName` / `EnThridName` / `EnLastName` | string | English name parts |
| `DateOfBirth` | datetime? | Gregorian date of birth |
| `HijriDateOfBirth` | string | Hijri date of birth |
| `IdentificationValue` | string | National ID value |
| `AgeEstimated` | bool? | Whether age is estimated |
| `Email` | string | Patient email |
| `MainPhone` | string | Primary phone |
| `Phones` | string | Additional phones |
| `IsActive` | bool? | Active flag |
| `DateOfDeath` | datetime? | Date of death (if deceased) |
| `CreatedBy` / `CreatedDate` | long? / datetime? | Creation audit |
| `UpdatedBy` / `UpdatedDate` | long? / datetime? | Last update audit |
| `EntityID` / `EntityKey` | long? / string | Business entity info |
| `DirectParentID` / `FinalParentID` | long? | Entity hierarchy |
| `PatientNote` | string | General patient note |
| `HealthId` | long | Health ID |
| `isYaqueenRetrieved` | bool | Retrieved from Yaqeen (KSA) |
| `YaqeenVerifiedFields` | long | Bitmask of Yaqeen-verified fields |
| `IsForeigner` | bool | Foreigner flag |
| `HasInfectiousDisease` | bool? | Infectious disease flag |

### Nested Lookup Objects (`SLKP_*`)

Each is `{ ID, Code, ArName, EnName }` (except Occupation: no `Code`; BloodGroup: `Name` / `Description`):

- `SLKP_Gender`
- `SLKP_MaritalStatus`
- `SLKP_Nationality`
- `SLKP_Religion`
- `SLKP_PatientClassification`
- `SLKP_Occupation`
- `SLKP_BloodGroup`
- `LKP_IdentificationType`

### `PatientDeathInfo` (nested object)

| Field | Type |
|---|---|
| `PatientID` | long |
| `MorgueRefrigeratorDatetime` | datetime? |
| `MorgueRefrigeratorNumber` | string |
| `TagNumber` | string |
| `NumberslabelsPrinted` | int? |
| `CreatedBy` / `CreatedDate` | long / datetime |
| `UpdatedBy` / `UpdatedDate` | long? / datetime? |

---

## 6. Business Summary (For Business Analyst)

- Business need: enable AI agents / client applications to retrieve a complete patient record using a national ID number instead of the internal patient ID.
- Supported ID types: **Egyptian National ID (14 digits)** and **KSA National ID / Iqama (10 digits)**.
- Search results are limited to the caller's own business unit — no cross-entity patient leakage.
- If no patient exists for the provided ID, the API returns `null` rather than an error, allowing the calling workflow to decide next steps (e.g., offer patient registration).
- The response includes demographic data, contact info, death information, and related lookups (gender, nationality, blood group, occupation, etc.).

---

## 7. Technical Changes (For Reference)

| File | Change |
|---|---|
| `Service/AIAgentIntegration/Controllers/MpiOperationsController.cs` | New `GetPatientByIdentificationValue` HTTP GET endpoint with input validation |
| `Utilities/AIAgentIntegration.Extensions/Helpers/IdentificationValueValidator.cs` | **New** — regex validator (14 or 10 digits) |
| `Model/AIAgentIntegration.Model/DTO/PatientDto.cs` | **New** — full patient response DTO |
| `Model/AIAgentIntegration.Model/DTO/PatientDeathInfoDto.cs` | **New** — death info DTO |
| `Model/AIAgentIntegration.Model/DTO/BaseLkpDto.cs` | **New** — base lookup DTOs (Gender, Nationality, Occupation, BloodGroup, etc.) |
| `DataRouting/.../IMpiDataRouting.cs` + `MpiDataRouting.cs` | Routing layer passthrough |
| `DataFlow/.../IMpiDataFlow.cs` + `MpiDataFlow.cs` | Calls MPI service `GetPatientByIdentificationValueForAI` via Service Locator with logging |

---

## 8. Frontend Integration Checklist

- [ ] Authenticate first via the Security Service (port `7000`) before calling AIAgentIntegration.Service (port `8090`)
- [ ] Handle `null` response (patient not found) gracefully
- [ ] Handle `400` for invalid ID format — display validation message before calling API when possible
- [ ] Trim user input before sending (spaces are rejected)
- [ ] Use `SLKP_*` nested objects for localized dropdown labels (`ArName` / `EnName`)
- [ ] Point requests to the correct environment base URL (see Section 2)
