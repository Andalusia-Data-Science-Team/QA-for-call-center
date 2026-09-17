# LangGraph Pipeline Topology

## Full Pipeline

```mermaid
graph TD
    START([START]) --> LC[load_call]

    LC -->|error| ERR[handle_error]
    LC --> LBC[load_behavioral_criteria]
    LC --> LCP[load_compliance_pillars]
    LC --> LRP[load_reservation_pillars]
    LC --> LOP[load_offer_pillars]
    LC --> LST[load_script_templates]
    LC --> LSW[load_scoring_weights]

    LBC & LCP & LRP & LOP & LST & LSW --> CR[criteria_ready]

    CR --> DI[detect_intent]

    DI -->|bank intent| VBI[validate_bank_information]
    DI -->|no bank| SBV[skip_bank_validation]
    DI -->|location intent| VL[validate_location]
    DI -->|no location| SLV[skip_location_validation]

    VBI & SBV & VL & SLV --> LBR[loc_bank_ready]

    LBR --> DII[detect_insurance_intent]
    DII -->|insurance| CPE[check_patient_eligibility]
    DII -->|continue| BM[booking_merge]
    CPE -->|eligible| BM
    CPE -->|not_eligible| HIP[handle_ineligible_patient]
    CPE -->|error| ERR
    HIP --> BM

    BM -->|validate_doctor| VD[validate_doctor]
    BM -->|skip_doctor| SDV[skip_doctor_validation]
    VD & SDV --> BR[booking_ready]

    BR -->|booking or offer_only| EAD["extract_appointment_details\n(LLM)"]
    BR -->|skip_booking| IG[inference_gate]

    EAD -->|booking| VADB[verify_appointment_in_db]
    EAD -->|offer_only| OED[offer_extraction_done]
    OED --> IG

    VADB --> IRE["infer_reservation_evaluation\n(LLM)"]
    IRE -->|error| ERR
    IRE --> EIRV[enforce_ineligible_reservation_violation]
    EIRV --> IG

    IG --> FCO[fetch_crm_offers_for_call]
    IG --> FCS[fetch_crm_services_for_call]
    IG --> FCP[fetch_crm_packages_for_call]

    FCO --> IBE["infer_behavioral_evaluation\n(LLM)"]
    FCO --> ICE["infer_compliance_evaluation\n(LLM)"]
    FCO --> ISM["infer_script_matching\n(LLM)"]
    FCO --> IOE["infer_offer_evaluation\n(LLM)"]
    FCO -->|doctor scope needed| IDSV["infer_doctor_scope_validation\n(LLM)"]
    FCO -->|no scope| SDSV[skip_doctor_scope_validation]
    FCO -->|COE triggered| ICV["infer_coe_validation\n(LLM)"]
    FCO -->|no COE| SCV[skip_coe_validation]

    FCS --> ISE["infer_service_evaluation\n(LLM)"]
    FCP --> IPE["infer_package_evaluation\n(LLM)"]

    IBE -->|error| ERR
    IBE --> BD[behavioral_done]
    ICE -->|error| ERR
    ICE --> CD[compliance_done]
    ISM -->|error| ERR
    ISM --> SD[script_done]
    IOE -->|error| ERR
    IOE --> OD[offer_done]
    ISE -->|error| ERR
    ISE --> SVD[service_done]
    IPE -->|error| ERR
    IPE --> PD[package_done]
    IDSV -->|error| ERR
    IDSV --> DSD[doctor_scope_done]
    SDSV --> DSD
    ICV -->|error| ERR
    ICV --> COED[coe_done]
    SCV --> COED

    BD & CD & SD & OD & SVD & PD & DSD & COED --> IR[inference_ready]

    IR --> VCL["validate_crm_lead\n(LLM)"]
    VCL -->|error| ERR
    VCL --> DFE[detect_faq_escalation]
    DFE -->|validate| VFR["validate_faq_record\n(LLM)"]
    DFE -->|skip| IOS["infer_overall_scoring\n(LLM)"]
    VFR -->|error| ERR
    VFR --> IOS
    IOS -->|error| ERR
    IOS --> AGG[aggregate_results]
    AGG -->|error| ERR
    AGG --> IC[integrity_check]
    IC --> SDB[save_to_database]
    SDB --> FIN[finalize]
    FIN --> END([END])
    ERR --> END

    style IBE fill:#1565c0,color:#ffffff
    style ICE fill:#1565c0,color:#ffffff
    style ISM fill:#1565c0,color:#ffffff
    style IOE fill:#1565c0,color:#ffffff
    style ISE fill:#1565c0,color:#ffffff
    style IPE fill:#1565c0,color:#ffffff
    style IDSV fill:#1565c0,color:#ffffff
    style ICV fill:#1565c0,color:#ffffff
    style IRE fill:#1565c0,color:#ffffff
    style EAD fill:#1565c0,color:#ffffff
    style IOS fill:#1565c0,color:#ffffff
    style VCL fill:#1565c0,color:#ffffff
    style VFR fill:#1565c0,color:#ffffff
    style VD fill:#e65100,color:#ffffff
    style VBI fill:#e65100,color:#ffffff
    style VL fill:#e65100,color:#ffffff
    style IG fill:#b71c1c,color:#ffffff
    style IR fill:#b71c1c,color:#ffffff
    style CR fill:#b71c1c,color:#ffffff
    style LBR fill:#b71c1c,color:#ffffff
    style BR fill:#b71c1c,color:#ffffff
    style BM fill:#b71c1c,color:#ffffff
    style SDB fill:#1b5e20,color:#ffffff
```

## Inference Fan-Out Stage (Detailed)

```
inference_gate (single entry — exactly 1 arrival per call)
    │
    ├──→ fetch_crm_offers_for_call
    │         │
    │         ├──→ infer_behavioral_evaluation  → behavioral_done ──┐
    │         ├──→ infer_compliance_evaluation  → compliance_done ──┤
    │         ├──→ infer_script_matching        → script_done ──────┤
    │         ├──→ infer_offer_evaluation       → offer_done ───────┤  fan-in
    │         ├──(doctor resolved+need)─→ infer_doctor_scope_validation → doctor_scope_done ──┤
    │         │   (else)─────────────────→ skip_doctor_scope_validation → doctor_scope_done ──┤
    │         ├──(COE trigger)──────────→ infer_coe_validation          → coe_done ───────────┤
    │         └── (no COE)─────────────→ skip_coe_validation            → coe_done ───────────┤
    │                                                                                          │
    ├──→ fetch_crm_services_for_call                                                           │
    │         └──→ infer_service_evaluation  → service_done ────────────────────────────────────┤
    │                                                                                          │
    └──→ fetch_crm_packages_for_call                                                           │
              └──→ infer_package_evaluation → package_done ─────────────────────────────────────┤
                                                                                               │
                                                                              inference_ready ◄─┘
                                                                     (barrier — 8 predecessors)
                                                                                               │
                                                                          validate_crm_lead (LLM)
                                                                                               │
                                                                      detect_faq_escalation
                                                                                               │
                                                                      infer_overall_scoring (LLM)
                                                                                               │
                                                                         aggregate_results
                                                                                               │
                                                                          integrity_check
                                                                                               │
                                                                          save_to_database
                                                                                               │
                                                                              finalize → END
```

## Offer Validation Logic

`fetch_crm_offers_for_call` fetches **all active CRM offers** for the call's
specialty (and gender when available) from Dynamics 365. It passes them as a
compact context block into `infer_offer_evaluation`.

`infer_offer_evaluation` performs **two-way validation**:

| Scenario | Outcome | Flag |
|---|---|---|
| Agent mentioned an offer AND it matches a live CRM offer | `SUITABLE_OFFER_RECOMMENDED` | ✅ positive |
| Agent mentioned an offer but details are wrong (price/date/specialty) | `OFFER_MISREPRESENTED` | ⚠️ C2B moderate |
| Agent mentioned an irrelevant/unmatched offer | `UNRELATED_OFFER_RECOMMENDED` | ⚠️ C2B moderate |
| A relevant CRM offer existed but agent never mentioned it | `OFFER_SKIPPED` | ⚠️ C2B moderate |
| Agent mentioned offer but omitted price or expiry | `INCOMPLETE_OFFER_PRESENTATION` | NC minor |
| Agent didn't ask patient to book after presenting offer | `MISSING_OFFER_CONFIRMATION_ASK` | NC minor |
| No active CRM offer for this specialty | `NO_OFFER_AVAILABLE` | — none |
| Non-booking call type | `OFFER_NOT_APPLICABLE` | — none |

## Key Design Decisions

### 1. Barrier Nodes (loop-prevention)
- **`criteria_ready`** (6 predecessors): waits for all 6 YAML loaders
- **`loc_bank_ready`** (4 predecessors): waits for bank pair + location pair
- **`booking_merge`** (≤3 predecessors): merges insurance/eligibility paths before doctor routing
- **`booking_ready`** (2 predecessors): waits for doctor branch — single entry into booking router
- **`inference_gate`** (1 predecessor per call): single fan-out point, never reached twice
- **`inference_ready`** (8 predecessors): one `*_done` per parallel branch

### 2. Why the old graph looped
The previous wiring had three root causes:
1. `loc_bank_ready` had **6** predecessors (bank×2 + location×2 + doctor×2) so it fired 3 times, triggering `extract_appointment_details` 3 times.
2. `infer_package_evaluation` was missing its `builder.add_conditional_edges` call (bare tuple) so `package_done` was never set and `inference_ready` retried.
3. `infer_doctor_scope_validation` and `infer_coe_validation` went directly to `inference_ready` (bypassing `*_done` barriers) giving it an inconsistent predecessor count.

### 3. Booking Branch
Runs **sequentially** (extract → verify → infer_reservation) before the parallel inference fan-out. Ineligible patients still pass through `booking_merge` → `booking_ready` → booking router so the reservation is checked for improper bookings.

### 4. Error Routing
Every fallible node has a conditional edge to `handle_error → END`.

### 5. Database Write
`save_to_database` is **non-fatal** — errors are logged but don't set `state["error"]`.

## Node Summary

| Node | Type | Description |
|------|------|-------------|
| `load_call` | validation | Entry — checks `CallTranscript` exists |
| `load_*` | loader | YAML criteria reads (lru_cached) |
| `criteria_ready` | barrier | Waits for 6 loaders |
| `detect_intent` | keyword scan | Booking/offer keywords (احجز, حجز, عرض…) |
| `validate_bank_information` | deterministic | KSA bank-account check |
| `validate_location` | deterministic | KSA branch/location check |
| `loc_bank_ready` | barrier | Waits for bank + location branches (4) |
| `detect_insurance_intent` | keyword scan | IQAMA / insurance detection |
| `check_patient_eligibility` | API | Beneficiary insurance eligibility check |
| `booking_merge` | barrier | Merges insurance/ineligible paths |
| `validate_doctor` | deterministic+LLM | Doctor CRM resolution + field validation |
| `booking_ready` | barrier | Single entry into booking router |
| `extract_appointment_details` | LLM | Date, doctor, specialty, patient, offer name |
| `verify_appointment_in_db` | SQL | Fuzzy-match reservation lookup |
| `infer_reservation_evaluation` | LLM | 6 reservation pillars |
| `inference_gate` | barrier | Single fan-out into parallel inference |
| `fetch_crm_offers_for_call` | CRM | Active offers for specialty + gender |
| `fetch_crm_services_for_call` | CRM | Active services for specialty |
| `fetch_crm_packages_for_call` | CRM | Active packages for specialty |
| `infer_behavioral_evaluation` | LLM | Tone, empathy, professionalism, red flags |
| `infer_compliance_evaluation` | LLM | 15 compliance pillars |
| `infer_script_matching` | LLM | Greeting/closing script adherence |
| `infer_offer_evaluation` | LLM | Two-way offer validation (agent + CRM) |
| `infer_service_evaluation` | LLM | Service price/name accuracy |
| `infer_package_evaluation` | LLM | Package price/name accuracy |
| `infer_doctor_scope_validation` | LLM | Doctor CRM scope vs patient clinical need |
| `infer_coe_validation` | LLM | Center of Excellence recommendation check |
| `inference_ready` | barrier | Waits for 8 `*_done` nodes |
| `validate_crm_lead` | LLM | CRM lead field accuracy |
| `detect_faq_escalation` | keyword | FAQ escalation claim detection |
| `validate_faq_record` | LLM | FAQ record match validation |
| `infer_overall_scoring` | LLM | Synthesizes all sub-results |
| `aggregate_results` | merge | Combines outputs → `QAAnalysisResult` |
| `integrity_check` | validation | Fixes escalation ↔ assessment mismatches |
| `save_to_database` | SQL INSERT | Persists to `[DWH].[AI].[Call_QA_Results]` |
| `finalize` | logging | Logs summary, closes trace |
| `handle_error` | error sink | Converts failure to safe error result |