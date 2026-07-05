```mermaid
graph TD
    A[load_call] --> B[load_behavioral_criteria]
    A --> C[load_compliance_pillars]
    A --> D[load_script_templates]
    A --> E[load_scoring_weights]

    B & C & D & E --> F["infer_behavioral_evaluation
LLM Call A"]
    B & C & D & E --> G["infer_compliance_evaluation
LLM Call B"]
    B & C & D & E --> H["infer_script_matching
LLM Call C"]

    F -->|error| ERR[handle_error]
    F --> DI[detect_intent]

    DI -->|booking| EAD[extract_appointment_details]
    DI -->|skip_booking| I

    EAD --> VAD[verify_appointment_in_db]
    VAD --> I

    G --> I
    H --> I

    I["infer_overall_scoring
LLM Call D"] --> J[aggregate_results]
    J --> K[integrity_check]
    K --> L[finalize]
```

```mermaid
graph TD
    F["infer_behavioral_evaluation
LLM Call A"] -->|error| ERR[handle_error]
    F --> DI[detect_intent]

    DI -->|booking| EAD[extract_appointment_details]
    DI -->|skip_booking| I["infer_overall_scoring
LLM Call D"]

    EAD --> VAD[verify_appointment_in_db]
    VAD --> I

    G["infer_compliance_evaluation
LLM Call B"] --> I
    H["infer_script_matching
LLM Call C"] --> I
```


criteria_ready
    │
    └──→ detect_intent (sequential)
              │
              ├──(booking)──→ extract → verify → infer_reservation_evaluation
              │                                          │
              └──(skip)──────────────────────────────────┘
                                   │ (always sequential, finishes completely)
                                   ▼
              ┌────────────── infer_behavioral_evaluation ◄── criteria_ready
              │               infer_compliance_evaluation ◄── criteria_ready
              │                        │ fan-in (exactly 2, unconditional)
              │                        ▼
              │                inference_ready
              │                        │
              │                        ▼
              └──────────────► infer_overall_scoring → aggregate → integrity_check → finalize