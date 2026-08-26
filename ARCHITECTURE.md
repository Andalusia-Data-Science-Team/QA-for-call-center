# Call QA Analysis System — Architecture & Workflow

*Reflects the current state of the codebase, including the config fixes, the
graph concurrency fix, the `upload-analyze-test` / direct-execution testing
tools, and the full bank/location feature build-out — bank, location, and
offers now live together in the shared `app/service_hub/` package (bank
and location used to be their own `app/bank_node/` and `app/location_node/`
folders; offers used to be `app/offers_node/`) while remaining independent
features/graph nodes. Written to replace the drift between `README.md` /
`diagram.md` and the actual code — treat this file as the one to keep
current going forward.*

---

## 1. What This System Does

A FastAPI backend that takes a call-center chat transcript (WhatsApp-style,
Arabic-first) and runs it through a LangGraph pipeline that produces a
structured QA report: pass/needs-review/escalate, compliance flags, agent
performance scoring, plus two deterministic checks specific to this
business — did the agent give the right **branch address** when asked, and
did the agent give the right **bank account/IBAN/bank name** when asked.
Results persist to SQL Server and are also surfaced through server-rendered
dashboards.

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Jinja2 (server-rendered dashboards, no SPA) |
| Orchestration | LangGraph (`StateGraph`) |
| Schema/validation | Pydantic v2 |
| Database | SQL Server (pyodbc / SQLAlchemy) — 4 separate systems (QA results, reservations, Robin HQ chats, Dynamics 365 CRM) |
| LLM | Anthropic / OpenAI / HuggingFace / OpenRouter — provider-agnostic client |
| CRM auth | MSAL (Azure AD, public client) against Dynamics 365's TDS endpoint |
| Arabic/matching | Custom normalization + `rapidfuzz` |

---

## 3. Folder Structure (current)

```
app/
├── main.py                      FastAPI app — all HTTP routes + direct-execution test runner
├── config.py                    Settings (env/.env)
├── agent/                       LangGraph pipeline — the orchestration core
│   ├── agent.py                   QAAgent façade — compiles + runs the graph
│   ├── graph.py                   Graph topology (nodes + edges)
│   ├── nodes.py                   All node implementations
│   └── state.py                   AgentState (shared graph state)
├── models/
│   ├── input.py                   CallTranscript, BatchCallTranscripts
│   └── output.py                  QAAnalysisResult, ComplianceFlag, AgentPerformance
├── prompts/
│   └── qa_prompt.py                SYSTEM_PROMPT + 6 focused prompt builders (now incl. location)
├── criteria/                     YAML evaluation policy (behavioral/compliance/offer/reservation/scoring)
├── services/                     Shared, domain-agnostic infrastructure
│   ├── llm_client.py                Provider-agnostic LLM client
│   ├── crm_connector.py             Shared Dynamics 365 connector (MSAL/connection/retry)
│   ├── criteria_loader.py           YAML → prompt text (lru_cached)
│   ├── sql_helpers.py               DB writes for QA results
│   ├── text_helpers.py              Arabic normalization, financial-ID masking, transcript-turn splitter
│   └── slots_check.py               ad-hoc script
├── service_hub/                  Shared package for the three QA-validation features below —
│   │                                each module is still fully self-contained/independent;
│   │                                only the package/folder is shared, not the logic.
│   ├── crm_bank.py                  Cached fetch of cr301_bankaccounts
│   ├── bank_validation.py           BU resolution + detection + validation (deterministic, no LLM)
│   ├── README_bank.md
│   ├── crm_location.py              Cached fetch of cr301_andalusialocations
│   ├── location_validation.py       Detection + branch resolution + address validation
│   ├── README_location.md
│   ├── crm_offers.py                Offer fetch + fuzzy/semantic matching
│   ├── crm_database.py              Doctor walk-in-price fetch (shrunk to just this)
│   ├── intent_detector.py, offer_search.py   orphaned — not imported by the app
│   └── README.md
├── CM/                            Robin HQ chat-retrieval subsystem (data ingestion)
├── SQL/                           All raw .sql files
├── templates/                     4 server-rendered dashboards
├── tests/                         pytest unit tests + eval.py smoke script
└── chats/                         sample/test transcripts
```

---

## 4. Entry Points

**Production:** `uvicorn app.main:app` (or `python main.py --serve`). Module-level
code builds `llm_client` and compiles the `QAAgent`'s graph **once** at
import time; every request reuses that singleton.

**Local direct-execution testing:** `python main.py` (no `uvicorn`, no
HTTP) — loads a JSON file (`TEST_JSON_PATH`, edit the constant near the
bottom of `main.py`), validates it with the same `CallTranscript` model
`/upload-analyze` uses, runs it through the same `analyzer.analyze()`, and
pretty-prints the result to the terminal. A separate import-path bootstrap
at the top of `main.py` (guarded by `__package__ in (None, "")`) makes this
work regardless of how the file is invoked, without affecting normal
`uvicorn`/`-m` loading at all.

**API endpoints:**

| Endpoint | Purpose |
|---|---|
| `POST /analyze-call` | Single transcript, strict schema |
| `POST /batch-analyze` | Up to 50 transcripts, concurrent, per-item error isolation |
| `POST /upload-analyze` | File upload, strict schema |
| `POST /upload-analyze-test` | File upload, **lenient** — fills missing/null required fields with obvious placeholders for that one request only |
| `POST /retrieve-and-analyze` | Pulls live chats from Robin HQ, converts, analyzes as a batch |
| `GET /health` | No dependencies |
| `POST /login`, `GET /qa-dashboard`, `/agents-dashboard`, `/qa-supervisor` | Auth + dashboards |
| `POST /escalations`, `/escalations/review`, `/escalations/dismiss` | Escalation review lifecycle (writes to `Call_QA_Results`) |
| `GET /agents/emails` | Agent email list from DWH |

---

## 5. LangGraph Pipeline — Full Topology

```
START
  │
  ▼
load_call ──(error)───────────────────────────────────────────────────┐
  │                                                                    │
  │ fan-out: 6 parallel criteria loaders (pure YAML reads, lru_cached) │
  ├→ load_behavioral_criteria ─┐                                      │
  ├→ load_compliance_pillars  ─┤                                      │
  ├→ load_script_templates    ─┤ fan-in                               │
  ├→ load_reservation_pillars ─┤                                      │
  ├→ load_offer_pillars       ─┤                                      │
  └→ load_scoring_weights     ─┘                                      │
                              │                                       │
                       criteria_ready  (barrier)                      │
                              │                                       │
                       detect_intent  (Arabic booking-keyword scan)   │
                              │                                       │
     fan-out: 2 PARALLEL, fully independent deterministic nodes       │
     ├→ validate_bank_information  (app.service_hub.bank_validation)   │
     └→ validate_location          (app.service_hub.location_validation)│
                    │        │                                        │
                    └───┬────┘                                        │
                  loc_bank_ready  (barrier)                           │
                              │                                       │
              ┌───────────────┴───────────────┐                       │
        (booking intent)              (no booking intent)             │
              │                               │                       │
    extract_appointment_details               │                      │
              │                               │                       │
    verify_appointment_in_db                  │                      │
              │                               │                       │
    infer_reservation_evaluation ──(error)────┼───────────────────────┤
              │                               │                       │
              └───────────┬───────────────────┘                       │
                    inference_gate  (barrier — single fan-out point)  │
                              │                                       │
     fan-out: parallel LLM calls + CRM offer fetch                    │
     ├→ infer_behavioral_evaluation ──(error)──┐                      │
     ├→ infer_compliance_evaluation ──(error)──┤                      │
     ├→ infer_script_matching  [dead code — always no-op]             │
     └→ fetch_crm_offers_for_call → infer_offer_evaluation ──(error)──┤
                                                │ fan-in               │
                                          inference_ready              │
                                                │                      │
                                    infer_overall_scoring ──(error)────┤
                                                │                      │
                                       aggregate_results ──(error)─────┤
                                                │                      │
                                        integrity_check                │
                                                │                      │
                                        save_to_database                │
                                                │                      │
                                            finalize            ┌──────┘
                                                │               │
                                               END ◄─ handle_error ◄───┘
```

**Key mechanics:**
- Any node writing `state["error"]` routes immediately to `handle_error`,
  which builds a safe `error`-assessment `QAAnalysisResult`. `error`/
  `error_node`/`result` all use a last-write-wins `Annotated` reducer so
  multiple parallel LLM nodes failing in the same step — e.g. no LLM key
  configured at all — routes cleanly instead of crashing with
  `InvalidUpdateError`.
- `infer_script_matching` is wired into the graph but its body is dead code
  (unreachable after `pass`) — a pre-existing issue, not part of this
  restructuring.

---

## 6. Bank & Location — Two Fully Independent Features

Originally one merged deterministic check, now two parallel graph nodes,
each self-contained like the offers feature (all three live in
`app/service_hub/`, but remain fully independent modules):

```
app/service_hub/bank_validation.py        app/service_hub/location_validation.py
   │ BU resolved FIRST, via                   │ detection: address/branch request
   │  BUSINESS_UNIT_KEYWORD_MAP                    (context-aware — "مكان" alone
   │  (app.models.input) — exact match             doesn't trigger; needs a place-noun)
   │  then fuzzy fallback, multi-word         │ branch resolution: distinctive-token
   │  aliases only                                 overlap (ubiquity-filtered, same
   │ scope-gated to {AFW,LIVE,AKW,ALW}             idea as crm_offers.py's generic-word
   │  — LCH/MKR/SNB stay valid BUs                 filtering), ranked by match count
   │  elsewhere, just out of bank scope            then ratio; fuzzy fallback for typos
   │ detection: IBAN / account-number /       │ address-answer scoring: token overlap
   │  bank-name / account-owner / general          against the branch's description,
   │ multi-account-per-BU disambiguation           same ubiquity filtering
   │  (narrows by named bank; flags
   │  AMBIGUOUS_MULTIPLE_ACCOUNTS when
   │  unresolved)
   │ exact identifier match — never fuzzy
   │ WRONG_FIELD_TYPE_PROVIDED when a
   │  valid-but-wrong-field answer is given
   ▼                                          ▼
app/service_hub/crm_bank.py                app/service_hub/crm_location.py
   │ cached fetch: cr301_bankaccounts           │ cached fetch: cr301_andalusialocations
   ▼                                           ▼
app/services/crm_connector.py  ◄────────────────┘
   (shared, domain-agnostic: MSAL auth, connection, retry —
    also used by app/service_hub/crm_offers.py)
```

Bank validation resolves its Business Unit from `BUSINESS_UNIT_KEYWORD_MAP`
directly — it does **not** depend on location's CRM fetch at all
(an earlier design routed BU resolution through location records; that
cross-dependency has been removed). The BU vocabularies differ between
layers and are bridged in exactly one place
(`bank_validation._APP_BU_TO_CRM_BU = {"LIVE": "AHJ"}`): app-level chat
aliases resolve to `LIVE` for the main Jeddah hospital, while the CRM's own
`cr301_bu`/`servhub_bulookupname`/`cr18c_bunname` fields store that same
branch as `"AHJ"` (verified via a live CRM query — `owningbusinessunitname`
is a constant `"ERS"` on every row and is never used for BU filtering).

Both nodes write to their own `AgentState` key (`bank_validation`,
`location_validation`) and their own `QAAnalysisResult` field.
`aggregate_results` independently turns either one's `is_violation=True`
into a C2B compliance flag; `integrity_check` downgrades a `pass` to
`needs_review` if either fired; `infer_overall_scoring`'s prompt injects
both as separate sub-evaluations (5 and 6) for the LLM's context.

The only thing the two share is
`app.services.text_helpers.split_transcript_by_speaker()` (turn-splitting)
— neither imports from the other.

---

## 7. Data Flow (single request)

```
Raw JSON  →  CallTranscript (Pydantic validation)  →  AgentState (mutated additively by each node)
   ↓
6× focused LLM calls + 2 deterministic checks (bank, location) — each writes its own state key
   ↓
aggregate_results — merges all sub-results into one dict, Pydantic-validates → QAAnalysisResult
   ↓
integrity_check — patches escalation/assessment consistency (incl. bank/location violations)
   ↓
save_to_database — flattened, one SQL row per ComplianceFlag (non-fatal on failure)
   ↓
JSON response (same QAAnalysisResult)
```

---

## 8. State & Output Schema

**`AgentState`** (`app/agent/state.py`) — key fields: `call`, 6× criteria
blocks, 4× LLM sub-eval dicts (`behavioral_eval`/`compliance_eval`/
`reservation_eval`/`offer_eval`/`scoring_eval`), `bank_validation`,
`location_validation`, `is_booking_intent`, `appointment_details`,
`appointment_verification`, `crm_offers_context`, `result`
(last-write-wins), `error`/`error_node` (last-write-wins), `node_trace`.

**`QAAnalysisResult`** (`app/models/output.py`) — `overall_assessment`,
`assessment_reasoning`, `compliance_flags[]`, `agent_performance`,
`escalation_required`/`reason`, plus `bank_validation: dict | None` and
`location_validation: dict | None` (independent — either can be populated
without the other).

---

## 9. External Integrations

| Service | Auth | Used by |
|---|---|---|
| Anthropic / OpenAI / HuggingFace / OpenRouter | API key | All 6 LLM-calling nodes (provider default: OpenRouter, per `config.py`) |
| SQL Server — DWH (QA results) | user/pass via `Passcode.json` (not present in this checkout) | `sql_helpers.py`, escalation endpoints |
| SQL Server — reservations | same | `verify_appointment_in_db` |
| SQL Server — Robin HQ | `CM/passcode.json` | `/retrieve-and-analyze`, `/agents/emails` |
| Dynamics 365 CRM (TDS) | MSAL / Azure AD | `crm_connector.py` → `crm_bank.py`, `crm_location.py`, `crm_offers.py`, `crm_database.py` |

---

## 10. Testing

```
app/tests/test_bank_validation.py       29 tests — BU resolution (exact + fuzzy),
                                         bank-validation scope gating, field-type
                                         detection (incl. account owner), multi-account
                                         disambiguation, bank-name matching, ambiguity,
                                         one-digit IBAN, multi-message aggregation,
                                         CRM-unavailable non-punitive degrade
app/tests/test_location_validation.py   11 tests — exact/spelling-variant/wrong address,
                                         dental-vs-general branch disambiguation,
                                         ambiguous ties, KSA-only filtering
app/tests/test_sql_helpers.py           2 tests — unaffected regression
```
42/42 pass. `test_output_models.py` still has a pre-existing broken
`from models.output import ...` import (missing the `app.` prefix) —
unrelated to this work, never fixed.

---

## 11. Known Open Items

- `app/Passcode.json` doesn't exist in this checkout — any DB write/lookup
  depending on it will fail until created locally.
- Secrets committed to git history (`app/CM/passcode.json`,
  `mail_client.py`'s hardcoded SMTP password).
- `mail_client.py` sends a real email as an import-time side effect — dead
  code, never imported by the running app, but dangerous if it ever is.
- `README.md`/`diagram.md` still describe an earlier, simpler pipeline —
  this file supersedes them for architecture purposes; consider updating or
  removing them to avoid two conflicting sources of truth.
- `infer_script_matching` remains dead code in a live graph position.
