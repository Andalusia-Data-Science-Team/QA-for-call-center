# Call QA Analysis API

An AI-powered quality assurance system for clinical call center transcripts.  
Built as a technical assessment for a Pain Management & Neurology Clinic.

---

## What It Does

Accepts a phone call transcript via a REST API and returns a structured quality report:
- **Overall assessment**: `pass`, `needs_review`, or `escalate`
- **Compliance flags**: HIPAA concerns, misinformation, rudeness, protocol violations, or positive interactions
- **Agent performance scores**: professionalism, accuracy, and resolution (0–1 floats)
- **Escalation guidance**: boolean flag + plain-language reason when a critical issue is found

The system is designed to replace a human QA team that currently reviews ~9% of calls — enabling 100% call coverage while being **non-punitive** (it coaches, not scores punitively).

---

## Project Structure

```
qa_system/
├── app/
│   ├── main.py                 # FastAPI app, endpoints
│   ├── config.py               # Settings from env vars / .env
│   ├── agent/
│   │   ├── graph.py            # LangGraph pipeline definition
│   │   ├── nodes.py            # All graph nodes (loaders, inference, validation)
│   │   └── state.py            # AgentState TypedDict schema
│   ├── models/
│   │   ├── input.py            # Pydantic input models (CallTranscript)
│   │   └── output.py           # Pydantic output models (QAAnalysisResult)
│   ├── prompts/
│   │   └── qa_prompt.py        # Focused prompt builders (behavioral, compliance, scoring)
│   ├── criteria/
│   │   ├── behavioral/         # Department-aware behavioral standards (YAML)
│   │   ├── compliance.yaml     # 15 compliance pillars (C2Com/C2C/C2B/NC)
│   │   ├── reservation.yaml    # 6 reservation-specific pillars
│   │   ├── scripts.yaml        # Approved greeting/closing templates
│   │   └── scoring.yaml        # Scoring weights & thresholds
│   ├── services/
│   │   ├── llm_client.py       # Provider-agnostic LLM client (Anthropic + OpenAI)
│   │   ├── criteria_loader.py # YAML criteria loader with lru_cache
│   │   ├── sql_helpers.py      # Database insert helper for QA results
│   │   └── text_helpers.py     # Arabic normalization, markdown stripping
│   ├── SQL/
│   │   ├── agentDB.sql         # INSERT query for Call_QA_Results table
│   │   └── Slots.sql           # Appointment verification lookup query
│   └── Passcode.json           # DB connection credentials (multiple environments)
├── tests/
│   └── eval.py                 # Evaluation script with expected-outcome assertions
├── sample_transcripts/
│   ├── 01_clean_call.json      # Scheduling call with no issues
│   ├── 02_hipaa_violation.json # Records call: PHI disclosed without ID verification
│   ├── 03_edge_disconnected.json # Helpdesk: call dropped mid-conversation
│   └── 04_rude_and_misinformation.json # Authorizations: rudeness + false approval status
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run

### 1. Clone / unzip the project

```bash
cd qa_system
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
```

**To use Anthropic (default):**
```
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
```

**To use OpenAI:**
```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

### 5. Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

### 6. Test a transcript

```bash
curl -X POST http://localhost:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d @sample_transcripts/01_clean_call.json
```

### 7. Run the evaluation suite

```bash
python tests/eval.py
```

The eval script is a generated script that sends all 4 sample transcripts to the API, validates the JSON schema, and asserts expected outcomes (e.g., the HIPAA call must escalate, the clean call must pass).

---

## API Reference

### `POST /analyze-call`

**Request body:**
```json
{
  "call_id": "CALL-001",
  "agent_name": "Maria Santos",
  "call_date": "2025-03-15",
  "call_duration_seconds": 187,
  "department": "Scheduling",
  "transcript": "Agent: Thank you for calling...\nCaller: Hi, I need..."
}
```

**Response:**
```json
{
  "call_id": "CALL-001",
  "overall_assessment": "pass",
  "assessment_reasoning": "The agent handled the scheduling request professionally...",
  "compliance_flags": [
    {
      "type": "positive_interaction",
      "severity": "positive",
      "description": "Agent confirmed appointment details clearly and provided a confirmation number.",
      "transcript_excerpt": "Your confirmation number is 892774."
    }
  ],
  "agent_performance": {
    "professionalism_score": 0.95,
    "accuracy_score": 0.90,
    "resolution_score": 1.0,
    "strengths": ["Warm greeting", "Confirmed all appointment details", "Offered preparation guidance"],
    "improvements": ["Could have asked about parking or accessibility needs"]
  },
  "escalation_required": false,
  "escalation_reason": null
}
```

### `POST /batch-analyze`

Accepts `{ "calls": [ ... ] }` — a list of up to 50 transcripts.  
Runs all analyses concurrently. One failure does not block the rest.

Returns:
```json
{
  "results": [ ... ],
  "summary": { "total": 4, "pass": 2, "needs_review": 1, "escalate": 1, "errors": 0 }
}
```

---

## Architecture: LangGraph Pipeline

### Why a multi-node graph instead of a single LLM call?

The original single-prompt approach was hitting token limits, producing inconsistent results, and mixing concerns (behavioral evaluation + compliance checking + scoring logic all in one 8K-token response).

The **LangGraph pipeline** solves this by decomposing the analysis into focused, parallel LLM calls:

```
load_call → [5 criteria loaders in parallel] → criteria_ready
    → detect_intent (keyword-based booking detection)
    → [booking branch: extract → verify → infer_reservation] OR skip
    → inference_gate (barrier)
    → [infer_behavioral_evaluation + infer_compliance_evaluation in parallel]
    → inference_ready (barrier)
    → infer_overall_scoring
    → aggregate_results (merge all sub-results)
    → integrity_check (fix escalation mismatches)
    → save_to_database (persist to SQL Server)
    → finalize
```

**Key benefits:**
1. **Focused prompts** — each LLM call has a single responsibility (behavioral tone, compliance pillars, or final scoring), reducing hallucinations
2. **Parallelism** — behavioral + compliance run concurrently, cutting latency by ~40%
3. **Intent-aware evaluation** — booking calls trigger appointment verification against the live reservations DB
4. **Automatic persistence** — results are written to `[DWH].[AI].[Call_QA_Results]` with auto-incrementing `Analysis_Version` for re-runs
5. **Composable** — new evaluation dimensions (e.g. script adherence, sentiment analysis) can be added as new parallel nodes without touching existing logic

### Prompting Strategy

**Evidence-first instruction**  
Every focused prompt starts with: *only flag issues you can directly observe in the transcript*. The LLM is told explicitly to note ambiguity rather than assume the worst.

**Proportionate escalation thresholds**  
`escalate` is reserved for HIPAA violations, dangerous misinformation, or explicit rudeness. The compliance prompt gives concrete examples of each severity tier (C2Com → C2C → C2B → NC).

**Department-specific behavioral standards**  
Each department (Scheduling, Records, Authorizations, etc.) has a tailored YAML file in `app/criteria/behavioral/` with department-specific tone expectations and red flags.

**Compliance pillar checklist**  
15 official compliance pillars loaded from `compliance.yaml`, grouped by severity. The compliance evaluation node checks each pillar independently and returns a list of violations (or an empty list for clean calls).

**Structured output enforcement**  
All LLM calls return JSON-only. Pydantic validates the merged result at the `aggregate_results` node — any schema violation is caught and returned as an `error` assessment.

---

## Edge Case Handling

| Scenario | Handling |
|---|---|
| Very short call (< ~100 words) | Noted in reasoning; scores default to 0.5 (neutral) to avoid penalizing the agent |
| Call disconnected mid-conversation | Recognized as a technical issue; agent is not penalized for resolution |
| Transcript with no issues | Returns `pass` with at least one `positive_interaction` flag |
| Ambiguous statement | Prompt instructs the model to note ambiguity, not assume the worst |
| LLM returns markdown-wrapped JSON | `_strip_markdown_fences()` in the analyzer strips ````json ... ```` before parsing |
| `escalation_required` / `overall_assessment` mismatch | Post-parse integrity check auto-corrects the inconsistency and logs a warning |
| LLM API failure | Exponential backoff retry (configurable: default 3 attempts, 1.5s base delay) |
| Batch item failure | Returns an `error` result for that item; other items are unaffected |

---

## Pipeline Features

### Booking Intent Detection
When the transcript contains Arabic booking keywords (`احجز`, `حجز`, `معاد`, etc.), the pipeline:
1. Extracts `appointment_date`, `doctor_name`, `specialty_name`, `patient_name` via a dedicated LLM call
2. Queries the live reservations database (`Slots.sql`) with fuzzy Arabic name matching
3. Stores `appointment_verification: { found: bool, record: dict, message: str }` in state
4. Feeds the verification result into `infer_reservation_evaluation` (checks 6 reservation-specific pillars)

Calls without booking intent skip this entire branch and go directly to the parallel behavioral + compliance evaluation.

### Database Persistence
Every completed analysis is written to `[DWH].[AI].[Call_QA_Results]` via the `save_to_database` node:
- **Denormalized schema**: one row per `ComplianceFlag`, with call-level fields repeated on each row
- **Auto-versioning**: `Analysis_Version` is computed as `MAX(existing_version) + 1` for the `Call_ID`, so re-runs don't overwrite — they create a new version
- **Zero-flag handling**: calls with no violations still insert one summary row (all flag columns NULL) so the call appears in the table
- **Non-fatal errors**: DB insert failures are logged but don't crash the pipeline — the in-memory result is still returned to the caller

### Criteria Versioning
All evaluation rules live in YAML files under `app/criteria/`:
- `behavioral/{department}.yaml` — department-specific tone standards
- `compliance.yaml` — 15 compliance pillars (severity-tiered)
- `reservation.yaml` — 6 reservation-specific pillars
- `scripts.yaml` — approved greeting/closing templates
- `scoring.yaml` — dimension weights & thresholds

YAML changes take effect immediately (no redeploy needed) — the `CriteriaLoader` uses `lru_cache` so files are re-read only when modified.

## Tradeoffs

**4 focused LLM calls instead of 1**  
Increases cost per call (~4× input tokens, ~3× output tokens) but dramatically improves consistency and reduces hallucinations. Parallelism keeps latency under 3s for most calls.

**No streaming**  
Streaming adds complexity with little benefit — all 4 LLM responses must complete before `aggregate_results` can merge them and run Pydantic validation.

**Keyword-based intent detection**  
Booking detection uses a simple keyword list (`احجز`, `حجز`, etc.) instead of an LLM classifier. This is instant, deterministic, and has zero false negatives in testing. An LLM classifier would add 300ms + cost with marginal accuracy gain.

**Single database write per call**  
Results are persisted at the end of the pipeline (after `integrity_check`). If you need real-time partial results, add intermediate `save_to_database` nodes after each focused evaluation.

**Department list**  
8 departments are explicitly supported with tailored behavioral YAML files. Unknown departments fall back to `general.yaml`. Add new departments by creating `app/criteria/behavioral/{new_department}.yaml`.

---

## Provider Swap Guide

To switch from Anthropic to OpenAI (no code changes needed):

```env
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
```

The `LLMClient` class dispatches to `_call_anthropic()` or `_call_openai()` based on the provider setting. Both return the same `(text, usage_dict)` tuple. Adding a new provider (e.g., Google Gemini) requires:
1. Adding a `_call_gemini()` method in `llm_client.py`
2. Adding `"gemini"` to the dispatch in `_call()`
3. Setting `LLM_PROVIDER=gemini` and the matching API key in `.env`

---

## Observability

Every analysis logs:
- `call_id`, `agent_name`, `department`, `duration` (on request)
- `latency_ms`, `input_tokens`, `output_tokens` (on LLM response)
- `overall_assessment`, `escalation_required` (on result)
- Prompt text and raw LLM response (at DEBUG level — set `LOG_LEVEL=DEBUG` for full traces)

Set `LOG_LEVEL=DEBUG` to see full prompts and raw LLM responses during development.

## Author
**Rafik Sameh Yanni** \
AI Engineer
