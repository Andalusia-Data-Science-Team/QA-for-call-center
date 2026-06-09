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
│   ├── models/
│   │   ├── input.py            # Pydantic input models (CallTranscript)
│   │   └── output.py           # Pydantic output models (QAAnalysisResult)
│   ├── prompts/
│   │   └── qa_prompt.py        # All prompt logic — system prompt, user prompt builder
│   └── services/
│       ├── llm_client.py       # Provider-agnostic LLM client (Anthropic + OpenAI)
│       └── analyzer.py         # Orchestration: prompt → LLM → parse → validate
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

## Prompting Strategy

### Why not just ask "is this call good or bad?"

The naive approach produces inconsistent, over-flagging results. A QA system that generates false positives destroys agent morale — exactly the problem the clinic wanted to solve.

Instead the prompt is structured around three principles:

**1. Evidence-first instruction**  
The system prompt's first rule: *only flag issues you can directly observe*. The LLM is told explicitly to note ambiguity rather than assume the worst. This prevents hallucinated violations.

**2. Proportionate escalation thresholds**  
`escalate` is reserved for HIPAA violations, dangerous misinformation, or explicit rudeness. The prompt gives concrete examples of each. Everything else is `needs_review` or `pass`. This prevents the common failure mode of over-escalating minor imperfections.

**3. Department-specific checklists**  
Each department has a tailored checklist injected into the user prompt:
- `Records`: Pay extra attention to PHI disclosure without identity verification
- `Authorizations`: Authorization status must not be misrepresented
- `Scheduling`: Appointment details (date/time/location) should be confirmed

This means a Scheduling call is evaluated on scheduling criteria, not generic QA criteria.

### Structured output enforcement

The user prompt includes the exact JSON schema the LLM must return. This is then validated by Pydantic — if the LLM returns invalid JSON or a wrong type, the error is caught and returned as a clean 502, not a server crash.

For Anthropic, we rely on the model's instruction-following ability (Claude is very reliable at JSON-only output with an explicit schema). For OpenAI, we additionally set `response_format: {"type": "json_object"}` as a second layer of enforcement.

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

## Tradeoffs

**Single LLM call per transcript (not a chain)**  
I chose one well-crafted prompt over a multi-step chain (e.g., first extract facts, then evaluate). This reduces latency and cost with minimal quality loss for this scope. A multi-step chain would add value for longer calls where a "fact extraction pass" could help focus the evaluation.

**No streaming**  
Streaming adds complexity with little benefit for a structured JSON endpoint — the full response must be received before Pydantic can validate it anyway.

**No caching**  
Each call is unique transcript content, so caching would have low hit rates. Redis caching would be valuable for repeated identical inputs (e.g., regression testing) but adds infrastructure complexity out of scope here.

**Pydantic v2**  
Used `model_validate()` and `model_copy()` (Pydantic v2 API). If running on a Pydantic v1 environment, minor changes to those calls are needed.

**Department list**  
8 departments are explicitly supported with tailored checklists. Any unknown department falls back to general healthcare standards. The list is easy to extend in `qa_prompt.py`.

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
