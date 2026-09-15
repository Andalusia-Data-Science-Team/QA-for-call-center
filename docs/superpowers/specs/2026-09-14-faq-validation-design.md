# FAQ Escalation Validation Design

## Goal

Add a conditional FAQ validation branch to the QA LangGraph pipeline. When an
agent says that a request was raised, sent, transferred, or escalated to the
responsible department, the branch must verify a same-day record in
`app/FAQs/FAQ_latest.csv` and compare that record with the authoritative call
metadata and transcript.

The feature must follow the existing focused-evaluation-node pattern and must
not modify the regulation YAML files or the FAQ CSV data.

## Trigger and Routing

A pure-Python detector normalizes Arabic and English transcript text and looks
for the combined concepts of:

- an action such as raising, submitting, sending, transferring, forwarding, or
  escalating a request; and
- a destination such as the concerned, responsible, or specialized department,
  team, or party.

The detector is phrase-family based rather than an exact string comparison, so
variants of `تم رفع الطلب للقسم المختص` route to FAQ validation. Action words
alone do not trigger the branch unless the text also indicates a request or a
responsible destination. The initial vocabulary remains deliberately narrow to
avoid sending ordinary booking transfers or unrelated uses of `رفع` through
the FAQ path.

The FAQ branch runs after the existing CRM-lead validation and before overall
scoring:

```text
validate_crm_lead
        |
detect_faq_escalation
        |
        +-- not triggered ----------------------+
        |                                       |
        +-- triggered --> validate_faq_record --+
                                                |
                                  infer_overall_scoring
```

The detection node writes `is_faq_escalation`. The validation node writes
`faq_lookup` and `faq_eval`. Calls that do not trigger the detector receive a
neutral skipped FAQ evaluation and proceed directly to overall scoring.

## CSV Lookup

The lookup utility lives in `app/FAQs/faq_validation.py` and reads the CSV with
Python's CSV support so all values remain strings. The CSV path is resolved
relative to the module and can be injected in tests.

Candidate rows must satisfy all of these conditions:

1. `mobile_phone` matches the call's `Patient_Phone` after removing whitespace,
   punctuation, and Saudi country-code variations.
2. `Date` exactly equals the call's `call_date` in `YYYY-MM-DD` form. Older or
   newer records are never used as fallback records.
3. Either `AgentEmail` matches the call's `agent_email` case-insensitively or
   `AgentName` matches `agent_name` after Unicode, Arabic, whitespace, and case
   normalization. A missing call email does not prevent a name match.

Email equality is preferred over name-only equality when candidates are
ranked. If several rows remain at the same identity rank, the highest numeric
`ID` is the selected record; non-numeric IDs rank below numeric IDs. The lookup
returns one of these explicit statuses:

- `found`: includes the selected normalized record and match metadata.
- `not_found`: the CSV was read successfully but no row met every mandatory
  phone/date/identity condition.
- `unavailable`: the file could not be read or its required columns are absent.

`not_found` is a QA outcome. `unavailable` is a technical failure and enters the
existing graph error path.

## Record Evaluation

For a found record, a focused prompt in `app/prompts/qa_prompt.py` receives the
call metadata, transcript, selected FAQ row, and the existing compliance
regulation text. It does not evaluate tone, greetings, offers, reservations, or
other unrelated criteria.

The evaluation checks:

- `AgentName` and `AgentEmail` against the call agent identity;
- `mobile_phone` and `Date` against the lookup keys;
- `CustomerName` against the patient/client identified in the transcript;
- `BU` against the call business unit, including the established `LIVE` and
  `AHJ` equivalence;
- `Inquiry` as a semantic representation of the customer's actual inquiry;
- whether `Response` is present or absent and whether its content agrees with
  the response communicated in the chat; and
- whether `End Call Result` is `In Progress` or `Closed` consistently with the
  actual conversation outcome and response state.

Minor case, spacing, spelling, transliteration, and harmless wording differences
are matches. A blank field is a mismatch when that field is required by the
observed call facts. The model returns field-level checks and no positive flags.
The normalizer enforces the public compliance-flag schema and limits FAQ flags
to four distinct underlying failures.

## Existing Regulation Mapping

No rules are added to or changed in YAML. FAQ findings cite and use only these
existing compliance-regulation entries:

- `C2B_017` — `Didn’t take the proper action on system such linking CST Contract
  etc.` A missing same-day FAQ record after an agent submission claim always
  produces this deterministic violation.
- `C2B_021` — `Missing / Wrong Field`. This covers wrong or missing agent,
  customer, business unit, inquiry, response, status, date, or phone data in an
  existing FAQ record.
- `C2C_023` — `Described wrong Information`. This applies when the agent tells
  the customer a response, status, or performed action that the FAQ evidence
  proves false.
- `C2C_024` — `Didn’t Escalate when required`. This applies only when the
  transcript and FAQ evidence establish that escalation was required but was
  not performed.

The same evidence cannot produce duplicate rule flags. `C2C_023` and
`C2C_024` require affirmative evidence; ambiguous or absent CSV values are
handled as `C2B_021` rather than inferred as customer-critical violations.

## Pipeline Integration

`AgentState` gains:

- `is_faq_escalation: Optional[bool]`
- `faq_lookup: dict[str, Any]`
- `faq_eval: dict[str, Any]`

`app/agent/nodes.py` gains:

- `detect_faq_escalation`, a no-LLM wrapper around the transcript detector; and
- `validate_faq_record`, which performs the lookup, returns deterministic
  missing-record output without an LLM call, invokes the focused prompt only
  for a found record, and normalizes model output.

The FAQ summary is added to `build_scoring_prompt`. `infer_overall_scoring`
serializes `faq_eval` along with existing focused evaluations. `aggregate_results`
merges `faq_flags` into `compliance_flags` before its existing deduplication and
Pydantic validation.

The graph registers both nodes and an FAQ conditional router. Any technical
error from FAQ lookup or evaluation uses `_error_router` and `handle_error`.
The no-trigger and successful-trigger paths converge exactly once at
`infer_overall_scoring`.

## Error Handling

- Missing or invalid call data remains the responsibility of `load_call`.
- A readable, schema-valid CSV with no matching row produces `C2B_017` and is
  not a graph error.
- A missing/unreadable CSV, decode failure, or missing required CSV column sets
  `error` and `error_node="validate_faq_record"`.
- Invalid LLM JSON uses the existing `_focused_llm_call` retry and error path.
- Model-supplied rule IDs or flag types outside the four allowed existing rules
  are discarded or converted to the applicable `C2B_021` field mismatch.
- Logs contain the call ID, lookup status, selected FAQ ID, and flag count, but
  do not log the full transcript or patient phone.

## Testing

Tests are added in `tests/test_faq_validation.py` and follow the existing pytest
style. They cover:

- exact and paraphrased Arabic escalation triggers plus non-trigger controls;
- English trigger equivalents;
- Arabic/name/email/phone normalization;
- mandatory same-day matching with no cross-date fallback;
- matching by email or normalized name and rejection of a different agent;
- highest-ID selection among equally ranked rows;
- `found`, `not_found`, and `unavailable` lookup states;
- deterministic `C2B_017` output for a missing row;
- normalization of `C2B_021`, `C2C_023`, and `C2C_024` flags without duplicates;
- prompt inclusion of every required FAQ field and the existing regulation
  context;
- FAQ routing for triggered and non-triggered calls;
- FAQ summary inclusion in scoring and FAQ flag inclusion in aggregation; and
- graph compilation with both branches converging on scoring.

Existing targeted tests and the full available pytest suite are run after the
new tests pass. No test writes to the production CSV.

## Files Changed

- Create `app/FAQs/faq_validation.py`.
- Create `tests/test_faq_validation.py`.
- Modify `app/agent/state.py`.
- Modify `app/agent/nodes.py`.
- Modify `app/agent/graph.py`.
- Modify `app/prompts/qa_prompt.py`.
- Do not modify any YAML regulation file or `app/FAQs/FAQ_latest.csv`.
