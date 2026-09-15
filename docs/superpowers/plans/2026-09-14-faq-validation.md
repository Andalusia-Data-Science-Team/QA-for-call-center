# FAQ Escalation Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route escalation-claim chats through a strict same-day FAQ CSV lookup and add evidence-based FAQ compliance findings to the existing QA result.

**Architecture:** A pure-Python FAQ module owns phrase detection, normalization, CSV lookup, and deterministic flag normalization. Two thin LangGraph nodes call that module; the found-record path uses one focused LLM prompt, while skipped and missing-record paths remain deterministic. FAQ output is then included in the existing overall-scoring prompt and final flag aggregation.

**Tech Stack:** Python 3, standard-library `csv`/`re`/`unicodedata`, Pydantic models, LangGraph, pytest, existing `LLMClient` and focused-call helper.

**Spec:** `docs/superpowers/specs/2026-09-14-faq-validation-design.md`

## Global Constraints

- Do not modify `app/FAQs/FAQ_latest.csv`.
- Do not update or add any YAML regulation rule.
- A FAQ row is eligible only on the exact call day.
- Lookup requires normalized phone and either matching agent email or agent name.
- Missing same-day record maps deterministically to existing rule `C2B_017`.
- Existing-record field problems map to `C2B_021`; proven false information and missed required escalation may map to `C2C_023` and `C2C_024`.
- Preserve all pre-existing uncommitted edits in shared pipeline files.

---

### Task 1: FAQ phrase detection and strict CSV lookup

**Files:**
- Create: `app/FAQs/faq_validation.py`
- Create: `tests/test_faq_validation.py`

**Interfaces:**
- Produces: `detect_faq_escalation(transcript: str) -> bool`
- Produces: `normalize_faq_text(value: object) -> str`
- Produces: `normalize_faq_phone(value: object) -> str`
- Produces: `lookup_faq_record(*, csv_path: Path | None, patient_phone: str, call_date: str, agent_name: str, agent_email: str | None) -> dict[str, Any]`
- Lookup output: `{"status": "found", "record": dict, "match": dict, "message": str}` or `{"status": "not_found"|"unavailable", "record": None, "message": str}`.

- [ ] **Step 1: Write failing trigger and normalization tests**

Add parametrized tests that name the exact behavior:

```python
@pytest.mark.parametrize(
    "transcript",
    [
        "Agent: تم رفع الطلب للقسم المختص وسيتم التواصل معك",
        "Agent: رفعنا استفسارك للجهة المعنية",
        "Agent: أرسلت الشكوى إلى الفريق المسؤول",
        "Agent: Your request was escalated to the concerned department.",
    ],
)
def test_detect_faq_escalation_accepts_semantic_phrase_variants(transcript):
    assert detect_faq_escalation(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "Patient: أريد رفع التقرير الطبي",
        "Agent: تم تحويلك إلى قسم الحجز",
        "Agent: القسم المختص متاح من التاسعة",
    ],
)
def test_detect_faq_escalation_rejects_unrelated_language(transcript):
    assert detect_faq_escalation(transcript) is False


def test_normalizers_align_arabic_names_and_saudi_phone_formats():
    assert normalize_faq_text("  أحمَد   عليّ ") == "احمد علي"
    assert normalize_faq_phone("+966 50-123-4567") == "501234567"
    assert normalize_faq_phone("0501234567") == "501234567"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k "detect_faq_escalation or normalizers"`

Expected: collection fails because `app.FAQs.faq_validation` does not exist.

- [ ] **Step 3: Implement normalization and concept-based detection**

Create `app/FAQs/faq_validation.py` with immutable required-column and phrase-family constants. Normalize NFKD Unicode, strip combining marks, normalize Arabic alef/yaa/taa marbuta, lowercase, and collapse punctuation/whitespace. Implement detection with these constraints:

```python
def detect_faq_escalation(transcript: str) -> bool:
    text = normalize_faq_text(transcript)
    strong_action = _contains_any(text, _RAISE_OR_ESCALATE_TERMS)
    transfer_action = _contains_any(text, _SEND_OR_TRANSFER_TERMS)
    has_request = _contains_any(text, _REQUEST_TERMS)
    has_destination = _contains_any(text, _DESTINATION_TERMS)
    return has_destination and (strong_action or (transfer_action and has_request))
```

Use narrow Arabic and English terms matching the tests, including normalized forms of `رفع`, `تصعيد`, `ارسل`, `حول`, `طلب`, `استفسار`, `شكوى`, `قسم`, `جهة`, `فريق`, `مختص`, `معني`, `مسؤول`, `raise`, `submit`, `send`, `forward`, `escalat`, `request`, `inquiry`, `complaint`, `department`, `team`, `concerned`, and `responsible`.

- [ ] **Step 4: Run the trigger tests and verify GREEN**

Run: `pytest -q tests/test_faq_validation.py -k "detect_faq_escalation or normalizers"`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing lookup tests**

Add a `write_faq_csv(tmp_path, rows)` helper using `csv.DictWriter` and the production header. Add tests proving:

```python
def test_lookup_requires_same_day_phone_and_agent_identity(tmp_path):
    csv_path = write_faq_csv(tmp_path, [
        faq_row(ID="10", Date="2026-09-13"),
        faq_row(ID="11", AgentEmail="other@example.com"),
        faq_row(ID="12", mobile_phone="0555999999"),
    ])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="0555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
    )
    assert result["status"] == "not_found"


def test_lookup_prefers_email_then_highest_id(tmp_path):
    csv_path = write_faq_csv(tmp_path, [
        faq_row(ID="20", AgentName="Mahmoud Atef Helmy", AgentEmail=""),
        faq_row(ID="21", AgentName="Different Spelling", AgentEmail="Mahmoud.Atef@Andalusiagroup.net"),
        faq_row(ID="22", AgentName="Different Spelling", AgentEmail="Mahmoud.Atef@Andalusiagroup.net"),
    ])
    result = lookup_faq_record(
        csv_path=csv_path,
        patient_phone="555123456",
        call_date="2026-09-14",
        agent_name="Mahmoud Atef Helmy",
        agent_email="mahmoud.atef@andalusiagroup.net",
    )
    assert result["status"] == "found"
    assert result["record"]["ID"] == "22"
    assert result["match"]["identity"] == "email"
```

Also test a name-only match when call email is absent, missing required columns as `unavailable`, and missing file as `unavailable`.

- [ ] **Step 6: Run lookup tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k lookup`

Expected: tests fail because `lookup_faq_record` is not implemented.

- [ ] **Step 7: Implement strict CSV lookup**

Use `csv.DictReader(path.open(encoding="utf-8-sig", newline=""))`. Validate this exact required-column set:

```python
FAQ_REQUIRED_COLUMNS = frozenset({
    "ID", "CustomerName", "mobile_phone", "BU", "Date", "Inquiry",
    "Response", "AgentName", "AgentEmail", "End Call Result",
})
```

Filter by normalized phone and stripped exact date, then score identity as `2` for matching nonblank emails or `1` for normalized names. Reject identity score `0`. Select with:

```python
selected = max(
    candidates,
    key=lambda item: (
        item["identity_rank"],
        item["id_is_numeric"],
        item["numeric_id"],
    ),
)
```

Catch `OSError`, `UnicodeError`, and `csv.Error` and return `unavailable` with a concise message. Do not include transcript content or the full phone in the result message.

- [ ] **Step 8: Run all core helper tests**

Run: `pytest -q tests/test_faq_validation.py`

Expected: trigger, normalization, and lookup tests pass.

- [ ] **Step 9: Commit the isolated helper slice**

```bash
git add app/FAQs/faq_validation.py tests/test_faq_validation.py
git commit -m "feat: add FAQ escalation lookup tools"
```

---

### Task 2: Deterministic violations, model normalization, and FAQ prompt

**Files:**
- Modify: `app/FAQs/faq_validation.py`
- Modify: `app/prompts/qa_prompt.py:988`
- Modify: `tests/test_faq_validation.py`

**Interfaces:**
- Consumes: lookup dictionaries from Task 1.
- Produces: `missing_faq_evaluation(message: str) -> dict[str, Any]`
- Produces: `normalize_faq_evaluation(data: dict[str, Any]) -> dict[str, Any]`
- Produces: `build_faq_validation_prompt(call: CallTranscript, lookup: dict[str, Any], compliance_pillars: str = "") -> str`
- Evaluation output keys: `faq_status`, `summary`, `field_checks`, `faq_flags`.

- [ ] **Step 1: Write failing deterministic and normalization tests**

Add tests asserting the missing-record result has exactly one moderate `C2B` flag whose description contains `C2B_017`. Add a normalization test with duplicated `C2C_023`, a valid `C2C_024`, and an unknown rule; assert valid rules are deduplicated and the unknown field mismatch becomes `C2B_021`.

```python
def test_missing_faq_record_is_deterministic_c2b_017():
    result = missing_faq_evaluation("No same-day FAQ record.")
    assert result["faq_status"] == "violation"
    assert result["faq_flags"] == [{
        "type": "C2B",
        "severity": "moderate",
        "description": "C2B_017: FAQ request was not recorded on the call date.",
        "transcript_excerpt": "N/A",
    }]
```

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k "missing_faq or normalize_faq_evaluation"`

Expected: imports or assertions fail because the evaluation helpers are absent.

- [ ] **Step 3: Implement rule-safe evaluation helpers**

Define the only allowed mappings in code as references to the unchanged YAML catalog:

```python
FAQ_RULES = {
    "C2B_017": ("C2B", "moderate"),
    "C2B_021": ("C2B", "moderate"),
    "C2C_023": ("C2C", "critical"),
    "C2C_024": ("C2C", "critical"),
}
```

Normalize field checks first. For each failed check, accept its allowed
`rule_id`; otherwise use `C2B_021`. Build schema-safe flags with the rule ID at
the start of `description`, deduplicate by `(rule_id, field, evidence)`, cap at
four flags, and set `faq_status` to `violation` when flags exist and `match`
otherwise. Never create positive flags.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

Run: `pytest -q tests/test_faq_validation.py -k "missing_faq or normalize_faq_evaluation"`

Expected: all selected tests pass.

- [ ] **Step 5: Write a failing prompt contract test**

Build a representative `CallTranscript` and lookup record. Assert that the
prompt contains the agent email, patient phone, same-day date, customer name,
BU, inquiry, response, end-call result, transcript, all four allowed rule IDs,
and the supplied regulation excerpt.

```python
prompt = build_faq_validation_prompt(
    call,
    {"status": "found", "record": record},
    compliance_pillars="C2B_017 existing-regulation excerpt",
)
for value in ("AgentEmail", "CustomerName", "Inquiry", "Response", "End Call Result"):
    assert value in prompt
assert "C2B_017 existing-regulation excerpt" in prompt
```

- [ ] **Step 6: Run the prompt test and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k prompt`

Expected: import fails because `build_faq_validation_prompt` does not exist.

- [ ] **Step 7: Implement the focused FAQ prompt**

Add the builder after `build_crm_lead_validation_prompt`. Serialize the selected
record with `json.dumps(..., ensure_ascii=False, default=str)`. Direct the model
to output:

```json
{
  "faq_status": "match | violation",
  "summary": "concise evidence-based summary",
  "field_checks": [
    {
      "field": "CSV column name",
      "matches": false,
      "expected": "call/transcript value",
      "actual": "CSV value",
      "rule_id": "C2B_021 | C2C_023 | C2C_024",
      "reason": "concise reason",
      "transcript_excerpt": "verbatim evidence or N/A"
    }
  ],
  "faq_flags": []
}
```

State explicitly that metadata/blank-field mismatches use `C2B_021`, false
customer-facing claims require affirmative evidence for `C2C_023`, and
`C2C_024` requires affirmative evidence of a required but unperformed
escalation. Include `LIVE=AHJ` equivalence and prohibit unrelated evaluation.

- [ ] **Step 8: Run Task 2 tests**

Run: `pytest -q tests/test_faq_validation.py`

Expected: all helper, evaluation, and prompt tests pass.

- [ ] **Step 9: Commit the evaluation slice**

```bash
git add app/FAQs/faq_validation.py app/prompts/qa_prompt.py tests/test_faq_validation.py
git commit -m "feat: add FAQ record evaluation prompt"
```

---

### Task 3: FAQ state and pipeline nodes

**Files:**
- Modify: `app/agent/state.py:23-57`
- Modify: `app/agent/nodes.py:43-74`
- Modify: `app/agent/nodes.py:1487-1575`
- Modify: `tests/test_faq_validation.py`

**Interfaces:**
- Consumes: Task 1 and Task 2 functions.
- Produces: `detect_faq_escalation(state: AgentState) -> dict`
- Produces: `validate_faq_record(state: AgentState, llm_client: LLMClient, csv_path: Path | None = None) -> dict`
- State output: `is_faq_escalation`, `faq_lookup`, `faq_eval`, optional `usage_list`, and `node_trace`.

- [ ] **Step 1: Write failing node tests**

Use `asyncio.run` to test nodes without requiring an async pytest plugin.
Assert the detector writes `True` for a matching transcript and writes a neutral
skipped evaluation for a non-trigger. Monkeypatch lookup for three validator
tests:

- `not_found` returns deterministic `C2B_017` without calling the LLM;
- `unavailable` returns `error_node == "validate_faq_record"`;
- `found` sends the prompt through a fake `LLMClient`, normalizes the returned
  field checks, and preserves usage tracking.

```python
result = asyncio.run(detect_faq_escalation_node({"call": call}))
assert result["is_faq_escalation"] is True
assert result["node_trace"] == ["detect_faq_escalation"]
```

Alias the imported node in the test as `detect_faq_escalation_node` to avoid a
name collision with the pure helper.

- [ ] **Step 2: Run node tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k "node or validate_faq_record"`

Expected: import fails because the nodes and state fields do not exist.

- [ ] **Step 3: Add FAQ state fields**

Add these fields beside CRM validation state:

```python
is_faq_escalation: Optional[bool]
faq_lookup: dict[str, Any]
faq_eval: dict[str, Any]
```

- [ ] **Step 4: Implement thin FAQ nodes**

Import the helper detector with an unambiguous alias:

```python
from app.FAQs.faq_validation import (
    detect_faq_escalation as transcript_has_faq_escalation,
    lookup_faq_record,
    missing_faq_evaluation,
    normalize_faq_evaluation,
)
```

The detector node returns a neutral skipped evaluation on false:

```python
return {
    "is_faq_escalation": detected,
    "faq_eval": {
        "faq_status": "pending" if detected else "skipped",
        "summary": "FAQ validation required." if detected else "No FAQ escalation claim detected.",
        "field_checks": [],
        "faq_flags": [],
    },
    "node_trace": _trace(state, "detect_faq_escalation"),
}
```

The validator calls `lookup_faq_record` using `call.Patient_Phone`,
`call.call_date`, `call.agent_name`, and `call.agent_email`. Handle statuses in
this exact order: `unavailable` to graph error, `not_found` to deterministic
evaluation, and `found` to `build_faq_validation_prompt` plus
`_focused_llm_call`. Pop `_usage` only after a successful found-record call.

- [ ] **Step 5: Run node tests and verify GREEN**

Run: `pytest -q tests/test_faq_validation.py -k "node or validate_faq_record"`

Expected: all selected tests pass.

- [ ] **Step 6: Commit the state/node slice**

```bash
git add app/agent/state.py app/agent/nodes.py tests/test_faq_validation.py
git commit -m "feat: add FAQ validation nodes"
```

---

### Task 4: Scoring, aggregation, and graph wiring

**Files:**
- Modify: `app/prompts/qa_prompt.py:381-485`
- Modify: `app/agent/nodes.py:1576-1619`
- Modify: `app/agent/nodes.py:1672-1757`
- Modify: `app/agent/graph.py:112-470`
- Modify: `tests/test_faq_validation.py`

**Interfaces:**
- Consumes: `state["faq_eval"]` from Task 3.
- Produces: `build_scoring_prompt(..., faq_summary: str = "") -> str`.
- Produces: `_faq_router(state: AgentState) -> Literal["validate", "skip"]`.
- Graph path: `validate_crm_lead -> detect_faq_escalation -> (validate_faq_record | infer_overall_scoring)`.

- [ ] **Step 1: Write failing scoring and aggregation tests**

Assert `build_scoring_prompt(..., faq_summary='{"faq_status":"violation"}')`
includes a `FAQ VALIDATION` section and the exact summary. Monkeypatch
`_focused_llm_call` while invoking `infer_overall_scoring` and assert the
captured prompt contains FAQ output. Build minimal aggregate state with one FAQ
flag and assert it appears in `result.compliance_flags`.

- [ ] **Step 2: Run scoring/aggregation tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k "scoring or aggregate"`

Expected: the scoring signature rejects `faq_summary` or the FAQ flag is absent.

- [ ] **Step 3: Add FAQ scoring and aggregation integration**

Add `faq_summary: str = ""` after `crm_lead_summary` in `build_scoring_prompt`
and render it as the next sub-evaluation. In `infer_overall_scoring`, serialize
`state.get("faq_eval") or {}` and pass it to the prompt builder. In
`aggregate_results`, bind `faq = state.get("faq_eval") or {}` and add
`faq.get("faq_flags", [])` to `all_flags` immediately after CRM lead flags.

- [ ] **Step 4: Run scoring/aggregation tests and verify GREEN**

Run: `pytest -q tests/test_faq_validation.py -k "scoring or aggregate"`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing router and compiled-graph tests**

Test `_faq_router({"is_faq_escalation": True}) == "validate"` and false/missing
values return `skip`. Compile with a non-calling fake LLM and inspect
`graph.get_graph().edges` to assert these directed pairs exist:

```python
expected_edges = {
    ("validate_crm_lead", "detect_faq_escalation"),
    ("detect_faq_escalation", "validate_faq_record"),
    ("detect_faq_escalation", "infer_overall_scoring"),
    ("validate_faq_record", "infer_overall_scoring"),
    ("validate_faq_record", "handle_error"),
}
assert expected_edges <= edge_pairs
```

- [ ] **Step 6: Run graph tests and verify RED**

Run: `pytest -q tests/test_faq_validation.py -k "faq_router or graph"`

Expected: router import or expected-edge assertion fails.

- [ ] **Step 7: Wire the conditional FAQ branch**

Import both new nodes. Add:

```python
def _faq_router(state: AgentState) -> Literal["validate", "skip"]:
    return "validate" if state.get("is_faq_escalation") else "skip"
```

Register `detect_faq_escalation` directly and register `validate_faq_record`
with `functools.partial(..., llm_client=llm_client)`. Replace the existing
`validate_crm_lead` success destination with `detect_faq_escalation`, then add:

```python
builder.add_conditional_edges(
    "detect_faq_escalation",
    _faq_router,
    {"validate": "validate_faq_record", "skip": "infer_overall_scoring"},
)
builder.add_conditional_edges(
    "validate_faq_record",
    _error_router,
    {"continue": "infer_overall_scoring", "handle_error": "handle_error"},
)
```

- [ ] **Step 8: Run all FAQ tests**

Run: `pytest -q tests/test_faq_validation.py`

Expected: all FAQ tests pass with no warnings caused by this feature.

- [ ] **Step 9: Commit the graph integration slice**

```bash
git add app/agent/graph.py app/agent/nodes.py app/prompts/qa_prompt.py tests/test_faq_validation.py
git commit -m "feat: wire FAQ validation into QA graph"
```

---

### Task 5: Regression and requirement verification

**Files:**
- Verify: `app/FAQs/faq_validation.py`
- Verify: `app/agent/state.py`
- Verify: `app/agent/nodes.py`
- Verify: `app/agent/graph.py`
- Verify: `app/prompts/qa_prompt.py`
- Verify: `tests/test_faq_validation.py`
- Verify unchanged: `app/FAQs/FAQ_latest.csv`
- Verify unchanged: `app/criteria/**/*.yaml`

**Interfaces:**
- Consumes all prior task outputs.
- Produces a verified implementation with no additional runtime interface.

- [ ] **Step 1: Run syntax compilation**

Run: `python -m compileall -q app/FAQs/faq_validation.py app/agent/state.py app/agent/nodes.py app/agent/graph.py app/prompts/qa_prompt.py tests/test_faq_validation.py`

Expected: exit code 0 with no output.

- [ ] **Step 2: Run targeted FAQ and CRM-lead regression tests**

Run: `pytest -q tests/test_faq_validation.py tests/test_crm_leads_validation.py test_crm_leads_validation.py`

Expected: all selected tests pass.

- [ ] **Step 3: Run the full available pytest suite**

Run: `pytest -q`

Expected: zero failures. If unrelated pre-existing failures exist, record their
exact test names and demonstrate that the targeted FAQ suite still passes.

- [ ] **Step 4: Verify protected data and regulations were not edited**

Run: `git status --short -- app/FAQs/FAQ_latest.csv app/criteria`

Expected: no new modifications attributable to this feature. Any pre-existing
status must match the baseline captured before implementation.

- [ ] **Step 5: Review the final diff**

Run: `git diff --check`

Expected: exit code 0.

Run: `git diff --stat HEAD~3..HEAD`

Expected: only the planned FAQ module, tests, state, node, graph, and prompt
changes plus the already committed design/plan documents are present in these
feature commits.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required a focused correction, add only the files changed for
that correction and commit them with:

```bash
git commit -m "fix: complete FAQ validation integration"
```

If no correction was required, do not create an empty commit.
