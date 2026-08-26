# Bank Node

Fully self-contained feature: deterministic KSA bank-account QA validation.
Mirrors `app/offers_node/`'s self-containment (data access + detection +
matching logic living together) and is a sibling of, not nested inside,
`app/location_node/` — the two features are independent graph nodes, not one
merged node, and this one has no CRM dependency on the other.

## Files

| File | Purpose |
|---|---|
| `crm_bank.py` | Fetches + caches active bank-account rows from Dynamics 365 (`cr301_bankaccounts`). |
| `bank_validation.py` | Business-Unit resolution + bank-request detection (which field — IBAN/account number/bank name/account owner?) + validation (does the agent's answer match an approved CRM record for that BU?). No LLM involved — fully deterministic. |

## Flow — Business Unit first

```
Transcript
   │  split_transcript_by_speaker()  (app.services.text_helpers)
   ▼
patient_text, agent_text
   │
   ▼
resolve_business_unit(patient_text)          — BUSINESS_UNIT_KEYWORD_MAP
   or resolve_business_unit(full transcript)    (app.models.input), exact
                                                  match → fuzzy fallback
   ▼
canonical BU  (e.g. "AKW", "LIVE", "AFW", "ALW", or one of LCH/MKR/SNB/None)
   │
   ▼
is_supported_bank_bu(bu)?  — scope is {AFW, LIVE, AKW, ALW} only
   │  NO  → NOT_APPLICABLE (not a violation — out of scope)
   ▼  YES
detect_arabic_bank_request(patient_text) OR identifiers already present?
   │  NO  → NOT_APPLICABLE (a BU mention alone is not a bank request —
   │         e.g. booking "في صحة الطفل" resolves AKW but isn't a bank ask)
   ▼  YES
fetch_bank_accounts()  (crm_bank.py, cached)
   ▼
_bank_rows_for_bu(bu, banks)   — canonical BU → CRM BU code, then filtered
   │                              against cr301_bu / servhub_bulookupname /
   │                              cr18c_bunname (NOT owningbusinessunitname —
   │                              verified constant across all rows)
   ▼
narrow by named bank, if the patient mentioned one  (multi-account BUs)
   ▼
detect requested field: IBAN / account number / bank name / account owner
   ▼
exact validation (financial identifiers) or approximate match
   (bank name via alias table; account owner via substring — free text)
   ▼
Bank result  → AgentState["bank_validation"]
```

## Business Unit vocabulary (important — read before touching BU logic)

Two different vocabularies refer to the same real-world branch:

- **App-level** (`BUSINESS_UNIT_KEYWORD_MAP`, `app/models/input.py`): Arabic
  chat aliases → `LIVE` for the main Jeddah hospital, plus `AKW`/`ALW`/`LCH`/
  `MKR`/`SNB`/`AFW`.
- **CRM-level** (`cr301_bankaccounts` — verified via a live query): the same
  branch is stored as `"AHJ"`, never `"LIVE"`. `owningbusinessunitname` is a
  constant (`"ERS"`) on every row and carries no BU signal at all.

`bank_validation.py`'s `_APP_BU_TO_CRM_BU = {"LIVE": "AHJ"}` is the **one**
place this translation happens — do not add a second special case for it
elsewhere. AFW/AKW/ALW need no translation; their app-level and CRM-level
codes already match. Bank validation's supported scope is
`{AFW, LIVE, AKW, ALW}` — LCH/MKR/SNB remain valid BUs elsewhere in the
project (booking, offers, ...), just outside this feature's scope.

## Entry point

`app.agent.nodes.validate_bank_information_node` is only reachable through
`app.agent.graph`'s conditional edge out of `detect_intent` —
`_bank_intent_router`, which reuses `bank_validation_needed()` /
`detect_bank_signals()` (which now requires *both* a supported BU *and*
bank-information context) rather than duplicating the check. When there is
no bank intent at all, the graph routes to `skip_bank_validation` instead:
`validate_bank_information` never executes, never appears in `node_trace`,
and never triggers a CRM fetch or business-unit bank resolution —
`skip_bank_validation` just writes the `NOT_APPLICABLE` state directly.
`validate_bank_information_node` keeps its own internal copy of the same
gate as a defensive fallback only (in case it's ever reached some other
way), not as the primary mechanism. When intent is present, the node calls
`validate_ksa_bank_information()` and stores the result in
`AgentState["bank_validation"]`.

Bank and location routing are independent conditional edges off the same
`detect_intent` node — all four combinations (neither/bank only/location
only/both) are supported, and each node/skip-path pair fans into the same
`loc_bank_ready` barrier before the booking split, exactly as before.

## Adding a new bank-related check

1. If it needs new CRM fields, extend `app/SQL/bank_accounts_query.sql` —
   the query stays in SQL, not embedded in Python.
2. Add the detection/matching logic to `bank_validation.py`.
3. If it needs a new BU alias, add it to `BUSINESS_UNIT_KEYWORD_MAP` in
   `app/models/input.py` (the project's single source of truth for BU
   aliases) — don't create a second mapping here.
4. `crm_bank.py` should not need to change for most additions — it's a thin,
   generic fetch+cache layer.
