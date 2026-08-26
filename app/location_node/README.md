# Location Node

Fully self-contained feature: deterministic KSA branch/location QA
validation. Mirrors `app/offers_node/`'s self-containment and is a sibling
of, not nested inside, `app/bank_node/` — the two features are independent
graph nodes, not one merged node.

## Files

| File | Purpose |
|---|---|
| `crm_location.py` | Fetches + caches active KSA branch/location rows from Dynamics 365 (`cr301_andalusialocations`). |
| `location_validation.py` | Detection (did the patient ask for a branch/address?) + branch resolution (which branch, layered exact-token → fuzzy matching) + answer validation (does the agent's address match?). No LLM — fully deterministic. |

## How it fits together

```
app/agent/nodes.py::validate_location_node            (graph node)
              │
              ▼  calls
app/location_node/location_validation.py               (detection + resolution + validation)
              │
              ▼  calls
app/location_node/crm_location.py                       (cached CRM fetch)
              │
              ▼  calls
app/services/crm_connector.py                            (shared, domain-agnostic:
                                                            MSAL auth, connection,
                                                            retry — also used by
                                                            app/offers_node/crm_offers.py
                                                            and app/bank_node/crm_bank.py)
```

`app/services/text_helpers.py` supplies the only things this module shares
with `bank_node`: `split_transcript_by_speaker()` (turn splitting) and
Arabic text normalisation. Everything location-specific — branch resolution,
address-token matching, ubiquity filtering — lives here and nowhere else.

## Entry point

`app.agent.nodes.validate_location_node` is only reachable through
`app.agent.graph`'s conditional edge out of `detect_intent` —
`_location_intent_router`, which reuses the SAME gate
(`location_validation_needed` / `detect_location_intent`) rather than
duplicating the check. When there is no location intent at all, the graph
routes to `skip_location_validation` instead: `validate_location` never
executes, never appears in `node_trace`, and never triggers a CRM fetch —
`skip_location_validation` just writes the `NOT_APPLICABLE` state directly.
`validate_location_node` keeps its own internal copy of the same gate as a
defensive fallback only (in case it's ever reached some other way), not as
the primary mechanism.

The gate itself: `patient_has_location_intent OR
agent_has_location_information`. Both booleans reuse the same context-aware
`detect_location_request()` detector (the unambiguous "عنوان"/"address"
trigger alone; weaker words like "فين"/"وين"/"مكان" only count alongside a
place-noun like "فرع"/"عيادة"/"مستشفى"), applied symmetrically to whichever
side is speaking — a patient asking for a location and an agent
proactively sharing one are treated as the same kind of signal. When
intent is present, the node fetches CRM data and calls
`validate_location_request()`, which stores the result in
`AgentState["location_validation"]`.

"موقع"/"لوكيشن"/"location" are treated as **ambiguous**, not strong, on
their own: a home-care/delivery conversation routinely uses the exact same
word for the CUSTOMER's own location ("ابعت موقعك", "بترسل لهم الموقع",
"شارك اللوكيشن", "نحتاج موقع حضرتك") as an Andalusia facility uses for its
own address ("موقع الفرع", "لوكيشن فرع السنابل"). `detect_location_request()`
tells the two apart: "موقع/لوكيشن" immediately followed by a place-noun
("الفرع"/"عيادة"/"مستشفى") always counts, a send/share verb or a dative
pronoun near "موقع/لوكيشن" with no place-noun never does. This must not be
weakened back to "any mention of موقع/لوكيشن counts" — that regressed to
flagging home-care/delivery calls as Andalusia branch-location requests.

## Debug visibility

`validate_location_request()` logs the full resolution/comparison chain at
`INFO` level (`logger = logging.getLogger("app.location_node.location_validation")`)
so a run can be inspected without a debugger — in order: how many KSA
records were considered, why a specific branch was resolved
(`resolution_source` — `branch_name`, `patient_text`, or `provided_address` —
plus the `matched_terms` that actually won), the exact CRM record selected
as the source of truth (branch/area/country/description/region), the
location text extracted from the chat, both normalized forms actually
compared, and the final `resolved_branch vs. crm_location vs. chat_location
→ match → outcome` line. The same two fields are also returned on the
result dict itself (`crm_location`, `provided_location`) so debugging
doesn't depend on log output alone. Validation always compares the chat
text against the **specific resolved branch's** CRM record — never "does
this address match anywhere in KSA" — so a real Andalusia address given
for the wrong branch still fails.

Alongside those structured `logger.info()` calls, `validate_location_request()`
also prints a concise, human-readable `[location] ...` terminal summary —
a sibling to this project's existing `[offers] ...` / `[crm_location] ...`
print-style logs, meant for quickly scanning a local test run without
parsing timestamped log lines. It's built from two tiny helpers
(`_loc_print` for a single line, `_loc_print_section` for a `title:` header
plus its indented fields) called at the same points the structured logs
are, so an early return (branch unresolved, ambiguous, no address, etc.)
only ever prints the sections that were actually computed — never the
whole transcript, only the already-extracted location-relevant fields.
This is purely additive local-dev/testing observability; it never replaces
the structured logs, and it never changes the returned result or any
matching behaviour. When the graph-level router skips `validate_location`
entirely (no location intent), `skip_location_validation` in
`app.agent.nodes` prints a single `[location] skipped | call_id=...
reason=no_location_intent` line instead of the full block.

## What "provided_location" actually is

It is **not** the whole agent-side transcript. `_extract_agent_location_text()`
scopes it to the location-relevant agent turn(s) only:

- It anchors on the location trigger — the patient's most recent
  location-intent turn, if any — and only looks at the agent turns that
  follow it (a pure proactive share with no patient request starts from
  the top of the call instead).
- Turns are grouped into runs of turns that are consecutive in the raw
  transcript; a patient turn in between ends a run, so two agent mentions
  "far apart" in the conversation are never merged into one answer.
- Within the first qualifying run, only turns that themselves carry a
  location signal (explicit vocabulary, or overlap with a real branch's
  distinctive tokens) or name a branch/place are kept — an unrelated aside
  in the same run (a booking or prescription remark) is dropped.

This is also why a real CRM address that happens to be mentioned in a
later, unrelated agent turn can never be picked up as if it answered an
earlier, different request — branch resolution's `provided_address`
fallback and the final comparison both use this same extracted text, not
the raw transcript.

## Matching approach (why it isn't a simple substring check)

Real branch names carry boilerplate a patient never repeats ("عيادات
أندلسية فرع الأمير سلطان" vs. how a patient actually asks: "فرع الأمير
سلطان"). `resolve_branch_candidates()` strips ubiquitous/boilerplate tokens
(the same idea `crm_offers.py` uses to stop generic words from confirming
an offer match) and ranks candidates by how many of a branch's *distinctive*
tokens the query covers — count first, then ratio, so a longer/more-specific
name only wins over a shorter one it's a superset of when the query actually
supplies evidence for the extra specificity (e.g. mentioning "الأسنان" to
pick the dental branch over the general one at the same address).
