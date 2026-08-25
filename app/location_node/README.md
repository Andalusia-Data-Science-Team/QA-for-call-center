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

`app.agent.nodes.validate_location_node` is the only caller. It gates on
`detect_location_request()` before touching the CRM, then calls
`validate_location_request()` and stores the result in
`AgentState["location_validation"]`.

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
