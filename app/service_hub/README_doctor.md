# Doctor Validation

Two independent checks, living in `app/service_hub/` alongside the sibling
bank/location/offers features:

1. **Deterministic doctor-information validation** (`doctor_validation.py`)
   — validates factual claims the Agent made about a specific doctor
   (name/degree/specialty/subspecialty/business unit/notes/scope/
   qualifications/examination age/walk-in fee) against the authoritative
   CRM record. No LLM.
2. **Semantic doctor-recommendation-suitability validation**
   (`app.prompts.qa_prompt.build_doctor_scope_prompt` +
   `app.agent.nodes.infer_doctor_scope_validation`) — a SEPARATE, LLM-based
   check: does the doctor's documented CRM scope of service reasonably
   cover what the patient described? This never merges with check 1, and
   the LLM is never allowed to pick a doctor itself — it only judges fit
   for the doctor check 1 already resolved.

## Files

| File | Purpose |
|---|---|
| `crm_doctors.py` | Fetches + caches the full doctor dataset (`cr301_newdoctordataset` joined with `cr301_table1` for fees) — spans both OPD and non-OPD doctors, not filtered by `cr301_opdflag`. |
| `doctor_validation.py` | Doctor-mention detection, CRM filtering/dedup, name resolution, per-field claim validation, and the applicability gate for the semantic check. |

## Entry points

`app.agent.nodes.validate_doctor_node` is only reachable through
`app.agent.graph`'s conditional edge out of `detect_intent` —
`_doctor_intent_router`, which reuses `doctor_validation_needed()` /
`detect_doctor_signals()` rather than duplicating the check. When there is
no named-doctor mention at all, the graph routes to
`skip_doctor_validation` instead: `validate_doctor` never executes, never
appears in `node_trace`, and never triggers a CRM fetch.
`validate_doctor_node` keeps its own internal copy of the same gate as a
defensive fallback only. Bank, location, and doctor routing are three fully
independent conditional edges off the same `detect_intent` node — every
combination is supported, and all three fan into the same `loc_bank_ready`
barrier before the booking split, exactly like bank/location already did.

`app.agent.nodes.infer_doctor_scope_validation` runs one hop past
`inference_gate` (via `fetch_crm_offers_for_call`), alongside behavioral/
compliance/script/offer, so all five inference branches stay at an EQUAL
hop count from `inference_gate` — a mismatched hop count there previously
caused the whole downstream chain (`infer_overall_scoring` →
`aggregate_results` → ... → `finalize`) to fire twice per call. Its own
applicability gate (`doctor_scope_validation_needed`) is checked INLINE
inside the node, not via a graph-level skip: when it fails, the node still
executes (appears in `node_trace`) but returns `NOT_APPLICABLE` without
ever calling the LLM — the same pattern `infer_offer_evaluation` already
uses for `NO_OFFER_AVAILABLE`.

## Authoritative filtering

Primary BU authority is `cr18c_buname` (NOT `cr301_businessunitname` — real
CRM data has been observed to disagree between the two on the same row).
Supported scope: `{AKW, AHJ, HJH, ALW, ADC, LCH, AFW}`. A doctor must also
be `statuscodename == "Active"`. Doctor identity is resolved against the
FULL deduplicated pool first (so a doctor who exists but fails one of
these conditions can still be recognised and reported as a `FAIL`, not a
bare "no such doctor"), then these conditions are checked as part of the
result.

`cr301_opdflag` is deliberately **not** part of this gate — a doctor who
isn't flagged OPD (e.g. a home-care or other non-OPD service context) can
still be a genuine, resolvable doctor. It is fetched and carried on every
doctor record purely as informational metadata (visible in logs and in
the resolved doctor evidence), never as a searchability filter — see
`_is_active()` in `doctor_validation.py`.

## Deduplication

Rows sharing `cr301_doctorkey` are merged (first non-null value per field
wins; conflicting non-null values are logged, never silently dropped). A
separate, real data quirk — the same physical doctor sometimes exists
under two DIFFERENT doctor keys (e.g. re-onboarded at a second business
unit) — is handled by a same-profile safety check: if every tied candidate
shares an identical name AND identical degree/specialty/subspecialty/BU,
resolution proceeds safely; otherwise (e.g. genuinely different BU/fee per
key) it correctly reports `AMBIGUOUS_DOCTOR`.

## Known limitation (documented, not silently papered over)

The CRM doctor dataset also contains non-human operational rows ("MRI .",
"X Ray Male", "Procedure Room", "TEST Doctor", ...) that pass the same
Active + supported-BU filter real doctors do — there is no reliable
CRM field to exclude them; their degree/specialty/BU values look identical
to real physician rows. This is handled defensively rather than by
inventing a filter: name resolution stays conservative (exact or
high-confidence multi-token match only, never a bare common word), so such
rows are only ever selected when the transcript explicitly names them —
which essentially never happens in a real conversation.

## Claim extraction is intentionally scoped, not exhaustive

Degree/specialty/business-unit/fee/examination-age claims use fairly
robust dedicated parsers. Scope-of-service/qualifications/doctor-notes
claims use a lighter-weight "trigger phrase + meaningful word overlap"
approach rather than an exhaustive Arabic NLP pipeline — this mirrors the
project's existing philosophy (see `bank_validation.py`'s
`_BANK_NAME_CANON`, an explicitly extensible, non-exhaustive alias table)
of preferring a documented, extensible starting point over an attempt at
total coverage that would inevitably still miss real phrasing anyway.
