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
| `specialty_taxonomy.py` | Shared, formal EN↔AR specialty/subspecialty NAME table (pediatric specialties included) — see "Specialty/subspecialty taxonomy" below. |

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

Primary BU authority for RECORD ELIGIBILITY is `cr18c_buname` (NOT
`cr301_businessunitname` — real CRM data has been observed to disagree
between the two on the same row, e.g. `cr301_businessunitname="MKR"` while
`cr18c_buname="ADC"` for the same doctor). Supported scope:
`{AKW, AHJ, HJH, ALW, ADC, LCH, AFW}`. A doctor must also be
`statuscodename == "Active"`. Doctor identity is resolved against the
FULL deduplicated pool first (so a doctor who exists but fails one of
these conditions can still be recognised and reported as a `FAIL`, not a
bare "no such doctor"), then these conditions are checked as part of the
result.

This eligibility policy is a separate question from whether a chat-stated
BU **claim** about an already-eligible doctor is correct — see
`_match_doctor_business_unit`/`_doctor_business_units` in
`doctor_validation.py`: a BU claim PASSes when it matches EITHER
`cr301_businessunitname` OR `cr18c_buname`, since the two fields are not
required to agree with each other, and a disagreement between them is
never itself a reason to fail a claim that matches one of them. The
per-field `business_unit` validation result additionally reports
`matched_crm_field` (which of the two fields matched) and
`crm_business_units` (both raw values), and the same two-field rule also
applies to BU-scoped doctor resolution (`bu_scoped_pool` and the BU
tie-break in `_resolve_and_validate_one_doctor`) — the call's own detected
BU can pick out a candidate via either of that candidate's two BU fields.
Record eligibility itself is untouched by any of this: it still runs on
`cr18c_buname` alone, exactly as above.

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

## Specialty/subspecialty taxonomy

`specialty_taxonomy.py` holds a SECOND, formal EN↔AR specialty-NAME table
(`SPECIALTY_EN_TO_AR`), deliberately separate from — and merged with, never
duplicating — `offer_search._AR_ALIAS`'s colloquial-phrase table:
`_AR_ALIAS` maps loose conversational phrases ("قلب", "اطفال انابيب") to a
coarse category for patient-facing offer search; `specialty_taxonomy`
instead maps the actual FORMAL specialty/subspecialty names CRM data and
an Agent's explicit claim can carry verbatim, in either language,
including pediatric subspecialties (`PEDIATRIC_SPECIALTIES_EN/AR`) offer
search has no reason to know about. Several EN names can describe the same
Arabic concept (e.g. "General Pediatrics"/"Pediatrics"/"Pediatric
medicine") — `canonicalize_specialty_name()` collapses those onto one
canonical EN bucket via an EXACT (never substring) lookup, which is what
keeps closely related specialties — "Pediatric Cardiology" vs
"Cardiology", "Oncology" vs "Medical Oncology" — as distinct, unambiguous
entries instead of accidentally merging.

`SPECIALTY_EN_TO_AR` is periodically extended (never replaced) with real
specialty/subspecialty display-name values as they're observed in
production CRM exports — including values that only differ from an
existing entry by case (e.g. both "Pediatric surgery" and "Pediatric
Surgery" are kept as their own keys, since real CRM rows carry both
spellings) and known CRM misspellings (e.g. "Anastasia" for Anesthesia,
"Neurospsychiatry"/"Neurospsychiry" for Neuropsychiatry, "Pain Managment"
for Pain Management, "Summar" for Summer). A misspelling is never silently
corrected in the table — it is kept as its own exact-match key so the raw
CRM spelling stays recognisable in evidence/logs — but it resolves through
`canonicalize_specialty_name()` to the same canonical concept as the
correctly-spelled name via their shared Arabic value.

**Required lookup order** for a claimed specialty/subspecialty/clinical
service against a resolved doctor (`_resolve_and_validate_one_doctor` in
`doctor_validation.py`):
1. CRM specialty fields — `cr301_specialtyname`, `cr18c_manualspecialtyname`.
2. CRM subspecialty fields — `cr301_subspecialtyname`, `cr18c_manualsubspecialtyname`.
3. CRM scope of service — `cr301_scopeofservicear`, then `cr301_scopeofservice`
   — tried ONLY once neither 1 nor 2 matched, using the RAW claim text (not
   just its resolved coarse category) so a genuine service phrase with no
   top-level specialty of its own (e.g. "الحشوات" — dental fillings) can
   still be confirmed against a doctor's own detailed scope text (see
   `_match_specialty_against_scope`). A mismatch against the general
   specialty never immediately FAILs when subspecialty or scope confirms
   the claim instead.

Outcome: a specialty/subspecialty/scope match PASSes that respective
field; no match anywhere the doctor has actual data FAILs; no populated
evidence in ANY of the 6 fields above is `NEEDS_REVIEW`.

**Pediatric guard** (`_specialty_values_match`/`_pediatric_flag` in
`doctor_validation.py`): before comparing two specialty values at all, if
exactly one side is pediatric (either its canonical taxonomy entry is
pediatric-flagged, or its raw text mentions a pediatric/infant/neonatal
marker word — see `_mentions_pediatric_context`) and the other is not,
they are an automatic mismatch, regardless of what the qualifier-stripped
core-token comparison below would otherwise conclude. This is what stops
"Pediatric Cardiology" from ever matching plain "Cardiology" while still
letting a claim like "غدد صماء أطفال" (which explicitly says "أطفال")
correctly match a "Pediatric Endocrinology" subspecialty.

**Scope-of-service fallback candidates** (`_scope_fallback_candidate_
phrases`): the raw Agent claim text, the resolved category's own EN name,
and that name's taxonomy-mapped Arabic equivalent — searched as
NORMALIZED SUBSTRINGS of the scope text (never bare word-overlap, which
risks a false PASS from shared generic vocabulary alone). A candidate
phrase that reduces to nothing but a generic word (`_is_generic_scope_
phrase` — "طب"/"جراحة"/"أطفال"/"علاج" alone, ...) is never accepted. The
result records `matched_crm_field`-style evidence — `match_source` (which
CRM field matched) and `matched_phrase` — on the `scope_of_service`
validated-field entry, additively (see `_field()`'s `**evidence`
parameter — the original `claimed`/`outcome`/`reference` keys are never
removed).

`_GENERIC_SPECIALTY_WORDS` (the vocabulary that decides whether a title's
tail is a specialty/service phrase rather than a person's name — see
`_rejected_candidate_reason`) is extended at import time with every
taxonomy EN/AR name's own first word, so a literal specialty/subspecialty
name an Agent states verbatim ("Pediatric Cardiology", "NICU",
"Pedodontic") is recognised as a specialty phrase the same way the
hardcoded colloquial words are — computed once from `specialty_taxonomy`,
never a second, hand-maintained list of specialty names to keep in sync.

## Unconditional checks vs. optional (claim-driven) fields

Once a doctor is resolved, two different validation contracts apply:

**Always validated, regardless of what the Agent said:**
- Identity/name resolution against CRM.
- Record eligibility (`statuscodename == "Active"` + a supported
  `cr18c_buname` — `cr301_opdflag` stays informational only, never an
  eligibility filter).
- Specialty/subspecialty and business_unit and scope_of_service ARE
  claim-driven (validated only when the Agent actually made that specific
  claim), but they are not part of the "5 optional fields" contract below
  — e.g. a stated business-unit/branch claim can PASS or genuinely FAIL
  (see `resolve_business_unit`/`canonical_doctor_bu` in
  `_resolve_and_validate_one_doctor`), it just never appears at all when
  no BU/branch was mentioned. A BU claim PASSes against either of the
  doctor's two CRM BU fields — see "Authoritative filtering" above.

**Optional (claim-driven) fields** — `_OPTIONAL_CLAIM_FIELDS` in
`doctor_validation.py` — validated ONLY when the Agent explicitly makes a
claim about the resolved doctor:
- `degree` (rank: consultant/specialist/professor/GP/resident/registrar/
  senior registrar — a bare دكتور/دكتورة/د/Dr title is never itself a claim)
- `doctor_notes`
- `qualifications`
- `examination_age` — high priority: checked (and logged with full
  diagnostics) immediately after specialty, before the lower-priority
  free-text optional fields (`doctor_notes`/`scope_of_service`/
  `qualifications`) and before `walkin_fee`. Recognises minimum-age,
  children/adults-only, and a SPECIFIC age-range claim
  (`"من 5 لحد 15 سنة"` → `range:5-15`), each checked against the CRM
  range parsed by `parse_examination_age`.
- `walkin_fee`

When absent, an optional field is never added to `validated_fields`
(so it never counts toward `fields_checked`, and can never lower the
outcome). A concise diagnostic line — `optional_fields_not_mentioned=[...]`
— lists exactly which of the 5 were skipped, printed by
`_log_skipped_optional_fields` alongside (never instead of) the per-field
`_log_field_validation` diagnostic for every field that WAS checked.

All per-doctor claim extraction (degree, business_unit, doctor_notes,
scope_of_service, qualifications, examination_age, walkin_fee) is scoped
to THIS resolved doctor's own agent turn(s) via
`_agent_turns_text_for_doctor` — never a blind scan of the whole call's
agent-side text — so a claim about a DIFFERENT doctor in a multi-doctor
recommendation set is never attributed to this one.
