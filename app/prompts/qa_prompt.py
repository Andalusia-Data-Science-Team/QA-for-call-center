"""
qa_prompt.py — System prompt and focused user-prompt builders for the QA LangGraph pipeline.

Design philosophy
─────────────────
• SYSTEM_PROMPT is intentionally SHORT — role, output contract, and core
  principles only.  No compliance details are embedded here.

• The previous single build_user_prompt() has been SPLIT into four focused
  builders, one per evaluation node:

    build_behavioral_prompt()   → professionalism, tone, empathy, red flags
    build_compliance_prompt()   → 15 compliance pillars, flag C2Com/C2C/C2B/NC
    build_script_prompt()       → greeting/closing script adherence
    build_scoring_prompt()      → aggregate scores + overall_assessment

  Each builder receives ONLY the criteria it needs, keeping the LLM focused and
  reducing hallucination from irrelevant context.

• build_user_prompt() is kept for backward-compatibility (legacy CallAnalyzer).

• aggregate_results() (in nodes.py) merges the four sub-results into the single
  QAAnalysisResult schema — output shape is UNCHANGED.
"""

from app.models.input import CallTranscript

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (shared across all four focused LLM calls)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Quality Assurance Analyst for Andalusia Hospitals' call center.
Your task is to evaluate agent-patient call transcripts and produce a structured JSON quality report.

## CORE PRINCIPLES
1. EVIDENCE-BASED: Base every finding strictly on observable transcript content. Quote directly; never speculate.
2. PATIENT SAFETY FIRST: Inaccurate medical/appointment/medication information is a patient safety risk — flag it.
3. PROPORTIONATE: Distinguish critical violations from minor imperfections using the severity tiers below.
4. DEFINED SCOPE: Mention only the most critical severe violations. Report 0 to 4 violations at maximum. Do NOT over-evaluate.
5. DEVELOPMENTAL: Balance corrective feedback with recognition of positive agent behaviors.
6. STRUCTURED OUTPUT: Return ONLY a valid JSON object matching the schema provided. No prose, no markdown fences.

## SEVERITY TIER DEFINITIONS  (apply strictly)
- critical   → Direct patient safety risk: wrong medication name/dose, wrong doctor, dangerous misinformation,
               explicit patient aggression ignored, or a vulnerable patient actively harmed.
               A callback promise, a hold, or an unresolved prior inquiry is NEVER critical on its own.
- positive   → Agent exceeded expectations or demonstrated exemplary behavior.

## WHAT IS NOT A VIOLATION
- Promising a callback within a stated time window (30 min, 1 hour, etc.) is standard practice — NOT a violation
  unless the agent gave a demonstrably false or impossible commitment.
- A patient reporting a previous unresolved inquiry is historical context — do NOT penalize the current agent
  for prior team failures unless this agent also fails to address the issue in the current call.
- Asking a clarifying question or deferring to a specialist is good practice — NOT "failure to identify problem".
- Short transcripts or single-turn exchanges does not count as a violation.


## ASSESSMENT DECISION RULES
- overall_assessment = "escalate"     → dangerous misinformation, explicit aggression, or vulnerable patient harmed.
- overall_assessment = "needs_review" → One or more C2C/C2B violations observed but no escalation trigger.
- overall_assessment = "pass"         → No significant violations; include at least one positive compliance_flag.
- escalation_required must be true if and only if overall_assessment = "escalate".

## EDGE CASES
- Short transcript (< 100 words): note in assessment_reasoning; score conservatively at 0.5 for unobservable dimensions.
- Disconnected call: note abrupt end; do not penalize the agent.
- Foreign language: note language barrier; do not assess content accuracy.

All specific violation definitions, behavioral standards, script templates, and scoring weights
are provided in the user message below. Evaluate strictly against those criteria.
Be realistic, not overly harsh.
"""

# ─────────────────────────────────────────────────────────────────────────────
# NODE A — Behavioral Evaluation Prompt
#   Focus: professionalism, tone, empathy, active listening, prohibited
#          behaviors, red-flag phrases.
#   Output fields: behavioral_flags (subset of compliance_flags), strengths,
#                  improvements, professionalism_score.
# ─────────────────────────────────────────────────────────────────────────────

def build_behavioral_prompt(
    call: CallTranscript,
    behavioral_criteria: str = "",
) -> str:
    return f"""\
Evaluate the agent's BEHAVIORAL performance in the call below.
Focus exclusively on: tone, professionalism, empathy, active listening, prohibited phrases, and red-flag language.
Do NOT evaluate compliance pillars, script adherence, or scoring weights here.
If there is no response from the patient agent can use the script to greet and close the call. 

## SEVERITY REMINDER
- critical   → patient safety risk ONLY (wrong drug, wrong dose, aggression ignored).
- NOT a violation: callback promises, hold times, prior team failures reported by the patient.

DEFINED SCOPE: Report 0 to 2 violations at maximum. Do NOT flag normal service interactions as violations.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
BEHAVIORAL STANDARDS  (evaluate strictly against these)
════════════════════════════════════════════════════════════
{behavioral_criteria or "(not loaded)"}

════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "professionalism_score": <0.0–1.0>,
  "behavioral_flags": [
    {{
      "type": "<C2C | C2B | NC>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences citing the specific behavioral standard violated or met>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ],
  "strengths": ["<observed positive behavior 1>", "<observed positive behavior 2>"],
  "improvements": ["<behavioral improvement 1>", "<behavioral improvement 2>"]
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE B — Compliance Pillars Evaluation Prompt
#   Focus: the 15 official compliance pillars grouped C2Com → C2C → C2B → NC.
#   Output fields: compliance_flags (all pillar violations), escalation signals.
# ─────────────────────────────────────────────────────────────────────────────

def build_compliance_prompt(
    call: CallTranscript,
    compliance_pillars: str = "",
) -> str:
    return f"""\
Evaluate the call below against the official COMPLIANCE PILLARS only.
Flag every violation by its exact pillar name and type (C2Com / C2C / C2B / NC).
Do NOT evaluate behavioral tone, script adherence, or scoring weights here.

## SEVERITY REMINDER — apply before flagging anything as critical
- critical   → patient safety risk ONLY: wrong medication name/dosage stated, wrong doctor assigned,
               dangerous medical misinformation given as fact.
- moderate   → process failure without safety risk.
- minor      → small deviation, negligible impact.
- NOT a violation: a callback promise within a stated window, a patient citing a prior unresolved inquiry,
  the agent deferring a medication availability check to a specialist or pharmacy team.

## WHAT DOES NOT CONSTITUTE "INACCURATE INFORMATION"
  Saying "we will contact you within 30 minutes" is a service commitment — not inaccurate information.
  Only flag information accuracy if the agent stated a medically or factually wrong claim as truth.

## WHAT DOES NOT CONSTITUTE "FAILED TROUBLESHOOTING"
  If the agent acknowledged the patient's issue and gave a concrete next step (callback, escalation,
  transfer), troubleshooting was NOT failed — even if the root cause was not resolved in this call.

## AUTOMATED SYSTEM MESSAGES & CLOSING PROTOCOL
  System-generated messages (e.g., "The call has ended due to inactivity") are NOT violations if:
  - The agent used proper closing script before patient became unresponsive
  - The patient did not respond for an extended period (30+ seconds)
  - The agent attempted to re-engage or used proper farewell before system timeout
  Do NOT flag automated closings or system messages as violations when the agent followed proper protocol.
  This includes: "Close the call on time", "Didn't Commit to call script", "Professional Closing" violations.

DEFINED SCOPE: Report 0 to 2 violations at maximum. Do NOT over-evaluate.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
COMPLIANCE PILLARS  (flag violations by pillar name + type + id)
════════════════════════════════════════════════════════════
{compliance_pillars or "(not loaded)"}

════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "compliance_flags": [
    {{
      "type": "<C2Com | C2C | C2B | NC>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences referencing the exact pillar name>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ]
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# NODE C — Compliance Pillars Evaluation Prompt
#   Focus: the 15 official compliance pillars grouped C2Com → C2C → C2B → NC.
#   Output fields: compliance_flags (all pillar violations), escalation signals.
# ─────────────────────────────────────────────────────────────────────────────

def build_reservation_prompt(
    call: CallTranscript,
    appointment_verification: str,
    reservation_pillars: str = "",
) -> str:
    return f"""\
Evaluate the call below against the official RESERVATION PILLARS only.
Flag every violation by its exact pillar name and type (C2Com / C2C / C2B / NC).
Check the appointment details extracted from the transcript against the hospital's reservation database.
Do NOT evaluate behavioral tone, script adherence, or scoring weights here.

## SEVERITY REMINDER — apply before flagging anything as critical
- critical   → patient safety risk ONLY: wrong medication name/dosage stated, wrong doctor assigned,
               dangerous medical misinformation given as fact.
- moderate   → process failure without safety risk.
- minor      → small deviation, negligible impact.
- NOT a violation: a callback promise within a stated window, a patient citing a prior unresolved inquiry,
  the agent deferring a medication availability check to a specialist or pharmacy team.

## WHAT DOES NOT CONSTITUTE "INACCURATE INFORMATION"
  Saying "we will contact you within 30 minutes" is a service commitment — not inaccurate information.
  Only flag information accuracy if the agent stated a medically or factually wrong claim as truth.

## WHAT DOES NOT CONSTITUTE "FAILED TROUBLESHOOTING"
  If the agent acknowledged the patient's issue and gave a concrete next step (callback, escalation,
  transfer), troubleshooting was NOT failed — even if the root cause was not resolved in this call.

## AUTOMATED SYSTEM MESSAGES & CLOSING PROTOCOL
  System-generated messages (e.g., "The call has ended due to inactivity") are NOT violations if:
  - The agent used proper closing script before patient became unresponsive
  - The patient did not respond for an extended period (30+ seconds)
  - The agent attempted to re-engage or used proper farewell before system timeout
  Do NOT flag automated closings or system messages as violations when the agent followed proper protocol.
  This includes: "Close the call on time", "Didn't Commit to call script", "Professional Closing" violations.

DEFINED SCOPE: Report 0 to 2 violations at maximum. Do NOT over-evaluate.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
RESERVATION PILLARS  (flag violations by pillar name + type + id)
════════════════════════════════════════════════════════════
{reservation_pillars or "(not loaded)"}

════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
APPOINTMENT VERIFICATION
════════════════════════════════════════════════════════════
{appointment_verification or "(not loaded)"}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "compliance_flags": [
    {{
      "type": "<C2Com | C2C | C2B | NC>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences referencing the exact pillar name>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ]
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# NODE C — Script Template Matching Prompt
#   Focus: greeting and closing script adherence against approved templates.
#   Output fields: script_flags, accuracy_score.
# ─────────────────────────────────────────────────────────────────────────────

def build_script_prompt(
    call: CallTranscript,
    script_templates: str = "",
) -> str:
    return f"""\
Evaluate the agent's adherence to the APPROVED SCRIPT TEMPLATES in the call below.
Do not strictly enforce the exact scripts provided; instead, assess whether the conversation aligns with their intended purpose and conveys the expected concepts.
Do NOT evaluate compliance pillars, behavioral tone, or scoring weights here.
If there is no response from the patient agent can use the script to greet and close the call.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
APPROVED SCRIPT TEMPLATES  (reference for script adherence)
════════════════════════════════════════════════════════════
{script_templates or "(not loaded)"}

════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "accuracy_score": <0.0–1.0>,
  "script_flags": [
    {{
      "type": "<C2C | NC>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences — state which script element was used/missed>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ]
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE D — Overall Scoring Prompt
#   Focus: compute final dimension scores using the weights, synthesize all
#          sub-evaluation signals into the top-level assessment.
#   Receives: outputs from the three previous focused evaluations (as context)
#             plus the scoring weights.
#   Output fields: overall_assessment, assessment_reasoning, resolution_score,
#                  escalation_required, escalation_reason.
# ─────────────────────────────────────────────────────────────────────────────

def build_scoring_prompt(
    call: CallTranscript,
    scoring_weights: str = "",
    behavioral_summary: str = "",
    compliance_summary: str = "",
    reservation_summary: str = "",
    script_summary: str = "",
    # Masked deterministic bank/location outputs are scoring context, not an
    # LLM decision about whether an account number or address is correct —
    # each comes from its own independent graph node (both backed by
    # app.service_hub).
    bank_summary: str = "",
    location_summary: str = "",
    # Doctor validation is two independent checks, same pattern as bank/
    # location: doctor_summary is the deterministic factual-information
    # result (app.service_hub.doctor_validation), doctor_scope_summary is
    # the separate, semantic recommendation-suitability result (LLM-based,
    # app.agent.nodes.infer_doctor_scope_validation) — never merged into
    # one opaque call.
    doctor_summary: str = "",
    doctor_scope_summary: str = "",
) -> str:
    """
    Prompt that synthesizes the three focused sub-evaluations into a final score.
    The three summary strings are the raw JSON outputs from the preceding nodes,
    giving the LLM full context to produce a calibrated overall assessment.
    """
    return f"""\
You are producing the FINAL scoring and overall assessment for a call that has already been
evaluated in three separate passes. Your job is to:
  1. Review the sub-evaluation results provided below.
  2. Apply the scoring weights to produce the final dimension scores.
  3. Determine the overall_assessment ("pass" / "needs_review" / "escalate").
  4. Write the assessment_reasoning (2-4 sentences citing specific evidence).
  5. Confirm or correct escalation_required based on the compliance results.
  6. Merge all compliance_flags from all sub-evaluations into a single deduplicated list.
  7. Aggregate strengths and improvements from behavioral sub-evaluation.
  8. Assign Agent Classification based on the violation counts below.
  9. Assign Profiling Comment ONLY if there is a clear performance issue — otherwise omit it (null).

Agent Classification criteria:
A  -> No C2C, C2B, C2Com or NC violations
B  -> Less than 2 NC violations
C  -> 1 Critical violation  OR  more than 2 NC violations
D  -> More than 1 Critical violation

Profiling Comment options (use ONLY one of these exact strings, or null if none applies):
"Poor Knowledge" | "Poor System" | "Poor Process" | "Poor Report" |
"Poor Selling Skills" | "Poor Behavior" | "Poor Soft Skills"

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
SCORING WEIGHTS  (apply when computing dimension scores)
════════════════════════════════════════════════════════════
{scoring_weights or "(not loaded)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 1 — BEHAVIORAL
════════════════════════════════════════════════════════════
{behavioral_summary or "(not available)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 2 — COMPLIANCE PILLARS
════════════════════════════════════════════════════════════
{compliance_summary or "(not available)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 3 — RESERVATION PILLARS
════════════════════════════════════════════════════════════
{reservation_summary or "(not available)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 4 — SCRIPT MATCHING
════════════════════════════════════════════════════════════
{script_summary or "(not available)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 5 — DETERMINISTIC KSA BANK VALIDATION
════════════════════════════════════════════════════════════
{bank_summary or "(not applicable)"}

════════════════════════════════════════════════════════════
SUB-EVALUATION 6 — DETERMINISTIC KSA LOCATION VALIDATION
════════════════════════════════════════════════════════════
{location_summary or "(not applicable)"}

How to interpret the location result above — reason in this order, and
never let this override a confident deterministic PASS/FAIL:
  1. Is the location text actually about an Andalusia branch/facility at
     all? Home-care / home-service conversations (قسم الرعاية المنزلية,
     خدمة منزلية, زيارة منزلية, مرافق منزلي, تمريض منزلي, خدمة في المنزل)
     routinely use the same location words to ask for the PATIENT's OWN
     address so a coordinator can arrange delivery/visit there — e.g. an
     agent saying "وبترسل لهم الموقع" / "ابعت موقعك" / "شارك اللوكيشن" /
     "المنسق هيطلب موقع حضرتك" / "نحتاج موقعك لتقديم الخدمة المنزلية".
     That is the customer's own service address, never an Andalusia
     branch location. When the transcript is clearly this kind of
     home-service context, branch-location validation is simply out of
     scope: do not raise a location-related compliance violation or lower
     accuracy for it, whatever the deterministic outcome field says.
  2. Otherwise, check whether the patient asked about one branch but the
     agent redirected them to a different one (e.g. the requested branch
     doesn't offer the service, so the agent names another facility and
     gives ITS address instead). When that happens, judge the
     agent-provided branch and address — not the patient's
     originally-named branch. A transcript that clearly shows this kind
     of redirection, followed by the agent supplying the new branch's
     location, reflects the agent doing its job correctly, not an error —
     even if the deterministic resolver couldn't map that alternative
     branch name.
  3. With that context established, treat the deterministic outcome as
     the primary signal:
       - PASS: the provided location is correct. No violation.
       - FAIL: a confirmed mismatch between the provided address and the
         resolved branch's known location. Treat this as a real location
         error — do not soften or dismiss a confirmed FAIL.
       - BRANCH_UNRESOLVED: the matcher could not map a branch name to a
         CRM record. This is NOT automatically an agent error — it
         commonly happens in exactly the redirection case (an alternative
         branch name the matcher doesn't recognize) or the home-service
         case above. Do not create a compliance violation or lower the
         accuracy score from a bare BRANCH_UNRESOLVED; only flag a
         problem if the transcript itself shows the agent naming a
         wrong/nonexistent facility, or giving an address that conflicts
         with a branch you can otherwise identify from the conversation.
       - NOT_APPLICABLE: no branch/location request or agent-provided
         branch location was in scope. Do not infer any location
         violation.
  4. Only report a location-related violation when the transcript gives
     real evidence of incorrect information — a confirmed FAIL, or the
     agent clearly naming the wrong facility/address for what was asked.
     Never penalize based on a home-service address, and never penalize a
     plausible branch redirection just because it left the deterministic
     resolver unable to confirm the alternative branch.

════════════════════════════════════════════════════════════
SUB-EVALUATION 7 — DETERMINISTIC DOCTOR INFORMATION VALIDATION
════════════════════════════════════════════════════════════
{doctor_summary or "(not applicable)"}

How to interpret the doctor result above — never override a confident
deterministic PASS/FAIL:
  - PASS: every factual claim the agent actually made about the doctor
    (name/degree/specialty/subspecialty/business unit/notes/scope/
    qualifications/examination age/walk-in fee) matches the authoritative
    CRM record. Treat this as correct — no violation.
  - FAIL: a confirmed mismatch between something the agent stated and the
    CRM record (e.g. wrong degree, wrong fee, recommending an inactive or
    non-OPD doctor). Treat this as a real violation — do not soften or
    dismiss a confirmed FAIL.
  - DOCTOR_UNRESOLVED / AMBIGUOUS_DOCTOR: the deterministic matcher could
    not confidently identify which doctor was meant, or found multiple
    equally plausible records. This is NOT automatically an agent error —
    only flag a problem if the transcript itself gives clear evidence the
    agent was wrong (e.g. a name/detail that plainly doesn't correspond to
    any real doctor). A bare unresolved/ambiguous result on its own must
    not lower the score.
  - INSUFFICIENT_REFERENCE_DATA: CRM doctor data was unavailable. Never
    penalize the agent for this — it is a system/reference-data condition.
  - NOT_APPLICABLE: no specific doctor was mentioned or recommended. Do
    not infer any doctor-information violation.
  - A field the agent never mentioned is never checked (it will not appear
    under validated_fields at all) — do not penalize for information the
    agent simply didn't state.

════════════════════════════════════════════════════════════
SUB-EVALUATION 8 — DOCTOR RECOMMENDATION SUITABILITY (semantic, LLM-based)
════════════════════════════════════════════════════════════
{doctor_scope_summary or "(not applicable)"}

This is a SEPARATE check from Sub-Evaluation 7 — it judges whether the
resolved doctor's documented CRM scope of service is a reasonable fit for
what the patient described, not whether the agent's factual claims about
the doctor were correct.
  - SUITABLE: the recommendation fits the doctor's documented scope. No
    violation.
  - UNSUITABLE: the patient described a need that falls outside the
    doctor's documented scope/specialty — this MAY become a QA issue,
    since the agent recommended a doctor whose documented scope does not
    match what the patient needed. Ground any flag strictly in the
    provided reasoning/evidence, never in outside medical knowledge.
  - UNCLEAR: the available scope evidence was too limited to decide
    confidently. Do not automatically penalize an UNCLEAR result.
  - NOT_APPLICABLE: the patient did not describe a medical complaint/need,
    or no doctor was resolved, or no scope evidence was available. No
    doctor-recommendation penalty in any of these cases.

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "call_id": "{call.call_id}",
  "agent_name": "{call.agent_name}",
  "overall_assessment": "<pass | needs_review | escalate>",
  "assessment_reasoning": "<2-4 sentences citing specific transcript evidence>",
  "compliance_flags": [
    {{
      "type": "<C2Com | C2C | C2B | NC>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences — reference the pillar name if a violation>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ],
  "agent_performance": {{
    "professionalism_score": <0.0–1.0>,
    "Agent Classification": "<A | B | C | D>",
    "Profiling Comment": "<one of the 7 options above, or null>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
  }},
  "escalation_required": <true | false>,
  "escalation_reason": "<reason string or null>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE E — Offer Recommendation Evaluation Prompt
#   Focus: did the agent correctly identify, present, and handle a relevant
#          CRM promotional offer for the patient's specialty?
#
#   Four possible outcomes (one must be chosen):
#     SUITABLE_OFFER_RECOMMENDED   — agent did the right thing (positive flag)
#     OFFER_SKIPPED                — offer existed but agent did not mention it (C2B)
#     UNRELATED_OFFER_RECOMMENDED  — agent mentioned an irrelevant offer (C2B)
#     NO_OFFER_AVAILABLE           — no active offer for this specialty (no flag)
#     OFFER_NOT_APPLICABLE         — call type does not warrant offer check (no flag)
#     OFFER_MISREPRESENTED         — wrong price / date / specialty details (C2B)
#     INCOMPLETE_OFFER_PRESENTATION — offer mentioned but key details omitted (NC)
#     MISSING_OFFER_CONFIRMATION_ASK — offer shown but patient not asked to book (NC)
#
#   Output fields: offer_outcome, offer_flags, offer_reasoning.
# ─────────────────────────────────────────────────────────────────────────────

def build_offer_prompt(
    call: "CallTranscript",
    offer_pillars: str = "",
    crm_offers_context: str = "",
) -> str:
    """
    Evaluate whether the agent correctly handled CRM promotional offers during
    the call.

    Parameters
    ----------
    call : CallTranscript
        The call being evaluated.
    offer_pillars : str
        Compact text block from CriteriaLoader.offer_pillars() — contains the
        pillar definitions and evaluation guidance.
    crm_offers_context : str
        Optional: a JSON-serialised list of active CRM offers that were
        available for the patient's specialty at the time of the call.
        When provided, the LLM can compare what the agent said against the
        actual available offers.  Pass "" when offers cannot be fetched.
    """
    crm_section = (
        f"""\
════════════════════════════════════════════════════════════
AVAILABLE CRM OFFERS  (active offers for this specialty at call time)
════════════════════════════════════════════════════════════
{crm_offers_context}
"""
        if crm_offers_context
        else """\
════════════════════════════════════════════════════════════
AVAILABLE CRM OFFERS
════════════════════════════════════════════════════════════
(CRM offer data not available for this evaluation — infer from transcript only)
"""
    )

    return f"""\
Evaluate whether the agent correctly identified and presented a relevant promotional
offer to the patient during the call below.

## YOUR TASK
1. Read the transcript and determine whether the call type warrants an offer check
   (booking call, service inquiry, or explicit offer question from patient).
2. Identify whether the agent mentioned any promotional offer.
3. Compare what the agent said (if anything) against the available CRM offers
   provided below.
4. Choose exactly ONE outcome from the list and produce the appropriate flags.

## OUTCOME DEFINITIONS
- SUITABLE_OFFER_RECOMMENDED   : Agent identified and correctly presented a matching offer → positive flag.
- OFFER_SKIPPED                : A relevant offer existed (visible in CRM context or inferable) but the
                                 agent never mentioned it → C2B flag (moderate).
- UNRELATED_OFFER_RECOMMENDED  : Agent presented an offer that does not match the patient's specialty,
                                 service request, or gender → C2B flag (moderate).
- OFFER_MISREPRESENTED         : Agent stated incorrect price, discount percentage, expiry date, or
                                 specialty for an offer → C2B flag (moderate).
- INCOMPLETE_OFFER_PRESENTATION: Agent mentioned the offer correctly but omitted price or expiry date → NC flag (minor).
- MISSING_OFFER_CONFIRMATION_ASK: Agent presented the offer but did not ask the patient whether they
                                  want to proceed with it → NC flag (minor).
- NO_OFFER_AVAILABLE           : No active CRM offer exists for this specialty → no flag, no penalisation.
- OFFER_NOT_APPLICABLE         : Call is a complaint, emergency, admin, or follow-up with no booking
                                 intent → no flag, no penalisation.

## IMPORTANT RULES
- Do NOT penalise the agent when CRM offers context is absent — choose NO_OFFER_AVAILABLE or
  OFFER_NOT_APPLICABLE in ambiguous cases rather than guessing.
- If any miss presented information is present mention the correct information in the reasoning section.
- Do NOT penalise when the patient explicitly declined an offer the agent correctly presented.
- Short calls (< 90 seconds) and calls with no specialty signal → OFFER_NOT_APPLICABLE.
- Report at most 1 offer flag per call.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
OFFER RECOMMENDATION PILLARS  (evaluate strictly against these)
════════════════════════════════════════════════════════════
{offer_pillars or "(not loaded)"}

{crm_section}
════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "offer_outcome": "<SUITABLE_OFFER_RECOMMENDED | OFFER_SKIPPED | UNRELATED_OFFER_RECOMMENDED | OFFER_MISREPRESENTED | INCOMPLETE_OFFER_PRESENTATION | MISSING_OFFER_CONFIRMATION_ASK | NO_OFFER_AVAILABLE | OFFER_NOT_APPLICABLE>",
  "offer_reasoning": "<2-3 sentences citing specific transcript evidence for your choice>",
  "offer_flags": [
    {{
      "type": "<C2B | NC | positive>",
      "severity": "<critical | positive>",
      "description": "<1-2 sentences referencing the exact pillar name and outcome>",
      "transcript_excerpt": "<verbatim excerpt or 'N/A' when no mention in transcript>"
    }}
  ]
}}

IMPORTANT: offer_flags must be an EMPTY LIST [] when the outcome is
NO_OFFER_AVAILABLE or OFFER_NOT_APPLICABLE.
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE — Doctor Recommendation Suitability Prompt (semantic, LLM-based)
#   Focus: is the ALREADY-RESOLVED doctor's authoritative CRM scope of
#          service compatible with what the patient described?
#   This is the separate, semantic half of doctor validation — the
#   deterministic half (name/degree/specialty/BU/fee/... factual checks)
#   lives entirely in app.service_hub.doctor_validation and never touches
#   an LLM. This prompt receives ONLY the already-resolved doctor's CRM
#   fields (never the full CRM dataset) and must NEVER be asked to pick a
#   doctor itself — that decision is made deterministically before this
#   prompt is ever built.
# ─────────────────────────────────────────────────────────────────────────────

def build_doctor_scope_prompt(
    call: CallTranscript,
    patient_complaint: str = "",
    doctor_reference: str = "",
) -> str:
    """
    Determine whether a doctor's documented CRM scope of service reasonably
    covers a patient's stated medical complaint/need.

    Parameters
    ----------
    patient_complaint : str
        The patient's own words describing their symptom/condition/desired
        treatment (already extracted — never the full transcript re-parsed
        by the LLM).
    doctor_reference : str
        Compact JSON of ONLY the CRM fields needed for this decision
        (doctor_name_ar/en, degree, specialty, subspecialty, manual
        specialty/subspecialty, scope_of_service(_ar), doctor_notes,
        examination_age, qualifications, plus the deterministically
        precomputed has_detailed_scope/patient_age/age_eligibility_hint
        flags — see infer_doctor_scope_validation) — built from the
        ALREADY-RESOLVED doctor record, never the full CRM dataset.
    """
    return f"""\
You are checking whether a SPECIFIC, ALREADY-IDENTIFIED doctor's documented CRM scope of
service reasonably covers a patient's stated medical need. This is NOT a diagnosis task —
you are not determining what disease the patient has. You are only comparing the patient's
stated complaint against the doctor's DOCUMENTED scope, notes, subspecialty, and specialty.

## YOUR TASK
1. Read the patient's stated complaint/need below.
2. Read the doctor's authoritative CRM reference (already resolved — do not question or
   change which doctor this is; you are only judging fit, not identity).
3. Decide whether the complaint reasonably falls within the doctor's documented scope, using
   this evidence hierarchy in order of authority (highest first):
     1. scope_of_service / scope_of_service_ar — the most authoritative evidence: exact
        documented services/procedures. Prefer this over any category label below.
     2. doctor_notes — especially an EXPLICIT restriction or inclusion statement (e.g. "new
        cases only", "does not accept reviews except Sunday", a specific service the doctor
        does NOT provide, a branch restriction, an age/case-type restriction). An explicit
        note MAY OVERRIDE an otherwise-suitable scope/specialty reading in either direction.
     3. subspecialty / manual subspecialty — narrower than specialty, but still just a
        category label, not proof of exact coverage.
     4. specialty / manual specialty — the broadest, weakest tier. NEVER treat a specialty
        match alone as sufficient: two doctors can share the exact same specialty (e.g. both
        "Orthopedics") while one handles spine surgery and the other handles sports injuries/
        ACL reconstruction — a specialty label says nothing about which.
     5. examination_age / patient_age / age_eligibility_hint — see the age-eligibility rule
        below.
     6. qualifications / qualifications_ar — CONTEXT ONLY. A doctor's degree or years of
        experience is never proof they handle a specific condition; never use it alone to
        justify SUITABLE.
4. Choose exactly ONE outcome.

## SPECIALTY-ALONE SAFEGUARD (critical)
If has_detailed_scope is false (no scope_of_service/scope_of_service_ar text at all) and
doctor_notes gives no explicit relevant signal, do NOT return a confident SUITABLE based on
specialty or subspecialty alone — prefer UNCLEAR. A bare specialty/subspecialty label is
supporting context, never the primary decision source.

## OUTCOME DEFINITIONS
- SUITABLE     : The complaint clearly falls within the doctor's documented scope (or, absent
                 detailed scope text, a specific and directly-relevant subspecialty/note), and
                 no doctor_notes restriction rules it out.
- UNSUITABLE   : The complaint clearly falls OUTSIDE the doctor's documented scope, a
                 doctor_notes restriction explicitly excludes this kind of case, or
                 age_eligibility_hint is "outside_range".
- UNCLEAR      : The documented evidence is too limited, generic (e.g. specialty-only with no
                 detailed scope), or ambiguous to confidently decide either way — do NOT guess;
                 this is the safe default when evidence is thin.
- NOT_APPLICABLE : Use only if the reference data given to you is empty/unusable (this should
                 be rare — the caller already checked for a complaint and reference data
                 before invoking you).

## AGE-ELIGIBILITY RULE
age_eligibility_hint is precomputed deterministically from an explicitly-stated patient age
compared against examination_age — trust it when present: "outside_range" is strong evidence
toward UNSUITABLE; "within_range" supports (but does not alone prove) SUITABLE. When
age_eligibility_hint is null/absent (no explicit patient age was stated), do NOT infer or
penalize based on age — proceed on the other evidence only.

## MEDICAL-SAFETY RULES (read carefully)
- Do NOT diagnose the patient. Never state what disease/condition the patient "has".
- Only state whether the stated symptoms/need appear within or outside the doctor's
  DOCUMENTED scope — ground every claim in the CRM reference text given below, not in your
  own general medical knowledge.
- Do not invent scope-of-service content that isn't present in the reference below.

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}

════════════════════════════════════════════════════════════
PATIENT'S STATED COMPLAINT / NEED
════════════════════════════════════════════════════════════
{patient_complaint or "(not available)"}

════════════════════════════════════════════════════════════
DOCTOR CRM REFERENCE  (already resolved — the authoritative record for this specific doctor)
════════════════════════════════════════════════════════════
{doctor_reference or "(not available)"}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "outcome": "<SUITABLE | UNSUITABLE | UNCLEAR | NOT_APPLICABLE>",
  "patient_need_summary": "<1 sentence, patient's own words/summary — no diagnosis>",
  "doctor_scope_summary": "<1 sentence summarising the doctor's relevant documented scope>",
  "matched_scope_evidence": ["<verbatim snippet(s) from the CRM reference that drove your decision>"],
  "reasoning": "<2-3 sentences, grounded ONLY in the CRM reference text above>",
  "is_violation": <true if outcome == "UNSUITABLE", else false>
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY — build_user_prompt()
#   Kept for backward-compatibility with the non-graph CallAnalyzer path.
#   The new graph pipeline uses the four focused builders above instead.
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(
    call: CallTranscript,
    behavioral_criteria: str = "",
    compliance_pillars: str = "",
    script_templates: str = "",
    scoring_weights: str = "",
) -> str:
    """
    Assemble the full user prompt from the transcript + pre-rendered criteria blocks.
    LEGACY: used by CallAnalyzer (analyzer.py) only.
    The LangGraph pipeline now uses the four focused prompt builders above.
    """
    return f"""\
Analyze the following call transcript and return a JSON quality report.
Agent Classification is calculated based on the following criteria:
A	-> No C2C,C2B,C2COM and NC violation Per Transaction	
B	-> Less than 2 NC violations Per Transaction	
C	-> 1 Critical Per Transaction	More than 2 NC violations Per Transaction
D	-> More than 1 Critical Per Transaction	

════════════════════════════════════════════════════════════
CALL METADATA
════════════════════════════════════════════════════════════
Call ID   : {call.call_id}
Agent     : {call.agent_name}
Date      : {call.call_date}
Duration  : {call.call_duration_seconds}s
Department: {call.department}

════════════════════════════════════════════════════════════
BEHAVIORAL STANDARDS  (evaluate against these)
════════════════════════════════════════════════════════════
{behavioral_criteria or "(not loaded)"}

════════════════════════════════════════════════════════════
COMPLIANCE PILLARS  (flag violations by pillar name + type)
════════════════════════════════════════════════════════════
{compliance_pillars or "(not loaded)"}

════════════════════════════════════════════════════════════
APPROVED SCRIPT TEMPLATES  (reference for script adherence)
════════════════════════════════════════════════════════════
{script_templates or "(not loaded)"}

════════════════════════════════════════════════════════════
SCORING WEIGHTS  (apply when computing dimension scores)
════════════════════════════════════════════════════════════
{scoring_weights or "(not loaded)"}

════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════
{call.transcript}

════════════════════════════════════════════════════════════
OUTPUT SCHEMA  — return ONLY this JSON, no markdown fences
════════════════════════════════════════════════════════════
{{
  "call_id": "{call.call_id}",
  "agent_name": "{call.agent_name}",
  "overall_assessment": "<pass | needs_review | escalate>",
  "assessment_reasoning": "<2-4 sentences citing specific transcript evidence>",
  "compliance_flags": [
    {{
      "type": "<C2Com | C2C | C2B | NC>",
      "severity": "<critical | moderate | minor | positive>",
      "description": "<1-2 sentences — reference the pillar name if a violation>",
      "transcript_excerpt": "<verbatim excerpt>"
    }}
  ],
  "agent_performance": {{
    "professionalism_score": <0.0–1.0>,
    "Agent Classification": "<A | B | C | D>",
    "Profiling Comment": "<one of the 7 options above, or null>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
  }},
  "escalation_required": <true | false>,
  "escalation_reason": "<reason string or null>"
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE E — Appointment Details Extraction Prompt
#   Focus: extract appointment date, doctor name, and medical specialty from transcript.
#   Output fields: appointment_details (date, doctor_name, specialty_name).
# ─────────────────────────────────────────────────────────────────────────────

APPOINTMENT_EXTRACTION_PROMPT = """
You are a medical-call data extractor. Given the following call transcript, extract:
1. The requested appointment date (as an ISO-8601 string 2026-MM-DD if determinable, otherwise the exact phrase used, or null if not mentioned).
2. The doctor's full name (exactly as mentioned, or null if not mentioned).
3. The medical specialty name (e.g. "cardiology", "dermatology", or null if not mentioned).
4. Patient Name for the reservation (exactly as mentioned, or null if not mentioned).
5. The name of any promotional offer explicitly mentioned by name (e.g. "باقة الصحة المتكاملة", "عرض الليزر"), or null if no specific offer name was mentioned.

Do not guess or infer information that is not explicitly stated in the transcript.
Do not refine any information
Do Not include any additional commentary or explanation.
If there is not a clear date provided map between the call date :{date} and the nearest date of the week day mentioned for reservation from the call date in 2026. If the date is not mentioned, return null.

Respond ONLY with a valid JSON object with keys:"appointment_date", "doctor_name", "specialty_name","Patient_name","offer_name".

Transcript:
{transcript}
"""


# ─────────────────────────────────────────────────────────────────────────────
# NODE F — Doctor Validation Name-Extraction Prompt
#   Focus: same conversation-reading judgment APPOINTMENT_EXTRACTION_PROMPT
#   already gets right (e.g. it correctly returns doctor_name='وصال',
#   specialty_name='اورام' for a call that also contains "وصال بعيادة
#   الاورام" and "دكتور اورام") — reused here as its OWN dedicated prompt
#   for app.service_hub.doctor_validation's factual-information check,
#   never by calling or altering the appointment-extraction flow itself
#   (that node only runs on a booking intent; doctor validation must work
#   with or without one). Consumed by app.agent.nodes.
#   extract_doctor_semantic_context via the SAME app.agent.nodes.
#   _focused_llm_call helper appointment extraction uses.
#
#   The output is an ADDITIVE quality layer, never the sole gate: when it
#   returns nothing usable (LLM failure, or a genuinely doctor-less call),
#   app.service_hub.doctor_validation's existing deterministic candidate
#   extraction is left to decide on its own, unchanged — see
#   validate_doctor_information's semantic_doctor_name docstring. The one
#   exception is "doctor_role" == "agent_self_introduction": that is a
#   POSITIVE assertion the LLM actively made (never a default/empty
#   value), and app.agent.nodes.validate_doctor_node treats it as an
#   authoritative signal to skip Doctor Validation entirely — see that
#   node's docstring.
# ─────────────────────────────────────────────────────────────────────────────

DOCTOR_NAME_EXTRACTION_PROMPT = """
You are extracting which HUMAN DOCTOR, if any, is the active booking/inquiry target in this call-center transcript — nothing else.

════════════════════════════════════════════════════════════
RULE #1 (HIGHEST PRIORITY) — AGENT SELF-INTRODUCTION
════════════════════════════════════════════════════════════
Before accepting ANY doctor name, first determine whether that name belongs to the speaking AGENT introducing himself/herself, rather than to a doctor being discussed, recommended, or booked for the patient.

Names introduced in phrases such as "مع حضرتك دكتور X", "مع حضرتك د X", "معك دكتور X", "أنا دكتور X", "أنا الدكتور X", or any equivalent greeting/self-identification pattern (in Arabic or English: "this is Dr X", "you're speaking with Dr X", "I'm Dr X") are the AGENT'S OWN IDENTITY, never a doctor target. This applies to EVERY agent turn in the conversation, not only the first greeting — a call can be transferred through multiple agents, each introducing themselves the same way, and NONE of those names is ever a doctor target.

A Patient turn that merely ADDRESSES that same agent by the name/title they introduced themselves with — a greeting, a thanks, an acknowledgement ("حياك الله دكتور محمد", "شكرا دكتور محمد", "اهلا دكتور محمد") — does NOT change this. That name stays the agent's identity; it does not become patient_selected, agent_recommended, a booking_target, or a booking_confirmation just because the patient used it too.

When the ONLY doctor-shaped name(s) anywhere in the conversation come from self-introduction(s) (optionally echoed back by the patient), you MUST return:
  "doctor_name": null
  "doctor_role": "agent_self_introduction"
  "doctor_validation_needed": false
Do not partially "rescue" a self-introduced name into any other role.

This exclusion is NEVER global or permanent for that name string, only for THAT specific self-introduction mention: if a SEPARATE, later, independent part of the SAME conversation clearly and unambiguously names a real third-party clinical doctor for the patient to see (a recommendation, a booking, an inquiry) — even one who happens to share the same first name as an agent who introduced themselves earlier — that later mention IS a real doctor target and must be extracted normally, with doctor_role reflecting that real interaction (e.g. "agent_recommended", "patient_selected").

Examples that must produce doctor_name=null / doctor_role="agent_self_introduction" / doctor_validation_needed=false:
- "السلام عليكم مع حضرتك دكتور محمد من مجموعة اندلسية صحة"
- "السلام عليكم مع حضرتك د ابانوب من قسم الرعاية المنزلية"
- "مع حضرتك دكتور هشام" / "مع حضرتك د ابانوب"
- "انا الدكتور محمد" / "أنا دكتور يوسف"
- "السلام عليكم مع حضرتك دكتور محمد" followed later by "Patient: حياك الله دكتور محمد" / "شكرا دكتور محمد"

Examples that ARE real doctor targets (do NOT treat as self-introduction):
- "متاح معانا دكتور شريف" -> شريف, agent_recommended
- "متاح معنا الاستشاري اسامة عبد السلام" -> اسامة عبد السلام, agent_recommended
- "متاح معانا دكتورة وصال" -> وصال, agent_recommended
- "ابغى احجز مع دكتور احمد أفندي" -> احمد أفندي, patient_selected
- An agent who introduced themselves as "دكتور محمد" earlier, followed LATER by a genuinely separate "متاح معانا دكتور شريف عظام" -> شريف is still the valid target; محمد stays excluded

════════════════════════════════════════════════════════════
RULE #2 — ONLY A PLAUSIBLE HUMAN NAME, NEVER A FRAGMENT
════════════════════════════════════════════════════════════
Before returning a doctor name, you must be able to answer YES to this exact question: "Is this text actually being used as the name of a human doctor in this conversation?" If the answer is no, unclear, or the text is anything else, return null — do not guess, do not soften a rejection into a weaker candidate.

A دكتور/طبيب/Dr TITLE does NOT automatically mean whatever word or phrase follows it is a name. Read the surrounding sentence and judge what the word is actually doing.

Never return any of the following as a doctor name, even when they immediately follow a doctor title:
- a verb, verb phrase, or sentence fragment (e.g. "بعدين", "بتكتب", "بيكون", "لعمل", "وبعدها بتذهب للاستقبال", "تجنبا لطلب", "وبتسال الاستقبال")
- a medical specialty, subspecialty, or service name (e.g. "اورام", "عصب", "مخ واعصاب", "عصب ود تركيبات", "تركيبات") — "دكتور اورام" means the specialty is Oncology, not a doctor literally named "اورام"
- a department/team name (e.g. "التمريض" nursing, "قسم الاشعة" the radiology department) — "التمريض بيكون" is a sentence fragment about the nursing team, never a name
- a generic, unnamed doctor role or reference ("الطبيب المعالج", "الطبيب المختص", "اى طبيب", "زيارة طبيب", "مين", "مين / يوم ايه", "اخصائيين ممتازين")
- a restrictive/quantifying word attached to the title ("دكتور فقط" = "male doctors ONLY" — "فقط" is not a name; if a list follows such as "دكتور فقط\\nدكتور شريف\\nودكتور احمد", the real names are شريف and احمد, "فقط" is excluded)
- a question, booking/administrative/temporal phrase, location, price, or duration ("حجز موعد مع اى طبيب؟", "مواعيد متاحة", "في الطوارئ")
- any other text that is not, on its own, a plausible human personal name

A doctor name MAY be just ONE token (a first name only) when that is genuinely all the conversation ever states — never require two tokens, and never invent or pad a second token. Just as important: never let a real first name absorb an adjacent specialty/location/service word into the returned name. For example, "دكتورة وصال بعيادة الاورام" must yield doctor_name "وصال" (specialty context "اورام"), never "وصال بعياده" or "وصال بعيادة الاورام".

When the SAME doctor is referred to more than once in the conversation at different levels of completeness (e.g. "وصال" earlier, then the fuller "وصال محمد" later), return only the FULLEST name actually stated anywhere in the conversation — never both forms, never the shorter one once a fuller one appears.

If a doctor title appears with no name attached at all and no specific person is ever named ("ممكن نعمل زيارة طبيب في البداية", "ممكن نعمل زيارة طبيب عظام في البداية"), doctor_name is null — never invent a placeholder doctor. Still extract the specialty context when one is present ("عظام" in the second example).

════════════════════════════════════════════════════════════
SPECIALTY CONTEXT
════════════════════════════════════════════════════════════
Separately, extract the medical specialty/service/department context surrounding the doctor mention — or, when no doctor name qualifies at all, whatever specialty is still being discussed (e.g. "اورام" for oncology, "عصب" / "تركيبات" for endodontic/prosthodontic dental work, "الأنف والأذن والحنجرة" for ENT). This is conversational context only — never treat it as the doctor's authoritative CRM specialty, and never let it leak into the doctor name itself.

Do not guess or infer information that is not explicitly stated in the transcript. Do not include any additional commentary or explanation.

Respond ONLY with a valid JSON object, no markdown fences, with exactly these keys:
{{
  "doctor_name": "<the single fullest plausible human doctor name actually used as the active target in this conversation, or null if none qualifies>",
  "doctor_role": "<one of: agent_self_introduction, patient_selected, agent_recommended, booking_confirmation, mention, or null if no doctor name qualifies>",
  "doctor_context_specialty": "<the specialty/service/department phrase discussed around the doctor (or, if no doctor qualifies, around the request generally), or null if none is mentioned>",
  "doctor_validation_needed": <true if doctor_name is a genuine third-party clinical doctor target, false if it is null or the only mention was an agent self-introduction>
}}

Transcript:
{transcript}
"""