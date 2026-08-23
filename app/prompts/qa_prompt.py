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