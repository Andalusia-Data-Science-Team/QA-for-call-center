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
3. PROPORTIONATE: Distinguish critical violations from minor imperfections using the severity tiers in the criteria.
4. DEVELOPMENTAL: Balance corrective feedback with recognition of positive agent behaviors.
5. STRUCTURED OUTPUT: Return ONLY a valid JSON object matching the schema provided. No prose, no markdown fences.

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
just be realistic not overly harsh. If the transcript is too short to assess a dimension, score it at 0.5 and note that in the reasoning.
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
    """
    Prompt focused SOLELY on behavioral standards.
    Returns a JSON fragment with professionalism flags and agent strengths/improvements.
    """
    return f"""\
Evaluate the agent's BEHAVIORAL performance in the call below.
Focus exclusively on: tone, professionalism, empathy, active listening, prohibited phrases, and red-flag language.
Do NOT evaluate compliance pillars, script adherence, or scoring weights here.

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
      "severity": "<critical | moderate | minor | positive>",
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
    """
    Prompt focused SOLELY on the 15 compliance pillars.
    Returns a JSON fragment with all pillar violations and an escalation signal.
    """
    return f"""\
Evaluate the call below against the official COMPLIANCE PILLARS only.
Flag every violation by its exact pillar name and type (C2Com / C2C / C2B / NC).
Do NOT evaluate behavioral tone, script adherence, or scoring weights here.

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
      "severity": "<critical | moderate | minor | positive>",
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
    """
    Prompt focused SOLELY on greeting/closing script adherence.
    Returns a JSON fragment with script-adherence flags and an accuracy score.
    """
    return f"""\
Evaluate the agent's adherence to the APPROVED SCRIPT TEMPLATES in the call below.
Evaluate only the greeting and closing sections. Do not strictly enforce the exact scripts provided; instead, assess whether the conversation aligns with their intended purpose and conveys the expected concepts.
Do NOT evaluate compliance pillars, behavioral tone, or scoring weights here.

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
      "severity": "<critical | moderate | minor | positive>",
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
  1. Review the three sub-evaluation results provided below.
  2. Apply the scoring weights to produce the final dimension scores.
  3. Determine the overall_assessment ("pass" / "needs_review" / "escalate").
  4. Write the assessment_reasoning (2-4 sentences citing specific evidence).
  5. Confirm or correct escalation_required based on the compliance results.

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
SUB-EVALUATION 3 — SCRIPT MATCHING  
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
  "resolution_score": <0.0–1.0>,
  "escalation_required": <true | false>,
  "escalation_reason": "<reason string or null>"
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
    "accuracy_score": <0.0–1.0>,
    "resolution_score": <0.0–1.0>,
    "strengths": ["<strength 1>", "<strength 2>"],
    "improvements": ["<improvement 1>", "<improvement 2>"]
  }},
  "escalation_required": <true | false>,
  "escalation_reason": "<reason string or null>"
}}
"""
