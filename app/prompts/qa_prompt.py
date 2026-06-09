"""
qa_prompt.py — System prompt and user-prompt builder for the QA LangGraph pipeline.

Design philosophy
─────────────────
• The SYSTEM_PROMPT is intentionally SHORT.  It defines the analyst's role,
  output contract, and core analytical principles ONLY.  All compliance rules,
  pillar checklists, behavioral standards, scoring weights, and script templates
  are injected into the USER message by the criteria-loader nodes so they can
  be versioned, swapped, or extended without touching this file.

• build_user_prompt() accepts the four pre-rendered criteria blocks from the
  corresponding LangGraph nodes (load_behavioral_criteria, load_compliance_pillars,
  load_script_templates, load_scoring_weights).  Each block is inserted as a
  clearly labelled section so the LLM can locate any policy at a glance.
"""

from app.models.input import CallTranscript

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT  (role + output contract only — no compliance details)
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
- overall_assessment = "escalate"     → C2Com violation confirmed, dangerous misinformation, explicit aggression, or vulnerable patient harmed.
- overall_assessment = "needs_review" → One or more C2C/C2B violations observed but no escalation trigger.
- overall_assessment = "pass"         → No significant violations; include at least one positive compliance_flag.
- escalation_required must be true if and only if overall_assessment = "escalate".

## EDGE CASES
- Short transcript (< 100 words): note in assessment_reasoning; score conservatively at 0.5 for unobservable dimensions.
- Disconnected call: note abrupt end; do not penalize the agent.
- Foreign language: note language barrier; do not assess content accuracy.

All specific violation definitions, behavioral standards, script templates, and scoring weights
are provided in the user message below. Evaluate strictly against those criteria.
"""

# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT BUILDER
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

    Parameters
    ----------
    call                : The inbound CallTranscript.
    behavioral_criteria : Output of load_behavioral_criteria node (compact text).
    compliance_pillars  : Output of load_compliance_pillars node (compact text).
    script_templates    : Output of load_script_templates node (compact text).
    scoring_weights     : Output of load_scoring_weights node (compact text).

    All criteria parameters are optional — if empty the LLM falls back to its
    general knowledge, which is useful during unit tests or if a file is missing.
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
