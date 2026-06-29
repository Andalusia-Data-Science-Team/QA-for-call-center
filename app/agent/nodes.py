"""
LangGraph Node Definitions for the Call QA Analysis Pipeline.

Each node is a pure async function with the signature:
    async def node_name(state: AgentState) -> dict

Nodes return a *partial* state dict — LangGraph merges it back into the
running state automatically.

Current pipeline (in order):
  load_call        – validate the incoming CallTranscript & init trace
  build_prompt     – construct the system + user prompts
  llm_inference    – call the LLM (with retry logic from LLMClient)
  parse_response   – strip markdown fences & JSON-parse the raw text
  validate_output  – run Pydantic validation → QAAnalysisResult
  integrity_check  – fix escalation_required ↔ overall_assessment mismatches
  finalize         – stamp call_id, log summary, close trace

To add a NEW node (e.g. criteria_lookup, human_review, re_rank):
  1. Write your async function here following the same pattern.
  2. Import it in graph.py and add it to the graph.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.agent.state import AgentState
from app.models.output import QAAnalysisResult
from app.prompts.qa_prompt import (
    SYSTEM_PROMPT,
    build_behavioral_prompt,
    build_compliance_prompt,
    build_script_prompt,
    build_scoring_prompt,
    build_user_prompt,   # kept for legacy path
)
from app.services.criteria_loader import CriteriaLoader
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Shared loader instance — YAML files are lru_cache'd after first read
_criteria = CriteriaLoader()

# ---------------------------------------------------------------------------
# Helper – return a single-item list for the node_trace reducer.
# AgentState.node_trace uses operator.add as its reducer, so every node
# just returns ["node_name"] and LangGraph concatenates them automatically.
# ---------------------------------------------------------------------------

def _trace(_state: AgentState, node: str) -> list[str]:
    return [node]


# ---------------------------------------------------------------------------
# Node 1 – load_call
#   Validates the CallTranscript exists in state and initialises the trace.
# ---------------------------------------------------------------------------

async def load_call(state: AgentState) -> dict:
    """
    Entry-point node.
    Confirms the call transcript is present and resets any previous error.
    Extend here if you need to hydrate additional metadata (e.g. fetch
    patient history, look up CRM records) before analysis begins.
    """
    call = state.get("call")
    if call is None:
        return {
            "error": "AgentState.call is None — no transcript provided",
            "error_node": "load_call",
            "node_trace": _trace(state, "load_call"),
        }

    logger.info(
        "load_call | call_id=%s agent=%s dept=%s duration=%ss",
        call.call_id,
        call.agent_name,
        call.department,
        call.call_duration_seconds,
    )
    return {
        "error": None,
        "error_node": None,
        "node_trace": _trace(state, "load_call"),
    }


# ---------------------------------------------------------------------------
# Node 2a – load_behavioral_criteria
#   Loads the department-aware behavioral standards from YAML and renders
#   them as a compact plain-text block stored in state.
#   Isolated so you can swap files, add A/B variants, or version-gate
#   criteria without touching any other node.
# ---------------------------------------------------------------------------

async def load_behavioral_criteria(state: AgentState) -> dict:
    """
    Fetch and compact the behavioral policy for the call's department.
    Falls back to 'general' when a department-specific file is absent.
    Covers: script compliance, tone, empathy, prohibited behaviors, red flags.
    """
    dept = state["call"].department
    block = _criteria.behavioral(department=dept)
    logger.debug(
        "load_behavioral_criteria | call_id=%s dept=%s chars=%d",
        state["call"].call_id, dept, len(block),
    )
    return {
        "behavioral_criteria": block,
        "node_trace": _trace(state, "load_behavioral_criteria"),
    }


# ---------------------------------------------------------------------------
# Node 2b – load_compliance_pillars
#   Loads the 15 official compliance pillars from regulations.yaml, grouped
#   by severity tier (C2Com → C2C → C2B → NC) in compact form.
#   Isolated so the pillar set can be versioned or department-filtered
#   independently of behavioral criteria.
# ---------------------------------------------------------------------------

async def load_compliance_pillars(state: AgentState) -> dict:
    """
    Fetch and compact the full compliance pillar checklist.
    Strips ids, type-repetitions, and 'Other' catch-all items to minimise
    token count while preserving all actionable violation descriptions.
    """
    block = _criteria.compliance_pillars()
    logger.debug(
        "load_compliance_pillars | call_id=%s chars=%d",
        state["call"].call_id, len(block),
    )
    return {
        "compliance_pillars": block,
        "node_trace": _trace(state, "load_compliance_pillars"),
    }


# ---------------------------------------------------------------------------
# Node 2c – load_script_templates
#   Loads approved greeting and closing scripts.
#   Isolated so scripts can be updated per-campaign or per-language without
#   touching behavioral or compliance nodes.
# ---------------------------------------------------------------------------

async def load_script_templates(state: AgentState) -> dict:
    """
    Fetch and compact the approved greeting / closing script templates.
    The LLM uses these as the reference baseline for script-adherence flags.
    """
    block = _criteria.script_templates()
    logger.debug(
        "load_script_templates | call_id=%s chars=%d",
        state["call"].call_id, len(block),
    )
    return {
        "script_templates": block,
        "node_trace": _trace(state, "load_script_templates"),
    }


# ---------------------------------------------------------------------------
# Node 2d – load_scoring_weights
#   Loads the scoring dimension weights and pass/fail thresholds.
#   Isolated so scoring policy changes (e.g. reweighting empathy vs accuracy)
#   require only a YAML edit, no code change.
# ---------------------------------------------------------------------------

async def load_scoring_weights(state: AgentState) -> dict:
    """
    Fetch and compact the scoring weights (dimension breakdown, minimum
    passing score, critical-violation deduction).
    """
    block = _criteria.scoring_weights()
    logger.debug(
        "load_scoring_weights | call_id=%s chars=%d",
        state["call"].call_id, len(block),
    )
    return {
        "scoring_weights": block,
        "node_trace": _trace(state, "load_scoring_weights"),
    }


# ---------------------------------------------------------------------------
# ── FOCUSED INFERENCE NODES ─────────────────────────────────────────────────
#
# The original single build_prompt → llm_inference → parse_response →
# validate_output chain has been REPLACED by four focused inference nodes:
#
#   infer_behavioral_evaluation  – tone, empathy, professionalism, red flags
#   infer_compliance_evaluation  – 15 compliance pillars (C2Com/C2C/C2B/NC)
#   infer_script_matching        – greeting / closing script adherence
#   infer_overall_scoring        – synthesises the three above + scoring weights
#
# Each node calls the LLM with a narrow prompt, parses the response, and
# stores its result in a dedicated state key.  aggregate_results merges all
# four into the final QAAnalysisResult (schema unchanged).
# ---------------------------------------------------------------------------


async def _focused_llm_call(
    node_name: str,
    call_id: str,
    user_prompt: str,
    llm_client: LLMClient,
    state: AgentState,
) -> tuple[dict | None, dict | None]:
    """
    Internal helper: call the LLM, log usage, parse JSON.
    Returns (parsed_dict, error_dict).  Exactly one of the two will be None.
    """
    try:
        raw_text, usage = await llm_client.complete(SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.error("%s LLM call failed | call_id=%s | %s", node_name, call_id, exc)
        return None, {
            "error": f"{node_name}: LLM call failed: {exc}",
            "error_node": node_name,
            "node_trace": _trace(state, node_name),
        }

    logger.debug(
        "%s | call_id=%s latency=%.0fms tokens_in=%s tokens_out=%s",
        node_name,
        call_id,
        usage.get("latency_ms", 0),
        usage.get("input_tokens") or usage.get("prompt_tokens"),
        usage.get("output_tokens") or usage.get("completion_tokens"),
    )

    clean = _strip_markdown_fences(raw_text)
    try:
        data: dict = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error(
            "%s JSON parse error | call_id=%s snippet=%s | %s",
            node_name, call_id, raw_text[:300], exc,
        )
        return None, {
            "error": f"{node_name}: LLM returned invalid JSON: {exc}",
            "error_node": node_name,
            "node_trace": _trace(state, node_name),
        }

    return data, None


# ---------------------------------------------------------------------------
# Node A – infer_behavioral_evaluation
#   Focused LLM call: professionalism, tone, empathy, red flags.
#   Runs in PARALLEL with infer_compliance_evaluation & infer_script_matching.
# ---------------------------------------------------------------------------

async def infer_behavioral_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a behavioral-only prompt.
    Produces: professionalism_score, behavioral_flags, strengths, improvements.
    Result stored in state["behavioral_eval"].
    """
    call = state["call"]
    user_prompt = build_behavioral_prompt(
        call,
        behavioral_criteria=state.get("behavioral_criteria", ""),
    )
    logger.debug(
        "infer_behavioral_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_behavioral_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    return {
        "behavioral_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_behavioral_evaluation"),
    }


# ---------------------------------------------------------------------------
# Node B – infer_compliance_evaluation
#   Focused LLM call: 15 compliance pillars (C2Com / C2C / C2B / NC).
#   Runs in PARALLEL with infer_behavioral_evaluation & infer_script_matching.
# ---------------------------------------------------------------------------

async def infer_compliance_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a compliance-pillars-only prompt.
    Produces: compliance_flags, escalation_required, escalation_reason.
    Result stored in state["compliance_eval"].
    """
    call = state["call"]
    user_prompt = build_compliance_prompt(
        call,
        compliance_pillars=state.get("compliance_pillars", ""),
    )
    logger.debug(
        "infer_compliance_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_compliance_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    return {
        "compliance_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_compliance_evaluation"),
    }


# ---------------------------------------------------------------------------
# Node C – infer_script_matching
#   Focused LLM call: greeting / closing script adherence.
#   Runs in PARALLEL with infer_behavioral_evaluation & infer_compliance_evaluation.
# ---------------------------------------------------------------------------

async def infer_script_matching(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a script-adherence-only prompt.
    Produces: accuracy_score, script_flags.
    Result stored in state["script_eval"].
    """
    pass
    """ call = state["call"]
    user_prompt = build_script_prompt(
        call,
        script_templates=state.get("script_templates", ""),
    )
    logger.debug(
        "infer_script_matching | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_script_matching", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    return {
        "script_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_script_matching"),
    } """


# ---------------------------------------------------------------------------
# Node D – infer_overall_scoring
#   Focused LLM call: synthesise behavioral + compliance + script results
#   into the final overall_assessment + scores.
#   Runs AFTER the three parallel nodes complete (fan-in barrier).
# ---------------------------------------------------------------------------

async def infer_overall_scoring(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a scoring-only prompt that receives the three sub-results
    as context.  Produces: overall_assessment, assessment_reasoning,
    resolution_score, escalation_required, escalation_reason.
    Result stored in state["scoring_eval"].
    """
    call = state["call"]

    # Serialise the three sub-results as compact JSON strings for injection
    behavioral_summary = json.dumps(state.get("behavioral_eval") or {}, ensure_ascii=False)
    compliance_summary = json.dumps(state.get("compliance_eval") or {}, ensure_ascii=False)
    script_summary     = json.dumps(state.get("script_eval") or {},     ensure_ascii=False)

    user_prompt = build_scoring_prompt(
        call,
        scoring_weights=state.get("scoring_weights", ""),
        behavioral_summary=behavioral_summary,
        compliance_summary=compliance_summary,
        script_summary=script_summary,
    )
    logger.debug(
        "infer_overall_scoring | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_overall_scoring", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    return {
        "scoring_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_overall_scoring"),
    }


# ---------------------------------------------------------------------------
# Node E – aggregate_results
#   Fan-in barrier: merges the four sub-evaluation dicts into one
#   parsed_data dict that matches the QAAnalysisResult schema, then runs
#   Pydantic validation.  No LLM call — pure Python merge.
# ---------------------------------------------------------------------------

async def aggregate_results(state: AgentState) -> dict:
    """
    Merge sub-evaluation results into a single QAAnalysisResult-compatible dict.

    Merge strategy
    ──────────────
    • compliance_flags  = behavioral_flags  (Node A)
                        + compliance_flags  (Node B)
                        + script_flags      (Node C)
    • agent_performance = professionalism_score (A)
                        + accuracy_score        (C)
                        + resolution_score      (D)
                        + strengths             (A)
                        + improvements          (A)
    • overall_assessment, assessment_reasoning,
      escalation_required, escalation_reason     ← Node D (scoring)
    """
    call_id = state["call"].call_id

    behavioral = state.get("behavioral_eval") or {}
    compliance = state.get("compliance_eval") or {}
    script     = state.get("script_eval")     or {}
    scoring    = state.get("scoring_eval")    or {}

    # Merge all compliance flag lists from the three focused evaluations
    all_flags: list[dict] = (
        behavioral.get("behavioral_flags", [])
        + compliance.get("compliance_flags", [])
        + script.get("script_flags", [])
    )

    merged: dict = {
        "call_id": call_id,
        "agent_name": state["call"].agent_name,
        "overall_assessment": scoring.get("overall_assessment", "needs_review"),
        "assessment_reasoning": scoring.get("assessment_reasoning", ""),
        "compliance_flags": all_flags,
        "agent_performance": {
            "professionalism_score": behavioral.get("professionalism_score", 0.5),
            "accuracy_score":        script.get("accuracy_score", 0.5),
            "resolution_score":      scoring.get("resolution_score", 0.5),
            "strengths":    behavioral.get("strengths", []),
            "improvements": behavioral.get("improvements", []),
        },
        "escalation_required": scoring.get("escalation_required", False),
        "escalation_reason":   scoring.get("escalation_reason"),
    }

    logger.debug(
        "aggregate_results | call_id=%s flags=%d assessment=%s",
        call_id, len(all_flags), merged["overall_assessment"],
    )

    try:
        result = QAAnalysisResult.model_validate(merged)
    except ValidationError as exc:
        logger.error(
            "aggregate_results Pydantic error | call_id=%s errors=%s",
            call_id, exc.errors(),
        )
        return {
            "error": f"Aggregated result failed schema validation: {exc}",
            "error_node": "aggregate_results",
            "node_trace": _trace(state, "aggregate_results"),
        }

    return {
        "parsed_data": merged,
        "result": result,
        "node_trace": _trace(state, "aggregate_results"),
    }


# ---------------------------------------------------------------------------
# Node 6 – integrity_check
#   Ensures escalation_required ↔ overall_assessment are consistent.
# ---------------------------------------------------------------------------

async def integrity_check(state: AgentState) -> dict:
    """
    Fix LLM inconsistencies between escalation_required and overall_assessment.
    Extend here to add additional post-processing rules, e.g.:
      - Force 'escalate' if any C2Com flag is critical
      - Cap scores if certain pillars are violated
    """
    call_id = state["call"].call_id
    result = state["result"]

    if result.escalation_required and result.overall_assessment != "escalate":
        logger.warning(
            "integrity_check: escalation_required=True but assessment=%s — correcting | call_id=%s",
            result.overall_assessment,
            call_id,
        )
        result = result.model_copy(update={"overall_assessment": "escalate"})

    if result.overall_assessment == "escalate" and not result.escalation_required:
        logger.warning(
            "integrity_check: assessment='escalate' but escalation_required=False — correcting | call_id=%s",
            call_id,
        )
        result = result.model_copy(update={"escalation_required": True})

    return {
        "result": result,
        "node_trace": _trace(state, "integrity_check"),
    }


# ---------------------------------------------------------------------------
# Node 7 – finalize
#   Logs the outcome summary and closes out the node trace.
# ---------------------------------------------------------------------------

async def finalize(state: AgentState) -> dict:
    """
    Last node in the happy path.  Logs summary and stamps the trace.
    Extend here to:
      - Emit metrics to Prometheus / Datadog
      - Persist results to a database
      - Trigger downstream webhooks
    """
    result = state["result"]
    call_id = state["call"].call_id

    logger.info(
        "finalize | call_id=%s assessment=%s escalate=%s trace=%s",
        call_id,
        result.overall_assessment,
        result.escalation_required,
        " → ".join(state.get("node_trace", [])),
    )
    return {
        "node_trace": _trace(state, "finalize"),
    }


# ---------------------------------------------------------------------------
# Node – handle_error  (terminal error path)
#   Called whenever any node sets state["error"].  Builds a safe error result.
# ---------------------------------------------------------------------------

async def handle_error(state: AgentState) -> dict:
    """
    Terminal error handler.  Converts a pipeline failure into a minimal
    QAAnalysisResult so callers always receive a well-typed response.
    """
    call = state.get("call")
    call_id = call.call_id if call else "UNKNOWN"
    reason = state.get("error", "Unknown error")
    error_node = state.get("error_node", "unknown")

    logger.error(
        "handle_error | call_id=%s node=%s reason=%s",
        call_id,
        error_node,
        reason,
    )

    error_result = QAAnalysisResult.error_result(call_id, reason)
    return {
        "result": error_result,
        "node_trace": _trace(state, f"handle_error[{error_node}]"),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers that some LLMs insert despite instructions."""
    text = text.strip()
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text
