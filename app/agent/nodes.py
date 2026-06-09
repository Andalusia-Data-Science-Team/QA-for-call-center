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
from app.prompts.qa_prompt import SYSTEM_PROMPT, build_user_prompt
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
# Node 3 – build_prompt
#   Assembles the final system + user prompts from the call transcript AND
#   all four criteria blocks loaded by the preceding nodes.
#   Each criteria section is injected as a clearly labelled block so the LLM
#   can locate any specific policy without re-reading the whole prompt.
# ---------------------------------------------------------------------------

async def build_prompt(state: AgentState) -> dict:
    """
    Assemble the full prompt from the transcript + the 4 loaded criteria blocks.

    Prompt structure (user message):
      [CALL METADATA]
      [BEHAVIORAL STANDARDS]    ← from load_behavioral_criteria
      [COMPLIANCE PILLARS]      ← from load_compliance_pillars
      [SCRIPT TEMPLATES]        ← from load_script_templates
      [SCORING WEIGHTS]         ← from load_scoring_weights
      [TRANSCRIPT]
      [OUTPUT SCHEMA]
    """
    call = state["call"]
    system = SYSTEM_PROMPT
    user = build_user_prompt(
        call,
        behavioral_criteria=state.get("behavioral_criteria", ""),
        compliance_pillars=state.get("compliance_pillars", ""),
        script_templates=state.get("script_templates", ""),
        scoring_weights=state.get("scoring_weights", ""),
    )

    logger.debug(
        "build_prompt | call_id=%s system_len=%d user_len=%d",
        call.call_id, len(system), len(user),
    )
    return {
        "system_prompt": system,
        "user_prompt": user,
        "node_trace": _trace(state, "build_prompt"),
    }


# ---------------------------------------------------------------------------
# Node 3 – llm_inference
#   Sends the prompts to the LLM and stores the raw response + usage stats.
# ---------------------------------------------------------------------------

async def llm_inference(state: AgentState, llm_client: LLMClient) -> dict:
    """
    Call the LLM.  The LLMClient already handles retry + exponential backoff.
    Stores raw_llm_text and usage metadata for downstream nodes.
    """
    call_id = state["call"].call_id
    try:
        raw_text, usage = await llm_client.complete(
            state["system_prompt"],
            state["user_prompt"],
        )
    except Exception as exc:
        logger.error("llm_inference failed | call_id=%s | %s", call_id, exc)
        return {
            "error": f"LLM call failed: {exc}",
            "error_node": "llm_inference",
            "node_trace": _trace(state, "llm_inference"),
        }

    logger.debug(
        "llm_inference | call_id=%s latency=%.0fms tokens_in=%s tokens_out=%s",
        call_id,
        usage.get("latency_ms", 0),
        usage.get("input_tokens") or usage.get("prompt_tokens"),
        usage.get("output_tokens") or usage.get("completion_tokens"),
    )
    return {
        "raw_llm_text": raw_text,
        "usage": usage,
        "node_trace": _trace(state, "llm_inference"),
    }


# ---------------------------------------------------------------------------
# Node 4 – parse_response
#   Strips markdown fences and JSON-parses the raw LLM text.
# ---------------------------------------------------------------------------

async def parse_response(state: AgentState) -> dict:
    """
    Strip ```json ... ``` wrappers and parse to a plain dict.
    Extend here to handle other output formats (YAML, partial JSON repair,
    structured extraction fallback, etc.).
    """
    call_id = state["call"].call_id
    raw = state["raw_llm_text"]
    clean = _strip_markdown_fences(raw)

    try:
        data: dict = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error(
            "parse_response JSON error | call_id=%s snippet=%s | %s",
            call_id,
            raw[:300],
            exc,
        )
        return {
            "error": f"LLM returned invalid JSON: {exc}",
            "error_node": "parse_response",
            "node_trace": _trace(state, "parse_response"),
        }

    return {
        "parsed_data": data,
        "node_trace": _trace(state, "parse_response"),
    }


# ---------------------------------------------------------------------------
# Node 5 – validate_output
#   Runs Pydantic validation on the parsed dict → QAAnalysisResult.
# ---------------------------------------------------------------------------

async def validate_output(state: AgentState) -> dict:
    """
    Validate the parsed dict against the QAAnalysisResult Pydantic schema.
    Always overrides call_id with the one from the request (never trust the LLM).
    Extend here to add cross-field business-rule validation.
    """
    call_id = state["call"].call_id
    data = {**state["parsed_data"], "call_id": call_id}   # trust request call_id

    try:
        result = QAAnalysisResult.model_validate(data)
    except ValidationError as exc:
        logger.error(
            "validate_output Pydantic error | call_id=%s errors=%s",
            call_id,
            exc.errors(),
        )
        return {
            "error": f"LLM response failed schema validation: {exc}",
            "error_node": "validate_output",
            "node_trace": _trace(state, "validate_output"),
        }

    return {
        "result": result,
        "node_trace": _trace(state, "validate_output"),
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
