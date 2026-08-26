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
import unicodedata
import urllib.parse
from datetime import date as DateType
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from sqlalchemy import create_engine, text as sa_text

from arabic_reshaper import reshape

from app.agent.state import AgentState
from app.models.output import QAAnalysisResult
from app.prompts.qa_prompt import (
    SYSTEM_PROMPT,
    APPOINTMENT_EXTRACTION_PROMPT,
    build_behavioral_prompt,
    build_compliance_prompt,
    build_reservation_prompt,
    build_offer_prompt,
    build_script_prompt,
    build_scoring_prompt,
    build_user_prompt,   # kept for legacy path
)
from app.services.criteria_loader import CriteriaLoader
from app.services.llm_client import LLMClient
from app.services.sql_helpers import insert_qa_result
from app.bank_node.bank_validation import bank_validation_needed, validate_ksa_bank_information, detect_bank_signals
from app.location_node.location_validation import detect_location_intent, detect_location_signals, is_home_service_location_text, location_validation_needed, validate_location_request
from app.services.text_helpers import (
    _normalize_arabic,
    _arabic_like_pattern, 
    _strip_markdown_fences,
    _norm_score,
)

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
# Node 2c – load_reservation_pillars
#   Loads the 15 official reservation pillars from regulations.yaml, grouped
#   by severity tier (C2Com → C2C → C2B → NC) in compact form.
#   Isolated so the pillar set can be versioned or department-filtered
#   independently of behavioral criteria.
# ---------------------------------------------------------------------------

async def load_reservation_pillars(state: AgentState) -> dict:
    """
    Fetch and compact the full reservation pillar checklist.
    Strips ids, type-repetitions, and 'Other' catch-all items to minimise
    token count while preserving all actionable violation descriptions.
    """
    block = _criteria.reservation_pillars()
    logger.debug(
        "load_reservation_pillars | call_id=%s chars=%d",
        state["call"].call_id, len(block),
    )
    return {
        "reservation_pillars": block,
        "node_trace": _trace(state, "load_reservation_pillars"),
    }

# ---------------------------------------------------------------------------
# Node 2d – load_script_templates
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
# Node 2f – load_offer_pillars
#   Loads the offer recommendation evaluation pillars from offer_regulations.yaml.
#   Isolated so offer policy can be updated independently.
# ---------------------------------------------------------------------------

async def load_offer_pillars(state: AgentState) -> dict:
    """
    Fetch and compact the offer recommendation pillar checklist.
    Includes evaluation guidance (when to check, when not to penalise, outcomes).
    """
    block = _criteria.offer_pillars()
    logger.debug(
        "load_offer_pillars | call_id=%s chars=%d",
        state["call"].call_id, len(block),
    )
    return {
        "offer_pillars": block,
        "node_trace": _trace(state, "load_offer_pillars"),
    }


# ---------------------------------------------------------------------------
# Node C – infer_reservation_evaluation
#   Focused LLM call: 6 reservation pillars (C2Com / C2C / C2B / NC).
#   Runs in PARALLEL with infer_behavioral_evaluation & infer_script_matching.
# ---------------------------------------------------------------------------

async def infer_reservation_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a reservation-pillars-only prompt.
    Produces: reservation_flags, escalation_required, escalation_reason.
    Result stored in state["reservation_eval"].
    """
    call = state["call"]
    user_prompt = build_reservation_prompt(
        call,
        appointment_verification=state.get("appointment_verification", ""),
        reservation_pillars=state.get("reservation_pillars", ""),
    )
    logger.debug(
        "infer_reservation_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_reservation_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    return {
        "reservation_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_reservation_evaluation"),
    }

# ---------------------------------------------------------------------------
# Node G – fetch_crm_offers_for_call
#   Looks up active CRM offers for the patient's specialty using the same
#   crm_offers layer used by the booking chatbot.  Runs AFTER inference_gate
#   (so appointment_details.specialty_name is already in state) and BEFORE
#   infer_offer_evaluation so the LLM can cross-check what the agent said
#   against what was actually available in the CRM at call time.
#
#   Uses get_offers_for_specialty() from crm_offers.py with:
#     • specialty_en   — from appointment_details or call.department
#     • patient_gender — inferred from the patient's name (Arabic suffix heuristic)
#
#   On any failure (CRM unreachable, specialty missing, etc.) it stores ""
#   so infer_offer_evaluation falls back gracefully to transcript-only mode.
# ---------------------------------------------------------------------------

async def fetch_crm_offers_for_call(state: AgentState) -> dict:
    """
    Fetch live CRM offers for the call's specialty and serialise them as a
    compact JSON string for injection into the offer evaluation prompt.

    Specialty resolution priority:
      1. appointment_details["specialty_name"]  (extracted by LLM from transcript)
      2. call.department                         (metadata fallback)

    Gender is inferred from appointment_details["patient_name"] using the
    same Arabic suffix heuristic used by the chatbot.
    """
    call = state.get("call")
    call_id = call.call_id if call else "UNKNOWN"

    # ── Resolve specialty ────────────────────────────────────────────────
    details: dict = state.get("appointment_details") or {}
    specialty_en: str = (
        details.get("specialty_name")
        or (call.department if call else "")
        or ""
    ).strip()

    if not specialty_en:
        logger.info(
            "fetch_crm_offers_for_call | call_id=%s — no specialty resolved, skipping CRM fetch",
            call_id,
        )
        return {
            "crm_offers_context": "",
            "node_trace": _trace(state, "fetch_crm_offers_for_call"),
        }

    # ── Infer patient gender from name (Arabic suffix heuristic) ────────
    patient_name: str = details.get("patient_name") or ""
    patient_gender: str | None = None
    if patient_name:
        first = patient_name.strip().split()[0]
        if first.endswith(("اء", "ية", "ّة", "ة", "ى", "يه")):
            patient_gender = "female"
        # names that don't match the female suffixes → leave as None (unknown)

    logger.info(
        "fetch_crm_offers_for_call | call_id=%s specialty=%r gender=%s",
        call_id, specialty_en, patient_gender or "unknown",
    )

    # ── Fetch from CRM ───────────────────────────────────────────────────
    try:
        # Import here (not at module top) so the QA pipeline doesn't hard-fail
        # when the CRM is not configured — the node just returns empty context.
        from app.offers_node.crm_offers import get_offers_for_specialty, format_offer_card

        # If the transcript explicitly named an offer, use it as the service hint
        # so get_offers_for_specialty performs a direct-name lookup first.
        offer_name_hint: str = (details.get("offer_name") or "").strip()

        offers = get_offers_for_specialty(
            specialty_en=specialty_en,
            service_hint=offer_name_hint,  # direct offer name if mentioned, else ""
            specialty_only=not bool(offer_name_hint),  # skip specialty-only guard when name is known
            patient_gender=patient_gender,
        )
    except Exception as exc:
        logger.warning(
            "fetch_crm_offers_for_call | call_id=%s — CRM fetch failed (non-fatal): %s",
            call_id, exc,
        )
        return {
            "crm_offers_context": "",
            "node_trace": _trace(state, "fetch_crm_offers_for_call"),
        }

    if not offers:
        logger.info(
            "fetch_crm_offers_for_call | call_id=%s — no active offers found for specialty=%r",
            call_id, specialty_en,
        )
        return {
            "crm_offers_context": f"No active CRM offers found for specialty: {specialty_en!r}",
            "node_trace": _trace(state, "fetch_crm_offers_for_call"),
        }

    # ── Serialise offers as compact human-readable cards ─────────────────
    # Use the same format_offer_card formatter the chatbot uses so the LLM
    # sees the same representation the agent would have seen.
    lines: list[str] = [
        f"Active CRM offers for specialty: {specialty_en!r}  "
        f"(patient_gender={patient_gender or 'unknown'})",
        f"Total: {len(offers)} offer(s) returned",
        "",
    ]
    for i, offer in enumerate(offers, 1):
        status    = offer.get("Offer_Status", "")
        is_alt    = offer.get("_is_alternative", False)
        card      = format_offer_card(offer, lang="ar")
        flag      = " [PRIMARY]" if not is_alt else " [ALTERNATIVE]"
        lines.append(f"--- Offer {i}{flag}  status={status} ---")
        lines.append(card)
        lines.append("")

    crm_context = "\n".join(lines)
    logger.info(
        "fetch_crm_offers_for_call | call_id=%s — serialised %d offer(s) (%d chars)",
        call_id, len(offers), len(crm_context),
    )
    return {
        "crm_offers_context": crm_context,
        "node_trace": _trace(state, "fetch_crm_offers_for_call"),
    }


# ---------------------------------------------------------------------------
# Node F – infer_offer_evaluation
#   Focused LLM call: did the agent correctly identify, present, and handle a
#   relevant CRM promotional offer?
#   Runs AFTER fetch_crm_offers_for_call (sequential dependency) so it
#   always has real CRM data to cross-check against.
# ---------------------------------------------------------------------------

async def infer_offer_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with an offer-evaluation-only prompt.

    Evaluates the agent's offer recommendation behaviour against the four
    possible outcomes:
      • SUITABLE_OFFER_RECOMMENDED   — positive flag
      • OFFER_SKIPPED                — C2B moderate flag
      • UNRELATED_OFFER_RECOMMENDED  — C2B moderate flag
      • OFFER_MISREPRESENTED         — C2B moderate flag
      • INCOMPLETE_OFFER_PRESENTATION — NC minor flag
      • MISSING_OFFER_CONFIRMATION_ASK — NC minor flag
      • NO_OFFER_AVAILABLE           — no flag
      • OFFER_NOT_APPLICABLE         — no flag

    Result stored in state["offer_eval"].
    """
    call = state["call"]

    user_prompt = build_offer_prompt(
        call,
        offer_pillars=state.get("offer_pillars", ""),
        crm_offers_context=state.get("crm_offers_context") or "",
    )
    logger.debug(
        "infer_offer_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_offer_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    outcome = data.get("offer_outcome", "OFFER_NOT_APPLICABLE")
    logger.info(
        "infer_offer_evaluation | call_id=%s outcome=%s flags=%d",
        call.call_id,
        outcome,
        len(data.get("offer_flags", [])),
    )

    return {
        "offer_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_offer_evaluation"),
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
    reservation_summary = json.dumps(state.get("reservation_eval") or {}, ensure_ascii=False)
    script_summary     = json.dumps(state.get("script_eval") or {},     ensure_ascii=False)
    # Include the masked deterministic outcomes in scoring; otherwise these
    # graph nodes would run without being able to influence the final
    # assessment. Bank and location are independent nodes/state keys now.
    bank_summary       = json.dumps(state.get("bank_validation") or {}, ensure_ascii=False)
    location_summary   = json.dumps(state.get("location_validation") or {}, ensure_ascii=False)

    user_prompt = build_scoring_prompt(
        call,
        scoring_weights=state.get("scoring_weights", ""),
        behavioral_summary=behavioral_summary,
        compliance_summary=compliance_summary,
        reservation_summary=reservation_summary,
        script_summary=script_summary,
        bank_summary=bank_summary,
        location_summary=location_summary,
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
                        + reservation_flags (Node C)
                        + script_flags      (Node D)
    • agent_performance = professionalism_score  (A)
                        + Agent Classification   (scoring)
                        + Profiling Comment      (scoring)
                        + strengths              (A)
                        + improvements           (A)
    • resolution_score                           ← top-level from Node D (scoring)
    • overall_assessment, assessment_reasoning,
      escalation_required, escalation_reason     ← Node D (scoring)
    """
    call_id = state["call"].call_id

    behavioral  = state.get("behavioral_eval")  or {}
    compliance  = state.get("compliance_eval")  or {}
    reservation = state.get("reservation_eval") or {}
    offer       = state.get("offer_eval")       or {}
    script      = state.get("script_eval")      or {}
    scoring     = state.get("scoring_eval")     or {}
    bank        = state.get("bank_validation") or {}
    location    = state.get("location_validation") or {}

    # Normalise offer_flags: the offer node uses "C2B"/"NC"/"positive" as type
    # but the ComplianceFlag schema expects type in {C2Com,C2C,C2B,NC}.
    # Map "positive" type to "NC" with severity "positive" so Pydantic validates.
    _raw_offer_flags: list[dict] = offer.get("offer_flags", [])
    _offer_flags: list[dict] = []
    for f in _raw_offer_flags:
        flag = dict(f)
        if flag.get("type") == "positive":
            flag["type"] = "NC"
            flag["severity"] = "positive"
        _offer_flags.append(flag)

    # Persist masked C2B findings through the existing DWH flag path. Both
    # comparisons were already completed deterministically — bank and
    # location are independent nodes, so each gets its own flag when it
    # independently flags a violation.
    bank_flags = []
    if bank.get("is_violation"):
        bank_flags.append({
            "type": "C2B", "severity": "moderate",
            "description": bank.get("reason", "KSA bank-account validation failed."),
            "transcript_excerpt": "Bank information supplied by agent (identifier masked).",
        })
    location_flags = []
    if location.get("is_violation"):
        location_flags.append({
            "type": "C2B", "severity": "moderate",
            "description": location.get("reason", "KSA branch/location validation failed."),
            "transcript_excerpt": "Location information supplied by agent.",
        })

    # Merge all compliance flag lists from all focused evaluations
    all_flags: list[dict] = (
        behavioral.get("behavioral_flags", [])
        + compliance.get("compliance_flags", [])
        + reservation.get("reservation_flags", [])
        + _offer_flags
        + bank_flags
        + location_flags
        + script.get("script_flags", [])
        + scoring.get("compliance_flags", [])
    )

    # Deduplicate flags by (type + transcript_excerpt[:80])
    seen: set[tuple] = set()
    deduped_flags: list[dict] = []
    for flag in all_flags:
        key = (flag.get("type"), flag.get("transcript_excerpt", "")[:80])
        if key not in seen:
            seen.add(key)
            deduped_flags.append(flag)

    scoring_perf: dict = scoring.get("agent_performance", {})

    merged: dict = {
        "call_id":              call_id,
        "agent_name":           state["call"].agent_name,
        "overall_assessment":   scoring.get("overall_assessment", "needs_review"),
        "assessment_reasoning": scoring.get("assessment_reasoning", ""),
        "compliance_flags":     deduped_flags,
        "agent_performance": {
            # professionalism_score — prefer behavioral node, fall back to scoring
            "professionalism_score": _norm_score(
                behavioral.get("professionalism_score")
                or scoring_perf.get("professionalism_score", 0.5)
            ),
            # Agent Classification — required, from scoring node
            "Agent Classification": (
                scoring_perf.get("Agent Classification")
                or scoring.get("Agent Classification")
                or "D"
            ),
            # Profiling Comment — optional, null when absent
            "Profiling Comment": (
                scoring_perf.get("Profiling Comment")
                or scoring.get("Profiling Comment")
                or None
            ),
            # strengths & improvements — prefer behavioral node
            "strengths":    behavioral.get("strengths",    scoring_perf.get("strengths",    [])),
            "improvements": behavioral.get("improvements", scoring_perf.get("improvements", [])),
        },
        # resolution_score and accuracy_score removed
        "escalation_required": scoring.get("escalation_required", False),
        "escalation_reason":   scoring.get("escalation_reason"),
        "conversation_link":   state["call"].conversation_link,
        "bank_validation":  bank,
        "location_validation": location,
    }

    logger.debug(
        "aggregate_results | call_id=%s flags=%d assessment=%s classification=%s profiling=%s offer_outcome=%s",
        call_id,
        len(deduped_flags),
        merged["overall_assessment"],
        merged["agent_performance"].get("Agent Classification"),
        merged["agent_performance"].get("Profiling Comment"),
        offer.get("offer_outcome", "N/A"),
    )

    try:
        result = QAAnalysisResult.model_validate(merged)
    except ValidationError as exc:
        logger.error(
            "aggregate_results Pydantic error | call_id=%s errors=%s merged_keys=%s",
            call_id, exc.errors(), list(merged.keys()),
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

    # A bank or location violation cannot remain a pass even if the LLM
    # overlooks the deterministic scoring context; severity/escalation stay
    # policy-driven. Bank and location are independent checks — either one
    # alone is enough to trigger this.
    bank_violation = (state.get("bank_validation") or {}).get("is_violation")
    location_violation = (state.get("location_validation") or {}).get("is_violation")
    if (bank_violation or location_violation) and result.overall_assessment == "pass":
        result = result.model_copy(update={"overall_assessment": "needs_review"})

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
# Node 7 – save_to_database
#   Persists the final QAAnalysisResult to [Service_Hub].[AI].[Call_QA_Results].
#   Runs after integrity_check and before finalize so the DB write is
#   confirmed before the pipeline terminates.
#   Errors are logged but do NOT abort the pipeline (non-fatal).
# ---------------------------------------------------------------------------

async def save_to_database(state: AgentState) -> dict:
    """
    Persist the validated QAAnalysisResult to the SQL Server database.

    Uses ``insert_qa_result`` from ``app.services.sql_helpers``.  Any
    database error is caught, logged, and surfaced as a warning in the
    node trace rather than crashing the pipeline — the caller already has
    the in-memory result and should not lose it due to a transient DB issue.
    """
    result = state.get("result")
    call   = state.get("call")

    if result is None or call is None:
        logger.warning(
            "save_to_database | result or call is None — skipping DB write"
        )
        return {"node_trace": _trace(state, "save_to_database")}

    if result.overall_assessment == "error":
        logger.warning(
            "save_to_database | call_id=%s — skipping DB write for error result",
            call.call_id,
        )
        return {"node_trace": _trace(state, "save_to_database")}

    try:
        insert_qa_result(result, call, channel_name="Whatsapp")
        logger.info(
            "save_to_database | call_id=%s — result persisted to DB",
            call.call_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "save_to_database | call_id=%s — DB insert failed (non-fatal): %s",
            call.call_id,
            exc,
        )

    return {"node_trace": _trace(state, "save_to_database")}


# ---------------------------------------------------------------------------
# Node 8 – finalize
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
# Node – detect_intent
#   Inspects the call transcript for booking / appointment-related intent by
#   scanning for Arabic keywords such as "احجز", "حجز", "معاد" (and common
#   variants).  Stores two keys in state:
#     • is_booking_intent  (bool)  – True when any keyword is matched
#     • intent_label       (str)   – human-readable label for the intent
# ---------------------------------------------------------------------------

# Keywords that signal a booking or appointment-checking intent
_BOOKING_KEYWORDS: list[str] = [
    "احجز",    # "book" (imperative)
    "احجزلي",  # "book for me"
    "حجز",     # "booking / reservation"
    "حجزي",    # "my booking"
    "حجزك",    # "your booking"
    "حجزه",    # "his/her booking"
    "الحجز",   # "the booking"
    "معاد",    # "appointment"
    "المعاد",  # "the appointment"
    "معادي",   # "my appointment"
    "موعد",    # "appointment / time-slot"
    "المواعيد", # "the appointments"
    "مواعيد",  # "appointments"
    "تأكيد الحجز",
    "الموعد",
    "تم تأكيد الحجز"
]


async def detect_intent(state: AgentState) -> dict:
    """
    Keyword-based intent detection node.

    Scans the full transcript text for Arabic booking / appointment keywords.
    Sets:
      - ``is_booking_intent``: True when at least one keyword is found.
      - ``intent_label``: A short descriptive string ("booking_or_appointment"
        or "other").

    This node never raises — on any error it logs and defaults to
    ``is_booking_intent=False`` so the rest of the pipeline continues
    unaffected.
    """
    call = state.get("call")
    if call is None:
        logger.warning("detect_intent | call is None — skipping intent detection")
        return {
            "is_booking_intent": False,
            "intent_label": "other",
            "node_trace": _trace(state, "detect_intent"),
        }

    # Concatenate all transcript turns into one searchable string.
    # Turns may be plain strings or objects that carry a .text attribute.
    try:
        raw_turns = state.get("call").transcript or []
        joined = "".join(
            (turn if isinstance(turn, str) else turn.text)
            for turn in raw_turns
            if (turn if isinstance(turn, str) else getattr(turn, "text", None))
        )
        # reshape operates on the full joined string so Arabic ligatures are
        # normalised correctly — it does NOT split the text.
        transcript_text = joined       #reshape(joined)
        print(transcript_text)
        logger.debug("detect_intent | transcript_text=%s", transcript_text[:200])
    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_intent | failed to read transcript | %s", exc)
        return {
            "is_booking_intent": False,
            "intent_label": "other",
            "node_trace": _trace(state, "detect_intent"),
        }

    matched_keywords = [kw for kw in _BOOKING_KEYWORDS if kw in transcript_text]
    is_booking = bool(matched_keywords)
    intent_label = "booking_or_appointment" if is_booking else "other"

    logger.info(
        "detect_intent | call_id=%s is_booking=%s matched=%s",
        call.call_id,
        is_booking,
        matched_keywords or "none",
    )

    return {
        "is_booking_intent": is_booking,
        "intent_label": intent_label,
        "node_trace": _trace(state, "detect_intent"),
    }


# ---------------------------------------------------------------------------
# Node – extract_appointment_details
#   Uses the LLM to extract the three appointment attributes from the call
#   transcript:
#     • appointment_date   (str | None) – e.g. "2024-08-15" or a descriptive
#                                         phrase when no ISO date is spoken
#     • doctor_name        (str | None) – full name of the doctor mentioned
#     • specialty_name     (str | None) – medical specialty (e.g. "cardiology")
#
#   Stores them under the key ``appointment_details`` in state.
#   This node is a no-op (returns None for all three fields) when
#   ``is_booking_intent`` is False.
# ---------------------------------------------------------------------------


async def extract_appointment_details(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    LLM-based extraction node for appointment attributes.

    Reads the call transcript and extracts:
      - ``appointment_date``  (str | None)
      - ``doctor_name``       (str | None)
      - ``specialty_name``    (str | None)

    These are stored together in ``state["appointment_details"]``.
    When ``is_booking_intent`` is False the node skips the LLM call and
    stores ``None`` for every field.
    """
    if not state.get("is_booking_intent", False):
        logger.info(
            "extract_appointment_details | is_booking_intent=False — skipping"
        )
        return {
            "appointment_details": {
                "appointment_date": None,
                "doctor_name": None,
                "specialty_name": None,
            },
            "node_trace": _trace(state, "extract_appointment_details"),
        }

    call = state.get("call")
    if call is None:
        logger.warning(
            "extract_appointment_details | call is None — skipping"
        )
        return {
            "appointment_details": {
                "appointment_date": None,
                "doctor_name": None,
                "specialty_name": None,
            },
            "node_trace": _trace(state, "extract_appointment_details"),
        }

    try:
        transcript_text = " ".join(
            (turn if isinstance(turn, str) else turn.text)
            for turn in (call.transcript or [])
            if (turn if isinstance(turn, str) else getattr(turn, "text", None))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "extract_appointment_details | failed to read transcript | %s", exc
        )
        return {
            "appointment_details": {
                "appointment_date": None,
                "doctor_name": None,
                "specialty_name": None,
            },
            "node_trace": _trace(state, "extract_appointment_details"),
        }

    user_prompt = APPOINTMENT_EXTRACTION_PROMPT.format(transcript=call.transcript,date=call.call_date)

    data, err = await _focused_llm_call(
        "extract_appointment_details",
        call.call_id,
        user_prompt,
        llm_client,
        state,
    )
    if err:
        return err
    print(data)
    # Normalise keys to lowercase so we tolerate Patient_name / patient_name
    # variants returned by different LLMs.
    data_lower = {k.lower(): v for k, v in data.items()}
    details = {
        "appointment_date": data_lower.get("appointment_date"),
        "doctor_name":      data_lower.get("doctor_name"),
        "specialty_name":   data_lower.get("specialty_name"),
        "patient_name":     data_lower.get("patient_name"),
        "offer_name":       data_lower.get("offer_name"),
    }

    logger.info(
        "extract_appointment_details | call_id=%s date=%s doctor=%s specialty=%s patient=%s offer=%s",
        call.call_id,
        details["appointment_date"],
        details["doctor_name"],
        details["specialty_name"],
        details["patient_name"],
        details["offer_name"],
    )

    return {
        "appointment_details": details,
        "node_trace": _trace(state, "extract_appointment_details"),
    }


# ---------------------------------------------------------------------------
# Node – verify_appointment_in_db
#   Takes the three appointment attributes produced by
#   ``extract_appointment_details`` and queries the database to check whether
#   a matching reservation record exists.
#
#   Stores the lookup outcome in ``state["appointment_verification"]``:
#     • found      (bool)            – True when a matching record exists
#     • record     (dict | None)     – the raw DB row as a dict, or None
#     • message    (str)             – human-readable summary
# ---------------------------------------------------------------------------


async def verify_appointment_in_db(state: AgentState) -> dict:
    """
    Database-lookup node for appointment verification.

    Uses ``appointment_details`` from state to search for a reservation that
    matches ALL three criteria (doctor name, specialty, date).  Any missing
    attribute is treated as a wildcard (not applied as a filter).

    The node expects a ``db_session`` key in state that holds an async
    SQLAlchemy ``AsyncSession`` (or any object with an ``execute`` coroutine
    compatible with SQLAlchemy's async API).  When no session is present the
    node logs a warning and returns ``found=False``.

    Stores result under ``state["appointment_verification"]``.
    """
    details: dict = state.get("appointment_details") or {}
    appointment_date: Optional[str] = details.get("appointment_date")
    doctor_name: Optional[str] = details.get("doctor_name")
    specialty_name: Optional[str] = details.get("specialty_name")
    patient_name: Optional[str] = details.get("patient_name")
    patient_phone: Optional[str] = state.get("call").Patient_Phone

    call = state.get("call")
    call_id = call.call_id if call else "UNKNOWN"

    # Guard: nothing to look up when all three attributes are absent
    if not any([appointment_date, doctor_name, specialty_name, patient_name]):
        logger.info(
            "verify_appointment_in_db | call_id=%s — no attributes to filter on",
            call_id,
        )
        return {
            "appointment_verification": {
                "found": False,
                "record": None,
                "message": "No appointment attributes were extracted — cannot verify.",
            },
            "node_trace": _trace(state, "verify_appointment_in_db"),
        }

    try:
        app_dir = Path(__file__).resolve().parent.parent
        passcodes_path = app_dir / "Passcode.json"
        sql_path = app_dir / "SQL" / "Slots.sql"

        if not passcodes_path.exists():
            raise FileNotFoundError(f"Passcode file not found: {passcodes_path}")

        business_unit: str = (state.get("call").business_unit or "LIVE")
        print("business_unit",business_unit)
        with passcodes_path.open("r", encoding="utf-8") as handle:
            db_config = json.load(handle)["DB_NAMES"]
            if business_unit not in db_config:
                logger.warning(
                    "verify_appointment_in_db | call_id=%s — unknown business_unit=%s, "
                    "falling back to LIVE",
                    call_id, business_unit,
                )
                business_unit = "LIVE"
            passcodes = db_config[business_unit]

        server = passcodes["Server"]
        db_name = passcodes["Database"]
        uid = passcodes["UID"]
        pwd = passcodes["PWD"]
        driver = passcodes["driver"]

        params = urllib.parse.quote_plus(
            f"DRIVER={driver};"
            f"SERVER={server};"
            f"DATABASE={db_name};"
            f"UID={uid};"
            f"PWD={pwd};"
            f"Connection Timeout=300;"
        )
        engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
        logger.debug(
            "verify_appointment_in_db | call_id=%s — using DB engine for %s/%s",
            call_id,
            server,
            db_name,
        )

        query_text = (sql_path.read_text(encoding="utf-8"))
        query = sa_text(query_text)

        # Build fuzzy-match LIKE patterns that tolerate Arabic spelling
        # variants (alef forms, teh-marbuta, yeh forms, diacritics).
        doctor_pattern   = _arabic_like_pattern(doctor_name,  first_word_only=True)
        patient_pattern  = _arabic_like_pattern(patient_name, first_word_only=False)
        specialty_norm   = _normalize_arabic(specialty_name)

        bind_params: dict[str, object] = {
            "ReportDate":   appointment_date or None,
            "Specialty":    specialty_norm,
            "Doctor":       (doctor_pattern.strip().split()[0]) if doctor_pattern else None,
            "PatientPhone": ("0" + patient_phone) if patient_phone else None,
            "PatientName":  patient_pattern,
        }
        print("params",bind_params)

        logger.debug(
            "verify_appointment_in_db | call_id=%s bind_params=%s",
            call_id, bind_params,
        )

        with engine.connect() as connection:
            result = connection.execute(query, bind_params)
            row = result.mappings().first()

        if row:
            record = dict(row)
            logger.info(
                "verify_appointment_in_db | call_id=%s — reservation FOUND | record_id=%s",
                call_id,
                record.get("PatientArName", "N/A"),
            )
            return {
                "appointment_verification": {
                    "found": True,
                    "record": record,
                    "message": "Reservation found in the database.",
                },
                "node_trace": _trace(state, "verify_appointment_in_db"),
            }
        else:
            logger.info(
                "verify_appointment_in_db | call_id=%s — reservation NOT FOUND "
                "(date=%s, doctor=%s, specialty=%s)",
                call_id,
                appointment_date,
                doctor_name,
                specialty_name,
            )
            return {
                "appointment_verification": {
                    "found": False,
                    "record": None,
                    "message": (
                        f"No reservation found for doctor='{doctor_name}', "
                        f"specialty='{specialty_name}', date='{appointment_date}'."
                    ),
                },
                "node_trace": _trace(state, "verify_appointment_in_db"),
            }

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "verify_appointment_in_db | call_id=%s — DB query failed | %s",
            call_id,
            exc,
        )
        return {
            "appointment_verification": {
                "found": False,
                "record": None,
                "message": f"Database error during verification: {exc}",
            },
            "node_trace": _trace(state, "verify_appointment_in_db"),
        }


# ---------------------------------------------------------------------------
# Node – validate_bank_information (app.bank_node)
#   Fully independent of validate_location — separate graph node, separate
#   CRM fetch/cache (app.bank_node.crm_bank), separate state key
#   (bank_validation). The only thing the two share is the transcript-turn
#   splitter in app.services.text_helpers.
# ---------------------------------------------------------------------------
def _not_applicable_bank_result(resolved_bu: str | None) -> dict:
    """Shared NOT_APPLICABLE shape for when bank validation does not apply.
    Used by BOTH the graph-level skip path (see skip_bank_validation below —
    no bank intent at all, so validate_bank_information_node is never
    reached) and validate_bank_information_node's own internal gate, kept
    as a defensive fallback in case the node is ever reached without the
    graph router having confirmed intent first."""
    return {
        "outcome": "NOT_APPLICABLE", "applicable": False,
        "request_detected": False, "requested_business_unit": resolved_bu,
        "provided_identifiers": [], "is_violation": False,
        "reason": "No bank request within bank-validation scope (AFW/AHJ/AKW/ALW) was detected.",
    }


def skip_bank_validation(state: AgentState) -> dict:
    """Graph-level skip path, taken by app.agent.graph's bank-intent router
    (_bank_intent_router) when there is no bank intent at all — reusing the
    exact same gate validate_bank_information_node itself uses
    (bank_validation_needed / detect_bank_signals), so the decision is never
    duplicated. validate_bank_information is not on this path: no CRM
    fetch, no business-unit bank resolution beyond the cheap keyword signal
    already used for routing, and — deliberately — no node_trace entry for
    'validate_bank_information', since the node never actually ran.
    """
    call = state["call"]
    signals = detect_bank_signals(call)
    logger.info("bank validation skipped | call_id=%s reason=no_bank_intent", call.call_id)
    return {"bank_validation": _not_applicable_bank_result(signals[4])}


async def validate_bank_information_node(state: AgentState) -> dict:
    """Run deterministic KSA-only bank validation.

    Under normal graph execution this node is only reached at all when
    app.agent.graph's bank-intent router (_bank_intent_router) has already
    confirmed bank_validation_needed — see
    app.bank_node.bank_validation.bank_validation_needed /
    detect_bank_signals, which the router reuses directly rather than
    duplicating. The gate below is kept as a defensive fallback (not the
    primary mechanism any more): it re-checks the same condition so that
    even if this node were ever reached without going through the router,
    it would still degrade to a safe, non-punitive NOT_APPLICABLE instead
    of running CRM lookups it has no real signal to justify.

    Business-Unit resolution (app.bank_node.bank_validation.resolve_business_unit)
    is keyword-map-driven, not CRM-location-based — this node no longer
    depends on app.location_node's CRM fetch at all, unlike before.
    """
    call = state["call"]
    signals = detect_bank_signals(call)
    # Avoid CRM access for conversations with no supported BU / no bank
    # request / no agent-provided financial identifier; keeps unrelated
    # flows (and out-of-scope BUs) untouched.
    if not bank_validation_needed(call, signals):
        result = _not_applicable_bank_result(signals[4])
    else:
        try:
            from app.bank_node.crm_bank import fetch_bank_accounts
            banks = fetch_bank_accounts()
            # Missing reference data is non-punitive: correctness cannot be
            # verified safely, so report an unresolved system-data condition.
            if not banks:
                result = {
                    "outcome": "BUSINESS_UNIT_UNRESOLVED", "applicable": False,
                    "request_detected": True, "requested_business_unit": signals[4],
                    "provided_identifiers": [], "is_violation": False,
                    "reason": "Authoritative CRM bank data is unavailable; no agent violation was inferred.",
                }
            else:
                result = validate_ksa_bank_information(call, banks, signals)
        except Exception as exc:  # Reference-data failures are non-punitive.
            logger.warning("bank validation unavailable | call_id=%s | %s", call.call_id, exc)
            result = {
                "outcome": "BUSINESS_UNIT_UNRESOLVED", "applicable": False,
                "request_detected": True, "requested_business_unit": signals[4],
                "provided_identifiers": [], "is_violation": False,
                "reason": "Authoritative CRM bank data could not be read; no agent violation was inferred.",
            }
    logger.info("bank validation | call_id=%s outcome=%s bu=%s", call.call_id, result["outcome"], signals[4])
    return {"bank_validation": result, "node_trace": _trace(state, "validate_bank_information")}


def _not_applicable_location_result() -> dict:
    """Shared NOT_APPLICABLE shape for when location validation does not
    apply. Used by BOTH the graph-level skip path (see
    skip_location_validation below — no location intent at all, so
    validate_location_node is never reached) and validate_location_node's
    own internal gate, kept as a defensive fallback in case the node is
    ever reached without the graph router having confirmed intent first."""
    return {
        "outcome": "NOT_APPLICABLE", "applicable": False,
        "request_detected": False, "requested_branch": None,
        "crm_location": None, "provided_location": None,
        "match_confidence": None, "is_violation": False,
        "reason": "No location/address request or agent-provided address was detected.",
    }


def skip_location_validation(state: AgentState) -> dict:
    """Graph-level skip path, taken by app.agent.graph's location-intent
    router (_location_intent_router) when there is no location intent at
    all — reusing the exact same gate validate_location_node itself uses
    (location_validation_needed / detect_location_intent), so the decision
    is never duplicated. validate_location is not on this path: no CRM
    fetch, no branch resolution, and — deliberately — no node_trace entry
    for 'validate_location', since the node never actually ran.
    """
    call = state["call"]
    _requested, patient_text, agent_text = detect_location_signals(call)
    reason = "home_service_location" if is_home_service_location_text(patient_text, agent_text) else "no_location_intent"
    logger.info("location validation skipped | call_id=%s reason=%s", call.call_id, reason)
    print(f"[location] skipped | call_id={call.call_id} reason={reason}", flush=True)
    return {"location_validation": _not_applicable_location_result()}


# ---------------------------------------------------------------------------
# Node – validate_location (app.location_node)
#   Fully independent of validate_bank_information — see above.
# ---------------------------------------------------------------------------
async def validate_location_node(state: AgentState) -> dict:
    """Run deterministic KSA-only branch/location validation.

    Under normal graph execution this node is only reached at all when
    app.agent.graph's location-intent router (_location_intent_router) has
    already confirmed patient_has_location_intent OR
    agent_has_location_information — see
    location_node.location_validation.detect_location_intent /
    location_validation_needed, which the router reuses directly rather
    than duplicating. The gate below is kept as a defensive fallback (not
    the primary mechanism any more): it re-checks the same condition so
    that even if this node were ever reached without going through the
    router, it would still degrade to a safe, non-punitive NOT_APPLICABLE
    instead of running CRM lookups it has no real signal to justify.
    """
    call = state["call"]
    signals = detect_location_signals(call)
    requested, patient_text, agent_text = signals
    patient_intent, agent_intent = detect_location_intent(patient_text, agent_text)
    logger.info(
        "location intent | call_id=%s patient_request=%s agent_provided=%s",
        call.call_id, patient_intent, agent_intent,
    )
    if not location_validation_needed(call, signals):
        logger.info("location validation skipped | call_id=%s reason=no_location_intent", call.call_id)
        result = _not_applicable_location_result()
    else:
        try:
            from app.location_node.crm_location import fetch_ksa_locations
            locations = fetch_ksa_locations()
            if not locations:
                logger.warning("[location_validation] CRM location lookup failed | call_id=%s reason=empty_result", call.call_id)
                result = {
                    "outcome": "BRANCH_UNRESOLVED", "applicable": False,
                    "request_detected": requested, "requested_branch": None,
                    "crm_location": None, "provided_location": None,
                    "match_confidence": None, "is_violation": False,
                    "reason": "Authoritative CRM location data is unavailable; no agent violation was inferred.",
                }
            else:
                result = validate_location_request(
                    call, locations, patient_text=patient_text, agent_text=agent_text,
                )
        except Exception as exc:  # Reference-data failures are non-punitive.
            logger.warning("[location_validation] CRM location lookup failed | call_id=%s | %s", call.call_id, exc)
            result = {
                "outcome": "BRANCH_UNRESOLVED", "applicable": False,
                "request_detected": requested, "requested_branch": None,
                "crm_location": None, "provided_location": None,
                "match_confidence": None, "is_violation": False,
                "reason": "Authoritative CRM location data could not be read; no agent violation was inferred.",
            }
        logger.info(
            "location validation | call_id=%s outcome=%s branch=%s",
            call.call_id, result["outcome"], result.get("requested_branch"),
        )
    return {"location_validation": result, "node_trace": _trace(state, "validate_location")}
