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
    build_service_prompt,
    build_package_prompt,
    build_script_prompt,
    build_scoring_prompt,
    build_user_prompt,   # kept for legacy path
)
from app.services.criteria_loader import CriteriaLoader
from app.services.llm_client import LLMClient
from app.services.sql_helpers import insert_qa_result
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


def _eligibility_evaluation_context(eligibility_result: Optional[dict]) -> str:
    """Return an authoritative eligibility fact only when the API result is conclusive."""
    if not eligibility_result or eligibility_result.get("api_status") != "Success":
        return "No conclusive eligibility result is available."

    outcome = "ELIGIBLE" if eligibility_result.get("is_eligible") else "NOT ELIGIBLE"
    return (
        "Authoritative eligibility API result: the patient is "
        f"{outcome}. This is the reference outcome for this call."
    )


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


def _escape_json_string_controls(payload: str) -> str:
    """Escape raw control characters inside JSON strings emitted by an LLM."""
    repaired: list[str] = []
    in_string = False
    escaped = False
    for char in payload:
        if in_string:
            if escaped:
                repaired.append(char)
                escaped = False
            elif char == "\\":
                repaired.append(char)
                escaped = True
            elif char == '"':
                repaired.append(char)
                in_string = False
            elif char == "\n":
                repaired.append("\\n")
            elif char == "\r":
                repaired.append("\\r")
            elif char == "\t":
                repaired.append("\\t")
            else:
                repaired.append(char)
        else:
            repaired.append(char)
            if char == '"':
                in_string = True
    return "".join(repaired)


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
        # Models occasionally emit a raw newline inside a JSON string. Escape
        # only those invalid control characters before failing the QA run.
        try:
            data = json.loads(_escape_json_string_controls(clean))
            if not isinstance(data, dict):
                raise ValueError("repaired response is not a JSON object")
            logger.warning(
                "%s JSON repaired | call_id=%s original_error=%s",
                node_name, call_id, exc,
            )
        except (json.JSONDecodeError, ValueError) as repair_exc:
            logger.error(
                "%s JSON parse error | call_id=%s snippet=%s | %s",
                node_name, call_id, raw_text[:300], exc,
            )
            return None, {
                "error": f"{node_name}: LLM returned invalid JSON: {exc}; repair failed: {repair_exc}",
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
        eligibility_context=_eligibility_evaluation_context(
            state.get("eligibility_result")
        ),
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
        eligibility_result=state.get("eligibility_result"),
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


def _reservation_transcript_excerpt(state: AgentState) -> str:
    """Return a short verbatim agent turn to accompany a reservation finding."""
    call = state.get("call")
    transcript = getattr(call, "transcript", None)
    turns = [transcript] if isinstance(transcript, str) else (transcript or [])
    for turn in turns:
        text = turn if isinstance(turn, str) else getattr(turn, "text", "")
        for line in str(text).splitlines():
            if "agent:" in line.lower() and any(keyword in line for keyword in _BOOKING_KEYWORDS):
                return line.strip()
    return ""


async def enforce_ineligible_reservation_violation(state: AgentState) -> dict:
    """Ensure an ineligible patient with a persisted reservation gets C2C_005."""
    eligibility = state.get("eligibility_result") or {}
    verification = state.get("appointment_verification") or {}
    is_conclusively_ineligible = (
        eligibility.get("http_status") == 200
        and eligibility.get("is_eligible") is False
        and eligibility.get("api_status") in {"Fail", "Success"}
    )
    if not (is_conclusively_ineligible and verification.get("found") is True):
        return {"node_trace": _trace(state, "enforce_ineligible_reservation_violation")}

    reservation_eval = dict(state.get("reservation_eval") or {})
    flags = list(reservation_eval.get("reservation_flags", []))
    if not any("C2C_005" in flag.get("description", "") for flag in flags):
        flags.append(
            {
                "type": "C2C",
                "severity": "critical",
                "description": (
                    "Submitting the right action on the system / C2C_005: "
                    "Made wrong reservation for a patient whose eligibility check failed."
                ),
                "transcript_excerpt": _reservation_transcript_excerpt(state),
            }
        )
        reservation_eval["reservation_flags"] = flags
        logger.warning(
            "enforce_ineligible_reservation_violation | call_id=%s persisted reservation found for ineligible patient",
            state["call"].call_id,
        )

    return {
        "reservation_eval": reservation_eval,
        "node_trace": _trace(state, "enforce_ineligible_reservation_violation"),
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
    # Only use the specialty extracted from the transcript by the LLM.
    # call.department holds operational dept names ("Scheduling", "Helpdesk",
    # etc.) which are NOT medical specialties and must never be used here.
    details: dict = state.get("appointment_details") or {}
    specialty_en: str = (details.get("specialty_name") or "").strip()
    
    # Check if offer name is directly mentioned in the transcript
    offer_name_hint: str = (details.get("offer_name") or "").strip()
    
    # If no specialty but offer name is mentioned, we can still proceed
    # The enhanced fast-path matching will handle the direct offer name lookup
    if not specialty_en and not offer_name_hint:
        logger.info(
            "fetch_crm_offers_for_call | call_id=%s — no specialty or offer name extracted from transcript, skipping CRM fetch",
            call_id,
        )
        return {
            "crm_offers_context": "",
            "node_trace": _trace(state, "fetch_crm_offers_for_call"),
        }
    
    if not specialty_en and offer_name_hint:
        logger.info(
            "fetch_crm_offers_for_call | call_id=%s — no specialty extracted but offer name mentioned: %r, proceeding with direct name lookup",
            call_id, offer_name_hint,
        )

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
        from app.service_hub.crm_offers import get_offers_for_specialty, format_offer_card

        # When offer name is mentioned but specialty is missing, use a generic
        # specialty value that won't filter results. The direct name lookup
        # (fast-path) will find the offer regardless of specialty.
        effective_specialty = specialty_en or "General"
        
        offers = get_offers_for_specialty(
            specialty_en=effective_specialty,
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
        search_info = f"offer_name={offer_name_hint!r}" if offer_name_hint and not specialty_en else f"specialty={specialty_en!r}"
        logger.info(
            "fetch_crm_offers_for_call | call_id=%s — no active offers found for %s",
            call_id, search_info,
        )
        return {
            "crm_offers_context": f"No active CRM offers found for {search_info}",
            "node_trace": _trace(state, "fetch_crm_offers_for_call"),
        }

    # ── Serialise offers as compact human-readable cards ─────────────────
    # Use the same format_offer_card formatter the chatbot uses so the LLM
    # sees the same representation the agent would have seen.
    search_context = f"offer_name={offer_name_hint!r}" if offer_name_hint and not specialty_en else f"specialty: {specialty_en!r}"
    lines: list[str] = [
        f"Active CRM offers for {search_context}  "
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
# ── Helper: Extract service name from transcript ──────────────────────────

def _extract_service_name_from_transcript(transcript: str) -> str:
    """
    Extract ALL service names from transcript using medical/lab stopping words.
    Returns a newline-separated string of all extracted service names, or empty string.
    
    Examples:
        "عاوز أعمل فحص سونار" → "سونار"
        "محتاج تحليل صورة دم كاملة" → "صورة دم كاملة"
        "MRI BRAIN\nCT BRAIN" → "MRI BRAIN\nCT BRAIN"
    """
    if not transcript:
        return ""
    
    import re
    
    # Strategy 1: Extract every uppercase English service line independently.
    # Service lists are commonly one item per line followed by a price. Do not
    # use ``\s`` across the entire transcript because it consumes newlines and
    # merges multiple services into one candidate.
    _MEDICAL_KEYWORDS = {
        'MRI', 'CT', 'SCAN', 'X-RAY', 'X RAY', 'XRAY', 'U/S', 'ULTRASOUND', 'ECHO',
        'CBC', 'ESR', 'CRP', 'TSH', 'ALT', 'AST', 'ECG', 'EKG',
        'BRAIN', 'CHEST', 'ABDOMEN', 'PELVIS', 'SPINE', 'HEAD', 'NECK',
        'CONTRAST', 'WITHOUT', 'WITH', 'COMPLETE', 'BLOOD', 'COUNT',
        'BONE', 'WINDOW', 'SOFT', 'TISSUE', 'TEST', 'SCREENING',
        'CREATININE', 'UREA', 'NITROGEN', 'ANTIGEN', 'TUMOR', 'MARKER',
        'SERUM', 'SGPT', 'SGOT', 'DENSITOMETRY', 'SCOLIOSIS', 'KYPHOSIS',
    }
    _PRICE_SUFFIX = re.compile(
        r'\s+\d+(?:\.\d+)?\s*(?:sar|sr|ريال(?:\s+سعودي)?)\b.*$',
        re.IGNORECASE,
    )
    _SPEAKER_PREFIX = re.compile(r'^(?:agent|patient)\s*:\s*', re.IGNORECASE)

    # Prefer the agent's explicitly quoted service names. Patient phrasing can
    # be descriptive and may otherwise produce an unrelated fuzzy CRM match.
    agent_services: list[str] = []
    other_services: list[str] = []
    current_speaker = ''
    for raw_line in transcript.splitlines():
        # Transcript exports prefix only the first line of a multi-line turn.
        # Keep that speaker until a later explicit Agent:/Patient: prefix.
        speaker_match = _SPEAKER_PREFIX.match(raw_line)
        if speaker_match:
            current_speaker = raw_line.split(':', 1)[0].strip().lower()
        is_agent_line = current_speaker == 'agent'
        candidate = _SPEAKER_PREFIX.sub('', raw_line).strip()
        candidate = _PRICE_SUFFIX.sub('', candidate).strip(' -:')
        candidate_upper = candidate.upper()
        # Do not turn lowercase explanatory text into a service by uppercasing
        # it first. The dedicated lowercase fallback below handles true
        # mixed-case service requests when no technical service line is found.
        if not candidate or not re.search(r'[A-Z]', candidate):
            continue
        if not any(keyword in candidate_upper for keyword in _MEDICAL_KEYWORDS):
            continue
        services = agent_services if is_agent_line else other_services
        if candidate not in services:
            services.append(candidate)

    matched_services = agent_services or other_services
    if matched_services:
        logger.info(
            "_extract_service_name_from_transcript | extracted %d English service name(s): %s",
            len(matched_services),
            matched_services,
        )
        return "\n".join(matched_services)

    # Strategy 1b: Extract lowercase/mixed-case English medical phrases
    # Pattern: 2-4 English words that include medical terms (case-insensitive)
    # Examples: "anomaly scan", "glucose test", "blood pressure"
    _LOWERCASE_MEDICAL_KEYWORDS = [
        'scan', 'test', 'screening', 'check', 'examination',
        'ultrasound', 'echo', 'doppler', 'x-ray', 'xray',
        'anomaly', 'anatomy', 'glucose', 'blood', 'pressure',
        'thyroid', 'diabetes', 'cholesterol', 'liver', 'kidney',
        'prenatal', 'pregnancy', 'fetal', 'obstetric', 'cardiac'
    ]
    
    # Find phrases with 1-4 consecutive English words that contain a medical keyword
    lowercase_phrases = re.findall(
        r'\b([a-z]+(?:\s+[a-z]+){0,3})\b',
        transcript.lower()
    )
    
    for phrase in lowercase_phrases:
        # Check if phrase contains any medical keyword
        if any(keyword in phrase for keyword in _LOWERCASE_MEDICAL_KEYWORDS):
            # Validate: must be 2+ words OR a known single-word service
            words = phrase.strip().split()
            if len(words) >= 2 or phrase in {'ultrasound', 'echo', 'doppler', 'xray'}:
                logger.info(f"_extract_service_name_from_transcript | extracted lowercase English phrase: {phrase!r}")
                return phrase
    
    # Strategy 1c: Extract English medical abbreviations only (fallback)
    english_abbrev = re.findall(
        r'\b(?:U/S|CT|MRI|CBC|ESR|CRP|HbA1c|TSH|T3|T4|ALT|AST|ECG|EKG|X-RAY|XRAY)\b'
        r'(?:\s+[2-4]D)?',  # Optional dimension suffix (2D, 3D, 4D)
        transcript, 
        re.IGNORECASE
    )
    if english_abbrev:
        extracted = english_abbrev[0].strip()
        logger.info(f"_extract_service_name_from_transcript | extracted English abbrev: {extracted!r}")
        return extracted
    
    # Strategy 2: Extract complete Arabic medical service phrases
    # Match medical terms with their full descriptions (up to 10 words for complex services)
    _MEDICAL_PHRASE_PATTERNS = [
        # Ultrasound variations (موجات فوق الصوتية + dimensions)
        r'(?:ال)?موجات\s+فوق\s+الصوتية(?:\s+(?:ثنائي|ثلاثي|رباعي|رباعية)\s+الأبعاد)?(?:\s+\w+){0,3}',
        r'(?:ال)?سونار(?:\s+(?:ثنائي|ثلاثي|رباعي|رباعية)\s+الأبعاد)?(?:\s+\w+){0,3}',
        r'(?:ال)?إيكو(?:\s+\w+){0,3}',
        r'(?:ال)?ايكو(?:\s+\w+){0,3}',
        r'(?:ال)?دوبلر(?:\s+\w+){0,3}',
        
        # Radiology (أشعة + type)
        r'(?:ال)?أشعة(?:\s+مقطعية)?(?:\s+\w+){0,3}',
        r'(?:ال)?اشعة(?:\s+مقطعية)?(?:\s+\w+){0,3}',
        r'(?:ال)?رنين\s+(?:مغناطيسي)?(?:\s+\w+){0,3}',
        r'(?:ال)?تصوير(?:\s+\w+){0,4}',
        
        # Lab tests
        r'(?:صورة|تحليل)\s+دم(?:\s+(?:كاملة|شاملة))?',
        r'(?:تحليل|فحص)\s+(?:السكر|سكر)(?:\s+تراكمي)?',
        r'(?:تحليل|فحص)\s+(?:الكوليسترول|كوليسترول)',
        r'(?:تحليل|فحص)\s+(?:الغدة\s+الدرقية|غدة\s+درقية)',
        r'(?:تحليل|فحص)\s+(?:وظائف\s+(?:كلى|كبد))',
        r'(?:تحليل|فحص)\s+(?:فيتامين\s+\w+)',
    ]
    
    for pattern in _MEDICAL_PHRASE_PATTERNS:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            extracted = match.group(0).strip()
            logger.info(f"_extract_service_name_from_transcript | extracted medical phrase: {extracted!r}")
            return extracted
    
    # Strategy 3: Generic extraction using trigger keywords (fallback)
    _SERVICE_TRIGGERS = [
        r'(?:أريد|اريد|عايز|عاوز|محتاج|ممكن)\s+(?:عمل|اعمل|أعمل)?\s*',
        r'(?:فحص|فحوص|فحوصات)\s+',
        r'(?:تحليل|تحاليل)\s+',
        r'(?:أشعة|اشعة|الأشعة|الاشعة)\s+',
        r'(?:تصوير)\s+',
        r'(?:صورة|صور)\s+',
    ]
    
    trigger_pattern = '|'.join(_SERVICE_TRIGGERS)
    # Capture up to 10 words after trigger for complex service names
    # Stop at: punctuation, questions, branch mentions, or price indicators
    full_pattern = rf"(?:{trigger_pattern})((?:(?![؟،\n\.!]|هل |في |عندكم|فرع|بكام|كام|سعر|ريال)[^\s]{{2,}}\s*){{1,10}})"
    
    match = re.search(full_pattern, transcript, re.IGNORECASE)
    if match:
        service_name = match.group(1).strip()
        # Clean up the extracted name
        service_name = re.sub(r'\s+', ' ', service_name)  # normalize whitespace
        logger.info(f"_extract_service_name_from_transcript | extracted generic: {service_name!r}")
        return service_name
    
    return ""


# Node G2 – fetch_crm_services_for_call
#   Looks up active services for the call's specialty from the hospital SQL
#   Server DB (cr301_ksaservicedataset, active rows only).
#   Runs in parallel with fetch_crm_offers_for_call.
#   Stores a compact human-readable context string in crm_services_context.
# ---------------------------------------------------------------------------

async def fetch_crm_services_for_call(state: AgentState) -> dict:
    """
    Fetch live service records for the call's specialty and serialise them
    as a compact string for injection into evaluation prompts.

    Falls back gracefully to "" on any error so downstream nodes are unaffected.
    """
    call = state.get("call")
    call_id = call.call_id if call else "UNKNOWN"

    # Only use the specialty extracted from the transcript by the LLM.
    # call.department is an operational label ("Scheduling", "Helpdesk", etc.),
    # not a medical specialty — never use it as a specialty fallback.
    details: dict = state.get("appointment_details") or {}
    specialty_en: str = (details.get("specialty_name") or "").strip()

    # Extract service hint: prioritize direct transcript extraction over LLM-extracted offer_name
    # This catches English service names (e.g., "anomaly scan") that the LLM might miss or misclassify
    transcript: str = getattr(call, "transcript", "") or ""
    service_hint = _extract_service_name_from_transcript(transcript)
    
    # Fallback: use LLM-extracted offer_name if transcript extraction found nothing
    if not service_hint:
        service_hint = (details.get("offer_name") or "").strip()
    
    # Skip fetching only if BOTH specialty AND service_hint are missing
    if not specialty_en and not service_hint:
        logger.info(
            "fetch_crm_services_for_call | call_id=%s — no specialty or service hint extracted, skipping",
            call_id
        )
        return {"crm_services_context": "", "node_trace": _trace(state, "fetch_crm_services_for_call")}
    
    # Use a generic specialty when missing but service_hint exists
    # The direct name matching in get_services_for_specialty will handle it
    if not specialty_en and service_hint:
        specialty_en = "General"
        logger.info(
            "fetch_crm_services_for_call | call_id=%s — using generic specialty with service_hint=%r",
            call_id, service_hint
        )
    
    business_unit: str = (getattr(call, "business_unit", None) or "LIVE")

    # Split service_hint by newlines to handle multiple service names
    # (e.g., when agent mentions multiple services in one conversation)
    service_hints = [h.strip() for h in service_hint.split('\n') if h.strip()] if service_hint else []
    
    logger.info(
        "fetch_crm_services_for_call | call_id=%s specialty=%r bu=%s hints=%d: %s",
        call_id, specialty_en, business_unit, len(service_hints), service_hints or "(none)",
    )

    try:
        from app.service_hub.crm_services import get_services_for_specialty, format_service_card

        # Fetch services for each hint and aggregate results
        all_services = []
        seen_service_keys = set()  # Track by servicekey to deduplicate
        
        if service_hints:
            # Multiple hints: fetch services for each one
            for idx, hint in enumerate(service_hints, 1):
                services = get_services_for_specialty(
                    specialty_en=specialty_en,
                    service_hint=hint,
                    bu=business_unit,
                    top_n=3,  # Limit to top 3 per hint to avoid explosion
                )
                logger.info(
                    "fetch_crm_services_for_call | call_id=%s hint#%d (%r) → %d service(s)",
                    call_id, idx, hint, len(services)
                )
                # Deduplicate by servicekey
                for svc in services:
                    svc_key = svc.get("cr301_servicekey")
                    if svc_key and svc_key not in seen_service_keys:
                        seen_service_keys.add(svc_key)
                        all_services.append(svc)
        else:
            # No hints: fallback to specialty-only search
            all_services = get_services_for_specialty(
                specialty_en=specialty_en,
                service_hint="",
                bu=business_unit,
            )
        
        services = all_services
    except Exception as exc:
        logger.warning(
            "fetch_crm_services_for_call | call_id=%s — fetch failed (non-fatal): %s", call_id, exc
        )
        return {"crm_services_context": "", "node_trace": _trace(state, "fetch_crm_services_for_call")}

    if not services:
        logger.info(
            "fetch_crm_services_for_call | call_id=%s — no services for specialty=%r", call_id, specialty_en
        )
        return {
            "crm_services_context": f"No services found for specialty: {specialty_en!r}",
            "node_trace": _trace(state, "fetch_crm_services_for_call"),
        }

    # ── Log detailed service information (same format as offers) ──────────────
    for i, svc in enumerate(services, 1):
        is_primary = not svc.get("_is_alternative", False)
        flag = "[PRIMARY]" if is_primary else "[ALTERNATIVE]"
        
        # Collect all available fields for comprehensive logging
        log_fields = [
            ("cr301_title", svc.get("cr301_title")),
            ("cr301_service", svc.get("cr301_service")),
            ("cr301_servicear", svc.get("cr301_servicear")),
            ("cr301_specialtyname", svc.get("cr301_specialtyname")),
            ("cr301_price", svc.get("servhub_priced")),
            ("cr301_code", svc.get("cr301_code")),
            ("cr301_servicekey", svc.get("cr301_servicekey")),
            ("cr301_nphiescode", svc.get("cr301_nphiescode")),
            ("cr18c_buname", svc.get("cr18c_buname")),
            ("_service_matched", svc.get("_service_matched", False)),
        ]
        
        # Build log message with NULL warnings
        log_lines = [f"fetch_crm_services_for_call | call_id={call_id} — retrieved service #{i} {flag}:"]
        null_fields = []
        for field_name, field_value in log_fields:
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                null_fields.append(field_name)
                log_lines.append(f"  {field_name:<20}: NULL ⚠️")
            else:
                log_lines.append(f"  {field_name:<20}: {field_value}")
        
        if null_fields:
            log_lines.append(f"  ⚠️  WARNING: {len(null_fields)} NULL field(s): {', '.join(null_fields)}")
        
        logger.info("\n".join(log_lines))

    hint_summary = f" | hints: {', '.join(repr(h) for h in service_hints)}" if service_hints else ""
    lines: list[str] = [
        f"Active services for specialty: {specialty_en!r}  (bu={business_unit}){hint_summary}",
        f"Total: {len(services)} service(s) returned",
        "",
    ]
    for i, svc in enumerate(services, 1):
        flag = " [PRIMARY]" if not svc.get("_is_alternative") else " [ALTERNATIVE]"
        card = format_service_card(svc, lang="ar")
        lines.append(f"--- Service {i}{flag} ---")
        lines.append(card)
        lines.append("")

    context = "\n".join(lines)
    matched_services = [
        {
            "arabic_name": str(svc.get("cr301_servicear") or svc.get("cr301_service") or "N/A"),
            "english_name": str(svc.get("cr301_title") or "N/A"),
            "code": str(svc.get("cr301_code") or "N/A"),
            "price": str(svc.get("servhub_priced") or svc.get("cr301_price") or "N/A"),
        }
        for svc in services
        if svc.get("_service_matched") is True
    ]
    logger.info(
        "fetch_crm_services_for_call | call_id=%s — serialised %d service(s) from %d hint(s) (%d chars)",
        call_id, len(services), len(service_hints), len(context),
    )
    return {
        "crm_services_context": context,
        "crm_matched_services": matched_services,
        "node_trace": _trace(state, "fetch_crm_services_for_call"),
    }


# ---------------------------------------------------------------------------
# Node G3 – fetch_crm_packages_for_call
#   Looks up packages for the call's specialty from the hospital SQL Server DB
#   (cr301_ksaservicedataset where servicecategoryname = 'Package').
#   Runs in parallel with fetch_crm_offers_for_call.
#   Stores a compact human-readable context string in crm_packages_context.
# ---------------------------------------------------------------------------

async def fetch_crm_packages_for_call(state: AgentState) -> dict:
    """
    Fetch live package records for the call's specialty and serialise them
    as a compact string for injection into evaluation prompts.

    Falls back gracefully to "" on any error so downstream nodes are unaffected.
    """
    call = state.get("call")
    call_id = call.call_id if call else "UNKNOWN"

    # Only use the specialty extracted from the transcript by the LLM.
    # call.department is an operational label ("Scheduling", "Helpdesk", etc.),
    # not a medical specialty — never use it as a specialty fallback.
    details: dict = state.get("appointment_details") or {}
    specialty_en: str = (details.get("specialty_name") or "").strip()

    if not specialty_en:
        logger.info("fetch_crm_packages_for_call | call_id=%s — no specialty extracted from transcript, skipping", call_id)
        return {"crm_packages_context": "", "node_trace": _trace(state, "fetch_crm_packages_for_call")}

    service_hint: str = (details.get("offer_name") or "").strip()
    business_unit: str = (getattr(call, "business_unit", None) or "LIVE")

    logger.info(
        "fetch_crm_packages_for_call | call_id=%s specialty=%r bu=%s hint=%r",
        call_id, specialty_en, business_unit, service_hint or "(none)",
    )

    try:
        from app.service_hub.crm_packages import get_packages_for_specialty, format_package_card

        packages = get_packages_for_specialty(
            specialty_en=specialty_en,
            service_hint=service_hint,
            bu=business_unit,
        )
    except Exception as exc:
        logger.warning(
            "fetch_crm_packages_for_call | call_id=%s — fetch failed (non-fatal): %s", call_id, exc
        )
        return {"crm_packages_context": "", "node_trace": _trace(state, "fetch_crm_packages_for_call")}

    if not packages:
        logger.info(
            "fetch_crm_packages_for_call | call_id=%s — no packages for specialty=%r", call_id, specialty_en
        )
        return {
            "crm_packages_context": f"No packages found for specialty: {specialty_en!r}",
            "node_trace": _trace(state, "fetch_crm_packages_for_call"),
        }

    # ── Log detailed package information (same format as offers) ──────────────
    for i, pkg in enumerate(packages, 1):
        is_primary = not pkg.get("_is_alternative", False)
        flag = "[PRIMARY]" if is_primary else "[ALTERNATIVE]"
        
        # Collect all available fields for comprehensive logging
        log_fields = [
            ("cr301_title", pkg.get("cr301_title")),
            ("cr301_service", pkg.get("cr301_service")),
            ("cr301_servicear", pkg.get("cr301_servicear")),
            ("cr301_specialtyname", pkg.get("cr301_specialtyname")),
            ("cr301_price", pkg.get("cr301_price")),
            ("cr301_code", pkg.get("cr301_code")),
            ("cr301_servicekey", pkg.get("cr301_servicekey")),
            ("cr301_nphiescode", pkg.get("cr301_nphiescode")),
            ("cr18c_buname", pkg.get("cr18c_buname")),
            ("_service_matched", pkg.get("_service_matched", False)),
        ]
        
        # Build log message with NULL warnings
        log_lines = [f"fetch_crm_packages_for_call | call_id={call_id} — retrieved package #{i} {flag}:"]
        null_fields = []
        for field_name, field_value in log_fields:
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                null_fields.append(field_name)
                log_lines.append(f"  {field_name:<20}: NULL ⚠️")
            else:
                log_lines.append(f"  {field_name:<20}: {field_value}")
        
        if null_fields:
            log_lines.append(f"  ⚠️  WARNING: {len(null_fields)} NULL field(s): {', '.join(null_fields)}")
        
        logger.info("\n".join(log_lines))

    lines: list[str] = [
        f"Active packages for specialty: {specialty_en!r}  (bu={business_unit})",
        f"Total: {len(packages)} package(s) returned",
        "",
    ]
    for i, pkg in enumerate(packages, 1):
        flag = " [PRIMARY]" if not pkg.get("_is_alternative") else " [ALTERNATIVE]"
        card = format_package_card(pkg, lang="ar")
        lines.append(f"--- Package {i}{flag} ---")
        lines.append(card)
        lines.append("")

    context = "\n".join(lines)
    logger.info(
        "fetch_crm_packages_for_call | call_id=%s — serialised %d package(s) (%d chars)",
        call_id, len(packages), len(context),
    )
    return {
        "crm_packages_context": context,
        "node_trace": _trace(state, "fetch_crm_packages_for_call"),
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


def _format_matched_services_summary(matched_services: list[dict[str, str]]) -> str:
    """Build a complete audit summary from CRM records, outside the LLM JSON."""
    details = [
        "{arabic_name} / {english_name} (Code: {code}, Price: {price} SAR)".format(**service)
        for service in matched_services
    ]
    return "Agent accurately provided the details and price for the following CRM-matched services: " + "; ".join(details)


# ---------------------------------------------------------------------------
# Node H – infer_service_evaluation
#   Focused LLM call: did the agent correctly recommend services available
#   for the patient's specialty?
#   Runs AFTER fetch_crm_services_for_call (sequential dependency).
# ---------------------------------------------------------------------------

async def infer_service_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a service-evaluation-only prompt.

    Evaluates the agent's service recommendation behaviour against possible
    outcomes:
      • SUITABLE_SERVICE_RECOMMENDED    — positive flag
      • SERVICE_SKIPPED                 — C2B moderate flag
      • UNRELATED_SERVICE_RECOMMENDED   — C2B moderate flag
      • SERVICE_MISREPRESENTED          — C2B moderate flag
      • INCOMPLETE_SERVICE_PRESENTATION — NC minor flag
      • NO_SERVICE_AVAILABLE            — no flag
      • SERVICE_NOT_APPLICABLE          — no flag

    Result stored in state["service_eval"].
    """
    call = state["call"]

    user_prompt = build_service_prompt(
        call,
        crm_services_context=state.get("crm_services_context") or "",
    )
    logger.debug(
        "infer_service_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_service_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    outcome = data.get("service_outcome", "SERVICE_NOT_APPLICABLE")
    matched_services = state.get("crm_matched_services") or []
    if outcome == "SUITABLE_SERVICE_RECOMMENDED" and matched_services:
        summary = _format_matched_services_summary(matched_services)
        data["service_reasoning"] = summary
        flags = list(data.get("service_flags") or [])
        positive_flag = next((flag for flag in flags if flag.get("type") == "positive"), None)
        if positive_flag is None:
            flags.append(
                {
                    "type": "positive",
                    "severity": "positive",
                    "description": summary,
                    "transcript_excerpt": "N/A",
                }
            )
        else:
            positive_flag["description"] = summary
        data["service_flags"] = flags

    logger.info(
        "infer_service_evaluation | call_id=%s outcome=%s flags=%d",
        call.call_id,
        outcome,
        len(data.get("service_flags", [])),
    )

    return {
        "service_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_service_evaluation"),
    }


# ---------------------------------------------------------------------------
# Node I – infer_package_evaluation
#   Focused LLM call: did the agent correctly recommend packages available
#   for the patient's specialty?
#   Runs AFTER fetch_crm_packages_for_call (sequential dependency).
# ---------------------------------------------------------------------------

async def infer_package_evaluation(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Call the LLM with a package-evaluation-only prompt.

    Evaluates the agent's package recommendation behaviour against possible
    outcomes:
      • SUITABLE_PACKAGE_RECOMMENDED    — positive flag
      • PACKAGE_SKIPPED                 — C2B moderate flag
      • UNRELATED_PACKAGE_RECOMMENDED   — C2B moderate flag
      • PACKAGE_MISREPRESENTED          — C2B moderate flag
      • INCOMPLETE_PACKAGE_PRESENTATION — NC minor flag
      • NO_PACKAGE_AVAILABLE            — no flag
      • PACKAGE_NOT_APPLICABLE          — no flag

    Result stored in state["package_eval"].
    """
    call = state["call"]

    user_prompt = build_package_prompt(
        call,
        crm_packages_context=state.get("crm_packages_context") or "",
    )
    logger.debug(
        "infer_package_evaluation | call_id=%s prompt_len=%d",
        call.call_id, len(user_prompt),
    )

    data, err = await _focused_llm_call(
        "infer_package_evaluation", call.call_id, user_prompt, llm_client, state
    )
    if err:
        return err

    outcome = data.get("package_outcome", "PACKAGE_NOT_APPLICABLE")
    logger.info(
        "infer_package_evaluation | call_id=%s outcome=%s flags=%d",
        call.call_id,
        outcome,
        len(data.get("package_flags", [])),
    )

    return {
        "package_eval": data,
        "usage_list": [data.get("_usage", {})],
        "node_trace": _trace(state, "infer_package_evaluation"),
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

    user_prompt = build_scoring_prompt(
        call,
        scoring_weights=state.get("scoring_weights", ""),
        behavioral_summary=behavioral_summary,
        compliance_summary=compliance_summary,
        reservation_summary=reservation_summary,
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


def _filter_unsubstantiated_reservation_flags(
    flags: list[dict], state: AgentState
) -> list[dict]:
    """Keep verified bookings from being misreported as doctor/specialty errors."""
    verification = state.get("appointment_verification") or {}
    eligibility = state.get("eligibility_result") or {}
    is_conclusively_ineligible = (
        eligibility.get("http_status") == 200
        and eligibility.get("is_eligible") is False
        and eligibility.get("api_status") in {"Fail", "Success"}
    )
    if verification.get("found") is not True:
        return flags

    # A verified ineligible reservation may retain only the appropriate C2C_005
    # finding; doctor/specialty mismatch claims remain unsupported either way.
    unsupported_markers = (
        "wrong doctor",
        "wrong specialty",
        "wrong appointment",
        "doctor and specialty",
        "incorrect doctor",
        "incorrect specialty",
    )
    if not is_conclusively_ineligible:
        unsupported_markers += (
            "c2c_005",
            "wrong reservation",
            "booking error",
        )
    filtered = [
        flag for flag in flags
        if not any(marker in flag.get("description", "").lower() for marker in unsupported_markers)
    ]
    if len(filtered) != len(flags):
        logger.warning(
            "aggregate_results: removed %d unsupported reservation mismatch flag(s) for verified booking | call_id=%s",
            len(flags) - len(filtered),
            state["call"].call_id,
        )
    return filtered


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
    service     = state.get("service_eval")     or {}
    package     = state.get("package_eval")     or {}
    script      = state.get("script_eval")      or {}
    scoring     = state.get("scoring_eval")     or {}

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

    # Normalise service_flags
    _raw_service_flags: list[dict] = service.get("service_flags", [])
    _service_flags: list[dict] = []
    for f in _raw_service_flags:
        flag = dict(f)
        if flag.get("type") == "positive":
            flag["type"] = "NC"
            flag["severity"] = "positive"
        _service_flags.append(flag)

    # Normalise package_flags
    _raw_package_flags: list[dict] = package.get("package_flags", [])
    _package_flags: list[dict] = []
    for f in _raw_package_flags:
        flag = dict(f)
        if flag.get("type") == "positive":
            flag["type"] = "NC"
            flag["severity"] = "positive"
        _package_flags.append(flag)

    # Merge all compliance flag lists from all focused evaluations
    all_flags: list[dict] = (
        behavioral.get("behavioral_flags", [])
        + compliance.get("compliance_flags", [])
        + reservation.get("reservation_flags", [])
        + _offer_flags
        + _service_flags
        + _package_flags
        + script.get("script_flags", [])
        + scoring.get("compliance_flags", [])
    )

    all_flags = _filter_unsubstantiated_reservation_flags(all_flags, state)

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
    }

    logger.debug(
        "aggregate_results | call_id=%s flags=%d assessment=%s classification=%s profiling=%s offer_outcome=%s service_outcome=%s package_outcome=%s",
        call_id,
        len(deduped_flags),
        merged["overall_assessment"],
        merged["agent_performance"].get("Agent Classification"),
        merged["agent_performance"].get("Profiling Comment"),
        offer.get("offer_outcome", "N/A"),
        service.get("service_outcome", "N/A"),
        package.get("package_outcome", "N/A"),
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

    if result.overall_assessment == "escalate" and not result.escalation_required:
        logger.warning(
            "integrity_check: assessment='escalate' but escalation_required=False — correcting | call_id=%s",
            call_id,
        )
        result = result.model_copy(update={"escalation_required": True})

    has_critical_eligibility_error = any(
        flag.severity == "critical" and "C2C_030" in flag.description
        for flag in result.compliance_flags
    )
    if has_critical_eligibility_error:
        logger.warning(
            "integrity_check: critical eligibility misinformation detected | call_id=%s",
            call_id,
        )
        result = result.model_copy(
            update={
                "overall_assessment": "escalate",
                "escalation_required": True,
                "escalation_reason": (
                    "Critical eligibility misinformation: the agent stated an outcome "
                    "that conflicts with the eligibility API result."
                ),
            }
        )

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

# Keywords that signal a booking / appointment intent.
# Triggers the full sub-flow: extract → verify DB → infer_reservation_evaluation.
_BOOKING_KEYWORDS: list[str] = [
    "احجز",           # "book" (imperative)
    "احجزلي",         # "book for me"
    "حجز",            # "booking / reservation"
    "حجزي",           # "my booking"
    "حجزك",           # "your booking"
    "حجزه",           # "his/her booking"
    "الحجز",          # "the booking"
    "معاد",           # "appointment"
    "المعاد",         # "the appointment"
    "معادي",          # "my appointment"
    "موعد",           # "appointment / time-slot"
    "المواعيد",       # "the appointments"
    "مواعيد",         # "appointments"
    "تأكيد الحجز",
    "الموعد",
    "تم تأكيد الحجز",
]

# Keywords that signal an offer / package / discount inquiry.
# Triggers extract_appointment_details (to get specialty + offer name) but
# SKIPS the DB appointment verification step — there is no reservation to check.
_OFFER_KEYWORDS: list[str] = [
    "عرض",            # "offer"
    "العرض",          # "the offer"
    "عروض",           # "offers" (plural)
    "العروض",         # "the offers"
    "باقة",           # "package"
    "الباقة",         # "the package"
    "باقات",          # "packages"
    "خصم",            # "discount"
    "الخصم",          # "the discount"
    "تخفيض",          # "reduction / sale"
    "بروموشن",        # "promotion" (colloquial)
    "كود الخصم",      # "discount code"
]

_INSURANCE_QUESTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"كاش\s*(?:او|أو|ام)\s*ت(?:ا|أ)مين", re.IGNORECASE),
    re.compile(r"الحجز\s+كاش\s*(?:او|أو|ام)\s*ت(?:ا|أ)مين", re.IGNORECASE),
    re.compile(r"cash\s*(?:or|/)\s*insurance", re.IGNORECASE),
    re.compile(r"insured\s+or\s+cash", re.IGNORECASE),
]

_INSURED_REPLY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\binsured\b", re.IGNORECASE),
    re.compile(r"\binsurance\b", re.IGNORECASE),
    re.compile(r"\bمؤمن\b", re.IGNORECASE),
    re.compile(r"\bت(?:ا|أ)مين\b", re.IGNORECASE),
]

_CASH_OR_UNINSURED_REPLY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bcash\b", re.IGNORECASE),
    re.compile(r"\bnot\s+insured\b", re.IGNORECASE),
    re.compile(r"\bno\s+insurance\b", re.IGNORECASE),
    re.compile(r"\bكاش\b", re.IGNORECASE),
    re.compile(r"\bنقد(?:ي|ا)?\b", re.IGNORECASE),
    re.compile(r"(?:غير|مو)\s*مؤمن", re.IGNORECASE),
    re.compile(r"(?:ما|لا)\s*(?:عندي|لدي|يوجد)?\s*ت(?:ا|أ)مين", re.IGNORECASE),
    re.compile(r"بدون\s*ت(?:ا|أ)مين", re.IGNORECASE),
]

_IQAMA_NUMBER_PATTERN = re.compile(
    r"(?:iqama|i?qama|رقم\s*(?:ال)?(?:اقامة|إقامة)|(?:ال)?(?:اقامة|إقامة))?\D*([12]\d{9})",
    re.IGNORECASE,
)

_IDENTITY_OR_IQAMA_REQUEST_PATTERN = re.compile(
    r"(?:رقم\s*(?:ال)?(?:هوية|هويتك|اقامة|إقامة|اقامتك|إقامتك)|(?:iqama|i?qama)(?:\s*(?:number|no\.?))?)",
    re.IGNORECASE,
)


def _extract_iqama_number(text: str) -> str | None:
    """Return the first 10-digit iqama-like number found in the transcript."""
    if not text:
        return None

    match = _IQAMA_NUMBER_PATTERN.search(text)
    if match:
        return match.group(1)

    fallback = re.search(r"\b([12]\d{9})\b", text)
    if fallback:
        return fallback.group(1)

    return None


def _agent_requested_identity_or_iqama(transcript_text: str) -> bool:
    """Return True when the agent requests an identity or IQAMA number."""
    if not transcript_text:
        return False

    return any(
        "agent:" in line.lower() and _IDENTITY_OR_IQAMA_REQUEST_PATTERN.search(line)
        for line in transcript_text.splitlines()
    )


def _patient_declined_insurance(transcript_text: str) -> bool:
    """Return True when the patient explicitly selects cash or no insurance."""
    if not transcript_text:
        return False

    return any(
        "patient:" in line.lower()
        and any(pattern.search(line) for pattern in _CASH_OR_UNINSURED_REPLY_PATTERNS)
        for line in transcript_text.splitlines()
    )


def _patient_selected_insured(transcript_text: str) -> bool:
    """
    Detect the flow where the agent asks for payment type and the patient
    responds that the booking is insured.
    """
    if not transcript_text:
        return False

    lines = [line.strip() for line in transcript_text.splitlines() if line.strip()]

    for index, line in enumerate(lines):
        if "agent:" not in line.lower():
            continue
        if not any(pattern.search(line) for pattern in _INSURANCE_QUESTION_PATTERNS):
            continue

        for reply in lines[index + 1:index + 4]:
            if "agent:" in reply.lower():
                break
            if "patient:" in reply.lower() and any(
                pattern.search(reply) for pattern in _INSURED_REPLY_PATTERNS
            ):
                return True

    return False


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

    matched_booking = [kw for kw in _BOOKING_KEYWORDS if kw in transcript_text]
    matched_offer   = [kw for kw in _OFFER_KEYWORDS   if kw in transcript_text]
    is_booking = bool(matched_booking)
    # offer_only: offer keywords found but NO booking keywords
    # (if both appear, the booking flow already handles everything)
    is_offer = bool(matched_offer) and not is_booking

    if is_booking:
        intent_label = "booking_or_appointment"
    elif is_offer:
        intent_label = "offer_inquiry"
    else:
        intent_label = "other"

    logger.info(
        "detect_intent | call_id=%s is_booking=%s is_offer=%s matched_booking=%s matched_offer=%s",
        call.call_id,
        is_booking,
        is_offer,
        matched_booking or "none",
        matched_offer or "none",
    )

    return {
        "is_booking_intent": is_booking,
        "is_offer_intent":   is_offer,
        "intent_label":      intent_label,
        "node_trace": _trace(state, "detect_intent"),
    }


async def detect_insurance_intent(state: AgentState) -> dict:
    """Detect insurance flow from coverage selection or an identity/IQAMA request."""
    call = state.get("call")
    if call is None:
        return {
            "patient_is_insured": False,
            "patient_declined_insurance": False,
            "is_insurance_intent": False,
            "node_trace": _trace(state, "detect_insurance_intent"),
        }

    try:
        raw_turns = call.transcript or []
        transcript_text = "".join(
            turn if isinstance(turn, str) else turn.text
            for turn in raw_turns
            if (turn if isinstance(turn, str) else getattr(turn, "text", None))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_insurance_intent | failed to read transcript | %s", exc)
        transcript_text = ""

    patient_is_insured = _patient_selected_insured(transcript_text)
    patient_declined_insurance = _patient_declined_insurance(transcript_text)
    identity_or_iqama_requested = _agent_requested_identity_or_iqama(transcript_text)
    is_insurance_intent = (
        (patient_is_insured or identity_or_iqama_requested)
        and not patient_declined_insurance
    )
    iqama_number = state.get("iqama_number") or _extract_iqama_number(transcript_text)
    logger.info(
        "detect_insurance_intent | call_id=%s insured=%s declined=%s identity_or_iqama_requested=%s iqama=%s",
        call.call_id,
        patient_is_insured,
        patient_declined_insurance,
        identity_or_iqama_requested,
        iqama_number or "none",
    )
    return {
        "patient_is_insured": patient_is_insured,
        "patient_declined_insurance": patient_declined_insurance,
        "is_insurance_intent": is_insurance_intent,
        "iqama_number": iqama_number,
        "node_trace": _trace(state, "detect_insurance_intent"),
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
    When both ``is_booking_intent`` and ``is_offer_intent`` are False the node
    skips any work and stores ``None`` for every field.

    When ``is_offer_intent`` is True (and ``is_booking_intent`` is False), a
    fast regex scan extracts the offer name directly from the transcript
    (text immediately after عرض / باقة / خصم / ...) without calling the LLM.
    """
    is_booking = state.get("is_booking_intent", False)
    is_offer   = state.get("is_offer_intent",   False)

    # ── Fast path: offer-only – extract offer name via regex, skip LLM ────────
    if is_offer and not is_booking:
        call = state.get("call")
        call_id = call.call_id if call else "UNKNOWN"
        transcript: str = getattr(call, "transcript", "") or ""

        # Regex: capture 1–6 words after an offer trigger keyword.
        # Handles: "عرض X", "باقة X Y", etc.
        # Stops at sentence-ending punctuation, common filler words, OR discount indicators
        _OFFER_NAME_RE = re.compile(
            r"(?:عرض|باقة|باقات|العرض|الباقة|بروموشن)\s+"
            r"((?:(?![؟،\n]|هل |هلل|هليه|فيه|عندكم|\d+%|على|علي)[^\s]{2,}\s*){1,6})",
            re.IGNORECASE,
        )
        offer_name: str | None = None
        m = _OFFER_NAME_RE.search(transcript)
        if m:
            extracted = m.group(1).strip()
            # Validate: skip if extracted text is price/discount info
            # (contains %, SAR, ريال, price numbers, "على"/"علي", etc.)
            _DISCOUNT_INDICATORS = re.compile(
                r'(?:\d+%|%\d+|SAR|ريال|SR|على|علي|سعر|السعر|\d{3,})',
                re.IGNORECASE
            )
            if not _DISCOUNT_INDICATORS.search(extracted):
                offer_name = extracted
                logger.info(
                    "extract_appointment_details | call_id=%s — offer fast-path: extracted offer_name=%r",
                    call_id, offer_name,
                )
            else:
                logger.info(
                    "extract_appointment_details | call_id=%s — offer fast-path: rejected discount text %r (not an offer name)",
                    call_id, extracted,
                )
        else:
            logger.info(
                "extract_appointment_details | call_id=%s — offer fast-path: no offer name found after keyword",
                call_id,
            )

        return {
            "appointment_details": {
                "appointment_date": None,
                "doctor_name":      None,
                "specialty_name":   None,
                "patient_name":     None,
                "offer_name":       offer_name,
            },
            "node_trace": _trace(state, "extract_appointment_details"),
        }

    # ── Skip entirely when neither booking nor offer ───────────────────────
    if not is_booking:
        logger.info(
            "extract_appointment_details | is_booking_intent=False, is_offer_intent=False — skipping"
        )
        return {
            "appointment_details": {
                "appointment_date": None,
                "doctor_name": None,
                "specialty_name": None,
                "patient_name": None,
                "offer_name": None,
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

"""
eligibility_node.py
───────────────────
LangGraph node: check_patient_eligibility

Position in the booking graph
──────────────────────────────
  detect_intent
       │
       ▼ (booking intent confirmed)
  check_patient_eligibility          ← NEW
       │
       ├──(eligible)──────────────→  extract_appointment_details
       ├──(not_eligible)──────────→  handle_ineligible_patient
       └──(error / timeout)────────→ handle_error

What this node does
───────────────────
1. Reads `iqama_number` from state (set by detect_intent or injected by the API layer).
2. Calls Beneficiary_api() → /check-insurance → insurance coverage snapshot.
3. Parses the response:
     • ApiStatus == "Success" + at least one Insurance entry  → eligible
     • ApiStatus != "Success" or empty Insurance list         → not_eligible
     • Network / timeout / unexpected exception               → error
4. Writes the parsed result back into state so downstream nodes
   (infer_reservation_evaluation, scoring, etc.) can reference it.

NOTE: EligibilityService.process_visit() requires a visit_id (post-booking).
      Do NOT call it here. Wire it after the appointment is created in the DB.
"""

#from __future__ import annotations

import logging
import time
import random
from datetime import datetime
from typing import Any

from app.agent.state import AgentState          # adjust import to your project layout
from app.eligibilty.eligibility import Beneficiary_api

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_beneficiary_response(iqama: int, raw_response: Any) -> dict:
    """Parse the `/check-insurance` response into one authoritative outcome."""
    checked_at = datetime.now().isoformat(timespec="seconds")
    default = {
        "iqama_number": iqama,
        "http_status": None,
        "api_status": "Unknown",
        "is_eligible": False,
        "insurance": None,
        "error_code": None,
        "reason": "Invalid eligibility API response",
        "transaction_name": None,
        "checked_at": checked_at,
    }
    if not isinstance(raw_response, dict):
        return default

    http_status = raw_response.get("status_code")
    response = raw_response.get("response")
    if not isinstance(response, dict):
        return {**default, "http_status": http_status, "reason": "Missing response payload"}

    api_status = response.get("ApiStatus", "Unknown")
    insurance_list = response.get("Insurance")
    error_code = response.get("ErrorCode")
    error_description = response.get("ErrorDescription")
    transaction_name = response.get("TransactionName")
    insurance_entry = (
        insurance_list[0]
        if isinstance(insurance_list, list) and insurance_list and isinstance(insurance_list[0], dict)
        else None
    )

    is_eligible = (
        http_status == 200
        and api_status == "Success"
        and insurance_entry is not None
    )
    if is_eligible:
        reason = None
    elif http_status != 200:
        reason = f"Eligibility API returned HTTP status {http_status!r}"
    elif api_status == "Fail":
        reason = error_description or error_code or "Eligibility API reported no coverage"
    elif api_status == "Success":
        reason = "Eligibility API returned Success without a valid Insurance entry"
    else:
        reason = error_description or "Unknown eligibility API status"

    return {
        "iqama_number": iqama,
        "http_status": http_status,
        "api_status": api_status,
        "is_eligible": is_eligible,
        "insurance": insurance_entry,
        "error_code": error_code,
        "reason": reason,
        "transaction_name": transaction_name,
        "checked_at": checked_at,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

async def check_patient_eligibility(state: AgentState) -> dict:
    """
    LangGraph node — runs BEFORE appointment extraction.

    Reads
    -----
    state["iqama_number"]  : str | int   (required)

    Writes
    ------
    state["eligibility_result"]  : dict   (always set, even on error)
    state["error"]               : str    (only on hard failure)
    """
    raw_iqama = state.get("iqama_number")

    # ── Guard: iqama_number must be present ─────────────────────────────────
    if not raw_iqama:
        logger.warning("check_patient_eligibility: iqama_number missing from state")
        return {
            "eligibility_result": {
                "iqama_number": None,
                "api_status":   "Fail",
                "is_eligible":  False,
                "insurance":    None,
                "checked_at":   datetime.now().isoformat(timespec="seconds"),
                "reason":       "iqama_number not provided",
            }
        }

    # ── Normalise: convert to int (Beneficiary_api expects int) ─────────────
    try:
        iqama = int(str(raw_iqama).strip())
    except (ValueError, TypeError) as exc:
        logger.error("check_patient_eligibility: cannot coerce iqama to int — %s", exc)
        return {
            "error": f"Invalid Iqama number format: {raw_iqama!r}",
            "eligibility_result": {
                "iqama_number": raw_iqama,
                "api_status":   "Fail",
                "is_eligible":  False,
                "insurance":    None,
                "checked_at":   datetime.now().isoformat(timespec="seconds"),
                "reason":       f"Non-numeric Iqama: {raw_iqama}",
            },
        }

    # ── Call Beneficiary API ─────────────────────────────────────────────────
    logger.info("check_patient_eligibility: calling Beneficiary_api for iqama=%d", iqama)

    try:
        # Small random jitter — matches the pattern in Iqama_table()
        jitter_ms = random.uniform(10, 30)
        time.sleep(jitter_ms / 1000)

        raw_response = Beneficiary_api(iqama)

    except Exception as exc:
        logger.exception(
            "check_patient_eligibility: Beneficiary_api raised an exception: %s", exc
        )
        return {
            "error": f"Eligibility API call failed: {exc}",
            "eligibility_result": {
                "iqama_number": iqama,
                "api_status":   "Fail",
                "is_eligible":  False,
                "insurance":    None,
                "checked_at":   datetime.now().isoformat(timespec="seconds"),
                "reason":       str(exc),
            },
        }

    # ── Parse & classify ─────────────────────────────────────────────────────
    result = _parse_beneficiary_response(iqama, raw_response)

    if result["is_eligible"]:
        logger.info(
            "check_patient_eligibility: iqama=%d → ELIGIBLE (insurer: %s)",
            iqama,
            result["insurance"].get("InsuranceCompanyEN", "unknown"),
        )
    else:
        logger.warning(
            "check_patient_eligibility: iqama=%d → NOT ELIGIBLE (api_status=%s)",
            iqama,
            result["api_status"],
        )

    return {"eligibility_result": result}


# ─────────────────────────────────────────────────────────────────────────────
# Router (add this to graph.py)
# ─────────────────────────────────────────────────────────────────────────────

def _eligibility_router(state: AgentState):
    """
    Conditional edge after check_patient_eligibility.

    Returns
    -------
    "eligible"      → proceed to extract_appointment_details
    "not_eligible"  → route to handle_ineligible_patient (graceful rejection)
    "error"         → route to handle_error
    """
    if state.get("error"):
        return "error"

    result = state.get("eligibility_result", {})

    if result.get("is_eligible"):
        return "eligible"

    return "not_eligible"


# ─────────────────────────────────────────────────────────────────────────────
# Ineligible handler (simple, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_ineligible_patient(state: AgentState) -> dict:
    """
    Pure-Python node — no LLM call needed.
    Records the rejection reason and surfaces a human-readable message
    that the downstream response builder can include in the agent's reply.
    """
    result = state.get("eligibility_result", {})
    iqama  = result.get("iqama_number", "unknown")

    api_status = result.get("api_status", "Unknown")
    reason     = result.get("reason", "Insurance coverage not found")

    logger.info(
        "handle_ineligible_patient: iqama=%s api_status=%s reason=%s",
        iqama, api_status, reason,
    )

    return {
        "ineligible_reason": reason,
        "final_response": (
            "عذراً، لم نتمكن من التحقق من تغطيتك التأمينية. "
            "يرجى التواصل مع شركة التأمين أو زيارة أقرب فرع. "
            f"(رقم الإقامة: {iqama})"
        ),
    }

