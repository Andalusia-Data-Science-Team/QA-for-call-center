"""
Patient Profile Validation Nodes

Nodes for fetching and validating patient profile information from MPI.
"""
import logging
import os
from typing import Any

from app.agent.state import AgentState
from app.services.llm_client import LLMClient
from app.services.get_patient import (
    get_patient,
    get_token_for_env,
    login,
)

logger = logging.getLogger(__name__)


def _trace(_state: AgentState, node: str) -> list[str]:
    """Helper to return a single-item list for the node_trace reducer."""
    return [node]


async def fetch_patient_profile(state: AgentState) -> dict:
    """
    Fetch patient profile from MPI service when booking intent + iqama detected.
    
    Uses the get_patient service to retrieve profile based on iqama_number.
    Maps business_unit to the appropriate MPI environment.
    
    Result stored in state["patient_profile_result"].
    """
    call = state.get("call")
    if call is None:
        return {
            "error": "fetch_patient_profile: call is None",
            "error_node": "fetch_patient_profile",
            "patient_profile_result": {"found": False, "error": "No call context"},
            "node_trace": _trace(state, "fetch_patient_profile"),
        }
    
    iqama_number = state.get("iqama_number")
    if not iqama_number:
        logger.info(
            "fetch_patient_profile | call_id=%s skip=no_iqama",
            call.call_id,
        )
        return {
            "patient_profile_result": {
                "found": False,
                "profile": None,
                "error": "No iqama number available",
                "env": None,
            },
            "node_trace": _trace(state, "fetch_patient_profile"),
        }
    
    # Map business unit to MPI environment
    business_unit = getattr(call, "business_unit", None) or "LIVE"
    env_map = {
        "BU-AHJ": "AHJ",
        "LIVE": "AHJ",
        "BU-SNB": "SNB",
        "BU-AKW": "AKW",
        "BU-ALW": "ALW",
        "BU-ERS": "ERS",
        "BU-JDC": "JDC",
    }
    env = env_map.get(business_unit, "AHJ")
    
    logger.info(
        "fetch_patient_profile | call_id=%s iqama=%s bu=%s env=%s",
        call.call_id,
        iqama_number,
        business_unit,
        env,
    )
    
    try:
        # Get authentication token for the environment
        token = get_token_for_env(env)
        if not token:
            # Try to login if token not available
            username = os.getenv("MPI_SERVICE_USERNAME", "AiModule")
            password = os.getenv("MPI_SERVICE_PASSWORD", "123456")
            token = login(env, username, password)
        
        # Fetch patient profile
        response = get_patient(env, iqama_number, token=token)
        
        if response.status_code != 200:
            logger.warning(
                "fetch_patient_profile | call_id=%s status=%d",
                call.call_id,
                response.status_code,
            )
            return {
                "patient_profile_result": {
                    "found": False,
                    "profile": None,
                    "error": f"MPI returned status {response.status_code}",
                    "env": env,
                },
                "node_trace": _trace(state, "fetch_patient_profile"),
            }
        
        body = response.text.strip()
        if not body or body.lower() == "null":
            logger.info(
                "fetch_patient_profile | call_id=%s result=not_found",
                call.call_id,
            )
            return {
                "patient_profile_result": {
                    "found": False,
                    "profile": None,
                    "error": "Patient not found in MPI",
                    "env": env,
                },
                "node_trace": _trace(state, "fetch_patient_profile"),
            }
        
        try:
            profile = response.json()
        except ValueError:
            logger.error(
                "fetch_patient_profile | call_id=%s error=invalid_json",
                call.call_id,
            )
            return {
                "patient_profile_result": {
                    "found": False,
                    "profile": None,
                    "error": "Invalid JSON response from MPI",
                    "env": env,
                },
                "node_trace": _trace(state, "fetch_patient_profile"),
            }
        
        logger.info(
            "fetch_patient_profile | call_id=%s result=found patient_code=%s",
            call.call_id,
            profile.get("Code"),
        )
        
        return {
            "patient_profile_result": {
                "found": True,
                "profile": profile,
                "error": None,
                "env": env,
            },
            "node_trace": _trace(state, "fetch_patient_profile"),
        }
    
    except Exception as exc:
        logger.error(
            "fetch_patient_profile | call_id=%s error=%s",
            call.call_id,
            str(exc),
        )
        return {
            "patient_profile_result": {
                "found": False,
                "profile": None,
                "error": str(exc),
                "env": env,
            },
            "node_trace": _trace(state, "fetch_patient_profile"),
        }


async def validate_patient_profile(
    state: AgentState, llm_client: LLMClient
) -> dict:
    """
    Validate that agent collected all required patient information and
    acknowledged profile updates.
    
    Checks:
    1. Required fields present in MPI profile
    2. Agent asked for missing required fields
    3. Agent acknowledged any profile updates patient mentioned
    
    Produces C2C compliance flags when agent failed to collect or acknowledge.
    Result stored in state["patient_profile_eval"].
    """
    from app.agent.nodes import _focused_llm_call
    
    call = state["call"]
    profile_result = state.get("patient_profile_result") or {}
    
    # Skip if no iqama (profile fetch would have already skipped)
    if not state.get("iqama_number"):
        logger.info(
            "validate_patient_profile | call_id=%s skip=no_iqama",
            call.call_id,
        )
        return {
            "patient_profile_eval": {
                "applicable": False,
                "validation_status": "not_applicable",
                "missing_fields": [],
                "mismatches": [],
                "profile_updates": [],
                "compliance_flags": [],
            },
            "node_trace": _trace(state, "validate_patient_profile"),
        }
    
    # Skip if profile fetch failed
    if not profile_result.get("found"):
        logger.info(
            "validate_patient_profile | call_id=%s skip=profile_not_found",
            call.call_id,
        )
        return {
            "patient_profile_eval": {
                "applicable": False,
                "validation_status": "profile_not_found",
                "missing_fields": [],
                "mismatches": [],
                "profile_updates": [],
                "compliance_flags": [],
            },
            "node_trace": _trace(state, "validate_patient_profile"),
        }
    
    from app.prompts.qa_prompt import build_patient_profile_validation_prompt
    
    user_prompt = build_patient_profile_validation_prompt(
        call,
        profile_result=profile_result,
        compliance_pillars=state.get("compliance_pillars", ""),
    )
    
    logger.debug(
        "validate_patient_profile | call_id=%s prompt_len=%d",
        call.call_id,
        len(user_prompt),
    )
    
    data, err = await _focused_llm_call(
        "validate_patient_profile",
        call.call_id,
        user_prompt,
        llm_client,
        state,
    )
    if err:
        return err
    
    # Normalize the result
    validation_status = data.get("validation_status", "error")
    missing_fields = data.get("missing_fields", [])
    profile_updates = data.get("profile_updates", [])
    compliance_flags = data.get("compliance_flags", [])
    
    logger.info(
        "validate_patient_profile | call_id=%s status=%s missing=%d updates=%d flags=%d",
        call.call_id,
        validation_status,
        len(missing_fields),
        len(profile_updates),
        len(compliance_flags),
    )
    
    return {
        "patient_profile_eval": data,
        "usage_list": [data.pop("_usage", {})],
        "node_trace": _trace(state, "validate_patient_profile"),
    }
