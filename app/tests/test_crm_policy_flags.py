"""CRM findings must survive the policy matcher and affect the call outcome."""

from app.service_hub.crm_leads_validation import (
    missing_lead_evaluation,
    normalize_crm_lead_evaluation,
)
from app.services.regulation_catalog import matched_regulation_flags, policy_outcome


def test_wrong_last_call_result_becomes_system_action_c2b():
    evaluation = normalize_crm_lead_evaluation({
        "crm_lead_status": "violation",
        "field_checks": [{
            "field": "new_lastcallresult",
            "matches": False,
            "reason": "Recorded Inquiry only [01] instead of Not interested [02].",
        }],
        "crm_leads_flags": [{
            "type": "C2B",
            "attribute": "Compliance Report",
            "comment": "Submit wrong call reason",
            "description": "CRM lead mismatch in new_lastcallresult: wrong final outcome.",
            "transcript_excerpt": "Agent: chat closed after no response",
        }],
    })

    flags = matched_regulation_flags(evaluation["crm_leads_flags"])
    assert len(flags) == 1
    assert flags[0]["type"] == "C2B"
    assert flags[0]["attribute"] == "Displays positive company image"
    assert flags[0]["comment"] == "Didn’t take the proper action on system such linking CST Contract etc."
    assert policy_outcome(flags, "pass", False) == ("needs_review", False, "C")


def test_unflagged_last_call_result_field_check_gets_policy_label():
    evaluation = normalize_crm_lead_evaluation({
        "crm_lead_status": "violation",
        "crm_leads_flags": [],
        "field_checks": [{
            "field": "new_lastcallresult",
            "matches": False,
            "reason": "Wrong disposition",
        }],
    })

    flags = matched_regulation_flags(evaluation["crm_leads_flags"])
    assert len(flags) == 1
    assert flags[0]["comment"] == "Didn’t take the proper action on system such linking CST Contract etc."


def test_other_crm_field_mismatch_gets_wrong_field_policy_label():
    evaluation = normalize_crm_lead_evaluation({
        "crm_lead_status": "violation",
        "crm_leads_flags": [{
            "type": "C2B",
            "description": "CRM lead mismatch in new_clinicbu: wrong branch.",
            "transcript_excerpt": "N/A",
        }],
        "field_checks": [],
    })

    flags = matched_regulation_flags(evaluation["crm_leads_flags"])
    assert len(flags) == 1
    assert flags[0]["attribute"] == "Compliance Report"
    assert flags[0]["comment"] == "Missing / Wrong Field"


def test_missing_crm_lead_flag_is_policy_matched():
    flags = matched_regulation_flags(missing_lead_evaluation("Lead not found")["crm_leads_flags"])
    assert len(flags) == 1
    assert flags[0]["comment"] == "Missing / Wrong Field"
