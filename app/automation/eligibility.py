"""
Eligibility and attribution rules — the other unresolved decision, D2.

Kept in one file for the same reason as the roster: the rule is a business
decision that has not been signed off, and when it is, exactly one function
should need to change.

WHAT THE SQL ALREADY DID
────────────────────────
app/SQL/automation/day_candidates.sql attributes each conversation to the
human who sent its last non-system message. That is the proposal, not a
confirmed rule, and it is recorded on every candidate row as
AttributionRule = 'last_human_message' so a later change is visible in the
data rather than retroactively implied.

WHAT IS DECIDED HERE
────────────────────
Which of those attributed conversations may be evaluated. The policy below is
deliberately conservative: a conversation is excluded unless it clearly
qualifies, and every exclusion carries a reason that is persisted. Silence is
never evidence of eligibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ATTRIBUTION_RULE = "last_human_message"

# Minimum human-agent messages for a conversation to be worth evaluating. One
# is enough to establish the agent handled it; the QA rubric decides whether
# the handling was adequate, which is not this module's business.
MIN_HUMAN_MESSAGES = 1


@dataclass(frozen=True, slots=True)
class Eligibility:
    is_eligible: bool
    reason: Optional[str] = None


def assess(row) -> Eligibility:
    """
    Decide whether one candidate row may be sampled.

    The checks, in the order they are applied:

    1. **An attributed human agent.** No resolvable agent email means we cannot
       say whose work is being evaluated. Quarantined rather than guessed.

    2. **A human handling timestamp.** Needed to place the conversation in a
       reporting day and an interval.

    3. **At least one human-agent message.** Excludes bot-only and
       system-only interactions, which have no agent performance to assess.

    4. **A completed conversation.** ArchiveDateTime being set is the signal
       used, because it is unambiguous. The raw State and AnswerState codes
       are carried through but NOT used as rules — their meanings have not
       been confirmed, and inventing a status code is how a sampler silently
       excludes half a day's traffic.

    A conversation still open when its interval closes is simply not eligible
    for that interval. It will be picked up by the reconciliation pass once it
    completes, as a late arrival.
    """
    agent_email = _clean(row.get("AgentEmail"))
    if not agent_email:
        return Eligibility(False, "no_attributable_human_agent")

    if _missing(row.get("HandlingTimeUtc")):
        return Eligibility(False, "no_human_handling_timestamp")

    human_messages = _as_int(row.get("HumanMessageCount"))
    if human_messages < MIN_HUMAN_MESSAGES:
        return Eligibility(False, f"only_{human_messages}_human_messages")

    if _missing(row.get("ArchiveDateTimeUtc")):
        return Eligibility(False, "conversation_not_closed")

    return Eligibility(True, None)


def summarise(frame: pd.DataFrame) -> dict[str, int]:
    """Count exclusions by reason, for the operational log."""
    counts: dict[str, int] = {}
    for _, row in frame.iterrows():
        verdict = assess(row)
        key = "eligible" if verdict.is_eligible else (verdict.reason or "ineligible")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value)


def _as_int(value) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
