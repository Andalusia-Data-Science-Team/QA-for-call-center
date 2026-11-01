"""
Eligibility and attribution rules.

WHAT THE SQL ALREADY DID
────────────────────────
app/SQL/automation/automation_day_conversations.sql attributes each
conversation to its inbox owner — the same rule NEWFIX_CM_Chat_Search.sql and
the manual dashboard use. Every candidate row records
AttributionRule = 'inbox_owner', so if the rule ever changes the change is
visible in the data rather than retroactively implied.

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

ATTRIBUTION_RULE = "inbox_owner"


@dataclass(frozen=True, slots=True)
class Eligibility:
    is_eligible: bool
    reason: Optional[str] = None


def assess(row) -> Eligibility:
    """
    Decide whether one candidate row may be sampled.

    The checks, in the order they are applied:

    1. **An attributed agent.** No inbox-owner email means we cannot say whose
       work is being evaluated. Quarantined rather than guessed.

    2. **A handling timestamp** (the archive time). Needed to place the
       conversation in a reporting day and an interval.

    3. **The agent replied at least once.** AgentReplied comes from the same
       SenderWasOwner test NEWFIX uses for FirstResponse. A closed chat the
       owner never answered has no agent performance to assess.

    4. **A completed conversation.** ArchiveDateTime being set is the signal
       used, because it is unambiguous. The raw State and AnswerState codes
       are carried through but NOT used as rules — their meanings have not
       been confirmed, and inventing a status code is how a sampler silently
       excludes half a day's traffic.

    Candidates are windowed on ArchiveDateTime, so a conversation reaches the
    sampler in the interval it closed in and check 4 normally passes; it stays
    as a guard in case the source ever returns an unarchived row.
    """
    agent_email = _clean(row.get("AgentEmail"))
    if not agent_email:
        return Eligibility(False, "no_inbox_owner")

    if _missing(row.get("HandlingTimeUtc")):
        return Eligibility(False, "no_handling_timestamp")

    if _as_int(row.get("AgentReplied")) < 1:
        return Eligibility(False, "agent_never_replied")

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
