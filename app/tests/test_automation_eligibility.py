"""
Tests for the eligibility policy.

Attribution is the conversation's inbox owner, as NEWFIX_CM_Chat_Search.sql and
the manual dashboard attribute it. The policy is deliberately conservative — a
conversation is excluded unless it clearly qualifies — so these tests mostly
check that each exclusion fires AND carries a reason. A silent exclusion is
worse than a wrong one, because a reason can be reviewed and a silence cannot.
"""

import numpy as np
import pandas as pd
import pytest

from app.automation.eligibility import ATTRIBUTION_RULE, assess, summarise


def _row(**overrides) -> dict:
    base = {
        "ConversationId": "c1",
        "AgentEmail": "agent@andalusiagroup.net",
        "HandlingTimeUtc": pd.Timestamp("2026-10-29 06:35:00"),
        "AgentReplied": 1,
        "ArchiveDateTimeUtc": pd.Timestamp("2026-10-29 06:35:00"),
        "ConversationState": "Closed",
    }
    base.update(overrides)
    return base


def test_a_normal_closed_conversation_is_eligible():
    verdict = assess(_row())
    assert verdict.is_eligible
    assert verdict.reason is None


def test_attribution_rule_matches_newfix():
    """NEWFIX attributes a chat to its inbox owner; the sampler records the same."""
    assert ATTRIBUTION_RULE == "inbox_owner"


@pytest.mark.parametrize("missing", [None, "", np.nan])
def test_no_inbox_owner_is_quarantined(missing):
    verdict = assess(_row(AgentEmail=missing))
    assert not verdict.is_eligible
    assert verdict.reason == "no_inbox_owner"


def test_missing_handling_time_is_quarantined():
    verdict = assess(_row(HandlingTimeUtc=None))
    assert not verdict.is_eligible
    assert verdict.reason == "no_handling_timestamp"


@pytest.mark.parametrize("replied", [0, None, np.nan])
def test_conversation_the_agent_never_answered_is_excluded(replied):
    verdict = assess(_row(AgentReplied=replied))
    assert not verdict.is_eligible
    assert verdict.reason == "agent_never_replied"


def test_unarchived_row_is_excluded_if_the_source_ever_returns_one():
    verdict = assess(_row(ArchiveDateTimeUtc=None))
    assert not verdict.is_eligible
    assert verdict.reason == "conversation_not_closed"


def test_unconfirmed_state_codes_are_not_used_as_rules():
    """
    We must not invent the meaning of CM's State / AnswerState codes. A
    conversation with an unfamiliar state but a real archive time is still
    eligible; guessing the codes is how a sampler silently drops a day.
    """
    assert assess(_row(ConversationState="SomethingUnfamiliar", AnswerState=99)).is_eligible


def test_summarise_counts_every_row_exactly_once():
    frame = pd.DataFrame([
        _row(ConversationId="a"),
        _row(ConversationId="b", AgentReplied=0),
        _row(ConversationId="c", ArchiveDateTimeUtc=None),
        _row(ConversationId="d"),
    ])
    counts = summarise(frame)
    assert counts["eligible"] == 2
    assert counts["agent_never_replied"] == 1
    assert counts["conversation_not_closed"] == 1
    assert sum(counts.values()) == len(frame)
