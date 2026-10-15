"""
Tests for the eligibility policy (decision D2).

The policy is deliberately conservative — a conversation is excluded unless
it clearly qualifies — so these tests mostly check that each exclusion fires
AND carries a reason. A silent exclusion is worse than a wrong one, because
a reason can be reviewed and a silence cannot.
"""

import numpy as np
import pandas as pd
import pytest

from app.automation.eligibility import ATTRIBUTION_RULE, assess, summarise


def _row(**overrides) -> dict:
    base = {
        "ConversationId": "c1",
        "AgentEmail": "agent@andalusiagroup.net",
        "HandlingTimeUtc": pd.Timestamp("2026-10-29 06:30:00"),
        "HumanMessageCount": 4,
        "ArchiveDateTimeUtc": pd.Timestamp("2026-10-29 06:35:00"),
        "ConversationState": "Closed",
    }
    base.update(overrides)
    return base


def test_a_normal_closed_conversation_is_eligible():
    verdict = assess(_row())
    assert verdict.is_eligible
    assert verdict.reason is None


def test_attribution_rule_is_named_so_a_later_change_is_visible():
    assert ATTRIBUTION_RULE == "last_human_message"


@pytest.mark.parametrize("missing", [None, "", np.nan])
def test_no_attributable_agent_is_quarantined(missing):
    verdict = assess(_row(AgentEmail=missing))
    assert not verdict.is_eligible
    assert verdict.reason == "no_attributable_human_agent"


def test_missing_handling_time_is_quarantined():
    verdict = assess(_row(HandlingTimeUtc=None))
    assert not verdict.is_eligible
    assert verdict.reason == "no_human_handling_timestamp"


def test_bot_only_conversation_is_excluded():
    verdict = assess(_row(HumanMessageCount=0))
    assert not verdict.is_eligible
    assert verdict.reason == "only_0_human_messages"


def test_open_conversation_is_excluded_not_evaluated_early():
    """
    An unfinished chat has no complete handling to assess. It is not lost —
    reconciliation picks it up as a late arrival once it closes.
    """
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
        _row(ConversationId="b", HumanMessageCount=0),
        _row(ConversationId="c", ArchiveDateTimeUtc=None),
        _row(ConversationId="d"),
    ])
    counts = summarise(frame)
    assert counts["eligible"] == 2
    assert counts["only_0_human_messages"] == 1
    assert counts["conversation_not_closed"] == 1
    assert sum(counts.values()) == len(frame)
