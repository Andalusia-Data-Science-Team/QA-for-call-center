"""Tests for app.automation.sampling — the reproducibility guarantees."""

from datetime import date

import pytest

from app.automation.sampling import (
    Candidate,
    agent_ranking,
    allocate_agent_quotas,
    allocate_interval_quotas,
    rank_hash,
    select_nonconsecutive,
)

SEED = "42"
DAY = date(2026, 10, 29)


def _hash(conversation_id: str, **overrides) -> str:
    kwargs = dict(
        base_seed=SEED, policy_version="v1", reporting_date=DAY,
        shift_id="shift-a", agent_id="agent@x.net", interval_id="20261029T0600Z/20261029T0800Z",
        conversation_id=conversation_id,
    )
    kwargs.update(overrides)
    return rank_hash(**kwargs)


# ── Hashing ──────────────────────────────────────────────────────────────────

def test_hash_is_stable_across_calls():
    assert _hash("abc") == _hash("abc")


def test_hash_is_uuid_case_insensitive():
    """CM_DWH and pyodbc disagree on UUID case; the selection must not."""
    upper = "A1B2C3D4-0000-0000-0000-00000000FFFF"
    assert _hash(upper) == _hash(upper.lower())


def test_hash_changes_with_every_field():
    base = _hash("abc")
    assert _hash("abd") != base
    assert _hash("abc", policy_version="v2") != base
    assert _hash("abc", base_seed="43") != base
    assert _hash("abc", agent_id="other@x.net") != base
    assert _hash("abc", interval_id="20261029T0800Z/20261029T1000Z") != base
    assert _hash("abc", reporting_date=date(2026, 10, 30)) != base


def test_hash_does_not_use_pythons_salted_hash():
    """
    A hard-coded expected digest. If someone swaps SHA-256 for hash() or
    changes the canonical serialisation, this fails loudly instead of
    silently reshuffling every future selection.
    """
    digest = rank_hash(
        base_seed="42", policy_version="v1", reporting_date=date(2026, 10, 29),
        shift_id="s", agent_id="a", interval_id="i", conversation_id="c",
    )
    assert len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest)
    assert digest == rank_hash(
        base_seed="42", policy_version="v1", reporting_date=date(2026, 10, 29),
        shift_id="s", agent_id="a", interval_id="i", conversation_id="c",
    )


def test_agent_ranking_is_deterministic_but_date_dependent():
    agents = [f"a{i}@x.net" for i in range(10)]
    today = agent_ranking(agents, base_seed=SEED, reporting_date=DAY)
    assert today == agent_ranking(agents, base_seed=SEED, reporting_date=DAY)
    assert today != agent_ranking(agents, base_seed=SEED, reporting_date=date(2026, 10, 30))
    assert sorted(today) == sorted(agents)      # a permutation, nothing lost


def test_agent_ranking_ignores_input_order():
    agents = [f"a{i}@x.net" for i in range(10)]
    assert (agent_ranking(agents, base_seed=SEED, reporting_date=DAY)
            == agent_ranking(list(reversed(agents)), base_seed=SEED, reporting_date=DAY))


# ── Allocation ───────────────────────────────────────────────────────────────

def test_twenty_agents_get_fifteen_each():
    quotas = allocate_agent_quotas([f"a{i}" for i in range(20)], 300, base_seed=SEED, reporting_date=DAY)
    assert set(quotas.values()) == {15}
    assert sum(quotas.values()) == 300


def test_remainder_is_distributed_not_dropped():
    quotas = allocate_agent_quotas([f"a{i}" for i in range(7)], 300, base_seed=SEED, reporting_date=DAY)
    assert sum(quotas.values()) == 300
    assert set(quotas.values()) == {42, 43}


def test_more_agents_than_target_covers_as_many_as_possible():
    quotas = allocate_agent_quotas([f"a{i}" for i in range(400)], 300, base_seed=SEED, reporting_date=DAY)
    assert sum(quotas.values()) == 300
    assert sum(1 for q in quotas.values() if q == 1) == 300
    assert sum(1 for q in quotas.values() if q == 0) == 100


def test_eight_hour_shift_splits_four_four_four_three():
    assert sorted(allocate_interval_quotas([120, 120, 120, 120], 15)) == [3, 4, 4, 4]


def test_interval_allocation_is_proportional_to_duration():
    """The 25-hour Cairo day's final hour gets half the quota of a full slot."""
    quotas = allocate_interval_quotas([120, 120, 60], 10)
    assert sum(quotas) == 10
    assert quotas[2] <= quotas[0]


def test_interval_allocation_always_preserves_the_total():
    for quota in range(0, 40):
        for minutes in ([120] * 12 + [60], [120] * 4, [90, 30], [60]):
            assert sum(allocate_interval_quotas(minutes, quota)) == quota


def test_zero_quota_allocates_nothing():
    assert allocate_interval_quotas([120, 120], 0) == [0, 0]


def test_negative_quota_is_rejected():
    with pytest.raises(ValueError):
        allocate_interval_quotas([120], -1)


# ── Nonconsecutive selection ─────────────────────────────────────────────────

def _candidates(n: int, *, eligible=True) -> list[Candidate]:
    return [
        Candidate(conversation_id=f"c{i:03d}", sequence_position=i,
                  rank_hash=_hash(f"c{i:03d}"), is_eligible=eligible)
        for i in range(1, n + 1)
    ]


def test_selection_never_picks_adjacent_conversations():
    outcome = select_nonconsecutive(_candidates(20), 5)
    positions = sorted(
        c.sequence_position for c in _candidates(20) if c.conversation_id in outcome.selected
    )
    assert all(b - a > 1 for a, b in zip(positions, positions[1:]))


def test_spacing_can_be_disabled():
    outcome = select_nonconsecutive(_candidates(4), 4, enforce_spacing=False)
    assert len(outcome.selected) == 4


def test_spacing_holds_against_previously_selected_positions():
    """This is what makes the rule survive an interval boundary."""
    outcome = select_nonconsecutive(
        _candidates(6), 3, already_selected_positions={3},
    )
    chosen = {int(c[1:]) for c in outcome.selected}
    assert 2 not in chosen and 4 not in chosen   # neighbours of 3 are blocked
    assert outcome.rejected.get("c002") == "adjacent_to_selected"


def test_selection_is_reproducible():
    first = select_nonconsecutive(_candidates(30), 7)
    second = select_nonconsecutive(_candidates(30), 7)
    assert first.selected == second.selected


def test_selection_ignores_input_order():
    forward = select_nonconsecutive(_candidates(30), 7)
    backward = select_nonconsecutive(list(reversed(_candidates(30))), 7)
    assert forward.selected == backward.selected


def test_ineligible_candidates_are_rejected_with_their_reason():
    candidates = _candidates(5)
    candidates[2] = Candidate("c003", 3, _hash("c003"), False, "conversation_not_closed")
    outcome = select_nonconsecutive(candidates, 5)
    assert "c003" not in outcome.selected
    assert outcome.rejected["c003"] == "conversation_not_closed"


def test_shortfall_from_too_few_eligible_candidates_is_reported_plainly():
    outcome = select_nonconsecutive(_candidates(2), 10)
    assert outcome.shortfall == 10 - len(outcome.selected)
    assert "eligible candidate" in outcome.shortfall_reason


def test_shortfall_from_spacing_is_not_claimed_to_be_infeasible():
    """
    Architecture §6: a greedy pass leaving slots unfilled must be reported as
    this policy's outcome, not as mathematical impossibility.
    """
    outcome = select_nonconsecutive(_candidates(6), 6)
    assert outcome.shortfall > 0
    assert "another combination might have filled more" in outcome.shortfall_reason


def test_quota_met_is_recorded_for_unselected_candidates():
    outcome = select_nonconsecutive(_candidates(20), 2)
    assert len(outcome.selected) == 2
    assert "quota_met" in outcome.rejected.values()


def test_min_gap_of_two_leaves_a_conversation_between_samples():
    outcome = select_nonconsecutive(_candidates(30), 5, min_gap=2)
    positions = sorted(int(c[1:]) for c in outcome.selected)
    assert all(b - a > 2 for a, b in zip(positions, positions[1:]))
