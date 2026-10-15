"""
Deterministic sampling: ranking, allocation and the nonconsecutive rule.

Everything here is a pure function of its arguments. No clock, no database, no
configuration lookup, no randomness beyond the hash. That is deliberate: the
acceptance criterion is that frozen inputs produce byte-identical selections
across processes and restarts, and the only reliable way to get that is for
the selection logic to have no hidden inputs at all.

THE CANONICAL SERIALISATION IS FIXED HERE AND NOWHERE ELSE.
Changing the field order, the separator, the case of a UUID or the date format
changes every future selection. Any such change requires a new
QA_SAMPLING_POLICY_VERSION, which is why the version is an input to the hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

__all__ = [
    "Candidate",
    "SelectionOutcome",
    "rank_hash",
    "agent_ranking",
    "allocate_agent_quotas",
    "allocate_interval_quotas",
    "select_nonconsecutive",
]

_SEPARATOR = "|"


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def _canonical(value) -> str:
    """
    Canonical text for one hash field.

    UUIDs are lowercased because CM_DWH and pyodbc disagree on case, and a
    selection that depended on which one produced the string would not be
    reproducible. Dates are ISO. Everything else is str() with surrounding
    whitespace removed.
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    text = str(value).strip()
    # A UUID in any case folds to one canonical form.
    if len(text) == 36 and text.count("-") == 4:
        return text.lower()
    return text


def rank_hash(
    *,
    base_seed: str,
    policy_version: str,
    reporting_date: date,
    shift_id: str,
    agent_id: str,
    interval_id: str,
    conversation_id: str,
) -> str:
    """
    The deterministic ranking score for one candidate.

    SHA-256 of the canonically serialised field list, in this exact order.
    Never ``hash()``: Python salts it per process, so a restart would reshuffle
    every selection and reproducibility would be silently lost.

    Including the agent, interval and conversation means each agent's sample is
    independent of every other agent's, while one shared base seed keeps the
    whole day reproducible from a single recorded value.
    """
    payload = _SEPARATOR.join(
        _canonical(v) for v in (
            base_seed,
            policy_version,
            reporting_date,
            shift_id,
            agent_id,
            interval_id,
            conversation_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def agent_ranking(agents: Sequence[str], *, base_seed: str, reporting_date: date) -> list[str]:
    """
    A date-seeded deterministic ordering of agents.

    Used to distribute the ``target mod A`` remainder. Seeding by date means
    the same agents do not receive the extra conversation every day, while the
    order stays reproducible for any given day.
    """
    return sorted(
        agents,
        key=lambda a: (
            hashlib.sha256(
                _SEPARATOR.join(
                    _canonical(v) for v in (base_seed, reporting_date, "agent-rank", a)
                ).encode("utf-8")
            ).hexdigest(),
            _canonical(a),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Allocation
# ─────────────────────────────────────────────────────────────────────────────

def allocate_agent_quotas(
    agents: Sequence[str],
    target: int,
    *,
    base_seed: str,
    reporting_date: date,
) -> dict[str, int]:
    """
    Split the daily target equally across agents, distributing the remainder
    deterministically.

    Equal allocation is the default because the objective is agent coverage,
    not traffic-weighted monitoring (architecture §5). With 20 agents and a
    target of 300 every agent gets 15.

    When there are more agents than the target allows, each agent that can be
    covered gets 1 and the rest get 0 — the caller is responsible for rotating
    coverage across days and reporting who was left out. Returning zeros rather
    than raising keeps that an operational fact rather than a crash.
    """
    if target < 0:
        raise ValueError(f"target must be non-negative, got {target}")
    if not agents:
        return {}

    ranked = agent_ranking(agents, base_seed=base_seed, reporting_date=reporting_date)
    base, remainder = divmod(target, len(ranked))
    return {
        agent: base + (1 if position < remainder else 0)
        for position, agent in enumerate(ranked)
    }


def allocate_interval_quotas(
    interval_minutes: Sequence[int],
    quota: int,
    *,
    tie_break_seed: str = "",
) -> list[int]:
    """
    Split an agent's quota across their intervals, proportional to duration,
    by the largest-remainder method.

    An eight-hour shift with quota 15 over four two-hour intervals yields
    4, 4, 4, 3 — the shortfall lands on one interval rather than being spread
    as fractions. Ties in the remainder are broken by a seeded hash of the
    interval index, not by position, so the same interval does not always lose.

    Durations are in *elapsed* minutes, which is why a DST-straddling interval
    is still 120: see app.automation.calendar.split_intervals.
    """
    if quota < 0:
        raise ValueError(f"quota must be non-negative, got {quota}")
    if not interval_minutes:
        return []
    total = sum(interval_minutes)
    if total <= 0:
        raise ValueError("Total interval duration must be positive")

    exact = [quota * m / total for m in interval_minutes]
    floors = [int(e) for e in exact]
    leftover = quota - sum(floors)

    if leftover:
        order = sorted(
            range(len(exact)),
            key=lambda i: (
                -(exact[i] - floors[i]),                     # biggest remainder first
                hashlib.sha256(f"{tie_break_seed}|{i}".encode("utf-8")).hexdigest(),
            ),
        )
        for i in order[:leftover]:
            floors[i] += 1

    assert sum(floors) == quota, "largest-remainder allocation must preserve the total"
    return floors


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Candidate:
    """One conversation competing for a slot."""

    conversation_id: str
    sequence_position: int
    rank_hash: str
    is_eligible: bool = True
    ineligible_reason: Optional[str] = None


@dataclass(slots=True)
class SelectionOutcome:
    """
    What the selection actually did, including why it fell short.

    ``shortfall`` and ``shortfall_reason`` are first-class rather than inferred:
    a shortfall is an operational fact to report, and the architecture is
    explicit that a greedy pass leaving slots unfilled must not be described as
    mathematical infeasibility.
    """

    selected: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)   # conversation_id -> reason
    shortfall: int = 0
    shortfall_reason: Optional[str] = None


def select_nonconsecutive(
    candidates: Iterable[Candidate],
    quota: int,
    *,
    already_selected_positions: Iterable[int] = (),
    enforce_spacing: bool = True,
    min_gap: int = 1,
) -> SelectionOutcome:
    """
    Pick up to ``quota`` candidates in hash order, skipping any that sit next
    to an already-selected conversation.

    The spacing rule is defined over ``sequence_position``, which numbers ALL
    conversations the agent handled in the shift — including ones ineligible
    for evaluation. "Not back to back" means the agent did not have two
    consecutive conversations sampled, and that is a fact about what they
    handled, not about what happened to be eligible.

    ``already_selected_positions`` carries selections committed in earlier
    intervals and on the previous day, which is how the rule holds across
    interval and midnight boundaries.

    ``min_gap`` of 1 forbids adjacency; 2 would require a gap of one
    conversation between samples, and so on.

    Greedy, and knowingly so. Walking in hash order and taking what fits can
    leave a slot unfilled where some other combination would have filled it.
    That is reported as a shortfall rather than hidden, because changing to a
    combinatorial search would break the property that matters more: that the
    same inputs always produce the same selection.
    """
    if quota < 0:
        raise ValueError(f"quota must be non-negative, got {quota}")

    taken = set(already_selected_positions)
    outcome = SelectionOutcome()

    # Hash order, with the conversation id as a deterministic tie-break.
    ordered = sorted(candidates, key=lambda c: (c.rank_hash, _canonical(c.conversation_id)))

    for candidate in ordered:
        if len(outcome.selected) >= quota:
            outcome.rejected[candidate.conversation_id] = "quota_met"
            continue

        if not candidate.is_eligible:
            outcome.rejected[candidate.conversation_id] = (
                candidate.ineligible_reason or "ineligible"
            )
            continue

        if enforce_spacing and any(
            abs(candidate.sequence_position - position) <= min_gap for position in taken
        ):
            outcome.rejected[candidate.conversation_id] = "adjacent_to_selected"
            continue

        outcome.selected.append(candidate.conversation_id)
        taken.add(candidate.sequence_position)

    outcome.shortfall = quota - len(outcome.selected)
    if outcome.shortfall > 0:
        eligible = sum(1 for c in ordered if c.is_eligible)
        if eligible < quota:
            outcome.shortfall_reason = (
                f"only {eligible} eligible candidate(s) for a quota of {quota}"
            )
        else:
            blocked = sum(1 for r in outcome.rejected.values() if r == "adjacent_to_selected")
            outcome.shortfall_reason = (
                f"{eligible} eligible candidate(s) for a quota of {quota}, but the "
                f"spacing rule blocked {blocked} of them under this greedy pass; "
                f"another combination might have filled more"
            )
    return outcome
