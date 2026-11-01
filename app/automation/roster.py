"""
The agent roster: the chat agents in CM, not DWH.AI.Agents (see
fetch_active_agents for why).

There is no shift or working-period model. Sampling covers every active agent
across the whole reporting day, which is what the architecture asks for —
"covering the full 24-hour operating day".

WHY THERE IS NO SHIFT MODEL
───────────────────────────
There was one. It allocated each agent's quota across the hours they actually
worked, behind a `ShiftSource` interface with three implementations, a JSON
config file and a provenance column recording which guess had been used.

It never produced more than one answer. `DWH.AI.Agents` has no working-period
columns and we cannot add any, so there was no real shift feed to read; every
agent fell through to a whole-day default. The abstraction cost a config file,
two settings and a decision that could not be resolved, and bought nothing.

Removed. If a genuine shift feed ever appears, the place to reintroduce it is
`plan_day`, which is the only caller that cares about an agent's time bounds.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fetch_active_agents(cm_handler, *, day_start_utc, lookback_days: int) -> list[dict]:
    """
    The chat agents to plan a reporting day for, taken from CM itself.

    The roster is everyone who owned — and replied in — a conversation closed
    during the ``lookback_days`` before ``day_start_utc``. The window ends where
    the reporting day begins, so the roster is fixed before the day starts and
    is not shaped by that day's traffic.

    WHY NOT DWH.AI.Agents
    ─────────────────────
    It is not the list of people who work in CM. On 2026-10-17 only 3 of the
    18 inbox owners were in it, and the busiest CM agents were missing
    entirely; planning against it gave quota to 46 people, 43 of whom had no
    chats, and the day finished at 12 of 300. The manual dashboard already
    takes its agent list from CM (CM_users.sql), and NEWFIX attributes a chat
    to its inbox owner, so the roster comes from the same source.

    An agent with no closed conversations in the look-back window — a new
    starter, or someone back from leave — is not on that day's roster and gets
    no quota until the next day's plan picks them up.
    """
    from datetime import timedelta

    from app.CM.candidate_queries import fetch_cm_roster

    frame = fetch_cm_roster(
        cm_handler, day_start_utc - timedelta(days=lookback_days), day_start_utc,
    )
    agents = frame.to_dict(orient="records") if not frame.empty else []

    logger.info(
        "fetch_active_agents | %d CM agent(s) handled chats in the %d day(s) before %s",
        len(agents), lookback_days, day_start_utc.isoformat(),
    )
    if not agents:
        logger.error(
            "fetch_active_agents | no CM agent closed a conversation in the "
            "look-back window. Planning cannot allocate quota against an empty roster."
        )
    return agents


def agent_emails(agents) -> list[str]:
    """
    Deduplicated, sorted agent email addresses.

    Sorted so planning is deterministic regardless of the order the database
    happened to return rows in; the date-seeded ranking in
    app.automation.sampling then decides who receives the remainder.
    """
    seen = {
        str(a.get("Agent_Email_Address") or "").strip()
        for a in agents
        if str(a.get("Agent_Email_Address") or "").strip()
    }
    return sorted(seen)


def agent_details(agents) -> dict[str, dict]:
    """Email -> the agent's row, for stamping id and name onto quota records."""
    details: dict[str, dict] = {}
    for agent in agents:
        email = str(agent.get("Agent_Email_Address") or "").strip()
        if email:
            details.setdefault(email, agent)
    return details
