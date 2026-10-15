"""
The agent roster.

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


def fetch_active_agents(dwh_engine) -> list[dict]:
    """
    Active human agents from DWH.AI.Agents. Read-only.

    Takes the **DWH** engine, not the CM handler. The two live on different
    servers (CM_DWH is reached from the CM connection through a linked server;
    AI.Agents is on the DWH instance), and querying the agents table over the
    CM connection would depend on a linked server that may not exist.

    Agent_Email_Password is deliberately not selected. The automation has no
    use for it and it should not travel through this process.
    """
    from sqlalchemy import text as sa_text

    query = sa_text(
        "SELECT Agent_id, Agent_Name, Department, is_Active, "
        "       Agent_Email_Address, Level, Supervisor_Name "
        "  FROM [DWH].[AI].[Agents] "
        " WHERE is_Active = 1 AND Agent_Email_Address IS NOT NULL "
        " ORDER BY Agent_Email_Address"
    )
    with dwh_engine.connect() as conn:
        agents = [dict(row._mapping) for row in conn.execute(query)]

    logger.info("fetch_active_agents | %d active agent(s)", len(agents))
    if not agents:
        logger.error(
            "fetch_active_agents | DWH.AI.Agents returned no active agents. "
            "Planning cannot allocate quota against an empty roster."
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
