"""
Daily planning: freeze the roster, allocate quota, lay out the intervals.

Runs once per reporting day, before the first interval is due. Everything it
writes is frozen for the day: the roster, the per-agent quota, the interval
layout, the seed and the policy version. Later corrections are recorded as
reallocations rather than by recomputing, so a committed selection is never
retroactively justified by a different plan.

There is no shift model. Every active agent is sampled across the whole
reporting day, split into QA_SAMPLING_INTERVAL_MINUTES slices so that coverage
is spread over the full 24 hours rather than concentrated wherever traffic
happened to be heaviest. See app/automation/roster.py for why.

Idempotent. Planning a day that already has a run returns the existing run
rather than creating a second one — the unique index on
(ReportingDate, SamplingPolicyVersion) would reject it anyway, but returning
early makes a duplicate scheduler tick a no-op instead of an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from app.automation import repository as repo
from app.automation.calendar import reporting_day, snapshot_cutoff, split_intervals
from app.automation.config import AutomationSettings
from app.automation.db import transaction
from app.automation.roster import agent_details, agent_emails
from app.automation.sampling import agent_ranking, allocate_agent_quotas, allocate_interval_quotas

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlanResult:
    run_id: str
    reporting_date: date
    created: bool
    agent_count: int
    interval_count: int
    allocated: int
    target: int
    notes: list[str]


def plan_day(
    conn,
    *,
    reporting_date: date,
    settings: AutomationSettings,
    agents: Sequence[dict],
    roster_version: Optional[str] = None,
) -> PlanResult:
    """
    Create (or return) the frozen plan for one reporting day.

    ``agents`` is the roster as returned by roster.fetch_active_agents.
    """
    tz = settings.tz
    day = reporting_day(reporting_date, tz)
    notes: list[str] = []

    existing = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
    if existing is not None:
        logger.info("plan_day | run already exists for %s (%s)", reporting_date, existing["RunId"])
        return PlanResult(
            run_id=existing["RunId"], reporting_date=reporting_date, created=False,
            agent_count=existing["RosterAgentCount"] or 0, interval_count=0,
            allocated=0, target=existing["DailyTarget"], notes=["run already existed"],
        )

    emails = agent_emails(agents)
    if not emails:
        raise ValueError(
            f"No active agents for {reporting_date}. Without a roster there is "
            f"nothing to allocate quota against — this is a setup dependency, "
            f"not a reason to infer the day's agents from early traffic."
        )
    details = agent_details(agents)

    if len(emails) > settings.daily_target:
        notes.append(
            f"{len(emails)} active agents exceed the daily target of "
            f"{settings.daily_target}; {len(emails) - settings.daily_target} agent(s) "
            f"receive no quota today and must be covered by rotation"
        )
        logger.warning("plan_day | %s", notes[-1])

    quotas = allocate_agent_quotas(
        emails, settings.daily_target,
        base_seed=settings.sampling_seed, reporting_date=reporting_date,
    )
    ranking = {agent: i for i, agent in enumerate(
        agent_ranking(emails, base_seed=settings.sampling_seed, reporting_date=reporting_date)
    )}

    if day.is_irregular:
        notes.append(
            f"{reporting_date} is a {day.hours:g}-hour day in {settings.reporting_timezone} "
            f"(DST transition); the daily budget is unchanged and the interval "
            f"layout absorbs the difference"
        )
        logger.info("plan_day | %s", notes[-1])

    # The interval layout is the same for every agent — the whole reporting
    # day — so it is computed once rather than per agent.
    intervals = split_intervals(day.start_utc, day.end_utc, settings.sampling_interval_minutes)
    if not intervals:
        raise ValueError(f"Reporting day {reporting_date} produced no intervals")

    interval_count = 0
    allocated = 0

    with transaction(conn):
        run_id = repo.create_run(
            conn,
            reporting_date=reporting_date,
            timezone_name=settings.reporting_timezone,
            day_start_utc=day.start_utc,
            day_end_utc=day.end_utc,
            policy_version=settings.sampling_policy_version,
            base_seed=settings.sampling_seed,
            daily_target=settings.daily_target,
            interval_minutes=settings.sampling_interval_minutes,
            grace_minutes=settings.source_grace_minutes,
            exclude_adjacent=settings.exclude_adjacent_chats,
            min_gap_minutes=settings.min_chat_gap_minutes,
            roster_version=roster_version or "DWH.AI.Agents",
            roster_agent_count=len(emails),
        )

        for agent in emails:
            agent_quota = quotas[agent]
            row = details.get(agent, {})

            repo.insert_agent_quota(
                conn, run_id=run_id, agent_email=agent,
                work_start_utc=day.start_utc, work_end_utc=day.end_utc,
                quota=agent_quota, remainder_rank=ranking.get(agent),
                agent_id=_str_or_none(row.get("Agent_id")),
                agent_name=row.get("Agent_Name"),
            )

            interval_quotas = allocate_interval_quotas(
                [i.minutes for i in intervals], agent_quota,
                tie_break_seed=f"{settings.sampling_seed}|{agent}",
            )

            for interval, quota in zip(intervals, interval_quotas):
                repo.insert_interval(
                    conn, run_id=run_id, agent_email=agent,
                    interval_id=interval.interval_id, index=interval.index,
                    start_utc=interval.start_utc, end_utc=interval.end_utc,
                    cutoff_utc=snapshot_cutoff(interval, settings.source_grace_minutes),
                    quota=quota,
                )
                interval_count += 1
                allocated += quota

    logger.info(
        "plan_day | %s run=%s agents=%d intervals=%d allocated=%d/%d",
        reporting_date, run_id, len(emails), interval_count, allocated, settings.daily_target,
    )

    return PlanResult(
        run_id=run_id, reporting_date=reporting_date, created=True,
        agent_count=len(emails), interval_count=interval_count,
        allocated=allocated, target=settings.daily_target, notes=notes,
    )


def _str_or_none(value) -> Optional[str]:
    return None if value is None else str(value)
