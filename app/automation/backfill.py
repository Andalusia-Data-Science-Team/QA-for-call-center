"""
Deficit backfill: getting the day to its target.

THE PROBLEM THIS SOLVES
───────────────────────
Quota is allocated evenly across an agent's intervals, but real traffic is
not evenly distributed. An interval allotted 2 conversations that only saw 1
loses the other slot, and nothing in the per-interval pass can recover it.
Measured on simulated but realistic traffic, that lost 5% of the target on a
busy day and more than half of it on a thin one:

    20 agents x 60 chats   ->  284 / 300
    30 agents x 30 chats   ->  243 / 300
    25 agents x 12 chats   ->  135 / 300

Architecture §7 anticipates exactly this: "Underfilled interval quota may
first move to other eligible intervals for the same agent. After the day
closes, unused agent quota may be redistributed to other agents with
capacity, using deterministic rounds."

HOW IT WORKS
────────────
Entirely from the ledger. Every candidate ever considered is already stored —
including the rejected ones, with eligibility, rank hash and sequence
position — so backfill needs no new CM_DWH query at all. That is the payoff
for keeping rejected candidates rather than only selections.

  Round 1, per agent: fill each agent up to their OWN quota from their
      unselected eligible candidates across every interval of the day.
      Preserves the intended per-agent balance.

  Round 2, across agents: if the day is still short, hand out the remaining
      budget one conversation at a time, cycling through agents in the
      date-seeded deterministic order, so the extra coverage is spread rather
      than dumped on whichever agent happens to be busiest.

Both rounds keep every guarantee the first pass had: hash ordering, the
nonconsecutive rule against all existing selections, the one-job-per-
conversation index, and the daily cap.

WHAT IT WILL NOT DO
───────────────────
It never substitutes a conversation for one whose evaluation failed, and it
never relaxes spacing or eligibility to hit the number. A day that genuinely
lacks enough adequately-spaced eligible conversations finishes short, and the
shortfall is reported. Hitting 300 by lowering the bar would make the number
meaningless.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.automation import repository as repo
from app.automation.config import AutomationSettings
from app.automation.db import transaction, utc_text
from app.automation.sampling import Candidate, agent_ranking, select_nonconsecutive

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass(slots=True)
class BackfillReport:
    run_id: str
    reserved_before: int
    reserved_after: int
    target: int
    round_one_added: int = 0
    round_two_added: int = 0
    agents_topped_up: list[str] = field(default_factory=list)
    agents_over_quota: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def added(self) -> int:
        return self.reserved_after - self.reserved_before

    @property
    def deficit(self) -> int:
        return max(0, self.target - self.reserved_after)


def _spare_candidates(conn, run_id: str, agent: str) -> list[Candidate]:
    """
    An agent's eligible, unselected, unreserved candidates for the day.

    Excludes anything already reserved under this policy version, including by
    another run, so the unique index is not relied on to catch the common case.
    One row per conversation: a conversation seen in two interval snapshots
    appears twice otherwise, and the lowest rank hash is the one the first pass
    would have used.
    """
    rows = conn.execute(
        """
        SELECT c.ConversationId, c.SequencePosition, MIN(c.RankHash) AS RankHash
          FROM QA_SamplingCandidates c
          JOIN QA_SamplingRuns r ON r.RunId = c.RunId
         WHERE c.RunId = ?
           AND c.AgentEmailAddress = ?
           AND c.IsEligible = 1
           AND c.IsSelected = 0
           AND NOT EXISTS (
                 SELECT 1 FROM QA_EvaluationJobs j
                  WHERE j.ConversationId = c.ConversationId
                    AND j.SamplingPolicyVersion = r.SamplingPolicyVersion
                    AND j.Status <> 'superseded')
         GROUP BY c.ConversationId, c.SequencePosition
        """,
        (run_id, agent),
    ).fetchall()
    return [
        Candidate(
            conversation_id=row["ConversationId"],
            sequence_position=row["SequencePosition"],
            rank_hash=row["RankHash"],
            is_eligible=True,
        )
        for row in rows
    ]


def _take(
    conn,
    *,
    run_id: str,
    agent: str,
    want: int,
    policy_version: str,
    reporting_date: date,
    daily_target: int,
    enforce_spacing: bool,
) -> int:
    """
    Reserve up to *want* more conversations for one agent. Returns how many.

    Spacing is checked against every position already selected for this agent,
    re-read each call so successive rounds see each other's work.
    """
    if want <= 0:
        return 0

    spare = _spare_candidates(conn, run_id, agent)
    if not spare:
        return 0

    taken = repo.selected_positions(conn, run_id, agent)
    taken |= repo.trailing_positions_from_previous_run(
        conn, agent, reporting_date - timedelta(days=1)
    )

    outcome = select_nonconsecutive(
        spare, want, already_selected_positions=taken, enforce_spacing=enforce_spacing,
    )
    if not outcome.selected:
        return 0

    snapshot_id = str(uuid.uuid4())
    reserved, _skipped = repo.reserve_jobs(
        conn,
        run_id=run_id, snapshot_id=snapshot_id, agent_email=agent,
        interval_id="backfill", policy_version=policy_version,
        reporting_date=reporting_date,
        conversation_ids=outcome.selected, daily_target=daily_target,
    )

    if reserved:
        placeholders = ", ".join("?" for _ in outcome.selected[:len(reserved)])
        conn.execute(
            f"UPDATE QA_SamplingCandidates "
            f"   SET IsSelected = 1, SelectionReason = 'backfill' "
            f" WHERE RunId = ? AND AgentEmailAddress = ? "
            f"   AND ConversationId IN ({placeholders})",
            [run_id, agent, *outcome.selected[:len(reserved)]],
        )
    return len(reserved)


def backfill_run(
    conn,
    settings: AutomationSettings,
    *,
    reporting_date: date,
    allow_cross_agent: bool = True,
) -> Optional[BackfillReport]:
    """Top the day up towards its target, without lowering any standard."""
    run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
    if run is None:
        return None

    run_id = run["RunId"]
    target = run["DailyTarget"]
    enforce_spacing = bool(run["ExcludeAdjacentChats"])

    before = repo.reserved_count(conn, run_id)
    report = BackfillReport(
        run_id=run_id, reserved_before=before, reserved_after=before, target=target,
    )
    if before >= target:
        report.notes.append("already at target; nothing to backfill")
        return report

    quotas = {
        row["AgentEmailAddress"]: (row["FinalQuota"], row["Reserved"])
        for row in repo.agent_coverage(conn, run_id)
    }

    # ── Round 1: each agent up to their own quota ───────────────────────────
    with transaction(conn):
        for agent in sorted(quotas):
            quota, reserved = quotas[agent]
            gained = _take(
                conn, run_id=run_id, agent=agent, want=quota - reserved,
                policy_version=run["SamplingPolicyVersion"],
                reporting_date=reporting_date, daily_target=target,
                enforce_spacing=enforce_spacing,
            )
            if gained:
                report.round_one_added += gained
                report.agents_topped_up.append(agent)

    report.reserved_after = repo.reserved_count(conn, run_id)
    logger.info(
        "backfill | round 1 (within quota) added %d; now %d/%d",
        report.round_one_added, report.reserved_after, target,
    )

    # ── Round 2: spread the remainder across agents with capacity ───────────
    if allow_cross_agent and report.reserved_after < target:
        order = agent_ranking(
            sorted(quotas), base_seed=run["BaseSeed"], reporting_date=reporting_date,
        )
        exhausted: set[str] = set()

        with transaction(conn):
            while report.reserved_after < target and len(exhausted) < len(order):
                progressed = False
                for agent in order:
                    if report.reserved_after >= target:
                        break
                    if agent in exhausted:
                        continue
                    gained = _take(
                        conn, run_id=run_id, agent=agent, want=1,
                        policy_version=run["SamplingPolicyVersion"],
                        reporting_date=reporting_date, daily_target=target,
                        enforce_spacing=enforce_spacing,
                    )
                    if gained:
                        report.round_two_added += gained
                        report.reserved_after += gained
                        progressed = True
                        if agent not in report.agents_over_quota:
                            report.agents_over_quota.append(agent)
                    else:
                        exhausted.add(agent)
                if not progressed:
                    break

        logger.info(
            "backfill | round 2 (cross-agent) added %d; now %d/%d",
            report.round_two_added, report.reserved_after, target,
        )

    report.reserved_after = repo.reserved_count(conn, run_id)

    # Record the reallocation so the audit shows who ended up over their
    # original quota and why.
    if report.agents_over_quota:
        with transaction(conn):
            for agent in report.agents_over_quota:
                conn.execute(
                    "UPDATE QA_AgentQuotas SET ReallocationReason = ? "
                    " WHERE RunId = ? AND AgentEmailAddress = ?",
                    ("received redistributed quota from agents whose intervals "
                     "had too few eligible conversations", run_id, agent),
                )

    if report.deficit:
        report.notes.append(
            f"still {report.deficit} short of {target} after backfill: the day "
            f"did not contain enough eligible, adequately-spaced conversations. "
            f"Spacing and eligibility were NOT relaxed to close the gap."
        )

    logger.info(
        "backfill | %s %d -> %d of %d (round1=%d round2=%d deficit=%d)",
        reporting_date, before, report.reserved_after, target,
        report.round_one_added, report.round_two_added, report.deficit,
    )
    return report
