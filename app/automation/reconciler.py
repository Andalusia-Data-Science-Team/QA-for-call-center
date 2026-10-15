"""
End-of-day reconciliation: fill feasible deficits, then close the run.

Runs once the reporting day has ended plus a grace period. Three jobs, in
order, because each depends on the one before:

  1. Recover anything stuck — expired leases back to the queue.
  2. Fill the deficit, if there is one and if it is feasible.
  3. Close the run and record the final accounting, including what was NOT
     achieved and why.

Step 3 happens even when steps 1 and 2 achieve nothing. A day that fell short
must still close with its shortfall recorded; leaving it open would make the
deficit invisible rather than reported.

WHAT RECONCILIATION DOES NOT DO
───────────────────────────────
It never substitutes an easier conversation for one whose evaluation failed.
Once 300 jobs are reserved, those are the 300: a failed job is retried, not
swapped out. Replacing failures with fresh conversations would turn a quality
signal into a selection bias, quietly dropping exactly the conversations the
pipeline finds hardest.
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

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass(slots=True)
class ReconcileReport:
    run_id: str
    reporting_date: date
    target: int
    reserved: int
    completed: int
    failed: int
    outstanding: int
    deficit: int
    recovered_leases: int = 0
    uncovered_agents: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    closed: bool = False


def reconcile_day(
    conn,
    settings: AutomationSettings,
    *,
    reporting_date: date,
    now: Optional[datetime] = None,
    close: bool = True,
) -> Optional[ReconcileReport]:
    """Reconcile and optionally close one reporting day's run."""
    moment = now or datetime.now(UTC)

    run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
    if run is None:
        logger.info("reconcile | no run for %s", reporting_date)
        return None

    recovered = repo.recover_expired_leases(conn, moment)

    undispatched = conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals "
        " WHERE RunId = ? AND Status IN ('pending', 'snapshotting')",
        (run["RunId"],),
    ).fetchone()[0]

    # Backfill only once every interval has reported. Running it earlier would
    # hand out slots that a not-yet-dispatched interval's own candidates should
    # take, and on a run with nothing dispatched it is pure noise — it can only
    # ever report "added 0", which reads like a data problem rather than the
    # "too early" that it is.
    filled = None
    if not undispatched:
        from app.automation.backfill import backfill_run
        filled = backfill_run(conn, settings, reporting_date=reporting_date)

    coverage = repo.run_coverage(conn, run["RunId"])
    uncovered = repo.uncovered_agents(conn, run["RunId"])

    report = ReconcileReport(
        run_id=run["RunId"],
        reporting_date=reporting_date,
        target=run["DailyTarget"],
        reserved=coverage["JobsReserved"] if coverage else 0,
        completed=coverage["JobsCompleted"] if coverage else 0,
        failed=coverage["JobsFailed"] if coverage else 0,
        outstanding=(coverage["JobsOutstanding"] + coverage["JobsAwaitingRetry"]) if coverage else 0,
        deficit=(run["DailyTarget"] - coverage["JobsReserved"]) if coverage else run["DailyTarget"],
        recovered_leases=recovered,
        uncovered_agents=uncovered,
    )

    if report.outstanding:
        report.notes.append(
            f"{report.outstanding} job(s) still pending or awaiting retry; "
            f"the run stays open so the worker can finish them"
        )
        close = False

    # A run whose intervals have not been dispatched yet must never be closed.
    #
    # The planner loop plans today AND yesterday (to recover a missed day) and
    # then reconciles yesterday. On the very first start that meant yesterday's
    # run was created and closed in the same pass, seconds apart, with zero
    # reserved — and since due_intervals only considers 'active' and
    # 'reconciling' runs, closing it made those 552 intervals permanently
    # undispatchable. Observed on the first live run.
    if undispatched:
        report.notes.append(
            f"{undispatched} interval(s) have not been dispatched yet; the run "
            f"stays open so the dispatcher can still process them"
        )
        close = False

    if filled and filled.added:
        report.notes.append(
            f"backfill added {filled.added} conversation(s) "
            f"({filled.round_one_added} within agent quota, "
            f"{filled.round_two_added} redistributed across agents)"
        )

    if report.deficit > 0:
        if undispatched:
            # Saying "the day did not produce enough conversations" here would
            # be a false diagnosis: nothing has been looked at yet.
            report.notes.append(
                f"reserved {report.reserved} of {report.target} so far — "
                f"{undispatched} interval(s) are still waiting for the "
                f"dispatcher, so no conclusion about the day's volume can be "
                f"drawn yet"
            )
        else:
            report.notes.append(
                f"reserved {report.reserved} of {report.target}: the day did not "
                f"produce enough eligible, adequately-spaced conversations under "
                f"policy {run['SamplingPolicyVersion']}"
            )

    if report.failed:
        report.notes.append(
            f"{report.failed} job(s) failed permanently and were NOT replaced "
            f"with substitute conversations — see QA_EvaluationJobs.LastError"
        )

    if uncovered and not undispatched:
        report.notes.append(
            f"{len(uncovered)} agent(s) with quota ended the day with nothing "
            f"reserved: {', '.join(uncovered[:5])}"
            + (" ..." if len(uncovered) > 5 else "")
        )

    if close:
        with transaction(conn):
            repo.set_run_status(
                conn, run["RunId"], "closed",
                ClosedAtUtc=utc_text(moment),
                ReservedCount=report.reserved,
                CompletedCount=report.completed,
                ShortfallCount=max(0, report.deficit),
                ShortfallReason="; ".join(report.notes)[:1000] or None,
            )
        report.closed = True

    logger.info(
        "reconcile | %s run=%s reserved=%d/%d completed=%d failed=%d outstanding=%d closed=%s",
        reporting_date, run["RunId"], report.reserved, report.target,
        report.completed, report.failed, report.outstanding, report.closed,
    )
    for note in report.notes:
        logger.info("reconcile | note: %s", note)
    return report


def day_is_reconcilable(
    conn, settings: AutomationSettings, reporting_date: date, now: Optional[datetime] = None
) -> bool:
    """
    Has the reporting day ended, plus grace?

    Uses the run's own stored DayEndUtc rather than recomputing it, so a
    tzdata change cannot move the boundary of a day that is already underway —
    which matters precisely because Africa/Cairo's transitions have moved
    before.
    """
    from app.automation.db import parse_utc

    run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
    if run is None:
        return False
    moment = now or datetime.now(UTC)
    return moment >= parse_utc(run["DayEndUtc"]) + timedelta(minutes=settings.source_grace_minutes)
