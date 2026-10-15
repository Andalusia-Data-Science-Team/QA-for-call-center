"""
All reads and writes against the local SQLite ledger.

Every other automation module goes through here rather than writing SQL, so
the transaction boundaries and the status transitions live in one place and
can be reasoned about as a whole.

The guarantees that matter are carried by the schema's constraints, not by
this code — see app/SQL/automation/sqlite/001_ledger.sql. What this module
adds is correct use of them: reservations inside one transaction, claims under
BEGIN IMMEDIATE, and treating an IntegrityError on the budget index as a
normal "someone else got there first" rather than an error.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from app.automation.db import next_fence_token, transaction, utc_text

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────

def get_run(conn: sqlite3.Connection, reporting_date: date, policy_version: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM QA_SamplingRuns "
        " WHERE ReportingDate = ? AND SamplingPolicyVersion = ? AND Status <> 'abandoned'",
        (reporting_date.isoformat(), policy_version),
    ).fetchone()


def create_run(
    conn: sqlite3.Connection,
    *,
    reporting_date: date,
    timezone_name: str,
    day_start_utc: datetime,
    day_end_utc: datetime,
    policy_version: str,
    base_seed: str,
    daily_target: int,
    interval_minutes: int,
    grace_minutes: int,
    exclude_adjacent: bool,
    min_gap_minutes: int,
    roster_version: Optional[str] = None,
    roster_agent_count: Optional[int] = None,
) -> str:
    run_id = str(uuid.uuid4())
    now = utc_text()
    conn.execute(
        """
        INSERT INTO QA_SamplingRuns
            (RunId, ReportingDate, ReportingTimezone, DayStartUtc, DayEndUtc,
             SamplingPolicyVersion, BaseSeed, DailyTarget, SamplingIntervalMinutes,
             SourceGraceMinutes, ExcludeAdjacentChats, MinChatGapMinutes,
             RosterVersion, RosterAgentCount, Status, PlannedAtUtc,
             CreatedAtUtc, UpdatedAtUtc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """,
        (run_id, reporting_date.isoformat(), timezone_name,
         utc_text(day_start_utc), utc_text(day_end_utc),
         policy_version, base_seed, daily_target, interval_minutes,
         grace_minutes, int(exclude_adjacent), min_gap_minutes,
         roster_version, roster_agent_count, now, now, now),
    )
    return run_id


def set_run_status(conn: sqlite3.Connection, run_id: str, status: str, **fields) -> None:
    assignments = ["Status = ?", "UpdatedAtUtc = ?"]
    params: list = [status, utc_text()]
    for column, value in fields.items():
        assignments.append(f"{column} = ?")
        params.append(value)
    params.append(run_id)
    conn.execute(f"UPDATE QA_SamplingRuns SET {', '.join(assignments)} WHERE RunId = ?", params)


# ─────────────────────────────────────────────────────────────────────────────
# Quotas and intervals
# ─────────────────────────────────────────────────────────────────────────────

def insert_agent_quota(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    agent_email: str,
    work_start_utc: datetime,
    work_end_utc: datetime,
    quota: int,
    remainder_rank: Optional[int] = None,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO QA_AgentQuotas
            (RunId, AgentEmailAddress, AgentId, AgentName, WorkStartUtc, WorkEndUtc,
             OriginalQuota, FinalQuota, RemainderRank, CreatedAtUtc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, agent_email, agent_id, agent_name,
         utc_text(work_start_utc), utc_text(work_end_utc),
         quota, quota, remainder_rank, utc_text()),
    )


def insert_interval(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    agent_email: str,
    interval_id: str,
    index: int,
    start_utc: datetime,
    end_utc: datetime,
    cutoff_utc: datetime,
    quota: int,
) -> None:
    conn.execute(
        """
        INSERT INTO QA_SamplingIntervals
            (RunId, AgentEmailAddress, IntervalId, IntervalIndex,
             IntervalStartUtc, IntervalEndUtc, SnapshotCutoffUtc, Quota, CreatedAtUtc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, agent_email, interval_id, index,
         utc_text(start_utc), utc_text(end_utc), utc_text(cutoff_utc), quota, utc_text()),
    )


def due_intervals(conn: sqlite3.Connection, now: Optional[datetime] = None) -> list[sqlite3.Row]:
    """
    Every interval whose snapshot cutoff has passed and which is not yet
    terminal, oldest first.

    No time lower bound: an interval missed while the scheduler was down is
    simply still due, which is what makes downtime self-healing rather than a
    permanent hole in the day.
    """
    return conn.execute(
        """
        SELECT i.*, r.SamplingPolicyVersion, r.BaseSeed, r.ReportingDate,
               r.ExcludeAdjacentChats, r.MinChatGapMinutes, r.DailyTarget
          FROM QA_SamplingIntervals i
          JOIN QA_SamplingRuns r ON r.RunId = i.RunId
         WHERE i.Status IN ('pending', 'snapshotting')
           AND i.SnapshotCutoffUtc <= ?
           AND r.Status IN ('active', 'reconciling')
         ORDER BY i.SnapshotCutoffUtc, i.AgentEmailAddress
        """,
        (utc_text(now),),
    ).fetchall()


def set_interval_status(conn: sqlite3.Connection, interval_row_id: int, status: str, **fields) -> None:
    assignments = ["Status = ?"]
    params: list = [status]
    for column, value in fields.items():
        assignments.append(f"{column} = ?")
        params.append(value)
    params.append(interval_row_id)
    conn.execute(
        f"UPDATE QA_SamplingIntervals SET {', '.join(assignments)} WHERE IntervalRowId = ?",
        params,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candidates and sequence positions
# ─────────────────────────────────────────────────────────────────────────────

def existing_sequence_positions(conn: sqlite3.Connection, run_id: str, agent_email: str) -> dict[str, int]:
    """
    Sequence positions already frozen for this agent in this run.

    Reused rather than recomputed so that a conversation keeps the position it
    was given when first seen — late arrivals must not renumber committed
    selections (architecture §6).
    """
    return {
        row["ConversationId"]: row["SequencePosition"]
        for row in conn.execute(
            "SELECT DISTINCT ConversationId, SequencePosition FROM QA_SamplingCandidates "
            " WHERE RunId = ? AND AgentEmailAddress = ?",
            (run_id, agent_email),
        )
    }


def selected_positions(conn: sqlite3.Connection, run_id: str, agent_email: str) -> set[int]:
    """
    Positions already selected for this agent, including earlier intervals.

    This is what makes the nonconsecutive rule hold across interval boundaries.
    Midnight is handled by the caller, which also passes the previous run's
    trailing positions.
    """
    return {
        row[0] for row in conn.execute(
            "SELECT SequencePosition FROM QA_SamplingCandidates "
            " WHERE RunId = ? AND AgentEmailAddress = ? AND IsSelected = 1",
            (run_id, agent_email),
        )
    }


def trailing_positions_from_previous_run(
    conn: sqlite3.Connection, agent_email: str, previous_date: date
) -> set[int]:
    """Selections at the tail of the previous reporting day, for midnight spacing."""
    return {
        row[0] for row in conn.execute(
            """
            SELECT c.SequencePosition
              FROM QA_SamplingCandidates c
              JOIN QA_SamplingRuns r ON r.RunId = c.RunId
             WHERE c.AgentEmailAddress = ? AND r.ReportingDate = ? AND c.IsSelected = 1
            """,
            (agent_email, previous_date.isoformat()),
        )
    }


def insert_candidates(conn: sqlite3.Connection, rows: Sequence[dict]) -> int:
    conn.executemany(
        """
        INSERT OR IGNORE INTO QA_SamplingCandidates
            (SnapshotId, ConversationId, RunId, AgentEmailAddress, IntervalId,
             AttributionRule, HandlingTimeUtc, SequencePosition, RankHash,
             IsEligible, EligibilityReason, IsSelected, SelectionReason,
             ConversationState, SourceLastUpdatedUtc, IsLateArrival, CreatedAtUtc)
        VALUES
            (:SnapshotId, :ConversationId, :RunId, :AgentEmailAddress, :IntervalId,
             :AttributionRule, :HandlingTimeUtc, :SequencePosition, :RankHash,
             :IsEligible, :EligibilityReason, :IsSelected, :SelectionReason,
             :ConversationState, :SourceLastUpdatedUtc, :IsLateArrival, :CreatedAtUtc)
        """,
        rows,
    )
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Jobs
# ─────────────────────────────────────────────────────────────────────────────

def reserved_count(conn: sqlite3.Connection, run_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM QA_EvaluationJobs WHERE RunId = ? AND Status <> 'superseded'",
        (run_id,),
    ).fetchone()[0]


def reserve_jobs(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    snapshot_id: str,
    agent_email: str,
    interval_id: str,
    policy_version: str,
    reporting_date: date,
    conversation_ids: Sequence[str],
    daily_target: int,
) -> tuple[list[str], list[str]]:
    """
    Reserve jobs for the selected conversations, respecting the daily budget.

    Returns ``(reserved_job_ids, skipped_conversation_ids)``.

    Two protections, both deliberately not relying on application logic alone:

    * The budget is re-checked against the live reserved count inside the
      caller's transaction, so two dispatchers cannot both see room for the
      last slot.
    * A conversation already reserved under this policy version raises
      IntegrityError on UX_QA_Jobs_Conv_Policy. That is caught and treated as
      "already taken" rather than as a failure — it is the normal outcome of a
      reconciliation pass meeting a conversation a regular pass already took.
    """
    reserved: list[str] = []
    skipped: list[str] = []
    remaining = daily_target - reserved_count(conn, run_id)

    for conversation_id in conversation_ids:
        if remaining <= 0:
            skipped.append(conversation_id)
            continue
        job_id = str(uuid.uuid4())
        try:
            conn.execute(
                """
                INSERT INTO QA_EvaluationJobs
                    (JobId, RunId, SnapshotId, ConversationId, AgentEmailAddress,
                     IntervalId, SamplingPolicyVersion, ReportingDate, Status, CreatedAtUtc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (job_id, run_id, snapshot_id, conversation_id, agent_email,
                 interval_id, policy_version, reporting_date.isoformat(), utc_text()),
            )
        except sqlite3.IntegrityError:
            skipped.append(conversation_id)
            continue
        reserved.append(job_id)
        remaining -= 1

    return reserved, skipped


def claim_job(
    conn: sqlite3.Connection,
    *,
    owner: str,
    lease_minutes: int,
    now: Optional[datetime] = None,
) -> Optional[sqlite3.Row]:
    """
    Atomically claim one claimable job, or return None.

    SQLite has neither ``UPDATE ... OUTPUT`` nor ``READPAST``, so this is a
    read-modify-write under BEGIN IMMEDIATE — which takes the write lock up
    front. The default deferred transaction would let two workers read the
    same row and have one fail on upgrade; taking the lock immediately turns
    that race into a short wait governed by busy_timeout.
    """
    moment = now or datetime.now(UTC)
    with transaction(conn):
        row = conn.execute(
            """
            SELECT JobId FROM QA_EvaluationJobs
             WHERE Status IN ('pending', 'retry_wait')
               AND (NextAttemptUtc IS NULL OR NextAttemptUtc <= ?)
             ORDER BY CreatedAtUtc, JobId
             LIMIT 1
            """,
            (utc_text(moment),),
        ).fetchone()
        if row is None:
            return None

        fence = next_fence_token(conn)
        conn.execute(
            """
            UPDATE QA_EvaluationJobs
               SET Status = 'running', LeaseOwner = ?, LeaseToken = ?,
                   LeaseExpiresUtc = ?, Attempts = Attempts + 1,
                   FenceToken = ?, StartedAtUtc = ?
             WHERE JobId = ?
            """,
            (owner, str(uuid.uuid4()),
             utc_text(moment + timedelta(minutes=lease_minutes)),
             fence, utc_text(moment), row["JobId"]),
        )
        return conn.execute("SELECT * FROM QA_EvaluationJobs WHERE JobId = ?", (row["JobId"],)).fetchone()


def heartbeat(conn: sqlite3.Connection, job_id: str, fence_token: int, lease_minutes: int) -> bool:
    """
    Extend a lease. Returns False if the worker has been superseded.

    The fence check is the whole point: a worker whose lease expired and whose
    job was reclaimed holds a lower token than the row now carries, and must
    stop rather than keep working and overwrite its replacement.
    """
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE QA_EvaluationJobs SET LeaseExpiresUtc = ? "
            " WHERE JobId = ? AND FenceToken = ? AND Status = 'running'",
            (utc_text(datetime.now(UTC) + timedelta(minutes=lease_minutes)), job_id, fence_token),
        )
        return cursor.rowcount > 0


def complete_job(conn: sqlite3.Connection, job_id: str, fence_token: int, **provenance) -> bool:
    """Mark a job completed, if this worker still holds the fence."""
    assignments = ["Status = 'completed'", "CompletedAtUtc = ?", "LastError = NULL"]
    params: list = [utc_text()]
    for column, value in provenance.items():
        assignments.append(f"{column} = ?")
        params.append(value)
    params.extend([job_id, fence_token])
    with transaction(conn):
        cursor = conn.execute(
            f"UPDATE QA_EvaluationJobs SET {', '.join(assignments)} "
            f" WHERE JobId = ? AND FenceToken = ?",
            params,
        )
        return cursor.rowcount > 0


def fail_job(
    conn: sqlite3.Connection,
    job_id: str,
    fence_token: int,
    *,
    error: str,
    failure_class: str,
    max_attempts: int,
    backoff_base_seconds: float = 30.0,
) -> str:
    """
    Record a failure and decide whether the job retries.

    Transient failures back off exponentially and become claimable again.
    Provider credit exhaustion waits without consuming a job attempt.
    Permanent and input failures go straight to 'failed' — retrying a
    malformed transcript just burns the same error three times. Exhausting
    attempts also means 'failed'.

    Returns the status the job ended up in.
    """
    with transaction(conn):
        row = conn.execute(
            "SELECT Attempts FROM QA_EvaluationJobs WHERE JobId = ? AND FenceToken = ?",
            (job_id, fence_token),
        ).fetchone()
        if row is None:
            logger.warning("fail_job | job %s: fence token stale, another worker owns it", job_id)
            return "superseded"

        attempts = row["Attempts"]
        if failure_class == "quota":
            conn.execute(
                "UPDATE QA_EvaluationJobs SET Status = 'retry_wait', NextAttemptUtc = ?, "
                "       Attempts = MAX(0, Attempts - 1), LastError = ?, FailureClass = 'transient', "
                "       LeaseOwner = NULL, LeaseExpiresUtc = NULL "
                " WHERE JobId = ? AND FenceToken = ?",
                (utc_text(datetime.now(UTC) + timedelta(minutes=5)),
                 error[:2000], job_id, fence_token),
            )
            return "quota_wait"

        retryable = failure_class == "transient" and attempts < max_attempts

        if retryable:
            delay = backoff_base_seconds * (2 ** (attempts - 1))
            conn.execute(
                "UPDATE QA_EvaluationJobs SET Status = 'retry_wait', NextAttemptUtc = ?, "
                "       LastError = ?, FailureClass = ?, LeaseOwner = NULL, LeaseExpiresUtc = NULL "
                " WHERE JobId = ?",
                (utc_text(datetime.now(UTC) + timedelta(seconds=delay)), error[:2000], failure_class, job_id),
            )
            return "retry_wait"

        conn.execute(
            "UPDATE QA_EvaluationJobs SET Status = 'failed', LastError = ?, FailureClass = ?, "
            "       CompletedAtUtc = ?, LeaseOwner = NULL, LeaseExpiresUtc = NULL "
            " WHERE JobId = ?",
            (error[:2000], failure_class, utc_text(), job_id),
        )
        return "failed"


def recover_expired_leases(conn: sqlite3.Connection, now: Optional[datetime] = None) -> int:
    """
    Return jobs whose worker died to the claimable pool.

    Safe because of the fence token: the dead worker, if it ever wakes, holds
    a stale token and every write it attempts is rejected.
    """
    moment = now or datetime.now(UTC)
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE QA_EvaluationJobs SET Status = 'retry_wait', NextAttemptUtc = ?, "
            "       LeaseOwner = NULL, LeaseExpiresUtc = NULL, "
            "       LastError = 'lease expired; worker presumed dead' "
            " WHERE Status = 'running' AND LeaseExpiresUtc IS NOT NULL AND LeaseExpiresUtc <= ?",
            (utc_text(moment), utc_text(moment)),
        )
        recovered = cursor.rowcount
    if recovered:
        logger.warning("recover_expired_leases | returned %d job(s) to the queue", recovered)
    return recovered


# ─────────────────────────────────────────────────────────────────────────────
# Result links (the §2.2 protocol)
# ─────────────────────────────────────────────────────────────────────────────

def get_result_link(conn: sqlite3.Connection, job_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM QA_ResultLinks WHERE JobId = ?", (job_id,)).fetchone()


def begin_persist_attempt(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    call_id: str,
    pre_insert_max_version: int,
    attempt_started_utc: datetime,
) -> None:
    """
    Record the intent to insert, and COMMIT IT BEFORE the DWH insert happens.

    This row is the only evidence a retry will have that an insert may already
    have landed. Writing it afterwards would defeat the entire protocol.
    """
    with transaction(conn):
        existing = conn.execute("SELECT Status FROM QA_ResultLinks WHERE JobId = ?", (job_id,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO QA_ResultLinks (JobId, CallId, Status, PreInsertMaxVersion, AttemptStartedUtc) "
                "VALUES (?, ?, 'attempting', ?, ?)",
                (job_id, call_id, pre_insert_max_version, utc_text(attempt_started_utc)),
            )
        else:
            conn.execute(
                "UPDATE QA_ResultLinks SET Status = 'attempting', PreInsertMaxVersion = ?, "
                "       AttemptStartedUtc = ?, Attempts = Attempts + 1 WHERE JobId = ?",
                (pre_insert_max_version, utc_text(attempt_started_utc), job_id),
            )


def complete_persist(
    conn: sqlite3.Connection,
    *,
    job_id: str,
    analysis_version: int,
    rows_inserted: Optional[int],
    method: str,
    notes: Optional[str] = None,
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE QA_ResultLinks SET Status = 'linked', AnalysisVersion = ?, RowsInserted = ?, "
            "       LinkedAtUtc = ?, LinkMethod = ?, Notes = COALESCE(?, Notes) WHERE JobId = ?",
            (analysis_version, rows_inserted, utc_text(), method, notes, job_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def run_coverage(conn: sqlite3.Connection, run_id: str) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM QA_RunCoverage WHERE RunId = ?", (run_id,)).fetchone()


def agent_coverage(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT q.AgentEmailAddress, q.OriginalQuota, q.FinalQuota,
               COUNT(j.JobId)                                           AS Reserved,
               SUM(CASE WHEN j.Status = 'completed' THEN 1 ELSE 0 END)  AS Completed
          FROM QA_AgentQuotas q
          LEFT JOIN QA_EvaluationJobs j
                 ON j.RunId = q.RunId AND j.AgentEmailAddress = q.AgentEmailAddress
                AND j.Status <> 'superseded'
         WHERE q.RunId = ?
         GROUP BY q.AgentEmailAddress, q.OriginalQuota, q.FinalQuota
         ORDER BY q.AgentEmailAddress
        """,
        (run_id,),
    ).fetchall()


def uncovered_agents(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Agents with quota who ended the day with nothing reserved."""
    return [
        row[0] for row in conn.execute(
            """
            SELECT q.AgentEmailAddress
              FROM QA_AgentQuotas q
             WHERE q.RunId = ? AND q.FinalQuota > 0
               AND NOT EXISTS (
                   SELECT 1 FROM QA_EvaluationJobs j
                    WHERE j.RunId = q.RunId AND j.AgentEmailAddress = q.AgentEmailAddress
                      AND j.Status <> 'superseded')
             ORDER BY q.AgentEmailAddress
            """,
            (run_id,),
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Repair
# ─────────────────────────────────────────────────────────────────────────────

def repair_run(conn: sqlite3.Connection, run_id: str) -> dict:
    """
    Undo the two states that make a run permanently stuck.

    Both were observed on the first live run and both are now prevented at
    source; this exists to recover ledgers written before those fixes, and as
    an operator escape hatch if a new way of getting stuck is ever found.

    * Intervals marked 'failed' by a transient DB error. due_intervals only
      returns 'pending'/'snapshotting', so a failed interval is never retried
      and its quota is lost for good.

    * A run closed before its intervals were dispatched. due_intervals also
      requires the run to be 'active' or 'reconciling', so a prematurely
      closed run makes every one of its intervals undispatchable.

    Returns what it changed. Intervals that failed for a non-transient reason
    are left alone — those are real errors and resetting them would just loop.
    """
    from app.CM.candidate_queries import is_transient

    failed = conn.execute(
        "SELECT IntervalRowId, ShortfallReason FROM QA_SamplingIntervals "
        " WHERE RunId = ? AND Status = 'failed'",
        (run_id,),
    ).fetchall()

    retryable = [
        row["IntervalRowId"] for row in failed
        if row["ShortfallReason"] and is_transient(Exception(row["ShortfallReason"]))
    ]

    reopened = False
    with transaction(conn):
        if retryable:
            conn.executemany(
                "UPDATE QA_SamplingIntervals SET Status = 'pending', "
                "       ShortfallReason = 'reset by repair after a transient failure' "
                " WHERE IntervalRowId = ?",
                [(row_id,) for row_id in retryable],
            )

        pending = conn.execute(
            "SELECT COUNT(*) FROM QA_SamplingIntervals "
            " WHERE RunId = ? AND Status IN ('pending', 'snapshotting')",
            (run_id,),
        ).fetchone()[0]

        run = conn.execute(
            "SELECT Status FROM QA_SamplingRuns WHERE RunId = ?", (run_id,)
        ).fetchone()
        if pending and run and run["Status"] == "closed":
            conn.execute(
                "UPDATE QA_SamplingRuns SET Status = 'active', ClosedAtUtc = NULL, "
                "       ShortfallReason = NULL, UpdatedAtUtc = ? WHERE RunId = ?",
                (utc_text(), run_id),
            )
            reopened = True

    return {
        "intervals_failed": len(failed),
        "intervals_reset": len(retryable),
        "intervals_left_failed": len(failed) - len(retryable),
        "run_reopened": reopened,
        "intervals_now_pending": pending,
    }
