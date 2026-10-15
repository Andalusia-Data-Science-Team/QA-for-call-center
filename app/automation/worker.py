"""
The evaluation worker: claim a job, fetch its transcript, evaluate, persist.

The whole file is organised around one problem. We have no DDL rights on DWH,
so there is no QA_Job_Id column and no unique index to reject a retry's
duplicate insert. Idempotency has to come from the protocol in
docs/AUTOMATION_IMPLEMENTATION_PLAN.md §2.2, and the protocol only works if
its steps happen in the right order:

    check link -> read version -> COMMIT intent -> insert -> record outcome

Committing the intent before the insert is the part that is easy to get wrong
and impossible to detect in testing that never crashes. ``_persist_guard``
below is the only place that ordering is expressed.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.automation import repository as repo
from app.automation.config import AutomationSettings
from app.automation.db import parse_utc, transaction, utc_text
from app.models.input import CallTranscript
from app.services.evaluation_service import (
    EvaluationOutcome,
    evaluate_conversation,
    transcript_checksum,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


def worker_identity() -> str:
    """Host and pid, so an abandoned lease names the process that held it."""
    return f"{socket.gethostname()}:{os.getpid()}"


@dataclass(slots=True)
class WorkerReport:
    claimed: int = 0
    completed: int = 0
    recovered: int = 0
    failed: int = 0
    retried: int = 0
    skipped_already_done: int = 0
    quota_paused: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Crash recovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_orphaned_result(dwh_engine, call_id: str, link) -> Optional[dict]:
    """
    Did a previous attempt's insert actually land before the process died?

    Looks for a row with this Call_ID whose Analysis_Version is above the
    version recorded before the attempt AND whose Created_at is at or after
    the attempt started. Both conditions are needed: the version alone cannot
    distinguish our insert from a manual re-analysis, and the timestamp alone
    cannot distinguish it from a row that was already there.

    Returns the adopted row, or None if no insert landed.

    The residual risk is documented in the schema and in the plan: a manual
    re-analysis of the same Call_ID inside the same window would be adopted as
    ours. Call_QA_Results has no column we may add to disambiguate and
    Created_at has one-second resolution, so this is as tight as the window
    gets without DDL rights.
    """
    from sqlalchemy import text as sa_text

    with dwh_engine.connect() as conn:
        row = conn.execute(
            sa_text(
                "SELECT TOP 1 Analysis_Version, COUNT(*) OVER (PARTITION BY Analysis_Version) AS RowCountForVersion "
                "  FROM [DWH].[AI].[Call_QA_Results] "
                " WHERE Call_ID = :cid AND Analysis_Version > :prev AND Created_at >= :since "
                " ORDER BY Analysis_Version ASC"
            ),
            {
                "cid": call_id,
                "prev": link["PreInsertMaxVersion"],
                "since": parse_utc(link["AttemptStartedUtc"]).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ).fetchone()

    if row is None:
        return None
    return {"analysis_version": int(row[0]), "rows_inserted": int(row[1])}


def _current_max_version(dwh_engine, call_id: str) -> int:
    from sqlalchemy import text as sa_text

    with dwh_engine.connect() as conn:
        row = conn.execute(
            sa_text(
                "SELECT COALESCE(MAX(Analysis_Version), 0) FROM [DWH].[AI].[Call_QA_Results] "
                " WHERE Call_ID = :cid"
            ),
            {"cid": call_id},
        ).fetchone()
    return int(row[0]) if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# One job
# ─────────────────────────────────────────────────────────────────────────────

async def process_job(
    conn,
    cm_handler,
    dwh_engine,
    analyzer,
    job,
    settings: AutomationSettings,
    report: WorkerReport,
) -> None:
    """Run one claimed job all the way to a terminal status."""
    from app.CM.candidate_queries import fetch_transcripts_by_ids, rows_for_conversation

    job_id = job["JobId"]
    fence = job["FenceToken"]
    conversation_id = job["ConversationId"]

    # ── Step 1: has this job already been persisted? ────────────────────────
    link = repo.get_result_link(conn, job_id)
    if link is not None and link["Status"] == "linked":
        logger.info("worker | job %s already linked (version %s) — not re-evaluating",
                    job_id, link["AnalysisVersion"])
        repo.complete_job(conn, job_id, fence)
        report.skipped_already_done += 1
        return

    # ── Step 1b: did a previous attempt's insert land before we died? ───────
    if link is not None and link["Status"] == "attempting":
        try:
            adopted = _find_orphaned_result(dwh_engine, link["CallId"], link)
        except Exception as exc:                                   # noqa: BLE001
            logger.error("worker | job %s: recovery lookup failed: %s", job_id, exc)
            adopted = None

        if adopted:
            logger.warning(
                "worker | job %s: recovered an orphaned insert (version %d) from a "
                "previous attempt — adopting it instead of re-running inference",
                job_id, adopted["analysis_version"],
            )
            repo.complete_persist(
                conn, job_id=job_id,
                analysis_version=adopted["analysis_version"],
                rows_inserted=adopted["rows_inserted"],
                method="recovered",
                notes="adopted after crash between DWH insert and link write",
            )
            repo.complete_job(conn, job_id, fence)
            report.recovered += 1
            return

    # ── Step 2: retrieve the transcript ─────────────────────────────────────
    try:
        frame = fetch_transcripts_by_ids(cm_handler, [conversation_id])
        rows = rows_for_conversation(frame, conversation_id)
        if not rows:
            raise ValueError(f"no transcript rows returned for conversation {conversation_id}")
        call = CallTranscript.from_cm_rows(rows)
    except Exception as exc:                                       # noqa: BLE001
        from app.services.evaluation_service import classify_failure
        status = repo.fail_job(
            conn, job_id, fence,
            error=f"transcript retrieval: {exc}",
            failure_class=classify_failure(exc),
            max_attempts=settings.max_attempts,
        )
        _count(report, status)
        return

    # Freeze the input on first successful retrieval. A later attempt that
    # computes a different checksum is evaluating a conversation that changed
    # after it was selected, which the architecture requires be visible rather
    # than silently attached to a supposedly identical retry.
    checksum = transcript_checksum(call)
    if job["InputChecksum"] and job["InputChecksum"] != checksum:
        logger.warning(
            "worker | job %s: transcript changed since selection (checksum %s -> %s); "
            "evaluating the current version and recording the change",
            job_id, job["InputChecksum"][:12], checksum[:12],
        )
    with transaction(conn):
        conn.execute(
            "UPDATE QA_EvaluationJobs SET InputChecksum = ?, "
            "       InputFrozenAtUtc = COALESCE(InputFrozenAtUtc, ?) "
            " WHERE JobId = ?",
            (checksum, utc_text(), job_id),
        )

    # ── Step 3: commit the intent BEFORE the insert can happen ──────────────
    attempt_started = datetime.now(UTC)
    try:
        pre_version = _current_max_version(dwh_engine, call.call_id)
    except Exception as exc:                                       # noqa: BLE001
        status = repo.fail_job(
            conn, job_id, fence, error=f"version probe: {exc}",
            failure_class="transient", max_attempts=settings.max_attempts,
        )
        _count(report, status)
        return

    repo.begin_persist_attempt(
        conn, job_id=job_id, call_id=call.call_id,
        pre_insert_max_version=pre_version, attempt_started_utc=attempt_started,
    )

    # ── Step 4: evaluate (the graph persists as node 7) ─────────────────────
    outcome: EvaluationOutcome = await evaluate_conversation(analyzer, call, job_id=job_id)

    # LangGraph runs save_to_database in a child task. The committed insert's
    # ContextVar update may not propagate back to this worker task, even though
    # the DWH row exists. Read back using the intent we committed above before
    # treating a missing context outcome as a failed evaluation.
    link_method = "inserted"
    if (
        not outcome.persisted
        and outcome.result is not None
        and outcome.error == "evaluation succeeded but no DWH row was recorded"
    ):
        try:
            link = repo.get_result_link(conn, job_id)
            adopted = _find_orphaned_result(dwh_engine, call.call_id, link)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("worker | job %s: post-evaluation read-back failed: %s", job_id, exc)
            adopted = None
        if adopted:
            logger.info(
                "worker | job %s: confirmed committed DWH version %d by read-back",
                job_id, adopted["analysis_version"],
            )
            outcome = EvaluationOutcome(
                result=outcome.result, persisted=True,
                analysis_version=adopted["analysis_version"],
                rows_inserted=adopted["rows_inserted"],
            )
            link_method = "recovered"

    # ── Step 5: record what landed ──────────────────────────────────────────
    if outcome.persisted:
        repo.complete_persist(
            conn, job_id=job_id,
            analysis_version=outcome.analysis_version,
            rows_inserted=outcome.rows_inserted,
            method=link_method,
        )
        if repo.complete_job(
            conn, job_id, fence,
            ModelName=getattr(analyzer.llm_client, "model", None),
            SourceLastUpdatedUtc=utc_text(),
        ):
            report.completed += 1
        else:
            logger.warning(
                "worker | job %s: result persisted but the lease was lost; a "
                "replacement worker will find the link and complete it",
                job_id,
            )
        return

    status = repo.fail_job(
        conn, job_id, fence,
        error=outcome.error or "evaluation produced no persisted result",
        failure_class=outcome.failure_class or "transient",
        max_attempts=settings.max_attempts,
    )
    _count(report, status)


def _count(report: WorkerReport, status: str) -> None:
    if status == "quota_wait":
        report.quota_paused = True
        report.retried += 1
    elif status == "retry_wait":
        report.retried += 1
    elif status == "failed":
        report.failed += 1


# ─────────────────────────────────────────────────────────────────────────────
# The loop
# ─────────────────────────────────────────────────────────────────────────────

async def work_once(
    conn, cm_handler, dwh_engine, analyzer, settings: AutomationSettings,
    *, should_stop=None,
) -> WorkerReport:
    """Drain claimable jobs, pausing after quota exhaustion or shutdown."""
    report = WorkerReport()
    report.recovered += repo.recover_expired_leases(conn)
    owner = worker_identity()

    while True:
        if report.quota_paused or (should_stop is not None and should_stop()):
            break
        job = repo.claim_job(conn, owner=owner, lease_minutes=settings.lease_duration_minutes)
        if job is None:
            break
        report.claimed += 1
        logger.info(
            "worker | claimed job %s (conversation %s, attempt %d)",
            job["JobId"], job["ConversationId"], job["Attempts"],
        )
        await process_job(conn, cm_handler, dwh_engine, analyzer, job, settings, report)

    if report.claimed:
        logger.info(
            "worker | claimed=%d completed=%d recovered=%d retried=%d failed=%d skipped=%d",
            report.claimed, report.completed, report.recovered,
            report.retried, report.failed, report.skipped_already_done,
        )
    return report
