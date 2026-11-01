"""
The automation process entrypoint.

    python -m app.automation.runner                 # the daemon
    python -m app.automation.runner plan                 # today in the reporting timezone
    python -m app.automation.runner dispatch
    python -m app.automation.runner work
    python -m app.automation.runner reconcile --date YYYY-MM-DD
    python -m app.automation.runner status               # today in the reporting timezone
    python -m app.automation.runner backfill --date YYYY-MM-DD
    QA_LEDGER_DB_PATH=/tmp/qa-dry-run.sqlite3 python -m app.automation.runner dry-run --date YYYY-MM-DD
    python -m app.automation.runner repair                  # unstick a wedged ledger
    python -m app.automation.runner replan --date YYYY-MM-DD  # re-plan against the CM roster

Deliberately NOT wired into FastAPI's lifespan. Uvicorn with several workers
would start one scheduler per worker, and three schedulers planning the same
day is how a budget of 300 becomes 900. A single process owns scheduling; the
ledger's constraints are the backstop if that is ever violated.

The planner and worker run as independent asyncio tasks with their own
cadences, so a slow evaluation cannot delay the dispatcher and a stalled
dispatch cannot starve the worker.

`dry-run` is the important one before go-live: it plans and dispatches a whole
day with **no LLM calls at all**, so the selection can be inspected and
re-run for determinism without spending a token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.automation import repository as repo
from app.automation.config import AutomationSettings, automation_settings
from app.automation.calendar import reporting_day
from app.automation.db import open_ledger
from app.automation.backfill import backfill_run
from app.automation.dispatcher import dispatch_once
from app.automation.planner import plan_day
from app.automation.reconciler import day_is_reconcilable, reconcile_day
from app.automation.roster import fetch_active_agents
from app.automation.worker import work_once

logger = logging.getLogger("app.automation.runner")

UTC = timezone.utc
_shutdown = asyncio.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Resources
# ─────────────────────────────────────────────────────────────────────────────

def _cm_handler(settings: AutomationSettings):
    """A connected CM_DWH handler, reusing the existing credential handling."""
    import sys as _sys
    from pathlib import Path

    cm_dir = Path(__file__).resolve().parent.parent / "CM"
    if str(cm_dir) not in _sys.path:
        _sys.path.insert(0, str(cm_dir))
    from CM_DB_Handler import CMDatabaseHandler              # noqa: E402

    handler = CMDatabaseHandler(json_file=str(cm_dir / "passcode.json"), db_key="CM")
    if not handler.connect(query_timeout=300):
        raise ConnectionError("Could not connect to CM_DWH")
    return handler


def _dwh_engine():
    from app.services.sql_helpers import _build_engine
    return _build_engine()


def _analyzer():
    from app.agent import QAAgent
    from app.config import settings as app_settings
    from app.services.llm_client import LLMClient

    return QAAgent(llm_client=LLMClient(provider=app_settings.llm_provider, model=app_settings.llm_model))


class _CMSession:
    """
    One CM_DWH connection, opened on first use and kept for the life of a loop.

    The loops used to open a fresh connection on every iteration and close it
    straight after — every 60 seconds for an idle worker — which is the
    endless "Connected / Database connection closed" output. A dropped
    connection is still recovered: the query layer reconnects on transient
    errors, and a loop that fails outright calls reset() so the next iteration
    starts clean.
    """

    def __init__(self, settings: AutomationSettings):
        self._settings = settings
        self._handler = None

    def handler(self):
        if self._handler is None:
            self._handler = _cm_handler(self._settings)
        return self._handler

    def reset(self) -> None:
        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:                                      # noqa: BLE001
                pass  # already gone
        self._handler = None

    close = reset


def _roster(cm_handler, settings: AutomationSettings, reporting_date: date) -> list[dict]:
    """
    The CM agents to plan *reporting_date* for — see roster.fetch_active_agents.
    The look-back ends where that reporting day starts.
    """
    day = reporting_day(reporting_date, settings.tz)
    return fetch_active_agents(
        cm_handler, day_start_utc=day.start_utc,
        lookback_days=settings.roster_lookback_days,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_plan(settings: AutomationSettings, reporting_date: date) -> int:
    conn = open_ledger(settings.ledger_db_path)
    handler = _cm_handler(settings)
    try:
        agents = _roster(handler, settings, reporting_date)
    finally:
        handler.close()
    result = plan_day(conn, reporting_date=reporting_date, settings=settings, agents=agents)
    print(json.dumps({
        "run_id": result.run_id, "date": str(result.reporting_date),
        "created": result.created, "agents": result.agent_count,
        "intervals": result.interval_count,
        "allocated": result.allocated, "target": result.target,
        "notes": result.notes,
    }, indent=2))
    return 0


def cmd_dispatch(settings: AutomationSettings) -> int:
    conn = open_ledger(settings.ledger_db_path)
    handler = _cm_handler(settings)
    try:
        report = dispatch_once(conn, handler, settings)
    finally:
        handler.close()
    print(json.dumps({
        "intervals": report.intervals_processed, "candidates": report.candidates_seen,
        "reserved": report.jobs_reserved, "shortfalls": report.shortfalls,
        "errors": report.errors,
    }, indent=2))
    return 0 if not report.errors else 1


def cmd_work(settings: AutomationSettings) -> int:
    conn = open_ledger(settings.ledger_db_path)
    handler = _cm_handler(settings)
    try:
        report = asyncio.run(work_once(conn, handler, _dwh_engine(), _analyzer(), settings))
    finally:
        handler.close()
    print(json.dumps(report.__dict__, indent=2))
    return 0


def cmd_reconcile(settings: AutomationSettings, reporting_date: date) -> int:
    conn = open_ledger(settings.ledger_db_path)
    report = reconcile_day(conn, settings, reporting_date=reporting_date)
    if report is None:
        print(json.dumps({"error": f"no run for {reporting_date}"}))
        return 1
    payload = {k: v for k, v in report.__dict__.items()}
    payload["reporting_date"] = str(payload["reporting_date"])
    print(json.dumps(payload, indent=2))
    return 0


def cmd_dry_run(settings: AutomationSettings, reporting_date: date) -> int:
    """
    Plan and dispatch a whole reporting day with NO inference.

    Jobs are reserved and then left pending — the worker is never started, so
    not one token is spent. This is the gate before enabling automation: it
    proves the selection is reproducible and the coverage is real while the
    LLM is entirely out of the picture.
    """
    conn = open_ledger(settings.ledger_db_path)
    handler = _cm_handler(settings)
    try:
        plan = plan_day(conn, reporting_date=reporting_date, settings=settings,
                        agents=_roster(handler, settings, reporting_date))

        # Dispatch with the clock wound past the end of the day so that every
        # interval is due at once, rather than only those whose cutoff has
        # genuinely passed.
        run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
        from app.automation.db import parse_utc
        end = parse_utc(run["DayEndUtc"]) + timedelta(hours=1)
        report = dispatch_once(conn, handler, settings, now=end)
    finally:
        handler.close()

    selected = conn.execute(
        "SELECT ConversationId FROM QA_SamplingCandidates c "
        " JOIN QA_SamplingRuns r ON r.RunId = c.RunId "
        " WHERE r.ReportingDate = ? AND c.IsSelected = 1 ORDER BY ConversationId",
        (reporting_date.isoformat(),),
    ).fetchall()

    import hashlib
    fingerprint = hashlib.sha256(
        "\n".join(r[0] for r in selected).encode("utf-8")
    ).hexdigest()

    print(json.dumps({
        "run_id": plan.run_id,
        "date": str(reporting_date),
        "agents": plan.agent_count,
        "intervals": plan.interval_count,
        "candidates": report.candidates_seen,
        "selected": len(selected),
        "reserved": report.jobs_reserved,
        "target": settings.daily_target,
        "selection_fingerprint": fingerprint,
        "shortfalls": report.shortfalls[:20],
        "errors": report.errors[:20],
        "note": "NO inference was run. Re-run on a fresh ledger and compare "
                "selection_fingerprint to prove determinism.",
    }, indent=2))
    return 0


def cmd_replan(settings: AutomationSettings, reporting_date: date) -> int:
    """
    Plan a day again against the current roster, keeping its existing jobs.

    For a day that was planned against the wrong agent list — every run made
    while the roster came from DWH.AI.Agents. The old run is abandoned, a new
    plan is made from the CM roster, and the old run's jobs move onto it, so
    nothing already reserved or evaluated is repeated and the daily budget
    still counts them.
    """
    conn = open_ledger(settings.ledger_db_path)
    old = repo.get_run(conn, reporting_date, settings.sampling_policy_version)

    handler = _cm_handler(settings)
    try:
        agents = _roster(handler, settings, reporting_date)
    finally:
        handler.close()
    if not agents:
        print(json.dumps({"error": "the CM roster is empty; the existing plan was left untouched"}))
        return 1

    if old is not None:
        repo.abandon_run(conn, old["RunId"])
    try:
        plan = plan_day(conn, reporting_date=reporting_date, settings=settings, agents=agents)
    except Exception:
        if old is not None:
            repo.reactivate_run(conn, old["RunId"], old["Status"])
        raise

    moved = repo.adopt_jobs(conn, old["RunId"], plan.run_id) if old is not None else 0
    print(json.dumps({
        "date": str(reporting_date),
        "abandoned_run": old["RunId"] if old is not None else None,
        "new_run": plan.run_id,
        "agents": plan.agent_count,
        "allocated": plan.allocated,
        "target": plan.target,
        "jobs_carried_over": moved,
        "notes": plan.notes,
    }, indent=2))
    return 0


def cmd_repair(settings: AutomationSettings, reporting_date: Optional[date]) -> int:
    """
    Reset transiently-failed intervals and reopen prematurely-closed runs.

    Without --date, repairs every run in the ledger.
    """
    conn = open_ledger(settings.ledger_db_path)
    if reporting_date is not None:
        run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
        runs = [(run["RunId"], str(reporting_date))] if run else []
    else:
        runs = [(r["RunId"], r["ReportingDate"]) for r in conn.execute(
            "SELECT RunId, ReportingDate FROM QA_SamplingRuns ORDER BY ReportingDate"
        )]

    results = {}
    for run_id, day in runs:
        results[day] = repo.repair_run(conn, run_id)
    print(json.dumps(results, indent=2))
    return 0


def cmd_backfill(settings: AutomationSettings, reporting_date: date) -> int:
    conn = open_ledger(settings.ledger_db_path)
    result = backfill_run(conn, settings, reporting_date=reporting_date)
    if result is None:
        print(json.dumps({"error": f"no run for {reporting_date}"}))
        return 1
    print(json.dumps({
        "run_id": result.run_id,
        "reserved_before": result.reserved_before,
        "reserved_after": result.reserved_after,
        "target": result.target,
        "added": result.added,
        "within_quota": result.round_one_added,
        "redistributed": result.round_two_added,
        "deficit": result.deficit,
        "notes": result.notes,
    }, indent=2))
    return 0


def cmd_status(settings: AutomationSettings, reporting_date: date) -> int:
    conn = open_ledger(settings.ledger_db_path)
    run = repo.get_run(conn, reporting_date, settings.sampling_policy_version)
    if run is None:
        print(json.dumps({"error": f"no run for {reporting_date}"}))
        return 1
    coverage = repo.run_coverage(conn, run["RunId"])
    print(json.dumps({
        "run_id": run["RunId"],
        "date": str(reporting_date),
        "timezone": run["ReportingTimezone"],
        "day_hours": run["DayHours"],
        "status": run["Status"],
        "coverage": dict(coverage) if coverage else None,
        "uncovered_agents": repo.uncovered_agents(conn, run["RunId"]),
    }, indent=2, default=str))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# The daemon
# ─────────────────────────────────────────────────────────────────────────────

async def _planner_loop(conn, settings: AutomationSettings) -> None:
    """Plan today (and yesterday, if it was missed) then reconcile closed days."""
    session = _CMSession(settings)
    try:
        while not _shutdown.is_set():
            try:
                today = datetime.now(settings.tz).date()
                # Only touch CM when a day actually needs a plan — the roster
                # query takes ~30 seconds and is pointless once both exist.
                for day in (today, today - timedelta(days=1)):
                    if settings.automation_start_date and day < settings.automation_start_date:
                        continue
                    if repo.get_run(conn, day, settings.sampling_policy_version) is None:
                        agents = _roster(session.handler(), settings, day)
                        plan_day(conn, reporting_date=day, settings=settings, agents=agents)

                yesterday = today - timedelta(days=1)
                if (
                    (not settings.automation_start_date or yesterday >= settings.automation_start_date)
                    and day_is_reconcilable(conn, settings, yesterday)
                ):
                    reconcile_day(conn, settings, reporting_date=yesterday)
            except Exception:                                      # noqa: BLE001
                logger.exception("planner loop iteration failed; continuing")
                session.reset()

            await _sleep_or_shutdown(600)
    finally:
        session.close()


async def _dispatcher_loop(conn, settings: AutomationSettings) -> None:
    session = _CMSession(settings)
    try:
        while not _shutdown.is_set():
            try:
                # Connect only when an interval is actually due.
                if repo.due_intervals(conn, min_reporting_date=settings.automation_start_date):
                    dispatch_once(conn, session.handler(), settings)
            except Exception:                                      # noqa: BLE001
                logger.exception("dispatcher loop iteration failed; continuing")
                session.reset()
            await _sleep_or_shutdown(settings.dispatch_interval_seconds)
    finally:
        session.close()


async def _worker_loop(settings: AutomationSettings) -> None:
    # Each worker owns its SQLite connection. Multiple asyncio workers can
    # await LLM calls concurrently without sharing connection transaction state.
    conn = open_ledger(settings.ledger_db_path)
    try:
        analyzer = _analyzer()
        dwh = _dwh_engine()
        session = _CMSession(settings)
        try:
            while not _shutdown.is_set():
                try:
                    # Checked in the ledger first, so an idle worker never
                    # opens a CM connection — previously it connected and
                    # disconnected every minute with nothing to do.
                    if not repo.has_claimable_job(
                        conn, min_reporting_date=settings.automation_start_date,
                    ):
                        await _sleep_or_shutdown(60)
                        continue
                    report = await work_once(
                        conn, session.handler(), dwh, analyzer, settings,
                        should_stop=_shutdown.is_set,
                    )
                    # A 402 cannot improve through immediate retries. Keep jobs
                    # available and try again after the provider can be funded.
                    delay = 300 if report.quota_paused else (5 if report.claimed else 60)
                    await _sleep_or_shutdown(delay)
                except Exception:                                  # noqa: BLE001
                    logger.exception("worker loop iteration failed; continuing")
                    session.reset()
                    await _sleep_or_shutdown(60)
        finally:
            session.close()
    finally:
        conn.close()


async def _sleep_or_shutdown(seconds: float) -> None:
    try:
        await asyncio.wait_for(_shutdown.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def _daemon(settings: AutomationSettings) -> int:
    if not settings.automation_enabled:
        logger.error(
            "QA_AUTOMATION_ENABLED is false. Refusing to start the daemon. "
            "Individual subcommands still run for testing."
        )
        return 2

    conn = open_ledger(settings.ledger_db_path)
    logger.info(
        "automation starting | tz=%s target=%d policy=%s ledger=%s concurrency=%d",
        settings.reporting_timezone, settings.daily_target,
        settings.sampling_policy_version, settings.ledger_db_path,
        settings.worker_concurrency,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except NotImplementedError:
            pass  # Windows

    await asyncio.gather(
        _planner_loop(conn, settings),
        _dispatcher_loop(conn, settings),
        *[_worker_loop(settings) for _ in range(settings.worker_concurrency)],
    )
    logger.info("automation stopped")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(prog="app.automation.runner", description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="daemon",
        choices=["daemon", "plan", "dispatch", "work", "reconcile",
                 "dry-run", "status", "repair", "backfill", "replan"],
    )
    parser.add_argument("--date", help="Reporting date, YYYY-MM-DD. Defaults to today in the reporting timezone.")
    args = parser.parse_args(argv)

    settings = automation_settings()
    reporting_date = (
        date.fromisoformat(args.date) if args.date else datetime.now(settings.tz).date()
    )

    if args.command == "daemon":
        return asyncio.run(_daemon(settings))
    if args.command == "plan":
        return cmd_plan(settings, reporting_date)
    if args.command == "dispatch":
        return cmd_dispatch(settings)
    if args.command == "work":
        return cmd_work(settings)
    if args.command == "reconcile":
        return cmd_reconcile(settings, reporting_date)
    if args.command == "dry-run":
        return cmd_dry_run(settings, reporting_date)
    if args.command == "status":
        return cmd_status(settings, reporting_date)
    if args.command == "repair":
        return cmd_repair(settings, date.fromisoformat(args.date) if args.date else None)
    if args.command == "backfill":
        return cmd_backfill(settings, reporting_date)
    if args.command == "replan":
        return cmd_replan(settings, reporting_date)
    return 1


if __name__ == "__main__":
    sys.exit(main())
