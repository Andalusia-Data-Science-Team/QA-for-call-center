"""
The automation process entrypoint.

    python -m app.automation.runner                 # the daemon
    python -m app.automation.runner plan   --date 2026-10-29
    python -m app.automation.runner dry-run --date 2026-10-29
    python -m app.automation.runner dispatch
    python -m app.automation.runner work
    python -m app.automation.runner reconcile --date 2026-10-29
    python -m app.automation.runner status --date 2026-10-29
    python -m app.automation.runner backfill --date 2026-10-29
    python -m app.automation.runner repair                  # unstick a wedged ledger

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


def _roster(dwh_engine) -> list[dict]:
    """The active agent roster. There is no shift model — see roster.py."""
    return fetch_active_agents(dwh_engine)


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_plan(settings: AutomationSettings, reporting_date: date) -> int:
    conn = open_ledger(settings.ledger_db_path)
    result = plan_day(conn, reporting_date=reporting_date, settings=settings,
                      agents=_roster(_dwh_engine()))
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
                        agents=_roster(_dwh_engine()))

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
    while not _shutdown.is_set():
        try:
            today = datetime.now(settings.tz).date()
            roster = _roster(_dwh_engine())
            for day in (today, today - timedelta(days=1)):
                if repo.get_run(conn, day, settings.sampling_policy_version) is None:
                    plan_day(conn, reporting_date=day, settings=settings, agents=roster)

            yesterday = today - timedelta(days=1)
            if day_is_reconcilable(conn, settings, yesterday):
                reconcile_day(conn, settings, reporting_date=yesterday)
        except Exception:                                          # noqa: BLE001
            logger.exception("planner loop iteration failed; continuing")

        await _sleep_or_shutdown(600)


async def _dispatcher_loop(conn, settings: AutomationSettings) -> None:
    while not _shutdown.is_set():
        try:
            handler = _cm_handler(settings)
            try:
                dispatch_once(conn, handler, settings)
            finally:
                handler.close()
        except Exception:                                          # noqa: BLE001
            logger.exception("dispatcher loop iteration failed; continuing")
        await _sleep_or_shutdown(settings.dispatch_interval_seconds)


async def _worker_loop(settings: AutomationSettings) -> None:
    # Each worker owns its SQLite connection. Multiple asyncio workers can
    # await LLM calls concurrently without sharing connection transaction state.
    conn = open_ledger(settings.ledger_db_path)
    try:
        analyzer = _analyzer()
        dwh = _dwh_engine()
        while not _shutdown.is_set():
            try:
                handler = _cm_handler(settings)
                try:
                    report = await work_once(
                        conn, handler, dwh, analyzer, settings,
                        should_stop=_shutdown.is_set,
                    )
                finally:
                    handler.close()
                # A 402 cannot improve through immediate retries. Keep jobs
                # available and try again after the provider can be funded.
                delay = 300 if report.quota_paused else (5 if report.claimed else 60)
                await _sleep_or_shutdown(delay)
            except Exception:                                      # noqa: BLE001
                logger.exception("worker loop iteration failed; continuing")
                await _sleep_or_shutdown(60)
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
                 "dry-run", "status", "repair", "backfill"],
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
    return 1


if __name__ == "__main__":
    sys.exit(main())
