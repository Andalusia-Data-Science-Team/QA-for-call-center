"""
Tests for the fixes made after the first live days.

Each test here corresponds to something observed in production:

* the daemon printed an endless connect/close loop, because every loop
  iteration opened and closed its own CM connection, idle or not;
* yesterday's sequence numbers blocked unrelated conversations today — an
  interval with 14 eligible candidates and a quota of 1 selected none;
* days were planned against DWH.AI.Agents, which held only 3 of the 18 people
  actually working in CM, and needed re-planning without losing their jobs;
* NEWFIX's columns do not populate agent_email, so the worker must stamp it.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.automation import repository as repo
from app.automation import runner
from app.automation.calendar import reporting_day
from app.automation.config import AutomationSettings
from app.automation.db import open_ledger, utc_text
from app.automation.dispatcher import dispatch_once
from app.automation.planner import plan_day

UTC = timezone.utc
DAY = date(2026, 10, 18)


def _settings(**overrides) -> AutomationSettings:
    base = dict(
        automation_enabled=True, daily_target=20, reporting_timezone="Africa/Cairo",
        sampling_interval_minutes=120, sampling_seed="42", sampling_policy_version="v1",
        exclude_adjacent_chats=True,
    )
    base.update(overrides)
    return AutomationSettings(**base)


def _agents(n=2):
    return [{"Agent_Email_Address": f"a{i}@andalusiagroup.net", "Agent_Name": f"A{i}",
             "Agent_id": str(i)} for i in range(n)]


class FakeCM:
    """One closed, replied conversation every 20 minutes per agent."""

    def __init__(self, day, agents, cadence=20):
        self.day, self.agents, self.cadence = day, agents, cadence

    def rows(self, start, end):
        rows = []
        for agent in self.agents:
            moment = self.day.start_utc
            while moment < self.day.end_utc:
                if start <= moment < end:
                    naive = moment.replace(tzinfo=None)
                    rows.append({
                        "ConversationId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{agent}|{moment}")),
                        "AgentEmail": agent, "AgentFullName": agent,
                        "HandlingTimeUtc": naive, "ArchiveDateTimeUtc": naive,
                        "StartDateTimeUtc": naive - timedelta(minutes=15),
                        "LastUpdatedUtc": naive, "ConversationState": 2, "AnswerState": 1,
                        "AgentReplied": 1, "WebStoreName": "AHJ",
                    })
                moment += timedelta(minutes=self.cadence)
        return pd.DataFrame(rows)


@pytest.fixture()
def fake_cm(monkeypatch):
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: handler.rows(start, end),
    )


# ── Connection reuse ─────────────────────────────────────────────────────────

class CountingHandler:
    opened = 0
    closed = 0

    def __init__(self):
        CountingHandler.opened += 1

    def close(self):
        CountingHandler.closed += 1


@pytest.fixture()
def counting(monkeypatch):
    CountingHandler.opened = CountingHandler.closed = 0
    monkeypatch.setattr(runner, "_cm_handler", lambda settings: CountingHandler())
    return CountingHandler


def test_session_reuses_one_connection(counting):
    session = runner._CMSession(_settings())
    first = session.handler()
    for _ in range(50):
        assert session.handler() is first
    assert counting.opened == 1
    assert counting.closed == 0


def test_session_reset_closes_and_reopens_once(counting):
    session = runner._CMSession(_settings())
    session.handler()
    session.reset()
    session.handler()
    assert (counting.opened, counting.closed) == (2, 1)


def test_session_close_is_safe_when_nothing_was_opened(counting):
    runner._CMSession(_settings()).close()
    assert (counting.opened, counting.closed) == (0, 0)


def test_idle_worker_never_opens_a_cm_connection(tmp_path, monkeypatch, counting):
    """
    The connect/close loop in the daemon output came from an idle worker
    opening a connection every 60 seconds. With nothing claimable it must
    not connect at all.
    """
    settings = _settings(ledger_db_path=str(tmp_path / "ledger.sqlite3"))
    monkeypatch.setattr(runner, "_analyzer", lambda: object())
    monkeypatch.setattr(runner, "_dwh_engine", lambda: object())

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 5:
            runner._shutdown.set()

    monkeypatch.setattr(runner, "_sleep_or_shutdown", fake_sleep)
    runner._shutdown.clear()
    try:
        asyncio.run(runner._worker_loop(settings))
    finally:
        runner._shutdown.clear()

    assert len(sleeps) == 5
    assert counting.opened == 0


def test_dispatcher_does_not_connect_when_nothing_is_due(tmp_path, monkeypatch, counting):
    settings = _settings()
    conn = open_ledger(tmp_path / "ledger.sqlite3")

    async def stop_after_one(seconds):
        runner._shutdown.set()

    monkeypatch.setattr(runner, "_sleep_or_shutdown", stop_after_one)
    runner._shutdown.clear()
    try:
        asyncio.run(runner._dispatcher_loop(conn, settings))
    finally:
        runner._shutdown.clear()
    assert counting.opened == 0


def test_has_claimable_job_reflects_the_queue(tmp_path, fake_cm):
    settings = _settings()
    day = reporting_day(DAY, settings.tz)
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    assert repo.has_claimable_job(conn) is False

    plan_day(conn, reporting_date=DAY, settings=settings, agents=_agents())
    dispatch_once(conn, FakeCM(day, [a["Agent_Email_Address"] for a in _agents()]), settings,
                  now=day.end_utc + timedelta(hours=1))
    assert repo.has_claimable_job(conn) is True

    conn.execute("UPDATE QA_EvaluationJobs SET Status = 'completed'")
    assert repo.has_claimable_job(conn) is False


def test_start_date_keeps_older_intervals_out_of_dispatch(tmp_path, fake_cm):
    settings = _settings(automation_start_date=DAY + timedelta(days=1))
    day = reporting_day(DAY, settings.tz)
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    agents = _agents()
    plan_day(conn, reporting_date=DAY, settings=settings, agents=agents)

    report = dispatch_once(
        conn, FakeCM(day, [agent["Agent_Email_Address"] for agent in agents]),
        settings, now=day.end_utc + timedelta(hours=1),
    )

    assert report.intervals_processed == 0
    assert conn.execute("SELECT COUNT(*) FROM QA_EvaluationJobs").fetchone()[0] == 0


# ── Spacing across midnight ──────────────────────────────────────────────────

def _seed_previous_day(conn, settings, agents, last_selected: bool):
    """A previous-day run whose agents' last conversation was or was not selected."""
    if isinstance(agents, str):
        agents = [agents]
    prev = DAY - timedelta(days=1)
    pday = reporting_day(prev, settings.tz)
    run_id = repo.create_run(
        conn, reporting_date=prev, timezone_name=settings.reporting_timezone,
        day_start_utc=pday.start_utc, day_end_utc=pday.end_utc,
        policy_version=settings.sampling_policy_version, base_seed=settings.sampling_seed,
        daily_target=settings.daily_target, interval_minutes=120, grace_minutes=20,
        exclude_adjacent=True, min_gap_minutes=0,
    )
    rows = []
    # Many selected positions, deliberately sitting on what will be TODAY's numbers.
    for agent in agents:
        for position in range(1, 41):
            selected = (position % 2 == 0) if position < 40 else last_selected
            rows.append({
                "SnapshotId": "s", "ConversationId": f"prev-{agent}-{position}", "RunId": run_id,
                "AgentEmailAddress": agent, "IntervalId": "i", "AttributionRule": "inbox_owner",
                "HandlingTimeUtc": utc_text(pday.start_utc + timedelta(minutes=position)),
                "SequencePosition": position, "RankHash": "0" * 64, "IsEligible": 1,
                "EligibilityReason": None, "IsSelected": int(selected), "SelectionReason": None,
                "ConversationState": None, "SourceLastUpdatedUtc": None, "IsLateArrival": 0,
                "CreatedAtUtc": utc_text(),
            })
    repo.insert_candidates(conn, rows)


def test_yesterdays_numbers_do_not_block_today(tmp_path, fake_cm):
    """
    Regression: yesterday's selected positions were merged into today's set,
    but each day numbers from 1. On 2026-10-17 that blocked 14 of 14 eligible
    candidates for a quota of 1.
    """
    settings = _settings(daily_target=40)
    agents = _agents(2)
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    _seed_previous_day(conn, settings, [a["Agent_Email_Address"] for a in agents],
                       last_selected=False)

    day = reporting_day(DAY, settings.tz)
    plan_day(conn, reporting_date=DAY, settings=settings, agents=agents)
    report = dispatch_once(conn, FakeCM(day, [a["Agent_Email_Address"] for a in agents]),
                           settings, now=day.end_utc + timedelta(hours=1))

    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    assert repo.reserved_count(conn, run["RunId"]) == 40
    assert not any("spacing rule blocked" in s for s in report.shortfalls)


def test_midnight_rule_blocks_only_todays_first_conversation(tmp_path):
    settings = _settings()
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    agent = "a0@andalusiagroup.net"

    _seed_previous_day(conn, settings, agent, last_selected=True)
    assert repo.midnight_boundary_positions(
        conn, agent, DAY - timedelta(days=1), "v1") == {0}


def test_midnight_rule_blocks_nothing_when_yesterday_ended_unselected(tmp_path):
    settings = _settings()
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    agent = "a0@andalusiagroup.net"

    _seed_previous_day(conn, settings, agent, last_selected=False)
    assert repo.midnight_boundary_positions(
        conn, agent, DAY - timedelta(days=1), "v1") == set()


def test_midnight_rule_with_no_previous_day(tmp_path):
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    assert repo.midnight_boundary_positions(conn, "x@y", DAY - timedelta(days=1), "v1") == set()


# ── Re-planning ──────────────────────────────────────────────────────────────

def test_replan_keeps_existing_jobs_and_the_budget(tmp_path, fake_cm, monkeypatch):
    """
    A day planned against the wrong roster is re-planned against the CM roster.
    Jobs already reserved move to the new run, so nothing is repeated and the
    daily target still counts them.
    """
    settings = _settings(daily_target=20, ledger_db_path=str(tmp_path / "ledger.sqlite3"))
    conn = open_ledger(settings.ledger_db_path)
    day = reporting_day(DAY, settings.tz)

    wrong = [{"Agent_Email_Address": "not-in-cm@andalusiagroup.net", "Agent_Name": "x", "Agent_id": "x"},
             {"Agent_Email_Address": "a0@andalusiagroup.net", "Agent_Name": "a0", "Agent_id": "0"}]
    plan_day(conn, reporting_date=DAY, settings=settings, agents=wrong)
    dispatch_once(conn, FakeCM(day, ["a0@andalusiagroup.net"]), settings,
                  now=day.start_utc + timedelta(hours=6), backfill=False)
    old = repo.get_run(conn, DAY, "v1")
    carried = repo.reserved_count(conn, old["RunId"])
    assert carried > 0
    conn.close()

    monkeypatch.setattr(runner, "_cm_handler", lambda s: CountingHandler())
    monkeypatch.setattr(runner, "_roster", lambda handler, s, d: _agents(2))
    monkeypatch.setattr(runner, "automation_settings", lambda: settings)
    assert runner.cmd_replan(settings, DAY) == 0

    conn = open_ledger(settings.ledger_db_path)
    new = repo.get_run(conn, DAY, "v1")
    assert new["RunId"] != old["RunId"]
    assert conn.execute("SELECT Status FROM QA_SamplingRuns WHERE RunId = ?",
                        (old["RunId"],)).fetchone()[0] == "abandoned"
    assert repo.reserved_count(conn, new["RunId"]) == carried

    # Finishing the day must not exceed the target, carried jobs included.
    dispatch_once(conn, FakeCM(day, [a["Agent_Email_Address"] for a in _agents(2)]), settings,
                  now=day.end_utc + timedelta(hours=1))
    assert repo.reserved_count(conn, new["RunId"]) == settings.daily_target
    total, distinct = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ConversationId) FROM QA_EvaluationJobs").fetchone()
    assert total == distinct


def test_replan_restores_the_old_run_if_planning_fails(tmp_path, monkeypatch):
    settings = _settings(ledger_db_path=str(tmp_path / "ledger.sqlite3"))
    conn = open_ledger(settings.ledger_db_path)
    plan_day(conn, reporting_date=DAY, settings=settings, agents=_agents())
    old = repo.get_run(conn, DAY, "v1")
    conn.close()

    monkeypatch.setattr(runner, "_cm_handler", lambda s: CountingHandler())
    monkeypatch.setattr(runner, "_roster", lambda handler, s, d: _agents())

    def boom(*a, **k):
        raise RuntimeError("planning failed")

    monkeypatch.setattr(runner, "plan_day", boom)
    with pytest.raises(RuntimeError):
        runner.cmd_replan(settings, DAY)

    conn = open_ledger(settings.ledger_db_path)
    assert repo.get_run(conn, DAY, "v1")["RunId"] == old["RunId"]


def test_replan_refuses_an_empty_roster(tmp_path, monkeypatch):
    settings = _settings(ledger_db_path=str(tmp_path / "ledger.sqlite3"))
    conn = open_ledger(settings.ledger_db_path)
    plan_day(conn, reporting_date=DAY, settings=settings, agents=_agents())
    old = repo.get_run(conn, DAY, "v1")
    conn.close()

    monkeypatch.setattr(runner, "_cm_handler", lambda s: CountingHandler())
    monkeypatch.setattr(runner, "_roster", lambda handler, s, d: [])
    assert runner.cmd_replan(settings, DAY) == 1

    conn = open_ledger(settings.ledger_db_path)
    assert repo.get_run(conn, DAY, "v1")["RunId"] == old["RunId"]


# ── Roster ───────────────────────────────────────────────────────────────────

def test_roster_lookback_ends_where_the_reporting_day_starts(monkeypatch):
    """Fixed before the day begins, so the day's own traffic cannot shape it."""
    from app.automation import roster

    seen = {}

    def fake_roster(handler, start, end):
        seen["window"] = (start, end)
        return pd.DataFrame(_agents(3))

    monkeypatch.setattr("app.CM.candidate_queries.fetch_cm_roster", fake_roster)
    day = reporting_day(DAY, _settings().tz)
    agents = roster.fetch_active_agents(object(), day_start_utc=day.start_utc, lookback_days=14)

    assert len(agents) == 3
    assert seen["window"] == (day.start_utc - timedelta(days=14), day.start_utc)
