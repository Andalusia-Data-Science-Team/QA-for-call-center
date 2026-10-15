"""
Tests for deficit backfill and the recovery paths.

The point of backfill is to reach the daily target when the data allows, and
to report an honest shortfall when it does not. Both halves are tested here,
because a backfill that hits 300 by relaxing spacing or eligibility would be
worse than one that falls short.
"""

from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.automation import repository as repo
from app.automation.backfill import backfill_run
from app.automation.calendar import reporting_day
from app.automation.config import AutomationSettings
from app.automation.db import open_ledger
from app.automation.dispatcher import dispatch_once
from app.automation.planner import plan_day

UTC = timezone.utc
DAY = date(2026, 10, 29)


def _settings(target=300, **overrides) -> AutomationSettings:
    base = dict(
        automation_enabled=True, daily_target=target,
        reporting_timezone="Africa/Cairo", sampling_interval_minutes=120,
        sampling_seed="42", sampling_policy_version="v1",
        exclude_adjacent_chats=True,
    )
    base.update(overrides)
    return AutomationSettings(**base)


class UnevenCM:
    """
    Traffic that is NOT evenly spread across intervals — which is the whole
    reason backfill exists. An even fake would hide the bug it fixes.
    """

    def __init__(self, day, agents, per_agent, seed=7, closed=True):
        self.day, self.agents = day, agents
        self.per_agent, self.seed, self.closed = per_agent, seed, closed
        self._cache = None

    def _all(self):
        if self._cache is not None:
            return self._cache
        rng = random.Random(self.seed)
        span = (self.day.end_utc - self.day.start_utc).total_seconds()
        rows = []
        for agent in self.agents:
            for _ in range(self.per_agent):
                moment = self.day.start_utc + timedelta(seconds=rng.uniform(0, span))
                rows.append({
                    "ConversationId": str(uuid.uuid4()), "AgentEmail": agent,
                    "AgentFullName": agent, "AgentPersonId": "p",
                    "HandlingTimeUtc": moment.replace(tzinfo=None),
                    "InboxOwnerEmail": agent, "OwnerDiffersFromHandler": 0,
                    "ConversationState": "Closed", "AnswerState": 1,
                    "ArchiveDateTimeUtc": (moment.replace(tzinfo=None) + timedelta(minutes=5)
                                           if self.closed else None),
                    "StartDateTimeUtc": moment.replace(tzinfo=None),
                    "LastUpdatedUtc": moment.replace(tzinfo=None),
                    "HumanMessageCount": 4, "TotalMessageCount": 8,
                    "WasForwarded": 0, "WebStoreName": "AHJ",
                })
        self._cache = pd.DataFrame(rows)
        return self._cache

    def rows(self, start, end):
        df = self._all()
        if df.empty:
            return df
        mask = ((df["HandlingTimeUtc"] >= start.replace(tzinfo=None))
                & (df["HandlingTimeUtc"] < end.replace(tzinfo=None)))
        return df.loc[mask]


@pytest.fixture()
def run_day(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: handler.rows(start, end),
    )

    def build(agent_count, per_agent, target=300, backfill=True, closed=True):
        settings = _settings(target)
        day = reporting_day(DAY, settings.tz)
        agents = [f"a{i}@x.net" for i in range(agent_count)]
        conn = open_ledger(tmp_path / f"{agent_count}-{per_agent}-{target}-{backfill}.sqlite3")
        plan_day(
            conn, reporting_date=DAY, settings=settings,
            agents=[{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                    for i, a in enumerate(agents)],
        )
        fake = UnevenCM(day, agents, per_agent, closed=closed)
        report = dispatch_once(conn, fake, settings,
                               now=day.end_utc + timedelta(hours=2), backfill=backfill)
        run = repo.get_run(conn, DAY, settings.sampling_policy_version)
        return conn, settings, run, report

    return build


# ── Reaching the target ──────────────────────────────────────────────────────

def test_busy_day_reaches_the_full_300(run_day):
    conn, settings, run, report = run_day(20, 60, target=300)
    assert repo.reserved_count(conn, run["RunId"]) == 300


def test_300_reserved_conversations_are_all_distinct(run_day):
    """'300 unique conversation ids' is the actual requirement."""
    conn, settings, run, report = run_day(20, 60, target=300)
    total, distinct = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ConversationId) FROM QA_EvaluationJobs"
    ).fetchone()
    assert total == distinct == 300


def test_without_backfill_the_day_falls_short(run_day):
    """
    Guards the fix itself. Even quotas over uneven traffic lose slots that no
    later interval can use; if this ever starts passing at 300, backfill has
    silently become a no-op.
    """
    conn, settings, run, report = run_day(20, 60, target=300, backfill=False)
    assert repo.reserved_count(conn, run["RunId"]) < 300


def test_backfill_is_what_closes_the_gap(run_day):
    conn, settings, run, report = run_day(20, 60, target=300, backfill=False)
    before = repo.reserved_count(conn, run["RunId"])

    result = backfill_run(conn, settings, reporting_date=DAY)
    assert result.added > 0
    assert result.reserved_after == 300
    assert repo.reserved_count(conn, run["RunId"]) == 300


def test_uneven_agent_volumes_still_reach_the_target(tmp_path, monkeypatch):
    """Some agents busy, some quiet — redistribution must cover the difference."""
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: handler.rows(start, end),
    )
    settings = _settings(300)
    day = reporting_day(DAY, settings.tz)
    agents = [f"a{i}@x.net" for i in range(20)]

    class Mixed(UnevenCM):
        def _all(self):
            if self._cache is not None:
                return self._cache
            rng = random.Random(3)
            span = (self.day.end_utc - self.day.start_utc).total_seconds()
            rows = []
            for i, agent in enumerate(self.agents):
                for _ in range(5 + i * 6):          # 5 .. 119 conversations
                    moment = self.day.start_utc + timedelta(seconds=rng.uniform(0, span))
                    rows.append({
                        "ConversationId": str(uuid.uuid4()), "AgentEmail": agent,
                        "AgentFullName": agent, "AgentPersonId": "p",
                        "HandlingTimeUtc": moment.replace(tzinfo=None),
                        "InboxOwnerEmail": agent, "OwnerDiffersFromHandler": 0,
                        "ConversationState": "Closed", "AnswerState": 1,
                        "ArchiveDateTimeUtc": moment.replace(tzinfo=None) + timedelta(minutes=5),
                        "StartDateTimeUtc": moment.replace(tzinfo=None),
                        "LastUpdatedUtc": moment.replace(tzinfo=None),
                        "HumanMessageCount": 4, "TotalMessageCount": 8,
                        "WasForwarded": 0, "WebStoreName": "AHJ",
                    })
            self._cache = pd.DataFrame(rows)
            return self._cache

    conn = open_ledger(tmp_path / "mixed.sqlite3")
    plan_day(conn, reporting_date=DAY, settings=settings,
             agents=[{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                     for i, a in enumerate(agents)])
    dispatch_once(conn, Mixed(day, agents, 0), settings, now=day.end_utc + timedelta(hours=2))

    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    assert repo.reserved_count(conn, run["RunId"]) == 300


# ── Not cheating to hit the number ───────────────────────────────────────────

def test_backfill_never_breaks_the_spacing_rule(run_day):
    conn, settings, run, report = run_day(20, 60, target=300)
    for agent in {r[0] for r in conn.execute(
        "SELECT DISTINCT AgentEmailAddress FROM QA_EvaluationJobs"
    )}:
        positions = sorted(r[0] for r in conn.execute(
            "SELECT SequencePosition FROM QA_SamplingCandidates "
            " WHERE AgentEmailAddress = ? AND IsSelected = 1", (agent,)
        ))
        assert all(b - a > 1 for a, b in zip(positions, positions[1:])), agent


def test_backfill_never_selects_an_ineligible_conversation(run_day):
    conn, settings, run, report = run_day(20, 60, target=300, closed=False)
    assert conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingCandidates WHERE IsSelected = 1 AND IsEligible = 0"
    ).fetchone()[0] == 0
    assert repo.reserved_count(conn, run["RunId"]) == 0


def test_backfill_never_exceeds_the_target(run_day):
    conn, settings, run, report = run_day(20, 200, target=300)
    assert repo.reserved_count(conn, run["RunId"]) == 300


def test_a_thin_day_falls_short_honestly(run_day):
    """
    Not enough conversations exist. Backfill must report the deficit rather
    than relax a rule to manufacture coverage.
    """
    conn, settings, run, report = run_day(10, 4, target=300)
    reserved = repo.reserved_count(conn, run["RunId"])
    assert reserved < 300

    result = backfill_run(conn, settings, reporting_date=DAY)
    assert result.deficit > 0
    assert any("NOT relaxed" in n for n in result.notes)


def test_backfill_is_idempotent(run_day):
    conn, settings, run, report = run_day(20, 60, target=300)
    after_first = repo.reserved_count(conn, run["RunId"])
    second = backfill_run(conn, settings, reporting_date=DAY)
    assert second.added == 0
    assert repo.reserved_count(conn, run["RunId"]) == after_first


def test_redistribution_is_recorded_for_audit(run_day):
    conn, settings, run, report = run_day(20, 60, target=300, backfill=False)
    result = backfill_run(conn, settings, reporting_date=DAY)
    if result.agents_over_quota:
        reasons = {r[0] for r in conn.execute(
            "SELECT ReallocationReason FROM QA_AgentQuotas "
            " WHERE RunId = ? AND ReallocationReason IS NOT NULL", (run["RunId"],)
        )}
        assert any("redistributed" in r for r in reasons)


def test_backfill_waits_until_every_interval_has_reported(tmp_path, monkeypatch):
    """
    Backfilling mid-day would hand out slots that a later interval's own
    candidates should have taken, biasing the day towards whoever ran first.
    """
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: handler.rows(start, end),
    )
    settings = _settings(300)
    day = reporting_day(DAY, settings.tz)
    agents = [f"a{i}@x.net" for i in range(20)]
    conn = open_ledger(tmp_path / "partial.sqlite3")
    plan_day(conn, reporting_date=DAY, settings=settings,
             agents=[{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                     for i, a in enumerate(agents)])

    # Only a third of the day is due.
    report = dispatch_once(conn, UnevenCM(day, agents, 60), settings,
                           now=day.start_utc + timedelta(hours=8))
    assert report.backfilled == 0

    remaining = conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals WHERE Status = 'pending'"
    ).fetchone()[0]
    assert remaining > 0


# ── Repair ───────────────────────────────────────────────────────────────────

def test_repair_resets_transiently_failed_intervals(run_day):
    """
    A 'Communication link failure' must not permanently delete an interval's
    quota — which is what happened on the first live run.
    """
    conn, settings, run, report = run_day(5, 10, target=50)
    conn.execute(
        "UPDATE QA_SamplingIntervals SET Status = 'failed', "
        "       ShortfallReason = '[08S01] Communication link failure (0)' "
        " WHERE RunId = ?", (run["RunId"],)
    )

    result = repo.repair_run(conn, run["RunId"])
    assert result["intervals_reset"] > 0
    assert result["intervals_left_failed"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals WHERE Status = 'failed'"
    ).fetchone()[0] == 0


def test_repair_leaves_genuine_errors_failed(run_day):
    """Resetting a real error would just loop on it forever."""
    conn, settings, run, report = run_day(5, 10, target=50)
    conn.execute(
        "UPDATE QA_SamplingIntervals SET Status = 'failed', "
        "       ShortfallReason = \"Invalid column name 'Frobnicate'\" "
        " WHERE RunId = ?", (run["RunId"],)
    )
    result = repo.repair_run(conn, run["RunId"])
    assert result["intervals_reset"] == 0
    assert result["intervals_left_failed"] > 0


def test_repair_reopens_a_run_closed_before_dispatch(run_day):
    conn, settings, run, report = run_day(5, 10, target=50)
    conn.execute(
        "UPDATE QA_SamplingIntervals SET Status = 'pending' WHERE RunId = ?", (run["RunId"],))
    conn.execute(
        "UPDATE QA_SamplingRuns SET Status = 'closed' WHERE RunId = ?", (run["RunId"],))

    result = repo.repair_run(conn, run["RunId"])
    assert result["run_reopened"] is True
    assert repo.get_run(conn, DAY, settings.sampling_policy_version)["Status"] == "active"


def test_repair_leaves_a_genuinely_finished_run_closed(run_day):
    conn, settings, run, report = run_day(5, 10, target=50)
    conn.execute("UPDATE QA_SamplingRuns SET Status = 'closed' WHERE RunId = ?", (run["RunId"],))
    assert repo.repair_run(conn, run["RunId"])["run_reopened"] is False


def test_reconcile_will_not_close_an_undispatched_run(tmp_path, monkeypatch):
    """
    The first live run planned yesterday and closed it seconds later with zero
    reserved, making 552 intervals undispatchable.
    """
    from app.automation.reconciler import reconcile_day

    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: pd.DataFrame(),
    )
    settings = _settings(300)
    day = reporting_day(DAY, settings.tz)
    conn = open_ledger(tmp_path / "undispatched.sqlite3")
    plan_day(conn, reporting_date=DAY, settings=settings,
             agents=[{"Agent_Email_Address": f"a{i}@x.net", "Agent_Name": "a", "Agent_id": i}
                     for i in range(5)])

    report = reconcile_day(conn, settings, reporting_date=DAY,
                           now=day.end_utc + timedelta(hours=1))
    assert not report.closed
    assert any("not been dispatched" in n for n in report.notes)
    assert repo.get_run(conn, DAY, settings.sampling_policy_version)["Status"] == "active"
