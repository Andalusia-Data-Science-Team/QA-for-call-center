"""
End-to-end: plan a reporting day, dispatch every interval, check the result.

CM_DWH is replaced by a fake that returns a deterministic conversation stream,
so these tests exercise the real planner, dispatcher, sampler, eligibility
rules and ledger without touching a database we do not control.

The reporting day used throughout is 2026-10-29 — Cairo's real 25-hour
DST-transition day — so the irregular-day handling is on the main path of
every test here rather than in a corner case.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.automation import repository as repo
from app.automation.calendar import reporting_day
from app.automation.config import AutomationSettings
from app.automation.db import open_ledger, parse_utc
from app.automation.dispatcher import dispatch_once
from app.automation.planner import plan_day
from app.automation.reconciler import reconcile_day

UTC = timezone.utc
DAY = date(2026, 10, 29)          # 25 hours in Africa/Cairo
AGENTS = [f"agent{i}@andalusiagroup.net" for i in range(1, 5)]


def _settings(**overrides) -> AutomationSettings:
    base = dict(
        automation_enabled=True,
        daily_target=40,
        reporting_timezone="Africa/Cairo",
        sampling_interval_minutes=120,
        source_grace_minutes=20,
        sampling_seed="42",
        sampling_policy_version="v1",
        exclude_adjacent_chats=True,
    )
    base.update(overrides)
    return AutomationSettings(**base)


class FakeCM:
    """
    Stands in for a connected CM_DWH handler.

    Generates a reproducible conversation stream: each agent handles one
    conversation every 20 minutes across the whole day. Conversation ids are
    derived from the agent and the timestamp, so the same fake always produces
    the same data — which is what lets the determinism test mean something.
    """

    def __init__(self, day, agents=AGENTS, *, closed=True, cadence_minutes=20):
        self.day = day
        self.agents = agents
        self.closed = closed
        self.cadence = cadence_minutes
        self.query_count = 0

    def rows(self, start, end):
        rows = []
        for agent in self.agents:
            moment = self.day.start_utc
            while moment < self.day.end_utc:
                if start <= moment < end:
                    cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{agent}|{moment.isoformat()}"))
                    rows.append({
                        "ConversationId": cid,
                        "AgentEmail": agent,
                        "AgentFullName": agent.split("@")[0],
                        "AgentPersonId": str(uuid.uuid5(uuid.NAMESPACE_DNS, agent)),
                        "HandlingTimeUtc": moment.replace(tzinfo=None),
                        "InboxOwnerEmail": agent,
                        "OwnerDiffersFromHandler": 0,
                        "ConversationState": "Closed",
                        "AnswerState": 1,
                        "ArchiveDateTimeUtc": (
                            moment.replace(tzinfo=None) + timedelta(minutes=5) if self.closed else None
                        ),
                        "StartDateTimeUtc": moment.replace(tzinfo=None) - timedelta(minutes=10),
                        "LastUpdatedUtc": moment.replace(tzinfo=None) + timedelta(minutes=5),
                        "HumanMessageCount": 4,
                        "TotalMessageCount": 9,
                        "WasForwarded": 0,
                        "WebStoreName": "AHJ",
                    })
                moment += timedelta(minutes=self.cadence)
        return pd.DataFrame(rows)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """A ledger, a fake CM, and the dispatcher wired to the fake."""
    settings = _settings()
    day = reporting_day(DAY, settings.tz)
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    fake = FakeCM(day)

    def fake_fetch(handler, start, end):
        handler.query_count += 1
        return handler.rows(start, end)

    monkeypatch.setattr("app.CM.candidate_queries.fetch_day_candidates", fake_fetch)

    agents_rows = [{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                   for i, a in enumerate(AGENTS, 1)]

    yield conn, fake, settings, agents_rows, day
    conn.close()


def _plan_and_dispatch(conn, fake, settings, agents_rows, day):
    plan = plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)
    report = dispatch_once(conn, fake, settings, now=day.end_utc + timedelta(hours=1))
    return plan, report


# ── Planning ─────────────────────────────────────────────────────────────────

def test_plan_covers_the_whole_25_hour_day(env):
    conn, fake, settings, agents_rows, day = env
    plan = plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)

    assert plan.created
    assert plan.agent_count == len(AGENTS)
    # Each agent's whole-day period splits into 13 intervals on a 25-hour day.
    assert plan.interval_count == len(AGENTS) * 13
    assert plan.allocated == settings.daily_target
    assert any("25-hour day" in n for n in plan.notes)


def test_plan_is_idempotent(env):
    conn, fake, settings, agents_rows, day = env
    first = plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)
    second = plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)
    assert second.run_id == first.run_id
    assert not second.created


def test_plan_fails_loudly_with_no_roster(env):
    conn, fake, settings, agents_rows, day = env
    with pytest.raises(ValueError, match="setup dependency"):
        plan_day(conn, reporting_date=DAY, settings=settings, agents=[])


def test_quota_is_split_equally_across_agents(env):
    conn, fake, settings, agents_rows, day = env
    plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    per_agent = conn.execute(
        "SELECT AgentEmailAddress, SUM(OriginalQuota) FROM QA_AgentQuotas "
        " WHERE RunId = ? GROUP BY AgentEmailAddress", (run["RunId"],)
    ).fetchall()
    assert {row[1] for row in per_agent} == {settings.daily_target // len(AGENTS)}


# ── Dispatch ─────────────────────────────────────────────────────────────────

def test_failed_shared_window_is_queried_once_then_retried_next_pass(env, monkeypatch):
    conn, fake, settings, agents_rows, day = env
    plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)
    cutoff = day.start_utc + timedelta(hours=2, minutes=settings.source_grace_minutes)
    calls = 0

    def broken_fetch(_handler, _start, _end):
        nonlocal calls
        calls += 1
        raise ConnectionError("Communication link failure")

    monkeypatch.setattr("app.CM.candidate_queries.fetch_day_candidates", broken_fetch)
    first = dispatch_once(conn, fake, settings, now=cutoff, backfill=False)
    assert calls == 1
    assert first.intervals_processed == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals WHERE Status = 'pending' "
        "AND SnapshotCutoffUtc <= ?", (cutoff.isoformat(timespec="milliseconds").replace("+00:00", "Z"),)
    ).fetchone()[0] == len(AGENTS)

    def recovered_fetch(_handler, start, end):
        nonlocal calls
        calls += 1
        return fake.rows(start, end)

    monkeypatch.setattr("app.CM.candidate_queries.fetch_day_candidates", recovered_fetch)
    second = dispatch_once(conn, fake, settings, now=cutoff, backfill=False)
    assert calls == 2
    assert second.intervals_processed == len(AGENTS)


def test_dispatch_reserves_the_whole_budget(env):
    conn, fake, settings, agents_rows, day = env
    plan, report = _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    assert report.intervals_processed == plan.interval_count
    assert report.jobs_reserved == settings.daily_target
    assert not report.errors


def test_dispatch_never_exceeds_the_daily_budget(env):
    """The budget is the point of the whole exercise."""
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    assert repo.reserved_count(conn, run["RunId"]) <= settings.daily_target


def test_dispatch_twice_does_not_double_reserve(env):
    """A duplicate scheduler tick must be a no-op, not a second day's worth."""
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    after_first = repo.reserved_count(conn, run["RunId"])

    dispatch_once(conn, fake, settings, now=day.end_utc + timedelta(hours=2))
    assert repo.reserved_count(conn, run["RunId"]) == after_first


def test_one_job_per_conversation(env):
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    total, distinct = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT ConversationId) FROM QA_EvaluationJobs"
    ).fetchone()
    assert total == distinct


def test_every_agent_is_covered(env):
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    assert repo.uncovered_agents(conn, run["RunId"]) == []


def test_selected_conversations_are_never_adjacent(env):
    """The nonconsecutive rule, checked across interval boundaries."""
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    for agent in AGENTS:
        positions = sorted(
            row[0] for row in conn.execute(
                "SELECT SequencePosition FROM QA_SamplingCandidates "
                " WHERE AgentEmailAddress = ? AND IsSelected = 1", (agent,)
            )
        )
        assert all(b - a > 1 for a, b in zip(positions, positions[1:])), (
            f"{agent} has adjacent selections: {positions}"
        )


def test_candidates_are_retained_with_rejection_reasons(env):
    """Rejected candidates are the audit trail; losing them loses the 'why'."""
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    rejected = conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingCandidates WHERE IsSelected = 0"
    ).fetchone()[0]
    assert rejected > 0
    unexplained = conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingCandidates "
        " WHERE IsSelected = 0 AND SelectionReason IS NULL"
    ).fetchone()[0]
    assert unexplained == 0


def test_open_conversations_are_excluded_with_a_reason(env):
    conn, fake, settings, agents_rows, day = env
    fake.closed = False
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    assert repo.reserved_count(
        conn, repo.get_run(conn, DAY, settings.sampling_policy_version)["RunId"]
    ) == 0
    reasons = {row[0] for row in conn.execute(
        "SELECT DISTINCT EligibilityReason FROM QA_SamplingCandidates WHERE IsEligible = 0"
    )}
    assert reasons == {"conversation_not_closed"}


def test_sparse_day_reports_a_shortfall_rather_than_inventing_coverage(env):
    conn, fake, settings, agents_rows, day = env
    fake.cadence = 600          # one conversation every 10 hours per agent
    plan, report = _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    assert report.jobs_reserved < settings.daily_target
    assert report.shortfalls
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    coverage = repo.run_coverage(conn, run["RunId"])
    assert coverage["ReservationDeficit"] > 0


def test_missed_intervals_are_picked_up_later(env):
    """Scheduler downtime must not leave a permanent hole in the day."""
    conn, fake, settings, agents_rows, day = env
    plan_day(conn, reporting_date=DAY, settings=settings, agents=agents_rows)

    # Dispatch as if we only woke up a third of the way through the day.
    early = dispatch_once(conn, fake, settings, now=day.start_utc + timedelta(hours=8))
    assert early.intervals_processed > 0

    # Now wake up after the day ended: every remaining interval is still due.
    late = dispatch_once(conn, fake, settings, now=day.end_utc + timedelta(hours=1))
    assert late.intervals_processed > 0

    still_pending = conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals WHERE Status IN ('pending','snapshotting')"
    ).fetchone()[0]
    assert still_pending == 0


# ── Determinism, the acceptance criterion ────────────────────────────────────

def test_same_inputs_produce_byte_identical_selections(tmp_path, monkeypatch):
    """
    Architecture §11: frozen inputs produce identical selected ids across
    repeated runs AND process restarts. Two independent ledgers, same seed.
    """
    settings = _settings()
    day = reporting_day(DAY, settings.tz)
    agents_rows = [{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                   for i, a in enumerate(AGENTS, 1)]

    def run_once(path):
        conn = open_ledger(path)
        fake = FakeCM(day)
        monkeypatch.setattr(
            "app.CM.candidate_queries.fetch_day_candidates",
            lambda handler, start, end: handler.rows(start, end),
        )
        plan_day(conn, reporting_date=DAY, settings=settings,
                 agents=agents_rows)
        dispatch_once(conn, fake, settings, now=day.end_utc + timedelta(hours=1))
        selected = [r[0] for r in conn.execute(
            "SELECT ConversationId FROM QA_SamplingCandidates "
            " WHERE IsSelected = 1 ORDER BY ConversationId"
        )]
        conn.close()
        return selected

    first = run_once(tmp_path / "a.sqlite3")
    second = run_once(tmp_path / "b.sqlite3")

    assert first == second
    assert len(first) == settings.daily_target


def test_a_different_seed_produces_a_different_selection(tmp_path, monkeypatch):
    """Otherwise the seed is decorative and 'reproducible' means nothing."""
    day = reporting_day(DAY, _settings().tz)
    agents_rows = [{"Agent_Email_Address": a, "Agent_Name": a, "Agent_id": i}
                   for i, a in enumerate(AGENTS, 1)]
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_day_candidates",
        lambda handler, start, end: handler.rows(start, end),
    )

    def run_with_seed(path, seed):
        settings = _settings(sampling_seed=seed)
        conn = open_ledger(path)
        plan_day(conn, reporting_date=DAY, settings=settings,
                 agents=agents_rows)
        dispatch_once(conn, FakeCM(day), settings, now=day.end_utc + timedelta(hours=1))
        selected = {r[0] for r in conn.execute(
            "SELECT ConversationId FROM QA_SamplingCandidates WHERE IsSelected = 1"
        )}
        conn.close()
        return selected

    assert run_with_seed(tmp_path / "s1.sqlite3", "42") != run_with_seed(tmp_path / "s2.sqlite3", "99")


# ── Reconciliation ───────────────────────────────────────────────────────────

def test_reconcile_reports_outstanding_work_and_keeps_the_run_open(env):
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)

    report = reconcile_day(conn, settings, reporting_date=DAY,
                           now=day.end_utc + timedelta(hours=1))
    assert report.outstanding == settings.daily_target    # nothing evaluated yet
    assert not report.closed
    assert any("still pending" in n for n in report.notes)


def test_reconcile_closes_a_finished_day(env):
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    conn.execute("UPDATE QA_EvaluationJobs SET Status = 'completed', CompletedAtUtc = '2026-10-29T22:00:00.000Z'")

    report = reconcile_day(conn, settings, reporting_date=DAY,
                           now=day.end_utc + timedelta(hours=1))
    assert report.closed
    assert report.completed == settings.daily_target
    assert repo.get_run(conn, DAY, settings.sampling_policy_version)["Status"] == "closed"


def test_reconcile_records_failures_without_substituting_conversations(env):
    """
    Once reserved, a failed job is retried — never swapped for an easier
    conversation. Substitution would bias the sample against hard chats.
    """
    conn, fake, settings, agents_rows, day = env
    _plan_and_dispatch(conn, fake, settings, agents_rows, day)
    run = repo.get_run(conn, DAY, settings.sampling_policy_version)
    before = repo.reserved_count(conn, run["RunId"])

    conn.execute("UPDATE QA_EvaluationJobs SET Status = 'completed'")
    conn.execute(
        "UPDATE QA_EvaluationJobs SET Status = 'failed', LastError = 'provider refused' "
        " WHERE JobId IN (SELECT JobId FROM QA_EvaluationJobs LIMIT 3)"
    )

    report = reconcile_day(conn, settings, reporting_date=DAY,
                           now=day.end_utc + timedelta(hours=1))
    assert report.failed == 3
    assert any("NOT replaced" in n for n in report.notes)
    assert repo.reserved_count(conn, run["RunId"]) == before
