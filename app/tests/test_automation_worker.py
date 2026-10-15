"""
Tests for the worker: claiming, leases, fencing and the persistence protocol.

The protocol under test replaces the QA_Job_Id column we cannot add to DWH.
Its correctness depends entirely on the ORDER of its steps, so the tests here
simulate crashes between those steps rather than only testing the happy path —
a happy-path test would pass against a completely broken ordering.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest

from app.automation import repository as repo
from app.automation.config import AutomationSettings
from app.automation.db import open_ledger, utc_text
from app.automation.worker import process_job, work_once, worker_identity

UTC = timezone.utc
DAY = date(2026, 10, 29)


def _settings(**overrides) -> AutomationSettings:
    base = dict(
        automation_enabled=True, daily_target=10, reporting_timezone="Africa/Cairo",
        max_attempts=3, lease_duration_minutes=10, sampling_policy_version="v1",
    )
    base.update(overrides)
    return AutomationSettings(**base)


@pytest.fixture()
def ledger(tmp_path):
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    yield conn
    conn.close()


@pytest.fixture()
def job(ledger):
    """A run with a single pending job."""
    now = utc_text()
    run_id = str(uuid.uuid4())
    ledger.execute(
        """
        INSERT INTO QA_SamplingRuns
            (RunId, ReportingDate, ReportingTimezone, DayStartUtc, DayEndUtc,
             SamplingPolicyVersion, BaseSeed, DailyTarget, SamplingIntervalMinutes,
             SourceGraceMinutes, ExcludeAdjacentChats, MinChatGapMinutes,
             Status, CreatedAtUtc, UpdatedAtUtc)
        VALUES (?, '2026-10-29', 'Africa/Cairo', '2026-10-28T21:00:00.000Z',
                '2026-10-29T22:00:00.000Z', 'v1', '42', 10, 120, 20, 1, 0,
                'active', ?, ?)
        """,
        (run_id, now, now),
    )
    job_id = str(uuid.uuid4())
    ledger.execute(
        """
        INSERT INTO QA_EvaluationJobs
            (JobId, RunId, SnapshotId, ConversationId, AgentEmailAddress, IntervalId,
             SamplingPolicyVersion, ReportingDate, Status, CreatedAtUtc)
        VALUES (?, ?, ?, 'conv-1', 'agent@x.net', 'i1', 'v1', '2026-10-29', 'pending', ?)
        """,
        (job_id, run_id, str(uuid.uuid4()), now),
    )
    return job_id


# ── Claiming, leases and fencing ─────────────────────────────────────────────

def test_claim_marks_the_job_running_with_a_lease(ledger, job):
    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    assert claimed["JobId"] == job
    assert claimed["Status"] == "running"
    assert claimed["LeaseOwner"] == "w1"
    assert claimed["FenceToken"] is not None
    assert claimed["Attempts"] == 1


def test_a_running_job_cannot_be_claimed_twice(ledger, job):
    assert repo.claim_job(ledger, owner="w1", lease_minutes=10) is not None
    assert repo.claim_job(ledger, owner="w2", lease_minutes=10) is None


def test_expired_lease_returns_the_job_to_the_queue(ledger, job):
    first = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    later = datetime.now(UTC) + timedelta(minutes=30)

    assert repo.recover_expired_leases(ledger, later) == 1
    second = repo.claim_job(ledger, owner="w2", lease_minutes=10, now=later)
    assert second is not None
    assert second["LeaseOwner"] == "w2"
    assert second["FenceToken"] > first["FenceToken"]


def test_a_superseded_worker_cannot_complete_the_job(ledger, job):
    """
    The fencing guarantee: a worker whose lease expired and whose job was
    reclaimed must not be able to write its result over the replacement's.
    """
    stale = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    later = datetime.now(UTC) + timedelta(minutes=30)
    repo.recover_expired_leases(ledger, later)
    fresh = repo.claim_job(ledger, owner="w2", lease_minutes=10, now=later)

    assert repo.complete_job(ledger, job, stale["FenceToken"]) is False
    assert repo.complete_job(ledger, job, fresh["FenceToken"]) is True
    assert ledger.execute(
        "SELECT Status FROM QA_EvaluationJobs WHERE JobId = ?", (job,)
    ).fetchone()[0] == "completed"


def test_heartbeat_fails_once_superseded(ledger, job):
    stale = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    assert repo.heartbeat(ledger, job, stale["FenceToken"], 10) is True

    later = datetime.now(UTC) + timedelta(minutes=30)
    repo.recover_expired_leases(ledger, later)
    repo.claim_job(ledger, owner="w2", lease_minutes=10, now=later)

    assert repo.heartbeat(ledger, job, stale["FenceToken"], 10) is False


def test_worker_identity_names_host_and_pid():
    assert ":" in worker_identity()


# ── Failure classification and retry ─────────────────────────────────────────

def test_transient_failure_schedules_a_retry(ledger, job):
    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    status = repo.fail_job(
        ledger, job, claimed["FenceToken"],
        error="provider timed out", failure_class="transient", max_attempts=3,
    )
    assert status == "retry_wait"
    row = ledger.execute("SELECT NextAttemptUtc, LeaseOwner FROM QA_EvaluationJobs WHERE JobId = ?",
                         (job,)).fetchone()
    assert row["NextAttemptUtc"] is not None
    assert row["LeaseOwner"] is None       # lease released for the next worker


def test_permanent_failure_does_not_retry(ledger, job):
    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    status = repo.fail_job(
        ledger, job, claimed["FenceToken"],
        error="invalid api key", failure_class="permanent", max_attempts=3,
    )
    assert status == "failed"


def test_credit_exhaustion_does_not_consume_job_attempts(ledger, job):
    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    status = repo.fail_job(
        ledger, job, claimed["FenceToken"],
        error="OpenRouter insufficient credits (HTTP 402)",
        failure_class="quota", max_attempts=3,
    )
    row = ledger.execute(
        "SELECT Status, Attempts, NextAttemptUtc FROM QA_EvaluationJobs WHERE JobId = ?",
        (job,),
    ).fetchone()
    assert status == "quota_wait"
    assert row["Status"] == "retry_wait"
    assert row["Attempts"] == 0
    assert row["NextAttemptUtc"] is not None


def test_attempts_are_exhausted_then_the_job_fails(ledger, job):
    for expected in ("retry_wait", "retry_wait", "failed"):
        claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10,
                                 now=datetime.now(UTC) + timedelta(hours=1))
        assert claimed is not None
        status = repo.fail_job(
            ledger, job, claimed["FenceToken"],
            error="timeout", failure_class="transient", max_attempts=3,
        )
        assert status == expected


def test_backoff_grows_between_attempts(ledger, job):
    first = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    repo.fail_job(ledger, job, first["FenceToken"], error="t",
                  failure_class="transient", max_attempts=5, backoff_base_seconds=10)
    delay_one = _next_attempt(ledger, job) - datetime.now(UTC)

    second = repo.claim_job(ledger, owner="w1", lease_minutes=10,
                            now=datetime.now(UTC) + timedelta(hours=1))
    repo.fail_job(ledger, job, second["FenceToken"], error="t",
                  failure_class="transient", max_attempts=5, backoff_base_seconds=10)
    delay_two = _next_attempt(ledger, job) - datetime.now(UTC)

    assert delay_two > delay_one


def _next_attempt(conn, job_id):
    from app.automation.db import parse_utc
    return parse_utc(conn.execute(
        "SELECT NextAttemptUtc FROM QA_EvaluationJobs WHERE JobId = ?", (job_id,)
    ).fetchone()[0])


# ── The persistence protocol ─────────────────────────────────────────────────

class FakeDWH:
    """
    A stand-in for DWH.AI.Call_QA_Results that records inserted versions.

    ``rows`` is a list of (call_id, analysis_version, created_at).
    """

    def __init__(self):
        self.rows: list[tuple[str, int, datetime]] = []

    def insert(self, call_id, created_at=None):
        version = max((v for c, v, _ in self.rows if c == call_id), default=0) + 1
        self.rows.append((call_id, version, created_at or datetime.now(UTC)))
        return version


def test_recovery_adopts_an_orphaned_insert(ledger, job, monkeypatch):
    """
    The crash that the whole protocol exists for: the DWH insert landed, then
    the process died before the link was written. A retry must adopt the
    existing row, NOT run inference again and insert a second version.
    """
    from app.automation import worker

    attempt_start = datetime.now(UTC) - timedelta(minutes=5)
    repo.begin_persist_attempt(
        ledger, job_id=job, call_id="conv-1",
        pre_insert_max_version=4, attempt_started_utc=attempt_start,
    )

    monkeypatch.setattr(
        worker, "_find_orphaned_result",
        lambda engine, call_id, link: {"analysis_version": 5, "rows_inserted": 3},
    )

    inference_calls = []

    class NeverCalled:
        llm_client = type("c", (), {"model": "m"})()

        async def analyze(self, call, *, job_id=None):
            inference_calls.append(job_id)
            raise AssertionError("inference must not run for a recovered job")

    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    report = worker.WorkerReport()
    asyncio.run(process_job(ledger, None, None, NeverCalled(), claimed, _settings(), report))

    assert inference_calls == []
    assert report.recovered == 1

    link = repo.get_result_link(ledger, job)
    assert link["Status"] == "linked"
    assert link["AnalysisVersion"] == 5
    assert link["LinkMethod"] == "recovered"
    assert ledger.execute(
        "SELECT Status FROM QA_EvaluationJobs WHERE JobId = ?", (job,)
    ).fetchone()[0] == "completed"


def test_an_already_linked_job_is_never_re_evaluated(ledger, job):
    """Cheapest protection: if the link says done, do not call the LLM at all."""
    from app.automation import worker

    repo.begin_persist_attempt(
        ledger, job_id=job, call_id="conv-1", pre_insert_max_version=0,
        attempt_started_utc=datetime.now(UTC),
    )
    repo.complete_persist(ledger, job_id=job, analysis_version=1,
                          rows_inserted=2, method="inserted")

    class NeverCalled:
        llm_client = type("c", (), {"model": "m"})()

        async def analyze(self, call, *, job_id=None):
            raise AssertionError("inference must not run for a linked job")

    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    report = worker.WorkerReport()
    asyncio.run(process_job(ledger, None, None, NeverCalled(), claimed, _settings(), report))

    assert report.skipped_already_done == 1
    assert ledger.execute(
        "SELECT Status FROM QA_EvaluationJobs WHERE JobId = ?", (job,)
    ).fetchone()[0] == "completed"


def test_intent_is_committed_before_any_insert_can_happen(ledger, job, monkeypatch):
    """
    Ordering test. The 'attempting' row must be durable BEFORE inference runs,
    because inference is what triggers the DWH insert. If the write happened
    afterwards, a crash during inference would leave no recovery evidence.
    """
    from app.automation import worker

    observed: dict[str, object] = {}

    class RecordingAnalyzer:
        llm_client = type("c", (), {"model": "m"})()

        async def analyze(self, call, *, job_id=None):
            # At this moment the intent must already be on disk.
            link = repo.get_result_link(ledger, job_id)
            observed["status_at_inference_time"] = link["Status"] if link else None
            observed["pre_version"] = link["PreInsertMaxVersion"] if link else None
            raise RuntimeError("crash during inference")

    monkeypatch.setattr(worker, "_current_max_version", lambda engine, call_id: 7)
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_transcripts_by_ids",
        lambda handler, ids: _fake_frame(),
    )
    monkeypatch.setattr(
        "app.CM.candidate_queries.rows_for_conversation",
        lambda frame, cid: _fake_rows(),
    )
    monkeypatch.setattr(
        "app.models.input.CallTranscript.from_cm_rows",
        classmethod(lambda cls, rows: _fake_call()),
    )

    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    report = worker.WorkerReport()
    asyncio.run(process_job(ledger, object(), object(), RecordingAnalyzer(), claimed, _settings(), report))

    assert observed["status_at_inference_time"] == "attempting"
    assert observed["pre_version"] == 7
    assert report.retried == 1


def test_committed_insert_is_linked_when_graph_loses_context(ledger, job, monkeypatch):
    """LangGraph runs save nodes in child tasks, so ContextVar writes may not return."""
    from app.automation import worker
    from app.services.evaluation_service import EvaluationOutcome

    monkeypatch.setattr(worker, "_current_max_version", lambda engine, call_id: 4)
    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_transcripts_by_ids",
        lambda handler, ids: _fake_frame(),
    )
    monkeypatch.setattr(
        "app.CM.candidate_queries.rows_for_conversation",
        lambda frame, cid: _fake_rows(),
    )
    monkeypatch.setattr(
        "app.models.input.CallTranscript.from_cm_rows",
        classmethod(lambda cls, rows: _fake_call()),
    )

    async def evaluated(_analyzer, _call, *, job_id):
        return EvaluationOutcome(
            result=object(), persisted=False,
            error="evaluation succeeded but no DWH row was recorded",
            failure_class="transient",
        )

    def read_back(_engine, call_id, link):
        assert call_id == "conv-1"
        assert link["Status"] == "attempting"
        assert link["PreInsertMaxVersion"] == 4
        return {"analysis_version": 5, "rows_inserted": 2}

    monkeypatch.setattr(worker, "evaluate_conversation", evaluated)
    monkeypatch.setattr(worker, "_find_orphaned_result", read_back)
    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    report = worker.WorkerReport()
    analyzer = type("Analyzer", (), {"llm_client": type("Client", (), {"model": "test"})()})()
    asyncio.run(process_job(ledger, object(), object(), analyzer, claimed, _settings(), report))

    assert report.completed == 1
    assert repo.get_result_link(ledger, job)["AnalysisVersion"] == 5
    assert ledger.execute(
        "SELECT Status FROM QA_EvaluationJobs WHERE JobId = ?", (job,)
    ).fetchone()[0] == "completed"


def _fake_frame():
    import pandas as pd
    return pd.DataFrame([{"UniqueId": "conv-1"}])


def _fake_rows():
    return [{"UniqueId": "conv-1"}]


def _fake_call():
    from app.models.input import CallTranscript
    return CallTranscript(
        call_id="conv-1", agent_name="A", Patient_Phone="123456789",
        call_date="2026-10-29", call_duration_seconds=60, department="Helpdesk",
        transcript="Agent: hello", message_count=1,
    )


@pytest.mark.parametrize("code", [0, 2])
def test_raw_cm_answer_state_code_does_not_block_a_chat(code):
    from app.models.input import CallTranscript

    call = _fake_call().model_copy(update={"answer_state": None})
    payload = call.model_dump()
    payload["answer_state"] = code
    assert CallTranscript.model_validate(payload).answer_state == str(code)


def test_missing_transcript_fails_the_job_rather_than_evaluating_nothing(ledger, job, monkeypatch):
    from app.automation import worker

    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_transcripts_by_ids",
        lambda handler, ids: __import__("pandas").DataFrame(),
    )
    monkeypatch.setattr(
        "app.CM.candidate_queries.rows_for_conversation", lambda frame, cid: [],
    )

    class NeverCalled:
        llm_client = type("c", (), {"model": "m"})()

        async def analyze(self, call, *, job_id=None):
            raise AssertionError("must not evaluate without a transcript")

    claimed = repo.claim_job(ledger, owner="w1", lease_minutes=10)
    report = worker.WorkerReport()
    asyncio.run(process_job(ledger, object(), object(), NeverCalled(), claimed, _settings(), report))

    status = ledger.execute("SELECT Status FROM QA_EvaluationJobs WHERE JobId = ?", (job,)).fetchone()[0]
    assert status in ("retry_wait", "failed")
    assert repo.get_result_link(ledger, job) is None   # no intent recorded, nothing to recover


def test_work_once_drains_the_queue_and_recovers_leases(ledger, job, monkeypatch):
    from app.automation import worker

    monkeypatch.setattr(
        "app.CM.candidate_queries.fetch_transcripts_by_ids",
        lambda handler, ids: __import__("pandas").DataFrame(),
    )
    monkeypatch.setattr("app.CM.candidate_queries.rows_for_conversation", lambda frame, cid: [])

    class Analyzer:
        llm_client = type("c", (), {"model": "m"})()

        async def analyze(self, call, *, job_id=None):
            raise AssertionError("not reached")

    report = asyncio.run(work_once(ledger, object(), object(), Analyzer(), _settings()))
    assert report.claimed == 1
    assert repo.claim_job(ledger, owner="w2", lease_minutes=10) is None   # queue drained


def test_work_once_pauses_after_credit_exhaustion(ledger, job, monkeypatch):
    from app.automation import worker

    second_job = str(uuid.uuid4())
    ledger.execute(
        "INSERT INTO QA_EvaluationJobs "
        "(JobId,RunId,SnapshotId,ConversationId,AgentEmailAddress,IntervalId,"
        "SamplingPolicyVersion,ReportingDate,Status,CreatedAtUtc) "
        "SELECT ?,RunId,?,? ,AgentEmailAddress,IntervalId,"
        "SamplingPolicyVersion,ReportingDate,'pending',CreatedAtUtc "
        "FROM QA_EvaluationJobs WHERE JobId=?",
        (second_job, str(uuid.uuid4()), "conv-2", job),
    )

    async def exhausted(conn, _cm, _dwh, _analyzer, claimed, settings, report):
        status = repo.fail_job(
            conn, claimed["JobId"], claimed["FenceToken"],
            error="OpenRouter insufficient credits (HTTP 402)",
            failure_class="quota", max_attempts=settings.max_attempts,
        )
        worker._count(report, status)

    monkeypatch.setattr(worker, "process_job", exhausted)
    report = asyncio.run(work_once(ledger, object(), object(), object(), _settings()))
    assert report.claimed == 1
    assert report.quota_paused is True
    assert ledger.execute(
        "SELECT COUNT(*) FROM QA_EvaluationJobs WHERE Status='pending'"
    ).fetchone()[0] == 1
