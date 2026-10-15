"""
Tests for the local SQLite ledger (app.automation.db + the schema it applies).

The ledger exists because we have no DDL rights on DWH. These tests therefore
focus on the constraints that carry the correctness guarantees the original
SQL Server design would have enforced — above all the one-job-per-conversation
budget index and the result-link protocol that replaces the QA_Job_Id column
we cannot add.
"""

import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app.automation.calendar import day_intervals, reporting_day
from app.automation.db import (
    SCHEMA_VERSION,
    apply_schema,
    next_fence_token,
    open_ledger,
    parse_utc,
    transaction,
    utc_text,
)

UTC = timezone.utc


@pytest.fixture()
def ledger(tmp_path):
    conn = open_ledger(tmp_path / "ledger.sqlite3")
    yield conn
    conn.close()


def _new_run(conn, *, reporting_date="2026-10-29", policy="v1", target=300, status="active") -> str:
    """Insert a run row and return its id. Cairo's 25-hour day by default."""
    run_id = str(uuid.uuid4())
    now = utc_text()
    conn.execute(
        """
        INSERT INTO QA_SamplingRuns
            (RunId, ReportingDate, ReportingTimezone, DayStartUtc, DayEndUtc,
             SamplingPolicyVersion, BaseSeed, DailyTarget, SamplingIntervalMinutes,
             SourceGraceMinutes, ExcludeAdjacentChats, MinChatGapMinutes,
             Status, CreatedAtUtc, UpdatedAtUtc)
        VALUES (?, ?, 'Africa/Cairo', '2026-10-28T21:00:00.000Z', '2026-10-29T22:00:00.000Z',
                ?, '42', ?, 120, 20, 1, 0, ?, ?, ?)
        """,
        (run_id, reporting_date, policy, target, status, now, now),
    )
    return run_id


def _new_job(conn, run_id, *, conversation_id=None, policy="v1", status="pending") -> str:
    job_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO QA_EvaluationJobs
            (JobId, RunId, SnapshotId, ConversationId, AgentEmailAddress, IntervalId,
             SamplingPolicyVersion, ReportingDate, Status, CreatedAtUtc)
        VALUES (?, ?, ?, ?, 'agent@andalusiagroup.net', '20261029T0600Z/20261029T0800Z',
                ?, '2026-10-29', ?, ?)
        """,
        (job_id, run_id, str(uuid.uuid4()), conversation_id or str(uuid.uuid4()),
         policy, status, utc_text()),
    )
    return job_id


# ── Schema bootstrap ─────────────────────────────────────────────────────────

def test_schema_creates_every_expected_object(ledger):
    names = {r[0] for r in ledger.execute(
        "SELECT name FROM sqlite_master WHERE name LIKE 'QA%'"
    )}
    assert {
        "QA_SamplingRuns", "QA_AgentQuotas", "QA_SamplingIntervals",
        "QA_SamplingCandidates", "QA_EvaluationJobs", "QA_ResultLinks",
        "QA_FenceCounter", "QA_RunCoverage", "QA_SchemaVersion",
    } <= names


def test_migration_drops_the_shift_provenance_column(tmp_path):
    """
    A ledger created before the shift model was removed must migrate cleanly
    rather than failing on a column the code no longer writes.
    """
    import sqlite3 as sq
    from app.automation.db import apply_schema, connect

    path = tmp_path / "old.sqlite3"
    conn = connect(path)
    apply_schema(conn)
    # Recreate the pre-002 shape by adding the column back.
    conn.execute("ALTER TABLE QA_AgentQuotas ADD COLUMN WorkPeriodSource TEXT")
    assert any(r[1] == "WorkPeriodSource"
               for r in conn.execute("PRAGMA table_info(QA_AgentQuotas)"))

    assert apply_schema(conn) == SCHEMA_VERSION
    assert not any(r[1] == "WorkPeriodSource"
                   for r in conn.execute("PRAGMA table_info(QA_AgentQuotas)"))
    conn.close()


def test_schema_has_no_shift_concept(ledger):
    sql = (Path(__file__).resolve().parent.parent
           / "SQL" / "automation" / "sqlite" / "001_ledger.sql").read_text()
    assert "WorkPeriodSource" not in sql


def test_apply_schema_is_idempotent(ledger):
    assert apply_schema(ledger) == SCHEMA_VERSION
    assert apply_schema(ledger) == SCHEMA_VERSION
    assert ledger.execute("SELECT COUNT(*) FROM QA_SchemaVersion").fetchone()[0] == 1


def test_required_pragmas_are_set(ledger):
    assert ledger.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert ledger.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_newer_ledger_is_refused_rather_than_written_to(ledger):
    ledger.execute(
        "INSERT INTO QA_SchemaVersion (Version, AppliedAtUtc, Description) VALUES (?, ?, 'future')",
        (SCHEMA_VERSION + 1, utc_text()),
    )
    with pytest.raises(RuntimeError, match="newer ledger"):
        apply_schema(ledger)


def test_foreign_keys_are_enforced(ledger):
    """REFERENCES is decorative in SQLite unless the pragma is on."""
    with pytest.raises(sqlite3.IntegrityError):
        _new_job(ledger, str(uuid.uuid4()))  # run does not exist


# ── Timestamps ───────────────────────────────────────────────────────────────

def test_timestamp_text_sorts_chronologically():
    """
    The ledger compares timestamps as TEXT, so lexicographic order must equal
    chronological order — otherwise 'due intervals' queries silently miss rows.
    """
    base = datetime(2026, 10, 29, 21, 0, tzinfo=UTC)
    moments = [base + timedelta(seconds=s) for s in (0, 1, 59, 60, 3600, 86400)]
    rendered = [utc_text(m) for m in moments]
    assert rendered == sorted(rendered)


def test_utc_text_rejects_naive_datetimes():
    with pytest.raises(ValueError, match="tz-aware"):
        utc_text(datetime(2026, 10, 29, 21, 0))


def test_utc_text_normalises_non_utc_input_to_utc():
    from zoneinfo import ZoneInfo

    cairo_noon = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo("Africa/Cairo"))
    assert utc_text(cairo_noon) == "2026-01-15T10:00:00.000Z"   # UTC+2 in winter


def test_timestamp_roundtrip():
    moment = datetime(2026, 10, 29, 21, 0, 0, 123000, tzinfo=UTC)
    assert parse_utc(utc_text(moment)) == moment


# ── The budget guarantee ─────────────────────────────────────────────────────

def test_one_job_per_conversation_per_policy_version(ledger):
    """
    The index, not application logic, is what stops a second dispatcher pass
    from reserving the same conversation twice and blowing the daily budget.
    """
    run = _new_run(ledger)
    conversation = str(uuid.uuid4())
    _new_job(ledger, run, conversation_id=conversation)

    with pytest.raises(sqlite3.IntegrityError):
        _new_job(ledger, run, conversation_id=conversation)


def test_superseded_job_frees_the_conversation_for_reevaluation(ledger):
    """An intentional reevaluation campaign must be possible; an accidental one must not."""
    run = _new_run(ledger)
    conversation = str(uuid.uuid4())
    first = _new_job(ledger, run, conversation_id=conversation)

    ledger.execute("UPDATE QA_EvaluationJobs SET Status = 'superseded' WHERE JobId = ?", (first,))
    _new_job(ledger, run, conversation_id=conversation)  # now allowed

    assert ledger.execute(
        "SELECT COUNT(*) FROM QA_EvaluationJobs WHERE ConversationId = ?", (conversation,)
    ).fetchone()[0] == 2


def test_same_conversation_allowed_under_a_new_policy_version(ledger):
    run_v1 = _new_run(ledger, policy="v1")
    run_v2 = _new_run(ledger, policy="v2")
    conversation = str(uuid.uuid4())
    _new_job(ledger, run_v1, conversation_id=conversation, policy="v1")
    _new_job(ledger, run_v2, conversation_id=conversation, policy="v2")  # must not raise


def test_one_live_run_per_date_and_policy(ledger):
    _new_run(ledger, reporting_date="2026-10-29", policy="v1")
    with pytest.raises(sqlite3.IntegrityError):
        _new_run(ledger, reporting_date="2026-10-29", policy="v1")


def test_abandoned_run_frees_the_date(ledger):
    first = _new_run(ledger, reporting_date="2026-10-29", policy="v1")
    ledger.execute("UPDATE QA_SamplingRuns SET Status = 'abandoned' WHERE RunId = ?", (first,))
    _new_run(ledger, reporting_date="2026-10-29", policy="v1")  # must not raise


# ── Generated columns, against the real Cairo 25-hour day ────────────────────

def test_day_hours_is_computed_for_the_25_hour_cairo_day(ledger):
    _new_run(ledger)
    assert ledger.execute("SELECT DayHours FROM QA_SamplingRuns").fetchone()[0] == pytest.approx(25.0)


def test_interval_minutes_matches_the_calendar_module(ledger):
    """
    The generated column and app.automation.calendar must agree, or quota
    allocated proportionally to duration will not match the intervals stored.
    """
    run = _new_run(ledger)
    day = reporting_day(datetime(2026, 10, 29).date(), __import__("zoneinfo").ZoneInfo("Africa/Cairo"))
    intervals = day_intervals(day, 120)

    for interval in intervals:
        ledger.execute(
            """
            INSERT INTO QA_SamplingIntervals
                (RunId, AgentEmailAddress, IntervalId, IntervalIndex,
                 IntervalStartUtc, IntervalEndUtc, SnapshotCutoffUtc, Quota, CreatedAtUtc)
            VALUES (?, 'agent@andalusiagroup.net', ?, ?, ?, ?, ?, 15, ?)
            """,
            (run, interval.interval_id, interval.index,
             utc_text(interval.start_utc), utc_text(interval.end_utc),
             utc_text(interval.end_utc + timedelta(minutes=20)), utc_text()),
        )

    stored = [r[0] for r in ledger.execute(
        "SELECT IntervalMinutes FROM QA_SamplingIntervals ORDER BY IntervalIndex"
    )]
    assert stored == [i.minutes for i in intervals]
    assert stored == [120] * 12 + [60]     # the 25-hour day's short final interval


def test_duplicate_interval_snapshot_is_blocked(ledger):
    run = _new_run(ledger)
    args = (run, "agent@andalusiagroup.net", "20261029T0600Z/20261029T0800Z", 3,
            "2026-10-29T06:00:00.000Z", "2026-10-29T08:00:00.000Z",
            "2026-10-29T08:20:00.000Z", 15, utc_text())
    sql = """
        INSERT INTO QA_SamplingIntervals
            (RunId, AgentEmailAddress, IntervalId, IntervalIndex,
             IntervalStartUtc, IntervalEndUtc, SnapshotCutoffUtc, Quota, CreatedAtUtc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    ledger.execute(sql, args)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute(sql, args)


# ── STRICT typing and CHECK constraints ──────────────────────────────────────

def test_strict_tables_reject_a_mistyped_value(ledger):
    run = _new_run(ledger)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute("UPDATE QA_SamplingRuns SET DailyTarget = 'three hundred' WHERE RunId = ?", (run,))


def test_invalid_job_status_is_rejected(ledger):
    run = _new_run(ledger)
    job = _new_job(ledger, run)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute("UPDATE QA_EvaluationJobs SET Status = 'nearly_done' WHERE JobId = ?", (job,))


def test_rank_hash_must_be_a_full_sha256(ledger):
    run = _new_run(ledger)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute(
            """
            INSERT INTO QA_SamplingCandidates
                (SnapshotId, ConversationId, RunId, AgentEmailAddress, IntervalId,
                 AttributionRule, HandlingTimeUtc, SequencePosition, RankHash,
                 IsEligible, CreatedAtUtc)
            VALUES (?, ?, ?, 'a@b.net', 'i', 'last_human_message', ?, 1, 'tooshort', 1, ?)
            """,
            (str(uuid.uuid4()), str(uuid.uuid4()), run, utc_text(), utc_text()),
        )


# ── Fencing ──────────────────────────────────────────────────────────────────

def test_fence_tokens_are_monotonic(ledger):
    with transaction(ledger):
        tokens = [next_fence_token(ledger) for _ in range(5)]
    assert tokens == sorted(tokens)
    assert len(set(tokens)) == 5


def test_fence_counter_survives_reconnect(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    first = open_ledger(path)
    with transaction(first):
        issued = next_fence_token(first)
    first.close()

    second = open_ledger(path)
    with transaction(second):
        assert next_fence_token(second) > issued
    second.close()


# ── Transactions ─────────────────────────────────────────────────────────────

def test_transaction_rolls_back_on_error(ledger):
    run = _new_run(ledger)
    before = ledger.execute("SELECT COUNT(*) FROM QA_EvaluationJobs").fetchone()[0]

    with pytest.raises(RuntimeError):
        with transaction(ledger):
            _new_job(ledger, run)
            raise RuntimeError("boom")

    assert ledger.execute("SELECT COUNT(*) FROM QA_EvaluationJobs").fetchone()[0] == before


def test_transaction_commits_on_success(ledger):
    run = _new_run(ledger)
    with transaction(ledger):
        _new_job(ledger, run)
    assert ledger.execute("SELECT COUNT(*) FROM QA_EvaluationJobs").fetchone()[0] == 1


# ── The result-link protocol (what replaces the QA_Job_Id column) ────────────

def test_result_link_cannot_claim_linked_without_a_version(ledger):
    """
    A 'linked' row asserts that a specific Analysis_Version landed in DWH.
    Allowing that claim with a NULL version would make recovery unable to tell
    a completed persist from an abandoned one.
    """
    run = _new_run(ledger)
    job = _new_job(ledger, run)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute(
            """
            INSERT INTO QA_ResultLinks
                (JobId, CallId, Status, PreInsertMaxVersion, AttemptStartedUtc)
            VALUES (?, 'call-1', 'linked', 0, ?)
            """,
            (job, utc_text()),
        )


def test_result_link_is_one_per_job(ledger):
    run = _new_run(ledger)
    job = _new_job(ledger, run)
    sql = """
        INSERT INTO QA_ResultLinks
            (JobId, CallId, Status, PreInsertMaxVersion, AttemptStartedUtc)
        VALUES (?, 'call-1', 'attempting', 0, ?)
    """
    ledger.execute(sql, (job, utc_text()))
    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute(sql, (job, utc_text()))


def test_attempting_row_is_durable_evidence_for_crash_recovery(tmp_path):
    """
    The protocol's core claim: the 'attempting' row is committed BEFORE the
    DWH insert, so a process that dies immediately after the insert still
    leaves behind the PreInsertMaxVersion a retry needs in order to tell
    whether the insert landed.

    Simulated by committing the attempt, dropping the connection without any
    further write, and reopening.
    """
    path = tmp_path / "ledger.sqlite3"
    conn = open_ledger(path)
    run = _new_run(conn)
    job = _new_job(conn, run)

    with transaction(conn):
        conn.execute(
            """
            INSERT INTO QA_ResultLinks
                (JobId, CallId, Status, PreInsertMaxVersion, AttemptStartedUtc)
            VALUES (?, 'conv-abc', 'attempting', 4, ?)
            """,
            (job, utc_text(datetime(2026, 10, 29, 6, 30, tzinfo=UTC))),
        )
    conn.close()   # the crash

    recovered = open_ledger(path)
    row = recovered.execute(
        "SELECT Status, PreInsertMaxVersion, AttemptStartedUtc FROM QA_ResultLinks WHERE JobId = ?",
        (job,),
    ).fetchone()
    assert row["Status"] == "attempting"
    assert row["PreInsertMaxVersion"] == 4
    assert parse_utc(row["AttemptStartedUtc"]) == datetime(2026, 10, 29, 6, 30, tzinfo=UTC)
    recovered.close()


def test_link_method_records_how_the_link_was_established(ledger):
    run = _new_run(ledger)
    job = _new_job(ledger, run)
    ledger.execute(
        """
        INSERT INTO QA_ResultLinks
            (JobId, CallId, Status, PreInsertMaxVersion, AttemptStartedUtc,
             AnalysisVersion, RowsInserted, LinkedAtUtc, LinkMethod)
        VALUES (?, 'conv-abc', 'linked', 4, ?, 5, 3, ?, 'recovered')
        """,
        (job, utc_text(), utc_text()),
    )
    assert ledger.execute("SELECT LinkMethod FROM QA_ResultLinks").fetchone()[0] == "recovered"

    with pytest.raises(sqlite3.IntegrityError):
        ledger.execute("UPDATE QA_ResultLinks SET LinkMethod = 'guessed' WHERE JobId = ?", (job,))


# ── Coverage view ────────────────────────────────────────────────────────────

def test_coverage_view_separates_failures_from_completions(ledger):
    run = _new_run(ledger, target=5)
    for status in ("completed", "completed", "failed", "retry_wait", "pending"):
        _new_job(ledger, run, status=status)
    # A superseded job must not count against the budget.
    _new_job(ledger, run, status="superseded")

    row = ledger.execute("SELECT * FROM QA_RunCoverage WHERE RunId = ?", (run,)).fetchone()
    assert row["JobsReserved"] == 5
    assert row["JobsCompleted"] == 2
    assert row["JobsFailed"] == 1
    assert row["JobsAwaitingRetry"] == 1
    assert row["JobsOutstanding"] == 1
    assert row["ReservationDeficit"] == 0
    assert row["DayHours"] == pytest.approx(25.0)


def test_coverage_view_reports_a_reservation_deficit(ledger):
    run = _new_run(ledger, target=300)
    for _ in range(3):
        _new_job(ledger, run)
    row = ledger.execute("SELECT * FROM QA_RunCoverage WHERE RunId = ?", (run,)).fetchone()
    assert row["ReservationDeficit"] == 297
