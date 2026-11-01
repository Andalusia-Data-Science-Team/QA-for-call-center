"""
Tests for the CM_DWH access layer and the SQL it runs.

These cannot hit CM_DWH, so they test the two things that are checkable
without it and that the old path got wrong: that ids are BOUND rather than
interpolated, and that a shortfall between requested and retrieved
conversations is detected rather than silently swallowed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from pathlib import Path

import pandas as pd
import pytest

from app.CM import candidate_queries as cq
from app.CM.chat_sanitize import sanitize_df, sanitize_rows

UTC = timezone.utc
SQL_DIR = Path(__file__).resolve().parent.parent / "SQL" / "automation"

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE = re.compile(r"--[^\n]*")


def sql_code(name: str) -> str:
    """
    The SQL with its comments stripped.

    Necessary because these files document the very anti-patterns they avoid —
    'TOP (10)' and 'DATEADD(HOUR, 3' both appear in the explanatory headers.
    Asserting against the raw text would match the explanation, not the code.
    """
    text = (SQL_DIR / name).read_text()
    return _COMMENT_LINE.sub("", _COMMENT_BLOCK.sub("", text))


# ── The SQL itself ───────────────────────────────────────────────────────────

NEWFIX = SQL_DIR.parent / "NEWFIX_CM_Chat_Search.sql"


def _normalise(sql: str) -> str:
    return " ".join(sql.split())


def test_transcript_query_is_newfix_with_only_the_selection_changed():
    """
    The automation copy must build the same transcript the manual dashboard
    builds. Everything from TopConversations' consumer onwards — the message
    joins, the column list, the DATEADD presentation — must match NEWFIX,
    apart from the added MessageId column and its ORDER BY key.
    """
    ours = _normalise(sql_code("automation_chat_by_ids.sql"))
    theirs = _normalise(_COMMENT_LINE.sub("", _COMMENT_BLOCK.sub("", NEWFIX.read_text())))

    def tail(sql):
        return sql[sql.index("SELECT tc.*"):]

    expected = tail(theirs).replace(
        "m.SenderWasOwner,", "m.UniqueId AS MessageId, m.SenderWasOwner,", 1
    ).replace("tc.UniqueId, m.CreationDateTime;", "tc.UniqueId, m.CreationDateTime, m.UniqueId;")
    assert tail(ours) == expected

    # The ConvSummary CTE is NEWFIX's too, up to its WHERE clause.
    def conv_summary_head(sql):
        return sql[sql.index("WITH ConvSummary AS"):sql.index("WHERE", sql.index("WITH ConvSummary AS"))]
    assert conv_summary_head(ours) == conv_summary_head(theirs)


def test_transcript_query_selects_by_id_list_without_a_top():
    sql = sql_code("automation_chat_by_ids.sql")
    assert "c.UniqueId IN ({id_placeholders})" in sql
    assert sql.count("{id_placeholders}") == 1
    assert "TOP (10" not in sql and "TOP (100)" not in sql
    assert ":AgentEmail" not in sql and "@FilterDate" not in sql


def test_newfix_itself_is_untouched_by_the_automation():
    """The dashboard's query keeps its own filters; the automation never edits it."""
    sql = NEWFIX.read_text()
    assert ":AgentEmail" in sql and "@FilterDate" in sql
    assert "{id_placeholders}" not in sql


def test_candidate_query_attributes_by_inbox_owner_like_newfix():
    sql = _normalise(sql_code("automation_day_conversations.sql"))
    assert "c.InboxOwnerPersonUniqueId = owner.UniqueId" in sql
    assert "owner.UserEmailAddress AS AgentEmail" in sql


def test_candidate_query_windows_on_archive_time_with_two_bound_values():
    """
    Archive time, so every candidate is already closed when its interval is
    snapshotted. Direct column predicates, so the linked server filters first.
    """
    sql = sql_code("automation_day_conversations.sql")
    assert "c.ArchiveDateTime >= ?" in sql
    assert "c.ArchiveDateTime < ?" in sql
    assert sql.count("?") == 2


def test_candidate_query_has_no_correlated_apply_and_no_top():
    """Row-by-row OUTER APPLYs over the linked server timed out in production."""
    sql = sql_code("automation_day_conversations.sql").upper()
    assert "APPLY" not in sql
    assert "TOP" not in sql


def test_candidate_query_reports_whether_the_agent_replied():
    sql = sql_code("automation_day_conversations.sql")
    assert "m.SenderWasOwner = 1" in sql
    assert "AS AgentReplied" in sql


def test_automation_window_queries_apply_no_fixed_timezone_offset():
    """
    Window bounds come from calendar.py, which handles Cairo's DST. A fixed
    +3 in a filter would be wrong for half the year.
    """
    for name in ("automation_day_conversations.sql", "automation_active_agents.sql"):
        assert "DATEADD" not in sql_code(name).upper(), name


def test_roster_query_reads_cm_inbox_owners_not_dwh_agents():
    sql = sql_code("automation_active_agents.sql")
    assert "[DWH].[AI].[Agents]" not in sql
    assert "c.InboxOwnerPersonUniqueId = owner.UniqueId" in sql
    assert "AS Agent_Email_Address" in sql
    assert sql.count("?") == 2


def test_no_question_marks_hide_in_comments():
    """ODBC drivers can mistake a '?' in a comment for a parameter marker."""
    for name in ("automation_day_conversations.sql", "automation_active_agents.sql",
                 "automation_chat_by_ids.sql"):
        raw = (SQL_DIR / name).read_text()
        assert raw.count("?") == sql_code(name).count("?"), name


# ── Binding ──────────────────────────────────────────────────────────────────

class RecordingHandler:
    """Captures the SQL and params that would have gone to the driver."""

    def __init__(self, result: pd.DataFrame | None = None):
        self.connection = object()
        self.calls: list[tuple[str, list]] = []
        self._result = result if result is not None else pd.DataFrame()


@pytest.fixture()
def recorder(monkeypatch):
    handler = RecordingHandler()

    def fake_execute(h, sql, params):
        h.calls.append((sql, list(params)))
        return h._result

    monkeypatch.setattr(cq, "_execute", fake_execute)
    return handler


def test_ids_are_passed_as_bound_parameters(recorder, monkeypatch):
    ids = [f"0000000{i}-0000-0000-0000-000000000000" for i in range(3)]
    cq.fetch_transcripts_by_ids(recorder, ids)

    sql, params = recorder.calls[0]
    # Each id is bound once and appears nowhere in the text.
    assert params == ids
    for cid in ids:
        assert cid not in sql
    assert sql.count("?") == len(ids)


def test_placeholder_count_matches_the_id_count(recorder):
    cq.fetch_transcripts_by_ids(recorder, ["a", "b", "c", "d", "e"])
    sql, params = recorder.calls[0]
    assert len(params) == 5
    assert ", ".join("?" for _ in range(5)) in sql


def test_an_id_containing_sql_is_bound_not_interpolated(recorder):
    """
    The regex-substitution path in CM_DB_Handler would splice this into the
    text. Binding makes it inert data.
    """
    nasty = "'); DROP TABLE Conversations; --"
    cq.fetch_transcripts_by_ids(recorder, [nasty])
    sql, params = recorder.calls[0]
    assert nasty in params
    assert "DROP TABLE" not in sql


def test_long_id_lists_are_chunked(recorder):
    cq.fetch_transcripts_by_ids(recorder, [f"id-{i}" for i in range(450)], chunk_size=200)
    assert len(recorder.calls) == 3
    assert [len(params) for _, params in recorder.calls] == [200, 200, 50]


def test_empty_id_list_runs_no_query(recorder):
    assert cq.fetch_transcripts_by_ids(recorder, []).empty
    assert recorder.calls == []


def test_duplicate_ids_are_collapsed(recorder):
    cq.fetch_transcripts_by_ids(recorder, ["a", "a", "b"])
    _, params = recorder.calls[0]
    assert params == ["a", "b"]


# ── Shortfall detection ──────────────────────────────────────────────────────

def test_missing_conversations_are_reported_not_swallowed(monkeypatch, caplog):
    """
    df_to_batch truncated to 50 with a warnings.warn nobody sees. A shortfall
    here must reach the log at ERROR.
    """
    handler = RecordingHandler(pd.DataFrame([{"UniqueId": "a"}]))
    monkeypatch.setattr(cq, "_execute", lambda h, sql, params: h._result)

    with caplog.at_level("ERROR"):
        cq.fetch_transcripts_by_ids(handler, ["a", "b", "c"])

    assert "MISSING 2" in caplog.text


def test_full_retrieval_logs_no_loss(monkeypatch, caplog):
    handler = RecordingHandler(pd.DataFrame([{"UniqueId": "a"}, {"UniqueId": "b"}]))
    monkeypatch.setattr(cq, "_execute", lambda h, sql, params: h._result)

    with caplog.at_level("INFO"):
        cq.fetch_transcripts_by_ids(handler, ["a", "b"])

    assert "no loss" in caplog.text


def test_uuid_case_difference_does_not_discard_a_retrieved_transcript(monkeypatch, caplog):
    lower = "740ced91-bcc6-f111-bf5b-000d3aa9dc2f"
    upper = lower.upper()
    handler = RecordingHandler(pd.DataFrame([{
        "UniqueId": upper,
        "CreationDateTime": datetime(2026, 10, 13, 9),
        "MessageId": "m1",
        "Content": "hello",
    }]))
    monkeypatch.setattr(cq, "_execute", lambda h, sql, params: h._result)

    with caplog.at_level("INFO"):
        frame = cq.fetch_transcripts_by_ids(handler, [lower])
    assert "MISSING" not in caplog.text
    assert len(cq.rows_for_conversation(frame, lower)) == 1


# ── Window validation ────────────────────────────────────────────────────────

def test_naive_bounds_are_rejected(recorder):
    with pytest.raises(ValueError, match="tz-aware"):
        cq.fetch_day_candidates(recorder, datetime(2026, 10, 29, 6), datetime(2026, 10, 29, 8, tzinfo=UTC))


def test_empty_window_is_rejected(recorder):
    moment = datetime(2026, 10, 29, 6, tzinfo=UTC)
    with pytest.raises(ValueError, match="Empty window"):
        cq.fetch_day_candidates(recorder, moment, moment)


def test_candidate_window_is_bound_as_naive_utc(recorder):
    """CM_DWH stores naive datetimes holding UTC."""
    cq.fetch_day_candidates(
        recorder,
        datetime(2026, 10, 29, 6, tzinfo=UTC),
        datetime(2026, 10, 29, 8, tzinfo=UTC),
    )
    sql, params = recorder.calls[0]
    assert params == [datetime(2026, 10, 29, 6), datetime(2026, 10, 29, 8)]
    assert all(p.tzinfo is None for p in params)
    assert sql.count("?") == len(params)


def test_candidate_window_converts_other_offsets_to_utc(recorder):
    """A Cairo-summer (UTC+3) bound must reach the server as its UTC value."""
    cairo_summer = timezone(timedelta(hours=3))
    cq.fetch_day_candidates(
        recorder,
        datetime(2026, 10, 29, 9, tzinfo=cairo_summer),
        datetime(2026, 10, 29, 11, tzinfo=cairo_summer),
    )
    _, params = recorder.calls[0]
    assert params == [datetime(2026, 10, 29, 6), datetime(2026, 10, 29, 8)]


def test_roster_lookback_is_bound_as_naive_utc(recorder):
    cq.fetch_cm_roster(
        recorder,
        datetime(2026, 10, 4, 21, tzinfo=UTC),
        datetime(2026, 10, 18, 21, tzinfo=UTC),
    )
    sql, params = recorder.calls[0]
    assert params == [datetime(2026, 10, 4, 21), datetime(2026, 10, 18, 21)]
    assert "Agent_Email_Address" in sql


def test_roster_rejects_naive_and_empty_windows(recorder):
    with pytest.raises(ValueError, match="tz-aware"):
        cq.fetch_cm_roster(recorder, datetime(2026, 10, 4), datetime(2026, 10, 18, tzinfo=UTC))
    moment = datetime(2026, 10, 18, tzinfo=UTC)
    with pytest.raises(ValueError, match="Empty look-back"):
        cq.fetch_cm_roster(recorder, moment, moment)


# ── NEWFIX result quirks ─────────────────────────────────────────────────────

def test_duplicate_webstorename_column_is_dropped(monkeypatch):
    """
    NEWFIX selects tc.* and then tc.WebStoreName again. A duplicated label
    makes df[col] return a DataFrame and breaks sanitising.
    """
    frame = pd.DataFrame(
        [["c1", "m1", "AHJ", "AHJ"]],
        columns=["UniqueId", "MessageId", "WebStoreName", "WebStoreName"],
    )
    handler = RecordingHandler(frame)
    monkeypatch.setattr(cq, "_execute", lambda h, sql, params: h._result)
    df = cq.fetch_transcripts_by_ids(handler, ["c1"])
    assert list(df.columns) == ["UniqueId", "MessageId", "WebStoreName"]


def test_duplicated_message_rows_are_removed(monkeypatch):
    frame = pd.DataFrame([
        {"UniqueId": "c1", "MessageId": "m1", "Content": "hi"},
        {"UniqueId": "c1", "MessageId": "m1", "Content": "hi"},
        {"UniqueId": "c1", "MessageId": "m2", "Content": "bye"},
    ])
    handler = RecordingHandler(frame)
    monkeypatch.setattr(cq, "_execute", lambda h, sql, params: h._result)
    df = cq.fetch_transcripts_by_ids(handler, ["c1"])
    assert list(df["MessageId"]) == ["m1", "m2"]


# ── Sanitisation ─────────────────────────────────────────────────────────────

def test_nan_text_columns_become_empty_strings():
    """The validators call .strip(); a float NaN would crash them."""
    df = pd.DataFrame([{"Content": None, "SenderType": "User", "UniqueId": "a"}])
    assert sanitize_df(df)["Content"].iloc[0] == ""


def test_rows_are_sorted_by_timestamp_then_message_id():
    """from_cm_rows sorts by timestamp alone, leaving ties non-deterministic."""
    same = datetime(2026, 10, 29, 6, 0)
    df = pd.DataFrame([
        {"UniqueId": "c", "CreationDateTime": same, "MessageId": "m2", "Content": "second"},
        {"UniqueId": "c", "CreationDateTime": same, "MessageId": "m1", "Content": "first"},
    ])
    assert [r["MessageId"] for r in sanitize_rows(df)] == ["m1", "m2"]
