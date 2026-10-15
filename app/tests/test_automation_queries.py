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

def test_candidate_sql_has_no_top_clause():
    """The TOP (10) in NEWFIX_CM_Chat_Search.sql is why automation needs its own."""
    sql = sql_code("day_candidates.sql").upper()
    assert "TOP (10)" not in sql
    # OUTER APPLY ... TOP (1) is a per-conversation lookup, not a result cap.
    assert "SELECT TOP" not in sql.replace("SELECT TOP (1)", "")


def test_candidate_sql_attributes_by_last_human_message_not_inbox_owner():
    sql = sql_code("day_candidates.sql")
    assert "p.Discriminator = 'User'" in sql
    assert "ORDER BY m.CreationDateTime DESC" in sql
    # The inbox owner is returned for comparison, but not used to attribute.
    assert "InboxOwnerEmail" in sql
    assert "lastHuman.AgentEmail" in sql


def test_candidate_sql_applies_no_fixed_timezone_offset():
    """A hard-coded +3 is wrong for Africa/Cairo for half the year."""
    assert "DATEADD(HOUR, 3," not in sql_code("day_candidates.sql")
    assert "DATEADD(HOUR, 3," not in sql_code("transcripts_by_ids.sql")


def test_transcript_sql_deduplicates_messages():
    sql = sql_code("transcripts_by_ids.sql")
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY m.Conversation_UniqueId, m.UniqueId" in sql
    assert "d.rn = 1" in sql


def test_transcript_sql_orders_deterministically():
    """Equal timestamps must not leave message order to chance."""
    sql = sql_code("transcripts_by_ids.sql")
    assert "ORDER BY\n    t.UniqueId,\n    d.CreationDateTime,\n    d.MessageId" in sql


def test_transcript_sql_uses_a_placeholder_token_not_interpolated_values():
    raw = (SQL_DIR / "transcripts_by_ids.sql").read_text()
    # Exactly two, both in executable SQL — one per CTE. A third occurrence in
    # a comment would also be substituted, which is harmless but misleading.
    assert raw.count("{id_placeholders}") == 2
    assert sql_code("transcripts_by_ids.sql").count("{id_placeholders}") == 2


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
    # Each id is bound twice (once per CTE) and appears nowhere in the text.
    assert params == ids + ids
    for cid in ids:
        assert cid not in sql
    assert sql.count("?") == len(ids) * 2


def test_placeholder_count_matches_the_id_count(recorder):
    cq.fetch_transcripts_by_ids(recorder, ["a", "b", "c", "d", "e"])
    sql, params = recorder.calls[0]
    assert len(params) == 10
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
    assert [len(params) for _, params in recorder.calls] == [400, 400, 100]


def test_empty_id_list_runs_no_query(recorder):
    assert cq.fetch_transcripts_by_ids(recorder, []).empty
    assert recorder.calls == []


def test_duplicate_ids_are_collapsed(recorder):
    cq.fetch_transcripts_by_ids(recorder, ["a", "a", "b"])
    _, params = recorder.calls[0]
    assert params == ["a", "b", "a", "b"]


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


def test_candidate_query_runs_on_remote_server_with_safe_utc_literals(recorder):
    """The full candidate join must execute on CM_DWH, not over four-part names."""
    cq.fetch_day_candidates(
        recorder,
        datetime(2026, 10, 29, 6, tzinfo=UTC),
        datetime(2026, 10, 29, 8, tzinfo=UTC),
    )
    sql, params = recorder.calls[0]
    assert sql.startswith("SELECT * FROM OPENQUERY([CM_DWH], '")
    assert "[CM_DWH].[msc_sub_35033_XYLJV]" not in sql
    assert sql.count("2026-10-29T06:00:00.000000") == 2
    assert sql.count("2026-10-29T08:00:00.000000") == 2
    assert "''User''" in sql
    assert "?" not in sql
    assert params == []


def test_candidate_query_converts_aware_bounds_to_utc(recorder):
    cairo = timezone(timedelta(hours=3))
    cq.fetch_day_candidates(
        recorder,
        datetime(2026, 10, 29, 9, tzinfo=cairo),
        datetime(2026, 10, 29, 11, tzinfo=cairo),
    )
    sql, _ = recorder.calls[0]
    assert "2026-10-29T06:00:00.000000" in sql
    assert "2026-10-29T08:00:00.000000" in sql
    assert "2026-10-29T09:00:00.000000" not in sql


def test_candidate_query_is_driven_from_messages_not_conversations():
    """
    The first version filtered on an OUTER APPLY's result, forcing evaluation
    against every conversation in the warehouse. Over a linked server that
    timed out and killed the connection — 179 intervals failed on the first
    live run. The Touched CTE narrows by an indexed date column first.
    """
    sql = sql_code("day_candidates.sql")
    assert "WITH Touched AS" in sql
    assert sql.index("WITH Touched AS") < sql.index("FROM Touched t")
    assert "m.CreationDateTime >= ?" in sql


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
