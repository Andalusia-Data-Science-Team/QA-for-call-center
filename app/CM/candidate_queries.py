"""
Read-only access to CM_DWH for the automated sampler.

Why this exists rather than reusing CMDatabaseHandler.execute_query_from_file
─────────────────────────────────────────────────────────────────────────────
That method does not bind parameters. It rewrites the SQL text with a regex
(`re.sub(rf':\\s*{key}\\b', sql_value, query)`, CM_DB_Handler.py:166-178),
quoting strings by doubling apostrophes. For the manual dashboard's two scalar
inputs that is survivable. For a list of selected conversation ids it is not:
the value count varies per call, and a text-substituted identifier list is
both injection-prone and silently wrong if any id is NULL or oddly typed.

All three automation queries are derived from NEWFIX_CM_Chat_Search.sql, the
query the manual dashboard uses, and live in app/SQL/automation/:

    automation_active_agents.sql      the roster: CM inbox owners
    automation_day_conversations.sql  candidates: one row per conversation
    automation_chat_by_ids.sql        transcripts: NEWFIX, selected by id

Every value reaches the driver as a bound `?` parameter. The only SQL text
built at runtime is the number of placeholder marks for an id list, derived
from `len(ids)` and never from the ids themselves.

The existing handler is reused for connection management only — it already
handles credentials, retry on transient TCP failures and reconnection.

READ ONLY. Nothing in this module writes to, or alters, anything in CM_DWH.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

_SQL_DIR = Path(__file__).resolve().parent.parent / "SQL" / "automation"

# SQL Server's hard ceiling is 2100 parameters per statement and
# automation_chat_by_ids binds each id once. We chunk
# far below that: a chunk is also the unit of work held in memory as a DataFrame
# of full message content, and 200 conversations of chat is already sizeable.
MAX_IDS_PER_QUERY = 200


def _read_sql(name: str) -> str:
    path = _SQL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Automation SQL not found: {path}")
    return path.read_text(encoding="utf-8")


# Driver message fragments that mean "the connection or the network broke",
# as opposed to a real SQL error that would fail identically on every retry.
# Same reasoning as CM_DB_Handler._TRANSIENT_DB_ERROR_MARKERS, and the same
# unavoidable approach: pyodbc surfaces both kinds as one exception type, so
# the message text is the only signal.
_TRANSIENT_MARKERS = (
    "communication link failure",
    "wait operation timed out",
    "timeout expired",
    "query timeout",
    "server is not found or not accessible",
    "connection is busy",
    "tcp provider",
    "connection was recovered",
    # pyodbc's own message when the handle was closed underneath us, e.g. the
    # dispatcher's finally-block closing the handler while a cached frame is
    # still being drained. Recoverable by reconnecting, like the rest.
    "closed connection",
    "the cursor's connection has been closed",
    "connection not open",
)


class TransientQueryError(Exception):
    """A query failure the caller should retry rather than treat as permanent."""


def is_transient(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _execute(
    handler,
    sql: str,
    params: Sequence,
    *,
    max_retries: int = 2,
    retry_delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Run a query, reconnecting once the link dies.

    WHY THE RETRY IS HERE AND NOT LEFT TO THE CALLER
    ────────────────────────────────────────────────
    A 'Communication link failure' leaves the pyodbc connection object dead.
    Every subsequent query on it fails the same way, so one blip cascades into
    every remaining interval of the dispatch pass failing. That is exactly
    what happened on the first live run: one timeout, then 179 intervals
    failed in a row on a connection nobody had re-established.

    Reconnecting is the caller's only real option, and doing it here keeps it
    out of every call site. CMDatabaseHandler.execute_query_from_file already
    does the same thing for the manual path; this is the parameterised
    equivalent.
    """
    import time
    import warnings

    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 2):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*DBAPI2.*")
                return pd.read_sql(sql, handler.connection, params=list(params))
        except Exception as exc:                                   # noqa: BLE001
            last_exc = exc
            if not is_transient(exc) or attempt > max_retries:
                raise

            delay = retry_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "query failed transiently (attempt %d/%d) — reconnecting and "
                "retrying in %.1fs: %s",
                attempt, max_retries + 1, delay, exc,
            )
            try:
                handler.close()
            except Exception:
                pass            # already dead; nothing to clean up
            time.sleep(delay)
            if not handler.connect(query_timeout=600):
                raise TransientQueryError(
                    f"reconnect failed after a transient error: {exc}"
                ) from exc

    raise TransientQueryError(str(last_exc))


def fetch_day_candidates(
    handler,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> pd.DataFrame:
    """
    Every conversation closed in ``[window_start_utc, window_end_utc)``, with
    its agent — the inbox owner, as NEWFIX attributes it.

    One query covers the whole roster for the window, rather than one query per
    agent per interval: the dispatcher runs every ten minutes and a per-agent
    fan-out would multiply that by the roster size for no extra information.

    Bounds must be tz-aware. They are sent as naive UTC because CM_DWH stores
    naive datetimes holding UTC. All predicates in the SQL are direct column
    comparisons on Conversations, so the linked server filters first; measured
    at about 8 seconds per 2-hour window.

    Returns an empty frame when nothing matches, so callers never have to
    special-case None.
    """
    for name, value in (("window_start_utc", window_start_utc), ("window_end_utc", window_end_utc)):
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{name} must be tz-aware UTC, got naive {value!r}")
    if window_end_utc <= window_start_utc:
        raise ValueError(f"Empty window: {window_start_utc} -> {window_end_utc}")

    params = (_naive_utc(window_start_utc), _naive_utc(window_end_utc))
    df = _execute(handler, _read_sql("automation_day_conversations.sql"), params)
    logger.info(
        "fetch_day_candidates | %s -> %s | %d conversations, %d distinct agents",
        window_start_utc.isoformat(),
        window_end_utc.isoformat(),
        len(df),
        df["AgentEmail"].nunique() if not df.empty else 0,
    )
    return df


def fetch_cm_roster(
    handler,
    lookback_start_utc: datetime,
    lookback_end_utc: datetime,
) -> pd.DataFrame:
    """
    The agents who handled chats in CM during the look-back window.

    Columns are Agent_Email_Address, Agent_Name, Agent_id and
    ConversationCount — the first three use DWH.AI.Agents' names so the
    planner's roster helpers work unchanged.
    """
    for name, value in (("lookback_start_utc", lookback_start_utc), ("lookback_end_utc", lookback_end_utc)):
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError(f"{name} must be tz-aware UTC, got naive {value!r}")
    if lookback_end_utc <= lookback_start_utc:
        raise ValueError(f"Empty look-back: {lookback_start_utc} -> {lookback_end_utc}")

    params = (_naive_utc(lookback_start_utc), _naive_utc(lookback_end_utc))
    return _execute(handler, _read_sql("automation_active_agents.sql"), params)


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def fetch_transcripts_by_ids(
    handler,
    conversation_ids: Iterable[str],
    *,
    chunk_size: int = MAX_IDS_PER_QUERY,
) -> pd.DataFrame:
    """
    Full, deduplicated message rows for exactly the conversations named.

    Chunked to stay under SQL Server's parameter ceiling; chunks are
    concatenated, so the caller sees one frame regardless.

    **Reconciles what it asked for against what it got.** A silent shortfall
    here is precisely the failure mode that `df_to_batch`'s truncation created,
    so a missing conversation is logged loudly rather than quietly dropped.
    The caller decides what to do about it — this function does not raise,
    because one unretrievable conversation must not fail a whole batch.
    """
    ids = [str(cid) for cid in conversation_ids]
    if not ids:
        return pd.DataFrame()

    unique_ids = list(dict.fromkeys(ids))   # de-dupe, preserve order
    if len(unique_ids) != len(ids):
        logger.warning(
            "fetch_transcripts_by_ids | %d duplicate id(s) in request, de-duplicated",
            len(ids) - len(unique_ids),
        )

    template = _read_sql("automation_chat_by_ids.sql")
    frames: list[pd.DataFrame] = []

    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start:start + chunk_size]
        placeholders = ", ".join("?" for _ in chunk)       # count only, never values
        sql = template.replace("{id_placeholders}", placeholders)
        frames.append(_execute(handler, sql, chunk))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not df.empty:
        # NEWFIX selects tc.* and then tc.WebStoreName again, so the result has
        # two WebStoreName columns. A duplicated label makes df[col] return a
        # DataFrame instead of a Series and breaks row sanitising.
        df = df.loc[:, ~df.columns.duplicated()]
        # NEWFIX's joins can repeat a message; keep one row per message.
        if "MessageId" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["UniqueId", "MessageId"], keep="first")
            if len(df) != before:
                logger.info(
                    "fetch_transcripts_by_ids | removed %d duplicated message row(s)",
                    before - len(df),
                )

    # SQL Server returns GUIDs in uppercase on some driver paths while the
    # sampler stores canonical lowercase IDs. Compare IDs case-insensitively.
    retrieved = {str(cid).casefold() for cid in df["UniqueId"].unique()} if not df.empty else set()
    missing = [cid for cid in unique_ids if cid.casefold() not in retrieved]
    if missing:
        logger.error(
            "fetch_transcripts_by_ids | requested %d conversations, retrieved %d — "
            "MISSING %d: %s",
            len(unique_ids), len(retrieved), len(missing),
            ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
        )
    else:
        logger.info(
            "fetch_transcripts_by_ids | %d conversations, %d message rows, no loss",
            len(retrieved), len(df),
        )
    return df


def rows_for_conversation(df: pd.DataFrame, conversation_id: str) -> list[dict]:
    """
    The message rows for one conversation, shaped for
    ``CallTranscript.from_cm_rows``.

    Deliberately bypasses ``Chat_Retriever.df_to_batch``, which caps at 50
    conversations and warns rather than raises. The automation evaluates one
    conversation per job, so the batch model's ``max_length=50`` never applies.

    NaN text columns are coerced to empty strings for the same reason
    ``_sanitize_df`` does it: the model's field validators call ``.strip()``
    and would otherwise receive a float.
    """
    from app.CM.chat_sanitize import sanitize_rows   # local import: optional dependency path

    subset = df.loc[df["UniqueId"].astype(str).str.casefold() == str(conversation_id).casefold()]
    if subset.empty:
        return []
    return sanitize_rows(subset)
