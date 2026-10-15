"""
Local SQLite ledger for the automated QA sampler.

The division of responsibility, which is not negotiable:

    DWH (SQL Server)    read + insert only, through the existing code paths.
                        No CREATE, no ALTER, no DROP. We have no DDL rights
                        and must not need any.

    SQLite (this file)  everything the automation owns — runs, quotas,
                        intervals, candidate snapshots, jobs, leases and the
                        job-to-result links.

So the ledger is local state about remote data. Two consequences worth holding
onto:

* The ledger can be rebuilt from DWH plus the policy, but **not for free** —
  the frozen candidate snapshots and rank hashes are the audit trail, and
  losing them loses reproducibility of past selections. The file belongs on
  durable storage and in whatever backup the host already runs.

* A result in DWH with no corresponding local link is indistinguishable from
  a manual evaluation. QA_ResultLinks is the only record that an automated
  job produced a given row, which is why the persistence protocol in the
  schema commits its intent *before* the insert rather than after.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
_SQL_DIR = Path(__file__).resolve().parent.parent / "SQL" / "automation" / "sqlite"
_SCHEMA_PATH = _SQL_DIR / "001_ledger.sql"

# ISO-8601 UTC with milliseconds and a literal Z. Chosen so that TEXT
# comparison is chronological comparison; every timestamp written to the
# ledger goes through utc_text().
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


def utc_text(moment: Optional[datetime] = None) -> str:
    """
    Render an instant as the ledger's canonical timestamp string.

    Naive datetimes are rejected rather than assumed to be UTC: the whole
    point of app.automation.calendar is that an unlabelled timestamp in an
    Egypt-timezone system is a bug waiting to happen.
    """
    if moment is None:
        moment = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise ValueError(
            f"utc_text requires a tz-aware datetime, got {moment!r}. "
            f"Stamp timestamps as UTC at the point they are read."
        )
    return moment.astimezone(timezone.utc).strftime(_TS_FORMAT)[:-3] + "Z"


def parse_utc(text: str) -> datetime:
    """Inverse of utc_text. Always returns a tz-aware UTC datetime."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def connect(db_path: str | Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Open the ledger with the pragmas the automation depends on.

    * ``foreign_keys=ON`` — off by default in SQLite, so the REFERENCES
      clauses in the schema are decorative until this is set.
    * ``journal_mode=WAL`` — readers do not block the writer, so the dashboard
      can query coverage while the worker is persisting. Persistent: set once,
      stored in the file.
    * ``busy_timeout`` — makes brief write contention wait instead of raising
      'database is locked' immediately. Matters once worker concurrency > 1.
    * ``synchronous=FULL`` — the persistence protocol relies on the
      'attempting' row being durable *before* the DWH insert. NORMAL in WAL
      mode can lose the most recent commits on an OS crash, which is exactly
      the evidence crash recovery needs. The cost is a few ms per commit on a
      ledger doing a few hundred writes a day.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=timeout, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def apply_schema(conn: sqlite3.Connection) -> int:
    """
    Create the ledger schema if it is not already there, apply any pending
    migrations, and return the schema version in force.

    Idempotent, so it runs on every startup. Migrations are numbered scripts
    beside 001_ledger.sql and are applied in order; each is guarded by a
    condition rather than relying on the recorded version alone, because
    SQLite lacks 'IF EXISTS' on most ALTER forms and a half-applied migration
    must not wedge the ledger.
    """
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Ledger schema not found: {_SCHEMA_PATH}")

    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    existing = conn.execute("SELECT MAX(Version) FROM QA_SchemaVersion").fetchone()[0]

    if existing is not None and existing > SCHEMA_VERSION:
        raise RuntimeError(
            f"Ledger is at schema version {existing} but this code expects "
            f"{SCHEMA_VERSION}. Running older code against a newer ledger risks "
            f"writing rows it does not understand — upgrade the application."
        )

    # 002: the shift/working-period model was removed. Drop its provenance
    # column from ledgers created before that.
    if _column_exists(conn, "QA_AgentQuotas", "WorkPeriodSource"):
        conn.executescript((_SQL_DIR / "002_drop_shifts.sql").read_text(encoding="utf-8"))
        logger.info("Automation ledger migrated: dropped QA_AgentQuotas.WorkPeriodSource")

    if existing is None:
        conn.execute(
            "INSERT INTO QA_SchemaVersion (Version, AppliedAtUtc, Description) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, utc_text(), "Automation ledger, full-day sampling"),
        )
        logger.info("Automation ledger created at schema version %d", SCHEMA_VERSION)
        return SCHEMA_VERSION

    if existing < SCHEMA_VERSION:
        conn.execute(
            "INSERT INTO QA_SchemaVersion (Version, AppliedAtUtc, Description) VALUES (?, ?, ?)",
            (SCHEMA_VERSION, utc_text(), "Removed the shift/working-period model"),
        )
        return SCHEMA_VERSION

    return existing


def open_ledger(db_path: str | Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    """Connect and ensure the schema is present. The normal entry point."""
    conn = connect(db_path, timeout=timeout)
    apply_schema(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    """
    Run a block in a transaction, committing on success and rolling back on
    any exception.

    ``immediate=True`` issues BEGIN IMMEDIATE, taking the write lock up front.
    This is the right default for read-modify-write sequences such as claiming
    a job or reserving quota: SQLite's default deferred transaction takes the
    lock only at the first write, so two workers can both read, then one gets
    SQLITE_BUSY on upgrade. Taking it immediately turns that race into a short
    wait governed by busy_timeout.

    The connection is opened with ``isolation_level=None``, so Python's
    implicit transaction handling is off and these BEGIN/COMMIT statements are
    the only ones in play.
    """
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def next_fence_token(conn: sqlite3.Connection) -> int:
    """
    Take the next fencing token. Must be called inside a write transaction.

    Monotonic across the whole ledger, so it also orders claims: a worker
    holding a lower token than the job row's current one has been superseded
    and must not write its result back.
    """
    row = conn.execute(
        "UPDATE QA_FenceCounter SET NextValue = NextValue + 1 WHERE Id = 1 "
        "RETURNING NextValue - 1"
    ).fetchone()
    return int(row[0])
