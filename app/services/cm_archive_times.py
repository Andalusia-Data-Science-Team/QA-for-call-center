"""Read CM archive timestamps for QA result previews without changing stored rows."""

from __future__ import annotations

import urllib.parse
from typing import Sequence

from sqlalchemy import bindparam, create_engine, text


def _make_engine(creds: dict):
    params = urllib.parse.quote_plus(
        f"DRIVER={creds['driver']};"
        f"SERVER={creds['Server']};"
        f"DATABASE={creds['Database']};"
        f"UID={creds['UID']};"
        f"PWD={creds['PWD']};"
        f"Connection Timeout=10;"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


def lookup_archive_datetimes(call_ids: Sequence[str], creds: dict) -> dict[str, str]:
    """Return CM's UTC+3 ArchiveDateTime by conversation id, to the second."""
    ids = list(dict.fromkeys(str(call_id).upper() for call_id in call_ids if call_id))
    if not ids:
        return {}

    query = text("""
        SELECT c.UniqueId AS CallId,
               DATEADD(HOUR, 3, c.ArchiveDateTime) AS ArchiveDateTime
        FROM [CM_DWH].[msc_sub_35033_XYLJV].[dbo].[Conversations] c
        WHERE c.UniqueId IN :ids
          AND c.ArchiveDateTime IS NOT NULL
    """).bindparams(bindparam("ids", expanding=True))

    engine = _make_engine(creds)
    found: dict[str, str] = {}
    try:
        with engine.connect() as conn:
            for offset in range(0, len(ids), 200):
                rows = conn.execute(query, {"ids": ids[offset:offset + 200]}).mappings()
                for row in rows:
                    found[str(row["CallId"]).upper()] = row["ArchiveDateTime"].isoformat(
                        timespec="seconds"
                    )
    finally:
        engine.dispose()
    return found
