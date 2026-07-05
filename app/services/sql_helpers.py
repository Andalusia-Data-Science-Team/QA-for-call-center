"""
SQL helper utilities for the Call QA Analysis System.

Functions
─────────
insert_qa_result   – Persists a completed QAAnalysisResult to the
                     [DWH].[AI].[Call_QA_Results] table using the query
                     defined in app/SQL/agentDB.sql.

Schema contract
───────────────
The table is denormalized: one row per ComplianceFlag, with all call-level
fields repeated on every row.  When a call has zero flags a single summary
row is inserted with NULL in the flag columns so the call still appears in
the table.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text as sa_text

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_VERSION = 1

_APP_DIR = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_engine():
    """Return a SQLAlchemy engine for the DWH / AI write account."""
    passcodes_path = _APP_DIR / "Passcode.json"
    if not passcodes_path.exists():
        raise FileNotFoundError(f"Passcode file not found: {passcodes_path}")

    with passcodes_path.open("r", encoding="utf-8") as fh:
        passcodes = json.load(fh)["DB_NAMES"]["DWH"]

    params = urllib.parse.quote_plus(
        f"DRIVER={passcodes['driver']};"
        f"SERVER={passcodes['Server']};"
        f"DATABASE={passcodes['Database']};"
        f"UID={passcodes['UID']};"
        f"PWD={passcodes['PWD']};"
        f"Connection Timeout=300;"
    )
    return create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))


def _weighted_score(result: QAAnalysisResult) -> float:
    """
    Weighted composite score: professionalism 40 %, accuracy 30 %, resolution 30 %.
    Passed_Threshold is a computed column in the DB derived from this value.
    """
    perf = result.agent_performance
    return round(
        perf.professionalism_score * 0.40
        + perf.accuracy_score      * 0.30
        + perf.resolution_score    * 0.30,
        4,
    )


def _call_level_params(
    result: QAAnalysisResult,
    call: CallTranscript,
    agent_id: Optional[str],
    channel_name: Optional[str],
    weighted: float,
    created_at: str,
) -> dict:
    """Build the call-level fields shared by every row for this call."""
    perf = result.agent_performance
    return {
        "Call_ID":               call.call_id,
        "Analysis_Version":      ANALYSIS_VERSION,
        "Agent_ID":              agent_id,
        "Agent_Name":            call.agent_name,
        "Department":            call.department,
        "Channel_Name":          channel_name,
        "Call_Date":             call.call_date,
        "Call_Duration_Seconds": call.call_duration_seconds,
        "Overall_Assessment":    result.overall_assessment,
        "Assessment_Reasoning":  result.assessment_reasoning,
        "Professionalism_Score": perf.professionalism_score,
        "Accuracy_Score":        perf.accuracy_score,
        "Resolution_Score":      perf.resolution_score,
        "Strengths":             json.dumps(perf.strengths,    ensure_ascii=False),
        "Improvements":          json.dumps(perf.improvements, ensure_ascii=False),
        "Escalation_Required":   1 if result.escalation_required else 0,
        "Escalation_Reason":     result.escalation_reason,
        "Weighted_Score":        weighted,
        "Created_at":            created_at,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def insert_qa_result(
    result: QAAnalysisResult,
    call: CallTranscript,
    *,
    agent_id: Optional[str] = None,
    channel_name: Optional[str] = "WhatsApp",
) -> None:
    """
    Insert QA evaluation results into [DWH].[AI].[Call_QA_Results].

    One row is inserted per ComplianceFlag detected in the call.  The
    call-level fields (scores, assessment, agent name, …) are repeated on
    every row.  When no flags were detected a single row is inserted with
    NULL in all flag columns so the call still has a record in the table.

    Parameters
    ──────────
    result       : Validated QAAnalysisResult from the pipeline.
    call         : Original CallTranscript (agent name, date, duration, …).
    agent_id     : Internal agent identifier (optional).
    channel_name : Channel label, e.g. "Phone", "Chat" (optional).

    Notes
    ─────
    • Passed_Threshold is a computed column — it is NOT inserted.
    • All rows for the same call share the same Created_at timestamp.

    Raises
    ──────
    FileNotFoundError  – Passcode.json is missing.
    sqlalchemy.*       – any database error is re-raised after logging.
    """
    sql_path = _APP_DIR / "SQL" / "agentDB.sql"
    query = sa_text(sql_path.read_text(encoding="utf-8"))

    weighted   = _weighted_score(result)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    engine = _build_engine()
    with engine.begin() as conn:
        # Resolve the next Analysis_Version for this Call_ID.
        # If the call has never been analysed before → version 1.
        # If it already exists → increment the current max by 1 so re-runs
        # are stored as new versions rather than silently overwriting.
        version_row = conn.execute(
            sa_text(
                "SELECT COALESCE(MAX(Analysis_Version), 0) AS max_ver "
                "FROM [DWH].[AI].[Call_QA_Results] "
                "WHERE Call_ID = :cid"
            ),
            {"cid": call.call_id},
        ).fetchone()
        analysis_version = (version_row[0] if version_row else 0) + 1

        base = _call_level_params(result, call, agent_id, channel_name, weighted, created_at)
        base["Analysis_Version"] = analysis_version

        flags = result.compliance_flags

        if flags:
            rows = []
            for flag in flags:
                rows.append({
                    **base,
                    "Compliance_Flag_Type": flag.type,
                    "Severity":             flag.severity,
                    "Violation":            flag.type,
                    "Flag_Description":     flag.description,
                    "Transcript_Excerpt":   flag.transcript_excerpt,
                })
        else:
            # No flags — insert one summary row with NULL flag columns
            rows = [{
                **base,
                "Compliance_Flag_Type": None,
                "Severity":             None,
                "Violation":            None,
                "Flag_Description":     None,
                "Transcript_Excerpt":   None,
            }]

        logger.debug(
            "insert_qa_result | call_id=%s assessment=%s weighted=%.4f "
            "version=%d rows=%d",
            call.call_id, result.overall_assessment, weighted,
            analysis_version, len(rows),
        )

        for row in rows:
            conn.execute(query, row)

    logger.info(
        "insert_qa_result | call_id=%s version=%d — %d row(s) inserted successfully",
        call.call_id, analysis_version, len(rows),
    )
