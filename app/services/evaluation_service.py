"""
The single evaluation entry point shared by the dashboard and the automation.

Thin by design. `QAAgent.analyze()` already owns the pipeline, including
persistence; this adds the things a caller outside the request cycle needs:
a stable classification of failures, and the persist outcome that the
automation's idempotency protocol depends on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.models.input import CallTranscript
from app.models.output import QAAnalysisResult
from app.services import persist_context

logger = logging.getLogger(__name__)


# Substrings identifying a failure worth retrying. Mirrors the reasoning in
# CM_DB_Handler._TRANSIENT_DB_ERROR_MARKERS: providers surface transient and
# permanent problems as the same exception type, so the message text is the
# only signal available.
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection reset", "connection aborted",
    "temporarily unavailable", "service unavailable", "bad gateway",
    "too many requests", "rate limit", "overloaded", "502", "503", "504", "429",
    "communication link failure", "wait operation",
)

# Problems that will fail identically on every retry. Retrying these just
# spends the attempt budget and the token budget to reach the same place.
_PERMANENT_MARKERS = (
    "invalid api key", "unauthorized", "authentication", "forbidden",
    "model not found", "does not exist", "insufficient_quota", "permission was denied",
)

_QUOTA_MARKERS = ("insufficient credits", "credits exhausted", "http 402")


@dataclass(slots=True)
class EvaluationOutcome:
    """Everything the worker needs to decide what happens to the job."""

    result: Optional[QAAnalysisResult]
    persisted: bool
    analysis_version: Optional[int] = None
    rows_inserted: Optional[int] = None
    error: Optional[str] = None
    failure_class: Optional[str] = None      # 'transient' | 'permanent' | 'input'


def classify_failure(exc: BaseException) -> str:
    """
    Bucket an exception for retry purposes.

    Defaults to 'transient' for anything unrecognised, which is the
    conservative choice: a wrongly-retried permanent failure costs three
    attempts, while a wrongly-abandoned transient one silently loses a
    conversation from the day's coverage.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _QUOTA_MARKERS):
        return "quota"
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return "permanent"
    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return "input"
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return "transient"
    return "transient"


def transcript_checksum(call: CallTranscript) -> str:
    """
    A stable fingerprint of the evaluation input.

    Recorded on the job at first retrieval so that a retry can tell whether it
    is re-evaluating the same conversation or a changed one. Covers the fields
    that actually reach the model; metadata the rubric never sees is excluded
    so that an unrelated DWH touch does not look like a transcript change.
    """
    payload = json.dumps(
        {
            "call_id": call.call_id,
            "transcript": call.transcript,
            "message_count": call.message_count,
            "agent_email": call.agent_email,
            "call_date": call.call_date,
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def evaluate_conversation(
    analyzer,
    call: CallTranscript,
    *,
    job_id: Optional[str] = None,
) -> EvaluationOutcome:
    """
    Evaluate one conversation and report precisely what happened.

    The distinction that matters: a QAAnalysisResult being returned does NOT
    mean it was persisted. The graph's save_to_database node treats a failed
    write as non-fatal for the manual path, so an automated caller must check
    ``persisted`` rather than assuming a result implies a row in DWH.
    """
    persist_context.reset()

    try:
        result = await analyzer.analyze(call, job_id=job_id)
    except Exception as exc:                                       # noqa: BLE001
        failure_class = classify_failure(exc)
        logger.error(
            "evaluate_conversation | call_id=%s job=%s failed (%s): %s",
            call.call_id, job_id, failure_class, exc,
        )
        return EvaluationOutcome(
            result=None, persisted=False, error=str(exc), failure_class=failure_class,
        )

    if result.overall_assessment == "error":
        # The pipeline caught the problem itself and returned an error result.
        # Nothing was persisted — save_to_database skips error assessments.
        reason = result.assessment_reasoning or "pipeline returned an error result"
        return EvaluationOutcome(
            result=result, persisted=False, error=reason,
            failure_class=classify_failure(RuntimeError(reason)),
        )

    outcome = persist_context.last()
    if outcome is None:
        return EvaluationOutcome(
            result=result, persisted=False,
            error="evaluation succeeded but no DWH row was recorded",
            failure_class="transient",
        )

    return EvaluationOutcome(
        result=result, persisted=True,
        analysis_version=outcome.analysis_version,
        rows_inserted=outcome.rows_inserted,
    )
