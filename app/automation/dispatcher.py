"""
The dispatcher loop: find due intervals, snapshot candidates, reserve jobs.

Runs every QA_DISPATCH_INTERVAL_SECONDS. For each interval whose cutoff has
passed, it freezes the candidate set, runs the deterministic selection, and
enqueues jobs — all in one transaction, so a crash mid-interval leaves either
a complete snapshot with its jobs or nothing at all.

Missed intervals are not special. ``repo.due_intervals`` has no lower time
bound, so an interval the scheduler slept through is simply still due and gets
picked up on the next tick. That is what makes scheduler downtime self-healing
rather than a permanent hole in the day.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from app.automation import eligibility, repository as repo
from app.automation.backfill import backfill_run
from app.automation.config import AutomationSettings
from app.automation.db import transaction, utc_text
from app.automation.sampling import Candidate, rank_hash, select_nonconsecutive

logger = logging.getLogger(__name__)

UTC = timezone.utc


@dataclass(slots=True)
class DispatchReport:
    intervals_processed: int = 0
    candidates_seen: int = 0
    jobs_reserved: int = 0
    backfilled: int = 0
    shortfalls: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def dispatch_once(
    conn,
    cm_handler,
    settings: AutomationSettings,
    *,
    now: Optional[datetime] = None,
    backfill: bool = True,
) -> DispatchReport:
    """One dispatcher pass over every due interval, then top up any short run."""
    from app.CM.candidate_queries import fetch_day_candidates

    moment = now or datetime.now(UTC)
    report = DispatchReport()

    due = repo.due_intervals(conn, moment)
    if not due:
        logger.debug("dispatch | nothing due at %s", moment.isoformat())
        return report

    logger.info("dispatch | %d interval(s) due at %s", len(due), moment.isoformat())

    # One CM_DWH query per distinct interval window, shared across all the
    # agents whose intervals share it — the candidate query covers the whole
    # roster, so fetching per agent would multiply the cost for no gain.
    windows: dict[tuple[str, str], pd.DataFrame] = {}
    window_errors: dict[tuple[str, str], Exception] = {}

    for interval in due:
        key = (interval["IntervalStartUtc"], interval["IntervalEndUtc"])
        try:
            if key in window_errors:
                raise window_errors[key]
            if key not in windows:
                try:
                    windows[key] = fetch_day_candidates(
                        cm_handler,
                        _parse(interval["IntervalStartUtc"]),
                        _parse(interval["IntervalEndUtc"]),
                    )
                except Exception as exc:
                    window_errors[key] = exc
                    raise
            _process_interval(conn, interval, windows[key], settings, report, moment)
        except Exception as exc:                                  # noqa: BLE001
            from app.CM.candidate_queries import is_transient

            message = f"interval {interval['IntervalId']} for {interval['AgentEmailAddress']}: {exc}"
            report.errors.append(message)

            # A transient DB failure must NOT make the interval terminal.
            # due_intervals only returns 'pending'/'snapshotting', so marking a
            # link failure as 'failed' loses that interval for good — one
            # connection blip permanently deleted 179 intervals' worth of the
            # daily budget on the first live run. Transient goes back to
            # pending and is retried on the next tick; only a genuine error
            # (bad SQL, missing column, unparseable data) is terminal.
            if is_transient(exc):
                logger.warning("dispatch | transient, will retry | %s", message)
                with transaction(conn):
                    repo.set_interval_status(
                        conn, interval["IntervalRowId"], "pending",
                        ShortfallReason=f"transient, retrying: {str(exc)[:400]}",
                    )
                # The shared connection is probably dead; drop the cached
                # frames so the next interval re-queries rather than reusing
                # a result fetched before the break.
                windows.clear()
            else:
                logger.exception("dispatch | permanent | %s", message)
                with transaction(conn):
                    repo.set_interval_status(
                        conn, interval["IntervalRowId"], "failed",
                        ShortfallReason=str(exc)[:500],
                    )

    # Top up any run that is short once its intervals are all in.
    #
    # Per-interval quotas are even but traffic is not, so an interval that saw
    # fewer conversations than its quota leaves slots unfilled that no later
    # interval can use. Without this the day lands well under target — 284/300
    # on simulated busy traffic, 135/300 on thin. backfill_run redistributes
    # from the already-frozen candidate snapshots, so it costs no extra
    # CM_DWH query and relaxes no rule.
    if backfill:
        for run_id, reporting_date in _runs_touched(conn, due):
            if _run_is_fully_dispatched(conn, run_id):
                result = backfill_run(conn, settings, reporting_date=reporting_date)
                if result and result.added:
                    report.jobs_reserved += result.added
                    report.backfilled += result.added

    logger.info(
        "dispatch | processed=%d candidates=%d reserved=%d (backfilled=%d) "
        "shortfalls=%d errors=%d",
        report.intervals_processed, report.candidates_seen,
        report.jobs_reserved, report.backfilled,
        len(report.shortfalls), len(report.errors),
    )
    return report


def _runs_touched(conn, due) -> list[tuple[str, date]]:
    seen: dict[str, date] = {}
    for interval in due:
        seen.setdefault(interval["RunId"], date.fromisoformat(interval["ReportingDate"]))
    return list(seen.items())


def _run_is_fully_dispatched(conn, run_id: str) -> bool:
    """
    True when no interval of the run is still waiting.

    Backfilling before every interval has reported would hand out slots that a
    later interval's own candidates should have taken, biasing the day towards
    whoever was processed first.
    """
    return conn.execute(
        "SELECT COUNT(*) FROM QA_SamplingIntervals "
        " WHERE RunId = ? AND Status IN ('pending', 'snapshotting')",
        (run_id,),
    ).fetchone()[0] == 0


def _process_interval(conn, interval, window_df, settings, report, moment) -> None:
    agent = interval["AgentEmailAddress"]
    run_id = interval["RunId"]
    quota = interval["Quota"]
    reporting_date = date.fromisoformat(interval["ReportingDate"])

    agent_rows = (
        window_df.loc[window_df["AgentEmail"].astype(str).str.strip().str.lower() == agent.lower()]
        if not window_df.empty else pd.DataFrame()
    )
    report.candidates_seen += len(agent_rows)

    if agent_rows.empty or quota == 0:
        with transaction(conn):
            repo.set_interval_status(
                conn, interval["IntervalRowId"], "empty",
                CandidateCount=len(agent_rows), SelectedCount=0,
                ShortfallCount=quota, SnapshottedAtUtc=utc_text(moment),
                ShortfallReason=("no candidates in interval" if agent_rows.empty else "zero quota"),
            )
        report.intervals_processed += 1
        if quota:
            report.shortfalls.append(f"{agent} {interval['IntervalId']}: no candidates (quota {quota})")
        return

    snapshot_id = str(uuid.uuid4())

    with transaction(conn):
        # Sequence positions are frozen the first time a conversation is seen
        # and never renumbered: a late arrival takes the next free position
        # rather than displacing anything already committed.
        known = repo.existing_sequence_positions(conn, run_id, agent)
        next_position = (max(known.values()) + 1) if known else 1

        candidates: list[Candidate] = []
        rows_to_store: list[dict] = []

        ordered = agent_rows.sort_values(["HandlingTimeUtc", "ConversationId"], kind="mergesort")
        for _, row in ordered.iterrows():
            conversation_id = str(row["ConversationId"]).lower()
            is_late = conversation_id in known
            position = known.get(conversation_id, next_position)
            if conversation_id not in known:
                known[conversation_id] = position
                next_position += 1

            verdict = eligibility.assess(row)
            digest = rank_hash(
                base_seed=interval["BaseSeed"],
                policy_version=interval["SamplingPolicyVersion"],
                reporting_date=reporting_date,
                shift_id=agent,
                agent_id=agent,
                interval_id=interval["IntervalId"],
                conversation_id=conversation_id,
            )

            candidates.append(Candidate(
                conversation_id=conversation_id,
                sequence_position=position,
                rank_hash=digest,
                is_eligible=verdict.is_eligible,
                ineligible_reason=verdict.reason,
            ))
            rows_to_store.append({
                "SnapshotId": snapshot_id,
                "ConversationId": conversation_id,
                "RunId": run_id,
                "AgentEmailAddress": agent,
                "IntervalId": interval["IntervalId"],
                "AttributionRule": eligibility.ATTRIBUTION_RULE,
                "HandlingTimeUtc": _text(row.get("HandlingTimeUtc")),
                "SequencePosition": position,
                "RankHash": digest,
                "IsEligible": int(verdict.is_eligible),
                "EligibilityReason": verdict.reason,
                "IsSelected": 0,
                "SelectionReason": None,
                "ConversationState": _str_or_none(row.get("ConversationState")),
                "SourceLastUpdatedUtc": _text(row.get("LastUpdatedUtc")),
                "IsLateArrival": int(is_late),
                "CreatedAtUtc": utc_text(moment),
            })

        # Spacing must hold across intervals and across midnight, so the
        # already-selected set spans this run and the tail of the previous day.
        taken = repo.selected_positions(conn, run_id, agent)
        taken |= repo.trailing_positions_from_previous_run(
            conn, agent, reporting_date - timedelta(days=1)
        )

        outcome = select_nonconsecutive(
            candidates, quota,
            already_selected_positions=taken,
            enforce_spacing=bool(interval["ExcludeAdjacentChats"]),
        )

        selected = set(outcome.selected)
        for stored in rows_to_store:
            if stored["ConversationId"] in selected:
                stored["IsSelected"] = 1
                stored["SelectionReason"] = "selected"
            else:
                stored["SelectionReason"] = outcome.rejected.get(stored["ConversationId"])

        repo.insert_candidates(conn, rows_to_store)

        reserved, skipped = repo.reserve_jobs(
            conn,
            run_id=run_id, snapshot_id=snapshot_id, agent_email=agent,
            interval_id=interval["IntervalId"],
            policy_version=interval["SamplingPolicyVersion"],
            reporting_date=reporting_date,
            conversation_ids=outcome.selected,
            daily_target=interval["DailyTarget"],
        )

        shortfall = quota - len(reserved)
        reason = outcome.shortfall_reason
        if skipped:
            reason = (
                f"{len(skipped)} conversation(s) already reserved elsewhere or the "
                f"daily budget was exhausted" + (f"; {reason}" if reason else "")
            )

        repo.set_interval_status(
            conn, interval["IntervalRowId"], "snapshotted",
            SnapshotId=snapshot_id,
            CandidateCount=len(candidates),
            SelectedCount=len(reserved),
            ShortfallCount=shortfall,
            ShortfallReason=(reason or None),
            SnapshottedAtUtc=utc_text(moment),
        )

    report.intervals_processed += 1
    report.jobs_reserved += len(reserved)
    if shortfall > 0:
        report.shortfalls.append(f"{agent} {interval['IntervalId']}: short {shortfall} — {reason}")
    logger.info(
        "dispatch | %s %s | candidates=%d selected=%d quota=%d%s",
        agent, interval["IntervalId"], len(candidates), len(reserved), quota,
        f" SHORT {shortfall}" if shortfall else "",
    )


def _parse(text: str) -> datetime:
    from app.automation.db import parse_utc
    return parse_utc(text)


def _text(value) -> str:
    if value is None or pd.isna(value):
        return utc_text()
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=UTC)
        return utc_text(moment)
    return utc_text(pd.Timestamp(value).to_pydatetime().replace(tzinfo=UTC))


def _str_or_none(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)
