"""
Reporting-day and sampling-interval arithmetic for the automated QA sampler.

Why this module exists
──────────────────────
The existing chat SQL (app/SQL/NEWFIX_CM_Chat_Search.sql) hard-codes the
reporting timezone as ``DATEADD(HOUR, 3, ...)``.  That is correct for
Asia/Riyadh all year, but **wrong for Africa/Cairo for roughly half of it**:
Egypt reinstated daylight saving in 2023, so Cairo is UTC+2 (EET) in winter
and UTC+3 (EEST) from the last Friday of April to the last Thursday of
October.  A fixed +3 offset silently misplaces every conversation in the
first and last hour of a winter reporting day.

Three consequences drive the design here, and each one has a test:

1. **A reporting day is not always 24 hours.**  Cairo's 2026 transitions fall
   at 2026-04-23 22:00 UTC and 2026-10-29 21:00 UTC, which makes the local day
   2026-04-24 twenty-three hours long and 2026-10-29 twenty-five hours long.
   The daily budget stays at 300 regardless; only the interval layout changes.

2. **Some local wall-clock times do not exist, and some happen twice.**  On
   2026-04-24 the local times 00:00:01 through 00:59:59 never occur.  On
   2026-10-29 the hour 23:00-23:59 occurs twice.  A roster that says an agent
   starts at 00:30 must still resolve to exactly one instant.

3. **All filtering happens in UTC.**  Local time is used to decide *where the
   boundaries are* and for display; every value handed to SQL is a tz-aware
   UTC instant.  Nothing downstream should ever apply an offset itself.

All functions here are pure.  No database, no configuration lookup, no clock
reads except where a default is explicitly documented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

__all__ = [
    "ReportingDay",
    "Interval",
    "resolve_local",
    "local_midnight_utc",
    "reporting_day",
    "reporting_day_of",
    "split_intervals",
    "day_intervals",
    "snapshot_cutoff",
]

UTC = timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# Local wall-clock resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_local(naive: datetime, tz: ZoneInfo, *, prefer_earliest: bool = True) -> datetime:
    """
    Turn a naive local wall-clock datetime into a single unambiguous instant.

    Two cases that a bare ``naive.replace(tzinfo=tz)`` gets wrong:

    *Nonexistent* times, skipped by a spring-forward transition.  Cairo moves
    00:00 EET straight to 01:00 EEST on 2026-04-24, so a roster entry of 00:30
    on that date names a time that never happened.  It is moved forward by the
    size of the gap — 00:30 becomes 01:30 EEST — because a shift that was
    scheduled to start half an hour after midnight should start at the first
    real instant at or after that, not an hour early.

    *Ambiguous* times, which occur twice at a fall-back transition.  Cairo
    repeats 23:00-23:59 on 2026-10-29.  ``prefer_earliest`` (the default)
    selects the first occurrence, i.e. the one still on DST.  This is the safe
    default for a boundary: the resulting reporting day is the longer one, so
    no conversation falls outside every interval.

    Parameters
    ──────────
    naive : datetime
        Must be naive (``tzinfo is None``); passing an aware datetime is a
        programming error and raises, rather than silently re-interpreting it.
    """
    if naive.tzinfo is not None:
        raise ValueError(
            f"resolve_local expects a naive wall-clock datetime, got {naive!r} "
            f"with tzinfo={naive.tzinfo!r}"
        )

    fold = 0 if prefer_earliest else 1
    aware = naive.replace(tzinfo=tz, fold=fold)

    # Round-trip through UTC. For a nonexistent local time the round trip
    # lands on a *different* wall clock, which is how we detect the gap.
    round_tripped = aware.astimezone(UTC).astimezone(tz).replace(tzinfo=None)
    if round_tripped == naive:
        return aware

    # Nonexistent: shift forward by the size of the transition's jump.
    before = naive.replace(tzinfo=tz, fold=0).utcoffset()
    after = naive.replace(tzinfo=tz, fold=1).utcoffset()
    assert before is not None and after is not None  # ZoneInfo always supplies both
    gap = after - before
    return (naive + gap).replace(tzinfo=tz, fold=1)


def local_midnight_utc(day: date, tz: ZoneInfo) -> datetime:
    """The UTC instant at which the local calendar day *day* begins."""
    return resolve_local(datetime.combine(day, time(0, 0)), tz).astimezone(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting day
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ReportingDay:
    """
    One reporting day as the half-open UTC interval ``[start_utc, end_utc)``.

    ``date`` is the *local* calendar date that names the day; the UTC bounds
    are what gets passed to SQL.  ``hours`` is 23, 24 or 25 for a DST-observing
    zone — read it rather than assuming 24.
    """

    date: date
    timezone_name: str
    start_utc: datetime
    end_utc: datetime

    @property
    def duration(self) -> timedelta:
        return self.end_utc - self.start_utc

    @property
    def hours(self) -> float:
        return self.duration.total_seconds() / 3600.0

    @property
    def is_irregular(self) -> bool:
        """True when a DST transition makes this day other than 24 hours."""
        return self.duration != timedelta(hours=24)

    def contains(self, instant: datetime) -> bool:
        """Half-open membership test. *instant* must be tz-aware."""
        if instant.tzinfo is None:
            raise ValueError(f"contains() requires a tz-aware instant, got {instant!r}")
        return self.start_utc <= instant.astimezone(UTC) < self.end_utc

    def __str__(self) -> str:
        return f"{self.date.isoformat()} [{self.timezone_name}] {self.hours:g}h"


def reporting_day(day: date, tz: ZoneInfo) -> ReportingDay:
    """
    Build the reporting day for the local calendar date *day*.

    The interval runs from local midnight to the next local midnight, derived
    from the tz database, so a DST transition inside the day produces a 23- or
    25-hour span rather than a misaligned 24-hour one.
    """
    start = local_midnight_utc(day, tz)
    end = local_midnight_utc(day + timedelta(days=1), tz)
    if end <= start:
        raise ValueError(
            f"Non-increasing reporting-day bounds for {day} in {tz}: {start} -> {end}"
        )
    return ReportingDay(date=day, timezone_name=str(tz), start_utc=start, end_utc=end)


def reporting_day_of(instant: datetime, tz: ZoneInfo) -> date:
    """
    The local reporting date that a UTC instant belongs to.

    This is the function that assigns a conversation to a day, so it takes the
    handling timestamp straight from CM_DWH (UTC) and never a pre-offset value.
    """
    if instant.tzinfo is None:
        raise ValueError(
            f"reporting_day_of requires a tz-aware instant, got {instant!r}. "
            f"Timestamps read from CM_DWH must be stamped as UTC at the edge."
        )
    return instant.astimezone(tz).date()


# ─────────────────────────────────────────────────────────────────────────────
# Sampling intervals
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Interval:
    """
    One sampling interval, half-open ``[start_utc, end_utc)``.

    ``interval_id`` is part of the deterministic sampling hash (architecture
    §6), so its format is fixed here and must not change without bumping
    QA_SAMPLING_POLICY_VERSION: changing it silently reshuffles every future
    selection.  It is derived from the UTC bounds alone, which makes it stable
    across processes, hosts and DST transitions.
    """

    index: int
    start_utc: datetime
    end_utc: datetime

    @property
    def interval_id(self) -> str:
        return (
            f"{self.start_utc.strftime('%Y%m%dT%H%MZ')}"
            f"/{self.end_utc.strftime('%Y%m%dT%H%MZ')}"
        )

    @property
    def duration(self) -> timedelta:
        return self.end_utc - self.start_utc

    @property
    def minutes(self) -> int:
        return int(self.duration.total_seconds() // 60)

    def contains(self, instant: datetime) -> bool:
        if instant.tzinfo is None:
            raise ValueError(f"contains() requires a tz-aware instant, got {instant!r}")
        return self.start_utc <= instant.astimezone(UTC) < self.end_utc

    def local_bounds(self, tz: ZoneInfo) -> tuple[datetime, datetime]:
        """The interval in local time, for display and operator logs only."""
        return self.start_utc.astimezone(tz), self.end_utc.astimezone(tz)

    def __str__(self) -> str:
        return f"#{self.index} {self.interval_id} ({self.minutes}m)"


def split_intervals(
    start_utc: datetime,
    end_utc: datetime,
    minutes: int,
) -> list[Interval]:
    """
    Split ``[start_utc, end_utc)`` into consecutive intervals of *minutes*,
    with a shorter final interval when the span does not divide evenly.

    Intervals advance by **elapsed UTC time**, not by local wall clock.  That
    is the deliberate choice: each interval is a real two hours of agent work,
    so allocating quota proportionally to interval duration (architecture §5)
    stays meaningful across a DST transition.  The alternative — stepping the
    local clock by two hours — would produce one interval that is really one
    or three hours long while claiming to be two.

    A 25-hour Cairo day at 120 minutes yields twelve 2-hour intervals plus a
    final 1-hour one; a 23-hour day yields eleven plus a final 1-hour one.
    """
    if minutes <= 0:
        raise ValueError(f"Interval length must be positive, got {minutes}")
    if start_utc.tzinfo is None or end_utc.tzinfo is None:
        raise ValueError("split_intervals requires tz-aware bounds")
    if end_utc <= start_utc:
        return []

    start_utc = start_utc.astimezone(UTC)
    end_utc = end_utc.astimezone(UTC)
    step = timedelta(minutes=minutes)

    intervals: list[Interval] = []
    cursor = start_utc
    index = 0
    while cursor < end_utc:
        nxt = min(cursor + step, end_utc)
        intervals.append(Interval(index=index, start_utc=cursor, end_utc=nxt))
        cursor = nxt
        index += 1
    return intervals


def day_intervals(day: ReportingDay, minutes: int) -> list[Interval]:
    """Split a whole reporting day into sampling intervals."""
    return split_intervals(day.start_utc, day.end_utc, minutes)


def snapshot_cutoff(interval: Interval, grace_minutes: int) -> datetime:
    """
    The UTC instant at which an interval becomes eligible for snapshotting.

    The grace period absorbs CM_DWH replication lag: conversations handled just
    before the interval ended may not have landed yet.  QA_SOURCE_GRACE_MINUTES
    starts at 20 as an assumption to be tuned against measured lag, not as a
    known-correct value.
    """
    if grace_minutes < 0:
        raise ValueError(f"grace_minutes must be non-negative, got {grace_minutes}")
    return interval.end_utc + timedelta(minutes=grace_minutes)
